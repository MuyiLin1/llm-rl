"""
experiment_algorithm1.py  (AmazonCat-13K version)

Confidence-Gated Meta-Algorithm (Algorithm 1 from update.tex)
Uses CACHED LLM predictions (no runtime LLM calls needed).

Architecture:
  Same as WikiLinks version — deterministic confidence-gated routing
  between a frozen LLM and adaptive LinUCB, with passive covariance
  warm-start.

  LLM uncertainty:     u_LLM    = ||phi(x, a_LLM)|| / sqrt(c)
  LinUCB uncertainty:  u_LinUCB = sqrt(phi^T V^{-1} phi)

  if u_LLM < u_LinUCB  →  use LLM
  else                  →  use LinUCB

  Passive covariance update when LLM is routed:
      V ← V + phi(x, a_LLM) phi(x, a_LLM)^T

AmazonCat differences from WikiLinks:
  - Multi-label: reward = 1 if chosen action in true label set
  - 13,330 actions (much larger than WikiLinks ~2000)
  - LLM predictions come from pre-computed cache files
  - Features: phi(x,a) = x * z_a (element-wise, 768d for v1)

Usage:
  python -u experiment_algorithm1.py --llm_type small --llm_confidence 100
  python -u experiment_algorithm1.py --llm_type base  --llm_confidence 100
  python -u experiment_algorithm1.py --llm_type small --llm_confidence 100 --passive_full
  python -u experiment_algorithm1.py --llm_type large --llm_confidence 100 --passive_full \
      --seed 1 --alpha_decay 0.9985 --max_samples 50000 --router gate
  python -u experiment_algorithm1.py --llm_type large --router corral   --passive_full
  python -u experiment_algorithm1.py --llm_type large --router offline --offline_tau 5000

Outputs per run (seed and router are in every filename):
  results_<tag>.json    1,000-step running means (unchanged format)
  perround_<tag>.npz    r_router / r_cold / r_llm / routed, one entry per round
  checkpoint_<tag>.npz  V_inv/b state for --resume
"""

import torch
import argparse
import numpy as np
import json
import gzip
import pickle
import gc
import sys
import os
import warnings
from scipy.optimize import brentq
warnings.filterwarnings('ignore', category=RuntimeWarning)


# ================================================================
# Sherman-Morrison rank-1 inverse update: O(d^2)
# ================================================================
def sm_update(V_inv, x):
    # Skip update if feature has NaN/inf or near-zero norm
    if not np.all(np.isfinite(x)) or np.linalg.norm(x) < 1e-10:
        return V_inv
    Vx = V_inv @ x
    denom = 1.0 + x @ Vx
    if abs(denom) < 1e-12:
        return V_inv  # skip near-singular updates
    return V_inv - np.outer(Vx, Vx) / denom


# ================================================================
# Helpers
# ================================================================
class EasyAcc:
    def __init__(self):
        self.n = 0
        self.sum = 0

    def __iadd__(self, v):
        self.n += 1
        self.sum += v
        return self

    def mean(self):
        return self.sum / max(self.n, 1)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs): pass
    def __len__(self): return self.Xs.shape[0]
    def __getitem__(self, i): return i  # just return index

sys.modules['__main__'].MyDataset = MyDataset


def load_dataset(model_family='v1'):
    if model_family == 'v1':
        path = 'mydataset.0.pickle.gz'
    else:
        path = f'mydataset.0.{model_family}.pickle.gz'
    print(f"Loading {path} ...")
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)


def load_llm_cache(model_family, llm_type):
    path = f'llm_cache_amazon_{model_family}_{llm_type}.pkl.gz'
    print(f"Loading {path} ...")
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)


def get_entity_embeddings(ds):
    d = {}
    for k in ds.labelfeats:
        idx, embd = ds.labelfeats[k]
        d[idx] = np.array(embd, dtype=np.float32)
    Z = np.stack([d[i] for i in sorted(d.keys())])
    # Replace NaN/inf with 0 to avoid corrupting LinUCB matrices
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    bad = np.sum(np.linalg.norm(Z, axis=1) == 0)
    if bad > 0:
        print(f"  Warning: {bad}/{Z.shape[0]} entities have zero-norm embeddings")
    return Z


def get_context_embeddings(ds):
    Xs = ds.Xs.numpy() if isinstance(ds.Xs, torch.Tensor) else ds.Xs
    d = Xs.shape[1] // 2
    cx = Xs[:, :d].astype(np.float32)
    cx = np.nan_to_num(cx, nan=0.0, posinf=0.0, neginf=0.0)
    return cx



# ================================================================
# Action scoring: the O(K d^2) bottleneck, on GPU when one is free
# ================================================================
def pick_device(requested='auto', need_mib=512, max_util=90):
    """'auto' uses CUDA only if a GPU exists with room and is not saturated."""
    if requested == 'cpu':
        return 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        # Hard failure by design: a study must not silently mix a float32 GPU
        # path with a float64 CPU path across its runs.
        raise SystemExit("--device cuda requested but no CUDA device is available")
    try:
        if not torch.cuda.is_available():
            return 'cpu'
        free_b, _ = torch.cuda.mem_get_info()
        free_mib = free_b / (1024 ** 2)
        util = None
        try:
            import subprocess
            out = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu',
                                  '--format=csv,noheader,nounits'],
                                 capture_output=True, text=True, timeout=10)
            util = int(out.stdout.strip().split('\n')[0])
        except Exception:
            pass
        if requested == 'cuda':
            print(f"  GPU forced: {free_mib:.0f} MiB free, utilization {util}%")
            return 'cuda'
        if free_mib < need_mib:
            print(f"  GPU has only {free_mib:.0f} MiB free (< {need_mib}); using CPU")
            return 'cpu'
        if util is not None and util > max_util:
            print(f"  GPU utilization {util}% (> {max_util}); using CPU")
            return 'cpu'
        print(f"  GPU available: {free_mib:.0f} MiB free, utilization {util}%")
        return 'cuda'
    except Exception as e:
        print(f"  GPU probe failed ({e}); using CPU")
        return 'cpu'


def make_scorer(Z, device, fast32):
    """Return score(V_inv, theta, cx, alpha) -> argmax action index.

    V_inv and b always stay float64 on the host: only this throwaway scoring
    product is done in float32, so the Sherman-Morrison state never accumulates
    single-precision error.
    """
    if device == 'cuda':
        Zt = torch.as_tensor(np.ascontiguousarray(Z), dtype=torch.float32,
                             device='cuda')

        def score_gpu(V_inv, theta, cx, alpha):
            Cx = ((V_inv * cx[:, None]) * cx[None, :]).astype(np.float32)
            Cxt = torch.as_tensor(Cx, device='cuda')
            w = torch.as_tensor((cx * theta).astype(np.float32), device='cuda')
            var = ((Zt @ Cxt) * Zt).sum(1).clamp_min(0)
            ucb = Zt @ w + alpha * var.sqrt()
            ucb = torch.nan_to_num(ucb, nan=float('-inf'))
            return int(torch.argmax(ucb).item())
        return score_gpu

    if fast32:
        Z32 = Z.astype(np.float32)

        def score_f32(V_inv, theta, cx, alpha):
            Cx = ((V_inv * cx[:, None]) * cx[None, :]).astype(np.float32)
            var = np.einsum('ij,ij->i', Z32 @ Cx, Z32)
            ucb = Z32 @ (cx * theta).astype(np.float32) + alpha * np.sqrt(np.maximum(var, 0))
            return int(np.argmax(np.nan_to_num(ucb, nan=-np.inf)))
        return score_f32

    def score_f64(V_inv, theta, cx, alpha):
        Cx = (V_inv * cx[:, None]) * cx[None, :]
        var = np.sum((Z @ Cx) * Z, axis=1)
        ucb = Z @ (cx * theta) + alpha * np.sqrt(np.maximum(var, 0))
        return int(np.argmax(np.nan_to_num(ucb, nan=-np.inf)))
    return score_f64


# ================================================================
# Main experiment loop
# ================================================================
def learnOnline(dataset, llm_cache, seed, llm_type, model_family='v1',
                linucb_alpha=1.0, lam=0.1, alpha_decay=0.9985,
                llm_confidence=100.0, passive_full=False,
                max_samples=None, resume=False,
                router='gate', offline_tau=5000,
                corral_eta=1.0, corral_pmin=0.2, fast32=False, device='auto',
                dataset_name='amazon', decay_per='round'):
    np.random.seed(seed)

    if isinstance(dataset, dict):                    # slim .npz payload
        entity_emb  = dataset['Z']
        contexts    = dataset['contexts']
        true_labels = dataset['ys']
    else:
        entity_emb = get_entity_embeddings(dataset)  # (K, d)
        contexts   = get_context_embeddings(dataset) # (N, 768)
        true_labels = dataset.ys                     # list of lists
    llm_labels  = llm_cache['labels']                # (N,) — LLM's argmax prediction

    N_total = len(true_labels)
    N = N_total
    K = entity_emb.shape[0]
    d = entity_emb.shape[1]

    if max_samples is not None:
        N = min(N, max_samples)

    print(f"Samples: {N}, Actions: {K}, Feature dim: {d}")
    print(f"LLM type: {llm_type} | alpha={linucb_alpha} | lam={lam}")
    print(f"LLM confidence (c): {llm_confidence}")
    print(f"Passive update mode: {'full (V + b)' if passive_full else 'covariance only (V)'}")
    print(f"Alpha decay: {alpha_decay}")
    print(f"Router mode: {router}"
          + (f" (offline_tau={offline_tau})" if router == 'offline' else "")
          + (f" (eta={corral_eta}, p_min={corral_pmin})" if router == 'corral' else ""))

    # Shuffle order (reproducible). Unchanged from the gate runs: the online
    # stream is a permutation of the first N samples, so results stay comparable.
    perm = np.random.permutation(N)

    # --- Warm LinUCB (receives passive covariance updates from LLM rounds) ---
    V_inv_warm = (1.0 / lam) * np.eye(d, dtype=np.float64)
    b_warm = np.zeros(d, dtype=np.float64)

    # --- Cold LinUCB (independent baseline, no help from LLM) ---
    V_inv_cold = (1.0 / lam) * np.eye(d, dtype=np.float64)
    b_cold = np.zeros(d, dtype=np.float64)

    # LLM uncertainty: u = ||phi|| / sqrt(c)
    inv_sqrt_c = 1.0 / np.sqrt(llm_confidence)

    # --- Accumulators ---
    acc_router = EasyAcc()
    acc_llm    = EasyAcc()
    acc_cold   = EasyAcc()

    llm_rounds   = 0
    total_rounds = 0

    history = {
        'step': [], 'hybrid': [], 'llm': [], 'linucb_cold': [],
        'llm_fraction': [],
        'config': {
            'algorithm': 'confidence_gated_routing',
            'dataset': dataset_name,
            'model_family': model_family,
            'features': f'elementwise_{entity_emb.shape[1]}d',
            'linucb_alpha': linucb_alpha, 'lam': lam,
            'llm_type': llm_type, 'alpha_decay': alpha_decay,
            'llm_confidence': llm_confidence,
            'passive_full': passive_full,
            'num_samples': N, 'num_actions': K,
            'seed': seed, 'router': router,
            'offline_tau': offline_tau if router == 'offline' else None,
            'corral_eta': corral_eta if router == 'corral' else None,
            'corral_pmin': corral_pmin if router == 'corral' else None,
        }
    }

    tag = f"algo1_{dataset_name}_{model_family}_{llm_type}_c{llm_confidence}"
    if passive_full:
        tag += "_pfull"
    if router != 'gate':
        tag += f"_{router}"
    if decay_per != 'round':
        tag += f"_decay{decay_per}"
    tag += f"_s{seed}"
    fname = f"results_{tag}.json"
    ckpt_fname = f"checkpoint_{tag}.npz"
    perround_fname = f"perround_{tag}.npz"

    # Per-round logs (Phase 2: A_tau / V_tau / B_tau are one numpy line each
    # once these exist). int8 keeps 50k x 4 arrays at 200 KB.
    pr_router = np.zeros(N, dtype=np.int8)
    pr_cold   = np.zeros(N, dtype=np.int8)
    pr_llm    = np.zeros(N, dtype=np.int8)
    pr_routed = np.zeros(N, dtype=np.int8)
    # Which action each policy played. Needed to compute coverage, and hence the
    # price ratio of Theorem thm:general-price -- a reward alone cannot say which
    # directions an action covered.
    pr_a_cold   = np.zeros(N, dtype=np.int32)
    pr_a_router = np.zeros(N, dtype=np.int32)

    def save_perround(upto):
        np.savez_compressed(perround_fname,
                            r_router=pr_router[:upto], r_cold=pr_cold[:upto],
                            r_llm=pr_llm[:upto], routed=pr_routed[:upto],
                            a_cold=pr_a_cold[:upto], a_router=pr_a_router[:upto],
                            step=upto, seed=seed, router=router)

    print(f"Results file: {fname}")
    print(f"Checkpoint file: {ckpt_fname}")
    print(f"Per-round log: {perround_fname}")
    print(f"Total samples: {N}")
    print("=" * 90)

    # Pre-normalize entity embeddings for faster UCB computation
    # For element-wise features phi(x,a) = x * z_a, we precompute z_a
    Z = entity_emb.astype(np.float64)  # (K, d)
    dev = pick_device(device)
    if dev == 'cuda':
        fast32 = True          # the GPU path is float32 scoring by construction
    score = make_scorer(Z, dev, fast32)
    print(f"Scoring device: {dev} ({'float32' if fast32 else 'float64'} scoring, "
          f"float64 state)")

    log_interval = 1000  # log every 1000 samples
    ckpt_interval = 5000  # save full matrix checkpoint every 5000 steps

    # --- Resume from checkpoint if available ---
    resume_step = 0
    if resume and os.path.exists(ckpt_fname):
        print(f"Loading checkpoint from {ckpt_fname} ...")
        ckpt = np.load(ckpt_fname, allow_pickle=True)
        V_inv_warm = ckpt['V_inv_warm']
        b_warm     = ckpt['b_warm']
        V_inv_cold = ckpt['V_inv_cold']
        b_cold     = ckpt['b_cold']
        resume_step = int(ckpt['step'])
        llm_rounds   = int(ckpt['llm_rounds'])
        total_rounds = int(ckpt['total_rounds'])
        # Restore accumulators from saved JSON history
        if os.path.exists(fname):
            with open(fname) as f:
                history = json.load(f)
            n_logged = len(history['step'])
            # Restore cumulative sums from last logged mean × count
            last_step = history['step'][-1] if history['step'] else 0
            acc_router.sum = history['hybrid'][-1] * last_step if history['hybrid'] else 0
            acc_router.n   = last_step
            acc_llm.sum    = history['llm'][-1] * last_step if history['llm'] else 0
            acc_llm.n      = last_step
            acc_cold.sum   = history['linucb_cold'][-1] * last_step if history['linucb_cold'] else 0
            acc_cold.n     = last_step
        if os.path.exists(perround_fname):
            pr = np.load(perround_fname)
            k = int(pr['step'])
            pr_router[:k] = pr['r_router']; pr_cold[:k] = pr['r_cold']
            pr_llm[:k]    = pr['r_llm'];    pr_routed[:k] = pr['routed']
            if 'a_cold' in pr.files:
                pr_a_cold[:k] = pr['a_cold']; pr_a_router[:k] = pr['a_router']
            # The JSON is written every log_interval but the checkpoint only every
            # ckpt_interval, so its last entry can be AHEAD of resume_step and the
            # means above would then double-count. The per-round log is saved with
            # the checkpoint, so rebuild the accumulators from it instead, and drop
            # any history entries past the checkpoint.
            if k >= resume_step:
                j = resume_step
                acc_router.sum = float(pr_router[:j].sum()); acc_router.n = j
                acc_llm.sum    = float(pr_llm[:j].sum());    acc_llm.n    = j
                acc_cold.sum   = float(pr_cold[:j].sum());   acc_cold.n   = j
                llm_rounds     = int(pr_routed[:j].sum())
                total_rounds   = j
                keep = sum(1 for s in history['step'] if s <= resume_step)
                for key in ('step', 'hybrid', 'llm', 'linucb_cold', 'llm_fraction'):
                    history[key] = history[key][:keep]
            else:
                print(f"  Warning: per-round log ends at {k} but checkpoint is at "
                      f"{resume_step}; accumulators restored from the JSON means.")
        print(f"Resuming from step {resume_step}/{N} "
              f"(llm_rounds={llm_rounds}, total={total_rounds})")
    elif resume:
        print(f"No checkpoint found at {ckpt_fname}, starting from scratch.")

    # --- Offline warm start (row 1): acquisition is not charged. Pull the
    # LLM's cached proposal on held-out contexts drawn from OUTSIDE the online
    # stream, then run plain LinUCB with no routing. ---
    if router == 'offline' and resume_step == 0:
        pool = np.arange(N, N_total)
        if pool.size == 0:
            raise SystemExit("offline warm start needs samples outside the online "
                             "stream; lower --max_samples")
        n_off = min(offline_tau, pool.size)
        ho = np.random.default_rng(seed + 1000).choice(pool, size=n_off, replace=False)
        off_correct = 0
        for j in ho:
            j = int(j)
            a = int(llm_labels[j])
            r = 1.0 if a in true_labels[j] else 0.0
            off_correct += r
            xo = contexts[j].astype(np.float64) * Z[a]
            V_inv_warm = sm_update(V_inv_warm, xo)
            b_warm += r * xo
        print(f"Offline warm start: {n_off} held-out LLM pulls, "
              f"accuracy {off_correct/max(n_off,1):.4f} (not charged as regret)")

    # --- CORRAL state: two bases, index 0 = LinUCB, index 1 = LLM. Mirrors
    # corral_clip.py (log-barrier OMD, brentq normaliser, p_min on LinUCB). ---
    invpalgo = np.array([2.0, 2.0])
    corral_rng = np.random.default_rng(seed + 2000)

    def corral_probs():
        p = 1.0 / invpalgo
        p0 = max(corral_pmin, p[0])
        return np.array([p0, 1.0 - p0])

    def corral_update(played, invprop, reward):
        invp = invpalgo.copy()
        invp[played] += (corral_eta / 2.0) * (-reward) * invprop
        invp += 1.0 - invp.min()
        zlb, zub = 0.0, 1.0
        while np.sum(1.0 / (invp + zub)) > 1.0:
            zlb, zub = zub, zub * 2.0
        root = brentq(lambda z: 1.0 - np.sum(1.0 / (invp + z)), zlb, zub)
        return invp + root

    for step_i in range(resume_step, N):
        idx = perm[step_i]
        cx = contexts[idx].astype(np.float64)  # (d,)
        ys_true = true_labels[idx]              # list of correct labels
        llm_choice = int(llm_labels[idx])

        # 'round' is the schedule v5 states; 'batch' reproduces the pre-v5
        # WikiLinks runs, whose anneal advanced once per minibatch of 128.
        exponent = step_i if decay_per == 'round' else step_i // 128
        current_alpha = max(0.1, linucb_alpha * (alpha_decay ** exponent))

        total_rounds += 1

        # Joint features for all actions: phi(x, a) = x * z_a  → (K, d)
        # This is expensive with 13K actions. We avoid materializing the full matrix.
        # Instead, compute theta^T phi = (V_inv @ b)^T (x * z) = (theta * x)^T z
        # and phi^T V_inv phi for selected actions only.

        # ===========================================================
        # TRACK 1: LLM only (static baseline)
        # ===========================================================
        r_llm = 1.0 if llm_choice in ys_true else 0.0
        acc_llm += r_llm

        # ===========================================================
        # TRACK 2: Cold LinUCB
        # ===========================================================
        theta_cold = V_inv_cold @ b_cold  # (d,)
        cold_choice = score(V_inv_cold, theta_cold, cx, current_alpha)
        # UCB scores: for each action a, score = phi(x,a)^T theta + alpha * sqrt(phi^T V_inv phi)
        # phi(x,a) = x * z_a, so phi^T theta = (x * z_a)^T theta = x^T (z_a * theta)
        # = sum_j x_j * z_{a,j} * theta_j
        r_cold = 1.0 if cold_choice in ys_true else 0.0
        acc_cold += r_cold

        x_cold = cx * Z[cold_choice]  # phi(x, a_cold)
        V_inv_cold = sm_update(V_inv_cold, x_cold)
        b_cold += r_cold * x_cold

        # ===========================================================
        # TRACK 3: Confidence-Gated Router (Algorithm 1)
        # ===========================================================
        theta_warm = V_inv_warm @ b_warm
        linucb_choice = score(V_inv_warm, theta_warm, cx, current_alpha)
        # Feature vectors
        x_llm    = cx * Z[llm_choice]
        x_linucb = cx * Z[linucb_choice]

        # Who plays this round?
        if router == 'gate':
            # Deterministic confidence gate (Algorithm 1)
            u_llm    = inv_sqrt_c * np.linalg.norm(x_llm)
            u_linucb = np.sqrt(max(0.0, x_linucb @ V_inv_warm @ x_linucb))
            use_llm = bool(u_llm < u_linucb)
            invprop = 1.0
        elif router == 'corral':
            p = corral_probs()
            use_llm = bool(corral_rng.random() < p[1])
            invprop = 1.0 / p[1 if use_llm else 0]
        else:  # 'offline' — warm start already applied, no routing
            use_llm = False
            invprop = 1.0

        if use_llm:
            router_choice = llm_choice
            r_router = 1.0 if router_choice in ys_true else 0.0
            llm_rounds += 1

            V_inv_warm = sm_update(V_inv_warm, x_llm)
            if passive_full:
                b_warm += r_router * x_llm
        else:
            router_choice = linucb_choice
            r_router = 1.0 if router_choice in ys_true else 0.0

            V_inv_warm = sm_update(V_inv_warm, x_linucb)
            b_warm += r_router * x_linucb

        if router == 'corral':
            invpalgo = corral_update(1 if use_llm else 0, invprop, r_router)

        acc_router += r_router

        # Per-round log (Phase 2)
        pr_router[step_i] = int(r_router)
        pr_cold[step_i]   = int(r_cold)
        pr_llm[step_i]    = int(r_llm)
        pr_routed[step_i] = int(use_llm)
        pr_a_cold[step_i]   = cold_choice
        pr_a_router[step_i] = router_choice

        # ===========================================================
        # Logging
        # ===========================================================
        if (step_i + 1) % log_interval == 0 or step_i == N - 1:
            llm_frac = llm_rounds / max(total_rounds, 1)
            history['step'].append(step_i + 1)
            history['hybrid'].append(acc_router.mean())
            history['llm'].append(acc_llm.mean())
            history['linucb_cold'].append(acc_cold.mean())
            history['llm_fraction'].append(llm_frac)

            print(
                f"Step {step_i+1:7d}/{N} | "
                f"Router: {acc_router.mean():.4f} | "
                f"LLM: {acc_llm.mean():.4f} | "
                f"Cold LinUCB: {acc_cold.mean():.4f} | "
                f"LLM%: {llm_frac:.1%}"
            )

            with open(fname, "w") as f:
                json.dump(history, f)

        # Save matrix checkpoint every ckpt_interval steps
        if (step_i + 1) % ckpt_interval == 0:
            save_perround(step_i + 1)
            np.savez(ckpt_fname,
                     V_inv_warm=V_inv_warm, b_warm=b_warm,
                     V_inv_cold=V_inv_cold, b_cold=b_cold,
                     step=step_i + 1,
                     llm_rounds=llm_rounds,
                     total_rounds=total_rounds)

    print("=" * 90)
    print(
        f"FINAL | Router: {acc_router.mean():.4f} | "
        f"LLM: {acc_llm.mean():.4f} | "
        f"Cold LinUCB: {acc_cold.mean():.4f}"
    )
    print(f"LLM routing: {llm_rounds}/{total_rounds} = {llm_rounds/max(total_rounds,1):.1%}")
    with open(fname, "w") as f:
        json.dump(history, f)
    save_perround(N)

    # A / V / B straight off the per-round logs (v5's decomposition, binary
    # reward: regret is 1-r, so differences of rewards are differences of regret)
    routed = pr_routed[:N].astype(bool)
    A = float((pr_router[:N][routed]  - pr_cold[:N][routed]).sum())
    V = float((pr_router[:N][~routed] - pr_cold[:N][~routed]).sum())
    B = float((pr_router[:N][~routed] - pr_llm[:N][~routed]).sum())
    tot = float(pr_router[:N].sum() - pr_cold[:N].sum())
    print(f"Decomposition: A={A:.0f}  V={V:.0f}  B={B:.0f}  "
          f"(A+V={A+V:.0f} must equal sum(r_router)-sum(r_cold)={tot:.0f})")
    print(f"Saved: {fname}, {perround_fname}")
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Algorithm 1: Confidence-Gated Routing (AmazonCat-13K)'
    )
    parser.add_argument('--seed',           type=int,   default=1)
    parser.add_argument('--model_family',   type=str,   default='v1',
                        choices=['v1', 'v3'],
                        help='v1 = Flan-T5/768d, v3 = Qwen/1024d')
    parser.add_argument('--llm_type',       type=str,   default='small',
                        choices=['small', 'base', 'large'])
    parser.add_argument('--linucb_alpha',   type=float, default=1.0)
    parser.add_argument('--lam',            type=float, default=0.1)
    parser.add_argument('--llm_confidence', type=float, default=100.0)
    parser.add_argument('--passive_full',   action='store_true')
    parser.add_argument('--alpha_decay',    type=float, default=0.9985,
                        help='v5 states 0.9985; the pre-v5 runs used 0.9999')
    parser.add_argument('--router',         type=str,   default='gate',
                        choices=['gate', 'corral', 'offline'],
                        help='gate = Algorithm 1; corral = randomized master; '
                             'offline = warm start from held-out LLM pulls, no routing')
    parser.add_argument('--offline_tau',    type=int,   default=5000,
                        help='--router offline: number of held-out LLM pulls')
    parser.add_argument('--corral_eta',     type=float, default=1.0)
    parser.add_argument('--corral_pmin',    type=float, default=0.2,
                        help='--router corral: floor on the LinUCB probability')
    parser.add_argument('--max_samples',    type=int,   default=None,
                        help='Limit number of samples (for quick testing)')
    parser.add_argument('--resume',         action='store_true',
                        help='Resume from checkpoint if available')
    parser.add_argument('--decay_per',      type=str,   default='round',
                        choices=['round', 'batch'],
                        help="'round' = v5's alpha_t = max(0.1, decay^t). 'batch' "
                             "advances the anneal once per 128 rounds, reproducing "
                             "the pre-v5 WikiLinks runs.")
    parser.add_argument('--dataset_name',   type=str,   default='amazon',
                        help='goes into every output filename so WikiLinks and '
                             'AmazonCat runs never collide')
    parser.add_argument('--device',         type=str,   default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help="'auto' uses the GPU when one is free and not "
                             "saturated, else CPU. Results are unchanged either "
                             "way apart from float32 rounding in the scoring.")
    parser.add_argument('--fast32',         action='store_true',
                        help='Do the O(K d^2) UCB scoring in float32 while keeping '
                             'V_inv and b in float64: ~3.9x faster per round, and '
                             'the Sherman-Morrison state never sees single precision.')
    parser.add_argument('--slim',           type=str,   default=None,
                        help='Path to amazon_<family>_slim.npz from make_slim_npz.py. '
                             'Loads in seconds and uses ~250 MB instead of ~11 GB, '
                             'so many runs fit on one machine.')
    args = parser.parse_args()

    if args.slim:
        print(f"Loading slim dataset {args.slim} ...")
        z = np.load(args.slim)
        flat, offs = z['y_flat'], z['y_offsets']
        ds = {'Z': z['Z'], 'contexts': z['contexts'],
              'ys': [flat[offs[i]:offs[i+1]].tolist() for i in range(len(offs) - 1)]}
        cache = {'labels': z[f'llm_{args.llm_type}']}
        if int(cache['labels'].min()) < 0:
            print("  Warning: this cache does not cover every kept sample "
                  "(v3 partial cache); rounds with label -1 count as errors.")
    else:
        ds = load_dataset(args.model_family)
        cache = load_llm_cache(args.model_family, args.llm_type)

    learnOnline(
        ds, cache, seed=args.seed,
        llm_type=args.llm_type,
        model_family=args.model_family,
        linucb_alpha=args.linucb_alpha,
        lam=args.lam,
        llm_confidence=args.llm_confidence,
        passive_full=args.passive_full,
        alpha_decay=args.alpha_decay,
        max_samples=args.max_samples,
        resume=args.resume,
        router=args.router,
        offline_tau=args.offline_tau,
        corral_eta=args.corral_eta,
        corral_pmin=args.corral_pmin,
        fast32=args.fast32,
        device=args.device,
        dataset_name=args.dataset_name,
        decay_per=args.decay_per,
    )
