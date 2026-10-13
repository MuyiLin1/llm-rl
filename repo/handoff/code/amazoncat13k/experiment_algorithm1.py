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
# Main experiment loop
# ================================================================
def learnOnline(dataset, llm_cache, seed, llm_type, model_family='v1',
                linucb_alpha=1.0, lam=0.1, alpha_decay=0.9999,
                llm_confidence=100.0, passive_full=False,
                max_samples=None, resume=False):
    np.random.seed(seed)

    entity_emb = get_entity_embeddings(dataset)     # (K, d)
    contexts   = get_context_embeddings(dataset)     # (N, 768)
    true_labels = dataset.ys                         # list of lists
    llm_labels  = llm_cache['labels']                # (N,) — LLM's argmax prediction

    N = len(true_labels)
    K = entity_emb.shape[0]
    d = entity_emb.shape[1]

    if max_samples is not None:
        N = min(N, max_samples)

    print(f"Samples: {N}, Actions: {K}, Feature dim: {d}")
    print(f"LLM type: {llm_type} | alpha={linucb_alpha} | lam={lam}")
    print(f"LLM confidence (c): {llm_confidence}")
    print(f"Passive update mode: {'full (V + b)' if passive_full else 'covariance only (V)'}")
    print(f"Alpha decay: {alpha_decay}")

    # Shuffle order (reproducible)
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
            'dataset': 'amazoncat13k',
            'model_family': model_family,
            'features': f'elementwise_{entity_emb.shape[1]}d',
            'linucb_alpha': linucb_alpha, 'lam': lam,
            'llm_type': llm_type, 'alpha_decay': alpha_decay,
            'llm_confidence': llm_confidence,
            'passive_full': passive_full,
            'num_samples': N, 'num_actions': K,
        }
    }

    tag = f"algo1_amazon_{model_family}_{llm_type}_c{llm_confidence}"
    if passive_full:
        tag += "_pfull"
    fname = f"results_{tag}.json"
    ckpt_fname = f"checkpoint_{tag}.npz"
    print(f"Results file: {fname}")
    print(f"Checkpoint file: {ckpt_fname}")
    print(f"Total samples: {N}")
    print("=" * 90)

    # Pre-normalize entity embeddings for faster UCB computation
    # For element-wise features phi(x,a) = x * z_a, we precompute z_a
    Z = entity_emb.astype(np.float64)  # (K, d)

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
        print(f"Resuming from step {resume_step}/{N} "
              f"(llm_rounds={llm_rounds}, total={total_rounds})")
    elif resume:
        print(f"No checkpoint found at {ckpt_fname}, starting from scratch.")

    for step_i in range(resume_step, N):
        idx = perm[step_i]
        cx = contexts[idx].astype(np.float64)  # (d,)
        ys_true = true_labels[idx]              # list of correct labels
        llm_choice = int(llm_labels[idx])

        current_alpha = max(0.1, linucb_alpha * (alpha_decay ** step_i))

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
        # UCB scores: for each action a, score = phi(x,a)^T theta + alpha * sqrt(phi^T V_inv phi)
        # phi(x,a) = x * z_a, so phi^T theta = (x * z_a)^T theta = x^T (z_a * theta)
        # = sum_j x_j * z_{a,j} * theta_j
        weighted_cold = cx * theta_cold   # (d,)
        mu_cold = Z @ weighted_cold       # (K,) — mean reward estimates

        # For UCB bonus: phi^T V_inv phi = (x*z_a)^T V_inv (x*z_a)
        # = z_a^T diag(x) V_inv diag(x) z_a
        # Let M = V_inv * outer(x, x) element-wise, then bonus = z_a^T M z_a
        # Or: let w = V_inv @ (x * z_a), bonus = (x * z_a) @ w
        # For all K actions efficiently:
        # Vx = V_inv * x (broadcast: scale columns by x)
        # Then for each a: bonus_a = (x * z_a)^T (Vx @ z_a) ... still O(K*d^2)
        # Actually: phi_a = x * z_a, phi_a^T V_inv phi_a = sum_ij x_i z_{a,i} V_inv_{ij} x_j z_{a,j}
        # = (x * z_a)^T V_inv (x * z_a)
        # Let Cx = diag(x) @ V_inv @ diag(x), then bonus_a = z_a^T Cx z_a
        # Cx = (V_inv * x[:, None]) * x[None, :]  — element-wise scaling
        # Then bonus = diag(Z @ Cx @ Z^T) = sum over columns of (Z @ Cx) * Z

        Cx_cold = (V_inv_cold * cx[:, None]) * cx[None, :]  # (d, d)
        ZCx_cold = Z @ Cx_cold  # (K, d)
        var_cold = np.sum(ZCx_cold * Z, axis=1)  # (K,)
        ucb_cold = mu_cold + current_alpha * np.sqrt(np.maximum(var_cold, 0))
        ucb_cold = np.nan_to_num(ucb_cold, nan=-np.inf)
        cold_choice = int(np.argmax(ucb_cold))

        r_cold = 1.0 if cold_choice in ys_true else 0.0
        acc_cold += r_cold

        x_cold = cx * Z[cold_choice]  # phi(x, a_cold)
        V_inv_cold = sm_update(V_inv_cold, x_cold)
        b_cold += r_cold * x_cold

        # ===========================================================
        # TRACK 3: Confidence-Gated Router (Algorithm 1)
        # ===========================================================
        theta_warm = V_inv_warm @ b_warm
        weighted_warm = cx * theta_warm
        mu_warm = Z @ weighted_warm

        Cx_warm = (V_inv_warm * cx[:, None]) * cx[None, :]
        ZCx_warm = Z @ Cx_warm
        var_warm = np.sum(ZCx_warm * Z, axis=1)
        ucb_warm = mu_warm + current_alpha * np.sqrt(np.maximum(var_warm, 0))
        ucb_warm = np.nan_to_num(ucb_warm, nan=-np.inf)
        linucb_choice = int(np.argmax(ucb_warm))

        # Feature vectors
        x_llm    = cx * Z[llm_choice]
        x_linucb = cx * Z[linucb_choice]

        # Geometric uncertainties
        u_llm    = inv_sqrt_c * np.linalg.norm(x_llm)
        u_linucb = np.sqrt(max(0.0, x_linucb @ V_inv_warm @ x_linucb))

        # Deterministic routing
        if u_llm < u_linucb:
            # LLM is more confident → defer to LLM
            router_choice = llm_choice
            r_router = 1.0 if router_choice in ys_true else 0.0
            llm_rounds += 1

            V_inv_warm = sm_update(V_inv_warm, x_llm)
            if passive_full:
                b_warm += r_router * x_llm
        else:
            # LinUCB is confident → route to LinUCB
            router_choice = linucb_choice
            r_router = 1.0 if router_choice in ys_true else 0.0

            V_inv_warm = sm_update(V_inv_warm, x_linucb)
            b_warm += r_router * x_linucb

        acc_router += r_router

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
    print(f"Saved: {fname}")
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
    parser.add_argument('--alpha_decay',    type=float, default=0.9999)
    parser.add_argument('--max_samples',    type=int,   default=None,
                        help='Limit number of samples (for quick testing)')
    parser.add_argument('--resume',         action='store_true',
                        help='Resume from checkpoint if available')
    args = parser.parse_args()

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
    )
