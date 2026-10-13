"""
experiment_algorithm1.py

Confidence-Gated Meta-Algorithm (Algorithm 1 from the paper)
─────────────────────────────────────────────────────────────

Implements the paper's theoretical framework: a deterministic routing
switchboard between a frozen LLM estimator and an adaptive LinUCB learner,
with passive covariance warm-start.

Architecture
────────────
  The master algorithm computes geometric uncertainty scores each round:

    LLM uncertainty:     u_LLM    = ‖φ(x, a_LLM)‖ / √c
        Models the LLM's "offline" confidence from pretraining.
        c = effective offline sample count (higher → lower uncertainty).

    LinUCB uncertainty:  u_LinUCB = √(φ(x, a_LinUCB)ᵀ V⁻¹ φ(x, a_LinUCB))
        Standard ridge regression variance from the online design matrix.

  Deterministic routing:
    if u_LLM < u_LinUCB  →  defer to frozen LLM
    else                  →  use adaptive LinUCB

  Passive covariance update (the warm-start mechanism):
    When the LLM is routed, LinUCB STILL updates its design matrix V
    with the feature vector of the LLM's chosen action:
        V ← V + φ(x, a_LLM) φ(x, a_LLM)ᵀ
    This shrinks LinUCB's confidence ellipsoid WITHOUT requiring the
    reward signal, allowing faster hand-off.

  "Reward Multiplication" variant (--passive_full):
    Optionally update both V and b during passive rounds (stronger
    empirically, but deviates from the strict proof in the paper).

Why this matters
────────────────
  - Early phase: LLM uncertainty is low (c is large), LinUCB is uninformed
    → router defers to LLM for good early performance.
  - Over time: passive V updates + active LinUCB rounds shrink LinUCB's
    uncertainty → deterministic hand-off to LinUCB.
  - Theoretical regret bound is maintained because routing is deterministic
    (no importance weighting needed).

Recommended usage (proving the hand-off with a weaker LLM):
  python experiment_algorithm1.py --llm_type small --llm_confidence 100
  python experiment_algorithm1.py --llm_type base  --llm_confidence 100
  python experiment_algorithm1.py --llm_type large --llm_confidence 100

The --llm_confidence parameter controls how long the LLM phase lasts:
  Higher c → LLM has lower uncertainty → longer LLM phase before hand-off.
  Lower  c → earlier switch to LinUCB.

Baselines tracked (all on the same data stream):
  Router     — the deterministic confidence-gated switchboard
  LLM only   — frozen LLM (argmax cosine similarity)
  Cold LinUCB — LinUCB without any warm-start (no passive updates)
"""

import torch
import argparse
import numpy as np
import json
import gzip
import pickle
import gc
from scipy.optimize import brentq
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer


# ================================================================
# Sherman-Morrison rank-1 inverse update: O(d^2)
# ================================================================
def sm_update(V_inv, x):
    Vx = V_inv @ x
    return V_inv - np.outer(Vx, Vx) / (1.0 + x @ Vx)


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
    def __init__(self): pass
    def __len__(self): return self.Xs.shape[0]
    def __getitem__(self, i):
        return (self.Xs[i], self.ys[i], self.pre_texts[i], self.post_texts[i],
                self.text_entities[i], i)


def loadMyDataset(threshold):
    with gzip.open(f'mydataset.{threshold}.pickle.gz', 'rb') as f:
        return pickle.load(f)


def load_model_and_tokenizer(model_name, device):
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    return model, tokenizer


def generate_mask_fillings(sentences, model, tokenizer, device):
    texts = [f"question: {s.replace('[MASK]', '<extra_id_0>')}" for s in sentences]
    inp = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    out = model.generate(
        inp.input_ids.to(device),
        attention_mask=inp.attention_mask.to(device),
        max_new_tokens=20
    )
    return [
        tokenizer.decode(ids, skip_special_tokens=True).split('<extra_id_0>')[-1].strip()
        for ids in out
    ]


def language_model_scores(pre_texts, post_texts, t5_tokenizer, t5_model,
                           entity_model, entity_embd, num_entities, cos, device):
    """
    Returns:
      labels : (batch_size,) — LLM's argmax predictions
      sims   : (batch_size, num_entities) — cosine similarities
    """
    sentences = [pre + ' [MASK]. ' + post for pre, post in zip(pre_texts, post_texts)]
    predicted = generate_mask_fillings(sentences, t5_model, t5_tokenizer, device)
    pred_embd = torch.FloatTensor(entity_model.encode(predicted)).to(device)
    pred_embd = pred_embd.unsqueeze(1).repeat(1, num_entities, 1)
    ent_exp = entity_embd.unsqueeze(0).repeat(len(pre_texts), 1, 1).to(device)
    sims = cos(pred_embd, ent_exp)                           # (batch, num_entities)
    labels = torch.argmax(sims, dim=-1).detach()             # (batch,)
    return labels, sims.detach().cpu().numpy()


def get_embd(dataset):
    d = {}
    for k in dataset.labelfeats:
        id_, embd = dataset.labelfeats[k]
        d[id_] = torch.FloatTensor(embd)
    return torch.stack([d[k] for k in sorted(d)])


# ================================================================
# Main experiment loop
# ================================================================
def learnOnline(dataset, batch_size, cuda, seed, llm_type,
                linucb_alpha=1.0, lam=0.1, alpha_decay=0.9985,
                llm_confidence=100.0, passive_full=False,
                llm_cache_path=None, router='gate', offline_tau=5000,
                corral_eta=1.0, corral_pmin=0.2, decay_per='round'):
    """
    Parameters
    ----------
    llm_confidence : float
        Effective "offline sample count" for the LLM's design matrix.
        The LLM's geometric uncertainty is  u = ‖φ‖ / √c.
        Higher c → longer LLM phase before hand-off.
    passive_full : bool
        If False (default): passive rounds update only V (covariance).
        If True: passive rounds update both V and b (reward vector).
        The paper's proofs use V-only; b-too is an empirical variant.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    entity_embd = get_embd(dataset)
    num_entities = entity_embd.shape[0]

    if cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"LLM: flan-t5-{llm_type} | alpha={linucb_alpha} | lam={lam}")
    print(f"LLM confidence (c): {llm_confidence}")
    print(f"Passive update mode: {'full (V + b)' if passive_full else 'covariance only (V)'}")
    print(f"Num actions: {num_entities}")

    print(f"Router mode: {router}")
    cached_labels = None
    if llm_cache_path:
        print(f"Loading LLM cache: {llm_cache_path}")
        with gzip.open(llm_cache_path, 'rb') as f:
            cache = pickle.load(f)
        cached_labels = cache['labels']
        print(f"  Cache: {len(cached_labels)} samples — no GPU needed")
        t5_model = t5_tokenizer = entity_model = cos = None
    else:
        t5_model, t5_tokenizer = load_model_and_tokenizer(f"google/flan-t5-{llm_type}", device)
        entity_model = SentenceTransformer('bert-base-nli-mean-tokens', device=device)
        cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    # Dedicated generator so shuffle order is identical across LLM sizes
    data_gen = torch.Generator()
    data_gen.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
        generator=data_gen
    )

    action_features = entity_embd.cpu().numpy()  # (num_actions, 768)
    d = action_features.shape[1]                 # 768
    offline_holdout = None
    if router == 'offline':
        if cached_labels is None:
            raise SystemExit("--router offline requires --llm_cache")
        g_ho = torch.Generator(); g_ho.manual_seed(seed + 1000)
        perm_all = torch.randperm(len(dataset), generator=g_ho).numpy()
        offline_holdout = perm_all[:offline_tau]
        online_idx = perm_all[offline_tau:]
        base_ds = dataset
        dataset = torch.utils.data.Subset(base_ds, online_idx)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0,
            pin_memory=True, generator=data_gen)
        print(f"Offline warm start: {len(offline_holdout)} held-out LLM pulls, "
              f"online stream {len(online_idx)} rounds")

    print(f"Feature dim: {d}")

    # --- Warm LinUCB (receives passive covariance updates from LLM rounds) ---
    V_inv_warm = (1.0 / lam) * np.eye(d)
    b_warm = np.zeros(d)

    # --- Cold LinUCB (independent baseline, no help from LLM) ---
    V_inv_cold = (1.0 / lam) * np.eye(d)
    b_cold = np.zeros(d)

    # LLM's geometric uncertainty uses a fixed "offline" design matrix V_LLM = c * I
    # u_LLM(phi) = sqrt(phi^T (c*I)^{-1} phi) = ||phi|| / sqrt(c)
    inv_sqrt_c = 1.0 / np.sqrt(llm_confidence)

    # --- Accumulators ---
    acc_router = EasyAcc()    # Confidence-gated router
    acc_llm    = EasyAcc()    # LLM only (frozen baseline)
    acc_cold   = EasyAcc()    # Cold LinUCB (no warm-start)

    # --- Routing diagnostics ---
    llm_rounds   = 0          # cumulative rounds LLM was chosen by router
    total_rounds = 0

    history = {
        'batch': [], 'hybrid': [], 'llm': [], 'linucb_cold': [],
        'llm_fraction': [],    # fraction of rounds LLM was chosen (for hand-off plot)
        'config': {
            'algorithm': 'confidence_gated_routing',
            'features': 'elementwise_768d',
            'linucb_alpha': linucb_alpha, 'lam': lam,
            'llm_type': llm_type, 'alpha_decay': alpha_decay,
            'llm_confidence': llm_confidence,
            'passive_full': passive_full,
            'decay_per': decay_per,
        }
    }

    tag = f"algo1_{llm_type}_c{llm_confidence}"
    if passive_full:
        tag += "_pfull"
    if decay_per != 'round':
        tag += f"_decay{decay_per}"
    if router != 'gate':
        tag += f"_{router}"
    tag += f"_s{seed}"
    fname = f"results_{tag}.json"
    perround_fname = f"perround_{tag}.npz"
    history['config'].update({'seed': seed, 'router': router,
                              'offline_tau': offline_tau if router == 'offline' else None,
                              'llm_cache': llm_cache_path})

    n_rounds = len(dataset)
    pr_router = np.zeros(n_rounds, dtype=np.int8)
    pr_cold   = np.zeros(n_rounds, dtype=np.int8)
    pr_llm    = np.zeros(n_rounds, dtype=np.int8)
    pr_routed = np.zeros(n_rounds, dtype=np.int8)

    # Offline warm start: pull the LLM's cached proposal on the held-out samples
    # and fold them into (V, b). Acquisition regret is not charged (row 1).
    if offline_holdout is not None:
        off_correct = 0
        for j in offline_holdout:
            j = int(j)
            cxo = base_ds.Xs[j].cpu().numpy()[:d]
            a = int(cached_labels[j])
            r = 1.0 if a == int(base_ds.ys[j]) else 0.0
            off_correct += r
            xo = action_features[a] * cxo
            V_inv_warm = sm_update(V_inv_warm, xo)
            b_warm += r * xo
        print(f"  held-out LLM accuracy {off_correct/max(len(offline_holdout),1):.4f} "
              f"(not charged as regret)")

    # CORRAL state: index 0 = LinUCB, index 1 = LLM (mirrors corral_clip.py)
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
        return invp + brentq(lambda z: 1.0 - np.sum(1.0 / (invp + z)), zlb, zub)

    print(f"Results file: {fname}")
    print(f"Per-round log: {perround_fname}")
    print(f"Total batches: {len(loader)}")
    print("=" * 90)

    for bno, (Xs, ys, pre_texts, post_texts, text_entities, indices) in enumerate(loader):

        # 'round' matches the paper's alpha_t = max(0.1, 0.9985^t); 'batch' is the
        # pre-v5 behaviour (one decay step per batch of 128, so the floor was only
        # reached around round 196k instead of round 1.5k).
        if decay_per == 'batch':
            current_alpha = max(0.1, linucb_alpha * (alpha_decay ** bno))
        Xs, ys = Xs.to(device), ys.to(device)

        if cached_labels is not None:
            llm_labels = torch.tensor(np.asarray(cached_labels)[indices.numpy()])
            llm_sims = None
        else:
            with torch.no_grad():
                llm_labels, llm_sims = language_model_scores(
                    pre_texts, post_texts, t5_tokenizer, t5_model,
                    entity_model, entity_embd, num_entities, cos, device
                )

        for i in range(len(Xs)):
            context_x   = Xs[i].cpu().numpy()
            d_ctx       = context_x.shape[0] // 2
            cx          = context_x[:d_ctx]            # pre-text embedding (768-dim)
            true_label  = ys[i].item()
            llm_choice  = llm_labels[i].item()

            # Base features: element-wise product (K, 768)
            joint_base = action_features * cx

            total_rounds += 1
            if decay_per == 'round':
                current_alpha = max(0.1, linucb_alpha * (alpha_decay ** (total_rounds - 1)))

            # ===========================================================
            # TRACK 1: LLM only (static baseline)
            # ===========================================================
            r_llm = 1.0 if llm_choice == true_label else 0.0
            acc_llm += r_llm

            # ===========================================================
            # TRACK 2: Cold LinUCB (no warm-start, acts every round)
            # ===========================================================
            theta_cold = V_inv_cold @ b_cold
            Vf_cold    = joint_base @ V_inv_cold
            vars_cold  = np.sum(Vf_cold * joint_base, axis=1)
            ucb_cold   = joint_base @ theta_cold + current_alpha * np.sqrt(np.maximum(vars_cold, 0))
            cold_choice = int(np.argmax(ucb_cold))
            r_cold = 1.0 if cold_choice == true_label else 0.0
            acc_cold += r_cold

            x_cold = joint_base[cold_choice]
            V_inv_cold = sm_update(V_inv_cold, x_cold)
            b_cold += r_cold * x_cold

            # ===========================================================
            # TRACK 3: Confidence-Gated Router (Algorithm 1)
            # ===========================================================

            # --- LinUCB's proposal (UCB action selection) ---
            theta_warm = V_inv_warm @ b_warm
            Vf_warm    = joint_base @ V_inv_warm
            vars_warm  = np.sum(Vf_warm * joint_base, axis=1)
            ucb_warm   = joint_base @ theta_warm + current_alpha * np.sqrt(np.maximum(vars_warm, 0))
            linucb_choice = int(np.argmax(ucb_warm))

            # --- Feature vectors for each proposal ---
            x_llm    = joint_base[llm_choice]
            x_linucb = joint_base[linucb_choice]

            # --- Geometric uncertainty scores ---
            # LLM: u = sqrt(φᵀ (c·I)⁻¹ φ) = ‖φ‖ / √c
            u_llm = inv_sqrt_c * np.linalg.norm(x_llm)

            # LinUCB: u = sqrt(φᵀ V⁻¹ φ)  (standard ridge variance)
            u_linucb = np.sqrt(max(0.0, x_linucb @ V_inv_warm @ x_linucb))

            # --- Routing rule ---
            if router == 'gate':
                use_llm = bool(u_llm < u_linucb)
                invprop = 1.0
            elif router == 'corral':
                p = corral_probs()
                use_llm = bool(corral_rng.random() < p[1])
                invprop = 1.0 / p[1 if use_llm else 0]
            else:   # offline: warm start already folded in, no routing
                use_llm = False
                invprop = 1.0

            if use_llm:
                # ─── LLM is more confident → defer to LLM ───
                router_choice = llm_choice
                r_router = 1.0 if router_choice == true_label else 0.0
                llm_rounds += 1

                # PASSIVE COVARIANCE UPDATE:
                # Update LinUCB's design matrix with the LLM's chosen feature.
                # This is the "reward-relevant warm-start coverage" mechanism.
                # LinUCB's confidence ellipsoid shrinks even though it didn't act.
                V_inv_warm = sm_update(V_inv_warm, x_llm)

                if passive_full:
                    # Empirical variant: also update b with observed reward
                    b_warm += r_router * x_llm
                # Else: b is NOT updated (paper's theoretical version).
                # θ = V⁻¹ b stays near 0 during LLM phase, but V becomes
                # well-conditioned. When LinUCB takes over, it refines θ
                # efficiently because V is already well-conditioned.

            else:
                # ─── LinUCB is confident enough → route to LinUCB ───
                router_choice = linucb_choice
                r_router = 1.0 if router_choice == true_label else 0.0

                # FULL UPDATE: both V and b
                V_inv_warm = sm_update(V_inv_warm, x_linucb)
                b_warm += r_router * x_linucb

            if router == 'corral':
                invpalgo = corral_update(1 if use_llm else 0, invprop, r_router)

            acc_router += r_router

            k = total_rounds - 1
            pr_router[k] = int(r_router); pr_cold[k] = int(r_cold)
            pr_llm[k]    = int(r_llm);    pr_routed[k] = int(use_llm)

        # ===========================================================
        # Logging every 10 batches
        # ===========================================================
        if bno % 10 == 0:
            llm_frac = llm_rounds / max(total_rounds, 1)
            history['batch'].append(bno)
            history['hybrid'].append(acc_router.mean())
            history['llm'].append(acc_llm.mean())
            history['linucb_cold'].append(acc_cold.mean())
            history['llm_fraction'].append(llm_frac)

            print(
                f"Batch {bno:4d} | "
                f"Router: {acc_router.mean():.4f} | "
                f"LLM only: {acc_llm.mean():.4f} | "
                f"Cold LinUCB: {acc_cold.mean():.4f} | "
                f"LLM routing: {llm_frac:.1%}"
            )

            with open(fname, "w") as f:
                json.dump(history, f)

        del Xs, ys, pre_texts, post_texts, text_entities, indices, llm_labels, llm_sims
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()

    print("=" * 90)
    print(
        f"FINAL | Router: {acc_router.mean():.4f} | "
        f"LLM only: {acc_llm.mean():.4f} | "
        f"Cold LinUCB: {acc_cold.mean():.4f}"
    )
    print(f"LLM routing: {llm_rounds}/{total_rounds} = {llm_rounds/max(total_rounds,1):.1%}")

    with open(fname, "w") as f:
        json.dump(history, f)
    m = total_rounds
    np.savez_compressed(perround_fname, r_router=pr_router[:m], r_cold=pr_cold[:m],
                        r_llm=pr_llm[:m], routed=pr_routed[:m], step=m,
                        seed=seed, router=router)
    routed = pr_routed[:m].astype(bool)
    A = float((pr_router[:m][routed]  - pr_cold[:m][routed]).sum())
    V = float((pr_router[:m][~routed] - pr_cold[:m][~routed]).sum())
    B = float((pr_router[:m][~routed] - pr_llm[:m][~routed]).sum())
    tot = float(pr_router[:m].sum() - pr_cold[:m].sum())
    print(f"Decomposition: A={A:.0f}  V={V:.0f}  B={B:.0f}  "
          f"(A+V={A+V:.0f} must equal sum(r_router)-sum(r_cold)={tot:.0f})")
    print(f"Results saved to: {fname}, {perround_fname}")
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Algorithm 1: Confidence-Gated Routing with Passive Warm-Start'
    )
    parser.add_argument('--seed',           type=int,   default=1)
    parser.add_argument('--llm_type',       type=str,   default='small',
                        choices=['small', 'base', 'large'],
                        help='LLM size. Use "small" to clearly show the hand-off.')
    parser.add_argument('--linucb_alpha',   type=float, default=1.0,
                        help='UCB exploration bonus scale')
    parser.add_argument('--lam',            type=float, default=0.1,
                        help='LinUCB ridge regularization')
    parser.add_argument('--llm_confidence', type=float, default=100.0,
                        help='Effective offline sample count c for LLM design matrix. '
                             'Higher → longer LLM phase before hand-off. '
                             'Try 50, 100, 500.')
    parser.add_argument('--passive_full',   action='store_true',
                        help='If set, passive updates include reward in b. '
                             'Default: covariance only (paper\'s theoretical version).')
    parser.add_argument('--llm_cache',      type=str,   default=None,
                        help='Path to llm_cache_v1_<size>.pkl.gz. With a cache the run '
                             'is pure CPU numpy (no Flan-T5 load, no GPU).')
    parser.add_argument('--router',         type=str,   default='gate',
                        choices=['gate', 'corral', 'offline'])
    parser.add_argument('--offline_tau',    type=int,   default=5000,
                        help='--router offline: held-out LLM pulls for the warm start')
    parser.add_argument('--corral_eta',     type=float, default=1.0)
    parser.add_argument('--corral_pmin',    type=float, default=0.2)
    parser.add_argument('--decay_per',      type=str,   default='round',
                        choices=['round', 'batch'],
                        help="'round' = the paper's alpha_t = max(0.1, 0.9985^t). "
                             "'batch' reproduces the pre-v5 runs (decay per batch of 128).")
    args = parser.parse_args()

    mydata = loadMyDataset(2000)
    learnOnline(
        mydata, batch_size=128, cuda=True, seed=args.seed,
        llm_type=args.llm_type,
        linucb_alpha=args.linucb_alpha, lam=args.lam,
        llm_confidence=args.llm_confidence,
        passive_full=args.passive_full,
        llm_cache_path=args.llm_cache,
        router=args.router,
        offline_tau=args.offline_tau,
        corral_eta=args.corral_eta,
        corral_pmin=args.corral_pmin,
        decay_per=args.decay_per,
    )
