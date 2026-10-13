"""
compute_diagnostics.py  (AmazonCat-13K version)

Compute regime diagnostics from Section 3.1 of the paper.

PHASE 1 — Dataset-only diagnostics (LLM-free, match the 3 theoretical axes):

  Axis A. Feature Coherence (alpha_hat):
     Median of minimum pairwise cosine similarity across entity embeddings.
     Threshold: alpha_hat >= -0.5 -> LinUCB suffices (Lemma 2).
     NOTE: With 13,330 entities the full (K, K) cosine matrix is ~700 MB.
     We compute it in blocks to stay memory-friendly.

  Axis B. Effective Rank Ratio (r_hat / sqrt(T)):
     Effective rank of the CONTEXT covariance, divided by sqrt(T).
     Threshold: r_hat / sqrt(T) < 1 is the easy-mode indicator.

  Axis C. Action-Gap Distribution (oracle linear model):
     Fit an oracle theta* via ridge regression on joint features phi(x,a) = x * z_a.
     AmazonCat is MULTI-LABEL: reward(t, a) = 1 if a in ys[t], else 0.
     The RHS of the normal equation sums over ALL correct labels per sample.
     Then compute per-sample gap = mu_t(best) - mu_t(2nd best).
     Threshold: frac(gap <= 0.01) > 0.5 -> "heavy" tail (hard).

PHASE 2 — LLM suitability diagnostic (requires LLM cache):

  B+C. Subspace Mismatch (mean principal angle):
     Angle between the rank-r subspace of the LLM-induced design matrix
     and the context covariance.

Usage:
  # Phase 1 only (no LLM needed) for v1
  python -u compute_diagnostics.py --model_family v1

  # Phase 1 + Phase 2 (with LLM caches)
  python -u compute_diagnostics.py --model_family v1 --llm_check

  # Specific LLM types only (e.g. skip large if not cached yet)
  python -u compute_diagnostics.py --model_family v1 --llm_check --llm_types small base

Inputs:
  mydataset.0.pickle.gz           — v1 dataset (Xs: 1.18M x 1536, labelfeats: 13330 x 768d)
  mydataset.0.v3.pickle.gz        — v3 dataset (Xs: 1.18M x 2048, labelfeats: 13330 x 1024d)
  llm_cache_amazon_v1_{small,base,large}.pkl.gz — LLM caches (only for Phase 2)
"""

import argparse
import gzip
import pickle
import sys
import numpy as np
import torch
from scipy.linalg import svd, subspace_angles


# ================================================================
# Dataset loading (need the class stub for pickle)
# ================================================================
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs): pass
    def __len__(self): return self.Xs.shape[0]
    def __getitem__(self, i): return self.Xs[i]

sys.modules['__main__'].MyDataset = MyDataset


def load_dataset(family):
    """Load the pickle dataset for the given model family."""
    if family == 'v1':
        path = 'mydataset.0.pickle.gz'
    elif family == 'v3':
        path = 'mydataset.0.v3.pickle.gz'
    else:
        raise ValueError(f"Unknown family: {family}")
    print(f"Loading {path} ...")
    with gzip.open(path, 'rb') as f:
        ds = pickle.load(f)
    return ds


def load_llm_cache(family, llm_type):
    """Load the pre-computed LLM predictions."""
    path = f'llm_cache_amazon_{family}_{llm_type}.pkl.gz'
    print(f"Loading {path} ...")
    with gzip.open(path, 'rb') as f:
        cache = pickle.load(f)
    return cache


def get_entity_embeddings(ds):
    """Extract entity feature matrix from dataset.labelfeats.
    Returns: (num_entities, embed_dim) numpy array, sorted by entity ordinal.
    """
    d = {}
    for k in ds.labelfeats:
        idx, embd = ds.labelfeats[k]
        d[idx] = np.array(embd, dtype=np.float32)
    return np.stack([d[i] for i in sorted(d.keys())])


def get_context_embeddings(ds):
    """Extract context embeddings from Xs.
    Xs is (N, 2*embed_dim) = concat of title and content embeddings.
    We use the first half (title) as the context representation.
    Returns: (N, embed_dim) numpy array.
    """
    Xs = ds.Xs.numpy() if isinstance(ds.Xs, torch.Tensor) else ds.Xs
    d = Xs.shape[1] // 2
    return Xs[:, :d].astype(np.float32)


def get_true_labels(ds):
    """Extract ground-truth label lists from dataset.
    AmazonCat is multi-label: ys[i] = list of correct entity ordinals.
    Returns: list of lists.
    """
    return ds.ys


# ================================================================
# Axis A: Feature Coherence (alpha_hat)
# ================================================================
def compute_feature_coherence(entity_embeddings, block_size=2000):
    """
    Median of minimum pairwise cosine similarity between action features.

    Paper definition: alpha_hat = median_a min_{a' != a} cos(z_a, z_{a'})

    With 13,330 entities the full cosine matrix is ~710 MB in float32.
    We compute it in row-blocks to keep peak memory reasonable.
    """
    K, d = entity_embeddings.shape
    print(f"  Computing feature coherence for {K} entities (block_size={block_size}) ...")

    # Normalize
    norms = np.linalg.norm(entity_embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = entity_embeddings / norms

    min_cosines = np.empty(K, dtype=np.float32)

    for start in range(0, K, block_size):
        end = min(start + block_size, K)
        # (block, K) cosine similarities
        block_cos = normed[start:end] @ normed.T  # (block, K)

        # Mask out self-similarity
        for i in range(end - start):
            block_cos[i, start + i] = np.inf

        min_cosines[start:end] = np.min(block_cos, axis=1)
        print(f"    processed entities {start}..{end-1}")

    alpha_hat = float(np.median(min_cosines))
    print(f"  Feature Coherence (alpha_hat): {alpha_hat:.4f}")
    print(f"    mean min-cosine: {np.mean(min_cosines):.4f}")
    print(f"    min  min-cosine: {np.min(min_cosines):.4f}")
    print(f"    max  min-cosine: {np.max(min_cosines):.4f}")
    return alpha_hat


# ================================================================
# Axis B: Effective Rank Ratio (r_hat / sqrt(T))
# ================================================================
def compute_effective_rank_ratio(context_embeddings, T, n_samples=5000, seed=42):
    """
    Effective rank of the CONTEXT covariance (not joint feature covariance).

    r_hat = tr(Sigma_hat) / ||Sigma_hat||_op
    Operationally: r_hat / sqrt(T) < 1 is the easy-mode check.
    """
    rng = np.random.RandomState(seed)
    N = context_embeddings.shape[0]
    idx = rng.choice(N, size=min(n_samples, N), replace=False)
    contexts_sub = context_embeddings[idx]  # (n_samples, d)

    n = len(idx)

    print(f"  Computing context covariance from {n} samples ...")

    # Context second-moment: (d, d)
    cov = (contexts_sub.T @ contexts_sub) / n

    # Eigenvalues (symmetric matrix)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[::-1]  # descending

    trace = np.sum(eigvals)
    max_eig = eigvals[0]
    r_hat = trace / max_eig
    # Paper formula: Axis A saturates when r log T <= c sqrt(T)
    # Equivalently: r log(T) / sqrt(T) <= c_0
    # We use c_0 = 0.1 as the saturation threshold
    ratio = r_hat * np.log(T) / np.sqrt(T)

    print(f"  Effective Rank (r_hat): {r_hat:.2f}")
    print(f"  sqrt(T): {np.sqrt(T):.2f}")
    print(f"  log(T): {np.log(T):.2f}")
    print(f"  Effective Rank Ratio (r_hat * log(T) / sqrt(T)): {ratio:.4f}")
    print(f"    top-5 eigenvalues: {eigvals[:5]}")
    return r_hat, ratio, cov


# ================================================================
# Axis C: Action-Gap Distribution (oracle linear model, LLM-free)
#   ** MULTI-LABEL version for AmazonCat **
# ================================================================
def compute_gap_distribution(context_embeddings, entity_embeddings,
                             true_labels_list, n_fit=50000,
                             reg_lambda=1.0, seed=42, eta_values=None):
    """
    Oracle action gap under the fitted linear model.

    Fits theta* via ridge regression on joint features phi(x,a) = x * z_a.
    AmazonCat is multi-label: reward(t, a) = 1 if a in ys[t], else 0.

    Normal equation using Hadamard identity:
      A = (X^T X) * (Z^T Z)    -- same as single-label (all actions contribute)
      b = sum_i sum_{a in ys[i]} (x_i * z_a)  -- sum over ALL correct labels

    Then computes per-sample gap = mu_t(best) - mu_t(2nd best).
    """
    if eta_values is None:
        eta_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

    rng = np.random.RandomState(seed)
    N, d = context_embeddings.shape
    K = entity_embeddings.shape[0]

    # --- Step 1: Fit oracle theta* via ridge regression ---
    n = min(n_fit, N)
    idx = rng.choice(N, size=n, replace=False)
    X_fit = context_embeddings[idx].astype(np.float64)  # (n, d)
    Z = entity_embeddings.astype(np.float64)             # (K, d)

    print(f"  Fitting oracle theta* from {n} samples x {K} actions (multi-label) ...")

    # LHS: A = (X^T X) * (Z^T Z)  -- Hadamard product (d, d)
    XtX = X_fit.T @ X_fit          # (d, d)
    ZtZ = Z.T @ Z                  # (d, d)
    A = XtX * ZtZ                  # Hadamard product (d, d)

    # RHS: sum over all correct labels per sample
    # b = sum_i sum_{a in ys[i]} (x_i * z_a)
    b = np.zeros(d, dtype=np.float64)
    for local_i, global_i in enumerate(idx):
        x_i = X_fit[local_i]                  # (d,)
        for a in true_labels_list[global_i]:   # iterate over correct labels
            b += x_i * Z[a]                    # Hadamard: x_i * z_a

    # Ridge: theta* = (A + lambda I)^{-1} b
    A += reg_lambda * np.eye(d, dtype=np.float64)
    theta_star = np.linalg.solve(A, b)    # (d,)

    print(f"  theta* norm: {np.linalg.norm(theta_star):.6f}")
    print(f"  theta* range: [{theta_star.min():.6f}, {theta_star.max():.6f}]")

    # --- Step 2: Compute gaps over all samples ---
    # mu_t(a) = <x_t * z_a, theta*> = (x_t * theta*) . z_a
    print(f"  Computing oracle gaps for {N} samples x {K} actions ...")

    Z_f32 = entity_embeddings                          # (K, d)
    theta_f32 = theta_star.astype(np.float32)          # (d,)

    batch_size = 5000  # smaller batches: (5000, 13330) ~ 260 MB per mu
    all_gaps = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_ctx = context_embeddings[start:end]      # (B, d)

        # Weighted context: x_t * theta*
        weighted = batch_ctx * theta_f32[None, :]      # (B, d)

        # Expected rewards for all actions: mu_t(a) = weighted @ Z^T
        mu = weighted @ Z_f32.T                        # (B, K)

        # Gap = mu(best) - mu(2nd best)
        top2_idx = np.argpartition(mu, -2, axis=1)[:, -2:]  # (B, 2)
        top2_vals = np.take_along_axis(mu, top2_idx, axis=1)  # (B, 2)
        top2_sorted = np.sort(top2_vals, axis=1)              # ascending
        gaps = top2_sorted[:, 1] - top2_sorted[:, 0]          # (B,)
        all_gaps.append(gaps)

        if (start // batch_size) % 20 == 0:
            print(f"    processed {end}/{N} samples ...")

    gaps = np.concatenate(all_gaps)

    # Oracle accuracy: how often is the oracle's best action in the true label set?
    batch_size_check = 5000
    oracle_correct = 0
    for start in range(0, N, batch_size_check):
        end = min(start + batch_size_check, N)
        batch_ctx = context_embeddings[start:end]
        weighted = batch_ctx * theta_f32[None, :]
        mu = weighted @ Z_f32.T
        oracle_best = np.argmax(mu, axis=1)
        for local_i in range(end - start):
            if oracle_best[local_i] in true_labels_list[start + local_i]:
                oracle_correct += 1
    oracle_acc = oracle_correct / N

    print(f"  Oracle linear model accuracy (top-1 in true set): {oracle_acc:.4f} ({oracle_acc*100:.2f}%)")
    print(f"  Action-Gap Distribution (oracle linear model):")
    print(f"    mean gap: {np.mean(gaps):.6f}")
    print(f"    median gap: {np.median(gaps):.6f}")
    print(f"    std gap: {np.std(gaps):.6f}")
    print(f"    min gap: {np.min(gaps):.6f}")
    print(f"    max gap: {np.max(gaps):.6f}")
    print(f"    frac(gap <= 0): {np.mean(gaps <= 0):.4f}")

    tail_label = "low"
    for eta in eta_values:
        frac = np.mean(gaps <= eta)
        print(f"    frac(gap <= {eta:.3f}): {frac:.4f}")
        if eta == 0.01 and frac > 0.5:
            tail_label = "heavy"

    print(f"  Tail classification: {tail_label}")
    return gaps, tail_label


# ================================================================
# LLM Suitability Check: Subspace Mismatch (mean principal angle)
# ================================================================
def compute_subspace_mismatch(context_embeddings, entity_embeddings,
                              llm_labels, r_hat, cov_ctx=None,
                              n_samples=5000, seed=42):
    """
    PHASE 2 -- LLM suitability check (requires LLM predictions).

    Mean principal angle between:
      - The rank-r subspace of the LLM-induced design matrix
      - The rank-r subspace of the context covariance
    """
    rng = np.random.RandomState(seed)
    N = context_embeddings.shape[0]
    idx = rng.choice(N, size=min(n_samples, N), replace=False)
    contexts_sub = context_embeddings[idx]
    labels_sub = llm_labels[idx]

    d = contexts_sub.shape[1]
    r = max(1, min(int(round(r_hat)), d - 1))

    print(f"  Computing subspace mismatch (rank={r}) from {len(idx)} samples ...")

    # LLM-induced covariance: phi(x_i, a_LLM_i) for each sample
    llm_features = contexts_sub * entity_embeddings[labels_sub]  # (n, d)
    cov_llm = (llm_features.T @ llm_features) / len(idx)  # (d, d)

    # Context covariance (reuse from Axis B if available)
    if cov_ctx is None:
        cov_ctx = (contexts_sub.T @ contexts_sub) / len(idx)

    # Top-r eigenvectors of each covariance
    _, _, Vh_llm = svd(cov_llm)
    _, _, Vh_ctx = svd(cov_ctx)
    U_llm = Vh_llm[:r, :].T   # (d, r)
    U_ctx = Vh_ctx[:r, :].T   # (d, r)

    # Principal angles between the two subspaces
    angles_rad = subspace_angles(U_llm, U_ctx)
    angles_deg = np.degrees(angles_rad)
    mean_angle = float(np.mean(angles_deg))

    print(f"  Subspace Mismatch:")
    print(f"    mean principal angle: {mean_angle:.1f} degrees")
    print(f"    all angles (deg): {np.round(angles_deg, 1).tolist()}")
    return mean_angle


# ================================================================
# Main pipeline
# ================================================================
def run_phase1(family, horizon=None):
    """Phase 1: Dataset-only regime diagnostics (LLM-free)."""
    ds = load_dataset(family)
    entity_emb = get_entity_embeddings(ds)
    context_emb = get_context_embeddings(ds)
    true_labels = get_true_labels(ds)
    T = horizon if horizon is not None else len(ds)

    print(f"\n{'='*70}")
    print(f"PHASE 1 -- Dataset Regime Diagnostics (LLM-free): {family}")
    print(f"  Contexts: {context_emb.shape}, Entities: {entity_emb.shape}, T={T}")
    print(f"{'='*70}")

    # Axis A: Feature Coherence
    print(f"\n--- Axis A: Feature Coherence ---")
    alpha_hat = compute_feature_coherence(entity_emb)

    # Axis B: Effective Rank Ratio
    print(f"\n--- Axis B: Effective Rank Ratio ---")
    r_hat, ratio, cov_ctx = compute_effective_rank_ratio(context_emb, T)

    # Axis C: Action-Gap Distribution (multi-label oracle)
    print(f"\n--- Axis C: Action-Gap Distribution (oracle, multi-label) ---")
    gaps, tail_label = compute_gap_distribution(context_emb, entity_emb, true_labels)

    # Regime prediction from 3 axes
    print(f"\n{'='*70}")
    print(f"PHASE 1 SUMMARY: {family}")
    print(f"  alpha_hat (coherence):     {alpha_hat:.4f}")
    print(f"  r_hat:                     {r_hat:.2f}")
    print(f"  r_hat * log(T) / sqrt(T):  {ratio:.4f}")
    print(f"  gap tail:                  {tail_label}")

    axes_saturated = 0
    if alpha_hat >= -0.5:
        axes_saturated += 1
        print(f"  Axis A: SATURATED (alpha_hat={alpha_hat:.2f} >= -0.5)")
    else:
        print(f"  Axis A: DEVIATING (alpha_hat={alpha_hat:.2f} < -0.5)")

    if ratio < 0.1:
        axes_saturated += 1
        print(f"  Axis B: SATURATED (r_hat*log(T)/sqrt(T)={ratio:.4f} < 0.1)")
    else:
        print(f"  Axis B: DEVIATING (r_hat*log(T)/sqrt(T)={ratio:.4f} >= 0.1)")

    if tail_label == "low":
        axes_saturated += 1
        print(f"  Axis C: SATURATED (low tail)")
    else:
        print(f"  Axis C: DEVIATING (heavy tail)")

    if axes_saturated >= 2:
        regime = "LinUCB-suffices"
    elif axes_saturated == 1:
        regime = "gap-suffices / marginal"
    else:
        regime = "warm-start-essential"

    print(f"\n  PREDICTED REGIME: {regime}")
    print(f"{'='*70}\n")

    return {
        'family': family,
        'alpha_hat': alpha_hat,
        'r_hat': r_hat,
        'ratio_r_sqrtT': ratio,
        'gap_tail': tail_label,
        'regime': regime,
        '_context_emb': context_emb,
        '_entity_emb': entity_emb,
        '_true_labels': true_labels,
        '_cov_ctx': cov_ctx,
    }


def run_phase2(phase1_result, llm_type):
    """Phase 2: LLM suitability check (requires LLM cache)."""
    family = phase1_result['family']
    r_hat = phase1_result['r_hat']
    context_emb = phase1_result['_context_emb']
    entity_emb = phase1_result['_entity_emb']
    true_labels = phase1_result['_true_labels']
    cov_ctx = phase1_result['_cov_ctx']

    cache = load_llm_cache(family, llm_type)

    print(f"\n{'='*70}")
    print(f"PHASE 2 -- LLM Suitability Check: {family} / {llm_type}")
    print(f"{'='*70}")

    print(f"\n--- Subspace Mismatch ---")
    mean_angle = compute_subspace_mismatch(
        context_emb, entity_emb, cache['labels'], r_hat, cov_ctx=cov_ctx
    )

    # LLM accuracy: is the LLM's predicted label in the true label set?
    llm_preds = cache['labels']
    N = len(true_labels)
    llm_correct = 0
    for i in range(N):
        if llm_preds[i] in true_labels[i]:
            llm_correct += 1
    llm_acc = llm_correct / N

    print(f"\n  LLM accuracy (top-1 in true set): {llm_acc:.4f} ({llm_acc*100:.2f}%)")
    print(f"  Subspace mismatch angle: {mean_angle:.1f} deg")

    if mean_angle > 45:
        print(f"  VERDICT: HIGH MISMATCH")
    elif mean_angle > 20:
        print(f"  VERDICT: MODERATE MISMATCH")
    else:
        print(f"  VERDICT: LOW MISMATCH")

    print(f"{'='*70}\n")

    return {
        'family': family,
        'llm_type': llm_type,
        'subspace_angle': mean_angle,
        'llm_accuracy': llm_acc,
    }


def main():
    parser = argparse.ArgumentParser(description='AmazonCat-13K regime diagnostics')
    parser.add_argument('--model_family', type=str, default='v1',
                        choices=['v1', 'v3', 'all'],
                        help='Which model family dataset to diagnose')
    parser.add_argument('--llm_check', action='store_true',
                        help='Also run Phase 2 (LLM suitability)')
    parser.add_argument('--llm_types', nargs='+', default=['small', 'base', 'large'],
                        help='Which LLM sizes to check (default: small base large)')
    parser.add_argument('--n_samples', type=int, default=5000,
                        help='Number of contexts to subsample for covariance computation')
    parser.add_argument('--horizon', type=int, default=None,
                        help='Experiment horizon T for Axis A check (default: dataset size)')
    args = parser.parse_args()

    families = ['v1', 'v3'] if args.model_family == 'all' else [args.model_family]

    # ---- Phase 1 ----
    phase1_results = []
    for family in families:
        res = run_phase1(family, horizon=args.horizon)
        phase1_results.append(res)

    # Phase 1 summary table
    print("\n" + "=" * 90)
    print("PHASE 1 SUMMARY -- AmazonCat-13K Dataset Regime Diagnostics (LLM-free)")
    print("=" * 90)
    print(f"{'Dataset':<15} {'alpha_hat':>10} {'r_hat':>8} {'r/sqrtT':>10} {'gap_tail':>10} {'regime':<25}")
    print("-" * 90)
    for r in phase1_results:
        print(f"{r['family']:<15} {r['alpha_hat']:>10.4f} {r['r_hat']:>8.2f} {r['ratio_r_sqrtT']:>10.4f} {r['gap_tail']:>10} {r['regime']:<25}")
    print("=" * 90)

    # ---- Phase 2 (optional) ----
    if args.llm_check:
        phase2_results = []
        for p1 in phase1_results:
            for llm_type in args.llm_types:
                res = run_phase2(p1, llm_type)
                phase2_results.append(res)

        print("\n" + "=" * 90)
        print("PHASE 2 SUMMARY -- LLM Suitability Check")
        print("=" * 90)
        print(f"{'Config':<25} {'LLM Acc':>10} {'Angle':>10} {'Verdict':<30}")
        print("-" * 90)
        for r in phase2_results:
            name = f"{r['family']}/{r['llm_type']}"
            if r['subspace_angle'] > 45:
                verdict = "HIGH MISMATCH"
            elif r['subspace_angle'] > 20:
                verdict = "MODERATE MISMATCH"
            else:
                verdict = "LOW MISMATCH"
            print(f"{name:<25} {r['llm_accuracy']*100:>9.2f}% {r['subspace_angle']:>9.1f} deg {verdict:<30}")
        print("=" * 90)


if __name__ == '__main__':
    main()
