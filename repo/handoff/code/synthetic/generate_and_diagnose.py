"""
generate_and_diagnose.py — Synthetic Datasets for Warm-Start-Essential Regime

Creates 3 synthetic datasets designed to make ALL 3 axes deviate,
then runs diagnostics and Algorithm 1 on them.

Option A: "Standard-basis arms" (d=200, K=200, T=5000)
  - Entity embeddings = standard basis vectors (perfectly orthogonal)
  - Context = isotropic Gaussian
  - θ* = small random vector → vanishing gaps
  - Mirrors Proposition 1 in the paper

Option B: "Hidden subspace" (d=200, K=100, T=5000)
  - θ* lives in a rank-20 subspace
  - Actions have signal in the 20-dim subspace + noise in the rest
  - LinUCB must explore all d dimensions; LLM focuses on the right 20

Option C: "Random orthogonal arms" (d=100, K=100, T=5000)
  - Entity embeddings = random orthogonal vectors (QR)
  - θ* sparse in first 20 dims
  - Contexts = isotropic Gaussian

For each dataset, we:
  1. Compute Phase 1 diagnostics (3 axes)
  2. Simulate "LLMs" at different accuracy levels
  3. Compute Phase 2 diagnostic (subspace mismatch) for each LLM
  4. Run Algorithm 1 for configs that predict warm-start-essential

Usage:
  python -u generate_and_diagnose.py              # Full pipeline
  python -u generate_and_diagnose.py --phase1     # Only diagnostics
  python -u generate_and_diagnose.py --run A      # Run Algorithm 1 on Option A only
"""

import argparse
import json
import numpy as np
from scipy.linalg import svd, subspace_angles, qr
from dataclasses import dataclass


# ================================================================
# Dataset generation
# ================================================================
@dataclass
class SyntheticDataset:
    name: str
    description: str
    d: int           # feature dimension
    K: int           # number of actions
    T: int           # horizon
    Z: np.ndarray    # entity embeddings (K, d)
    theta_star: np.ndarray  # true reward parameter (d,)
    noise_std: float
    contexts: np.ndarray  # (T, d) — pre-generated contexts for diagnostics
    # LLM simulations stored separately


def generate_option_A(T=5000, seed=42):
    """Option A: Anti-podal basis arms (d=100, K=200).

    Take 100 standard basis vectors + their negatives → 200 actions.
    Every entity has a "partner" with cosine = -1, so alpha_hat = -1.
    """
    rng = np.random.RandomState(seed)
    d = 100
    K = 200

    # Entity embeddings = standard basis + negatives (anti-podal)
    Z_pos = np.eye(d, dtype=np.float32)       # (100, 100)
    Z_neg = -np.eye(d, dtype=np.float32)      # (100, 100)
    Z = np.concatenate([Z_pos, Z_neg], axis=0)  # (200, 100)

    # theta* = small random vector (makes gaps vanishing)
    theta_star = rng.randn(d).astype(np.float32) * 0.01

    # Contexts = isotropic Gaussian
    contexts = rng.randn(T, d).astype(np.float32)

    return SyntheticDataset(
        name="A",
        description="Anti-podal basis arms (d=100, K=200)",
        d=d, K=K, T=T,
        Z=Z,
        theta_star=theta_star,
        noise_std=0.1,
        contexts=contexts,
    )


def generate_option_B(T=5000, seed=42):
    """Option B: Hidden subspace with anti-podal pairs (d=200, K=100).

    50 random unit vectors in R^200 + their negatives → 100 actions.
    theta* lives in a rank-20 subspace.
    """
    rng = np.random.RandomState(seed)
    d = 200
    K = 100
    signal_dim = 20

    # theta* lives in first 20 dimensions only
    theta_star = np.zeros(d, dtype=np.float32)
    theta_star[:signal_dim] = rng.randn(signal_dim).astype(np.float32) * 0.05

    # Entity embeddings: 50 random + 50 negatives (anti-podal)
    Z_half = rng.randn(K // 2, d).astype(np.float32)
    Z_half = Z_half / (np.linalg.norm(Z_half, axis=1, keepdims=True) + 1e-8)
    Z = np.concatenate([Z_half, -Z_half], axis=0)  # (100, 200)

    # Contexts = isotropic Gaussian (high effective rank)
    contexts = rng.randn(T, d).astype(np.float32)

    return SyntheticDataset(
        name="B",
        description="Hidden subspace + anti-podal (d=200, K=100)",
        d=d, K=K, T=T,
        Z=Z,
        theta_star=theta_star,
        noise_std=0.1,
        contexts=contexts,
    )


def generate_option_C(T=5000, seed=42):
    """Option C: Random orthogonal + anti-podal arms (d=50, K=100).

    50 orthogonal vectors in R^50 + their negatives → 100 actions.
    theta* sparse in first 20 dims.
    Using d=50 with K=100 ensures both Axis A deviates and Axis B deviates.
    """
    rng = np.random.RandomState(seed)
    d = 50
    K = 100
    signal_dim = 20

    # Entity embeddings: 50 orthogonal vectors + negatives via QR
    M = rng.randn(d, d).astype(np.float64)
    Q, _ = qr(M)  # Q is (d, d) orthogonal
    Z_half = Q.T.astype(np.float32)  # (50, 50) — orthonormal rows
    Z = np.concatenate([Z_half, -Z_half], axis=0)  # (100, 50) — anti-podal

    # theta* sparse: only first 20 dims are nonzero
    theta_star = np.zeros(d, dtype=np.float32)
    theta_star[:signal_dim] = rng.randn(signal_dim).astype(np.float32) * 0.03

    # Contexts = isotropic Gaussian
    contexts = rng.randn(T, d).astype(np.float32)

    return SyntheticDataset(
        name="C",
        description="Orthogonal + anti-podal arms (d=50, K=100, signal_dim=20)",
        d=d, K=K, T=T,
        Z=Z,
        theta_star=theta_star,
        noise_std=0.1,
        contexts=contexts,
    )


# ================================================================
# Simulated LLM (noisy oracle)
# ================================================================
def simulate_llm_predictions(ds, accuracy_target=0.3, noise_scale=None, seed=123):
    """
    Simulate an LLM that picks actions based on noisy knowledge of the reward.

    The LLM computes approximate action scores and picks the argmax.
    We tune noise_scale so that the resulting accuracy ≈ accuracy_target.
    """
    rng = np.random.RandomState(seed)
    T, d = ds.contexts.shape
    K = ds.K

    # True scores: mu(x, a) = <theta*, x ⊙ z_a> for all actions
    # = (x * theta*) @ Z^T
    weighted = ds.contexts * ds.theta_star[None, :]  # (T, d)
    true_scores = weighted @ ds.Z.T  # (T, K)

    # Find noise_scale by binary search if not given
    if noise_scale is None:
        # Binary search for the noise that gives target accuracy
        lo, hi = 0.0, 10.0
        for _ in range(30):
            mid = (lo + hi) / 2
            noisy = true_scores + rng.randn(T, K) * mid
            acc = np.mean(noisy.argmax(axis=1) == true_scores.argmax(axis=1))
            if acc > accuracy_target:
                lo = mid
            else:
                hi = mid
        noise_scale = (lo + hi) / 2

    # Generate final predictions with the calibrated noise
    rng2 = np.random.RandomState(seed + 1000)
    noisy_scores = true_scores + rng2.randn(T, K) * noise_scale
    llm_labels = noisy_scores.argmax(axis=1).astype(np.int64)

    # Actual accuracy
    oracle_labels = true_scores.argmax(axis=1)
    actual_acc = np.mean(llm_labels == oracle_labels)

    return llm_labels, actual_acc, noise_scale


# ================================================================
# Diagnostics (same as paper)
# ================================================================
def compute_feature_coherence(Z):
    """Axis A: alpha_hat = median_a min_{a'≠a} cos(z_a, z_a')."""
    K, d = Z.shape
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = Z / norms

    cos_matrix = normed @ normed.T  # (K, K)
    np.fill_diagonal(cos_matrix, np.inf)  # exclude self
    min_cosines = np.min(cos_matrix, axis=1)
    alpha_hat = float(np.median(min_cosines))
    return alpha_hat


def compute_effective_rank_ratio(contexts, T, n_samples=None):
    """Axis B: r_hat / sqrt(T)."""
    if n_samples is not None and n_samples < contexts.shape[0]:
        rng = np.random.RandomState(42)
        idx = rng.choice(contexts.shape[0], n_samples, replace=False)
        ctx = contexts[idx]
    else:
        ctx = contexts

    n = ctx.shape[0]
    cov = (ctx.T @ ctx) / n  # (d, d)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[::-1]  # descending

    trace = np.sum(eigvals)
    max_eig = eigvals[0]
    r_hat = trace / max_eig
    ratio = r_hat / np.sqrt(T)
    return r_hat, ratio, cov


def compute_gap_distribution(ds):
    """Axis C: fraction of oracle gaps <= 0.01."""
    contexts = ds.contexts
    Z = ds.Z
    theta = ds.theta_star
    T = ds.T

    # mu(x, a) = <theta*, x ⊙ z_a> = (x * theta*) @ Z^T
    weighted = contexts * theta[None, :]  # (T, d)
    mu = weighted @ Z.T  # (T, K)

    # Gap = mu(best) - mu(2nd best)
    top2_idx = np.argpartition(mu, -2, axis=1)[:, -2:]
    top2_vals = np.take_along_axis(mu, top2_idx, axis=1)
    top2_sorted = np.sort(top2_vals, axis=1)
    gaps = top2_sorted[:, 1] - top2_sorted[:, 0]

    frac_01 = float(np.mean(gaps <= 0.01))
    tail_label = "heavy" if frac_01 > 0.5 else "low"

    # Oracle accuracy (vs uniform random)
    oracle_best = mu.argmax(axis=1)

    return gaps, frac_01, tail_label, oracle_best


def compute_subspace_mismatch(contexts, Z, llm_labels, r_hat, cov_ctx=None):
    """Phase 2: mean principal angle between LLM design matrix and context covariance."""
    d = contexts.shape[1]
    r = max(1, min(int(round(r_hat)), d - 1))

    # LLM-induced features: phi(x_i, a_LLM_i) = x_i * z_{a_LLM_i}
    llm_features = contexts * Z[llm_labels]  # (T, d)
    cov_llm = (llm_features.T @ llm_features) / len(contexts)  # (d, d)

    # Context covariance
    if cov_ctx is None:
        cov_ctx = (contexts.T @ contexts) / len(contexts)

    # Top-r subspaces
    _, _, Vh_llm = svd(cov_llm)
    _, _, Vh_ctx = svd(cov_ctx)
    U_llm = Vh_llm[:r, :].T   # (d, r)
    U_ctx = Vh_ctx[:r, :].T   # (d, r)

    angles_rad = subspace_angles(U_llm, U_ctx)
    angles_deg = np.degrees(angles_rad)
    mean_angle = float(np.mean(angles_deg))
    return mean_angle, angles_deg


# ================================================================
# Algorithm 1: Confidence-Gated Routing
# ================================================================
def run_algorithm1(ds, llm_labels, c=100.0, lam=0.1, alpha=1.0,
                   alpha_decay=0.9999, seed=42):
    """
    Run Algorithm 1 (confidence-gated routing) on synthetic dataset.

    Returns dict with cumulative rewards for Router, LLM-only, Cold-LinUCB.
    """
    rng = np.random.RandomState(seed)
    d = ds.d
    K = ds.K
    T = ds.T
    Z = ds.Z
    theta_star = ds.theta_star

    # Initialize
    V_inv_warm = np.eye(d, dtype=np.float64) / lam
    b_warm = np.zeros(d, dtype=np.float64)
    V_inv_cold = np.eye(d, dtype=np.float64) / lam
    b_cold = np.zeros(d, dtype=np.float64)

    cum_reward_router = 0.0
    cum_reward_llm = 0.0
    cum_reward_cold = 0.0
    llm_count = 0

    history = []

    for t in range(T):
        x_t = ds.contexts[t].astype(np.float64)

        # --- Cold LinUCB action ---
        theta_cold = V_inv_cold @ b_cold
        weighted_cold = x_t * theta_cold
        mu_cold = Z.astype(np.float64) @ weighted_cold

        # UCB for cold
        Cx_cold = V_inv_cold @ x_t
        ucb_terms_cold = np.zeros(K)
        for a in range(K):
            za = Z[a].astype(np.float64)
            phi_a = x_t * za
            ucb_terms_cold[a] = alpha * np.sqrt(max(0, phi_a @ V_inv_cold @ phi_a))
        a_cold = int(np.argmax(mu_cold + ucb_terms_cold))

        # --- Router decision ---
        a_llm = int(llm_labels[t])
        phi_llm = x_t * Z[a_llm].astype(np.float64)

        # LLM confidence: norm of feature / sqrt(c)
        u_llm = np.linalg.norm(phi_llm) / np.sqrt(c)

        # LinUCB confidence on the LLM's action
        u_linucb = np.sqrt(max(0, phi_llm @ V_inv_warm @ phi_llm))

        if u_llm < u_linucb:
            # Defer to LLM
            a_router = a_llm
            llm_count += 1
        else:
            # Use warm LinUCB
            theta_warm = V_inv_warm @ b_warm
            weighted_warm = x_t * theta_warm
            mu_warm = Z.astype(np.float64) @ weighted_warm
            ucb_terms_warm = np.zeros(K)
            for a in range(K):
                za = Z[a].astype(np.float64)
                phi_a = x_t * za
                ucb_terms_warm[a] = alpha * np.sqrt(max(0, phi_a @ V_inv_warm @ phi_a))
            a_router = int(np.argmax(mu_warm + ucb_terms_warm))

        # --- Observe rewards ---
        # True reward for each action: r(x,a) = <theta*, x⊙z_a> + noise
        def get_reward(action):
            phi = x_t * Z[action].astype(np.float64)
            return float(theta_star.astype(np.float64) @ phi) + rng.randn() * ds.noise_std

        r_router = get_reward(a_router)
        r_llm = get_reward(a_llm)
        r_cold = get_reward(a_cold)

        cum_reward_router += r_router
        cum_reward_llm += r_llm
        cum_reward_cold += r_cold

        # --- Update V, b (passive warm-start: always update with router's action) ---
        phi_router = x_t * Z[a_router].astype(np.float64)
        # Sherman-Morrison update for warm
        Vp = V_inv_warm @ phi_router
        denom = 1.0 + phi_router @ Vp
        V_inv_warm -= np.outer(Vp, Vp) / denom
        b_warm += r_router * phi_router

        # Cold update with cold's own action
        phi_cold_action = x_t * Z[a_cold].astype(np.float64)
        Vp_cold = V_inv_cold @ phi_cold_action
        denom_cold = 1.0 + phi_cold_action @ Vp_cold
        V_inv_cold -= np.outer(Vp_cold, Vp_cold) / denom_cold
        b_cold += r_cold * phi_cold_action

        alpha *= alpha_decay

        if (t + 1) % 1000 == 0 or t == T - 1:
            avg_router = cum_reward_router / (t + 1)
            avg_llm = cum_reward_llm / (t + 1)
            avg_cold = cum_reward_cold / (t + 1)
            history.append({
                'step': t + 1,
                'avg_router': avg_router,
                'avg_llm': avg_llm,
                'avg_cold': avg_cold,
                'llm_pct': llm_count / (t + 1) * 100,
            })
            print(f"  Step {t+1:>5d}/{T} | Router: {avg_router:.4f} | "
                  f"LLM: {avg_llm:.4f} | Cold: {avg_cold:.4f} | "
                  f"LLM%: {llm_count/(t+1)*100:.1f}%")

    return {
        'avg_router': cum_reward_router / T,
        'avg_llm': cum_reward_llm / T,
        'avg_cold': cum_reward_cold / T,
        'llm_pct': llm_count / T * 100,
        'history': history,
    }


# ================================================================
# Main pipeline
# ================================================================
def run_diagnostics(ds, llm_configs=None):
    """Run full diagnostic pipeline on a synthetic dataset."""
    print(f"\n{'='*70}")
    print(f"DATASET: Option {ds.name} — {ds.description}")
    print(f"  d={ds.d}, K={ds.K}, T={ds.T}, noise_std={ds.noise_std}")
    print(f"  theta* norm: {np.linalg.norm(ds.theta_star):.6f}")
    print(f"{'='*70}")

    # --- Phase 1: 3-Axis Diagnostics ---
    print(f"\n--- Axis A: Feature Coherence ---")
    alpha_hat = compute_feature_coherence(ds.Z)
    print(f"  alpha_hat = {alpha_hat:.4f}")
    if alpha_hat >= -0.5:
        print(f"  Status: SATURATED (>= -0.5)")
        a_status = "SATURATED"
    else:
        print(f"  Status: DEVIATING (< -0.5)")
        a_status = "DEVIATING"

    print(f"\n--- Axis B: Effective Rank Ratio ---")
    r_hat, ratio, cov_ctx = compute_effective_rank_ratio(ds.contexts, ds.T)
    print(f"  r_hat = {r_hat:.2f}")
    print(f"  r_hat / sqrt(T) = {ratio:.4f}")
    if ratio < 1.0:
        print(f"  Status: SATURATED (< 1)")
        b_status = "SATURATED"
    else:
        print(f"  Status: DEVIATING (>= 1)")
        b_status = "DEVIATING"

    print(f"\n--- Axis C: Action-Gap Distribution ---")
    gaps, frac_01, tail_label, oracle_labels = compute_gap_distribution(ds)
    print(f"  mean gap: {np.mean(gaps):.6f}")
    print(f"  median gap: {np.median(gaps):.6f}")
    print(f"  frac(gap <= 0.01): {frac_01:.4f}")
    print(f"  Tail: {tail_label}")
    if tail_label == "heavy":
        print(f"  Status: DEVIATING (heavy tail)")
        c_status = "DEVIATING"
    else:
        print(f"  Status: SATURATED (low tail)")
        c_status = "SATURATED"

    # Oracle accuracy (random baseline = 1/K)
    # For continuous rewards, "accuracy" = fraction where argmax matches
    random_acc = 1.0 / ds.K
    print(f"  Random baseline accuracy: {random_acc*100:.2f}%")

    # Regime prediction (2/3 axes deviating = warm-start regime)
    n_deviating = sum(1 for s in [a_status, b_status, c_status] if s == "DEVIATING")
    if n_deviating >= 3:
        regime = "warm-start-essential"
    elif n_deviating >= 2:
        regime = "warm-start-likely (2/3 deviate)"
    elif n_deviating == 1:
        regime = "marginal (1/3 deviate)"
    else:
        regime = "LinUCB-suffices"

    print(f"\n  Phase 1 Summary:")
    print(f"    Axis A: {a_status} (alpha_hat={alpha_hat:.4f})")
    print(f"    Axis B: {b_status} (r/sqrtT={ratio:.4f})")
    print(f"    Axis C: {c_status} (tail={tail_label}, frac≤0.01={frac_01:.4f})")
    print(f"    PREDICTED REGIME: {regime}")

    # --- Phase 2: LLM Subspace Mismatch ---
    if llm_configs is None:
        llm_configs = [
            ('oracle_30pct', 0.30),
            ('oracle_50pct', 0.50),
            ('oracle_70pct', 0.70),
        ]

    print(f"\n--- Phase 2: Subspace Mismatch (simulated LLMs) ---")
    phase2_results = []

    for llm_name, target_acc in llm_configs:
        llm_labels, actual_acc, noise_scale = simulate_llm_predictions(
            ds, accuracy_target=target_acc
        )
        angle, all_angles = compute_subspace_mismatch(
            ds.contexts, ds.Z, llm_labels, r_hat, cov_ctx=cov_ctx
        )
        verdict = "LOW" if angle < 45 else ("MODERATE" if angle < 70 else "HIGH")
        print(f"  {llm_name}: acc={actual_acc*100:.1f}%, angle={angle:.1f}°, verdict={verdict}")
        phase2_results.append({
            'llm_name': llm_name,
            'target_acc': target_acc,
            'actual_acc': actual_acc,
            'angle': angle,
            'verdict': verdict,
            'llm_labels': llm_labels,
        })

    return {
        'name': ds.name,
        'alpha_hat': alpha_hat,
        'r_hat': r_hat,
        'ratio': ratio,
        'tail': tail_label,
        'frac_01': frac_01,
        'regime': regime,
        'a_status': a_status,
        'b_status': b_status,
        'c_status': c_status,
        'phase2': phase2_results,
        'cov_ctx': cov_ctx,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase1', action='store_true',
                        help='Only run Phase 1+2 diagnostics (no Algorithm 1)')
    parser.add_argument('--run', type=str, default=None,
                        choices=['A', 'B', 'C', 'all'],
                        help='Run Algorithm 1 on specified option(s)')
    parser.add_argument('--T', type=int, default=5000,
                        help='Horizon (number of rounds)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("SYNTHETIC DATASET GENERATION & DIAGNOSTICS")
    print("=" * 70)

    # Generate all 3 datasets
    datasets = [
        generate_option_A(T=args.T, seed=args.seed),
        generate_option_B(T=args.T, seed=args.seed),
        generate_option_C(T=args.T, seed=args.seed),
    ]

    # Run diagnostics on all
    all_results = []
    for ds in datasets:
        result = run_diagnostics(ds)
        all_results.append(result)

    # Summary table
    print(f"\n\n{'='*90}")
    print("SUMMARY TABLE — All Synthetic Datasets")
    print("=" * 90)
    print(f"{'Option':<8} {'alpha_hat':>10} {'r/sqrtT':>8} {'tail':>6} {'frac≤.01':>9} "
          f"{'Regime':<25} {'Best LLM angle':>15}")
    print("-" * 90)
    for r in all_results:
        best_angle = min(p['angle'] for p in r['phase2'])
        print(f"{r['name']:<8} {r['alpha_hat']:>10.4f} {r['ratio']:>8.4f} "
              f"{r['tail']:>6} {r['frac_01']:>9.4f} "
              f"{r['regime']:<25} {best_angle:>14.1f}°")
    print("=" * 90)

    # Phase 2 detail
    print(f"\n{'='*90}")
    print("PHASE 2 DETAIL — Subspace Mismatch by LLM Type")
    print("=" * 90)
    print(f"{'Option':<8} {'LLM':<15} {'Accuracy':>9} {'Angle':>8} {'Verdict':<12}")
    print("-" * 90)
    for r in all_results:
        for p in r['phase2']:
            print(f"{r['name']:<8} {p['llm_name']:<15} {p['actual_acc']*100:>8.1f}% "
                  f"{p['angle']:>7.1f}° {p['verdict']:<12}")
    print("=" * 90)

    # Identify candidates for Algorithm 1 (at least 2/3 deviate + low mismatch)
    candidates = []
    for r, ds in zip(all_results, datasets):
        if "warm-start" in r['regime']:
            for p in r['phase2']:
                if p['angle'] < 45:  # Low mismatch — LLM can help
                    candidates.append((ds, r, p))

    print(f"\n\nCANDIDATES for Algorithm 1 (≥2/3 deviate + low mismatch angle < 45°):")
    if not candidates:
        print("  None found. Running Algorithm 1 on all datasets with best LLM anyway.")
        for r, ds in zip(all_results, datasets):
            best_p = min(r['phase2'], key=lambda x: x['angle'])
            candidates.append((ds, r, best_p))

    for ds, r, p in candidates:
        print(f"  Option {ds.name}: regime={r['regime']}, "
              f"LLM={p['llm_name']} (acc={p['actual_acc']*100:.1f}%, angle={p['angle']:.1f}°)")

    # Run Algorithm 1 if requested
    if args.phase1:
        print("\n(--phase1 mode: skipping Algorithm 1)")
        return

    run_targets = candidates
    if args.run is not None and args.run != 'all':
        run_targets = [(ds, r, p) for ds, r, p in candidates if ds.name == args.run]

    print(f"\n\n{'='*70}")
    print("ALGORITHM 1 — Confidence-Gated Routing")
    print("=" * 70)

    algo1_results = []
    for ds, diag, llm_info in run_targets:
        print(f"\n{'='*70}")
        print(f"Running Algorithm 1 on Option {ds.name} with LLM={llm_info['llm_name']}")
        print(f"  Regime prediction: {diag['regime']}")
        print(f"  LLM accuracy: {llm_info['actual_acc']*100:.1f}%")
        print(f"  Subspace angle: {llm_info['angle']:.1f}°")
        print(f"{'='*70}")

        result = run_algorithm1(ds, llm_info['llm_labels'], c=100.0)
        algo1_results.append({
            'option': ds.name,
            'regime': diag['regime'],
            'llm_name': llm_info['llm_name'],
            'llm_acc': llm_info['actual_acc'],
            'angle': llm_info['angle'],
            **{k: v for k, v in result.items() if k != 'history'},
        })

        mult = result['avg_router'] / result['avg_llm'] if result['avg_llm'] != 0 else float('inf')
        print(f"\n  RESULT: Router={result['avg_router']:.4f}, "
              f"LLM={result['avg_llm']:.4f}, Cold={result['avg_cold']:.4f}")
        print(f"  Multiplication (Router/LLM): {mult:.2f}x")
        print(f"  Warm-start lift: {(result['avg_router'] - result['avg_cold']):.4f}")

    # Final comparison
    if algo1_results:
        print(f"\n\n{'='*90}")
        print("FINAL RESULTS — Algorithm 1 Comparison")
        print("=" * 90)
        print(f"{'Option':<8} {'Regime':<22} {'LLM':<15} {'Router':>8} {'LLM':>8} "
              f"{'Cold':>8} {'Mult':>6} {'Lift':>8}")
        print("-" * 90)
        for r in algo1_results:
            mult = r['avg_router'] / r['avg_llm'] if r['avg_llm'] != 0 else float('inf')
            lift = r['avg_router'] - r['avg_cold']
            print(f"{r['option']:<8} {r['regime']:<22} {r['llm_name']:<15} "
                  f"{r['avg_router']:>8.4f} {r['avg_llm']:>8.4f} "
                  f"{r['avg_cold']:>8.4f} {mult:>6.2f}x {lift:>+8.4f}")
        print("=" * 90)

        # Save results
        out_file = "synthetic_results.json"
        with open(out_file, 'w') as f:
            json.dump(algo1_results, f, indent=2, default=str)
        print(f"\nSaved results to {out_file}")


if __name__ == '__main__':
    main()
