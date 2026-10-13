"""
experiment_synthetic.py — Clean Synthetic for Warm-Start Multiplication

Design mimics what makes AmazonCat work:
  - Large K relative to T → cold LinUCB is starved (T/K ≈ 10)
  - Low coherence α̂ → each arm must be learned independently
  - "LLM" frozen policy concentrates on reward-relevant arms
  - Element-wise features φ(x,a) = x ⊙ a

Parameters:
  d = 200       feature dimension
  K = 1000      number of arms (large!)
  T = 10000     horizon → T/K = 10 pulls/arm (starved)

Ground truth:
  θ* ∈ R^d     true reward parameter
  Arms:         a_k ~ Uniform(S^{d-1})  → pairwise α̂ ≈ 1/√d ≈ 0.07
  Contexts:     x ~ N(0, I_d)
  Features:     φ(x,a) = x ⊙ a
  Reward:       r = 1{argmax_k φ(x,a_k)·θ* == chosen arm}  (Precision@1)

The "LLM" (simulated frozen policy):
  - Has access to a noisy/partial version of θ*
  - Achieves ~10-20% accuracy (vs 0.1% random on K=1000)
  - Concentrates selections on a reward-informative subset of arms
  - This provides directional coverage parameter ν > 0

Why multiplication occurs:
  - Cold LinUCB uses UCB exploration → tries all 1000 arms → each gets ~10 pulls
    → insufficient to learn θ* along each arm's direction
  - Router defers to LLM early → LLM picks from ~100 "known" arms
    → those directions get dense coverage → θ* estimated well there
  - After warm-start phase, LinUCB exploits learned arms → cumulative reward jumps
  - Phase transition mirrors AmazonCat

Usage:
  python experiment_synthetic.py --T 10000 --c 1.0 --seeds 3
  python experiment_synthetic.py --T 10000 --c_sweep --seeds 3
  python experiment_synthetic.py --T 50000 --c 1.0 --seeds 3  (longer run on SSH)
"""

import numpy as np
import json
import argparse
import os
from numpy.linalg import norm


# ================================================================
# Dataset Generation
# ================================================================
def generate_dataset(d=200, K=1000, N=20000, llm_noise=0.3, seed=0):
    """
    Generate a synthetic contextual bandit dataset.
    
    Returns dict with:
      - contexts: (N, d) context vectors
      - arm_features: (K, d) arm feature vectors (unit norm, ~orthogonal)
      - true_labels: (N,) correct arm for each context
      - theta_star: (d,) true reward parameter
      - llm_labels: (N,) simulated LLM predictions
      - llm_accuracy: scalar
      
    The "LLM" model:
      θ_LLM = (1 - llm_noise) * θ* + llm_noise * random_direction
      Picks argmax_k φ(x, a_k)·θ_LLM
      
      With K=1000 and moderate noise, LLM achieves ~5-15% accuracy
      (vs 0.1% random). Crucially, LLM concentrates its picks on
      reward-relevant arms → provides directional warm-start.
    """
    rng = np.random.default_rng(seed)
    
    # --- θ* : true reward parameter (unit norm, dense) ---
    theta_star = rng.standard_normal(d)
    theta_star = theta_star / norm(theta_star)
    
    # --- Arm features: random unit vectors on S^{d-1} ---
    # With d=200, K=1000, pairwise |<a_i, a_j>| ≈ 1/√d ≈ 0.07
    arm_features = rng.standard_normal((K, d))
    arm_features = arm_features / norm(arm_features, axis=1, keepdims=True)
    
    # --- Contexts: x ~ N(0, I_d) ---
    contexts = rng.standard_normal((N, d))
    
    # --- True labels: argmax_k φ(x, a_k)·θ* ---
    # φ(x, a_k)·θ* = (x ⊙ a_k)·θ* = Σ_j x_j * a_{k,j} * θ*_j = x · (a_k ⊙ θ*)
    scoring_matrix = arm_features * theta_star[None, :]  # (K, d) — each row = a_k ⊙ θ*
    scores = contexts @ scoring_matrix.T  # (N, K) — score[i,k] = x_i · (a_k ⊙ θ*)
    true_labels = np.argmax(scores, axis=1)  # (N,)
    
    # --- Simulated "LLM" frozen policy ---
    # θ_LLM is a mixture: mostly θ* with some random corruption
    # This ensures the LLM "knows" the reward direction but imperfectly
    noise_dir = rng.standard_normal(d)
    noise_dir = noise_dir / norm(noise_dir)
    theta_llm = (1 - llm_noise) * theta_star + llm_noise * noise_dir
    theta_llm = theta_llm / norm(theta_llm)  # re-normalize to unit
    
    scoring_llm = arm_features * theta_llm[None, :]  # (K, d)
    scores_llm = contexts @ scoring_llm.T  # (N, K)
    llm_labels = np.argmax(scores_llm, axis=1)  # (N,)
    
    llm_accuracy = np.mean(llm_labels == true_labels)
    
    # --- Compute diagnostics ---
    # Coherence: average pairwise |<a_i, a_j>|
    # Sample 5000 pairs for efficiency
    n_pairs = 5000
    idx1 = rng.integers(0, K, n_pairs)
    idx2 = rng.integers(0, K, n_pairs)
    mask = idx1 != idx2
    coherences = np.abs(np.sum(arm_features[idx1[mask]] * arm_features[idx2[mask]], axis=1))
    alpha_hat = np.mean(coherences)
    
    # Effective rank of joint feature covariance
    # Sample some φ(x, a) vectors
    sample_size = min(5000, N)
    sample_idx = rng.integers(0, N, sample_size)
    sample_arms = rng.integers(0, K, sample_size)
    phi_samples = contexts[sample_idx] * arm_features[sample_arms]  # (sample_size, d)
    cov = phi_samples.T @ phi_samples / sample_size
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-10]
    r_hat = (np.sum(eigvals) ** 2) / np.sum(eigvals ** 2)
    
    print(f"Dataset generated:")
    print(f"  d={d}, K={K}, N={N}")
    print(f"  LLM accuracy: {llm_accuracy:.3f} (random: {1/K:.4f})")
    print(f"  Coherence α̂: {alpha_hat:.4f}")
    print(f"  Effective rank r̂: {r_hat:.1f}")
    print(f"  r̂/√T (at T={N}): {r_hat/np.sqrt(N):.3f}")
    print(f"  θ_LLM · θ* (cosine): {np.dot(theta_llm, theta_star) / (norm(theta_llm)*norm(theta_star)):.3f}")
    
    return {
        'contexts': contexts.astype(np.float32),
        'arm_features': arm_features.astype(np.float32),
        'true_labels': true_labels,
        'llm_labels': llm_labels,
        'theta_star': theta_star,
        'theta_llm': theta_llm,
        'llm_accuracy': llm_accuracy,
        'diagnostics': {
            'alpha_hat': float(alpha_hat),
            'r_hat': float(r_hat),
            'd': d, 'K': K, 'N': N,
        }
    }


# ================================================================
# Sherman-Morrison rank-1 inverse update
# ================================================================
def sm_update(V_inv, x):
    Vx = V_inv @ x
    denom = 1.0 + x @ Vx
    if abs(denom) < 1e-12:
        return V_inv
    return V_inv - np.outer(Vx, Vx) / denom


# ================================================================
# Algorithm 1: Confidence-Gated Router
# ================================================================
def run_experiment(dataset, T=10000, c=1.0, lam=0.1, alpha=1.0,
                   alpha_decay=0.9999, seed=42):
    """
    Run Algorithm 1 on the synthetic dataset.
    
    Returns dict with cumulative accuracies and routing stats.
    """
    rng = np.random.default_rng(seed)
    
    contexts = dataset['contexts'].astype(np.float64)
    Z = dataset['arm_features'].astype(np.float64)  # (K, d)
    true_labels = dataset['true_labels']
    llm_labels = dataset['llm_labels']
    
    N, d = contexts.shape
    K = Z.shape[0]
    T = min(T, N)
    
    # Random permutation of contexts
    perm = rng.permutation(N)[:T]
    
    # --- Router (warm-started LinUCB) ---
    V_inv_warm = (1.0 / lam) * np.eye(d, dtype=np.float64)
    b_warm = np.zeros(d, dtype=np.float64)
    
    # --- Cold LinUCB (independent) ---
    V_inv_cold = (1.0 / lam) * np.eye(d, dtype=np.float64)
    b_cold = np.zeros(d, dtype=np.float64)
    
    inv_sqrt_c = 1.0 / np.sqrt(c)
    
    # Accumulators
    cum_router = 0
    cum_llm = 0
    cum_cold = 0
    llm_rounds = 0
    
    # History for learning curves
    history = {'step': [], 'router': [], 'llm': [], 'cold': [], 'llm_frac': []}
    log_interval = max(1, T // 100)  # ~100 log points
    
    for t in range(T):
        idx = perm[t]
        cx = contexts[idx].astype(np.float64)
        y_true = true_labels[idx]
        y_llm = llm_labels[idx]
        
        current_alpha = max(0.1, alpha * (alpha_decay ** t))
        
        # --- LLM baseline ---
        r_llm = 1.0 if y_llm == y_true else 0.0
        cum_llm += r_llm
        
        # --- Cold LinUCB ---
        theta_cold = V_inv_cold @ b_cold
        weighted_cold = cx * theta_cold  # (d,)
        mu_cold = Z @ weighted_cold  # (K,)
        
        # UCB bonus: φ(x,a_k)^T V^{-1} φ(x,a_k) = (x⊙a_k)^T V^{-1} (x⊙a_k)
        # = a_k^T diag(x) V^{-1} diag(x) a_k
        Cx_cold = (V_inv_cold * cx[:, None]) * cx[None, :]  # (d, d)
        ZCx_cold = Z @ Cx_cold  # (K, d)
        var_cold = np.sum(ZCx_cold * Z, axis=1)  # (K,)
        ucb_cold = mu_cold + current_alpha * np.sqrt(np.maximum(var_cold, 0))
        cold_choice = int(np.argmax(ucb_cold))
        
        r_cold = 1.0 if cold_choice == y_true else 0.0
        cum_cold += r_cold
        
        # Update cold LinUCB
        phi_cold = cx * Z[cold_choice]
        V_inv_cold = sm_update(V_inv_cold, phi_cold)
        b_cold += r_cold * phi_cold
        
        # --- Router (Algorithm 1) ---
        theta_warm = V_inv_warm @ b_warm
        weighted_warm = cx * theta_warm
        mu_warm = Z @ weighted_warm
        
        Cx_warm = (V_inv_warm * cx[:, None]) * cx[None, :]
        ZCx_warm = Z @ Cx_warm
        var_warm = np.sum(ZCx_warm * Z, axis=1)
        ucb_warm = mu_warm + current_alpha * np.sqrt(np.maximum(var_warm, 0))
        linucb_choice = int(np.argmax(ucb_warm))
        
        # Feature vectors
        phi_llm = cx * Z[y_llm]
        phi_linucb = cx * Z[linucb_choice]
        
        # Uncertainties
        u_llm = inv_sqrt_c * norm(phi_llm)
        u_linucb = np.sqrt(max(0.0, phi_linucb @ V_inv_warm @ phi_linucb))
        
        # Route
        if u_llm < u_linucb:
            router_choice = y_llm
            r_router = 1.0 if router_choice == y_true else 0.0
            llm_rounds += 1
            V_inv_warm = sm_update(V_inv_warm, phi_llm)
            b_warm += r_router * phi_llm  # passive full update
        else:
            router_choice = linucb_choice
            r_router = 1.0 if router_choice == y_true else 0.0
            V_inv_warm = sm_update(V_inv_warm, phi_linucb)
            b_warm += r_router * phi_linucb
        
        cum_router += r_router
        
        # Log
        if (t + 1) % log_interval == 0 or t == T - 1:
            history['step'].append(t + 1)
            history['router'].append(cum_router / (t + 1))
            history['llm'].append(cum_llm / (t + 1))
            history['cold'].append(cum_cold / (t + 1))
            history['llm_frac'].append(llm_rounds / (t + 1))
    
    # Final results
    avg_router = cum_router / T
    avg_llm = cum_llm / T
    avg_cold = cum_cold / T
    llm_pct = llm_rounds / T
    mult = avg_router / max(avg_llm, avg_cold, 1e-8)
    
    return {
        'avg_router': avg_router,
        'avg_llm': avg_llm,
        'avg_cold': avg_cold,
        'llm_pct': llm_pct,
        'multiplication': mult,
        'T': T,
        'c': c,
        'seed': seed,
        'history': history,
    }


# ================================================================
# C-sweep
# ================================================================
def run_c_sweep(dataset, T=10000, seeds=None, c_values=None):
    """Run Algorithm 1 across multiple c values and seeds."""
    if seeds is None:
        seeds = [42, 43, 44]
    if c_values is None:
        c_values = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    
    results = {}
    best_mult = 0
    best_c = None
    
    for c in c_values:
        seed_results = []
        for s in seeds:
            r = run_experiment(dataset, T=T, c=c, seed=s)
            seed_results.append(r)
            print(f"  c={c:8.3f} seed={s}: router={r['avg_router']:.4f} "
                  f"cold={r['avg_cold']:.4f} llm={r['avg_llm']:.4f} "
                  f"mult={r['multiplication']:.3f} llm%={r['llm_pct']:.1%}")
        
        avg_router = np.mean([r['avg_router'] for r in seed_results])
        avg_cold = np.mean([r['avg_cold'] for r in seed_results])
        avg_llm = np.mean([r['avg_llm'] for r in seed_results])
        llm_pct = np.mean([r['llm_pct'] for r in seed_results])
        mult = avg_router / max(avg_llm, avg_cold, 1e-8)
        
        results[str(c)] = {
            'avg_router': avg_router,
            'avg_cold': avg_cold,
            'avg_llm': avg_llm,
            'llm_pct': llm_pct,
            'multiplication': mult,
            'std_router': np.std([r['avg_router'] for r in seed_results]),
            'std_cold': np.std([r['avg_cold'] for r in seed_results]),
        }
        
        if mult > best_mult:
            best_mult = mult
            best_c = c
    
    return results, best_c, best_mult


# ================================================================
# Main
# ================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=200, help='Feature dimension')
    parser.add_argument('--K', type=int, default=1000, help='Number of arms')
    parser.add_argument('--N', type=int, default=20000, help='Total contexts generated')
    parser.add_argument('--T', type=int, default=10000, help='Horizon (rounds to play)')
    parser.add_argument('--c', type=float, default=1.0, help='Confidence parameter')
    parser.add_argument('--c_sweep', action='store_true', help='Run c-sweep')
    parser.add_argument('--seeds', type=int, default=3, help='Number of seeds')
    parser.add_argument('--noise', type=float, default=0.3, help='LLM noise level (0=perfect, 1=random direction)')
    parser.add_argument('--data_seed', type=int, default=0, help='Dataset generation seed')
    parser.add_argument('--out', type=str, default=None, help='Output JSON path')
    args = parser.parse_args()
    
    print("=" * 70)
    print("SYNTHETIC WARM-START EXPERIMENT")
    print("=" * 70)
    
    # Generate dataset
    dataset = generate_dataset(d=args.d, K=args.K, N=args.N,
                               llm_noise=args.noise, seed=args.data_seed)
    
    print(f"\nExperiment config: T={args.T}, c={args.c}")
    print(f"Expected: T/K = {args.T/args.K:.1f} pulls/arm (starved regime)")
    print("=" * 70)
    
    seeds_list = [42 + i for i in range(args.seeds)]
    
    if args.c_sweep:
        print("\n--- C-SWEEP ---")
        results, best_c, best_mult = run_c_sweep(
            dataset, T=args.T, seeds=seeds_list
        )
        print(f"\nBest: c={best_c}, mult={best_mult:.3f}")
        
        out_path = args.out or f"results_synthetic_d{args.d}_K{args.K}_T{args.T}.json"
        output = {
            'config': {
                'd': args.d, 'K': args.K, 'N': args.N, 'T': args.T,
                'noise': args.noise, 'data_seed': args.data_seed,
            },
            'diagnostics': dataset['diagnostics'],
            'llm_accuracy': float(dataset['llm_accuracy']),
            'c_sweep': results,
            'best_c': best_c,
            'best_mult': best_mult,
        }
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Saved to {out_path}")
    else:
        print(f"\n--- Single run: c={args.c} ---")
        for s in seeds_list:
            r = run_experiment(dataset, T=args.T, c=args.c, seed=s)
            print(f"  seed={s}: router={r['avg_router']:.4f} "
                  f"cold={r['avg_cold']:.4f} llm={r['avg_llm']:.4f} "
                  f"mult={r['multiplication']:.3f} llm%={r['llm_pct']:.1%}")
