"""
c_sweep.py — Sweep c parameter for Algorithm 1 using existing LLM caches.

This re-runs ONLY the routing algorithm (no LLM inference needed).
Takes ~2 seconds per model per c value.

Usage:
  python -u c_sweep.py
"""

import gzip
import json
import os
import pickle

import numpy as np
from numpy.linalg import norm
from tqdm import tqdm


def sm_update(V_inv, phi):
    """Sherman-Morrison rank-1 update: V_inv -= (V_inv phi phi^T V_inv) / (1 + phi^T V_inv phi)"""
    Vp = V_inv @ phi
    denom = 1.0 + phi @ Vp
    if abs(denom) < 1e-12:
        return V_inv
    V_inv -= np.outer(Vp, Vp) / denom
    return V_inv


def run_algorithm1(dataset, llm_labels, T=2000, c=1.0, lam=0.1,
                   alpha=1.0, alpha_decay=0.9999, seed=42):
    doc_features = dataset['doc_features'].astype(np.float64)
    arm_features = dataset['arm_features'].astype(np.float64)
    true_labels = dataset['ys']
    N, d = doc_features.shape
    K = arm_features.shape[0]

    T = min(T, N)
    np.random.seed(seed)
    perm = np.random.permutation(N)[:T]

    V_inv_warm = np.eye(d, dtype=np.float64) / lam
    b_warm = np.zeros(d, dtype=np.float64)
    V_inv_cold = np.eye(d, dtype=np.float64) / lam
    b_cold = np.zeros(d, dtype=np.float64)

    inv_sqrt_c = 1.0 / np.sqrt(c)
    cum_router, cum_llm, cum_cold = 0.0, 0.0, 0.0
    llm_count = 0

    history = {'step': [], 'router': [], 'llm': [], 'cold': [], 'llm_frac': []}

    for t_i in range(T):
        idx = perm[t_i]
        x = doc_features[idx]
        true_y = true_labels[idx]
        llm_choice = int(llm_labels[idx])
        current_alpha = max(0.1, alpha * (alpha_decay ** t_i))

        r_llm = 1.0 if llm_choice == true_y else 0.0
        cum_llm += r_llm

        # Cold LinUCB
        theta_cold = V_inv_cold @ b_cold
        phi_all = x[None, :] * arm_features
        scores_cold = phi_all @ theta_cold
        V_phi = phi_all @ V_inv_cold
        var_cold = np.sum(V_phi * phi_all, axis=1)
        ucb_cold = scores_cold + current_alpha * np.sqrt(np.maximum(var_cold, 0))
        cold_choice = int(np.argmax(ucb_cold))
        r_cold = 1.0 if cold_choice == true_y else 0.0
        cum_cold += r_cold
        phi_cold = phi_all[cold_choice]
        V_inv_cold = sm_update(V_inv_cold, phi_cold)
        b_cold += r_cold * phi_cold

        # Router (Algorithm 1)
        # First compute LinUCB's own best arm
        theta_warm = V_inv_warm @ b_warm
        scores_warm = phi_all @ theta_warm
        V_phi_w = phi_all @ V_inv_warm
        var_warm = np.sum(V_phi_w * phi_all, axis=1)
        ucb_warm = scores_warm + current_alpha * np.sqrt(np.maximum(var_warm, 0))
        linucb_choice = int(np.argmax(ucb_warm))

        # Uncertainty on each method's OWN chosen arm
        phi_llm = phi_all[llm_choice]
        phi_linucb = phi_all[linucb_choice]
        u_llm = norm(phi_llm) * inv_sqrt_c
        u_linucb = np.sqrt(max(0, phi_linucb @ V_inv_warm @ phi_linucb))

        if u_llm < u_linucb:
            router_choice = llm_choice
            llm_count += 1
        else:
            router_choice = linucb_choice

        r_router = 1.0 if router_choice == true_y else 0.0
        cum_router += r_router
        phi_router = phi_all[router_choice]
        V_inv_warm = sm_update(V_inv_warm, phi_router)
        b_warm += r_router * phi_router

        if (t_i + 1) % 200 == 0 or t_i == T - 1:
            history['step'].append(t_i + 1)
            history['router'].append(cum_router / (t_i + 1))
            history['llm'].append(cum_llm / (t_i + 1))
            history['cold'].append(cum_cold / (t_i + 1))
            history['llm_frac'].append(llm_count / (t_i + 1))

    return {
        'avg_router': cum_router / T,
        'avg_llm': cum_llm / T,
        'avg_cold': cum_cold / T,
        'llm_pct': llm_count / T * 100,
        'T': T,
        'history': history,
    }


def main():
    # Config — fine grid around the transition zone + longer T
    c_values = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 1.0, 5.0]
    T = 5200  # use ALL documents
    seed = 42

    # Load dataset
    print("Loading dataset...")
    with gzip.open('synthetic_text_dataset_v3.pkl.gz', 'rb') as f:
        dataset = pickle.load(f)
    print(f"  {dataset['doc_features'].shape[0]} docs, {dataset['arm_features'].shape[0]} arms, d={dataset['doc_features'].shape[1]}")

    # Find all caches
    cache_files = sorted([f for f in os.listdir('.') if f.startswith('llm_cache_') and f.endswith('.pkl.gz')])
    print(f"  Found caches: {cache_files}")

    models = {}
    for cf in cache_files:
        model_key = cf.replace('llm_cache_', '').replace('.pkl.gz', '')
        with gzip.open(cf, 'rb') as f:
            cache = pickle.load(f)
        models[model_key] = cache
        print(f"  {model_key}: accuracy={cache['accuracy']*100:.2f}%")

    # Diagnostic: phi norm statistics
    doc_features = dataset['doc_features'].astype(np.float64)
    arm_features = dataset['arm_features'].astype(np.float64)
    print(f"\n  PHI NORM DIAGNOSTICS:")
    for model_key, cache in models.items():
        labels = cache['labels']
        phi_llm_all = doc_features * arm_features[labels.astype(int)]  # (N, d)
        norms = np.linalg.norm(phi_llm_all, axis=1)
        print(f"  {model_key}: phi_llm norm — min={norms.min():.4f} mean={norms.mean():.4f} "
              f"max={norms.max():.4f} std={norms.std():.4f}")
    print(f"  -> u_llm = norm/sqrt(c). For routing transition at step t, need u_llm ≈ u_linucb")

    # Run sweep
    print(f"\n{'='*80}")
    print(f"C-SWEEP: c = {c_values}")
    print(f"{'='*80}")

    all_results = {}

    for model_key, cache in models.items():
        print(f"\n{'─'*80}")
        print(f"Model: {model_key} (acc={cache['accuracy']*100:.2f}%)")
        print(f"{'─'*80}")
        print(f"  {'c':<8} {'Router':<8} {'LLM':<8} {'Cold':<8} {'Mult':<8} {'LLM%':<8} {'Note'}")
        print(f"  {'-'*60}")

        model_results = {}
        for c in tqdm(c_values, desc=f"  {model_key}", leave=False):
            res = run_algorithm1(dataset, cache['labels'], T=T, c=c, seed=seed)
            mult = res['avg_router'] / max(res['avg_llm'], res['avg_cold'], 1e-8)
            note = "*** MULT > 1 ***" if mult > 1.0 else ""
            print(f"  {c:<8.2f} {res['avg_router']:<8.4f} {res['avg_llm']:<8.4f} "
                  f"{res['avg_cold']:<8.4f} {mult:<8.4f} {res['llm_pct']:<8.1f} {note}")
            model_results[str(c)] = {
                'avg_router': res['avg_router'],
                'avg_llm': res['avg_llm'],
                'avg_cold': res['avg_cold'],
                'llm_pct': res['llm_pct'],
                'multiplication': float(mult),
                'history': res['history'],
            }
        all_results[model_key] = model_results

    # Save results
    out_path = "results_c_sweep.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print best c for each model
    print(f"\n{'='*80}")
    print("BEST c (highest multiplication factor) per model:")
    print(f"{'='*80}")
    for model_key, mres in all_results.items():
        best_c = max(mres.keys(), key=lambda k: mres[k]['multiplication'])
        best = mres[best_c]
        print(f"  {model_key:<16} best c={best_c:<6} mult={best['multiplication']:.4f} "
              f"router={best['avg_router']:.4f} llm%={best['llm_pct']:.1f}%")


if __name__ == '__main__':
    main()
