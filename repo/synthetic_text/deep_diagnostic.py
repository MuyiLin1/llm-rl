"""
deep_diagnostic.py — In-depth analysis of why multiplication is weak
and what needs to change in the dataset.

Runs Diagnostic #4 (subspace mismatch) + additional analyses:
1. Per-document LLM vs LinUCB complementarity
2. LinUCB convergence speed analysis  
3. Feature space geometry (phi norms, variance)
4. Where LLM is right but LinUCB is wrong (and vice versa)
5. Routing transition dynamics
"""

import gzip
import pickle
import numpy as np
from numpy.linalg import norm
from scipy.linalg import svd, subspace_angles


def load_all():
    print("Loading dataset...")
    with gzip.open('synthetic_text_dataset_v3.pkl.gz', 'rb') as f:
        dataset = pickle.load(f)
    
    print("Loading LLM caches...")
    import os
    caches = {}
    for cf in sorted(os.listdir('.')):
        if cf.startswith('llm_cache_') and cf.endswith('.pkl.gz'):
            model_key = cf.replace('llm_cache_', '').replace('.pkl.gz', '')
            with gzip.open(cf, 'rb') as f:
                caches[model_key] = pickle.load(f)
    return dataset, caches


def diagnostic4_detailed(dataset, llm_labels, label=""):
    """Extended Diagnostic 4: subspace angle + decomposition"""
    doc_features = dataset['doc_features'].astype(np.float64)
    arm_features = dataset['arm_features'].astype(np.float64)
    d = doc_features.shape[1]
    N = doc_features.shape[0]

    # Context covariance
    cov_ctx = (doc_features.T @ doc_features) / N
    
    # LLM-induced covariance (phi = x * z_llm)
    llm_features = doc_features * arm_features[llm_labels.astype(int)]
    cov_llm = (llm_features.T @ llm_features) / N

    # Effective rank
    eigvals_ctx = np.linalg.eigvalsh(cov_ctx)
    eigvals_ctx = eigvals_ctx[eigvals_ctx > 1e-10]
    r_hat = eigvals_ctx.sum() / eigvals_ctx.max()
    r = max(1, min(int(round(r_hat)), d - 1))

    # Subspace angles
    _, _, Vh_llm = svd(cov_llm)
    _, _, Vh_ctx = svd(cov_ctx)
    U_llm = Vh_llm[:r, :].T
    U_ctx = Vh_ctx[:r, :].T

    angles_rad = subspace_angles(U_llm, U_ctx)
    mean_angle = float(np.mean(np.degrees(angles_rad)))
    max_angle = float(np.max(np.degrees(angles_rad)))
    min_angle = float(np.min(np.degrees(angles_rad)))
    
    # Oracle theta* direction
    # theta* = argmax E[reward] for linear model; with our construction theta*=ones
    theta_star = np.ones(d)
    theta_star = theta_star / norm(theta_star)
    
    # How aligned is theta* with the top eigenvectors of context covariance?
    top_ctx = Vh_ctx[:r, :]  # (r, d)
    projection_ctx = norm(top_ctx @ theta_star)
    
    top_llm = Vh_llm[:r, :]  # (r, d)
    projection_llm = norm(top_llm @ theta_star)
    
    print(f"\n  DIAGNOSTIC 4 — {label}")
    print(f"    Effective rank r̂={r_hat:.2f}, using r={r}")
    print(f"    Subspace angle: mean={mean_angle:.1f}°, min={min_angle:.1f}°, max={max_angle:.1f}°")
    print(f"    theta* projection onto ctx top-{r}: {projection_ctx:.4f}")
    print(f"    theta* projection onto llm top-{r}: {projection_llm:.4f}")
    
    return {
        'mean_angle': mean_angle, 'max_angle': max_angle, 'min_angle': min_angle,
        'r': r, 'r_hat': r_hat,
        'theta_proj_ctx': projection_ctx, 'theta_proj_llm': projection_llm,
    }


def complementarity_analysis(dataset, caches):
    """Analyze where LLM and LinUCB are complementary vs redundant"""
    doc_features = dataset['doc_features'].astype(np.float64)
    arm_features = dataset['arm_features'].astype(np.float64)
    true_labels = dataset['ys']
    N, d = doc_features.shape
    K = arm_features.shape[0]
    
    print(f"\n{'='*70}")
    print("COMPLEMENTARITY ANALYSIS")
    print(f"{'='*70}")
    
    # Run LinUCB for full horizon to get its predictions
    np.random.seed(42)
    perm = np.random.permutation(N)
    
    V_inv = np.eye(d, dtype=np.float64) * 10  # 1/lam with lam=0.1
    b = np.zeros(d, dtype=np.float64)
    
    linucb_correct = np.zeros(N, dtype=bool)
    linucb_choices = np.zeros(N, dtype=int)
    
    for t_i in range(N):
        idx = perm[t_i]
        x = doc_features[idx]
        true_y = true_labels[idx]
        alpha = max(0.05, 0.5 * (0.9999 ** t_i))
        
        phi_all = x[None, :] * arm_features
        theta = V_inv @ b
        scores = phi_all @ theta
        V_phi = phi_all @ V_inv
        var = np.sum(V_phi * phi_all, axis=1)
        ucb = scores + alpha * np.sqrt(np.maximum(var, 0))
        choice = int(np.argmax(ucb))
        
        linucb_choices[idx] = choice
        linucb_correct[idx] = (choice == true_y)
        
        reward = 1.0 if choice == true_y else 0.0
        phi_chosen = phi_all[choice]
        Vp = V_inv @ phi_chosen
        denom = 1.0 + phi_chosen @ Vp
        if abs(denom) > 1e-12:
            V_inv -= np.outer(Vp, Vp) / denom
        b += reward * phi_chosen
    
    linucb_acc = linucb_correct.mean()
    print(f"\n  LinUCB overall accuracy (full horizon): {linucb_acc*100:.2f}%")
    
    # Early vs late LinUCB performance
    early_idx = perm[:1000]
    late_idx = perm[3000:]
    print(f"  LinUCB early (first 1000): {linucb_correct[early_idx].mean()*100:.2f}%")
    print(f"  LinUCB late (after 3000):  {linucb_correct[late_idx].mean()*100:.2f}%")
    
    for model_key, cache in caches.items():
        llm_labels = cache['labels'].astype(int)
        llm_correct = (llm_labels == true_labels)
        llm_acc = llm_correct.mean()
        
        # Complementarity metrics
        both_correct = (llm_correct & linucb_correct).sum()
        both_wrong = (~llm_correct & ~linucb_correct).sum()
        llm_only = (llm_correct & ~linucb_correct).sum()
        linucb_only = (~llm_correct & linucb_correct).sum()
        
        # Where LLM is uniquely helpful (early steps)
        llm_helps_early = (llm_correct[early_idx] & ~linucb_correct[early_idx]).sum()
        llm_helps_late = (llm_correct[late_idx] & ~linucb_correct[late_idx]).sum()
        
        oracle_combined = (llm_correct | linucb_correct).mean()
        
        print(f"\n  {model_key} (acc={llm_acc*100:.2f}%):")
        print(f"    Both correct: {both_correct} ({both_correct/N*100:.1f}%)")
        print(f"    Both wrong:   {both_wrong} ({both_wrong/N*100:.1f}%)")
        print(f"    LLM-only:     {llm_only} ({llm_only/N*100:.1f}%)")
        print(f"    LinUCB-only:  {linucb_only} ({linucb_only/N*100:.1f}%)")
        print(f"    Oracle (best of both): {oracle_combined*100:.2f}%")
        print(f"    LLM helps early (first 1000): {llm_helps_early} ({llm_helps_early/1000*100:.1f}%)")
        print(f"    LLM helps late (after 3000):  {llm_helps_late} ({llm_helps_late/len(late_idx)*100:.1f}%)")
        print(f"    ** Max possible multiplication: {oracle_combined / max(llm_acc, linucb_acc):.4f} **")


def feature_geometry(dataset, caches):
    """Analyze phi norm distribution and routing dynamics"""
    doc_features = dataset['doc_features'].astype(np.float64)
    arm_features = dataset['arm_features'].astype(np.float64)
    N, d = doc_features.shape
    K = arm_features.shape[0]
    
    print(f"\n{'='*70}")
    print("FEATURE GEOMETRY")
    print(f"{'='*70}")
    
    # All phis
    phi_all_norms = []
    for i in range(min(1000, N)):
        phi = doc_features[i:i+1, :] * arm_features  # (K, d)
        norms = np.linalg.norm(phi, axis=1)
        phi_all_norms.append(norms)
    phi_all_norms = np.array(phi_all_norms)  # (1000, K)
    
    print(f"\n  Phi norms across all arms:")
    print(f"    Overall:  mean={phi_all_norms.mean():.4f} std={phi_all_norms.std():.4f}")
    print(f"              min={phi_all_norms.min():.4f} max={phi_all_norms.max():.4f}")
    print(f"    Per-doc variance: {phi_all_norms.var(axis=1).mean():.6f}")
    print(f"    Per-arm variance: {phi_all_norms.var(axis=0).mean():.6f}")
    
    # Key insight: if phi norms are all similar, u_llm is essentially constant
    # and routing cannot be document-specific
    print(f"\n  Coefficient of variation (std/mean): {phi_all_norms.std()/phi_all_norms.mean():.4f}")
    print(f"  -> If CV is low, routing is binary (all-or-nothing), not adaptive")
    
    # LinUCB uncertainty decay
    print(f"\n  LinUCB uncertainty decay (u_linucb over time):")
    np.random.seed(42)
    perm = np.random.permutation(N)
    V_inv = np.eye(d, dtype=np.float64) * 10
    b = np.zeros(d, dtype=np.float64)
    true_labels = dataset['ys']
    
    checkpoints = [10, 50, 100, 200, 500, 1000, 2000, 3000, 5000]
    
    for t_i in range(min(5000, N)):
        idx = perm[t_i]
        x = doc_features[idx]
        phi_all = x[None, :] * arm_features
        
        if t_i + 1 in checkpoints:
            # Sample u_linucb for a random arm
            phi_sample = phi_all[0]
            u_linucb = np.sqrt(max(0, phi_sample @ V_inv @ phi_sample))
            # Also average over all arms for this document
            V_phi = phi_all @ V_inv
            var_all = np.sum(V_phi * phi_all, axis=1)
            u_avg = np.sqrt(np.maximum(var_all, 0)).mean()
            print(f"    t={t_i+1:5d}: u_linucb_avg={u_avg:.4f}")
        
        # Update
        theta = V_inv @ b
        scores = phi_all @ theta
        alpha = max(0.05, 0.5 * (0.9999 ** t_i))
        V_phi = phi_all @ V_inv
        var = np.sum(V_phi * phi_all, axis=1)
        ucb = scores + alpha * np.sqrt(np.maximum(var, 0))
        choice = int(np.argmax(ucb))
        
        reward = 1.0 if choice == true_labels[idx] else 0.0
        phi_chosen = phi_all[choice]
        Vp = V_inv @ phi_chosen
        denom = 1.0 + phi_chosen @ Vp
        if abs(denom) > 1e-12:
            V_inv -= np.outer(Vp, Vp) / denom
        b += reward * phi_chosen


def linucb_convergence(dataset):
    """Measure how quickly LinUCB converges — key to multiplication"""
    doc_features = dataset['doc_features'].astype(np.float64)
    arm_features = dataset['arm_features'].astype(np.float64)
    true_labels = dataset['ys']
    N, d = doc_features.shape
    
    print(f"\n{'='*70}")
    print("LINUCB CONVERGENCE SPEED")
    print(f"{'='*70}")
    
    np.random.seed(42)
    perm = np.random.permutation(N)
    V_inv = np.eye(d, dtype=np.float64) * 10
    b = np.zeros(d, dtype=np.float64)
    
    cum_reward = 0
    checkpoints = [100, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000, 5200]
    
    for t_i in range(N):
        idx = perm[t_i]
        x = doc_features[idx]
        alpha = max(0.05, 0.5 * (0.9999 ** t_i))
        
        phi_all = x[None, :] * arm_features
        theta = V_inv @ b
        scores = phi_all @ theta
        V_phi = phi_all @ V_inv
        var = np.sum(V_phi * phi_all, axis=1)
        ucb = scores + alpha * np.sqrt(np.maximum(var, 0))
        choice = int(np.argmax(ucb))
        
        reward = 1.0 if choice == true_labels[idx] else 0.0
        cum_reward += reward
        
        if t_i + 1 in checkpoints:
            print(f"    t={t_i+1:5d}: cumulative accuracy = {cum_reward/(t_i+1)*100:.2f}%")
        
        phi_chosen = phi_all[choice]
        Vp = V_inv @ phi_chosen
        denom = 1.0 + phi_chosen @ Vp
        if abs(denom) > 1e-12:
            V_inv -= np.outer(Vp, Vp) / denom
        b += reward * phi_chosen
    
    # Oracle accuracy
    theta_star = np.ones(d)
    oracle_rewards = 0
    for i in range(N):
        x = doc_features[i]
        phi_all = x[None, :] * arm_features
        scores = phi_all @ theta_star
        if int(np.argmax(scores)) == true_labels[i]:
            oracle_rewards += 1
    oracle_acc = oracle_rewards / N
    print(f"\n    Oracle (theta*=ones): {oracle_acc*100:.2f}%")
    print(f"    Gap (oracle - LinUCB_final): {(oracle_acc - cum_reward/N)*100:.2f}%")
    print(f"    -> Small gap means problem is EASY for LinUCB (bad for multiplication)")


def main():
    dataset, caches = load_all()
    
    print(f"\nDataset: {dataset['doc_features'].shape[0]} docs, "
          f"{dataset['arm_features'].shape[0]} arms, d={dataset['doc_features'].shape[1]}")
    print(f"Models: {list(caches.keys())}")
    
    # 1. Diagnostic 4 for each model
    print(f"\n{'='*70}")
    print("DIAGNOSTIC #4: SUBSPACE MISMATCH")
    print(f"{'='*70}")
    for model_key, cache in caches.items():
        diagnostic4_detailed(dataset, cache['labels'], label=model_key)
    
    # 2. Complementarity
    complementarity_analysis(dataset, caches)
    
    # 3. Feature geometry
    feature_geometry(dataset, caches)
    
    # 4. LinUCB convergence
    linucb_convergence(dataset)
    
    # 5. Summary & recommendations
    print(f"\n{'='*70}")
    print("DIAGNOSIS SUMMARY & RECOMMENDATIONS")
    print(f"{'='*70}")
    print("""
    For strong multiplication, we need:
    1. LinUCB to be SLOW to converge (large gap between early and late performance)
    2. LLM to be RIGHT where LinUCB is WRONG early on (complementarity)
    3. Phi norms to VARY across documents (so routing can be adaptive)
    4. LLM to NOT be right everywhere (otherwise just use LLM alone)
    
    Key metrics to check:
    - Oracle-vs-LinUCB-final gap: should be LARGE (problem is hard)
    - LLM-only-correct count: should be HIGH early, LOW late
    - Phi norm CV: should be HIGH (>0.5) for adaptive routing
    - Max multiplication (oracle best-of-both): should be >> 1.0
    """)


if __name__ == '__main__':
    main()
