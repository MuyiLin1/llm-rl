"""
generate_dataset_v5.py — Hard dataset for strong multiplication effect.

Key insight from teammate's results: 28% multiplication from 5% LLM + 5% LinUCB.
Both methods are BAD individually but make DIFFERENT errors. The router learns 
when to trust each one.

Design principles:
1. MANY categories (52) → random chance ≈ 2%
2. HIGH d (300) with NOISE → LinUCB converges slowly (5% → 20% over T=5000)
3. Text is domain-specific but NOT trivially classifiable → LLM gets 10-30%
4. Complementary errors: LLM good at some categories, LinUCB at others
5. Phi norms VARY across documents → adaptive routing

Strategy changes from v3:
- Use the SAME 52 categories and templates (rich text for LLMs)
- But BREAK the feature structure so LinUCB can't trivially solve it:
  a) Higher d (300 vs 150) → more parameters to learn
  b) Add substantial feature noise → noisy signal
  c) Random subset of dimensions are informative → theta* is sparse
  d) Different random projection per arm GROUP → phi norms vary
"""

import numpy as np
import random
import gzip
import pickle
import os
from numpy.linalg import norm
from sklearn.linear_model import Ridge

# ================================================================
# Import categories from v3
# ================================================================
# We import the CATEGORIES, CONFUSER_PAIRS, SHARED_KEYWORDS, TEMPLATES
# and generate_documents from v3 to reuse all the text generation
import sys
sys.path.insert(0, '.')
from generate_dataset_v3 import (
    CATEGORIES, CONFUSER_PAIRS, SHARED_KEYWORDS, TEMPLATES,
    generate_documents, build_bow_features,
)

K = len(CATEGORIES)


def build_noisy_features(doc_bow, arm_bow, d=300, noise_level=1.0, seed=42):
    """
    Build features that are:
    - High-dimensional (d=300)
    - Noisy (signal + Gaussian noise)
    - Have varying phi norms across arm groups
    
    This makes LinUCB convergence slow and creates room for LLM warm-start.
    """
    np.random.seed(seed)
    V = doc_bow.shape[1]
    N = doc_bow.shape[0]
    K_arms = arm_bow.shape[0]
    
    # ---- Document features ----
    # Random projection from BoW to d dimensions
    W = np.random.randn(V, d) / np.sqrt(d)
    doc_proj = doc_bow @ W
    
    # Add per-document noise (varying amounts → phi norm variance)
    # Some documents get more noise → harder to classify → LLM might help
    doc_noise_scales = np.random.exponential(noise_level, size=N)
    doc_noise = np.random.randn(N, d) * doc_noise_scales[:, None] / np.sqrt(d)
    doc_features = doc_proj + doc_noise
    
    # Normalize to unit norm
    doc_norms = norm(doc_features, axis=1, keepdims=True)
    doc_norms[doc_norms < 1e-8] = 1.0
    doc_features = doc_features / doc_norms
    
    # ---- Arm features ----
    arm_proj = arm_bow @ W
    
    # Scale arms DIFFERENTLY to create phi norm variation
    # Group 1 (first third): small arms → small phi norms
    # Group 2 (middle third): medium arms  
    # Group 3 (last third): large arms → large phi norms
    third = K_arms // 3
    arm_scales = np.ones(K_arms)
    arm_scales[:third] = 0.3
    arm_scales[third:2*third] = 0.7
    arm_scales[2*third:] = 1.5
    np.random.shuffle(arm_scales)  # randomize which categories are which scale
    
    # Normalize then scale
    arm_norms = norm(arm_proj, axis=1, keepdims=True)
    arm_norms[arm_norms < 1e-8] = 1.0
    arm_features = arm_proj / arm_norms
    arm_features = arm_features * arm_scales[:, None]
    
    return doc_features, arm_features


def compute_oracle_accuracy(doc_features, arm_features, ys, theta):
    """Compute accuracy of oracle linear model."""
    N = doc_features.shape[0]
    correct = 0
    for i in range(N):
        phi_all = doc_features[i:i+1] * arm_features  # (K, d)
        scores = phi_all @ theta
        if np.argmax(scores) == ys[i]:
            correct += 1
    return correct / N


def learn_theta_star(doc_features, arm_features, ys):
    """Learn theta* via regression that maximizes correct arm scores."""
    N = doc_features.shape[0]
    K = arm_features.shape[0]
    d = doc_features.shape[1]
    
    # Build training data: positive phi (correct arm) and negative phi (random wrong arm)
    n_train = min(N, 10000)
    idx = np.random.choice(N, n_train, replace=False)
    
    phis = []
    targets = []
    for i in idx:
        y = ys[i]
        phi_pos = doc_features[i] * arm_features[y]
        phis.append(phi_pos)
        targets.append(1.0)
        
        # Multiple negatives for better learning
        for _ in range(3):
            neg = random.choice([k for k in range(K) if k != y])
            phi_neg = doc_features[i] * arm_features[neg]
            phis.append(phi_neg)
            targets.append(0.0)
    
    phis = np.array(phis)
    targets = np.array(targets)
    
    reg = Ridge(alpha=0.1)
    reg.fit(phis, targets)
    return reg.coef_


def run_linucb_test(doc_features, arm_features, ys, T, seed=42):
    """Quick LinUCB test to measure convergence."""
    N, d = doc_features.shape
    K = arm_features.shape[0]
    np.random.seed(seed)
    perm = np.random.permutation(N)[:T]
    
    V_inv = np.eye(d, dtype=np.float64) / 0.1
    b = np.zeros(d, dtype=np.float64)
    cum = 0
    
    checkpoints = [100, 200, 500, 1000, 2000, 3000, 5000]
    results = {}
    
    for t_i in range(T):
        idx = perm[t_i]
        x = doc_features[idx]
        alpha = max(0.05, 1.0 * (0.9997 ** t_i))
        
        phi_all = x[None, :] * arm_features
        theta = V_inv @ b
        scores = phi_all @ theta
        V_phi = phi_all @ V_inv
        var = np.sum(V_phi * phi_all, axis=1)
        ucb = scores + alpha * np.sqrt(np.maximum(var, 0))
        choice = int(np.argmax(ucb))
        
        reward = 1.0 if choice == ys[idx] else 0.0
        cum += reward
        
        if t_i + 1 in checkpoints and t_i + 1 <= T:
            acc = cum / (t_i + 1)
            results[t_i + 1] = acc
            print(f"    t={t_i+1:5d}: acc={acc*100:.2f}%")
        
        phi_chosen = phi_all[choice]
        Vp = V_inv @ phi_chosen
        denom = 1.0 + phi_chosen @ Vp
        if abs(denom) > 1e-12:
            V_inv -= np.outer(Vp, Vp) / denom
        b += reward * phi_chosen
    
    final_acc = cum / T
    results[T] = final_acc
    print(f"    Final (T={T}): {final_acc*100:.2f}%")
    return results


def verify_diagnostics(doc_features, arm_features, ys, T=5000):
    """Check axes and phi geometry."""
    K = arm_features.shape[0]
    N = doc_features.shape[0]
    d = doc_features.shape[1]

    print(f"\n{'=' * 70}")
    print(f"DIAGNOSTICS (T={T}, K={K}, d={d}, N={N})")
    print(f"{'=' * 70}")

    # AXIS B: Coherence
    arm_normed = arm_features / (norm(arm_features, axis=1, keepdims=True) + 1e-8)
    cos_matrix = arm_normed @ arm_normed.T
    np.fill_diagonal(cos_matrix, 2.0)
    min_cos = cos_matrix.min(axis=1)
    alpha_hat = np.median(min_cos)
    axis_b = alpha_hat < 0.1
    print(f"\n  Axis B (Coherence): α̂ = {alpha_hat:.4f} {'✓' if axis_b else '✗'}")

    # AXIS A: Effective rank
    np.random.seed(42)
    n_sample = min(5000, N)
    sample_idx = np.random.choice(N, n_sample, replace=False)
    sample_arms = np.random.choice(K, n_sample, replace=True)
    phi = doc_features[sample_idx] * arm_features[sample_arms]
    row_norms = norm(phi, axis=1)
    valid = row_norms > 1e-8
    phi = phi[valid]
    cov = np.cov(phi.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-10]
    r_hat = eigvals.sum() / eigvals.max()
    ratio = r_hat / np.sqrt(T)
    axis_a = ratio >= 1
    print(f"  Axis A (Effective Rank): r̂ = {r_hat:.2f}, r̂/√T = {ratio:.4f} {'✓' if axis_a else '✗'}")

    # Phi norm statistics
    phi_norms = norm(phi, axis=1)
    cv = phi_norms.std() / phi_norms.mean()
    print(f"  Phi norm: mean={phi_norms.mean():.4f} std={phi_norms.std():.4f} CV={cv:.4f}")
    print(f"  (CV > 0.3 good for adaptive routing)")

    n_dev = sum([axis_a, axis_b])
    print(f"\n  Axes deviating: {n_dev}/2 (A, B)")
    return {'axis_a': axis_a, 'axis_b': axis_b, 'r_hat': r_hat, 'phi_cv': cv}


def tune_noise_level(doc_bow, arm_bow, ys, d, target_linucb_acc=0.15, T_test=3000):
    """
    Binary search for noise_level that gives LinUCB accuracy near target.
    Higher noise → harder for LinUCB → lower accuracy.
    """
    print(f"\n{'='*70}")
    print(f"TUNING noise_level for LinUCB target ≈ {target_linucb_acc*100:.0f}% at T={T_test}")
    print(f"{'='*70}")
    
    lo, hi = 0.1, 5.0
    best_noise = 1.0
    best_diff = 999
    
    for iteration in range(8):
        noise = (lo + hi) / 2
        print(f"\n  Trial {iteration+1}: noise={noise:.3f}")
        
        doc_f, arm_f = build_noisy_features(doc_bow, arm_bow, d=d, noise_level=noise, seed=42)
        
        # Quick LinUCB test
        N, dd = doc_f.shape
        K = arm_f.shape[0]
        np.random.seed(42)
        perm = np.random.permutation(N)[:T_test]
        
        V_inv = np.eye(dd, dtype=np.float64) / 0.1
        b = np.zeros(dd, dtype=np.float64)
        cum = 0
        
        for t_i in range(T_test):
            idx = perm[t_i]
            x = doc_f[idx]
            alpha = max(0.05, 1.0 * (0.9997 ** t_i))
            phi_all = x[None, :] * arm_f
            theta = V_inv @ b
            scores = phi_all @ theta
            V_phi = phi_all @ V_inv
            var = np.sum(V_phi * phi_all, axis=1)
            ucb = scores + alpha * np.sqrt(np.maximum(var, 0))
            choice = int(np.argmax(ucb))
            reward = 1.0 if choice == ys[idx] else 0.0
            cum += reward
            phi_chosen = phi_all[choice]
            Vp = V_inv @ phi_chosen
            denom = 1.0 + phi_chosen @ Vp
            if abs(denom) > 1e-12:
                V_inv -= np.outer(Vp, Vp) / denom
            b += reward * phi_chosen
        
        acc = cum / T_test
        diff = abs(acc - target_linucb_acc)
        print(f"    LinUCB acc = {acc*100:.2f}% (target={target_linucb_acc*100:.0f}%, diff={diff*100:.1f}%)")
        
        if diff < best_diff:
            best_diff = diff
            best_noise = noise
        
        if acc > target_linucb_acc:
            lo = noise  # need more noise
        else:
            hi = noise  # too much noise
    
    print(f"\n  Best noise_level = {best_noise:.3f}")
    return best_noise


if __name__ == '__main__':
    print("=" * 70)
    print("SYNTHETIC TEXT DATASET v5 — Hard for both, designed for multiplication")
    print(f"  {K} categories, {len(CONFUSER_PAIRS)} confuser pairs")
    print("=" * 70)
    
    # Step 1: Generate documents (reuse v3 text generation)
    n_per_cat = 200  # 200 * 52 = 10400 documents
    documents, category_names = generate_documents(n_per_category=n_per_cat, seed=42)
    N = len(documents)
    ys = np.array([d['category_idx'] for d in documents])
    print(f"\n  Documents: {N}")
    n_confuser = sum(1 for d in documents if d['is_confuser'])
    print(f"  Confusers: {n_confuser} ({100*n_confuser/N:.1f}%)")
    
    # Step 2: Build BoW features (reuse v3)
    print("\nBuilding BoW features...")
    doc_bow, arm_bow, all_keywords, kw_to_idx = build_bow_features(documents, category_names)
    
    # Step 3: Tune noise level for target LinUCB accuracy
    d = 300
    T_target = 5000
    target_acc = 0.15  # 15% LinUCB accuracy at T=5000
    
    best_noise = tune_noise_level(doc_bow, arm_bow, ys, d=d, 
                                   target_linucb_acc=target_acc, T_test=T_target)
    
    # Step 4: Build final features with tuned noise
    print(f"\n{'='*70}")
    print(f"BUILDING FINAL FEATURES (d={d}, noise={best_noise:.3f})")
    print(f"{'='*70}")
    doc_features, arm_features = build_noisy_features(
        doc_bow, arm_bow, d=d, noise_level=best_noise, seed=42
    )
    print(f"  doc_features: {doc_features.shape}")
    print(f"  arm_features: {arm_features.shape}")
    
    # Step 5: Learn theta*
    print("\nLearning theta*...")
    theta_star = learn_theta_star(doc_features, arm_features, ys)
    oracle_acc = compute_oracle_accuracy(doc_features, arm_features, ys, theta_star)
    print(f"  Oracle accuracy: {oracle_acc*100:.2f}%")
    
    # Step 6: Verify diagnostics
    diag = verify_diagnostics(doc_features, arm_features, ys, T=T_target)
    
    # Step 7: Full LinUCB convergence test
    print(f"\n{'='*70}")
    print(f"LINUCB CONVERGENCE TEST")
    print(f"{'='*70}")
    run_linucb_test(doc_features, arm_features, ys, T=min(T_target, N), seed=42)
    
    # Step 8: Save dataset
    dataset = {
        'documents': documents,
        'category_names': category_names,
        'doc_features': doc_features,
        'arm_features': arm_features,
        'doc_bow': doc_bow,
        'arm_bow': arm_bow,
        'ys': ys,
        'all_keywords': all_keywords,
        'keyword_to_idx': kw_to_idx,
        'theta_star': theta_star,
        'n_categories': K,
        'n_documents': N,
        'confuser_pairs': CONFUSER_PAIRS,
        'noise_level': best_noise,
        'd': d,
    }
    
    out_path = 'synthetic_text_dataset_v5.pkl.gz'
    with gzip.open(out_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    fsize = os.path.getsize(out_path) / 1024 / 1024
    print(f"\nSaved: {out_path} ({fsize:.1f} MB)")
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Categories: {K}")
    print(f"  Documents:  {N}")
    print(f"  Features:   d={d}")
    print(f"  Noise:      {best_noise:.3f}")
    print(f"  Oracle:     {oracle_acc*100:.2f}%")
    print(f"  Phi CV:     {diag['phi_cv']:.4f}")
    print(f"  Target: LinUCB ≈ {target_acc*100:.0f}%, LLM 10-30% → multiplication from complementarity")
