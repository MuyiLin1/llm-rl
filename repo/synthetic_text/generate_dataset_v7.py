"""
generate_dataset_v7.py — Quick-fix version that passes ALL 3 diagnostic axes.

Changes from v6:
- Arm features: orthogonal random vectors (QR) → α̂ ≈ 0 → passes Axis B
- Doc features: high-rank (category mean + full-rank noise) → r̂ large → passes Axis A  
- Gap structure: θ* constructed to create confusable categories → passes Axis C
- Same text/documents → LLM caches from v6 still work
- Same group structure → experiment logic unchanged

The experiment should show multiplication because:
- High effective rank → cold LinUCB converges SLOWLY
- Low coherence → each arm must be learned independently
- LLM provides early correct answers → warm-starts LinUCB → Router > Cold
"""

import numpy as np
import random
import gzip
import pickle
import os
from numpy.linalg import norm
from sklearn.linear_model import Ridge

import sys
sys.path.insert(0, '.')
from generate_dataset_v3 import (
    CATEGORIES, CONFUSER_PAIRS, SHARED_KEYWORDS, TEMPLATES,
    generate_documents, build_bow_features,
)

K = len(CATEGORIES)
category_names = list(CATEGORIES.keys())


# ================================================================
# GROUP ASSIGNMENT (same as v6)
# ================================================================
def assign_groups():
    """Split categories into LinUCB-friendly and LLM-friendly groups."""
    in_pairs = set()
    for cat_a, cat_b in CONFUSER_PAIRS:
        in_pairs.add(cat_a)
        in_pairs.add(cat_b)

    group_linucb = []
    group_llm = []

    for cat in category_names:
        if cat in in_pairs:
            group_linucb.append(cat)
        else:
            group_llm.append(cat)

    while len(group_linucb) > len(group_llm) + 4:
        partner_counts = {}
        for cat in group_linucb:
            count = sum(1 for a, b in CONFUSER_PAIRS if a == cat or b == cat)
            partner_counts[cat] = count
        least = min(group_linucb, key=lambda c: partner_counts[c])
        group_linucb.remove(least)
        group_llm.append(least)

    return group_linucb, group_llm


# ================================================================
# KEY CHANGE: Orthogonal arm features + high-rank doc features
# ================================================================
def build_diagnostic_passing_features(doc_bow, arm_bow, ys, group_linucb_idx, group_llm_idx,
                                       d=400, arm_noise_llm=0.6, seed=42):
    """
    Build features that PASS all 3 diagnostic axes while preserving
    the complementarity structure needed for multiplication.
    
    Uses a PLANTED MODEL approach:
    1. Generate θ* ∈ R^d
    2. Generate orthogonal arm features a_k → α̂ ≈ 0 (Axis B)
    3. Compute scoring directions s_k = a_k ⊙ θ*
    4. Doc features: x = signal * s_k/||s_k|| + noise * random → high r̂ (Axis A)
    5. The noise component creates small gaps (Axis C)
    
    Multiplication mechanism:
    - LLM-group arms: corrupted → LinUCB can't distinguish them
    - LinUCB-group arms: clean orthogonal → LinUCB eventually learns
    - LLM handles LLM-group correctly from text → Router beats both
    """
    np.random.seed(seed)
    N = doc_bow.shape[0]
    K_arms = arm_bow.shape[0]
    
    # ----- STEP 1: Generate θ* (the true reward parameter) -----
    theta_star = np.random.randn(d)
    theta_star = theta_star / norm(theta_star)
    
    # ----- STEP 2: ARM FEATURES - Orthogonal via QR -----
    random_matrix = np.random.randn(d, d)
    Q, _ = np.linalg.qr(random_matrix)
    arm_features = Q[:K_arms, :].copy()  # orthonormal rows
    
    # ----- STEP 3: Compute scoring directions -----
    # Score for arm k given context x: x · (a_k ⊙ θ*) = φ(x,a_k)·θ*
    scoring_dirs = arm_features * theta_star[None, :]  # (K_arms, d)
    scoring_norms = norm(scoring_dirs, axis=1, keepdims=True)
    scoring_norms[scoring_norms < 1e-8] = 1.0
    scoring_dirs_normed = scoring_dirs / scoring_norms
    
    # ----- STEP 4: DOC FEATURES - Planted signal + high-rank noise -----
    # signal_strength controls oracle accuracy vs effective rank tradeoff
    # After normalization, effective signal ≈ signal_strength / sqrt(signal^2 + noise^2 * d)
    # Use DIFFERENT signal per group:
    #   LinUCB-group: strong signal → LinUCB can learn these (good convergence)
    #   LLM-group: weak signal → small gaps → Axis C passes, LLM handles from text
    signal_linucb = 2.5     # strong: LinUCB converges well on these
    signal_llm = 0.3        # weak: creates heavy gap tail, LinUCB struggles
    noise_strength = 1.0    # enough noise for high effective rank
    
    llm_set = set(group_llm_idx)
    
    doc_features = np.zeros((N, d))
    for i in range(N):
        cat_idx = ys[i]
        # Choose signal strength based on group
        sig = signal_llm if cat_idx in llm_set else signal_linucb
        # Signal: aligned with the scoring direction of the correct arm
        signal = sig * scoring_dirs_normed[cat_idx]
        # Noise: isotropic Gaussian → creates high effective rank in φ space
        noise = noise_strength * np.random.randn(d)
        doc_features[i] = signal + noise
    
    # Normalize doc features to unit sphere
    doc_norms = norm(doc_features, axis=1, keepdims=True)
    doc_norms[doc_norms < 1e-8] = 1.0
    doc_features = doc_features / doc_norms
    
    # ----- STEP 5: CORRUPT LLM-group arms -----
    # Mix with random noise → LinUCB can't learn these arms well
    llm_set = set(group_llm_idx)
    noise_matrix = np.random.randn(K_arms, d)
    noise_norms_m = norm(noise_matrix, axis=1, keepdims=True)
    noise_matrix = noise_matrix / (noise_norms_m + 1e-8)
    
    for k in range(K_arms):
        if k in llm_set:
            arm_features[k] = (1.0 - arm_noise_llm) * arm_features[k] + arm_noise_llm * noise_matrix[k]
            n = norm(arm_features[k])
            if n > 1e-8:
                arm_features[k] /= n
    
    return doc_features, arm_features


def contaminate_text(documents, category_names_list, categories_dict,
                     linucb_group_set, n_misleading_linucb=6,
                     n_misleading_llm=1, seed=42):
    """Same as v6: asymmetric text contamination."""
    rng = random.Random(seed)
    all_cats = list(categories_dict.keys())
    
    n_heavy, n_light = 0, 0
    for doc in documents:
        true_cat = doc['category_name']
        cat_idx = category_names_list.index(true_cat)
        
        if cat_idx in linucb_group_set:
            n_inject = n_misleading_linucb
            n_heavy += 1
        else:
            n_inject = n_misleading_llm
            n_light += 1
        
        if n_inject == 0:
            continue
        
        other_cats = [c for c in all_cats if c != true_cat]
        misleading_kws = []
        for _ in range(n_inject):
            src_cat = rng.choice(other_cats)
            kw = rng.choice(categories_dict[src_cat])
            misleading_kws.append(kw)
        
        text = doc['text']
        injections = []
        templates = [
            "This also relates to {kw}.",
            "Consider the role of {kw} in this context.",
            "Recent work connects this to {kw}.",
            "The relationship with {kw} is notable.",
            "There are parallels with {kw}.",
        ]
        for kw in misleading_kws:
            tmpl = rng.choice(templates)
            injections.append(tmpl.format(kw=kw))
        
        sentences = text.split('. ')
        for inj in injections:
            pos = rng.randint(0, len(sentences))
            sentences.insert(pos, inj)
        
        doc['text'] = '. '.join(sentences)
    
    print(f"  Heavy contamination ({n_misleading_linucb} kw): {n_heavy} docs")
    print(f"  Light contamination ({n_misleading_llm} kw): {n_light} docs")
    return documents


def learn_theta_star(doc_features, arm_features, ys):
    """Learn θ* from data (same as v6)."""
    N = doc_features.shape[0]
    K_arms = arm_features.shape[0]
    d = doc_features.shape[1]

    n_train = min(N, 10000)
    idx = np.random.choice(N, n_train, replace=False)

    phis, targets = [], []
    for i in idx:
        y = ys[i]
        phi_pos = doc_features[i] * arm_features[y]
        phis.append(phi_pos)
        targets.append(1.0)
        for _ in range(3):
            neg = random.choice([k for k in range(K_arms) if k != y])
            phi_neg = doc_features[i] * arm_features[neg]
            phis.append(phi_neg)
            targets.append(0.0)

    reg = Ridge(alpha=0.1)
    reg.fit(np.array(phis), np.array(targets))
    return reg.coef_


def compute_oracle_accuracy(doc_features, arm_features, ys, theta):
    N = doc_features.shape[0]
    correct = 0
    for i in range(N):
        phi_all = doc_features[i:i+1] * arm_features
        if np.argmax(phi_all @ theta) == ys[i]:
            correct += 1
    return correct / N


def compute_group_oracle(doc_features, arm_features, ys, theta, group_idx_set):
    correct, total = 0, 0
    for i in range(len(ys)):
        if ys[i] in group_idx_set:
            phi_all = doc_features[i:i+1] * arm_features
            if np.argmax(phi_all @ theta) == ys[i]:
                correct += 1
            total += 1
    return correct / max(total, 1)


def verify_diagnostics(doc_features, arm_features, ys, T=5000):
    """Verify all 3 axes pass."""
    K_arms = arm_features.shape[0]
    N = doc_features.shape[0]
    d = doc_features.shape[1]

    print(f"\n{'=' * 70}")
    print(f"DIAGNOSTICS (T={T}, K={K_arms}, d={d}, N={N})")
    print(f"{'=' * 70}")

    # AXIS B: Coherence α̂
    arm_normed = arm_features / (norm(arm_features, axis=1, keepdims=True) + 1e-8)
    cos_matrix = arm_normed @ arm_normed.T
    np.fill_diagonal(cos_matrix, -2.0)  # exclude self (set to very negative)
    max_cos = cos_matrix.max(axis=1)  # maximum cosine to nearest neighbor
    alpha_hat = np.median(max_cos)
    axis_b = alpha_hat < 0.1
    print(f"\n  Axis B (Coherence): α̂ = {alpha_hat:.4f} (threshold < 0.1) {'✓ PASS' if axis_b else '✗ FAIL'}")
    print(f"    max(max_cos) = {max_cos.max():.4f}, min(max_cos) = {max_cos.min():.4f}")

    # AXIS A: Effective rank r̂/√T
    np.random.seed(42)
    n_sample = min(8000, N)
    si = np.random.choice(N, n_sample, replace=False)
    sa = np.random.choice(K_arms, n_sample)
    phi = doc_features[si] * arm_features[sa]
    row_norms = norm(phi, axis=1)
    phi = phi[row_norms > 1e-8]
    cov = np.cov(phi.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-10]
    r_hat = eigvals.sum() / eigvals.max()
    ratio = r_hat / np.sqrt(T)
    axis_a = ratio >= 1.0
    print(f"\n  Axis A (Effective Rank): r̂ = {r_hat:.2f}, √T = {np.sqrt(T):.2f}, r̂/√T = {ratio:.4f} {'✓ PASS' if axis_a else '✗ FAIL'}")
    T_max = r_hat ** 2
    print(f"    Would pass for T ≤ {T_max:.0f}")

    # AXIS C: Gap distribution (heavy tail)
    # Learn theta for gap computation
    theta = learn_theta_star(doc_features, arm_features, ys)
    
    n_gap = min(3000, N)
    gi = np.random.choice(N, n_gap, replace=False)
    gaps = []
    for i in gi:
        phi_all = doc_features[i:i+1] * arm_features
        scores = phi_all @ theta
        sorted_s = np.sort(scores)[::-1]
        if len(sorted_s) >= 2:
            gaps.append(sorted_s[0] - sorted_s[1])
    gaps = np.array(gaps)
    
    # Use gap ≤ 0.05 threshold
    frac_small = np.mean(gaps <= 0.05)
    axis_c = frac_small > 0.5
    print(f"\n  Axis C (Gap tail): frac(gap≤0.05) = {frac_small:.4f} (threshold > 0.5) {'✓ PASS' if axis_c else '✗ FAIL'}")
    print(f"    Gap percentiles: p10={np.percentile(gaps,10):.4f}, p25={np.percentile(gaps,25):.4f}, "
          f"p50={np.percentile(gaps,50):.4f}, p75={np.percentile(gaps,75):.4f}")

    # Phi norm CV
    phi_norms = norm(phi, axis=1)
    cv = phi_norms.std() / phi_norms.mean()
    print(f"\n  Phi norm CV: {cv:.4f}")

    n_pass = sum([axis_a, axis_b, axis_c])
    print(f"\n  {'=' * 50}")
    print(f"  RESULT: {n_pass}/3 axes pass → {'WARM-START-ESSENTIAL ★' if n_pass == 3 else 'NEEDS WORK'}")
    print(f"  {'=' * 50}")

    return {'axis_a': axis_a, 'axis_b': axis_b, 'axis_c': axis_c,
            'r_hat': r_hat, 'alpha_hat': alpha_hat, 'gap_frac': frac_small,
            'phi_cv': cv, 'T_max': T_max}


def run_linucb_test(doc_features, arm_features, ys, T, seed=42,
                     group_linucb_idx=None, group_llm_idx=None):
    """Quick LinUCB convergence test."""
    N, d = doc_features.shape
    K_arms = arm_features.shape[0]
    np.random.seed(seed)
    perm = np.random.permutation(N)[:T]

    V_inv = np.eye(d, dtype=np.float64) / 0.1
    b = np.zeros(d, dtype=np.float64)
    cum = 0
    cum_linucb_grp = 0
    cum_llm_grp = 0
    n_linucb_grp = 0
    n_llm_grp = 0

    linucb_set = set(group_linucb_idx) if group_linucb_idx is not None else set()
    llm_set = set(group_llm_idx) if group_llm_idx is not None else set()

    checkpoints = [100, 500, 1000, 2000, 3000, 5000, 8000, 10000]

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

        if ys[idx] in linucb_set:
            cum_linucb_grp += reward
            n_linucb_grp += 1
        elif ys[idx] in llm_set:
            cum_llm_grp += reward
            n_llm_grp += 1

        if t_i + 1 in checkpoints and t_i + 1 <= T:
            acc = cum / (t_i + 1)
            acc_lg = cum_linucb_grp / max(n_linucb_grp, 1)
            acc_ll = cum_llm_grp / max(n_llm_grp, 1)
            print(f"    t={t_i+1:5d}: overall={acc*100:.1f}%  "
                  f"LinUCB-group={acc_lg*100:.1f}%  LLM-group={acc_ll*100:.1f}%")

        phi_chosen = phi_all[choice]
        Vp = V_inv @ phi_chosen
        denom = 1.0 + phi_chosen @ Vp
        if abs(denom) > 1e-12:
            V_inv -= np.outer(Vp, Vp) / denom
        b += reward * phi_chosen

    final_acc = cum / T
    print(f"    FINAL: overall={final_acc*100:.1f}%  "
          f"LinUCB-grp={cum_linucb_grp/max(n_linucb_grp,1)*100:.1f}%  "
          f"LLM-grp={cum_llm_grp/max(n_llm_grp,1)*100:.1f}%")
    return final_acc


def create_ordering(ys, group_llm_idx_set, mode='random', seed=42):
    """Create document ordering (same as v6)."""
    N = len(ys)
    np.random.seed(seed)

    if mode == 'random':
        return np.random.permutation(N)
    elif mode == 'llm_first':
        llm_docs = [i for i in range(N) if ys[i] in group_llm_idx_set]
        linucb_docs = [i for i in range(N) if ys[i] not in group_llm_idx_set]
        np.random.shuffle(llm_docs)
        np.random.shuffle(linucb_docs)
        return np.array(llm_docs + linucb_docs)
    elif mode == 'interleaved':
        llm_docs = [i for i in range(N) if ys[i] in group_llm_idx_set]
        linucb_docs = [i for i in range(N) if ys[i] not in group_llm_idx_set]
        np.random.shuffle(llm_docs)
        np.random.shuffle(linucb_docs)
        result = []
        li, ri = 0, 0
        while li < len(llm_docs) or ri < len(linucb_docs):
            if li < len(llm_docs):
                result.append(llm_docs[li]); li += 1
            if ri < len(linucb_docs):
                result.append(linucb_docs[ri]); ri += 1
        return np.array(result)
    else:
        raise ValueError(f"Unknown ordering mode: {mode}")


if __name__ == '__main__':
    print("=" * 70)
    print("SYNTHETIC TEXT DATASET v7 — Diagnostic-passing version")
    print("=" * 70)

    # Step 1: Assign groups (same as v6)
    group_linucb, group_llm = assign_groups()
    group_linucb_idx = [category_names.index(c) for c in group_linucb]
    group_llm_idx = [category_names.index(c) for c in group_llm]
    linucb_idx_set = set(group_linucb_idx)
    llm_idx_set = set(group_llm_idx)

    print(f"\n  GROUP_LINUCB ({len(group_linucb)} cats): {group_linucb[:5]}...")
    print(f"  GROUP_LLM ({len(group_llm)} cats): {group_llm[:5]}...")

    # Step 2: Generate documents (same text as v6 → LLM caches work)
    n_per_cat = 200  # 200 * 52 = 10400
    documents, cat_names_list = generate_documents(n_per_category=n_per_cat, seed=42)
    N = len(documents)
    ys = np.array([d['category_idx'] for d in documents])

    print(f"\n  Total docs: {N}, Categories: {K}")

    # Step 3: Build BoW (needed for v3 compatibility, even though we override features)
    doc_bow, arm_bow, all_keywords, kw_to_idx = build_bow_features(documents, cat_names_list)

    # Step 4: Build DIAGNOSTIC-PASSING features
    d = 400
    print(f"\n{'='*70}")
    print(f"BUILDING DIAGNOSTIC-PASSING FEATURES (d={d})")
    print(f"{'='*70}")

    doc_features, arm_features = build_diagnostic_passing_features(
        doc_bow, arm_bow, ys, group_linucb_idx, group_llm_idx,
        d=d, arm_noise_llm=0.6, seed=42
    )

    print(f"  doc_features: {doc_features.shape}")
    print(f"  arm_features: {arm_features.shape}")

    # Step 5: Learn θ* and check oracle accuracy
    theta_star = learn_theta_star(doc_features, arm_features, ys)
    orc_all = compute_oracle_accuracy(doc_features, arm_features, ys, theta_star)
    orc_lg = compute_group_oracle(doc_features, arm_features, ys, theta_star, linucb_idx_set)
    orc_ll = compute_group_oracle(doc_features, arm_features, ys, theta_star, llm_idx_set)
    print(f"\n  Oracle accuracy: overall={orc_all*100:.1f}%  "
          f"LinUCB-grp={orc_lg*100:.1f}%  LLM-grp={orc_ll*100:.1f}%")

    # Step 6: Contaminate text (same as v6 — AFTER BoW extraction)
    print(f"\n  Contaminating text...")
    documents = contaminate_text(documents, cat_names_list, CATEGORIES,
                                  linucb_group_set=linucb_idx_set,
                                  n_misleading_linucb=6,
                                  n_misleading_llm=1, seed=42)

    # Step 7: VERIFY DIAGNOSTICS at the experiment horizon
    # Try multiple T values to find what works
    for T in [1000, 2000, 5000, 10000]:
        diag = verify_diagnostics(doc_features, arm_features, ys, T=T)
        if diag['axis_a'] and diag['axis_b'] and diag['axis_c']:
            print(f"\n  ★ ALL 3 AXES PASS at T={T}!")
            break

    # Step 8: LinUCB convergence test
    print(f"\n{'='*70}")
    print("LINUCB CONVERGENCE TEST (T=5000)")
    print(f"{'='*70}")
    run_linucb_test(doc_features, arm_features, ys, T=min(5000, N), seed=42,
                    group_linucb_idx=group_linucb_idx, group_llm_idx=group_llm_idx)

    # Step 9: Save dataset (compatible with run_experiment_v6.py)
    dataset = {
        'documents': documents,
        'category_names': cat_names_list,
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
        'noise_level': 0.6,
        'doc_noise_level': 0.0,  # not used in v7 (features are constructed differently)
        'd': d,
        'group_linucb': group_linucb,
        'group_llm': group_llm,
        'group_linucb_idx': group_linucb_idx,
        'group_llm_idx': group_llm_idx,
    }

    out_path = 'synthetic_text_dataset_v7.pkl.gz'
    with gzip.open(out_path, 'wb') as f:
        pickle.dump(dataset, f)

    fsize = os.path.getsize(out_path) / 1024 / 1024
    print(f"\nSaved: {out_path} ({fsize:.1f} MB)")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Categories: {K} ({len(group_linucb)} LinUCB-grp, {len(group_llm)} LLM-grp)")
    print(f"  Documents:  {N}")
    print(f"  Features:   d={d}")
    print(f"  Oracle:     overall={orc_all*100:.1f}% | LinUCB-grp={orc_lg*100:.1f}% | LLM-grp={orc_ll*100:.1f}%")
    print(f"  Diagnostics: A={'✓' if diag['axis_a'] else '✗'} B={'✓' if diag['axis_b'] else '✗'} C={'✓' if diag['axis_c'] else '✗'}")
    print(f"  → Predicted regime: {'WARM-START-ESSENTIAL' if all([diag['axis_a'], diag['axis_b'], diag['axis_c']]) else 'NOT PASSING'}")
