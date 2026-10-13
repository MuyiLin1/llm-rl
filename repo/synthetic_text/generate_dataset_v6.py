"""
generate_dataset_v6.py — Dataset with category-dependent complementarity.

Design principles for multiplication:
1. Split 52 categories into two groups:
   - GROUP_LINUCB (26 cats): CLEAN features, CONFUSING text
     → LinUCB learns these over time, LLM struggles
   - GROUP_LLM (26 cats): NOISY features, DISTINCTIVE text
     → LLM gets these right, LinUCB struggles with noisy features
2. All features share the same projected space (so passive V updates transfer)
3. LinUCB starts slow overall (~random) and climbs on GROUP_LINUCB cats
4. LLM is decent on GROUP_LLM cats, poor on GROUP_LINUCB cats
5. Hybrid = LLM on GROUP_LLM + LinUCB on GROUP_LINUCB > either alone

Ordering options for the experiment:
  - random: standard random permutation (fair baseline)
  - llm_first: GROUP_LLM docs first → passive V updates warm-start LinUCB
  - interleaved: alternate GROUP_LLM and GROUP_LINUCB docs
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
# GROUP ASSIGNMENT
# ================================================================
# GROUP_LINUCB: categories involved in confuser pairs (text is confusing)
# GROUP_LLM: categories NOT in confuser pairs, or one side of each pair

# Strategy: each confuser pair has two cats with similar text.
# Put BOTH confuser cats into GROUP_LINUCB (LLM struggles with these).
# Non-confuser cats go into GROUP_LLM (distinctive text, LLM handles these).

def assign_groups():
    """Split categories into LinUCB-friendly and LLM-friendly groups."""
    # Collect all categories that appear in confuser pairs
    in_pairs = set()
    for cat_a, cat_b in CONFUSER_PAIRS:
        in_pairs.add(cat_a)
        in_pairs.add(cat_b)

    group_linucb = []  # confusing text → LinUCB gets clean features
    group_llm = []     # distinctive text → LLM handles, noisy features

    for cat in category_names:
        if cat in in_pairs:
            group_linucb.append(cat)
        else:
            group_llm.append(cat)

    # If groups are too unbalanced, move some from the larger group
    # Target: roughly equal size
    while len(group_linucb) > len(group_llm) + 4:
        # Move a LinUCB cat that has fewest confuser partners to LLM group
        partner_counts = {}
        for cat in group_linucb:
            count = sum(1 for a, b in CONFUSER_PAIRS if a == cat or b == cat)
            partner_counts[cat] = count
        # Move the one with fewest partners (least confusing text)
        least = min(group_linucb, key=lambda c: partner_counts[c])
        group_linucb.remove(least)
        group_llm.append(least)

    return group_linucb, group_llm


def build_complementary_features(doc_bow, arm_bow, ys, group_linucb_idx, group_llm_idx,
                                  d=150, arm_noise=0.8, doc_noise=0.5, seed=42):
    """
    Build features with category-dependent ARM noise + arm scaling for phi CV.

    Key insight: corrupt ARM features for LLM-group categories so LinUCB can't
    distinguish those arms. The text stays distinctive → LLM handles them.
    LinUCB-group arms stay clean → LinUCB learns to classify those.

    Arm scaling (3 tiers) creates phi norm variation → non-binary routing.
    """
    np.random.seed(seed)
    V = doc_bow.shape[1]
    N = doc_bow.shape[0]
    K_arms = arm_bow.shape[0]

    # Shared random projection
    W = np.random.randn(V, d) / np.sqrt(d)

    # Project documents and arms
    doc_proj = doc_bow @ W
    arm_proj = arm_bow @ W

    # Normalize arm features first
    arm_norms_raw = norm(arm_proj, axis=1, keepdims=True)
    arm_norms_raw[arm_norms_raw < 1e-8] = 1.0
    arm_features = arm_proj / arm_norms_raw

    # CORRUPT arm features for LLM-group categories
    # Mix clean arm vector with random noise → makes these arms indistinguishable
    llm_set = set(group_llm_idx)
    arm_noise_matrix = np.random.randn(K_arms, d) / np.sqrt(d)
    for k in range(K_arms):
        if k in llm_set:
            # Mix: (1-arm_noise) * clean + arm_noise * random
            arm_features[k] = (1.0 - arm_noise) * arm_features[k] + arm_noise * arm_noise_matrix[k]
            # Re-normalize
            n = norm(arm_features[k])
            if n > 1e-8:
                arm_features[k] /= n

    # ARM SCALING: 3 tiers to create phi norm variation
    # This is critical for non-binary routing (phi CV > 0.3)
    third = K_arms // 3
    arm_scales = np.ones(K_arms)
    arm_scales[:third] = 0.3
    arm_scales[third:2*third] = 0.7
    arm_scales[2*third:] = 1.5
    np.random.shuffle(arm_scales)
    arm_features = arm_features * arm_scales[:, None]

    # Document features: light uniform noise for realism
    doc_noise_matrix = np.random.randn(N, d) / np.sqrt(d)
    doc_features = doc_proj + doc_noise * doc_noise_matrix
    doc_norms = norm(doc_features, axis=1, keepdims=True)
    doc_norms[doc_norms < 1e-8] = 1.0
    doc_features = doc_features / doc_norms

    return doc_features, arm_features


def contaminate_text(documents, category_names, categories_dict,
                     linucb_group_set, n_misleading_linucb=6,
                     n_misleading_llm=1, seed=42):
    """
    Post-process document text to add misleading keywords from OTHER categories.
    Called AFTER build_bow_features so LinUCB features are unaffected.
    
    ASYMMETRIC contamination:
    - LinUCB-group docs: HEAVY contamination (n_misleading_linucb keywords)
      → LLM can't classify these, must rely on LinUCB's clean features
    - LLM-group docs: LIGHT contamination (n_misleading_llm keywords)
      → LLM can still handle these from distinctive text
    """
    rng = random.Random(seed)
    all_cats = list(categories_dict.keys())
    
    n_heavy, n_light = 0, 0
    for doc in documents:
        true_cat = doc['category_name']
        cat_idx = category_names.index(true_cat)
        
        # Asymmetric: more contamination for LinUCB-group
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


def learn_theta_star(doc_features, arm_features, ys):
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


def run_linucb_test(doc_features, arm_features, ys, T, seed=42,
                     group_linucb_idx=None, group_llm_idx=None):
    """LinUCB convergence test with per-group accuracy tracking."""
    N, d = doc_features.shape
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

    checkpoints = [100, 200, 500, 1000, 2000, 3000, 5000]

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

    return cum / T


def create_ordering(ys, group_llm_idx_set, mode='random', seed=42):
    """Create document ordering for the experiment.

    Modes:
      random: standard random permutation
      llm_first: all GROUP_LLM docs first, then GROUP_LINUCB
      interleaved: alternate between groups
    """
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


def verify_diagnostics(doc_features, arm_features, ys, T=5000):
    K_arms = arm_features.shape[0]
    N = doc_features.shape[0]
    d = doc_features.shape[1]

    print(f"\n{'=' * 70}")
    print(f"DIAGNOSTICS (T={T}, K={K_arms}, d={d}, N={N})")
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
    axis_a = ratio >= 1
    print(f"  Axis A (Effective Rank): r̂ = {r_hat:.2f}, r̂/√T = {ratio:.4f} {'✓' if axis_a else '✗'}")

    # Phi norm CV
    phi_norms = norm(phi, axis=1)
    cv = phi_norms.std() / phi_norms.mean()
    print(f"  Phi norm CV: {cv:.4f} (>0.3 good for adaptive routing)")

    # AXIS C: Gap distribution
    n_gap = min(2000, N)
    gi = np.random.choice(N, n_gap, replace=False)
    gaps = []
    for i in gi:
        phi_all = doc_features[i:i+1] * arm_features
        scores = phi_all @ np.ones(d)  # approximate with ones
        sorted_s = np.sort(scores)[::-1]
        if len(sorted_s) >= 2:
            gaps.append(sorted_s[0] - sorted_s[1])
    gaps = np.array(gaps)
    frac_small = np.mean(gaps <= 0.05)
    axis_c = frac_small > 0.5
    print(f"  Axis C (Gap tail): frac(gap≤0.05) = {frac_small:.4f} {'✓' if axis_c else '✗'}")

    return {'axis_a': axis_a, 'axis_b': axis_b, 'axis_c': axis_c,
            'r_hat': r_hat, 'phi_cv': cv}


if __name__ == '__main__':
    print("=" * 70)
    print("SYNTHETIC TEXT DATASET v6 — Category-dependent complementarity")
    print("=" * 70)

    # Step 1: Assign groups
    group_linucb, group_llm = assign_groups()
    group_linucb_idx = [category_names.index(c) for c in group_linucb]
    group_llm_idx = [category_names.index(c) for c in group_llm]

    print(f"\n  GROUP_LINUCB ({len(group_linucb)} cats, clean features, confusing text):")
    for c in group_linucb:
        print(f"    {c}")
    print(f"\n  GROUP_LLM ({len(group_llm)} cats, noisy features, distinctive text):")
    for c in group_llm:
        print(f"    {c}")

    # Step 2: Generate documents
    n_per_cat = 200  # 200 * 52 = 10400
    documents, cat_names_list = generate_documents(n_per_category=n_per_cat, seed=42)
    N = len(documents)
    ys = np.array([d['category_idx'] for d in documents])
    doc_cats = ys  # global reference for build_complementary_features

    n_linucb_docs = sum(1 for y in ys if y in set(group_linucb_idx))
    n_llm_docs = N - n_linucb_docs
    print(f"\n  Total docs: {N}")
    print(f"  LinUCB-group docs: {n_linucb_docs} ({100*n_linucb_docs/N:.0f}%)")
    print(f"  LLM-group docs: {n_llm_docs} ({100*n_llm_docs/N:.0f}%)")

    # Step 3: Build BoW features
    print("\nBuilding BoW features...")
    doc_bow, arm_bow, all_keywords, kw_to_idx = build_bow_features(documents, cat_names_list)

    # Step 4: Build features with category-dependent ARM noise
    d = 400  # large d → slower LinUCB convergence → more value from passive warm-start

    # Tune arm_noise: controls how corrupted LLM-group arm features are
    # Higher arm_noise → harder for LinUCB on LLM-group, no effect on LinUCB-group
    print(f"\n{'='*70}")
    print(f"TESTING ARM NOISE LEVELS (d={d})")
    print(f"{'='*70}")

    llm_idx_set = set(group_llm_idx)
    linucb_idx_set = set(group_linucb_idx)

    # Step 5: Use best noise and build final dataset
    arm_noise_level = 0.8
    doc_noise_level = 0.3
    print(f"\n{'='*70}")
    print(f"BUILDING FINAL DATASET (d={d}, arm_noise={arm_noise_level}, doc_noise={doc_noise_level})")
    print(f"{'='*70}")

    doc_features, arm_features = build_complementary_features(
        doc_bow, arm_bow, ys, group_linucb_idx, group_llm_idx,
        d=d, arm_noise=arm_noise_level, doc_noise=doc_noise_level, seed=42
    )

    print(f"  doc_features: {doc_features.shape}")
    print(f"  arm_features: {arm_features.shape}")

    # Learn theta*
    theta_star = learn_theta_star(doc_features, arm_features, ys)

    # Oracle per group
    orc_all = compute_oracle_accuracy(doc_features, arm_features, ys, theta_star)
    orc_lg = compute_group_oracle(doc_features, arm_features, ys, theta_star, linucb_idx_set)
    orc_ll = compute_group_oracle(doc_features, arm_features, ys, theta_star, llm_idx_set)
    print(f"\n  Oracle: overall={orc_all*100:.1f}%  "
          f"LinUCB-grp={orc_lg*100:.1f}%  LLM-grp={orc_ll*100:.1f}%")

    # Step 6: Contaminate text ASYMMETRICALLY (AFTER BoW extraction, does NOT affect LinUCB features)
    n_misleading_linucb = 6  # heavy: LLM can't classify LinUCB-group docs
    n_misleading_llm = 1     # light: LLM can still handle LLM-group docs
    print(f"\n  Contaminating text (LinUCB-grp: {n_misleading_linucb} kw, LLM-grp: {n_misleading_llm} kw)...")
    documents = contaminate_text(documents, cat_names_list, CATEGORIES,
                                  linucb_group_set=linucb_idx_set,
                                  n_misleading_linucb=n_misleading_linucb,
                                  n_misleading_llm=n_misleading_llm, seed=42)
    print(f"  Done. Example contaminated doc:")
    print(f"    Cat: {documents[0]['category_name']}")
    print(f"    Text: {documents[0]['text'][:200]}...")

    # Diagnostics
    diag = verify_diagnostics(doc_features, arm_features, ys)

    # Full LinUCB convergence test
    print(f"\n{'='*70}")
    print("LINUCB CONVERGENCE TEST (T=5000)")
    print(f"{'='*70}")
    run_linucb_test(doc_features, arm_features, ys, T=5000, seed=42,
                    group_linucb_idx=group_linucb_idx, group_llm_idx=group_llm_idx)

    # Save
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
        'noise_level': arm_noise_level,
        'doc_noise_level': doc_noise_level,
        'd': d,
        'group_linucb': group_linucb,
        'group_llm': group_llm,
        'group_linucb_idx': group_linucb_idx,
        'group_llm_idx': group_llm_idx,
    }

    out_path = 'synthetic_text_dataset_v6.pkl.gz'
    with gzip.open(out_path, 'wb') as f:
        pickle.dump(dataset, f)

    fsize = os.path.getsize(out_path) / 1024 / 1024
    print(f"\nSaved: {out_path} ({fsize:.1f} MB)")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Categories: {K} ({len(group_linucb)} LinUCB-friendly, {len(group_llm)} LLM-friendly)")
    print(f"  Documents:  {N}")
    print(f"  Features:   d={d}")
    print(f"  Noise:      arm={arm_noise_level} (LLM-group arms), doc={doc_noise_level} (all)")
    print(f"  Oracle:     overall={orc_all*100:.1f}% | LinUCB-grp={orc_lg*100:.1f}% | LLM-grp={orc_ll*100:.1f}%")
    print(f"  Phi CV:     {diag['phi_cv']:.4f}")
    print(f"  Target: LinUCB learns LinUCB-grp, LLM handles LLM-grp → hybrid > both")
