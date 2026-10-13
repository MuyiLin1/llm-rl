"""
generate_dataset_v4.py — Redesigned dataset for STRONG multiplication effect.

KEY CHANGES vs v3 (which only got 1-2% multiplication):

Problem 1: LinUCB converges instantly (41% at t=100, oracle=46%)
  FIX: Use a COMPLEX theta* (not ones), higher d, and more categories so 
       LinUCB genuinely needs many samples to learn.

Problem 2: Phi norms are constant (CV=0.10) → routing is all-or-nothing
  FIX: Use HETEROGENEOUS arm features (varying magnitudes) so phi norms
       differ substantially across documents and arms.

Problem 3: LLM helps equally early and late (no temporal structure)
  FIX: Design "LLM-easy" categories (clear text cues the LLM can see) vs
       "LinUCB-easy" categories (numerical feature patterns the LLM misses).
       The LLM helps early on the LLM-easy ones while LinUCB is still learning.

Problem 4: Only 5200 documents with T=2000
  FIX: Generate 20000+ documents for longer horizon T=10000+

Design principle from the paper: Multiplication happens when:
  - LLM is good early (provides warm start)
  - LinUCB eventually surpasses LLM (learns the pattern)
  - Router correctly transitions from LLM→LinUCB over time

Strategy: Create TWO types of categories:
  Type A ("LLM-friendly"): Categories with clear textual descriptions
    → LLMs classify these well from text alone
    → LinUCB needs many examples because features are subtle
  Type B ("LinUCB-friendly"): Categories with confusing text but clean features
    → LLMs struggle (confuser text, ambiguous wording)
    → LinUCB learns quickly from feature patterns
    
This ensures LLM is COMPLEMENTARY to LinUCB (not redundant).
"""

import numpy as np
import random
import gzip
import pickle
from numpy.linalg import norm
from scipy.linalg import svd


# ================================================================
# CATEGORY DESIGN
# ================================================================

# Type A: LLM-friendly categories — clear distinct topics that LLMs can identify
# but with features that are harder for LinUCB (projections mix dimensions)
LLM_FRIENDLY_CATEGORIES = {
    "Quantum Mechanics": [
        "quantum entanglement", "wave function", "superposition",
        "uncertainty principle", "quantum tunneling", "photon",
        "spin state", "quantum decoherence", "Planck constant",
        "Schrodinger equation", "quantum field theory", "Hilbert space"
    ],
    "Machine Learning": [
        "neural network", "gradient descent", "backpropagation",
        "overfitting regularization", "convolutional layer", "transformer attention",
        "loss function optimization", "batch normalization", "dropout layer",
        "learning rate schedule", "weight initialization", "activation function"
    ],
    "Climate Science": [
        "greenhouse gas emission", "global temperature anomaly", "ice sheet melting",
        "sea level rise projection", "carbon dioxide concentration", "climate model simulation",
        "ocean heat content", "radiative forcing feedback", "permafrost thaw",
        "extreme weather attribution", "renewable energy transition", "carbon capture"
    ],
    "Neuroscience": [
        "synaptic plasticity", "neurotransmitter release", "action potential",
        "hippocampal memory", "cortical column", "neural oscillation",
        "brain imaging fMRI", "dopamine reward pathway", "axon myelination",
        "long-term potentiation", "glial cell function", "neurodegeneration"
    ],
    "Organic Chemistry": [
        "carbon bond formation", "nucleophilic substitution", "Grignard reagent",
        "aromatic ring system", "stereochemistry chirality", "catalytic hydrogenation",
        "functional group transformation", "reaction mechanism pathway",
        "spectroscopy characterization", "synthesis yield optimization",
        "protecting group strategy", "retrosynthetic analysis"
    ],
    "Evolutionary Biology": [
        "natural selection fitness", "genetic drift population", "speciation event",
        "phylogenetic tree analysis", "adaptation mechanism", "convergent evolution",
        "sexual selection display", "gene flow migration", "molecular clock dating",
        "fossil record evidence", "homologous structure", "coevolution dynamics"
    ],
    "Astrophysics": [
        "dark matter distribution", "cosmic microwave background", "gravitational wave",
        "black hole merger", "galaxy formation evolution", "supernova explosion",
        "neutron star collision", "redshift measurement", "Hubble constant tension",
        "exoplanet atmosphere", "stellar nucleosynthesis", "quasar luminosity"
    ],
    "Cryptography": [
        "public key infrastructure", "elliptic curve algorithm", "hash function collision",
        "symmetric encryption block", "digital signature verification",
        "zero knowledge proof", "quantum resistant lattice",
        "key exchange protocol", "random number generation",
        "homomorphic encryption computation", "blockchain consensus",
        "side channel attack"
    ],
    "Immunology": [
        "T cell receptor binding", "antibody specificity", "immune memory",
        "cytokine signaling cascade", "antigen presentation MHC",
        "B cell activation germinal", "innate immune response",
        "autoimmune disorder mechanism", "vaccine adjuvant",
        "complement system activation", "inflammatory response",
        "immunotherapy checkpoint"
    ],
    "Plate Tectonics": [
        "subduction zone dynamics", "mid-ocean ridge spreading",
        "continental drift evidence", "earthquake fault rupture",
        "volcanic arc formation", "lithospheric plate boundary",
        "mantle convection current", "transform fault motion",
        "oceanic crust formation", "mountain building orogeny",
        "seismic wave propagation", "hotspot volcanic chain"
    ],
}

# Type B: LinUCB-friendly categories — confusing text (overlapping language)
# but features have clean separable structure  
LINUCB_FRIENDLY_CATEGORIES = {
    "Signal Processing": [
        "frequency domain analysis", "filter response", "spectral estimation",
        "time series analysis", "sampling theorem", "noise reduction",
        "convolution operation", "bandwidth limitation", "modulation scheme",
        "waveform reconstruction", "phase detection", "digital conversion"
    ],
    "Control Systems": [
        "frequency response analysis", "filter design", "spectral method",
        "time domain analysis", "sampling control", "noise rejection",
        "convolution integral", "bandwidth allocation", "modulation feedback",
        "waveform generation", "phase margin", "digital controller"
    ],
    "Probability Theory": [
        "random variable distribution", "expectation calculation", "variance estimation",
        "conditional probability event", "Bayesian inference update",
        "central limit theorem", "Markov chain transition",
        "convergence guarantee", "hypothesis testing", "confidence interval",
        "maximum likelihood estimation", "sufficient statistic"
    ],
    "Statistical Mechanics": [
        "random state distribution", "expectation value ensemble", "variance fluctuation",
        "conditional partition function", "Bayesian statistical weight",
        "central limit scaling", "Markov process equilibrium",
        "convergence thermodynamic", "hypothesis ensemble", "confidence thermodynamic",
        "maximum entropy principle", "sufficient observable"
    ],
    "Fluid Dynamics": [
        "flow field velocity", "pressure gradient force", "turbulent mixing",
        "boundary condition layer", "Reynolds transport", "viscous dissipation",
        "vortex dynamics rotation", "compressible flow regime",
        "conservation momentum equation", "numerical simulation method",
        "stability criterion analysis", "drag coefficient estimation"
    ],
    "Heat Transfer": [
        "flow thermal convection", "temperature gradient force", "turbulent heat mixing",
        "boundary thermal layer", "Reynolds analogy transport", "viscous heating dissipation",
        "convective circulation rotation", "compressible thermal regime",
        "conservation energy equation", "numerical thermal simulation",
        "stability thermal criterion", "heat transfer coefficient"
    ],
    "Robotics": [
        "trajectory planning motion", "sensor fusion algorithm", "actuator control loop",
        "inverse kinematics solver", "obstacle avoidance path",
        "simultaneous localization mapping", "force feedback control",
        "manipulation grasp planning", "navigation autonomous system",
        "computer vision recognition", "reinforcement policy learning",
        "hardware interface communication"
    ],
    "Autonomous Vehicles": [
        "trajectory prediction planning", "sensor fusion perception", "actuator steering control",
        "inverse dynamics model", "obstacle detection avoidance",
        "simultaneous positioning navigation", "force brake control",
        "lane keeping planning", "navigation route system",
        "computer vision detection", "reinforcement driving learning",
        "vehicle communication protocol"
    ],
    "Database Systems": [
        "query optimization execution", "index structure design", "transaction isolation",
        "storage engine architecture", "replication consistency protocol",
        "sharding partition strategy", "cache invalidation policy",
        "schema migration evolution", "lock contention resolution",
        "distributed consensus algorithm", "backup recovery strategy",
        "access pattern optimization"
    ],
    "Distributed Computing": [
        "query routing execution", "index distributed design", "transaction coordination",
        "storage distributed architecture", "replication distributed protocol",
        "sharding distributed strategy", "cache distributed policy",
        "schema distributed evolution", "lock distributed resolution",
        "distributed agreement algorithm", "fault tolerance strategy",
        "load balancing optimization"
    ],
}

# Confuser pairs: LinUCB-friendly categories share similar TEXT
# but have different feature structures
CONFUSER_PAIRS = [
    ("Signal Processing", "Control Systems"),
    ("Probability Theory", "Statistical Mechanics"),
    ("Fluid Dynamics", "Heat Transfer"),
    ("Robotics", "Autonomous Vehicles"),
    ("Database Systems", "Distributed Computing"),
]


# ================================================================
# TEXT GENERATION
# ================================================================

# Templates for clear category documents (LLM-friendly)
CLEAR_TEMPLATES = [
    "This research paper investigates {kw1} in the field of {cat}. The study focuses on {kw2} and presents new findings related to {kw3}.",
    "A comprehensive study of {kw1} reveals important connections to {kw2}. The authors demonstrate advances in {kw3} within {cat}.",
    "Recent breakthroughs in {cat} have highlighted the importance of {kw1}. This work combines {kw2} with novel approaches to {kw3}.",
    "The {cat} community has long studied {kw1}. We present a unified framework connecting {kw2} and {kw3} through rigorous analysis.",
    "In this paper on {cat}, we examine {kw1} from a new perspective. Our approach leverages {kw2} to achieve improvements in {kw3}.",
]

# Templates for confusing documents (LinUCB-friendly — text is ambiguous)
CONFUSING_TEMPLATES = [
    "This work presents an analysis of {kw1} with applications to {kw2}. The methodology builds on {kw3} using established techniques.",
    "We study the properties of {kw1} and its relationship to {kw2}. Results show that {kw3} plays a central role in the framework.",
    "An investigation of {kw1} reveals connections to {kw2}. The theoretical foundation relies on {kw3} as a key component.",
    "The study of {kw1} has been advanced by recent work on {kw2}. We extend these ideas through {kw3} in a unified treatment.",
    "Building on prior work in {kw1}, this paper develops new results for {kw2} and demonstrates applications to {kw3}.",
]


def generate_documents(n_per_category=400, seed=42):
    """Generate text documents for all categories."""
    random.seed(seed)
    np.random.seed(seed)
    
    all_categories = {}
    all_categories.update(LLM_FRIENDLY_CATEGORIES)
    all_categories.update(LINUCB_FRIENDLY_CATEGORIES)
    
    category_names = list(all_categories.keys())
    K = len(category_names)
    
    # Track which are LLM-friendly vs LinUCB-friendly
    llm_friendly_idx = set(range(len(LLM_FRIENDLY_CATEGORIES)))
    linucb_friendly_idx = set(range(len(LLM_FRIENDLY_CATEGORIES), K))
    
    documents = []
    
    for cat_idx, cat_name in enumerate(category_names):
        keywords = all_categories[cat_name]
        is_llm_friendly = cat_idx in llm_friendly_idx
        
        # Choose template style based on category type
        if is_llm_friendly:
            templates = CLEAR_TEMPLATES
        else:
            templates = CONFUSING_TEMPLATES
        
        for doc_i in range(n_per_category):
            template = random.choice(templates)
            kws = random.sample(keywords, 3)
            
            # For confuser categories, sometimes mix in keywords from the paired category
            is_confuser_doc = False
            if not is_llm_friendly:
                # Find confuser pair
                partner = None
                for (a, b) in CONFUSER_PAIRS:
                    if cat_name == a:
                        partner = b
                        break
                    elif cat_name == b:
                        partner = a
                        break
                
                # 40% of docs get mixed keywords from partner
                if partner and random.random() < 0.40:
                    partner_kws = all_categories[partner]
                    # Replace 1-2 keywords with partner's
                    n_replace = random.randint(1, 2)
                    for r in range(n_replace):
                        kws[r] = random.choice(partner_kws)
                    is_confuser_doc = True
            
            if is_llm_friendly:
                text = template.format(kw1=kws[0], kw2=kws[1], kw3=kws[2], cat=cat_name)
            else:
                # Don't include category name in text for confusing categories!
                text = template.format(kw1=kws[0], kw2=kws[1], kw3=kws[2])
            
            documents.append({
                'text': text,
                'category_idx': cat_idx,
                'category_name': cat_name,
                'is_confuser': is_confuser_doc,
                'is_llm_friendly': is_llm_friendly,
            })
    
    random.shuffle(documents)
    return documents, category_names


# ================================================================
# FEATURE ENGINEERING — The key to making LinUCB hard
# ================================================================

def build_features(documents, category_names, d=200, seed=42):
    """
    Build features designed to create:
    1. High phi norm variance (so routing is document-specific)
    2. Slow LinUCB convergence (theta* is complex)
    3. Still solvable by LinUCB given enough data
    
    Strategy:
    - Document features: Dense random vectors (not sparse BoW) with varying magnitudes
    - Arm features: Structured but non-trivial embeddings
    - theta*: Learned from an oracle model (not just ones)
    """
    np.random.seed(seed)
    random.seed(seed)
    
    N = len(documents)
    K = len(category_names)
    all_categories = {}
    all_categories.update(LLM_FRIENDLY_CATEGORIES)
    all_categories.update(LINUCB_FRIENDLY_CATEGORIES)
    
    # --- Build vocabulary for BoW ---
    keyword_to_idx = {}
    all_keywords = []
    for cat_name in category_names:
        for kw in all_categories[cat_name]:
            if kw not in keyword_to_idx:
                keyword_to_idx[kw] = len(all_keywords)
                all_keywords.append(kw)
    
    V = len(all_keywords)
    
    # Document BoW
    doc_bow = np.zeros((N, V))
    for i, doc in enumerate(documents):
        text_lower = doc['text'].lower()
        for kw, idx in keyword_to_idx.items():
            if kw.lower() in text_lower:
                doc_bow[i, idx] = 1.0
    
    # Arm BoW
    arm_bow = np.zeros((K, V))
    for k, cat_name in enumerate(category_names):
        for kw in all_categories[cat_name]:
            if kw in keyword_to_idx:
                arm_bow[k, keyword_to_idx[kw]] = 1.0
    
    # --- SHARED random projection for both doc and arm features ---
    # Key: use the SAME projection so phi = doc * arm is meaningful
    W = np.random.randn(V, d) / np.sqrt(V)
    
    doc_features = doc_bow @ W
    arm_features = arm_bow @ W
    
    n_llm = len(LLM_FRIENDLY_CATEGORIES)
    
    # Normalize arm features to unit norm, then scale by type
    arm_mags = norm(arm_features, axis=1, keepdims=True)
    arm_mags[arm_mags < 1e-8] = 1.0
    arm_features = arm_features / arm_mags
    # LLM-friendly arms: smaller magnitude → phi norms smaller → routing prefers LLM
    arm_features[:n_llm] *= 0.4
    # LinUCB-friendly arms: larger magnitude → phi norms larger → routing prefers LinUCB
    arm_features[n_llm:] *= 1.2
    
    # Add heterogeneous noise to document features (creates phi norm variance)
    doc_noise_scale = np.random.exponential(0.2, size=N)
    doc_noise = np.random.randn(N, d) * doc_noise_scale[:, None] / np.sqrt(d)
    doc_features = doc_features + doc_noise
    
    # Normalize documents to unit norm (standard for LinUCB)
    doc_mags = norm(doc_features, axis=1, keepdims=True)
    doc_mags[doc_mags < 1e-8] = 1.0
    doc_features = doc_features / doc_mags
    
    # --- Build a complex theta* ---
    # With shared projection W, phi(x,a) = x*a where x = bow_doc@W, a = bow_arm@W
    # The natural theta for this is related to the projection structure.
    # Use regression on observed correct-arm phi vectors to find theta*.
    
    # Collect phi vectors for correct arms
    phis_correct = []
    phis_incorrect = []
    for i in range(min(N, 8000)):
        y = documents[i]['category_idx']
        phi_correct = doc_features[i] * arm_features[y]
        phis_correct.append(phi_correct)
        # Random wrong arm
        neg = random.choice([k for k in range(K) if k != y])
        phis_incorrect.append(doc_features[i] * arm_features[neg])
    
    phis_correct = np.array(phis_correct)
    phis_incorrect = np.array(phis_incorrect)
    
    # Solve: find theta s.t. phi_correct @ theta = 1, phi_incorrect @ theta = 0
    all_phi = np.vstack([phis_correct, phis_incorrect])
    all_targets = np.concatenate([np.ones(len(phis_correct)), np.zeros(len(phis_incorrect))])
    
    from sklearn.linear_model import Ridge
    reg = Ridge(alpha=0.01)
    reg.fit(all_phi, all_targets)
    theta_star = reg.coef_
    
    # Verify oracle accuracy
    correct = 0
    for i in range(N):
        y = documents[i]['category_idx']
        phi_all = doc_features[i:i+1] * arm_features  # (K, d)
        scores = phi_all @ theta_star
        if np.argmax(scores) == y:
            correct += 1
    oracle_acc = correct / N
    print(f"  Oracle accuracy (learned theta*): {oracle_acc*100:.2f}%")
    
    if oracle_acc < 0.3:
        print("  WARNING: Oracle accuracy too low — trying simple mean phi approach")
        # Fallback: theta* = mean of correct phi vectors
        theta_star = phis_correct.mean(axis=0)
        correct = 0
        for i in range(N):
            y = documents[i]['category_idx']
            phi_all = doc_features[i:i+1] * arm_features
            scores = phi_all @ theta_star
            if np.argmax(scores) == y:
                correct += 1
        oracle_acc = correct / N
        print(f"  Oracle accuracy (mean phi): {oracle_acc*100:.2f}%")
    
    return doc_features, arm_features, doc_bow, arm_bow, all_keywords, keyword_to_idx, theta_star


def verify_diagnostics(doc_features, arm_features, ys, T=5000):
    """Check all 3 axes."""
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
    print(f"\nAxis B (Coherence): α̂ = {alpha_hat:.4f}")
    axis_b = alpha_hat < 0.1

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
    print(f"\nAxis A (Effective Rank): r̂ = {r_hat:.2f}, r̂/√T = {ratio:.4f}")
    axis_a = ratio >= 1

    # AXIS C: Gap distribution
    theta_star_simple = np.ones(d)  # fallback
    n_test = min(3000, N)
    test_idx = np.random.choice(N, n_test, replace=False)
    gaps = []
    for idx in test_idx:
        phi_all = doc_features[idx:idx+1] * arm_features
        scores = phi_all @ theta_star_simple
        sorted_s = np.sort(scores)[::-1]
        gaps.append(sorted_s[0] - sorted_s[1])
    gaps = np.array(gaps)
    frac_heavy = np.mean(gaps <= 0.05)
    print(f"\nAxis C (Gap Distribution): frac(gap ≤ 0.05) = {frac_heavy:.4f}")
    axis_c = frac_heavy > 0.5

    n_dev = sum([axis_a, axis_b, axis_c])
    print(f"\nSUMMARY: A={'✓' if axis_a else '✗'} B={'✓' if axis_b else '✗'} C={'✓' if axis_c else '✗'} → {n_dev}/3 deviate")
    
    # Phi norm statistics
    phi_sample = doc_features[:1000, None, :] * arm_features[None, :, :]  # (1000, K, d)
    phi_norms = norm(phi_sample, axis=2)
    cv = phi_norms.std() / phi_norms.mean()
    print(f"\n  Phi norm: mean={phi_norms.mean():.4f} std={phi_norms.std():.4f} CV={cv:.4f}")
    print(f"  (CV > 0.3 needed for adaptive routing)")
    
    return {'axis_a': axis_a, 'axis_b': axis_b, 'axis_c': axis_c,
            'r_hat': r_hat, 'alpha_hat': alpha_hat, 'gap_frac': frac_heavy,
            'phi_cv': cv}


def run_quick_linucb(doc_features, arm_features, ys, T=5000, seed=42):
    """Quick LinUCB test to verify convergence speed."""
    N, d = doc_features.shape
    K = arm_features.shape[0]
    np.random.seed(seed)
    perm = np.random.permutation(N)[:T]
    
    V_inv = np.eye(d, dtype=np.float64) / 0.1
    b = np.zeros(d, dtype=np.float64)
    cum = 0
    
    checkpoints = [100, 500, 1000, 2000, 5000, 8000]
    print(f"\n  LinUCB convergence (T={T}):")
    
    for t_i in range(T):
        idx = perm[t_i]
        x = doc_features[idx]
        alpha = max(0.05, 1.0 * (0.9995 ** t_i))
        
        phi_all = x[None, :] * arm_features
        theta = V_inv @ b
        scores = phi_all @ theta
        V_phi = phi_all @ V_inv
        var = np.sum(V_phi * phi_all, axis=1)
        ucb = scores + alpha * np.sqrt(np.maximum(var, 0))
        choice = int(np.argmax(ucb))
        
        reward = 1.0 if choice == ys[idx] else 0.0
        cum += reward
        
        if t_i + 1 in checkpoints:
            print(f"    t={t_i+1:5d}: acc={cum/(t_i+1)*100:.2f}%")
        
        phi_chosen = phi_all[choice]
        Vp = V_inv @ phi_chosen
        denom = 1.0 + phi_chosen @ Vp
        if abs(denom) > 1e-12:
            V_inv -= np.outer(Vp, Vp) / denom
        b += reward * phi_chosen
    
    print(f"    Final: {cum/T*100:.2f}%")
    return cum / T


if __name__ == '__main__':
    print("=" * 70)
    print("SYNTHETIC TEXT DATASET v4 — Designed for strong multiplication")
    print("=" * 70)
    
    K = len(LLM_FRIENDLY_CATEGORIES) + len(LINUCB_FRIENDLY_CATEGORIES)
    print(f"\n{K} categories: {len(LLM_FRIENDLY_CATEGORIES)} LLM-friendly, "
          f"{len(LINUCB_FRIENDLY_CATEGORIES)} LinUCB-friendly")
    print(f"Confuser pairs: {len(CONFUSER_PAIRS)}")
    
    # Generate docs
    N_PER_CAT = 500  # 500 * 20 = 10000 documents
    documents, category_names = generate_documents(n_per_category=N_PER_CAT, seed=42)
    N = len(documents)
    print(f"\nTotal documents: {N}")
    n_confuser = sum(1 for d in documents if d['is_confuser'])
    n_llm_friendly = sum(1 for d in documents if d['is_llm_friendly'])
    print(f"  LLM-friendly: {n_llm_friendly} ({100*n_llm_friendly/N:.1f}%)")
    print(f"  Confuser docs: {n_confuser} ({100*n_confuser/N:.1f}%)")
    
    # Build features
    d = 200
    print(f"\nBuilding features (d={d})...")
    doc_features, arm_features, doc_bow, arm_bow, keywords, kw_to_idx, theta_star = \
        build_features(documents, category_names, d=d, seed=42)
    
    print(f"  doc_features: {doc_features.shape}")
    print(f"  arm_features: {arm_features.shape}")
    
    # Verify diagnostics
    ys = np.array([d['category_idx'] for d in documents])
    diag = verify_diagnostics(doc_features, arm_features, ys, T=5000)
    
    # Test LinUCB convergence
    run_quick_linucb(doc_features, arm_features, ys, T=min(8000, N), seed=42)
    
    # Save
    dataset = {
        'documents': documents,
        'category_names': category_names,
        'doc_features': doc_features,
        'arm_features': arm_features,
        'doc_bow': doc_bow,
        'arm_bow': arm_bow,
        'ys': ys,
        'all_keywords': keywords,
        'keyword_to_idx': kw_to_idx,
        'theta_star': theta_star,
        'n_categories': K,
        'n_documents': N,
        'confuser_pairs': CONFUSER_PAIRS,
        'llm_friendly_categories': list(LLM_FRIENDLY_CATEGORIES.keys()),
        'linucb_friendly_categories': list(LINUCB_FRIENDLY_CATEGORIES.keys()),
    }
    
    out_path = 'synthetic_text_dataset_v4.pkl.gz'
    with gzip.open(out_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"\nSaved to {out_path}")
    print(f"Size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")
    
    # Print sample texts
    print(f"\nSample texts:")
    for t in ['llm_friendly', 'confuser']:
        for doc in documents:
            if t == 'llm_friendly' and doc['is_llm_friendly']:
                print(f"  [LLM-friendly: {doc['category_name']}]")
                print(f"    {doc['text'][:120]}...")
                break
            elif t == 'confuser' and doc['is_confuser']:
                print(f"  [Confuser: {doc['category_name']}]")
                print(f"    {doc['text'][:120]}...")
                break


import os
