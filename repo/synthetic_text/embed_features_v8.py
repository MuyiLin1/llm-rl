"""
embed_features_v8.py — Replace BoW features with sentence embeddings

Uses sentence-transformers (all-MiniLM-L6-v2, 384d) to embed texts.
These features are:
  - Continuous (not binary)
  - Partially informative (similar categories overlap in embedding space)
  - Noisy (same category texts vary in embedding)
→ LinUCB struggles to converge → warm-start multiplication

Usage on SSH:
  python embed_features_v8.py              # default: all-MiniLM-L6-v2
  python embed_features_v8.py --model all-mpnet-base-v2   # 768d, better
  python embed_features_v8.py --d 200      # project to 200d after embedding
"""

import numpy as np
import gzip
import pickle
import argparse
from numpy.linalg import norm


def embed_texts(texts, model_name='all-MiniLM-L6-v2', batch_size=256):
    """Embed texts using sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    
    print(f"  Loading {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"  Embedding {len(texts)} texts (batch_size={batch_size})...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                              normalize_embeddings=True)
    return embeddings.astype(np.float32)


def build_arm_features(texts, ys, K, embeddings):
    """Build arm features as mean embedding per category."""
    d = embeddings.shape[1]
    arm_features = np.zeros((K, d), dtype=np.float32)
    for k in range(K):
        mask = ys == k
        if mask.sum() > 0:
            arm_features[k] = embeddings[mask].mean(axis=0)
    # Normalize arms
    norms = norm(arm_features, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    arm_features /= norms
    return arm_features


def compute_diagnostics(doc_features, arm_features, ys, K):
    """Compute α̂ (coherence) and r̂ (effective rank)."""
    # Coherence: average |<a_i, a_j>| for i≠j
    n_pairs = min(5000, K * (K - 1) // 2)
    coherences = []
    for _ in range(n_pairs):
        i, j = np.random.choice(K, 2, replace=False)
        coherences.append(abs(float(arm_features[i] @ arm_features[j])))
    alpha_hat = np.mean(coherences)
    
    # Effective rank of φ(x,a) = x ⊙ a covariance
    d = doc_features.shape[1]
    sample_size = min(5000, len(ys))
    idx = np.random.choice(len(ys), sample_size, replace=False)
    phi_samples = doc_features[idx] * arm_features[ys[idx]]
    cov = (phi_samples.T @ phi_samples) / sample_size
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-10]
    r_hat = (np.sum(eigvals) ** 2) / np.sum(eigvals ** 2)
    
    return {'alpha_hat': float(alpha_hat), 'r_hat': float(r_hat), 'd': d, 'K': K}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='synthetic_text_dataset_v8.pkl.gz')
    parser.add_argument('--output', type=str, default='synthetic_text_dataset_v8_emb.pkl.gz')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                        help='Sentence-transformer model name')
    parser.add_argument('--d', type=int, default=0,
                        help='Project to d dims (0 = use native embedding dim)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # Load existing dataset
    print(f"Loading {args.input}...")
    with gzip.open(args.input, 'rb') as f:
        dataset = pickle.load(f)
    
    texts = dataset['texts']
    ys = dataset['ys']
    K = len(dataset['category_names'])
    N = len(texts)
    print(f"  {N} docs, {K} categories")
    
    # Embed texts
    print(f"\nEmbedding with {args.model}...")
    embeddings = embed_texts(texts, model_name=args.model)
    embed_dim = embeddings.shape[1]
    print(f"  Embedding dim: {embed_dim}")
    
    # Optional projection to lower dimension
    if args.d > 0 and args.d < embed_dim:
        print(f"\nProjecting {embed_dim} → {args.d}...")
        W = np.random.randn(embed_dim, args.d).astype(np.float32) / np.sqrt(args.d)
        embeddings = embeddings @ W
        # Re-normalize
        norms = norm(embeddings, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        embeddings = embeddings / norms
        embed_dim = args.d
    
    # Build arm features (mean embedding per category)
    print("\nBuilding arm features...")
    arm_features = build_arm_features(texts, ys, K, embeddings)
    
    # Diagnostics
    print("\nDiagnostics...")
    diagnostics = compute_diagnostics(embeddings, arm_features, ys, K)
    print(f"  α̂ (coherence): {diagnostics['alpha_hat']:.4f}")
    print(f"  r̂ (eff. rank): {diagnostics['r_hat']:.1f}")
    print(f"  d: {embed_dim}")
    
    # Build output dataset
    out_dataset = {
        'doc_features': embeddings,
        'arm_features': arm_features,
        'ys': ys,
        'texts': texts,
        'category_names': dataset['category_names'],
        'diagnostics': diagnostics,
        'n_per_cat': dataset.get('n_per_cat', N // K),
        'embedding_model': args.model,
    }
    
    # Save
    print(f"\nSaving to {args.output}...")
    with gzip.open(args.output, 'wb') as f:
        pickle.dump(out_dataset, f)
    
    print(f"\nDone! {N} docs, K={K}, d={embed_dim}")
    print(f"Diagnostics: α̂={diagnostics['alpha_hat']:.4f}, r̂={diagnostics['r_hat']:.1f}")


if __name__ == '__main__':
    main()
