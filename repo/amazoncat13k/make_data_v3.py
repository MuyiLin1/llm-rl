"""
make_data_v3.py

Re-embed AmazonCat-13K using BAAI/bge-large-en-v1.5 (1024-dim)
instead of bert-base-nli-mean-tokens (768-dim).

This is needed for Qwen caches and Qwen-based bandit experiments,
since Qwen outputs are embedded with BGE and compared against
BGE entity embeddings.

Input:  mydataset.0.pickle.gz  (existing dataset with raw text fields)
Output: mydataset.0.v3.pickle.gz  (same structure, BGE embeddings)

Usage (run from amazoncat13k/):
    python -u make_data_v3.py

On a GPU this should take ~1-2 hours for 1.18M samples.
On CPU, expect ~6-8 hours.
"""

import torch
import numpy as np
import gzip
import pickle
import sys
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

ST_MODEL = 'BAAI/bge-large-en-v1.5'  # 1024-dim


# ================================================================
# Load existing v1 dataset (to get raw text + labels)
# ================================================================
class MyDataset(torch.utils.data.Dataset):
    """Stub for unpickling."""
    def __init__(self, *args, **kwargs): pass
    def __len__(self): return self.Xs.shape[0]
    def __getitem__(self, index):
        return self.Xs[index], self.text_titles[index], self.text_contents[index], self.ys[index]

sys.modules['__main__'].MyDataset = MyDataset


def load_v1():
    fname = 'mydataset.0.pickle.gz'
    print(f'Loading v1 dataset: {fname}')
    with gzip.open(fname, 'rb') as f:
        return pickle.load(f)


# ================================================================
# Re-embed entities with BGE
# ================================================================
def reembed_entities(v1_dataset, model):
    """Re-embed all entity label names with BGE."""
    # Extract label names sorted by ordinal
    label_names = {}
    for name, (ordinal, _old_emb) in v1_dataset.labelfeats.items():
        label_names[ordinal] = name

    sorted_names = [label_names[i] for i in range(len(label_names))]
    print(f"Re-embedding {len(sorted_names)} entity labels with {ST_MODEL}...")

    embeddings = model.encode(sorted_names, batch_size=256, show_progress_bar=True)

    # Rebuild labelfeats dict: {name: (ordinal, bge_embedding)}
    new_labelfeats = {}
    for name, (ordinal, _old_emb) in v1_dataset.labelfeats.items():
        new_labelfeats[name] = (ordinal, embeddings[ordinal])

    return new_labelfeats


# ================================================================
# Re-embed contexts (titles + contents) with BGE
# ================================================================
def reembed_contexts(v1_dataset, model, batch_size=64):
    """Re-embed all title and content texts with BGE."""
    N = len(v1_dataset)
    print(f"Re-embedding {N} titles + contents with {ST_MODEL}...")

    # Encode titles
    print("  Encoding titles...")
    title_embs = model.encode(
        v1_dataset.text_titles, batch_size=batch_size, show_progress_bar=True
    )  # (N, 1024)

    # Encode contents
    print("  Encoding contents...")
    content_embs = model.encode(
        v1_dataset.text_contents, batch_size=batch_size, show_progress_bar=True
    )  # (N, 1024)

    # Concatenate: (N, 2048)
    Xs = np.concatenate([title_embs, content_embs], axis=1)
    return torch.FloatTensor(Xs)


# ================================================================
# Build v3 dataset
# ================================================================
def main():
    print(f"Using SentenceTransformer: {ST_MODEL}")
    model = SentenceTransformer(ST_MODEL)

    # Load existing dataset
    v1 = load_v1()
    print(f"v1 dataset: {len(v1)} samples, Xs shape: {v1.Xs.shape}")
    print(f"  Entities: {len(v1.labelfeats)}")

    # Re-embed entities
    new_labelfeats = reembed_entities(v1, model)

    # Re-embed contexts
    new_Xs = reembed_contexts(v1, model)
    print(f"New Xs shape: {new_Xs.shape}")

    # Build new dataset object (reuse text fields and ys from v1)
    v3 = MyDataset()
    v3.Xs = new_Xs
    v3.ys = v1.ys
    v3.text_titles = v1.text_titles
    v3.text_contents = v1.text_contents
    v3.labelfeats = new_labelfeats

    print(f"v3 dataset: {len(v3)} samples, Xs shape: {v3.Xs.shape}")
    print(f"  Entity embedding dim: {list(v3.labelfeats.values())[0][1].shape}")

    # Save
    fname = 'mydataset.0.v3.pickle.gz'
    print(f"Saving to {fname}...")
    with gzip.open(fname, 'wb') as f:
        pickle.dump(v3, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Done! Saved {fname}")


if __name__ == '__main__':
    main()
