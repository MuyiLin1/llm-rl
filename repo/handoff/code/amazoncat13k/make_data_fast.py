"""Fast version of make_data.py — batch encodes all texts at once to maximize CPU utilization."""
import gzip
import json
import pickle
import numpy as np
import torch
from tqdm import tqdm

def load_gz_json(file_path):
    data = []
    with gzip.open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.decode('utf-8')))
    return data

# Set PyTorch to use all cores
torch.set_num_threads(torch.get_num_threads())
print(f"PyTorch using {torch.get_num_threads()} threads")

# Load labels
print("Loading labels...")
with open('AmazonCat-13K/Yf.txt', 'r', encoding='ISO-8859-1') as f:
    labels = [line.strip() for line in f]
print(f"  {len(labels)} labels")

# Load training data
print("Loading training data...")
import pandas as pd
trn_data = load_gz_json('AmazonCat-13K/trn.json.gz')
trn_df = pd.DataFrame(trn_data)
del trn_data  # free memory

titles = trn_df['title'].tolist()
contents = trn_df['content'].tolist()
ids = trn_df['target_ind'].tolist()
print(f"  {len(titles)} examples loaded")

# threshold=0 means use all labels that appear at least once
threshold = 0

# Build category set (labels with freq >= threshold)
print("Building category set...")
label_freq = {}
for target_ids in ids:
    for lid in target_ids:
        label_freq[lid] = label_freq.get(lid, 0) + 1

if threshold > 0:
    valid_labels = {lid for lid, freq in label_freq.items() if freq >= threshold}
else:
    valid_labels = set(label_freq.keys())
print(f"  {len(valid_labels)} categories with freq >= {threshold}")

# Build label-to-index mapping
label_to_idx = {}
for lid in sorted(valid_labels):
    label_to_idx[lid] = len(label_to_idx)
print(f"  {len(label_to_idx)} label indices")

# Load model
print("Loading sentence-transformers model...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Encode all category labels
print("Encoding category labels...")
label_texts = [labels[lid] for lid in sorted(valid_labels)]
label_embeddings = model.encode(label_texts, batch_size=512, show_progress_bar=True)
print(f"  Label embeddings shape: {label_embeddings.shape}")

# Build labelfeats dict matching original format: {label_text: (idx, embedding)}
labelfeats = {}
for i, lid in enumerate(sorted(valid_labels)):
    labelfeats[labels[lid]] = (i, label_embeddings[i])

# Encode in two phases, saving to disk to avoid OOM
import os
import gc

title_emb_path = 'title_embeddings.npy'
content_emb_path = 'content_embeddings.npy'

# Phase 1: Encode titles (skip if cached)
if not os.path.exists(title_emb_path):
    print("Encoding all titles...")
    title_embeddings = model.encode(titles, batch_size=256, show_progress_bar=True)
    np.save(title_emb_path, title_embeddings)
    print(f"  Saved to {title_emb_path}, shape: {title_embeddings.shape}")
    del title_embeddings
    gc.collect()
else:
    print(f"Titles already cached at {title_emb_path}")

# Phase 2: Encode contents (skip if cached)
if not os.path.exists(content_emb_path):
    print("Encoding all contents...")
    content_embeddings = model.encode(contents, batch_size=256, show_progress_bar=True)
    np.save(content_emb_path, content_embeddings)
    print(f"  Saved to {content_emb_path}, shape: {content_embeddings.shape}")
    del content_embeddings
    gc.collect()
else:
    print(f"Contents already cached at {content_emb_path}")

# Free model from memory before loading embeddings
del model
gc.collect()

# Now load embeddings and assemble dataset
print("Loading title embeddings from disk...")
title_embeddings = np.load(title_emb_path)
print(f"  Title embeddings shape: {title_embeddings.shape}")

print("Loading content embeddings from disk...")
content_embeddings = np.load(content_emb_path)
print(f"  Content embeddings shape: {content_embeddings.shape}")

# Assemble dataset
print("Assembling dataset...")
Xs = torch.from_numpy(np.concatenate([title_embeddings, content_embeddings], axis=1))
print(f"  Final Xs shape: {Xs.shape}")

# Free raw embeddings
del title_embeddings, content_embeddings
gc.collect()

ys = []
for i in range(len(titles)):
    target_ids = ids[i]
    y = [label_to_idx[lid] for lid in target_ids if lid in label_to_idx]
    ys.append(y)

text_titles = titles
text_contents = contents

# Create dataset object matching original MyDataset structure
class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        pass
    def __len__(self):
        return self.Xs.shape[0]
    def __getitem__(self, index):
        return self.Xs[index], self.text_titles[index], self.text_contents[index], self.ys[index]

dataset = MyDataset()
dataset.Xs = Xs
dataset.ys = ys
dataset.text_titles = text_titles
dataset.text_contents = text_contents
dataset.labelfeats = labelfeats

# Save
print("Saving to mydataset.0.pickle.gz...")
with gzip.open('mydataset.0.pickle.gz', 'wb') as handle:
    pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Done! Dataset has {len(dataset)} examples, feature dim={Xs.shape[1]}")
