"""Assemble dataset from cached .npy files and save as uncompressed pickle.
Then gzip from command line: gzip mydataset.0.pickle"""
import gzip
import json
import pickle
import numpy as np
import torch
import gc

# Load labels
print("Loading labels...")
with open('AmazonCat-13K/Yf.txt', 'r', encoding='ISO-8859-1') as f:
    labels = [line.strip() for line in f]
print(f"  {len(labels)} labels")

# Load training data (need titles, contents, target_ind for ys)
print("Loading training data...")
trn_data = []
with gzip.open('AmazonCat-13K/trn.json.gz', 'r') as f:
    for line in f:
        trn_data.append(json.loads(line.decode('utf-8')))

titles = [d['title'] for d in trn_data]
contents = [d['content'] for d in trn_data]
ids = [d['target_ind'] for d in trn_data]
del trn_data
gc.collect()
print(f"  {len(titles)} examples loaded")

# Build label mapping (threshold=0, all labels)
print("Building label mapping...")
label_freq = {}
for target_ids in ids:
    for lid in target_ids:
        label_freq[lid] = label_freq.get(lid, 0) + 1
valid_labels = set(label_freq.keys())
label_to_idx = {lid: i for i, lid in enumerate(sorted(valid_labels))}
print(f"  {len(label_to_idx)} labels")

# Encode label embeddings (need the model just for labels - small)
print("Loading model for label embeddings only...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')
label_texts = [labels[lid] for lid in sorted(valid_labels)]
label_embeddings = model.encode(label_texts, batch_size=512, show_progress_bar=True)
print(f"  Label embeddings shape: {label_embeddings.shape}")

labelfeats = {}
for i, lid in enumerate(sorted(valid_labels)):
    labelfeats[labels[lid]] = (i, label_embeddings[i])
del model, label_embeddings
gc.collect()

# Build ys
print("Building ys...")
ys = []
for target_ids in ids:
    y = [label_to_idx[lid] for lid in target_ids if lid in label_to_idx]
    ys.append(y)
del ids
gc.collect()

# Load embeddings with memory mapping and concatenate
print("Loading embeddings (memory-mapped)...")
title_emb = np.load('title_embeddings.npy', mmap_mode='r')
content_emb = np.load('content_embeddings.npy', mmap_mode='r')
print(f"  Title: {title_emb.shape}, Content: {content_emb.shape}")

print("Concatenating to Xs...")
Xs = torch.from_numpy(np.concatenate([title_emb, content_emb], axis=1))
print(f"  Xs shape: {Xs.shape}")
del title_emb, content_emb
gc.collect()

# Create dataset
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
dataset.text_titles = titles
dataset.text_contents = contents
dataset.labelfeats = labelfeats

# Save as UNCOMPRESSED pickle (gzip from CLI afterwards)
print("Saving to mydataset.0.pickle (uncompressed)...")
with open('mydataset.0.pickle', 'wb') as handle:
    pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Done! Now run: gzip mydataset.0.pickle")
print(f"Dataset has {len(dataset)} examples, feature dim={Xs.shape[1]}")
