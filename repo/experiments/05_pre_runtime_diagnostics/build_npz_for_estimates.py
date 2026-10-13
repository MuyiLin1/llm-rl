"""Build the .npz inputs pre_runtime_estimates.py needs, from the existing AmazonCat artifacts.

X[s,a,:] = phi(x_s, a) = context_s * Z[a]   (same joint features as experiment_algorithm1.py)
y[s]     = index (in axis 1) of the single true label kept for context s
expert[s]= index of the LLM's proposal for context s

Candidate set (A per context, << 13,330 -- the full set will not fit in memory):
  * one true label (the LLM's, if the LLM is correct on s, else the first true label)
  * the LLM's proposal
  * hard negatives: highest |<x_s, z_a>| entities, excluding ALL true labels of s
Dropping the other true labels makes the multi-label reward 1{a in ys} coincide with the
single-label reward 1{a == y} that pre_runtime_estimates.py assumes.
"""
import argparse, gzip, os, pickle, sys
import numpy as np, torch

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, *a, **k): pass
    def __len__(self): return self.Xs.shape[0]
    def __getitem__(self, i): return i
sys.modules['__main__'].MyDataset = MyDataset

ap = argparse.ArgumentParser()
ap.add_argument('--family', default='v1')
ap.add_argument('--llm_types', nargs='+', default=['small', 'base', 'large'])
ap.add_argument('--n_held', type=int, default=1000)
ap.add_argument('--A', type=int, default=128)
ap.add_argument('--seed', type=int, default=42)
args = ap.parse_args()

path = 'mydataset.0.pickle.gz' if args.family == 'v1' else 'mydataset.0.v3.pickle.gz'
print(f"loading {path} ...", flush=True)
with gzip.open(path, 'rb') as f:
    ds = pickle.load(f)

zd = {}
for k in ds.labelfeats:
    i, e = ds.labelfeats[k]; zd[i] = np.asarray(e, dtype=np.float32)
Z = np.stack([zd[i] for i in sorted(zd)])                      # (13330, d)
Xs = ds.Xs.numpy() if isinstance(ds.Xs, torch.Tensor) else ds.Xs
d = Xs.shape[1] // 2
N = Xs.shape[0]
ys = ds.ys
print(f"N={N} d={d} n_entities={Z.shape[0]}", flush=True)

# The v3 (BGE/Qwen) caches were only run over the first 50,000 samples
# (subset_indices == arange(50000)), so held-out contexts must come from there.
pool = N
if args.family == 'v3':
    pool = 50000
    print(f"family v3: sampling held-out contexts from the first {pool} "
          f"(the extent of the partial LLM caches)", flush=True)
rng = np.random.default_rng(args.seed)
idx = rng.choice(pool, size=args.n_held, replace=False)
C = Xs[idx, :d].astype(np.float32)                             # (n, d) contexts
del Xs, ds

sims = C @ Z.T                                                 # (n, 13330)
rank = np.argsort(-np.abs(sims), axis=1)                       # hard negatives first
del sims

for llm in args.llm_types:
    cp = f'llm_cache_amazon_{args.family}_{llm}.pkl.gz'
    if not os.path.exists(cp):
        cp = f'llm_cache_amazon_{args.family}_{llm}_partial50000.pkl.gz'
    print(f"loading {cp} ...", flush=True)
    with gzip.open(cp, 'rb') as f:
        cache = pickle.load(f)
    lab = np.asarray(cache['labels']); del cache

    n, A = args.n_held, args.A
    X = np.zeros((n, A, d), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64); ex = np.zeros(n, dtype=np.int64)
    n_llm_ok = 0
    for s in range(n):
        true = set(int(t) for t in ys[idx[s]])
        e = int(lab[idx[s]])
        if e in true:
            keep, n_llm_ok = e, n_llm_ok + 1
        else:
            keep = int(sorted(true)[0])
        cand = [keep] + ([e] if e != keep else [])
        for a in rank[s]:
            if len(cand) >= A: break
            a = int(a)
            if a in true or a == e: continue
            cand.append(a)
        cand = np.array(cand[:A])
        X[s] = C[s][None, :] * Z[cand]
        y[s] = 0
        ex[s] = 0 if e == keep else 1
    out = f'estimates_{args.family}_{llm}.npz'
    np.savez_compressed(out, X=X, y=y, expert=ex)
    print(f"wrote {out}: X{X.shape} held-out LLM acc {n_llm_ok/n:.3f}", flush=True)
