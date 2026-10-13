"""Condense a dataset pickle into a small .npz the runner can load in seconds.

Handles both datasets, so every configuration can go through the SAME runner
(experiment_algorithm1.py) rather than two programs with subtly different
semantics. WikiLinks differs only in having 311 actions instead of 13,330 and a
single correct label instead of a set; both are stored in the same ragged form.

The online experiments use only the first `--n_keep` samples (T = 50,000, plus a
pool outside the stream for --router offline), so the full 1.18M x 1536 tensor is
never needed at run time. Loading the pickle costs ~11 GB of RAM per process,
which is what stops the nine Phase 4 seed runs from running in parallel; this
file costs ~250 MB per process instead.

  python make_slim_npz.py --family v1 --n_keep 60000
  python experiment_algorithm1.py --slim amazon_v1_slim.npz --llm_type large ...

Contents: contexts (n_keep, d) float32, ragged true labels as flat+offsets,
entity embeddings Z, and the cached LLM proposals for each size.
"""
import argparse, gzip, os, pickle, sys
import numpy as np
import torch


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, *a, **k): pass
    def __len__(self): return self.Xs.shape[0]
    def __getitem__(self, i): return i
sys.modules['__main__'].MyDataset = MyDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='amazon', choices=['amazon', 'wiki'])
    ap.add_argument('--family', default='v1', choices=['v1', 'v3'])
    ap.add_argument('--llm_types', nargs='+', default=['small', 'base', 'large'])
    ap.add_argument('--n_keep', type=int, default=0,
                    help='samples to keep (0 = all; AmazonCat uses the online '
                         'horizon plus an offline pool, WikiLinks uses all 622k)')
    args = ap.parse_args()

    if args.dataset == 'amazon':
        path = ('mydataset.0.pickle.gz' if args.family == 'v1'
                else 'mydataset.0.v3.pickle.gz')
        cache_fmt = 'llm_cache_amazon_{fam}_{llm}.pkl.gz'
        cache_alt = 'llm_cache_amazon_{fam}_{llm}_partial50000.pkl.gz'
    else:
        path = ('mydataset.2000.pickle.gz' if args.family == 'v1'
                else 'mydataset.2000.v3.pickle.gz')
        cache_fmt = 'llm_cache_{fam}_{llm}.pkl.gz'
        cache_alt = cache_fmt
    print(f"loading {path} ...", flush=True)
    with gzip.open(path, 'rb') as f:
        ds = pickle.load(f)

    zd = {}
    for k in ds.labelfeats:
        i, e = ds.labelfeats[k]
        zd[i] = np.asarray(e, dtype=np.float32)
    Z = np.nan_to_num(np.stack([zd[i] for i in sorted(zd)]),
                      nan=0.0, posinf=0.0, neginf=0.0)

    Xs = ds.Xs.numpy() if isinstance(ds.Xs, torch.Tensor) else ds.Xs
    d = Xs.shape[1] // 2
    n = Xs.shape[0] if args.n_keep in (0, None) else min(args.n_keep, Xs.shape[0])
    contexts = np.nan_to_num(Xs[:n, :d].astype(np.float32),
                             nan=0.0, posinf=0.0, neginf=0.0)
    del Xs

    ys = ds.ys[:n]
    if hasattr(ys, 'numpy'):
        ys = ys.numpy()
    if np.ndim(ys[0]) == 0:            # WikiLinks: one correct label per context
        flat = np.asarray([int(y) for y in ys], dtype=np.int32)
        offs = np.arange(n + 1, dtype=np.int64)
    else:                              # AmazonCat: a set of correct labels
        flat = np.concatenate([np.asarray(y, dtype=np.int32) for y in ys])
        offs = np.cumsum([0] + [len(y) for y in ys]).astype(np.int64)
    del ds

    payload = dict(contexts=contexts, Z=Z, y_flat=flat, y_offsets=offs, n_keep=n)
    for llm in args.llm_types:
        cp = cache_fmt.format(fam=args.family, llm=llm)
        if not os.path.exists(cp):
            cp = cache_alt.format(fam=args.family, llm=llm)
        print(f"loading {cp} ...", flush=True)
        with gzip.open(cp, 'rb') as f:
            cache = pickle.load(f)
        lab = np.asarray(cache['labels'])
        del cache
        if len(lab) < n:      # v3 caches only cover the first 50,000
            lab = np.concatenate([lab, -np.ones(n - len(lab), dtype=lab.dtype)])
        payload[f'llm_{llm}'] = lab[:n].astype(np.int64)

    out = f'{args.dataset}_{args.family}_slim.npz'
    np.savez(out, **payload)
    print(f"wrote {out}: contexts{contexts.shape} Z{Z.shape} "
          f"({os.path.getsize(out)/1e6:.0f} MB)", flush=True)


if __name__ == '__main__':
    main()
