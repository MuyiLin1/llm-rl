"""Build the .npz inputs pre_runtime_estimates.py needs, for OneShotWikiLinks.

Simpler than the AmazonCat builder: WikiLinks has only 311 actions, so the full
action set fits and no hard-negative candidate subsampling is needed. Labels are
single (not multi-label), so the action index IS the label index.

  X[s,a,:] = phi(x_s, a) = context_s * Z[a]     (same joint features as the runner)
  y[s]     = true entity id for context s
  expert[s]= the cached LLM proposal for context s

  python build_npz_for_estimates.py --family v1 --llm_types small base large --n_held 500
  python build_npz_for_estimates.py --family v3 --llm_types small base large --n_held 500

Memory: n_held x 311 x d floats. At n_held=500, d=768 that is 478 MB (float32);
pre_runtime_estimates.py upcasts to float64 internally, so keep n_held modest.
"""
import argparse, gzip, pickle, sys
import numpy as np
import torch


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, *a, **k): pass
    def __len__(self): return self.Xs.shape[0]
    def __getitem__(self, i): return i
sys.modules['__main__'].MyDataset = MyDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', default='v1', choices=['v1', 'v2', 'v3'])
    ap.add_argument('--llm_types', nargs='+', default=['small', 'base', 'large'])
    ap.add_argument('--n_held', type=int, default=500)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--threshold', type=int, default=2000)
    args = ap.parse_args()

    path = (f'mydataset.{args.threshold}.pickle.gz' if args.family == 'v1'
            else f'mydataset.{args.threshold}.{args.family}.pickle.gz')
    print(f"loading {path} ...", flush=True)
    with gzip.open(path, 'rb') as f:
        ds = pickle.load(f)

    zd = {}
    for k in ds.labelfeats:
        i, e = ds.labelfeats[k]
        zd[i] = np.asarray(e, dtype=np.float32)
    Z = np.stack([zd[i] for i in sorted(zd)])            # (311, d)
    Xs = ds.Xs.numpy() if isinstance(ds.Xs, torch.Tensor) else ds.Xs
    d = Xs.shape[1] // 2
    N = Xs.shape[0]
    ys = np.asarray(ds.ys)
    A = Z.shape[0]
    print(f"N={N} d={d} actions={A}", flush=True)

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(N, size=args.n_held, replace=False)
    C = Xs[idx, :d].astype(np.float32)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    y = ys[idx].astype(np.int64)
    del Xs, ds

    # X is the same for every expert; only `expert` changes. Build it once.
    X = C[:, None, :] * Z[None, :, :]                    # (n, A, d) float32

    for llm in args.llm_types:
        cp = (f'llm_cache_{args.family}_{llm}.pkl.gz' if args.family != 'v1'
              else f'llm_cache_v1_{llm}.pkl.gz')
        print(f"loading {cp} ...", flush=True)
        with gzip.open(cp, 'rb') as f:
            cache = pickle.load(f)
        ex = np.asarray(cache['labels'])[idx].astype(np.int64)
        del cache
        out = f'estimates_wiki_{args.family}_{llm}.npz'
        np.savez_compressed(out, X=X, y=y, expert=ex)
        print(f"wrote {out}: X{X.shape}  held-out LLM acc {(ex == y).mean():.3f}",
              flush=True)


if __name__ == '__main__':
    main()
