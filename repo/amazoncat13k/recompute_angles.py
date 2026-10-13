"""Recompute the subspace-mismatch angle SEPARATELY FOR EACH EXPERT.

Table `tab:diagnostics` in v5 reports 85.1 deg for all three AmazonCat Flan-T5
sizes and 83.1 deg for all three Qwen sizes, flagged `TODO(check)` because the
angle depends on the expert's own proposals and so should differ across sizes
(as it does on WikiLinks). This recomputes it per expert in a single pass over
the dataset, using exactly the protocol of compute_diagnostics.py:

  r_hat = tr(Sigma_ctx)/||Sigma_ctx||_op          on n_samples contexts
  U_llm = top-r right singular vectors of  (x_s * z_{a_s^Exp}) second moment
  U_ctx = top-r right singular vectors of  the context second moment
  angle = mean principal angle between the two subspaces, in degrees

Usage (AmazonCat):
  python recompute_angles.py --family v1 --llm_types small base large
  python recompute_angles.py --family v3 --llm_types small base large
"""
import argparse, gzip, os, pickle, sys
import numpy as np
import torch
from scipy.linalg import svd, subspace_angles


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, *a, **k): pass
    def __len__(self): return self.Xs.shape[0]
    def __getitem__(self, i): return i
sys.modules['__main__'].MyDataset = MyDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', default='v1', choices=['v1', 'v3'])
    ap.add_argument('--llm_types', nargs='+', default=['small', 'base', 'large'])
    ap.add_argument('--n_samples', type=int, default=5000,
                    help='matches compute_diagnostics.py')
    ap.add_argument('--T', type=int, default=50000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    path = ('mydataset.0.pickle.gz' if args.family == 'v1'
            else 'mydataset.0.v3.pickle.gz')
    print(f"loading {path} ...", flush=True)
    with gzip.open(path, 'rb') as f:
        ds = pickle.load(f)

    zd = {}
    for k in ds.labelfeats:
        i, e = ds.labelfeats[k]
        zd[i] = np.asarray(e, dtype=np.float32)
    Z = np.stack([zd[i] for i in sorted(zd)])
    Xs = ds.Xs.numpy() if isinstance(ds.Xs, torch.Tensor) else ds.Xs
    d = Xs.shape[1] // 2
    N = Xs.shape[0]
    del ds

    # v3 caches only cover the first 50,000 samples
    pool = 50000 if args.family == 'v3' else N
    rng = np.random.RandomState(args.seed)
    idx = rng.choice(pool, size=min(args.n_samples, pool), replace=False)
    C = np.nan_to_num(Xs[idx, :d].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    del Xs
    n = len(idx)

    cov_ctx = (C.T @ C) / n
    ev = np.linalg.eigvalsh(cov_ctx)[::-1]
    r_hat = float(ev.sum() / ev[0])
    r = max(1, min(int(round(r_hat)), d - 1))
    print(f"n={n} d={d} r_hat={r_hat:.2f} -> rank r={r}  "
          f"(r_hat*log T/sqrt T = {r_hat*np.log(args.T)/np.sqrt(args.T):.3f})",
          flush=True)
    _, _, Vh_ctx = svd(cov_ctx)
    U_ctx = Vh_ctx[:r, :].T

    print(f"\n{'expert':>8} {'angle (deg)':>12} {'held-out acc':>13}")
    for llm in args.llm_types:
        cp = f'llm_cache_amazon_{args.family}_{llm}.pkl.gz'
        if not os.path.exists(cp):
            cp = f'llm_cache_amazon_{args.family}_{llm}_partial50000.pkl.gz'
        with gzip.open(cp, 'rb') as f:
            cache = pickle.load(f)
        lab = np.asarray(cache['labels'])
        del cache
        F = C * Z[lab[idx]]                       # (n, d) expert joint features
        cov_llm = (F.T @ F) / n
        _, _, Vh_llm = svd(cov_llm)
        U_llm = Vh_llm[:r, :].T
        ang = float(np.mean(np.degrees(subspace_angles(U_llm, U_ctx))))
        print(f"{llm:>8} {ang:>12.1f}", flush=True)


if __name__ == '__main__':
    main()
