"""Skew experiment: AmazonCat streams in which no category is present on more than a share `cap` of the contexts.
Contexts containing the most over-represented category are dropped at random (fixed seed) until every category is
at or below the cap; only the 50,000 contexts of the online stream are used. Every level is run for the same
horizon TS, short enough for the smallest pool."""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
STREAM = 50000
CAPS = {'bertcap30': None, 'bertcap20': 0.20, 'bertcap10': 0.10, 'bertcap05': 0.05}   # None = natural stream (29.8%)
TS = 14000

def _cap(fl, of, p, seed=0):
    n_all = len(of) - 1; keep = np.zeros(n_all, bool); keep[:STREAM] = True
    rng = np.random.RandomState(seed); lens = np.diff(of); owner = np.repeat(np.arange(n_all), lens)
    while True:
        cnt = np.bincount(fl, weights=np.repeat(keep, lens), minlength=fl.max()+1)
        n = keep.sum(); c = int(cnt.argmax())
        if cnt[c] <= p*n + 1e-9: return np.flatnonzero(keep)
        x = int(np.ceil((cnt[c] - p*n)/(1 - p)))
        idx = np.unique(owner[fl == c]); idx = idx[keep[idx]]
        keep[rng.choice(idx, min(x, len(idx)), replace=False)] = False

def pool(name, slim):
    """Context indices of the capped stream, cached next to this file."""
    p = CAPS[name]
    if p is None: return np.arange(STREAM)
    f = f'{HERE}/skew_pool_{name}.npy'
    if not os.path.exists(f):
        z = np.load(slim); np.save(f, _cap(z['y_flat'], z['y_offsets'], p))
    return np.load(f)

def order(name, slim, seed):
    """Seeded stream order; on the natural stream this is the first TS rounds of the paper's stream."""
    P = pool(name, slim); return P[np.random.RandomState(seed).permutation(len(P))][:TS]
