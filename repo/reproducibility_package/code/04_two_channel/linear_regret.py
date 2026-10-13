"""Analyze the linearized-reward runs (Tests 2 / 2b) with the metric that makes
sense under linear rewards: pseudo-regret against the oracle's best action,
    reg_t = max_a <theta_or, phi(x_t,a)> - <theta_or, phi(x_t, A_t)>,
computed post hoc from the logged actions. Multiplier := R_cold / R_router.
Prediction (Prop response-sum): under linear rewards the three experts give
the same multiplier, even with the response sum switched on.
"""
import argparse, json, os, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='linear'); ap.add_argument('--suffix', default='_pfull_linear')
ap.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 11)))
ap.add_argument('--logdir', default='study'); ap.add_argument('--online_T', type=int, default=50000)
a = ap.parse_args()
z = np.load('amazon_v1_slim.npz'); C = z['contexts']; Z = z['Z']
th = np.load('theta_or_v1.npy').astype(np.float32)
Zt = Z.astype(np.float32); Ct = (C[:a.online_T] * th[None, :]).astype(np.float32)   # (T, d)
best = np.empty(a.online_T, dtype=np.float32)
for i in range(0, a.online_T, 2000):                       # max over all 13,330 actions
    best[i:i+2000] = (Ct[i:i+2000] @ Zt.T).max(axis=1)
out = {}
print(f"{'expert':7} {'seed':>4} {'R_cold':>10} {'R_router':>10} {'mult':>7} {'routed%':>8}")
for llm in ('small', 'base', 'large'):
    for s in a.seeds:
        p = f"{a.logdir}/perround_algo1_{a.tag}_v1_{llm}_c100.0{a.suffix}_s{s}.npz"
        if not os.path.exists(p): continue
        L = np.load(p); n = int(L['step'])
        np.random.seed(s); perm = np.random.permutation(a.online_T)
        got_r = np.einsum('ij,ij->i', Ct[perm[:n]], Zt[L['a_router'][:n]])
        got_c = np.einsum('ij,ij->i', Ct[perm[:n]], Zt[L['a_cold'][:n]])
        Rr = float((best[perm[:n]] - got_r).sum()); Rc = float((best[perm[:n]] - got_c).sum())
        out[f'{llm}_s{s}'] = dict(R_cold=Rc, R_router=Rr, mult=Rc / Rr, routed=float(L['routed'][:n].mean()))
        print(f"{llm:7} {s:>4} {Rc:>10.3f} {Rr:>10.3f} {Rc/Rr:>7.3f} {100*out[f'{llm}_s{s}']['routed']:>7.2f}%")
json.dump(out, open(f'linear_regret_{a.tag}.json', 'w'), indent=2)
print("\nmeans:")
for llm in ('small', 'base', 'large'):
    v = [r['mult'] for k, r in out.items() if k.startswith(llm + '_')]
    if v: print(f"  {llm:7} multiplier {np.mean(v):.3f} +- {np.std(v, ddof=1) if len(v) > 1 else 0:.3f}   (n={len(v)})")
