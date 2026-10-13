"""Part C row for tab:inject50k: '+75 on 1471, oracle reward' (replay/inject50k_oracle.py).
Same statistics as replay/paper/stats.py (paired t on Router accuracy, exact McNemar on settling on 1471,
multiplier = mean acc / mean cold acc). Also checks that each oracle run is identical to the existing
'+75 on 1471' run up to its first routed changed answer, and reports the oracle rewards that replaced 1."""
import os, sys, json, numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); RP = os.path.join(os.path.dirname(HERE), 'replay')
sys.path.insert(0, RP); sys.path.insert(0, os.path.dirname(HERE))
from math import comb
def settle(a, cat=1471):   # = replay/paper/stats.py (not imported: it runs and rewrites stats.json on import)
    tail = a[int(0.8 * len(a)):]; return float((tail == cat).mean()) > 0.20
def mcnemar(x, y):         # = replay/paper/stats.py
    b = sum(1 for i, j in zip(x, y) if i and not j); c = sum(1 for i, j in zip(x, y) if j and not i)
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
OUT = os.path.join(RP, 'out'); S = range(1, 11)
ld = lambda n: np.load(os.path.join(OUT, n))
cold = np.array([ld(f'small_s{s}.npz')['r_cold'].mean() for s in S])
cond = {'none': [ld(f'small_s{s}.npz') for s in S]}
for c in ('seen', 'hidden', 'rand', 'oracle'): cond[c] = [ld(f'J50k_{c}_s{s}.npz') for s in S]
key = lambda c: 'full' if c == 'none' else 'router'
acc = {c: np.array([o[f'r_{key(c)}'].mean() for o in z]) for c, z in cond.items()}
st = {c: [settle(o[f'a_{key(c)}']) for o in z] for c, z in cond.items()}
R = {}
for c in cond:
    a = acc[c]
    R[c] = dict(acc=a.mean()*100, sd=a.std(ddof=1)*100, mult=a.mean()/cold.mean(),
                mult_sd_perseed=(a/cold).std(ddof=1), rr=(1-cold.mean())/(1-a.mean()), settle=int(sum(st[c])))
    if c != 'none':
        R[c]['routed_changed'] = float(np.mean([int(o['injected_routed']) for o in cond[c]]))
        for ref in ('none', 'seen'):
            if ref == c: continue
            t = stats.ttest_rel(a, acc[ref]); R[c][f'vs_{ref}'] = dict(t=t.statistic, p=t.pvalue, mcnemar=mcnemar(st[c], st[ref]))
# sanity: oracle == seen until the first routed changed round; oracle rewards fed on those rounds
from sim import Data
from inject50k import build, SLIM, N_ON
D = Data(SLIM); seen, rand, chg = build(D); th = np.load(os.path.join(RP, 'theta_or_v1.npy'))
chk, fed = [], []
for s, o, sn in zip(S, cond['oracle'], cond['seen']):
    np.random.seed(s); order = np.random.permutation(N_ON)
    t1 = np.where(o['oracle'])[0]; first = int(t1[0]) if t1.size else len(order)
    chk.append(dict(seed=s, first_changed_routed=first if t1.size else None,
                    identical_before=bool(np.array_equal(o['a_router'][:first+1], sn['a_router'][:first+1])),
                    identical_whole_run=bool(np.array_equal(o['a_router'], sn['a_router']))))
    fed += [float((D.C[order[t]].astype(np.float64) * D.Z[1471]) @ th) for t in t1]
R['checks'] = chk; R['oracle_rewards_fed'] = dict(n=len(fed), mean=float(np.mean(fed)), min=float(np.min(fed)),
                                                  max=float(np.max(fed)), values=fed)
R['cold'] = dict(acc=cold.mean()*100)
json.dump(R, open(os.path.join(HERE, 'out', 'partC.json'), 'w'), indent=1, default=float)
for c in ('none', 'seen', 'hidden', 'rand', 'oracle'):
    r = R[c]; v = {k: r.get(k) for k in ('vs_none', 'vs_seen')}
    print(f"{c:7s} acc {r['acc']:.2f}±{r['sd']:.2f}  mult {r['mult']:.2f}x (rr {r['rr']:.2f})  settle {r['settle']}/10  "
          f"routed {r.get('routed_changed')}  " + '  '.join(f"{k}: p={x['p']:.3f} t={x['t']:.2f} mcn={x['mcnemar']:.3f}" for k, x in v.items() if x))
print(chk); print('oracle rewards fed:', R['oracle_rewards_fed']['n'], 'mean', R['oracle_rewards_fed']['mean'])
