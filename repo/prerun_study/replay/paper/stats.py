"""Numbers for the paper from the replay (3d) and 50k injection runs. Writes stats.json."""
import os, json, numpy as np
from scipy import stats
from math import comb
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out')
E = ['small', 'base', 'large']; S = range(1, 11); K = ['full', 'nom', 'passive']
def ld(n): return np.load(os.path.join(OUT, n))
def settle(a, cat=1471):
    tail = a[int(0.8 * len(a)):]; return float((tail == cat).mean()) > 0.20
def mcnemar(x, y):
    b = sum(1 for i, j in zip(x, y) if i and not j); c = sum(1 for i, j in zip(x, y) if j and not i)
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
cold = np.array([ld(f'small_s{s}.npz')['r_cold'].mean() for s in S])
res = {'cold_mean': cold.mean() * 100, 'cold_std': cold.std(ddof=1) * 100, 'replay': {}}
acc, am, rr, st = {}, {}, {}, {}
for e in E:
    for k in K:
        z = [ld(f'{e}_s{s}.npz') for s in S]
        a = np.array([o[f'r_{k}'].mean() for o in z])
        acc[e, k] = a; am[e, k] = a / cold; rr[e, k] = (1 - cold) / (1 - a)
        st[e, k] = [settle(o[f'a_{k}']) for o in z]
        res['replay'][f'{e}_{k}'] = dict(acc=a.mean() * 100, sd=a.std(ddof=1) * 100,
            mult=a.mean() / cold.mean(), rr=(1 - cold.mean()) / (1 - a.mean()),
            mult_perseed_mean=am[e, k].mean(), mult_se=am[e, k].std(ddof=1) / np.sqrt(10),
            settle=int(sum(st[e, k])), routed=float(np.mean([o['routed'].sum() for o in z])))
    same = all(np.array_equal(ld(f'{e}_s{s}.npz')['routed'], ld(f'{e}_s{s}.npz')['routed']) for s in S)
T = {}
for k in K:
    T[k] = dict(spread=max(am[e, k].mean() for e in E) - min(am[e, k].mean() for e in E),
                large_small=stats.ttest_rel(am['large', k], am['small', k]).statistic,
                base_small=stats.ttest_rel(am['base', k], am['small', k]).statistic,
                large_base=stats.ttest_rel(am['large', k], am['base', k]).statistic,
                large_small_rr=stats.ttest_rel(rr['large', k], rr['small', k]).statistic,
                base_small_rr=stats.ttest_rel(rr['base', k], rr['small', k]).statistic)
res['tests'] = T
res['nom_vs_passive_t'] = {e: stats.ttest_rel(am[e, 'nom'], am[e, 'passive']).statistic for e in E}
res['full_minus_nom'] = {e: dict(drop=(am[e, 'full'] - am[e, 'nom']).mean(),
                                  t=stats.ttest_rel(am[e, 'full'], am[e, 'nom']).statistic) for e in E}
res['full_minus_passive'] = {e: dict(drop=(am[e, 'full'] - am[e, 'passive']).mean(),
                                  t=stats.ttest_rel(am[e, 'full'], am[e, 'passive']).statistic) for e in E}
res['cold_settle'] = int(sum(settle(ld(f'small_s{s}.npz')['a_cold']) for s in S))
# 50k injection
base = np.array([ld(f'small_s{s}.npz')['r_full'].mean() for s in S])
base_st = [settle(ld(f'small_s{s}.npz')['a_full']) for s in S]
inj = {'none': dict(acc=base, st=base_st)}
for c in ['seen', 'hidden', 'rand']:
    z = [ld(f'J50k_{c}_s{s}.npz') for s in S]
    inj[c] = dict(acc=np.array([o['r_router'].mean() for o in z]), st=[settle(o['a_router']) for o in z],
                  routed=np.mean([int(o['injected_routed']) for o in z]),
                  routed_list=[int(o['injected_routed']) for o in z])
R = {}
for c, v in inj.items():
    R[c] = dict(acc=v['acc'].mean() * 100, sd=v['acc'].std(ddof=1) * 100, mult=v['acc'].mean() / cold.mean(),
                rr=(1 - cold.mean()) / (1 - v['acc'].mean()), settle=int(sum(v['st'])), routed=v.get('routed'),
                routed_list=v.get('routed_list'))
    if c != 'none':
        R[c]['p_vs_none'] = stats.ttest_rel(v['acc'], base).pvalue
        R[c]['t_vs_none'] = stats.ttest_rel(v['acc'], base).statistic
        R[c]['mcn_vs_none'] = mcnemar(v['st'], base_st)
R['seen_vs_hidden'] = dict(t=stats.ttest_rel(inj['seen']['acc'], inj['hidden']['acc']).statistic,
                           p=stats.ttest_rel(inj['seen']['acc'], inj['hidden']['acc']).pvalue,
                           mcn=mcnemar(inj['seen']['st'], inj['hidden']['st']))
res['inject50k'] = R
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stats.json'), 'w'), indent=1,
          default=float)
print(json.dumps(res, indent=1, default=lambda x: round(float(x), 4)))
