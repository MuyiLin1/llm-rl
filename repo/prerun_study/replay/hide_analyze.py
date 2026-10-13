"""4d read-out: baseline small (A), +15 on 1471 with rewards (D), +15 on 1471 with rewards hidden (H)."""
import numpy as np, glob, os
from scipy import stats
HERE=os.path.dirname(os.path.abspath(__file__)); R=os.path.dirname(HERE)+'/reh_out'
T=10000
def load(pattern, rkey='r_router', akey='a_router'):
    out={}
    for p in glob.glob(pattern):
        if p.endswith('.tmp.npz'): continue
        s=int(p.rsplit('_s',1)[1].split('.')[0]); o=np.load(p)
        a=o[akey]; r=o[rkey]
        out[s]=dict(acc=r.mean(), lock=np.mean(a[int(.8*T):]==1471)>0.25,
                    hidden=int(o['hidden'].sum()) if 'hidden' in o.files else None)
    return out
cold=load(R+'/A_cold_s*.npz','r_cold','a_cold'); ca=np.mean([v['acc'] for v in cold.values()])
A=load(R+'/A_router_small_s*.npz'); D=load(R+'/D_router_small_plus1471_s*.npz'); H=load(HERE+'/out/H_small_plus1471_hide_s*.npz')
PR=load(R+'/P_router_small_plusrand_s*.npz')
S=sorted(set(A)&set(D)&set(H)); print(f"paired seeds: {len(S)}   cold acc {ca*100:.2f}%")
for name,X in (('baseline small',A),('+15 on 1471, rewards seen',D),('+15 on 1471, rewards HIDDEN',H),('+15 random, rewards seen',PR)):
    acc=np.array([X[s]['acc'] for s in S if s in X]); lk=np.array([X[s]['lock'] for s in S if s in X])
    print(f"{name:30s} acc {acc.mean()*100:5.2f}%  acc mult {acc.mean()/ca:.2f}x  settled on 1471 in {lk.sum()}/{len(lk)} seeds")
h=np.array([H[s]['hidden'] for s in S]); print(f"injected answers actually routed (and hidden) per seed: mean {h.mean():.2f}, zero in {np.sum(h==0)} seeds")
a=np.array([A[s]['acc'] for s in S]); d=np.array([D[s]['acc'] for s in S]); hh=np.array([H[s]['acc'] for s in S])
for lab,x,y in (('seen vs baseline',d,a),('hidden vs baseline',hh,a),('seen vs hidden',d,hh)):
    t=stats.ttest_rel(x,y); print(f"paired t, {lab:20s}: t={t.statistic:.2f}  p={t.pvalue:.3g}")
m=h>0
if m.sum()>=3:
    t=stats.ttest_rel(d[m],hh[m]); print(f"seeds where >=1 injected answer was routed (n={m.sum()}): seen {d[m].mean()*100:.2f}% vs hidden {hh[m].mean()*100:.2f}%  t={t.statistic:.2f} p={t.pvalue:.3g}")
