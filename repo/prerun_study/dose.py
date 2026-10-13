import json, numpy as np, glob, collections
from scipy import stats
T=10000
c=json.load(open('cause.json')); on=json.load(open('online10k.json'))
# online c=100 gate runs: k from cause.json, lock@10k from online10k.json
O=[]
for x in c:
    if x['tag']=='amazonv1' and x['c']==100.0:
        m=[y for y in on[f"amazonv1_{x['e']}"] if y['s']==x['s']][0]
        O.append(dict(src='online',e=x['e'],k=x['n_dom_correct'],lock=m['lock10'],acc=m['acc10'],routed=x['routed']))
R=[]
for p in glob.glob('reh_out/[AD]_router_*_s*.npz'):
    if p.endswith('.tmp.npz'): continue
    o=np.load(p); a,r,rt=o['a_router'],o['r_router'],o['routed']
    e=p.split('router_')[1].rsplit('_s',1)[0]
    R.append(dict(src='reh',e=e,k=int(((a==1471)&rt&(r>0)).sum()),lock=bool(np.mean(a[8000:]==1471)>0.25),acc=float(r.mean()),routed=int(rt.sum())))
def tab(rows,name):
    print(f"\n{name}: n={len(rows)}")
    by=collections.defaultdict(list)
    for x in rows: by[min(x['k'],5)].append(x)
    for k in sorted(by): L=[x['lock'] for x in by[k]]; print(f"  k={k}{'+' if k==5 else ' '}  lock {sum(L):2d}/{len(L):2d} = {np.mean(L)*100:5.1f}%   experts {dict(collections.Counter(x['e'] for x in by[k]))}")
tab(O,'ONLINE c=100 (lock at 10k)'); tab(R,'REHEARSAL A+D (held-out 50-60k)')
print("\nmean k by group:")
for e in ('small','base','large'):
    print(f"  {e:6} online k={np.mean([x['k'] for x in O if x['e']==e]):.2f} routed={np.mean([x['routed'] for x in O if x['e']==e]):.0f}   "
          f"reh k={np.mean([x['k'] for x in R if x['e']==e]):.2f} routed={np.mean([x['routed'] for x in R if x['e']==e]):.0f}")
# logistic P(lock|k) comparison: same curve? likelihood ratio test on pooled vs separate
import numpy.linalg as la
def fit(rows):
    k=np.array([x['k'] for x in rows]); y=np.array([x['lock'] for x in rows],float)
    # model P=1-(1-p0)(1-b)^k ; grid MLE
    best=(-1e9,None)
    for p0 in np.linspace(0.001,0.6,300):
        for b in np.linspace(0.01,0.95,300):
            P=np.clip(1-(1-p0)*(1-b)**k,1e-9,1-1e-9); ll=(y*np.log(P)+(1-y)*np.log(1-P)).sum()
            if ll>best[0]: best=(ll,(p0,b))
    return best
lo,po=fit(O); lr,pr=fit(R); lp,pp=fit(O+R)
print(f"\nseeding model P=1-(1-p0)(1-b)^k:  online p0={po[0]:.3f} b={po[1]:.3f} | rehearsal p0={pr[0]:.3f} b={pr[1]:.3f} | pooled p0={pp[0]:.3f} b={pp[1]:.3f}")
LR=2*(lo+lr-lp); print(f"  same-curve LR test: chi2={LR:.2f} df=2 p={stats.chi2.sf(LR,2):.3f}")
