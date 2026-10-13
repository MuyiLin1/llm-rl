"""Is basin occupancy a sufficient summary? Predict each configuration's mean Router accuracy
as sum_b P_hat(b) * abar(b), abar pooled across ALL other configurations (leave-config-out),
basins = {1471 (rank1), rank2, rank3, other-locked, none}. AmazonCat only; gate/offline/corral."""
import json, numpy as np, collections
B=[x for x in json.load(open('basins_online.json')) if x['tag'] in ('amazonv1','amazonv3','corral','off1600','off5000','nom') and x['routed']<25000]
def basin(t):
    if t['share']<=0.2: return 'none'
    return {1:'r1',2:'r2',3:'r3'}.get(t['rank'],'rX')
for fam in ('v1','v3'):
    X=[x for x in B if x['fam']==fam]
    grp=collections.defaultdict(list)
    for x in X: grp[(x['tag'],x['c'],x['e'])].append(x)
    cold={}
    for x in X: cold.setdefault(x['s'],x['cold'])
    grp[('cold',0,'-')]=[dict(router=t) for t in cold.values()]
    rows=[]
    for k,L in grp.items():
        others=[t['router'] for kk,LL in grp.items() if kk!=k for t in LL]
        ab=collections.defaultdict(list)
        for t in others: ab[basin(t)].append(t['acc'])
        abar={b:np.mean(v) for b,v in ab.items()}
        P=collections.Counter(basin(t['router']) for t in L)
        pred=sum(P[b]/len(L)*abar.get(b,np.mean([t['acc'] for t in others])) for b in P)
        obs=np.mean([t['router']['acc'] for t in L]); rows.append((k,len(L),obs,pred))
    o=np.array([r[2] for r in rows]); p=np.array([r[3] for r in rows])
    print(f"\n{fam}: {len(rows)} configurations, leave-config-out basin-mixture prediction")
    print(f"   R2 = {1-np.sum((o-p)**2)/np.sum((o-o.mean())**2):.3f}   mean |err| = {np.mean(np.abs(o-p))*100:.2f}pp   max |err| = {np.max(np.abs(o-p))*100:.2f}pp")
    for k,n,ob,pr in sorted(rows, key=lambda r:-abs(r[2]-r[3]))[:6]: print(f"   worst: {str(k):32} n={n:2} obs {ob*100:5.2f}% pred {pr*100:5.2f}%")
