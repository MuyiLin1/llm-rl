import numpy as np, collections
D='/Users/linmuyi/code/DMwithLLM/reproducibility_package/data/'
def load(f):
    z=np.load(D+f); flat,offs=z['y_flat'],z['y_offsets']; N=len(offs)-1
    return z,[flat[offs[i]:offs[i+1]] for i in range(N)],N
for fam in ('v1','v3'):
    z,ys,N=load(f'amazon_{fam}_slim.npz')
    for e in ('small','base','large'):
        lab=z[f'llm_{e}']
        for name,idx in (('held-out 50k-60k',np.arange(50000,60000)),('stream 0-50k',np.arange(0,50000))):
            valid=np.array([lab[i]>=0 for i in idx]); 
            hits=np.array([lab[i] in set(ys[i].tolist()) for i in idx])
            dom_prop=np.array([lab[i]==1471 for i in idx])
            dom_hit=dom_prop & hits
            if hits.sum()==0 and name.startswith('held'): 
                print(f"  {fam} {e:6} {name}: no valid expert proposals (labels range {lab[idx].min()}..{lab[idx].max()})"); continue
            print(f"  {fam} {e:6} {name:17} acc {hits.mean()*100:5.2f}%  proposes-1471 {dom_prop.mean()*100:5.2f}%  "
                  f"dominant-hit rate q_E {dom_hit.mean()*1e3:5.2f} per 1000")
    print()
# WikiLinks: is there a dominant action at all?
for f in ('wiki_v1_slim.npz',):
    z,ys,N=load(f)
    c=collections.Counter(int(y[0]) for y in ys)
    print("WikiLinks: contexts",N,"entities",z['Z'].shape[0],"top-5 entity prevalence:",[(a,round(k/N,4)) for a,k in c.most_common(5)])
