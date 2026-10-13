import numpy as np, glob, json, collections, re
out=[]
for fam in ('v1','v3'):
    z=np.load(f'amazon_{fam}_slim.npz'); fl,of=z['y_flat'],z['y_offsets']; n=min(50000,len(of)-1)
    prev=collections.Counter(l for i in range(n) for l in set(fl[of[i]:of[i+1]].tolist()))
    top=[k for k,_ in prev.most_common(6)]
    for p in sorted(glob.glob(f'perround_algo1_amazon{fam}_{fam}_*_pfull_s*.npz'))+sorted(glob.glob(f'perround_algo1_corral_{fam}_*_s*.npz')):
        m=re.match(r'perround_algo1_(.+?)_'+fam+r'_(small|base|large)_c([0-9.e+]+)_pfull(?:_\w+?)?_s(\d+)\.npz',p)
        if not m: continue
        tag,e,c,s=m.groups(); o=np.load(p)
        a,r,rt=o['a_router'],o['r_router'].astype(bool),o['routed'].astype(bool)
        if rt.mean()>0.5: continue
        k={str(L):int(((a==L)&rt&r).sum()) for L in top}
        # also: the Router's OWN (non-routed) correct plays of each top label during the first 3000 rounds
        own={str(L):int(((a[:3000]==L)&~rt[:3000]&r[:3000]).sum()) for L in top}
        out.append(dict(fam=fam,tag=tag,e=e,c=float(c),s=int(s),top=[int(x) for x in top],k=k,own=own,routed=int(rt.sum())))
json.dump(out,open('seeding_online.json','w')); print(len(out))
