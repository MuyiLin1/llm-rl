import numpy as np, glob, os
from scipy import stats
T=10000
def load(ph,mode,e):
    out={}
    for p in glob.glob(f"reh_out/{ph}_{mode}{'_'+e if e else ''}_s*.npz"):
        if p.endswith('.tmp.npz'): continue
        s=int(p.split('_s')[-1].split('.')[0]); o=np.load(p)
        a=o['a_router'] if mode=='router' else o['a_cold']; r=o['r_router'] if mode=='router' else o['r_cold']
        rt=o['routed'] if mode=='router' else np.zeros(T,bool)
        k=int(((a==1471)&rt&(r>0)).sum())   # correct 1471 plays on routed rounds
        out[s]=dict(lock=np.mean(a[int(.8*T):]==1471)>0.25, acc=r.mean(), k=k, share=np.mean(a[int(.8*T):]==1471), routed=int(rt.sum()))
    return out
cold=load('A','cold',None); ca=np.mean([v['acc'] for v in cold.values()])
for base_e, int_e in (('large','large_no1471'),('small','small_plus1471')):
    X=load('A','router',base_e); Y=load('D','router',int_e); S=sorted(set(X)&set(Y))
    lx=np.array([X[s]['lock'] for s in S]); ly=np.array([Y[s]['lock'] for s in S])
    ax=np.array([X[s]['acc'] for s in S]); ay=np.array([Y[s]['acc'] for s in S])
    kx=np.array([X[s]['k'] for s in S]); ky=np.array([Y[s]['k'] for s in S])
    b=((lx==1)&(ly==0)).sum(); c=((lx==0)&(ly==1)).sum()
    p_mc=stats.binomtest(max(b,c), b+c, 0.5).pvalue if b+c else 1.0
    t=stats.ttest_rel(ay,ax)
    print(f"\n{base_e} -> {int_e}  (n={len(S)} paired seeds)")
    print(f"  mean k (correct 1471 on routed): {kx.mean():.2f} -> {ky.mean():.2f}")
    print(f"  P_lock: {lx.mean()*100:.0f}% -> {ly.mean()*100:.0f}%   discordant lost={b} gained={c}  McNemar exact p={p_mc:.4f}")
    print(f"  acc:    {ax.mean()*100:.2f}% -> {ay.mean()*100:.2f}%   mult {ax.mean()/ca:.2f}x -> {ay.mean()/ca:.2f}x   paired t={t.statistic:.2f} p={t.pvalue:.2g}")
    print("  per-seed (lock before/after, acc before/after):")
    print("   "+"  ".join(f"s{s}:{int(X[s]['lock'])}{int(Y[s]['lock'])}" for s in S))
