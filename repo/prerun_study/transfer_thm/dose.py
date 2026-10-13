import numpy as np, sys
sys.path.insert(0,'.')
from sim_fast import run_fast
def types_for(T,k,p,q,rng):
    w=['D']*k+['R']*k+['0']*2*k; rng.shuffle(w)
    return list(w)+list(rng.choice(['D','R','0'],size=T-4*k,p=[p,q,1-p-q]))
for (p,q,Lfac) in ((0.3,0.05,0.9),(0.3,0.0,0.5),(0.5,0.1,0.9)):
    lam,L=0.1,2.0; c=min(p,0.5)/2*Lfac; beta=c*np.sqrt(lam)/L; T=5000
    print(f"\np={p} q={q} c={c:.3f} (fresh-arm index), T={T}")
    print(" k  settle-rate  lower-bound  returned-after-leaving  B-played-D  cold-regret==pT")
    for k in (1,2,3,4,6,8,12,16,24,32):
        N=400; settle=0; ret=0; bD=0; coldok=0
        for s in range(N):
            rng=np.random.default_rng(10_000*k+s); ty=types_for(T,k,p,q,rng)
            a,_=run_fast('A',ty,T,k,L,beta,lam); tail=a[4*k:]
            if all(x=='D' for x in tail): settle+=1
            else:
                i=next(j for j,x in enumerate(tail) if x!='D')
                if 'D' in tail[i:]: ret+=1
            b,_=run_fast('B',ty,T,k,L,beta,lam); bD+=('D' in b[4*k:])
            if s<20:
                cc,_=run_fast('cold',ty,T,k,L,beta,lam); coldok+=all(x=='F' for x in cc)
        lb=max(0.0,1-(2/p**2)*np.exp(-p*(k-2*c*lam)))
        print(f"{k:3d}  {settle/N:10.3f}  {lb:11.3f}  {ret:21d}  {bD:10d}  {coldok}/20")
