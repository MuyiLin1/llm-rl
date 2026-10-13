import numpy as np, sys
sys.path.insert(0,'.')
from sim_fast import run_fast
from scipy.optimize import brentq
def types_for(T,k,p,q,rng):
    w=['D']*k+['R']*k+['0']*2*k; rng.shuffle(w)
    return list(w)+list(rng.choice(['D','R','0'],size=T-4*k,p=[p,q,1-p-q]))
lam,L=0.1,2.0
print("== sharper bound: P(settle) >= 1-exp(-8(p-c)(k(1-2c)-c*lam)); exact Lundberg root also shown")
for p,q,c in ((0.3,0.05,0.135),(0.5,0.1,0.225),(0.2,0.0,0.09)):
    beta=c*np.sqrt(lam)/L; T=5000
    th=brentq(lambda t: p*np.exp(-t*(1-c))+(1-p)*np.exp(t*c)-1, 1e-6, 50)
    print(f" p={p} c={c}: 8(p-c)={8*(p-c):.3f}  Lundberg theta*={th:.3f}")
    for k in (1,2,3,4,6):
        N=2000; st=0
        for s in range(N):
            ty=types_for(T,k,p,q,np.random.default_rng(777*k+s)); a,_=run_fast('A',ty,T,k,L,beta,lam)
            st+=all(x=='D' for x in a[4*k:])
        h=k*(1-2*c)-c*lam
        print(f"   k={k}: settle {st/N:.3f}  bound(Hoeffding) {1-np.exp(-8*(p-c)*h):.3f}  bound(Lundberg) {1-np.exp(-th*h):.3f}")
print("\n== large exploration (c>1): does anyone escape?")
p,q=0.3,0.05
for c in (1.2,3.0):
    beta=c*np.sqrt(lam)/L; esc=0
    for s in range(200):
        ty=types_for(3000,16,p,q,np.random.default_rng(s)); a,_=run_fast('A',ty,3000,16,L,beta,lam); esc+=('D' in a[64:])
    print(f"  c={c}: runs in which A ever plays D after routing: {esc}/200")
