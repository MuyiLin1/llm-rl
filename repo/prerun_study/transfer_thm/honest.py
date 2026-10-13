"""'Honest' (uninformed) expert: same number of D and R plays as A and B (2k each),
but it picks which window rounds get D WITHOUT seeing the type, so its right answers
come at the natural rate. Does its warm start also settle on D?"""
import numpy as np
def run(ex_actions,types,T,k,L,beta,lam):
    c=beta*L/np.sqrt(lam); tau=4*k; nD=nR=0; sD=sR=0.0; acts=[]; right=0
    for t in range(T):
        y=types[t]
        if t<tau: a=ex_actions[t]
        else:
            uD=sD/(lam+nD)+beta/np.sqrt(lam+nD); uR=sR/(lam+nR)+beta/np.sqrt(lam+nR)
            m=max(uD,uR,c); a='F' if c>=m-1e-12 else ('D' if uD>=uR else 'R')
        r=1.0 if a==y else 0.0
        if t<tau: right+=r
        if a=='D': nD+=1; sD+=r
        elif a=='R': nR+=1; sR+=r
        acts.append(a)
    return acts,right
p,q,lam,L,T=0.3,0.05,0.1,2.0,5000; c=0.135; beta=c*np.sqrt(lam)/L
for k in (2,4,8):
    st=0; rights=[]; N=1000
    for s in range(N):
        rng=np.random.default_rng(s)
        w=['D']*k+['R']*k+['0']*2*k; rng.shuffle(w)
        ty=list(w)+list(rng.choice(['D','R','0'],size=T-4*k,p=[p,q,1-p-q]))
        ex=['D']*(2*k)+['R']*(2*k); rng.shuffle(ex)          # blind to the type
        a,right=run(ex,ty,T,k,L,beta,lam); rights.append(right)
        st+=all(x=='D' for x in a[4*k:])
    print(f"k={k}: honest expert right on {np.mean(rights):.1f} of {4*k} routed rounds on average; its hybrid settles on D in {st}/{N} runs")

print("\n-- eventual behaviour and regret (pseudo-regret w.r.t. means p,q,0) --")
mu={'D':p,'R':q,'F':0.0}
from sim_fast import run_fast
for k in (2,4,8):
    ev=0; regH=[]; regA=[]; regB=[]; regC=[]; N=500
    for s in range(N):
        rng=np.random.default_rng(s)
        w=['D']*k+['R']*k+['0']*2*k; rng.shuffle(w)
        ty=list(w)+list(rng.choice(['D','R','0'],size=T-4*k,p=[p,q,1-p-q]))
        ex=['D']*(2*k)+['R']*(2*k); rng.shuffle(ex)
        a,_=run(ex,ty,T,k,L,beta,lam)
        ev+=all(x=='D' for x in a[T//2:])
        regH.append(sum(p-mu[x] for x in a))
        for pol,store in (('A',regA),('B',regB),('cold',regC)):
            b,_=run_fast(pol,ty,T,k,L,beta,lam); store.append(sum(p-mu[x] for x in b))
    print(f"k={k}: honest hybrid on D for the whole second half in {ev}/{N} runs | mean regret: honest {np.mean(regH):.0f}, A {np.mean(regA):.0f}, B {np.mean(regB):.0f}, cold {np.mean(regC):.0f}")
