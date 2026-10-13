"""Variant: window adds j 'F'-type rounds (only that round's fresh arm is correct) and j '0b'-type rounds (nothing correct).
A: F->D (wrong), 0b->fresh (wrong).  B: F->fresh (right), 0b->D (wrong).  Both add j D-plays and j fresh plays.
B then has k+j correct routed answers vs A's k; Gram agrees on e1, e2 and on every fresh direction offered after tau."""
import numpy as np
def run_v(policy,types,T,k,j,L,beta,lam):
    c=beta*L/np.sqrt(lam); tau=4*k+2*j; nD=nR=0; sD=sR=0.0; acts=[]; rew=[]; freshplays=[]
    EA={'D':'D','R':'D','0':'R','F':'D','0b':'F'}; EB={'D':'R','R':'R','0':'D','F':'F','0b':'D'}
    for t in range(T):
        y=types[t]
        if policy!='cold' and t<tau: a=(EA if policy=='A' else EB)[y]
        else:
            uD=sD/(lam+nD)+beta/np.sqrt(lam+nD); uR=sR/(lam+nR)+beta/np.sqrt(lam+nR)
            m=max(uD,uR,c); a='F' if c>=m-1e-12 else ('D' if uD>=uR else 'R')
        r=1.0 if a==y else 0.0
        if a=='D': nD+=1; sD+=r
        elif a=='R': nR+=1; sR+=r
        else: freshplays.append(t)
        acts.append(a); rew.append(r)
    return acts,rew,(nD,nR,freshplays)
p,q,lam,L=0.3,0.05,0.1,2.0; c=0.135; beta=c*np.sqrt(lam)/L; T=5000
for k,j in ((4,4),(4,8),(6,12)):
    st=0; bD=0; acc=[]; N=500
    for s in range(N):
        rng=np.random.default_rng(s)
        w=['D']*k+['R']*k+['0']*2*k+['F']*j+['0b']*j; rng.shuffle(w)
        ty=list(w)+list(rng.choice(['D','R','0'],size=T-len(w),p=[p,q,1-p-q]))
        a,ra,(nDa,nRa,fa)=run_v('A',ty,T,k,j,L,beta,lam); b,rb,(nDb,nRb,fb)=run_v('B',ty,T,k,j,L,beta,lam)
        tau=4*k+2*j
        if s==0:
            print(f"k={k} j={j}: routed correct A={sum(ra[:tau]):.0f} B={sum(rb[:tau]):.0f}; window D-plays A={a[:tau].count('D')} B={b[:tau].count('D')}, R-plays A={a[:tau].count('R')} B={b[:tau].count('R')}, fresh A={a[:tau].count('F')} B={b[:tau].count('F')}")
        st+=all(x=='D' for x in a[tau:]); bD+=('D' in b[tau:])
    print(f"   A settles on D: {st}/{N};  B ever plays D after routing: {bD}/{N}")
