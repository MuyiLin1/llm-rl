"""Scalar version: V is diagonal, so LinUCB state = (n_D, s_D, n_R, s_R); every fresh arm has index c=beta*L/sqrt(lam)."""
import numpy as np
def run_fast(policy, types, T, k, L, beta, lam, extra_fresh_correct=0):
    c=beta*L/np.sqrt(lam); tau=4*k
    nD=nR=0; sD=sR=0.0; acts=[]; rew=[]
    for t in range(T):
        y=types[t]
        if policy!='cold' and t<tau:
            a={'A':{'D':'D','R':'D','0':'R'},'B':{'D':'R','R':'R','0':'D'}}[policy][y]
        else:
            uD=sD/(lam+nD)+beta/np.sqrt(lam+nD); uR=sR/(lam+nR)+beta/np.sqrt(lam+nR)
            m=max(uD,uR,c); a='F' if c>=m-1e-12 else ('D' if uD>=uR else 'R')
        r=1.0 if a==y else 0.0
        if a=='D': nD+=1; sD+=r
        elif a=='R': nR+=1; sR+=r
        acts.append(a); rew.append(r)
    return acts,rew
