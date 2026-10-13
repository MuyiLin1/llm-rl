import numpy as np
def run(policy,types,T,k,L,lam,alpha,coverage_only=False):
    tau=4*k; nD=nR=0; sD=sR=0.0; acts=[]
    for t in range(T):
        y=types[t]; a_t=alpha(t); c=a_t*L/np.sqrt(lam)
        if policy!='cold' and t<tau:
            a={'A':{'D':'D','R':'D','0':'R'},'B':{'D':'R','R':'R','0':'D'}}[policy][y]
        else:
            uD=sD/(lam+nD)+a_t/np.sqrt(lam+nD); uR=sR/(lam+nR)+a_t/np.sqrt(lam+nR)
            m=max(uD,uR,c); a='F' if c>=m-1e-12 else ('D' if uD>=uR else 'R')
        r=1.0 if a==y else 0.0
        upd_reward = not (coverage_only and t<tau and policy!='cold')
        if a=='D': nD+=1; sD+=r*upd_reward
        elif a=='R': nR+=1; sR+=r*upd_reward
        acts.append(a)
    return acts
def types_for(T,k,p,q,rng):
    w=['D']*k+['R']*k+['0']*2*k; rng.shuffle(w)
    return list(w)+list(rng.choice(['D','R','0'],size=T-4*k,p=[p,q,1-p-q]))
p,q,lam,L,T,k=0.3,0.05,0.1,2.0,5000,6
const=lambda t: 0.135*np.sqrt(lam)/L
print("coverage only (no reward sum on routed rounds), constant radius:")
for pol in ('A','B'):
    anyD=sum('D' in run(pol,types_for(T,k,p,q,np.random.default_rng(s)),T,k,L,lam,const,True)[4*k:] for s in range(300))
    print(f"  expert {pol}: runs where the hybrid ever plays D after routing: {anyD}/300")
# decaying schedule: alpha_t = a0*max(0.1, 0.9985^t) scaled so that the tail fresh index stays <= p/2 for t>tau
a0=0.3*np.sqrt(lam)/L   # fresh index at t=0 is 0.3 (> p/2); decays to 0.03
dec=lambda t: a0*max(0.1,0.9985**t)
print("decaying schedule (fresh-arm index starts at 0.30, floors at 0.03):")
for pol in ('cold','A','B'):
    st=0; anyD=0
    for s in range(300):
        a=run(pol,types_for(T,k,p,q,np.random.default_rng(s)),T,k,L,lam,dec)
        st+=all(x=='D' for x in a[4*k:]); anyD+=('D' in a[4*k:])
    print(f"  {pol:4}: settles on D {st}/300   ever plays D after routing {anyD}/300")
