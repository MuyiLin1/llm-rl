"""2a stronger version of Theorem 3: tail types D (p), R (q), F (f), none (rest). On an F round the
fresh arm pays 1. Expert A plays D on D/R/F rounds and R on none; expert B plays R on D/R, the fresh
arm on F, and D on none. Window (first 4k rounds) unchanged: k D, k R, 2k none.
Checks: B is strictly more accurate than A after tau (q+f > p) yet its hybrid never plays D;
cold regret (p-f)T; A settles w.p. >= 1-exp(-8(p-w)h); ratios; and the in-expectation corollary."""
import numpy as np
def run(policy, types, T, k, L, beta, lam):
    w=beta*L/np.sqrt(lam); tau=4*k; nD=nR=0; sD=sR=0.0; reg=0.0; gap=None
    for t in range(T):
        y=types[t]
        if policy!='cold' and t<tau:
            a={'A':{'D':'D','R':'D','0':'R'},'B':{'D':'R','R':'R','0':'D'}}[policy][y]
        else:
            uD=sD/(lam+nD)+beta/np.sqrt(lam+nD); uR=sR/(lam+nR)+beta/np.sqrt(lam+nR)
            m=max(uD,uR,w); a='F' if w>=m-1e-12 else ('D' if uD>=uR else 'R')
        r=1.0 if a==y else 0.0
        if a=='D': nD+=1; sD+=r
        elif a=='R': nR+=1; sR+=r
        reg+={'D':0.0,'R':P-Q,'F':P-F}[a]
        if t>=tau and a=='D' and gap is None: pass
    return reg, nD
rng=np.random.default_rng(0)
P,Q,F=0.30,0.10,0.25          # q+f=0.35 > p=0.30 : B more accurate; f<p : D still optimal
lam,L=1.0,2.0; w=0.135; beta=w*np.sqrt(lam)/L
print(f"p={P} q={Q} f={F}  A tail acc={P}  B tail acc={Q+F}  omega={w}")
def types_for(T,k):
    win=np.array(['D']*k+['R']*k+['0']*(2*k)); rng.shuffle(win)
    u=rng.random(T-4*k); tail=np.where(u<P,'D',np.where(u<P+Q,'R',np.where(u<P+Q+F,'F','0')))
    return np.concatenate([win,tail])
for k in (2,4,8):
    h=k*(1-2*w)-w*lam; bound=1-np.exp(-8*(P-w)*h)
    for T in (2000,20000):
        runs=300 if T==2000 else 100; RA=[];RB=[];RC=[];accA=[];accB=[];settle=0;Bd=0
        for _ in range(runs):
            ty=types_for(T,k); tail=ty[4*k:]
            accA.append(np.mean(tail=='D')); accB.append(np.mean((tail=='R')|(tail=='F')))
            ra,_=run('A',ty,T,k,L,beta,lam); rb,nDb=run('B',ty,T,k,L,beta,lam); rc,_=run('cold',ty,T,k,L,beta,lam)
            RA.append(ra);RB.append(rb);RC.append(rc); settle+=abs(ra-2*k*(P-Q))<1e-9; Bd+=(nDb>2*k)
        RA,RB,RC=map(np.array,(RA,RB,RC))
        print(f"k={k} T={T}: accA {np.mean(accA):.3f} accB {np.mean(accB):.3f} | cold {RC.mean():.1f} (=(p-f)T {(P-F)*T:.1f}) "
              f"| A settles {settle/runs:.3f} (bound {bound:.3f}) | B ever plays D after tau: {Bd} "
              f"| B min regret {RB.min():.1f} >= (T-2k)min(p-q,p-f) {(T-2*k)*min(P-Q,P-F):.1f} "
              f"| mean mult A {RC.mean()/RA.mean():.1f}  B {RC.mean()/RB.mean():.2f} (<= {(P-F)/min(P-Q,P-F)*T/(T-2*k):.2f})")
# in-expectation corollary: k chosen so that exp(-8(p-w)h) <= 1/T
print("\nexpectation corollary: k_T = ceil((ln T/(8(p-w)) + w*lam)/(1-2w))")
for T in (2000,20000,200000):
    k=int(np.ceil((np.log(T)/(8*(P-w))+w*lam)/(1-2*w))); runs=200 if T<=20000 else 40
    RA=[];RC=[]
    for _ in range(runs):
        ty=types_for(T,k); RA.append(run('A',ty,T,k,L,beta,lam)[0]); RC.append(run('cold',ty,T,k,L,beta,lam)[0])
    print(f"T={T} k={k}: mean R(A)={np.mean(RA):.2f} <= 2k(p-q)+(p-f) = {2*k*(P-Q)+(P-F):.2f};  mean mult {np.mean(RC)/np.mean(RA):.0f}  (T/log T scale: {T/np.log(T):.0f})")
