"""Exact (full-matrix) LinUCB on the candidate instance, to check the proof's claims.
Arms at round t: D=e1, R=e2, fresh b_t = L e_{t+2}. Hidden type y_t in {D,R,0}: which arm is correct.
Window t<=tau=4k: k D-type, k R-type, 2k 0-type (fixed order given by `window`).
Tail: i.i.d. P(D)=p, P(R)=q.  Learner sees only features (fixed per arm).
Expert A: D-type->D, R-type->D, 0-type->R.   Expert B: D-type->R, R-type->R, 0-type->D."""
import numpy as np
def run(policy, types, T, k, L, beta, lam, tie='fresh'):
    d=T+2; Vinv=np.eye(d)/lam; b=np.zeros(d); tau=4*k
    acts=[]; rew=[]; gram=np.zeros((d,d))
    for t in range(T):
        y=types[t]
        arms={'D':np.eye(d)[0],'R':np.eye(d)[1],'F':L*np.eye(d)[t+2]}
        if policy!='cold' and t<tau:
            a=expert(policy,y)
        else:
            th=Vinv@b
            ucb={n:x@th+beta*np.sqrt(x@Vinv@x) for n,x in arms.items()}
            m=max(ucb.values()); best=[n for n,v in ucb.items() if v>=m-1e-12]
            a='F' if 'F' in best else best[0]
        x=arms[a]; r=1.0 if a==y else 0.0
        if t<tau and policy!='cold': gram+=np.outer(x,x)
        Vx=Vinv@x; Vinv-=np.outer(Vx,Vx)/(1+x@Vx); b+=r*x
        acts.append(a); rew.append(r)
    return acts,rew,gram
def expert(pol,y):
    if pol=='A': return {'D':'D','R':'D','0':'R'}[y]
    if pol=='B': return {'D':'R','R':'R','0':'D'}[y]
def make_types(T,k,p,q,rng):
    w=['D']*k+['R']*k+['0']*2*k; rng.shuffle(w)
    tail=rng.choice(['D','R','0'],size=T-4*k,p=[p,q,1-p-q])
    return list(w)+list(tail)
if __name__=='__main__':
    p,q,lam,L=0.3,0.05,0.1,2.0
    c=p/2*0.9; beta=c*np.sqrt(lam)/L          # fresh-arm index c = beta L/sqrt(lam)
    T,k=300,8
    mu={'D':p,'R':q,'F':0.0}
    for seed in range(3):
        rng=np.random.default_rng(seed); types=make_types(T,k,p,q,rng)
        out={pol:run(pol,types,T,k,L,beta,lam) for pol in ('cold','A','B')}
        gA,gB=out['A'][2],out['B'][2]
        print(f"seed {seed}: Gram equal {np.allclose(gA,gB)}  routed correct A={sum(out['A'][1][:4*k]):.0f} B={sum(out['B'][1][:4*k]):.0f}")
        for pol in ('cold','A','B'):
            a=out[pol][0]; reg=sum(p-mu[x] for x in a)
            tail=a[4*k:]
            print(f"   {pol:4} pseudo-regret {reg:7.2f}  tail plays D={tail.count('D')} R={tail.count('R')} F={tail.count('F')}")
