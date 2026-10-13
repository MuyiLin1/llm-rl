"""3d rerun: transfer-removed and no-response-sum ablations with the routed rounds held FIXED.
One pass per (expert, seed) runs, in lockstep on the same stream:
  full    : the gated Router (sim.run semantics, c=100)
  nom     : routes exactly the rounds `full` routed, plays the LLM there, but its response sum
            gets <theta_or, phi> instead of the observed reward (transfer m removed)
  passive : same routed rounds, covariance only (no response-sum update on routed rounds)
  cold    : cold LinUCB (only in the 'small' jobs; it does not depend on the expert)
On non-routed rounds each learner plays its own LinUCB choice and sees the true reward.
Worker i of W takes jobs i, i+W, ... ; output written atomically."""
import sys, os, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'amazoncat13k'))
from sim import Data, sm_update
SLIM = '/Users/linmuyi/code/DMwithLLM/reproducibility_package/data/amazon_v1_slim.npz'
THETA = os.path.join(HERE, 'theta_or_v1.npy')
T, C_CONF, LAM, DECAY, ONLINE_T = 50000, 100.0, 0.1, 0.9985, 50000

def get_theta(D):
    if os.path.exists(THETA): return np.load(THETA)
    from transfer_decomp import fit_oracle
    held = np.arange(ONLINE_T, D.C.shape[0])
    print(f"fitting ridge oracle on {held.size} held-out contexts", flush=True)
    th = fit_oracle(D.C[held], D.Z, [D.ys[i] for i in held], 1.0)
    np.save(THETA, th); return th

def one(D, th, e, s, with_cold):
    d = D.d; np.random.seed(s); order = np.random.permutation(ONLINE_T)
    llm = D.llm[e]; isc = 1.0/np.sqrt(C_CONF)
    st = {k: [np.eye(d)/LAM, np.zeros(d)] for k in ('full', 'nom', 'passive', 'cold')}
    R = {k: np.zeros(T) for k in st}; A = {k: np.zeros(T, np.int64) for k in st}
    routed = np.zeros(T, bool)
    for t in range(T):
        j = int(order[t]); cx = D.C[j].astype(np.float64); ys = D.ys[j]
        alpha = max(0.1, DECAY**t)
        al = int(llm[j]); xl = cx*D.Z[al]; rl = 1.0 if al in ys else 0.0
        # full router: decides the routed rounds for everyone
        V, b = st['full']; a0 = D.score(V, V@b, cx, alpha); x0 = cx*D.Z[a0]
        use = bool(isc*np.linalg.norm(xl) < np.sqrt(max(0.0, x0@V@x0))); routed[t] = use
        for k in ('full', 'nom', 'passive'):
            V, b = st[k]
            if use:
                a, x, r = al, xl, rl
                if k == 'full': rb = r
                elif k == 'nom': rb = float(xl @ th)
                else: rb = 0.0
            else:
                a = a0 if k == 'full' else D.score(V, V@b, cx, alpha)
                x = cx*D.Z[a]; r = 1.0 if a in ys else 0.0; rb = r
            st[k][0] = sm_update(V, x); st[k][1] = b + rb*x; R[k][t] = r; A[k][t] = a
        if with_cold:
            V, b = st['cold']; a = D.score(V, V@b, cx, alpha); x = cx*D.Z[a]; r = 1.0 if a in ys else 0.0
            st['cold'][0] = sm_update(V, x); st['cold'][1] = b + r*x; R['cold'][t] = r; A['cold'][t] = a
    out = dict(routed=routed); out.update({f'r_{k}': R[k] for k in st}); out.update({f'a_{k}': A[k] for k in st})
    return out

if __name__ == '__main__':
    W, I = int(sys.argv[1]), int(sys.argv[2])
    jobs = [(e, s) for s in range(1, 11) for e in ('small', 'base', 'large')]
    mine = jobs[I::W]; D = Data(SLIM); th = get_theta(D)
    for e, s in mine:
        out = f"{HERE}/out/{e}_s{s}.npz"
        if os.path.exists(out): continue
        t0 = time.time(); o = one(D, th, e, s, with_cold=(e == 'small'))
        tmp = out + '.tmp.npz'; np.savez(tmp, **o); os.replace(tmp, out)
        m = {k: o[f'r_{k}'].mean()*100 for k in ('full', 'nom', 'passive')}
        print(f"[w{I}] {e} s{s}: full {m['full']:.2f}% nom {m['nom']:.2f}% passive {m['passive']:.2f}% "
              f"routed {int(o['routed'].sum())}  {time.time()-t0:.0f}s", flush=True)
    print(f"[w{I}] DONE", flush=True)
