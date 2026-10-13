"""Lean re-implementation of the runner's cold-LinUCB and gated-Router tracks
(experiment_algorithm1.py semantics), with float32 scoring on the Apple GPU and
float64 Sherman-Morrison state on the host -- the same split the CUDA path uses."""
import numpy as np, torch
DEV = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def sm_update(V_inv, x):
    if not np.all(np.isfinite(x)) or np.linalg.norm(x) < 1e-10: return V_inv
    Vx = V_inv @ x; den = 1.0 + x @ Vx
    if abs(den) < 1e-12: return V_inv
    return V_inv - np.outer(Vx, Vx) / den

class Data:
    def __init__(self, path):
        z = np.load(path)
        self.C = z['contexts']; self.Z = z['Z'].astype(np.float64)
        self.Zt = torch.as_tensor(np.ascontiguousarray(self.Z), dtype=torch.float32, device=DEV)
        fl, of = z['y_flat'], z['y_offsets']
        self.ys = [set(fl[of[i]:of[i+1]].tolist()) for i in range(len(of)-1)]
        self.llm = {k[4:]: z[k] for k in z.files if k.startswith('llm_')}
        self.d = self.Z.shape[1]
    @torch.no_grad()
    def score(self, V_inv, theta, cx, alpha):
        Cx = torch.as_tensor(((V_inv*cx[:,None])*cx[None,:]).astype(np.float32), device=DEV)
        w = torch.as_tensor((cx*theta).astype(np.float32), device=DEV)
        var = ((self.Zt @ Cx) * self.Zt).sum(1).clamp_min(0)
        ucb = self.Zt @ w + alpha * var.sqrt()
        return int(torch.argmax(torch.nan_to_num(ucb, nan=float('-inf'))).item())

def run(D, order, T, lam=0.1, decay=0.9985, llm=None, c=None, cold=True, router=True,
        warm=None):
    """order: context indices; llm: expert proposals (array over contexts) for the gate;
    warm: optional list of (context, action) pairs fed to the router's learner before round 1."""
    d = D.d; out = {}
    Vc = np.eye(d)/lam; bc = np.zeros(d)
    Vr = np.eye(d)/lam; br = np.zeros(d)
    if warm is not None:
        for (j, a) in warm:
            x = D.C[j].astype(np.float64)*D.Z[a]; r = 1.0 if a in D.ys[j] else 0.0
            Vr = sm_update(Vr, x); br += r*x
    isc = 1.0/np.sqrt(c) if c else None
    ac=np.zeros(T,np.int64); rc=np.zeros(T); ar=np.zeros(T,np.int64); rr=np.zeros(T); ro=np.zeros(T,bool)
    for t in range(T):
        j = int(order[t]); cx = D.C[j].astype(np.float64); ys = D.ys[j]
        alpha = max(0.1, decay**t)
        if cold:
            a = D.score(Vc, Vc@bc, cx, alpha); r = 1.0 if a in ys else 0.0
            x = cx*D.Z[a]; Vc = sm_update(Vc, x); bc += r*x; ac[t]=a; rc[t]=r
        if router:
            a0 = D.score(Vr, Vr@br, cx, alpha); x0 = cx*D.Z[a0]
            use = False
            if llm is not None and c is not None:
                al = int(llm[j]); xl = cx*D.Z[al]
                use = bool(isc*np.linalg.norm(xl) < np.sqrt(max(0.0, x0@Vr@x0)))
            a, x = (al, xl) if use else (a0, x0)
            r = 1.0 if a in ys else 0.0
            Vr = sm_update(Vr, x); br += r*x; ar[t]=a; rr[t]=r; ro[t]=use
    return dict(a_cold=ac, r_cold=rc, a_router=ar, r_router=rr, routed=ro)
