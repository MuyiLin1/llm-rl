"""Phase B simulator: cold LinUCB and the gated Router on AmazonCat-13K with configurable exploration
schedule, regularization and feature map. With feat='prod', sched='fast', amin=0.1, lam=0.1 it is the
same computation as prerun_study/sim.py (checked against the cluster runner action for action):
float64 Sherman-Morrison state on the host, float32 scoring on the GPU.

feat='prod'      phi(x,a) = x * z_a                     (the paper's features, d)
feat='prod+arm'  phi(x,a) = [x * z_a, z_a]              (adds a context-free category term, 2d)
sched='fast'     alpha_t = max(amin, 0.9985^t)          (the paper's schedule)
sched='slow'     alpha_t = max(amin, 0.9995^t)
sched='const'    alpha_t = amin"""
import os, time, numpy as np, torch
DEV = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
DECAY = {'fast': 0.9985, 'slow': 0.9995, 'const': None}

def sm_update(V_inv, x):
    if not np.all(np.isfinite(x)) or np.linalg.norm(x) < 1e-10: return V_inv
    Vx = V_inv @ x; den = 1.0 + x @ Vx
    if abs(den) < 1e-12: return V_inv
    return V_inv - np.outer(Vx, Vx) / den

def alpha(sched, amin, t):
    return amin if sched == 'const' else max(amin, DECAY[sched] ** t)

class Data:
    def __init__(self, path, feat='prod'):
        z = np.load(path)
        self.C = z['contexts']; self.Z = z['Z'].astype(np.float64); self.feat = feat
        self.Zt = torch.as_tensor(np.ascontiguousarray(self.Z), dtype=torch.float32, device=DEV)
        fl, of = z['y_flat'], z['y_offsets']
        self.ys = [set(fl[of[i]:of[i+1]].tolist()) for i in range(len(of)-1)]
        self.llm = {k[4:]: z[k] for k in z.files if k.startswith('llm_')}
        self.dz = self.Z.shape[1]; self.d = self.dz * (2 if feat == 'prod+arm' else 1)
    def phi(self, cx, a):
        v = cx * self.Z[a]
        return v if self.feat == 'prod' else np.concatenate([v, self.Z[a]])
    @torch.no_grad()
    def score(self, V_inv, theta, cx, alpha):
        if self.feat == 'prod':          # identical to sim.py
            Cx = torch.as_tensor(((V_inv*cx[:,None])*cx[None,:]).astype(np.float32), device=DEV)
            w = torch.as_tensor((cx*theta).astype(np.float32), device=DEV)
            var = ((self.Zt @ Cx) * self.Zt).sum(1).clamp_min(0)
            ucb = self.Zt @ w + alpha * var.sqrt()
        else:
            cxt = torch.as_tensor(cx.astype(np.float32), device=DEV)
            W = torch.cat([self.Zt * cxt, self.Zt], 1)
            Vt = torch.as_tensor(V_inv.astype(np.float32), device=DEV)
            th = torch.as_tensor(theta.astype(np.float32), device=DEV)
            var = ((W @ Vt) * W).sum(1).clamp_min(0)
            ucb = W @ th + alpha * var.sqrt()
        return int(torch.argmax(torch.nan_to_num(ucb, nan=float('-inf'))).item())

def run(D, spec, T=50000, ckpt=None, every=10000, progress=None, order=None):
    """spec: dict(kind='cold'|'router', expert, c, sched, amin, lam, seed). Returns a, r, routed arrays.
    Checkpoints to `ckpt` every `every` rounds and resumes from it. `order` overrides the seeded stream order."""
    s = spec['seed']
    if order is None: np.random.seed(s); order = np.random.permutation(50000)
    lam, sched, amin = spec['lam'], spec['sched'], spec['amin']
    router = spec['kind'] == 'router'
    if router: llm = D.llm[spec['expert']]; isc = 1.0/np.sqrt(spec['c'])
    if ckpt and os.path.exists(ckpt):
        z = np.load(ckpt); t0 = int(z['t']); V, b, A, R, RO = z['V'], z['b'], z['A'], z['R'], z['RO']
    else:
        t0 = 0; V = np.eye(D.d)/lam; b = np.zeros(D.d)
        A = np.zeros(T, np.int32); R = np.zeros(T, np.int8); RO = np.zeros(T, bool)
    start = time.time()
    for t in range(t0, T):
        j = int(order[t]); cx = D.C[j].astype(np.float64); ys = D.ys[j]; al_t = alpha(sched, amin, t)
        a0 = D.score(V, V@b, cx, al_t); x0 = D.phi(cx, a0); use = False
        if router:
            al = int(llm[j]); xl = D.phi(cx, al)
            use = bool(isc*np.linalg.norm(xl) < np.sqrt(max(0.0, x0@V@x0)))
        a, x = (al, xl) if use else (a0, x0); r = 1 if a in ys else 0
        V = sm_update(V, x); b += r*x; A[t] = a; R[t] = r; RO[t] = use
        if ckpt and ((t+1) % every == 0) and t+1 < T:
            tmp = ckpt + '.tmp.npz'; np.savez(tmp, t=t+1, V=V, b=b, A=A, R=R, RO=RO); os.replace(tmp, ckpt)
        if progress and (t+1) % 1000 == 0:
            progress(t+1, T, float(R[:t+1].mean()), (t+1-t0)/max(1e-9, time.time()-start))
    return A, R, RO
