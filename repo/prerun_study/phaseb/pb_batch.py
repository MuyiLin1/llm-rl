"""Run K learners (cold LinUCB or gated Router, any schedule / lambda / c) on the SAME seed in lockstep.
Per learner the arithmetic is that of pb_sim.run (and so of prerun_study/sim.py for the default setting):
host float64 Sherman-Morrison state; the GPU keeps a float64 mirror of V_inv updated with the same
elementwise operations, so the float32 scoring matrix is built on the GPU without copying d x d
matrices each round. One GPU sync per round for all K learners."""
import os, time, json, numpy as np, torch
from pb_sim import DEV, alpha

class Data:
    def __init__(self, path):
        z = np.load(path)
        self.C = z['contexts']; self.Z = z['Z'].astype(np.float64)
        self.Zt = torch.as_tensor(np.ascontiguousarray(self.Z), dtype=torch.float32, device=DEV)
        fl, of = z['y_flat'], z['y_offsets']
        self.ys = [set(fl[of[i]:of[i+1]].tolist()) for i in range(len(of)-1)]
        self.llm = {k[4:]: z[k] for k in z.files if k.startswith('llm_')}
        self.dz = self.Z.shape[1]

def phi(D, feat, cx, a):
    v = cx * D.Z[a]
    return v if feat == 'prod' else np.concatenate([v, D.Z[a]])

def run_batch(D, feat, seed, specs, T=50000, ckpt=None, every=5000, prog=None):
    """One host->GPU copy and one GPU->host copy per round for all K learners (fewer sync points under
    GPU sharing). The float64 mirror update of round t-1 is applied on the GPU at the start of round t."""
    K = len(specs); dz = D.dz; d = dz * (2 if feat == 'prod+arm' else 1)
    np.random.seed(seed); order = np.random.permutation(50000)
    f64 = torch.float64
    if ckpt and os.path.exists(ckpt):
        z = np.load(ckpt); t0 = int(z['t']); V = list(z['V']); b = list(z['b'])
        A, R, RO = z['A'], z['R'], z['RO']
    else:
        t0 = 0; V = [np.eye(d)/s['lam'] for s in specs]; b = [np.zeros(d) for _ in specs]
        A = np.zeros((K, T), np.int16); R = np.zeros((K, T), np.int8); RO = np.zeros((K, T), bool)
    Vg = [torch.as_tensor(v, dtype=f64, device=DEV) for v in V]
    llm = [D.llm[s['expert']] if s['kind'] == 'router' else None for s in specs]
    isc = [1.0/np.sqrt(s['c']) if s['kind'] == 'router' else None for s in specs]
    Zt = D.Zt; start = time.time()
    # buffer layout: cx | per learner: theta (d), Vx (d), den (1), flag (1)
    L = dz + K*(2*d + 2); off = [dz + k*(2*d + 2) for k in range(K)]
    hb = torch.empty(L, dtype=f64).pin_memory(); hbn = hb.numpy(); gb = torch.empty(L, dtype=f64, device=DEV)
    pend = [None]*K                                    # (Vx, den) of the previous round's update, per learner
    theta = [V[k] @ b[k] for k in range(K)]
    with torch.no_grad():
        for t in range(t0, T):
            j = int(order[t]); cx = D.C[j].astype(np.float64); ys = D.ys[j]
            hbn[:dz] = cx
            for k in range(K):
                o = off[k]; hbn[o:o+d] = theta[k]
                if pend[k] is None: hbn[o+2*d+1] = 0.0
                else: hbn[o+d:o+2*d] = pend[k][0]; hbn[o+2*d] = pend[k][1]; hbn[o+2*d+1] = 1.0
            gb.copy_(hb, non_blocking=True)
            cxg = gb[:dz]
            if feat != 'prod': W = torch.cat([Zt * cxg.float(), Zt], 1)
            picks = []
            for k, s in enumerate(specs):
                o = off[k]
                if pend[k] is not None:
                    Vxg = gb[o+d:o+2*d]; Vg[k] = Vg[k] - torch.outer(Vxg, Vxg) / gb[o+2*d]
                thg = gb[o:o+d]; al = alpha(s['sched'], s['amin'], t)
                if feat == 'prod':
                    Cx = ((Vg[k]*cxg[:, None])*cxg[None, :]).float()
                    var = ((Zt @ Cx) * Zt).sum(1).clamp_min(0)
                    ucb = Zt @ (cxg*thg).float() + al * var.sqrt()
                else:
                    var = ((W @ Vg[k].float()) * W).sum(1).clamp_min(0)
                    ucb = W @ thg.float() + al * var.sqrt()
                picks.append(torch.argmax(torch.nan_to_num(ucb, nan=float('-inf'))))
            picks = torch.stack(picks).cpu().numpy()
            for k, s in enumerate(specs):
                a0 = int(picks[k]); x0 = phi(D, feat, cx, a0); use = False
                if llm[k] is not None:
                    al_ = int(llm[k][j]); xl = phi(D, feat, cx, al_)
                    use = bool(isc[k]*np.linalg.norm(xl) < np.sqrt(max(0.0, x0@V[k]@x0)))
                a, x = (al_, xl) if use else (a0, x0); r = 1 if a in ys else 0
                pend[k] = None
                if np.all(np.isfinite(x)) and np.linalg.norm(x) >= 1e-10:        # == sim.sm_update
                    Vx = V[k] @ x; den = 1.0 + x @ Vx
                    if abs(den) >= 1e-12:
                        V[k] = V[k] - np.outer(Vx, Vx) / den; pend[k] = (Vx, den)
                b[k] = b[k] + r*x; theta[k] = V[k] @ b[k]; A[k, t] = a; R[k, t] = r; RO[k, t] = use
            if ckpt and (t+1) % every == 0 and t+1 < T:
                tmp = ckpt + '.tmp.npz'; np.savez(tmp, t=t+1, V=np.stack(V), b=np.stack(b), A=A, R=R, RO=RO); os.replace(tmp, ckpt)
            if prog and (t+1) % 500 == 0:
                rate = (t+1-t0)/max(1e-9, time.time()-start)
                with open(prog + '.tmp', 'w') as f:
                    json.dump(dict(t=t+1, T=T, K=K, acc=[float(R[k, :t+1].mean()) for k in range(K)],
                                   rounds_per_sec=rate, eta_min=(T-t-1)/max(rate, 1e-9)/60,
                                   updated=time.strftime('%m-%d %H:%M:%S')), f)
                os.replace(prog + '.tmp', prog)
    return A, R, RO
