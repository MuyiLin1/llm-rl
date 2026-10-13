"""Parts A and B of teammate-overlap-analysis.md (feature overlap of category 1471, BERT-768d).

No new bandit runs. Part B rebuilds the Router learner's state at round tau+1 from the logged plays of the
replay runs (prerun_study/replay/out/{expert}_s{seed}.npz: a_full, r_full, routed; same simulator and
seeds as tab:inject50k / the 3d replay): V = lam I + sum_{t<=tau} phi phi^T, b = sum_{t<=tau} r_t phi_t.
The rebuilt state is checked by re-scoring round tau+1 and comparing with the logged action.

Sampling seeds: A uses np.random.default_rng(0) on seed 1's stream; G4 uses default_rng(1);
B step 2 uses default_rng(0) per run on the rounds after tau.
  python overlap_analysis.py            -> out/partA.json, out/partB.csv, out/partB_step4.csv, states/*.npz
  python overlap_analysis.py p99        -> same Part B at tau_99 (99% of routed rounds done): *_p99.*"""
import os, sys, json, csv, time, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from sim import Data, DEV
SLIM = '/Users/linmuyi/code/DMwithLLM/reproducibility_package/data/amazon_v1_slim.npz'
REPLAY = os.path.join(os.path.dirname(HERE), 'replay', 'out')
OUT = os.path.join(HERE, 'out'); ST = os.path.join(HERE, 'states')
TAU_MODE = sys.argv[1] if len(sys.argv) > 1 else 'last'   # 'last' = request's tau; 'p99' = round by which 99% of routing is done
SFX = '' if TAU_MODE == 'last' else '_' + TAU_MODE
N_ON, LAM, DECAY, BETA, MU_A, SIG = 50000, 0.1, 0.9985, 0.1, 0.298, 0.5
DOM, G1 = 1471, [7961, 7892, 7891]
E, S = ['small', 'base', 'large'], range(1, 11)

def settle_cat(a):
    """Category with >20% of plays in the last fifth (stats.py rule), or None."""
    tail = a[int(0.8 * len(a)):]; c, n = np.unique(tail, return_counts=True)
    i = n.argmax(); return int(c[i]) if n[i] / len(tail) > 0.20 else None

def q(v):
    v = np.asarray(v, float)
    return dict(mean=v.mean(), sd=v.std(ddof=1), p5=np.percentile(v, 5), median=np.median(v),
                p95=np.percentile(v, 95), min=v.min(), max=v.max(), n=v.size)

def cos_group(X2, zd, Zb):
    """cos(phi(x,1471), phi(x,b)) for all rows of X (given X^2) and categories b (rows of Zb)."""
    num = X2 @ (zd[None, :] * Zb).T
    nu = np.sqrt(X2 @ zd**2); nb = np.sqrt(X2 @ (Zb**2).T)
    return num / (nu[:, None] * nb)

def part_a(D):
    np.random.seed(1); order = np.random.permutation(N_ON)
    idx = order[np.sort(np.random.default_rng(0).choice(N_ON, 5000, replace=False))]
    X = D.C[idx].astype(np.float64); X2 = X**2; zd = D.Z[DOM]
    U = X * zd; un = np.linalg.norm(U, axis=1)
    ubar = (U / un[:, None]).mean(0); ubar /= np.linalg.norm(ubar)
    cs = (U @ ubar) / un
    has = np.array([DOM in D.ys[j] for j in idx])
    A = {'n_sample': 5000, 'frac_1471_correct': has.mean(), 'A1': {}}
    for name, m in [('all', np.ones_like(has)), ('1471_correct', has), ('1471_not_correct', ~has)]:
        A['A1'][name] = dict(cos_to_ubar=q(cs[m]), norm_u=q(un[m]))
    # G2/G3 from cold LinUCB of the replay runs (seeds 1-10; the cold arm lives in the small_* files)
    acold = [np.load(f'{REPLAY}/small_s{s}.npz')['a_cold'] for s in S]
    c, n = np.unique(np.concatenate(acold), return_counts=True)
    o = np.argsort(-n); top = [(int(c[i]), int(n[i])) for i in o if c[i] != DOM][:20]
    cold_settle = {s: settle_cat(a) for s, a in zip(S, acold)}
    g3 = sorted({v for v in cold_settle.values() if v is not None and v != DOM})
    rest = np.setdiff1d(np.arange(D.Z.shape[0]), [DOM])
    g4 = np.sort(np.random.default_rng(1).choice(rest, 1000, replace=False))
    groups = {'G1': G1, 'G2': [b for b, _ in top], 'G3': g3, 'G4': g4.tolist()}
    A['G2_list'] = [dict(category=b, cold_plays=k) for b, k in top]
    A['cold_settle_by_seed'] = {str(s): v for s, v in cold_settle.items()}
    A['A2'] = {}; A['A3'] = {}
    for g, B in groups.items():
        cg = cos_group(X2, zd, D.Z[B]); A['A2'][g] = dict(pooled=q(cg.ravel()), n_categories=len(B))
        if g in ('G1', 'G3'):
            A['A2'][g]['per_category'] = {str(b): q(cg[:, i]) for i, b in enumerate(B)}
    zn = zd / np.linalg.norm(zd)
    for g, B in groups.items():
        zc = (D.Z[B] @ zn) / np.linalg.norm(D.Z[B], axis=1)
        A['A3'][g] = ({str(b): float(v) for b, v in zip(B, zc)} if g in ('G1', 'G3') else q(zc))
    A['A3']['all_other_categories'] = q((D.Z[rest] @ zn) / np.linalg.norm(D.Z[rest], axis=1))
    # what each coordinate of x^2 weights: overlap is sum_i x_i^2 z_d,i z_b,i
    A['x2_weight_concentration'] = dict(top10_coords_share_of_mean_x2=float(
        np.sort(X2.mean(0))[::-1][:10].sum() / X2.mean(0).sum()))
    return A, ubar, float(un.mean())

@torch.no_grad()
def widths2(D, Vinv, x):
    """v^T V^-1 v for v = x * z_b, all b (float32 on DEV, as in the runner's scorer)."""
    Cx = torch.as_tensor(((Vinv * x[:, None]) * x[None, :]).astype(np.float32), device=DEV)
    return ((D.Zt @ Cx) * D.Zt).sum(1).clamp_min(0).cpu().numpy().astype(np.float64)

def rebuild(D, e, s):
    z = np.load(f'{REPLAY}/{e}_s{s}.npz'); routed, a, r = z['routed'], z['a_full'], z['r_full']
    np.random.seed(s); order = np.random.permutation(N_ON)
    rt = np.where(routed)[0]
    tau = int(rt.max()) if TAU_MODE == 'last' else int(rt[int(np.ceil(0.99 * rt.size)) - 1])
    F = D.C[order[:tau + 1]].astype(np.float64) * D.Z[a[:tau + 1]]
    V = LAM * np.eye(D.d) + F.T @ F; b = F.T @ r[:tau + 1]
    Vinv = np.linalg.inv(V); Vinv = (Vinv + Vinv.T) / 2; th = Vinv @ b
    x1 = D.C[order[tau + 1]].astype(np.float64)
    ok = D.score(Vinv, th, x1, max(0.1, DECAY**(tau + 1))) == int(a[tau + 1])
    n1471 = int(np.sum(routed[:tau + 1] & (a[:tau + 1] == DOM) & (r[:tau + 1] == 1)))
    return dict(V=V, Vinv=Vinv, theta=th, tau=tau, alpha_tau=max(0.1, DECAY**tau), settled=settle_cat(a) == DOM,
                settle_cat=settle_cat(a), n_1471_correct_routed=n1471, n_routed=int(routed.sum()),
                check_action_tau1=bool(ok), order=order)

def part_b(D, ubar, mscale):
    rows, rows4 = [], []; others = np.setdiff1d(np.arange(D.Z.shape[0]), [DOM])
    for e in E:
        for s in S:
            t0 = time.time(); st = rebuild(D, e, s); Vinv, th, tau = st['Vinv'], st['theta'], st['tau']
            np.savez(f'{ST}/{e}_s{s}{SFX}.npz', Vinv=Vinv, theta=th, tau=tau, alpha_tau=st['alpha_tau'],
                     settled=st['settled'], n_1471_correct_routed=st['n_1471_correct_routed'])
            Vu = Vinv @ ubar; uVu = float(ubar @ Vu); N = 1 / uVu; mu = float(th @ ubar)
            post = st['order'][tau + 1:]
            ctx = post[np.sort(np.random.default_rng(0).choice(post.size, 200, replace=False))]
            RHO, IDX, mut, wid, top = [], [], [], [], []
            for j in ctx:
                x = D.C[j].astype(np.float64)
                rho_all = (D.Z @ (x * Vu)) / uVu
                I_all = D.Z @ (x * th) + BETA * np.sqrt(widths2(D, Vinv, x))
                RHO.append(rho_all[others]); IDX.append(I_all[others])
                ud = x * D.Z[DOM]; mut.append(float(th @ ud)); wid.append(BETA * np.sqrt(ud @ Vinv @ ud))
                top.append(int(np.argmax(I_all)) == DOM)
            RHO = np.concatenate(RHO); IDX = np.concatenate(IDX); lt = RHO < 1
            w = (IDX[lt] - RHO[lt] * mu) / (1 - RHO[lt]); om = w.max(); h = N * (mu - om)
            bound = (1 - np.exp(-2 * (MU_A - om) * h / SIG**2)) if (h > 0 and om < MU_A) else None
            # same quantities with the fixed direction at the scale of the real features (u = m * ubar)
            rs = RHO / mscale; lts = rs < 1; Ns = N / mscale**2; mus = mu * mscale
            ws = (IDX[lts] - RHO[lts] * mu) / (1 - rs[lts]); oms = ws.max(); hs = Ns * (mus - oms)
            bs = (1 - np.exp(-2 * (MU_A - oms) * hs / SIG**2)) if (hs > 0 and oms < MU_A) else None
            om99 = np.percentile(ws, 99); h99 = Ns * (mus - om99)   # margin with omega_99 in place of the max
            b99 = (1 - np.exp(-2 * (MU_A - om99) * h99 / SIG**2)) if (h99 > 0 and om99 < MU_A) else None
            rows.append(dict(seed=s, expert=e, tau=tau, alpha_tau=st['alpha_tau'], n_routed=st['n_routed'],
                n_1471_correct_routed=st['n_1471_correct_routed'], settled='yes' if st['settled'] else 'no',
                settle_cat=st['settle_cat'], N=N, mu_hat=mu, rho_max=RHO.max(), rho_99=np.percentile(RHO, 99),
                frac_rho_ge_1=1 - lt.mean(), omega=om, omega_99=np.percentile(w, 99), h=h,
                bound='n/a' if bound is None else bound,
                sc_N=Ns, sc_mu_hat=mus, sc_rho_max=rs.max(), sc_frac_rho_ge_1=1 - lts.mean(), sc_omega=oms,
                sc_omega_99=om99, sc_h=hs, sc_bound='n/a' if bs is None else bs,
                sc_h99=h99, sc_bound99='n/a' if b99 is None else b99,
                check_action_tau1=st['check_action_tau1']))
            mut = np.array(mut); wid = np.array(wid)
            rows4.append(dict(seed=s, expert=e, settled=rows[-1]['settled'], mu_t_mean=mut.mean(), mu_t_sd=mut.std(ddof=1),
                mu_t_p5=np.percentile(mut, 5), width_mean=wid.mean(), width_sd=wid.std(ddof=1),
                frac_1471_top_index=np.mean(top)))
            r = rows[-1]
            print(f"{e} s{s}: tau {tau} n1471 {r['n_1471_correct_routed']} settled {r['settled']} N {N:.1f} mu {mu:.3f} "
                  f"rho_max {r['rho_max']:.2f} frac>=1 {r['frac_rho_ge_1']:.3f} omega {om:.3f} h {h:.2f} "
                  f"top1471 {np.mean(top):.2f} check {st['check_action_tau1']} {time.time()-t0:.0f}s", flush=True)
    return rows, rows4

def wcsv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True); os.makedirs(ST, exist_ok=True)
    D = Data(SLIM)
    A, ubar, mscale = part_a(D); A['ubar_scale_mean_norm_u'] = mscale
    if TAU_MODE == 'last':
        np.save(f'{OUT}/ubar.npy', ubar)
        json.dump(A, open(f'{OUT}/partA.json', 'w'), indent=1, default=float); print('Part A written', flush=True)
    rows, rows4 = part_b(D, ubar, mscale)
    wcsv(f'{OUT}/partB{SFX}.csv', rows); wcsv(f'{OUT}/partB_step4{SFX}.csv', rows4); print('DONE', flush=True)
