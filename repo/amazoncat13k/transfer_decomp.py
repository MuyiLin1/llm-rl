"""Test 3: decompose the warm-start estimate into its expert-independent and
expert-dependent parts, and check which part orders the experts.

Under the misspecified model r = <theta_or, phi> + eps(phi) + noise, the ridge
estimate from the routed rounds alone is

    theta_hat_E = theta_or - lam (lam I + M_E)^{-1} theta_or          [ridge bias]
                + (lam I + M_E)^{-1} sum_s eps_s phi_s                 [transfer m_E]

where eps_s = r_s - <theta_or, phi_s> is the linear model's residual on the
expert's own picks. Given M_E, the transfer m_E is the ONLY expert-dependent term.

theta_or is the paper's ridge oracle (compute_diagnostics.py), fitted here on the
held-out block (contexts 50,000-59,999) so it never sees the online stream. Its
Hadamard closed form A = (X^T X) * (Z^T Z) covers all 13,330 actions.

Decision quality is measured on held-out contexts by the rank of the correct
action among all 13,330 (mean reciprocal rank; higher is better), for
  theta_or               the linear model's own best fit,
  theta_hat_E            what the expert's routed rounds actually left behind,
  theta_hat_E - m_E      the same with the transfer removed.
Prediction: the last row is expert-independent; the middle row orders
large > base > small; the difference is m_E.
"""
import argparse, json, os
import numpy as np


def fit_oracle(C, Z, ys, lam=1.0):
    """Paper's ridge oracle over all actions; O(n d^2 + K d^2) via the Hadamard form."""
    X = C.astype(np.float64); Zd = Z.astype(np.float64)
    A = (X.T @ X) * (Zd.T @ Zd)
    b = np.zeros(X.shape[1])
    for i in range(X.shape[0]):
        for a in ys[i]:
            b += X[i] * Zd[a]
    return np.linalg.solve(A + lam * np.eye(X.shape[1]), b)


def mrr(theta, C, Z, ys, batch=500):
    """Mean reciprocal rank of the (best) correct action under scores <theta, x*z_a>."""
    th = theta.astype(np.float32); out = []
    for i in range(0, C.shape[0], batch):
        S = (C[i:i+batch] * th[None, :]) @ Z.T          # (B, K)
        for j in range(S.shape[0]):
            s = S[j]; best = max(s[a] for a in ys[i+j])
            out.append(1.0 / (1 + int((s > best).sum())))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slim', default='amazon_v1_slim.npz')
    ap.add_argument('--logdir', default='study')
    ap.add_argument('--dataset_name', default='amazonv1')
    ap.add_argument('--family', default='v1')
    ap.add_argument('--llm_types', nargs='+', default=['small', 'base', 'large'])
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 11)))
    ap.add_argument('--lam', type=float, default=0.1, help='LinUCB ridge (the runner uses 0.1)')
    ap.add_argument('--oracle_lam', type=float, default=1.0, help='paper uses 1.0')
    ap.add_argument('--online_T', type=int, default=50000)
    ap.add_argument('--n_eval', type=int, default=4000)
    ap.add_argument('--out', default='transfer_decomp.json')
    args = ap.parse_args()

    z = np.load(args.slim)
    C, Z = z['contexts'], z['Z']
    flat, offs = z['y_flat'], z['y_offsets']
    ys = [set(flat[offs[i]:offs[i+1]].tolist()) for i in range(len(offs) - 1)]
    d = Z.shape[1]
    held = np.arange(args.online_T, C.shape[0])
    ev = held[:args.n_eval]

    oracle_path = f'theta_or_{args.family}.npy'
    if os.path.exists(oracle_path):
        theta_or = np.load(oracle_path)
    else:
        print(f"fitting ridge oracle on {held.size} held-out contexts ...", flush=True)
        theta_or = fit_oracle(C[held], Z, [ys[i] for i in held], args.oracle_lam)
        np.save(oracle_path, theta_or)
    print(f"theta_or: norm {np.linalg.norm(theta_or):.4f}   "
          f"held-out MRR {mrr(theta_or, C[ev], Z, [ys[i] for i in ev]):.4f}", flush=True)

    # Residual std of the linear model on held-out (correct pair + one random
    # negative per context). Test 2 uses it as the noise level of the linearized
    # reward: the counterfactual in which what the linear model cannot explain is
    # noise rather than structure. Also report the signal scale so sigma is
    # interpretable.
    rr = np.random.default_rng(0); res = []; sig_pos = []
    for i in ev:
        cx = C[i].astype(np.float64)
        a_pos = next(iter(ys[i])); a_neg = int(rr.integers(Z.shape[0]))
        pp = (cx * theta_or) @ Z[a_pos].astype(np.float64)
        pn = (cx * theta_or) @ Z[a_neg].astype(np.float64)
        sig_pos.append(pp)
        res += [1.0 - pp, (1.0 if a_neg in ys[i] else 0.0) - pn]
    stats = dict(resid_std=float(np.std(res)), pred_correct_mean=float(np.mean(sig_pos)),
                 pred_correct_std=float(np.std(sig_pos)), theta_norm=float(np.linalg.norm(theta_or)))
    json.dump(stats, open(f'oracle_stats_{args.family}.json', 'w'), indent=2)
    print(f"oracle residual std {stats['resid_std']:.4f}; <theta_or,phi(x,a*)> mean "
          f"{stats['pred_correct_mean']:.4f} sd {stats['pred_correct_std']:.4f}", flush=True)

    Zd = Z.astype(np.float64); lamI = args.lam * np.eye(d)
    results = {}
    print(f"\n{'expert':7} {'seed':>4} {'routed':>6} {'correct':>7} {'MRR(th_or)':>10} "
          f"{'MRR(th_hat)':>11} {'MRR(no m)':>10} {'|m|':>8} {'<m,phi*>-<m,phi>':>16}")
    for llm in args.llm_types:
        for s in args.seeds:
            p = f"{args.logdir}/perround_algo1_{args.dataset_name}_{args.family}_{llm}_c100.0_pfull_s{s}.npz"
            if not os.path.exists(p):
                continue
            L = np.load(p); n = int(L['step'])
            np.random.seed(s); perm = np.random.permutation(args.online_T)   # runner's stream
            rt = np.flatnonzero(L['routed'][:n].astype(bool))
            phi = C[perm[rt]].astype(np.float64) * Zd[L['a_router'][rt]]      # (m, d)
            r = L['r_router'][rt].astype(np.float64)
            M = phi.T @ phi
            V = lamI + M
            eps = r - phi @ theta_or                                          # residuals
            c = phi.T @ r
            theta_hat = np.linalg.solve(V, c)
            m = np.linalg.solve(V, phi.T @ eps)                               # transfer
            no_m = theta_hat - m                                              # = ridge-shrunk oracle
            # how much the transfer favours the correct action on held-out contexts
            Ce = C[ev].astype(np.float64); lift = []
            for k, i in enumerate(ev):
                sc = (Ce[k] * m) @ Zd.T
                lift.append(max(sc[a] for a in ys[i]) - sc.mean())
            row = dict(routed=int(rt.size), correct=int(r.sum()),
                       mrr_or=mrr(theta_or, C[ev], Z, [ys[i] for i in ev]),
                       mrr_hat=mrr(theta_hat, C[ev], Z, [ys[i] for i in ev]),
                       mrr_nom=mrr(no_m, C[ev], Z, [ys[i] for i in ev]),
                       m_norm=float(np.linalg.norm(m)), lift=float(np.mean(lift)),
                       eps_mean_correct=float(eps[r == 1].mean()) if (r == 1).any() else None,
                       eps_mean_wrong=float(eps[r == 0].mean()))
            results[f"{llm}_s{s}"] = row
            print(f"{llm:7} {s:>4} {row['routed']:>6} {row['correct']:>7} {row['mrr_or']:>10.4f} "
                  f"{row['mrr_hat']:>11.4f} {row['mrr_nom']:>10.4f} {row['m_norm']:>8.3f} "
                  f"{row['lift']:>16.5f}", flush=True)
    json.dump(results, open(args.out, 'w'), indent=2)

    print("\n=== means over seeds ===")
    print(f"{'expert':7} {'MRR(th_hat)':>11} {'MRR(no m)':>10} {'|m|':>8} {'lift':>10} "
          f"{'eps|correct':>12} {'eps|wrong':>10}")
    for llm in args.llm_types:
        rows = [v for k, v in results.items() if k.startswith(llm + '_')]
        if not rows: continue
        f = lambda key: np.mean([x[key] for x in rows if x[key] is not None])
        print(f"{llm:7} {f('mrr_hat'):>11.4f} {f('mrr_nom'):>10.4f} {f('m_norm'):>8.3f} "
              f"{f('lift'):>10.5f} {f('eps_mean_correct'):>12.4f} {f('eps_mean_wrong'):>10.4f}")


if __name__ == '__main__':
    main()
