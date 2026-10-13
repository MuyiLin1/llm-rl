"""Measure the price ratio of Theorem `thm:general-price` on a real configuration.

The theorem bounds the multiplier over cold LinUCB by a ratio of PRICES, where a
price is regret paid per unit of coverage of the directions that matter:

    p_U    = cold LinUCB's regret / the coverage of U its own pulls bought
    pi_U   = the expert's regret  / the coverage of U its routed actions bought
    P_U    = p_U / pi_U                                  (the predicted multiplier)

Everything in the paper so far measures the EFFECT (V_tau). This measures the
CAUSE, which is the quantity the theorem is actually about, and it is the only
statistic that can explain the negatives as well as the positives: an expert whose
actions miss U entirely has pi_U = infinity and buys nothing, however accurate it is.

Defining U on real data. The template has fixed arm vectors, so U is a set of
coordinate directions. Here phi(x,a) = x * z_a depends on the context, so we take U
empirically: the leading eigenvectors of the second-moment matrix of the features
cold LinUCB played on rounds it got WRONG after routing stopped. Those are exactly
the directions cold keeps paying for and has not resolved -- the real counterpart of
"the directions that trap cold LinUCB".

Usage:
  python measure_price.py --slim amazon_v1_slim.npz --dataset_name amazonv1 \
      --model_family v1 --llm_types small base large --seeds 1 2 3 --rank 50
"""
import argparse, glob, json, os
import numpy as np


def stream_order(seed, n):
    """Reproduce the runner's stream exactly: np.random.seed(seed); permutation(n)."""
    np.random.seed(seed)
    return np.random.permutation(n)


def second_moment(contexts, Z, idx, actions, d, chunk=20000):
    """Sum of phi phi^T over the given (context, action) pairs, in chunks."""
    M = np.zeros((d, d))
    for s in range(0, len(idx), chunk):
        i = idx[s:s + chunk]
        a = actions[s:s + chunk]
        if len(i) == 0:
            continue
        P = contexts[i].astype(np.float64) * Z[a]
        M += P.T @ P
    return M


def coverage(contexts, Z, idx, actions, U, chunk=20000):
    """Sum over rounds of ||P_U phi||^2 -- the coverage of U those actions bought."""
    tot = 0.0
    for s in range(0, len(idx), chunk):
        i = idx[s:s + chunk]
        a = actions[s:s + chunk]
        if len(i) == 0:
            continue
        P = contexts[i].astype(np.float64) * Z[a]
        tot += float(np.sum((P @ U) ** 2))
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slim', required=True)
    ap.add_argument('--dataset_name', required=True)
    ap.add_argument('--model_family', required=True)
    ap.add_argument('--llm_types', nargs='+', default=['small', 'base', 'large'])
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 11)))
    ap.add_argument('--rank', type=int, default=50,
                    help='how many leading directions of the trapped subspace to use')
    ap.add_argument('--logdir', default='study')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    z = np.load(args.slim)
    contexts, Z = z['contexts'], z['Z'].astype(np.float64)
    flat, offs = z['y_flat'], z['y_offsets']
    K, d = Z.shape
    N_total = len(offs) - 1
    results = {}

    for llm in args.llm_types:
        lab = z[f'llm_{llm}']
        per_seed = []
        for seed in args.seeds:
            tag = (f"{args.logdir}/perround_algo1_{args.dataset_name}_{args.model_family}"
                   f"_{llm}_c100.0_pfull_s{seed}.npz")
            if not os.path.exists(tag):
                continue
            pr = np.load(tag)
            if 'a_cold' not in pr.files:
                print(f"  {llm} s{seed}: per-round log has no actions, skipping")
                continue
            n = int(pr['step'])
            r_cold, r_router = pr['r_cold'][:n], pr['r_router'][:n]
            routed = pr['routed'][:n].astype(bool)
            a_cold, a_router = pr['a_cold'][:n], pr['a_router'][:n]
            perm = stream_order(seed, N_total)[:n]

            # --- U: directions cold is still paying for after routing stops ---
            tail = ~routed
            trapped = tail & (r_cold == 0)
            if trapped.sum() < args.rank:
                print(f"  {llm} s{seed}: only {trapped.sum()} trapped rounds, skipping")
                continue
            M = second_moment(contexts, Z, perm[trapped], a_cold[trapped], d)
            w, V = np.linalg.eigh(M)
            U = V[:, ::-1][:, :args.rank]                 # leading `rank` directions

            # --- cold's price: what it paid, and what that bought, on those rounds ---
            cold_regret = float((1 - r_cold[trapped]).sum())
            cold_cov = coverage(contexts, Z, perm[trapped], a_cold[trapped], U)
            p_U = cold_regret / cold_cov if cold_cov > 0 else np.inf

            # --- the expert's price over the routed rounds ---
            ridx = perm[routed]
            racts = lab[ridx]
            exp_correct = np.array([
                1.0 if racts[j] in set(flat[offs[ridx[j]]:offs[ridx[j] + 1]].tolist())
                else 0.0 for j in range(len(ridx))])
            exp_regret = float((1 - exp_correct).sum())
            exp_cov = coverage(contexts, Z, ridx, racts, U)
            pi_U = exp_regret / exp_cov if exp_cov > 0 else np.inf

            P_U = p_U / pi_U if np.isfinite(pi_U) and pi_U > 0 else 0.0
            obs = (r_router.sum() / max(r_cold.sum(), 1))
            per_seed.append(dict(seed=seed, routed=int(routed.sum()),
                                 cold_regret=cold_regret, cold_cov=cold_cov,
                                 exp_regret=exp_regret, exp_cov=exp_cov,
                                 p_U=p_U, pi_U=pi_U, P_U=P_U, observed=float(obs)))
            print(f"  {llm} s{seed}: routed={int(routed.sum()):6d}  "
                  f"cold price={p_U:.4g}  expert price={pi_U:.4g}  "
                  f"P_U={P_U:.2f}  observed={obs:.2f}", flush=True)
        results[llm] = per_seed

    out = args.out or f'price_{args.dataset_name}.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")
    print(f"\n{'expert':8} {'cold price':>12} {'expert price':>14} {'P_U (pred)':>12} {'observed':>10}")
    for llm, rs in results.items():
        if not rs:
            continue
        import statistics as st
        print(f"{llm:8} {st.mean(r['p_U'] for r in rs):>12.4g} "
              f"{st.mean(r['pi_U'] for r in rs):>14.4g} "
              f"{st.mean(r['P_U'] for r in rs):>12.2f} "
              f"{st.mean(r['observed'] for r in rs):>10.2f}")


if __name__ == '__main__':
    main()
