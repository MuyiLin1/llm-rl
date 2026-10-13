"""Pre-runtime dry run over the FULL action set.

Why this exists: pre_runtime_estimates.py must hold every candidate's features in
memory, so build_npz_for_estimates.py cuts the action set to A=128 per context.
In that reduced world cold LinUCB reaches ~65% accuracy instead of the ~10% it
gets among all 13,330 actions -- it is not starved, so a warm start has nothing
to repair and the dry run reports "no gain" for every expert, including the one
that multiplies by 2-5x in reality.

This script instead scores all 13,330 actions, exactly as experiment_algorithm1.py
does, by building phi(x,a) = x * z_a on the fly. It never materialises an (n, K, d)
tensor, so memory stays at ~250 MB.

Protocol (v5, Appendix `app:diagnostics`, "dry run"):
  * contexts are resampled from a held-out block DISJOINT from the online stream
  * cold:  LinUCB from lambda*I for T' rounds
  * warm:  play the expert's cached proposal for tau rounds (acquisition, charged
           at its held-out error rate), then LinUCB for the remaining T'-tau
  * implied multiplier = R_cold / (C_E(tau) + R_warm_tail)
Cold does not depend on the expert, so it is simulated once and shared.

  python dry_run_full.py --slim amazon_v1_slim.npz --T_dry 20000 --taus 1600 5000
"""
import argparse, json, time
import numpy as np
import torch

from experiment_algorithm1 import pick_device, make_scorer


def sm_update(V_inv, x):
    if not np.all(np.isfinite(x)) or np.linalg.norm(x) < 1e-10:
        return V_inv
    Vx = V_inv @ x
    denom = 1.0 + x @ Vx
    if abs(denom) < 1e-12:
        return V_inv
    return V_inv - np.outer(Vx, Vx) / denom


def linucb_rounds(order, contexts, Z, ys, V_inv, b, beta, label="", score=None):
    """Run LinUCB over the given stream; return the number of mistakes."""
    mistakes = 0
    t0 = time.time()
    for i, s in enumerate(order):
        cx = contexts[s].astype(np.float64)
        theta = V_inv @ b
        a = score(V_inv, theta, cx, beta)
        r = 1.0 if a in ys[s] else 0.0
        mistakes += 1.0 - r
        x = cx * Z[a]
        V_inv = sm_update(V_inv, x)
        b += r * x
        if label and (i + 1) % 2000 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"    [{label}] {i+1}/{len(order)} rounds, {rate:.1f}/s, "
                  f"acc so far {1 - mistakes/(i+1):.4f}", flush=True)
    return mistakes, V_inv, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slim', default='amazon_v1_slim.npz')
    ap.add_argument('--llm_types', nargs='+', default=['small', 'base', 'large'])
    ap.add_argument('--T_dry', type=int, default=20000)
    ap.add_argument('--taus', type=int, nargs='+', default=[1600, 5000])
    ap.add_argument('--lam', type=float, default=0.1)
    ap.add_argument('--beta', type=float, default=1.0)
    ap.add_argument('--online_T', type=int, default=50000,
                    help='size of the online stream; held-out contexts are taken '
                         'from AFTER this point so the dry run never sees it')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default='dryrun_full_results.json')
    ap.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    args = ap.parse_args()

    z = np.load(args.slim)
    contexts, Z = z['contexts'], z['Z']
    flat, offs = z['y_flat'], z['y_offsets']
    ys = [set(flat[offs[i]:offs[i+1]].tolist()) for i in range(len(offs) - 1)]
    K, d = Z.shape
    Z = Z.astype(np.float64)

    pool = np.arange(args.online_T, contexts.shape[0])
    if pool.size < 100:
        raise SystemExit("no held-out block: rebuild the slim file with a larger --n_keep")
    dev = pick_device(args.device)
    score = make_scorer(Z, dev, fast32=(dev != 'cpu'))
    print(f"Scoring device: {dev}")
    print(f"actions={K} dim={d} held-out pool={pool.size} "
          f"(indices {pool[0]}..{pool[-1]}, disjoint from the online stream)", flush=True)

    rng = np.random.default_rng(args.seed)
    stream = rng.choice(pool, size=args.T_dry, replace=True)

    print(f"\n=== cold LinUCB, {args.T_dry} rounds (shared by every expert) ===", flush=True)
    R_cold, _, _ = linucb_rounds(stream, contexts, Z, ys,
                                 (1.0 / args.lam) * np.eye(d), np.zeros(d),
                                 args.beta, label="cold", score=score)
    print(f"cold: {R_cold:.0f} mistakes, accuracy {1 - R_cold/args.T_dry:.4f}", flush=True)

    results = {'config': vars(args), 'R_cold': R_cold,
               'cold_acc': 1 - R_cold / args.T_dry, 'experts': {}}

    for llm in args.llm_types:
        lab = z[f'llm_{llm}']
        err = float(np.mean([0.0 if lab[s] in ys[s] else 1.0 for s in pool]))
        print(f"\n=== expert {llm}: held-out error {err:.4f} ===", flush=True)
        results['experts'][llm] = {'held_out_err': err, 'taus': {}}
        for tau in args.taus:
            V_inv = (1.0 / args.lam) * np.eye(d)
            b = np.zeros(d)
            acq = rng.choice(pool, size=tau, replace=True)
            for s in acq:                      # acquisition: play the expert
                cx = contexts[s].astype(np.float64)
                a = int(lab[s])
                r = 1.0 if a in ys[s] else 0.0
                x = cx * Z[a]
                V_inv = sm_update(V_inv, x)
                b += r * x
            tail = stream[tau:]
            R_warm, _, _ = linucb_rounds(tail, contexts, Z, ys, V_inv, b,
                                         args.beta, label=f"{llm} warm tau={tau}",
                                         score=score)
            C_E = tau * err
            mult = R_cold / (C_E + R_warm)
            print(f"  tau={tau}: C_E={C_E:.0f}  R_warm_tail={R_warm:.0f}  "
                  f"warm tail acc {1 - R_warm/max(len(tail),1):.4f}  "
                  f"implied multiplier ~ {mult:.2f}", flush=True)
            results['experts'][llm]['taus'][str(tau)] = {
                'C_E': C_E, 'R_warm_tail': R_warm, 'implied_multiplier': mult,
                'warm_tail_acc': 1 - R_warm / max(len(tail), 1)}
            with open(args.out, 'w') as f:
                json.dump(results, f, indent=2)

    print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
