"""Pre-runtime dry run over the FULL action set, one independent task per process.

Rewrite of dry_run_full.py for the full-horizon study. Two changes matter:

1. Task granularity. Within a seed, the cold pass and each expert's warm pass are
   completely independent simulations -- the warm pass starts from its own
   acquisition, not from cold's state. So a (seed, task) pair is the unit of work,
   which lets every pass run in its own process and lets a failed pass be retried
   alone instead of restarting a whole seed.

2. Reproducibility across decompositions. The evaluation stream and the
   acquisition draw come from separate, independently seeded generators, so a
   warm task produces exactly the same stream as the cold task of the same seed
   no matter which order tasks run in or which ones are retried.

Every task writes its JSON only after finishing, with "complete": true. A task
whose output file lacks that flag is treated as unfinished and re-run, so partial
results can never reach the aggregate.

  python dry_run_task.py --task cold --seed 1 --T_dry 50000
  python dry_run_task.py --task warm --seed 1 --llm_type large --tau 1600 --T_dry 50000
"""
import argparse, json, os, time
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


def streams(seed, pool, T_dry, tau_max):
    """Deterministic given the seed alone, independent of task decomposition."""
    stream = np.random.default_rng(seed).choice(pool, size=T_dry, replace=True)
    acq = np.random.default_rng(seed + 10_000).choice(pool, size=tau_max, replace=True)
    return stream, acq


def run_linucb(order, contexts, Z, ys, V_inv, b, beta, score, label, every=5000):
    mistakes = 0.0
    t0 = time.time()
    for i, s in enumerate(order):
        cx = contexts[s].astype(np.float64)
        a = score(V_inv, V_inv @ b, cx, beta)
        r = 1.0 if a in ys[s] else 0.0
        mistakes += 1.0 - r
        x = cx * Z[a]
        V_inv = sm_update(V_inv, x)
        b += r * x
        if (i + 1) % every == 0:
            print(f"    [{label}] {i+1}/{len(order)}  {(i+1)/(time.time()-t0):.1f}/s  "
                  f"acc {1 - mistakes/(i+1):.4f}", flush=True)
    return mistakes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=['cold', 'warm'])
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--llm_type', default=None, help='required for --task warm')
    ap.add_argument('--tau', type=int, default=1600, help='acquisition rounds (warm)')
    ap.add_argument('--slim', default='amazon_v1_slim.npz')
    ap.add_argument('--T_dry', type=int, default=50000)
    ap.add_argument('--tau_max', type=int, default=5000,
                    help='largest tau in the study; fixes the acquisition draw so '
                         'different taus are nested prefixes of one sequence')
    ap.add_argument('--lam', type=float, default=0.1)
    ap.add_argument('--beta', type=float, default=1.0)
    ap.add_argument('--online_T', type=int, default=50000)
    ap.add_argument('--outdir', default='dryfull')
    ap.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    args = ap.parse_args()

    if args.task == 'warm' and not args.llm_type:
        raise SystemExit("--task warm needs --llm_type")
    os.makedirs(args.outdir, exist_ok=True)
    name = (f"cold_s{args.seed}" if args.task == 'cold'
            else f"warm_{args.llm_type}_tau{args.tau}_s{args.seed}")
    out = os.path.join(args.outdir, name + '.json')
    if os.path.exists(out):
        try:
            if json.load(open(out)).get('complete'):
                print(f"{name}: already complete, skipping")
                return
        except Exception:
            pass

    z = np.load(args.slim)
    contexts, Z = z['contexts'], z['Z'].astype(np.float64)
    flat, offs = z['y_flat'], z['y_offsets']
    ys = [set(flat[offs[i]:offs[i+1]].tolist()) for i in range(len(offs) - 1)]
    K, d = Z.shape
    pool = np.arange(args.online_T, contexts.shape[0])
    if pool.size < 100:
        raise SystemExit("no held-out block; rebuild the slim file with a larger --n_keep")

    dev = pick_device(args.device)
    score = make_scorer(Z, dev, fast32=(dev != 'cpu'))
    stream, acq_all = streams(args.seed, pool, args.T_dry, args.tau_max)
    print(f"{name}: device={dev} actions={K} d={d} held-out={pool.size} "
          f"T_dry={args.T_dry}", flush=True)

    res = {'task': args.task, 'seed': args.seed, 'T_dry': args.T_dry,
           'lam': args.lam, 'beta': args.beta, 'device': dev, 'complete': False}
    t0 = time.time()

    if args.task == 'cold':
        m = run_linucb(stream, contexts, Z, ys, (1.0/args.lam)*np.eye(d),
                       np.zeros(d), args.beta, score, 'cold')
        res.update(mistakes=m, acc=1 - m/args.T_dry, rounds=int(args.T_dry))
        print(f"cold s{args.seed}: {m:.0f} mistakes, acc {1-m/args.T_dry:.4f}")
    else:
        lab = z[f'llm_{args.llm_type}']
        err = float(np.mean([0.0 if lab[s] in ys[s] else 1.0 for s in pool]))
        V_inv = (1.0/args.lam)*np.eye(d)
        b = np.zeros(d)
        for s in acq_all[:args.tau]:            # acquisition: play the expert
            cx = contexts[int(s)].astype(np.float64)
            a = int(lab[int(s)])
            r = 1.0 if a in ys[int(s)] else 0.0
            x = cx * Z[a]
            V_inv = sm_update(V_inv, x)
            b += r * x
        tail = stream[args.tau:]
        m = run_linucb(tail, contexts, Z, ys, V_inv, b, args.beta, score,
                       f"{args.llm_type} tau={args.tau}")
        res.update(llm_type=args.llm_type, tau=args.tau, held_out_err=err,
                   C_E=args.tau*err, R_warm_tail=m, rounds=int(len(tail)),
                   warm_tail_acc=1 - m/max(len(tail), 1))
        print(f"warm {args.llm_type} tau={args.tau} s{args.seed}: "
              f"C_E={args.tau*err:.0f} R_warm={m:.0f} acc {1-m/max(len(tail),1):.4f}")

    res['seconds'] = time.time() - t0
    res['complete'] = True
    tmp = out + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(res, f, indent=2)
    os.replace(tmp, out)                        # atomic: no half-written JSON
    print(f"wrote {out} in {res['seconds']/60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
