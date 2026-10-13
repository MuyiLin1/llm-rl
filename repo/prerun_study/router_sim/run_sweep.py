"""Grid sweep for the Router on the Theorem thm:transfer-separation instance.

One *cell* = (setting, k, T, design); all policies and all c values of a cell are simulated together
on the same type sequences (common random numbers). After each cell its rows (one per policy x c)
are appended to results.csv and flushed, and progress.json is rewritten, so a crash loses at most
the cell in flight. --resume skips cells already complete in results.csv.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
import zlib
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import METRICS, gen_types, make_configs, simulate  # noqa: E402

# (name, p, q, f, omega, lam, L)
BASE_SETTINGS = [
    ('s1', 0.3, 0.05, 0.0, 0.135, 0.1, 2.0),
    ('s2', 0.5, 0.1, 0.0, 0.225, 0.1, 2.0),
    ('s3', 0.2, 0.0, 0.0, 0.09, 0.1, 2.0),
    ('fgt', 0.3, 0.1, 0.25, 0.135, 1.0, 2.0),     # f > p - q: expert B more accurate
]
LARGE_OMEGA = 2.0   # = L/(L-1) for L=2: beta(L-1)/sqrt(lam) >= 1 and omega > p (valid-radius regime)


def settings_list(large=True):
    out = list(BASE_SETTINGS)
    if large:
        out += [(n + '_bigR', p, q, f, LARGE_OMEGA, lam, L) for (n, p, q, f, _, lam, L) in BASE_SETTINGS]
    return out


def c_values(lam, L):
    return [lam / L ** 2 * 0.5, 0.05, 1.0, 10.0, 100.0, 1000.0]


def c_label(c, lam, L):
    return 'never' if abs(c - lam / L ** 2 * 0.5) < 1e-15 else f'{c:g}'


def build_grid(a):
    if a.quick:
        sets = [s for s in settings_list() if s[0] in ('s1', 'fgt', 's1_bigR')]
        ks, Ts, designs = [1, 2], [500], ['scripted', 'adaptive']
        seeds = {500: a.seeds or 50}
    else:
        sets = settings_list(not a.no_large)
        ks = a.k
        Ts = a.T
        designs = a.designs
        seeds = {T: (a.seeds or (500 if T <= 2000 else 200)) for T in Ts}
    cells = []
    for (name, p, q, f, om, lam, L) in sets:
        for k in ks:
            for T in Ts:
                for design in designs:
                    cid = f'{name}|k={k}|T={T}|{design}'
                    cells.append(dict(cell_id=cid, setting=name, p=p, q=q, f=f, omega=om, lam=lam, L=L,
                                      k=k, T=T, design=design, seeds=seeds[T]))
    return cells


def bound_of(p, omega, lam, k):
    h = k * (1 - 2 * omega) - omega * lam
    if omega < p and h > 0:
        return h, 1 - math.exp(-8 * (p - omega) * h)
    return h, float('nan')


def mse(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float('nan')


def run_cell(cell, base_seed):
    p, q, f, om, lam, L, k, T, S, design = (cell[x] for x in ('p', 'q', 'f', 'omega', 'lam', 'L', 'k', 'T', 'seeds', 'design'))
    cv = c_values(lam, L)
    cfgs = make_configs(cv)
    rng = np.random.default_rng([base_seed, zlib.crc32(cell['cell_id'].encode())])
    types = gen_types(rng, T, k, S, p, q, f, design)
    t0 = time.time()
    out = simulate(types, cfgs, p, q, f, om, lam, L, k, design)
    wall = time.time() - t0
    h, bnd = bound_of(p, om, lam, k)
    idx = {(c.policy, None if c.kind != 'router' else c.c): i for i, c in enumerate(cfgs)}
    ic = idx[('cold', None)]
    cold_reg = out['regret'][ic]
    cold_rew = out['reward'][ic]
    beta = om * math.sqrt(lam) / L
    rows = []
    for i, cfg in enumerate(cfgs):
        row = {k_: v for k_, v in cell.items()}
        row.update(beta=beta, policy=cfg.policy, kind=cfg.kind, expert='-AB'[cfg.expert + 1],
                   noreward=int(cfg.noreward),
                   c='' if cfg.kind != 'router' else repr(cfg.c),
                   c_label='' if cfg.kind != 'router' else c_label(cfg.c, lam, L),
                   h=h, bound=bnd, tail_acc_A=p, tail_acc_B=q + f)
        for m in METRICS:
            row[m + '_mean'], row[m + '_se'] = mse(out[m][i])
        row['routed_frac_mean'] = row['routed_mean'] / T
        row['regret_min'] = float(out['regret'][i].min())
        row['regret_max'] = float(out['regret'][i].max())
        row['cold_regret_mean'] = float(cold_reg.mean())
        row['cold_reward_mean'] = float(cold_rew.mean())
        rm = row['regret_mean']
        row['ratio_cold_over'] = row['cold_regret_mean'] / rm if rm > 0 else float('inf')
        row['identical_to_cold'] = int(np.array_equal(out['regret'][i], cold_reg) and
                                       np.array_equal(out['reward'][i], cold_rew) and
                                       np.array_equal(out['nD'][i], out['nD'][ic]) and
                                       np.array_equal(out['nR'][i], out['nR'][ic]))
        # paired comparison against the hybrid with the same expert (and same noreward flag)
        if cfg.kind == 'router':
            hname = cfg.policy.replace('router', 'hybrid')
            ih = idx[(hname, None)]
            d = out['regret'][i] - out['regret'][ih]
            row['hyb_regret_mean'] = float(out['regret'][ih].mean())
            row['hyb_settled_tau_rate'] = float(out['settled_tau'][ih].mean())
            row['hyb_settled_half_rate'] = float(out['settled_half'][ih].mean())
            row['minus_hyb_regret_mean'], row['minus_hyb_regret_se'] = mse(d)
            row['identical_to_hyb'] = int(np.array_equal(out['regret'][i], out['regret'][ih]) and
                                          np.array_equal(out['reward'][i], out['reward'][ih]) and
                                          np.array_equal(out['nD'][i], out['nD'][ih]) and
                                          np.array_equal(out['nR'][i], out['nR'][ih]))
        else:
            for x in ('hyb_regret_mean', 'hyb_settled_tau_rate', 'hyb_settled_half_rate',
                      'minus_hyb_regret_mean', 'minus_hyb_regret_se', 'identical_to_hyb'):
                row[x] = ''
        row['cell_wall_s'] = wall
        row['n_rows_in_cell'] = len(cfgs)
        rows.append(row)
    return rows, wall


def load_done(csv_path):
    """cell_ids complete in csv; rewrites the csv without partially written cells."""
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return set()
    cnt = {}
    for r in rows:
        cnt[r['cell_id']] = cnt.get(r['cell_id'], 0) + 1
    done = {c for c in cnt if all(int(r['n_rows_in_cell']) == cnt[c] for r in rows if r['cell_id'] == c)}
    if len(done) < len(cnt):
        keep = [r for r in rows if r['cell_id'] in done]
        with open(csv_path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(keep)
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'full'))
    ap.add_argument('--quick', action='store_true', help='tiny smoke-test grid (<1 min)')
    ap.add_argument('--resume', action='store_true', help='skip cells already complete in results.csv')
    ap.add_argument('--seeds', type=int, default=None, help='seeds per cell (default 500 at T<=2000, 200 above)')
    ap.add_argument('--k', type=int, nargs='+', default=[1, 2, 4, 8])
    ap.add_argument('--T', type=int, nargs='+', default=[2000, 20000])
    ap.add_argument('--designs', nargs='+', default=['scripted', 'adaptive'])
    ap.add_argument('--no-large', action='store_true', help='skip the large-radius (omega=2) settings')
    ap.add_argument('--base-seed', type=int, default=20261009)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    csv_path = os.path.join(a.out, 'results.csv')
    prog_path = os.path.join(a.out, 'progress.json')
    log_path = os.path.join(a.out, 'run.log')
    cells = build_grid(a)
    if os.path.exists(csv_path) and not a.resume:
        sys.exit(f'{csv_path} exists; use --resume to continue it or choose another --out')
    done = load_done(csv_path) if a.resume else set()
    todo = [c for c in cells if c['cell_id'] not in done]
    cost = lambda c: c['T'] * c['seeds']
    total_cost = sum(cost(c) for c in todo)
    logf = open(log_path, 'a')

    def log(msg):
        line = f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}'
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()

    start = time.time()
    start_iso = datetime.now().isoformat(timespec='seconds')
    state = dict(pid=os.getpid(), start_time=start_iso, status='running', cells_total=len(cells),
                 cells_done=len(cells) - len(todo), current_cell=None, elapsed_s=0.0, eta_s=None,
                 out=a.out, argv=sys.argv)

    def write_prog():
        tmp = prog_path + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(state, fh, indent=1)
        os.replace(tmp, prog_path)

    log(f'start pid={os.getpid()} cells total={len(cells)} todo={len(todo)} (resumed {len(done)}) out={a.out}')
    write_prog()
    fieldnames = None
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, newline='') as fh:
            fieldnames = next(csv.reader(fh))
    done_cost = 0
    try:
        for c in todo:
            state['current_cell'] = c['cell_id']
            write_prog()
            rows, wall = run_cell(c, a.base_seed)
            new_file = fieldnames is None
            if new_file:
                fieldnames = list(rows[0].keys())
            with open(csv_path, 'a', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=fieldnames)
                if new_file:
                    w.writeheader()
                w.writerows(rows)
                fh.flush()
                os.fsync(fh.fileno())
            done_cost += cost(c)
            state['cells_done'] += 1
            el = time.time() - start
            state['elapsed_s'] = el
            state['eta_s'] = el / done_cost * (total_cost - done_cost)
            write_prog()
            g = {r['policy'] + ('@' + r['c_label'] if r['c_label'] else ''): r for r in rows}
            ra, rb = g.get('router_A@100', {}), g.get('router_B@100', {})
            log(f"done {state['cells_done']}/{len(cells)} {c['cell_id']} seeds={c['seeds']} {wall:.1f}s | "
                f"cold {g['cold']['regret_mean']:.1f} hybA {g['hybrid_A']['regret_mean']:.1f} "
                f"(settle {g['hybrid_A']['settled_tau_mean']:.3f}) hybB {g['hybrid_B']['regret_mean']:.1f} | "
                f"c=100: RA {ra.get('regret_mean', float('nan')):.1f} (settle {ra.get('settled_half_mean', float('nan')):.3f}, "
                f"routed {ra.get('routed_mean', float('nan')):.0f}) RB {rb.get('regret_mean', float('nan')):.1f} "
                f"(routed {rb.get('routed_mean', float('nan')):.0f}) | ETA {state['eta_s']:.0f}s")
        state['status'] = 'finished'
        state['current_cell'] = None
        state['eta_s'] = 0.0
        state['elapsed_s'] = time.time() - start
        write_prog()
        log(f'finished in {state["elapsed_s"]:.0f}s; rows in {csv_path}')
    except BaseException as e:
        state['status'] = f'error: {type(e).__name__}: {e}'
        write_prog()
        log(f'ERROR {type(e).__name__}: {e}')
        raise


if __name__ == '__main__':
    main()
