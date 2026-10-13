"""Status of a sweep: python3 status.py [--out results/full] [--watch N] [--cells M] [--c 100]"""
import argparse
import csv
import json
import os
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (TypeError, ValueError):
        return False


def hms(s):
    if s is None:
        return '?'
    s = int(s)
    return f'{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s'


def tail(path, n):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return fh.read().splitlines()[-n:]


def show(a):
    out = a.out
    pp = os.path.join(out, 'progress.json')
    if not os.path.exists(pp):
        print(f'no progress.json in {out}')
        return
    with open(pp) as fh:
        st = json.load(fh)
    is_alive = alive(st.get('pid'))
    if st.get('status') == 'finished':
        state = 'FINISHED'
    elif is_alive and st.get('status') == 'running':
        state = 'RUNNING'
    elif str(st.get('status', '')).startswith('error'):
        state = 'DIED-INCOMPLETE (' + st['status'] + ')'
    else:
        state = 'DIED-INCOMPLETE'
    done, tot = st.get('cells_done', 0), st.get('cells_total', 0)
    el = st.get('elapsed_s')
    if state == 'RUNNING':   # elapsed since last progress write
        try:
            el = (datetime.now() - datetime.fromisoformat(st['start_time'])).total_seconds()
        except Exception:
            pass
    print(f'[{datetime.now():%H:%M:%S}] {out}')
    print(f'state: {state}   pid {st.get("pid")} ({"alive" if is_alive else "not alive"})   started {st.get("start_time")}')
    print(f'cells: {done}/{tot} ({100 * done / max(tot, 1):.1f}%)   elapsed {hms(el)}   ETA {hms(st.get("eta_s"))}'
          f'   current: {st.get("current_cell")}')
    if state.startswith('DIED'):
        print(f'resume with:  python3 {os.path.join(HERE, "run_sweep.py")} --resume --out {out}  (or ./run.sh {out} --resume)')
    print('--- log tail ---')
    for line in tail(os.path.join(out, 'run.log'), a.log):
        print(line[:220])
    cp = os.path.join(out, 'results.csv')
    if not os.path.exists(cp):
        return
    with open(cp, newline='') as fh:
        rows = list(csv.DictReader(fh))
    cells = []
    by = {}
    for r in rows:
        if r['cell_id'] not in by:
            cells.append(r['cell_id'])
            by[r['cell_id']] = {}
        by[r['cell_id']][(r['policy'], r['c_label'])] = r
    f = lambda r, k, fmt='.1f': format(float(r[k]), fmt) if r and r.get(k, '') != '' else '-'
    print(f'--- last {a.cells} finished cells, Router at c={a.c} (regret mean; settle = all-D second half) ---')
    print(f'{"cell":34} {"cold":>7} {"hybA":>7} {"hybB":>7} {"RtrA":>7} {"RtrB":>7} '
          f'{"setlA":>6} {"setlHA":>6} {"routA":>6} {"routB":>6} {"cold/A":>7} {"cold/B":>7}')
    for cid in cells[-a.cells:]:
        g = by[cid]
        ra, rb = g.get(('router_A', a.c)), g.get(('router_B', a.c))
        print(f'{cid:34} {f(g[("cold", "")], "regret_mean"):>7} {f(g[("hybrid_A", "")], "regret_mean"):>7} '
              f'{f(g[("hybrid_B", "")], "regret_mean"):>7} {f(ra, "regret_mean"):>7} {f(rb, "regret_mean"):>7} '
              f'{f(ra, "settled_half_mean", ".3f"):>6} {f(g[("hybrid_A", "")], "settled_tau_mean", ".3f"):>6} '
              f'{f(ra, "routed_mean", ".0f"):>6} {f(rb, "routed_mean", ".0f"):>6} '
              f'{f(ra, "ratio_cold_over"):>7} {f(rb, "ratio_cold_over", ".2f"):>7}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'full'))
    ap.add_argument('--watch', type=float, default=0, help='refresh every N seconds')
    ap.add_argument('--cells', type=int, default=8)
    ap.add_argument('--log', type=int, default=5, help='log lines to show')
    ap.add_argument('--c', default='100', help='c label for the summary table (never, 0.05, 1, 10, 100, 1000)')
    a = ap.parse_args()
    if a.watch > 0:
        try:
            while True:
                os.system('clear')
                show(a)
                time.sleep(a.watch)
        except KeyboardInterrupt:
            pass
    else:
        show(a)


if __name__ == '__main__':
    main()
