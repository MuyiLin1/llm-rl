"""Decide the confidence parameter c from the tuning seeds, then report what is
still needed on the evaluation seeds.

Protocol (the one the paper follows):
    SELECT c on seeds 11-20      -- tuning block, never reported
    EVALUATE on seeds 1-10       -- the block every table reports

For each configuration family this script
  1. averages Router accuracy over the tuning seeds at every c in the sweep,
  2. picks the c with the highest mean,
  3. compares it with the c the paper currently reports,
  4. lists the evaluation runs at the winning c that do not yet exist,
  5. prints a ready-to-run command for the missing ones.

Usage:
    python pick_best_c.py --logdir study
"""
import argparse, glob, json, os, re
import statistics as st

FAMILIES = {                       # family -> (model_family, slim file, max_samples, c used in the paper)
    'amazonv1': ('v1', 'amazon_v1_slim.npz', 50000, '100.0'),
    'amazonv3': ('v3', 'amazon_v3_slim.npz', 50000, '0.1'),
    'wikiv1':   ('v1', 'wiki_v1_slim.npz',       0, '100.0'),
    'wikiv3':   ('v3', 'wiki_v3_slim.npz',       0, '10.0'),
}
EXPERTS = ['small', 'base', 'large']


def collect(logdir, family, seeds):
    """{c: {expert: {seed: (router, cold, llm)}}} from finished runs."""
    out = {}
    pat = os.path.join(logdir, f'results_algo1_{family}_*_c*_pfull_s*.json')
    for p in glob.glob(pat):
        m = re.search(rf'results_algo1_{family}_v\d_(\w+?)_c([\d.]+)_pfull_s(\d+)\.json$',
                      os.path.basename(p))
        if not m:
            continue
        expert, c, seed = m.group(1), m.group(2), int(m.group(3))
        if expert not in EXPERTS or seed not in seeds:
            continue
        try:
            h = json.load(open(p))
        except Exception:
            continue
        if not h.get('step') or h['step'][-1] == 0:
            continue
        out.setdefault(c, {}).setdefault(expert, {})[seed] = (
            h['hybrid'][-1], h['linucb_cold'][-1], h['llm'][-1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logdir', default='study')
    ap.add_argument('--tune_seeds', type=int, nargs='+', default=list(range(11, 21)))
    ap.add_argument('--eval_seeds', type=int, nargs='+', default=list(range(1, 11)))
    ap.add_argument('--min_seeds', type=int, default=8,
                    help='a c value is only eligible if this many tuning seeds finished')
    a = ap.parse_args()

    todo = []
    for fam, (mf, slim, maxs, paper_c) in FAMILIES.items():
        tune = collect(a.logdir, fam, set(a.tune_seeds))
        ev = collect(a.logdir, fam, set(a.eval_seeds))
        print(f"\n=== {fam} ===")
        if not tune:
            print("  no tuning runs yet"); continue
        rows = []
        for c, byexp in sorted(tune.items(), key=lambda kv: float(kv[0])):
            accs, ns = [], []
            for e in EXPERTS:
                d = byexp.get(e, {})
                ns.append(len(d))
                if d:
                    accs.append(st.mean(v[0] for v in d.values()))
            if not accs or min(ns) < a.min_seeds:
                print(f"  c={c:<8} incomplete ({ns} seeds) -- not eligible")
                continue
            rows.append((c, st.mean(accs), ns))
        if not rows:
            print("  nothing eligible yet"); continue
        print(f"  {'c':<10}{'mean Router acc over experts':>30}{'seeds':>10}")
        for c, m, ns in rows:
            print(f"  {c:<10}{m*100:>29.2f}%{min(ns):>10}")
        best = max(rows, key=lambda r: r[1])[0]
        same = abs(float(best) - float(paper_c)) < 1e-12
        print(f"  --> best c on tuning seeds: {best}   (paper currently reports c={paper_c})"
              f"   {'MATCH, no re-run needed' if same else '*** DIFFERENT ***'}")
        have = ev.get(best, {})
        missing = [(e, s) for e in EXPERTS for s in a.eval_seeds if s not in have.get(e, {})]
        if missing:
            print(f"  evaluation runs missing at c={best}: {len(missing)}")
            todo.append((fam, mf, slim, maxs, best, missing))
        else:
            print(f"  evaluation runs at c={best}: all {len(a.eval_seeds)} seeds present for every expert")

    if todo:
        print("\n\n=== commands for the missing evaluation runs ===")
        for fam, mf, slim, maxs, c, missing in todo:
            extra = f"--max_samples {maxs} " if maxs else ""
            print(f"\n# {fam}: {len(missing)} runs at c={c}")
            for e, s in missing:
                print(f"python experiment_algorithm1.py --slim {slim} --dataset_name {fam} "
                      f"--model_family {mf} --llm_type {e} --seed {s} --llm_confidence {c} "
                      f"--passive_full --alpha_decay 0.9985 --decay_per round {extra}--resume")
    else:
        print("\n\nNothing to re-run: every family's winning c already has all evaluation seeds.")


if __name__ == '__main__':
    main()
