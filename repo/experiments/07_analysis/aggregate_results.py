"""Aggregate the full-horizon dry run and the 10-seed AmazonCat experiment."""
import glob, json, os, re
import statistics as st

EXPERTS = ['small', 'base', 'large']
# Observed accuracy multipliers, 10 seeds, unified runner (Table tab:amazon-results).
OBS = {'small': 1.71, 'base': 2.72, 'large': 3.29}


def agg_dryrun(d='dryfull'):
    cold = {}
    for f in glob.glob(os.path.join(d, 'cold_s*.json')):
        r = json.load(open(f))
        if r.get('complete'):
            cold[r['seed']] = r
    warm = {}
    for f in glob.glob(os.path.join(d, 'warm_*.json')):
        r = json.load(open(f))
        if r.get('complete'):
            warm[(r['llm_type'], r['tau'], r['seed'])] = r
    print(f"=== FULL-HORIZON DRY RUN (T'=50,000) ===")
    accs = [cold[s]['acc'] for s in sorted(cold)]
    print(f"cold LinUCB in the dry run: {st.mean(accs)*100:.2f}% +- {st.stdev(accs)*100:.2f}%"
          f"   (real experiment: 9.5%)   seeds={len(accs)}\n")
    taus = sorted({k[1] for k in warm})
    for tau in taus:
        print(f"--- tau = {tau} ---")
        print(f"{'expert':8}{'dry-run multiplier':>22}{'warm tail acc':>16}"
              f"{'observed':>11}   per-seed")
        means = {}
        for m in EXPERTS:
            mults, tacc = [], []
            for s in sorted(cold):
                w = warm.get((m, tau, s))
                if not w:
                    continue
                mults.append(cold[s]['mistakes'] / (w['C_E'] + w['R_warm_tail']))
                tacc.append(w['warm_tail_acc'])
            means[m] = (st.mean(mults), st.stdev(mults) if len(mults) > 1 else 0.0)
            per = " ".join(f"{x:.2f}" for x in mults)
            print(f"{m:8}{means[m][0]:>13.3f} +-{means[m][1]:<7.3f}"
                  f"{st.mean(tacc)*100:>14.2f}%{OBS[m]:>9.2f}x   {per}")
        best = max(means, key=lambda k: means[k][0])
        others = max(means[k][0] for k in means if k != 'large')
        sep = means['large'][0] - others
        print(f"  ranks large first: {'YES' if best == 'large' else 'NO (' + best + ')'}"
              f"    large - next best = {sep:+.3f}"
              f"    (large's own sd = {means['large'][1]:.3f})")
        print(f"  separated by more than 1 sd: "
              f"{'YES' if sep > means['large'][1] else 'NO'}\n")


def agg_seeds(pattern='study/results_algo1_amazonv1_v1_*_c100.0_pfull_s*.json',
              extra='results_algo1_amazonv1_v1_*_c100.0_pfull_s*.json'):
    runs = {}
    for f in glob.glob(pattern) + glob.glob(extra):
        m = re.search(r'v1_(small|base|large)_c100\.0_pfull_s(\d+)\.json', f)
        if not m:
            continue
        h = json.load(open(f))
        if h['step'][-1] < h['config']['num_samples']:
            print(f"  SKIPPING incomplete: {f}")
            continue
        runs[(m.group(1), int(m.group(2)))] = h
    seeds = sorted({k[1] for k in runs})
    print(f"=== AmazonCat, alpha decay 0.9985, {len(seeds)} seeds: {seeds} ===\n")
    cold = [runs[('large', s)]['linucb_cold'][-1] for s in seeds if ('large', s) in runs]
    print(f"cold LinUCB: {st.mean(cold)*100:.2f}% +- {st.stdev(cold)*100:.2f}%"
          f"   range {min(cold)*100:.2f}-{max(cold)*100:.2f}%\n")
    print(f"{'expert':8}{'LLM':>7}{'Router (mean+-sd)':>21}{'mult':>8}   per-seed Router")
    for m in EXPERTS:
        rs = [runs[(m, s)]['hybrid'][-1] for s in seeds if (m, s) in runs]
        ll = [runs[(m, s)]['llm'][-1] for s in seeds if (m, s) in runs]
        cc = [runs[(m, s)]['linucb_cold'][-1] for s in seeds if (m, s) in runs]
        mult = st.mean(rs) / st.mean([max(l, c) for l, c in zip(ll, cc)])
        print(f"{m:8}{st.mean(ll)*100:>6.2f}%{st.mean(rs)*100:>13.2f} +-{st.stdev(rs)*100:<6.2f}"
              f"{mult:>7.2f}x   " + " ".join(f"{x*100:.1f}" for x in rs))
    return runs, seeds


def agg_decomp(runs, seeds):
    import numpy as np
    print(f"\n=== decomposition A / V / B (10 seeds) ===")
    print(f"{'expert':8}{'A':>10}{'V (mean+-sd)':>20}{'B':>10}   V per seed")
    for m in EXPERTS:
        A, V, B = [], [], []
        for s in seeds:
            f = f'seeds/perround_algo1_amazon_v1_{m}_c100.0_pfull_s{s}.npz'
            if not os.path.exists(f):
                f = f'perround_algo1_amazon_v1_{m}_c100.0_pfull_s{s}.npz'
            if not os.path.exists(f):
                continue
            z = np.load(f)
            r, c, l = z['r_router'], z['r_cold'], z['r_llm']
            rt = z['routed'].astype(bool)
            A.append(float((r[rt] - c[rt]).sum()))
            V.append(float((r[~rt] - c[~rt]).sum()))
            B.append(float((r[~rt] - l[~rt]).sum()))
        if not V:
            print(f"{m:8} no per-round logs found")
            continue
        print(f"{m:8}{st.mean(A):>10.0f}{st.mean(V):>12.0f} +-{st.stdev(V):<6.0f}"
              f"{st.mean(B):>10.0f}   " + " ".join(f"{x:.0f}" for x in V))
        print(f"{'':8}{'':10}  V>0 on {sum(1 for x in V if x>0)}/{len(V)} seeds")


if __name__ == '__main__':
    agg_dryrun()
    print()
    runs, seeds = agg_seeds()
    agg_decomp(runs, seeds)
