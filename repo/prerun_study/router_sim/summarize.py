"""Summarize a finished (or partial) sweep into summary.md and table.tex.

python3 summarize.py [--out results/full] [--c 100] [--T-table 20000]
"""
import argparse
import csv
import math
import os
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
C_ORDER = ['never', '0.05', '1', '10', '100', '1000']


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def load(out):
    with open(os.path.join(out, 'results.csv'), newline='') as fh:
        rows = list(csv.DictReader(fh))
    idx = {}
    for r in rows:
        idx[(r['setting'], int(r['k']), int(r['T']), r['design'], r['policy'], r['c_label'])] = r
    return rows, idx


def fmt(x, d=1):
    x = num(x)
    if math.isnan(x):
        return '-'
    if math.isinf(x):
        return 'inf'
    return f'{x:.{d}f}'


def md_table(header, rows):
    s = '| ' + ' | '.join(header) + ' |\n|' + '|'.join(['---'] * len(header)) + '|\n'
    for r in rows:
        s += '| ' + ' | '.join(str(x) for x in r) + ' |\n'
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'full'))
    ap.add_argument('--c', default='100', help='c label used in the headline comparisons and the LaTeX table')
    ap.add_argument('--T-table', type=int, default=None, help='T used in the LaTeX table (default: largest)')
    a = ap.parse_args()
    rows, idx = load(a.out)
    settings = list(OrderedDict.fromkeys(r['setting'] for r in rows))
    sinfo = {r['setting']: r for r in rows}
    ks = sorted({int(r['k']) for r in rows})
    Ts = sorted({int(r['T']) for r in rows})
    designs = list(OrderedDict.fromkeys(r['design'] for r in rows))
    cs = [c for c in C_ORDER if any(r['c_label'] == c for r in rows)]
    base = [s for s in settings if not s.endswith('_bigR')]
    big = [s for s in settings if s.endswith('_bigR')]
    g = lambda s, k, T, d, pol, c='': idx.get((s, k, T, d, pol, c))
    C = a.c
    md = []
    P = md.append

    def sdesc(s):
        r = sinfo[s]
        return (f"{s}: p={num(r['p']):g} q={num(r['q']):g} f={num(r['f']):g} omega={num(r['omega']):g} "
                f"lam={num(r['lam']):g} L={num(r['L']):g}")

    P(f'# Router on the transfer-separation instance\n\nSource: `{os.path.abspath(a.out)}/results.csv` '
      f'({len(rows)} rows). Headline c = {C}. Settings:\n')
    for s in settings:
        P(f'- {sdesc(s)}')
    P('\nDefinitions: regret = pseudo-regret vs theta* (gaps D 0, R p-q, fresh p-f); reward = number of '
      'reward-1 rounds; "settle (half)" = plays D on every round of the second half; "settle (tau)" = plays D on '
      'every round after tau = 4k (the theorem\'s event for H_tau). Router widths: LinUCB 1/sqrt(lam+n) for D/R, '
      'L/sqrt(lam) for an untried fresh arm; expert 1/sqrt(c) (L/sqrt(c) for fresh). c=never is lam/(2L^2).\n')

    # ---------------------------------------------------------------- 0 sanity
    P('## 0. Sanity: c = never-route\n')
    bad = []
    n_ok = 0
    for r in rows:
        if r['kind'] != 'router' or r['c_label'] != 'never':
            continue
        if r['design'] == 'adaptive':
            ok = r['identical_to_cold'] == '1' and num(r['routed_mean']) == 0
            want = 'cold'
        else:
            ok = r['identical_to_hyb'] == '1' and num(r['routed_after_tau_mean']) == 0
            want = 'hybrid'
        n_ok += ok
        if not ok:
            bad.append(f"{r['cell_id']} {r['policy']} != {want}")
    P(f'Adaptive design: Router(never) must equal cold on every seed; scripted design: it must equal H_tau '
      f'(same expert, same noreward flag). Passed {n_ok} / {n_ok + len(bad)} (cell, policy) pairs.' +
      ('' if not bad else '\n\nFAILURES:\n' + '\n'.join('- ' + b for b in bad)) + '\n')
    # theorem (ii)/(iii) checks: LinUCB-chosen D plays = nD - routed_nD
    chk = []
    for r in rows:
        if r['policy'] in ('hybrid_B', 'hybrid_A_noreward', 'hybrid_B_noreward') and not r['setting'].endswith('_bigR') \
                and r['design'] == 'scripted':
            chk.append(num(r['nD_mean']) - num(r['routed_nD_mean']))
    if chk:
        P(f'Theorem (ii)/(iii) on scripted cells: max over cells of mean LinUCB-chosen D plays (nD - routed D) for '
          f'hybrid_B, hybrid_A_noreward, hybrid_B_noreward = {max(chk):.3g} (theorem: 0).\n')
    cold_ok = [abs(num(r['regret_mean']) - (num(r['p']) - num(r['f'])) * int(r['T'])) < 1e-6
               for r in rows if r['policy'] == 'cold' and not r['setting'].endswith('_bigR')]
    if cold_ok:
        P(f'Cold regret = (p-f)T in {sum(cold_ok)}/{len(cold_ok)} small-radius cells.\n')

    # ---------------------------------------------------------------- 1 settling
    P(f'## 1. Does Router-A settle at least as often as H_tau-A? (scripted)\n')
    for T in Ts:
        hdr = ['setting', 'k', 'bound', 'H_A settle(tau)', 'H_A settle(half)'] + [f'R_A c={c}' for c in cs]
        tab = []
        for s in base:
            for k in ks:
                h = g(s, k, T, 'scripted', 'hybrid_A')
                if not h:
                    continue
                line = [s, k, fmt(h['bound'], 3), fmt(h['settled_tau_mean'], 3), fmt(h['settled_half_mean'], 3)]
                for c in cs:
                    r = g(s, k, T, 'scripted', 'router_A', c)
                    if not r:
                        line.append('-'); continue
                    flag = ''
                    if num(r['settled_half_mean']) + 1e-12 < num(h['settled_half_mean']) - 2 * math.sqrt(
                            max(num(h['settled_half_mean']) * (1 - num(h['settled_half_mean'])), 1e-12) / int(h['seeds'])):
                        flag = ' (<H)'
                    line.append(fmt(r['settled_half_mean'], 3) + flag)
                tab.append(line)
        if tab:
            P(f'T = {T}. Router settle = all-D second half. "(<H)" marks Router-A settling less often than H_A '
              f'(second-half definition) by more than 2 binomial s.e.\n')
            P(md_table(hdr, tab))
    # routing volume of router A
    P(f'Router-A routed rounds (mean; after tau in brackets), scripted:\n')
    for T in Ts:
        tab = []
        for s in base:
            for k in ks:
                line = [s, k]
                for c in cs:
                    r = g(s, k, T, 'scripted', 'router_A', c)
                    line.append('-' if not r else f"{fmt(r['routed_mean'], 0)} ({fmt(r['routed_after_tau_mean'], 0)})")
                if len(line) > 2 and any(x != '-' for x in line[2:]):
                    tab.append(line)
        if tab:
            P(f'T = {T}\n')
            P(md_table(['setting', 'k'] + [f'c={c}' for c in cs], tab))

    # ---------------------------------------------------------------- 2 router B
    P('## 2. Does Router-B become the expert? (routed fraction, scripted)\n')
    for T in Ts:
        tab = []
        for s in base:
            for k in ks:
                line = [s, k]
                for c in cs:
                    r = g(s, k, T, 'scripted', 'router_B', c)
                    line.append('-' if not r else f"{fmt(r['routed_frac_mean'], 3)} / D2={fmt(r['fracD_half_mean'], 2)}")
                tab.append(line)
        P(f'T = {T}. Entry = routed fraction / fraction of second-half rounds on D.\n')
        P(md_table(['setting', 'k'] + [f'c={c}' for c in cs], tab))

    # ---------------------------------------------------------------- 3 ratios
    P(f'## 3. Multipliers cold / Router-A vs cold / Router-B (c = {C})\n')
    for d in designs:
        tab = []
        for s in settings:
            for T in Ts:
                for k in ks:
                    ra, rb = g(s, k, T, d, 'router_A', C), g(s, k, T, d, 'router_B', C)
                    ha, hb, co = g(s, k, T, d, 'hybrid_A'), g(s, k, T, d, 'hybrid_B'), g(s, k, T, d, 'cold')
                    if not (ra and rb and co):
                        continue
                    tab.append([s, T, k, fmt(co['regret_mean']),
                                f"{fmt(ra['regret_mean'])}±{fmt(ra['regret_se'])}", f"{fmt(rb['regret_mean'])}±{fmt(rb['regret_se'])}",
                                fmt(ha['regret_mean']), fmt(hb['regret_mean']),
                                fmt(ra['ratio_cold_over'], 2), fmt(rb['ratio_cold_over'], 2),
                                fmt(ha['ratio_cold_over'], 2), fmt(hb['ratio_cold_over'], 2)])
        P(f'Design {d}.\n')
        P(md_table(['setting', 'T', 'k', 'cold', 'Router-A', 'Router-B', 'H_A', 'H_B',
                    'cold/RA', 'cold/RB', 'cold/H_A', 'cold/H_B'], tab))
    P('Ratios at every c (scripted, largest T, k=2 if present):\n')
    kk = 2 if 2 in ks else ks[0]
    tab = []
    for s in settings:
        line = [s]
        for c in cs:
            ra, rb = g(s, kk, Ts[-1], 'scripted', 'router_A', c), g(s, kk, Ts[-1], 'scripted', 'router_B', c)
            line.append('-' if not ra else f"{fmt(ra['ratio_cold_over'], 2)} / {fmt(rb['ratio_cold_over'], 2)}")
        tab.append(line)
    P(md_table(['setting'] + [f'c={c} (A/B)' for c in cs], tab))

    # ---------------------------------------------------------------- 4 f > p - q
    fg = [s for s in settings if s.startswith('fgt')]
    if fg:
        P('## 4. f > p - q: pseudo-regret vs realized reward (does the ranking flip?)\n\nExpert accuracies after tau: A = p, B = q + f. FLIP = the two measures rank the pair differently.\n')
        tab = []
        for s in fg:
            for d in designs:
                for T in Ts:
                    for k in ks:
                        co = g(s, k, T, d, 'cold')
                        if not co:
                            continue
                        pairs = [('H', ('hybrid_A', ''), ('hybrid_B', ''))] + \
                                [(f'R c={c}', ('router_A', c), ('router_B', c)) for c in cs if c != 'never']
                        for lab, pa, pb in pairs:
                            A, B = g(s, k, T, d, *pa), g(s, k, T, d, *pb)
                            if not (A and B):
                                continue
                            reg_says = 'A' if num(A['regret_mean']) < num(B['regret_mean']) else 'B'
                            rew_says = 'A' if num(A['reward_mean']) > num(B['reward_mean']) else 'B'
                            tab.append([s, d, T, k, lab, fmt(A['regret_mean']), fmt(B['regret_mean']),
                                        fmt(A['reward_mean']), fmt(B['reward_mean']), fmt(co['reward_mean']),
                                        reg_says, rew_says, 'FLIP' if reg_says != rew_says else ''])
        P(md_table(['setting', 'design', 'T', 'k', 'pair', 'regret A', 'regret B', 'reward A', 'reward B',
                    'reward cold', 'better by regret', 'better by reward', ''], tab))

    # ---------------------------------------------------------------- 5 scripted vs adaptive
    if 'scripted' in designs and 'adaptive' in designs:
        P(f'## 5. Scripted vs adaptive window (c = {C})\n')
        tab = []
        for s in settings:
            for T in Ts:
                for k in ks:
                    rs = [g(s, k, T, d, p_, C) for d in ('scripted', 'adaptive') for p_ in ('router_A', 'router_B')]
                    hs = [g(s, k, T, d, 'hybrid_A') for d in ('scripted', 'adaptive')]
                    if not all(rs) or not all(hs):
                        continue
                    tab.append([s, T, k,
                                f"{fmt(rs[0]['regret_mean'])} / {fmt(rs[2]['regret_mean'])}",
                                f"{fmt(rs[0]['settled_half_mean'], 3)} / {fmt(rs[2]['settled_half_mean'], 3)}",
                                f"{fmt(rs[0]['routed_mean'], 0)} / {fmt(rs[2]['routed_mean'], 0)}",
                                f"{fmt(rs[1]['regret_mean'])} / {fmt(rs[3]['regret_mean'])}",
                                f"{fmt(hs[0]['regret_mean'])} / {fmt(hs[1]['regret_mean'])}",
                                f"{fmt(hs[0]['settled_tau_mean'], 3)} / {fmt(hs[1]['settled_tau_mean'], 3)}"])
        P('Each entry: scripted / adaptive.\n')
        P(md_table(['setting', 'T', 'k', 'RA regret', 'RA settle', 'RA routed', 'RB regret', 'H_A regret', 'H_A settle(tau)'], tab))

    # ---------------------------------------------------------------- 6 noreward
    P(f'## 6. Coverage only (routed rewards withheld), scripted, c = {C}\n')
    tab = []
    for s in base:
        for T in Ts:
            for k in ks:
                r1, r2 = g(s, k, T, 'scripted', 'router_A_noreward', C), g(s, k, T, 'scripted', 'router_B_noreward', C)
                h1 = g(s, k, T, 'scripted', 'hybrid_A_noreward')
                if not (r1 and r2 and h1):
                    continue
                tab.append([s, T, k, fmt(h1['regret_mean']), fmt(num(h1['nD_mean']) - num(h1['routed_nD_mean']), 2),
                            fmt(r1['regret_mean']), fmt(r1['settled_half_mean'], 3), fmt(r2['regret_mean']),
                            fmt(g(s, k, T, 'scripted', 'cold')['regret_mean'])])
    P(md_table(['setting', 'T', 'k', 'H_A(nr) regret', 'H_A(nr) LinUCB D plays', 'R_A(nr) regret',
                'R_A(nr) settle', 'R_B(nr) regret', 'cold'], tab))

    # ---------------------------------------------------------------- 7 large radius
    if big:
        P(f'## 7. Large radius (omega = {fmt(sinfo[big[0]]["omega"], 2)} > p; valid-radius regime)\n')
        tab = []
        for s in big:
            for d in designs:
                for T in Ts:
                    for k in ks:
                        co = g(s, k, T, d, 'cold')
                        if not co:
                            continue
                        line = [s, d, T, k, fmt(co['regret_mean']),
                                fmt(g(s, k, T, d, 'hybrid_A')['ratio_cold_over'], 2),
                                fmt(g(s, k, T, d, 'hybrid_B')['ratio_cold_over'], 2)]
                        for c in cs:
                            ra, rb = g(s, k, T, d, 'router_A', c), g(s, k, T, d, 'router_B', c)
                            line.append('-' if not ra else
                                        f"{fmt(ra['ratio_cold_over'], 2)}/{fmt(rb['ratio_cold_over'], 2)} "
                                        f"(rt {fmt(ra['routed_frac_mean'], 2)}/{fmt(rb['routed_frac_mean'], 2)})")
                        tab.append(line)
        P('Entries: cold/Router-A / cold/Router-B (routed fraction A/B).\n')
        P(md_table(['setting', 'design', 'T', 'k', 'cold', 'cold/H_A', 'cold/H_B'] + [f'c={c}' for c in cs], tab))

    with open(os.path.join(a.out, 'summary.md'), 'w') as fh:
        fh.write('\n'.join(md) + '\n')

    # ---------------------------------------------------------------- LaTeX
    Tt = a.T_table or Ts[-1]
    Ttx = f'{Tt:,}'.replace(',', '{,}')
    f0 = [s for s in base if num(sinfo[s]['f']) == 0]
    colspec = '@{}ll' + 'c' * len(ks) + '@{}'
    head = '$(p,q,\\omega)$ & & ' + ' & '.join(f'$k={k}$' for k in ks) + ' \\\\'
    L = []
    L.append('% generated by prerun_study/router_sim/summarize.py')
    L.append('\\begin{table}[h]\n\\centering')
    r0 = sinfo[f0[0]] if f0 else sinfo[base[0]]
    nseeds = g(f0[0], ks[0], Tt, 'scripted', 'cold')['seeds'] if f0 and g(f0[0], ks[0], Tt, 'scripted', 'cold') else '?'
    L.append(f'\\caption{{The Router on the instance of Theorem~\\ref{{thm:transfer-separation}} '
             f'(${nseeds}$ seeds per cell, $\\lambda={num(r0["lam"]):g}$, $L={num(r0["L"]):g}$, $T={Ttx}$, $f=0$, '
             f'expert confidence $c={C}$, scripted window). Rows: frequency with which $\\Hyb^A_\\tau$ settles on $D$ '
             f'after round $\\tau$; frequency with which the Router with $\\Exp_A$ plays $D$ on every round of the '
             f'second half; the bound $1-e^{{-8(p-\\omega)h}}$ of Theorem~\\ref{{thm:transfer-separation}}(iv); '
             f'fraction of rounds the Router with $\\Exp_B$ routes to the expert; and the multipliers '
             f'$R_T(\\Cold)/R_T$ of the two Routers.}}')
    L.append('\\label{tab:router-check}\n\\small')
    L.append(f'\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{head}\n\\midrule')
    for i, s in enumerate(f0):
        r = sinfo[s]
        lab = f'$({num(r["p"]):g},\\,{num(r["q"]):g},\\,{num(r["omega"]):g})$'
        def rowvals(fn):
            out = []
            for k in ks:
                try:
                    out.append(fn(k))
                except (TypeError, KeyError):
                    out.append('--')
            return ' & '.join(out)
        lines = [
            ('$\\Hyb^A_\\tau$ settles', lambda k: f"${num(g(s, k, Tt, 'scripted', 'hybrid_A')['settled_tau_mean']):.3f}$"),
            ('Router-$A$ settles', lambda k: f"${num(g(s, k, Tt, 'scripted', 'router_A', C)['settled_half_mean']):.3f}$"),
            ('bound', lambda k: (lambda b: '--' if math.isnan(b) else f'${b:.3f}$')(num(g(s, k, Tt, 'scripted', 'hybrid_A')['bound']))),
            ('Router-$B$ routed', lambda k: f"${num(g(s, k, Tt, 'scripted', 'router_B', C)['routed_frac_mean']):.3f}$"),
            ('mult.\\ Router-$A$', lambda k: f"${num(g(s, k, Tt, 'scripted', 'router_A', C)['ratio_cold_over']):.1f}$"),
            ('mult.\\ Router-$B$', lambda k: f"${num(g(s, k, Tt, 'scripted', 'router_B', C)['ratio_cold_over']):.2f}$"),
        ]
        for j, (name, fn) in enumerate(lines):
            first = f'\\multirow{{{len(lines)}}}{{*}}{{{lab}}}' if j == 0 else ''
            L.append(f'{first} & {name} & {rowvals(fn)} \\\\')
        if i < len(f0) - 1:
            L.append('\\midrule')
    L.append('\\bottomrule\n\\end{tabular}\n\\end{table}')
    # f > p - q table
    if fg:
        s = fg[0]
        r = sinfo[s]
        L.append('')
        L.append('\\begin{table}[h]\n\\centering')
        L.append(f'\\caption{{The case $f>p-q$ ($p={num(r["p"]):g}$, $q={num(r["q"]):g}$, $f={num(r["f"]):g}$, '
                 f'$\\omega={num(r["omega"]):g}$, $\\lambda={num(r["lam"]):g}$, $L={num(r["L"]):g}$, $T={Ttx}$, $c={C}$, '
                 f'scripted window): mean pseudo-regret and mean realized reward (number of rounds with reward $1$).}}')
        L.append('\\label{tab:router-fgt}\n\\small')
        L.append(f'\\begin{{tabular}}{{@{{}}ll{"c" * len(ks)}@{{}}}}\n\\toprule\n & & '
                 + ' & '.join(f'$k={k}$' for k in ks) + ' \\\\\n\\midrule')
        pols = [('$\\Cold$', 'cold', ''), ('$\\Hyb^A_\\tau$', 'hybrid_A', ''), ('$\\Hyb^B_\\tau$', 'hybrid_B', ''),
                ('Router-$A$', 'router_A', C), ('Router-$B$', 'router_B', C)]
        for mi, (mname, key) in enumerate((('regret', 'regret_mean'), ('reward', 'reward_mean'))):
            for j, (pn, pol, c) in enumerate(pols):
                first = f'\\multirow{{{len(pols)}}}{{*}}{{{mname}}}' if j == 0 else ''
                vals = []
                for k in ks:
                    x = g(s, k, Tt, 'scripted', pol, c)
                    vals.append('--' if not x else f'${num(x[key]):.1f}$')
                L.append(f'{first} & {pn} & ' + ' & '.join(vals) + ' \\\\')
            if mi == 0:
                L.append('\\midrule')
        L.append('\\bottomrule\n\\end{tabular}\n\\end{table}')
    with open(os.path.join(a.out, 'table.tex'), 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    print(f'wrote {os.path.join(a.out, "summary.md")} and {os.path.join(a.out, "table.tex")}')


if __name__ == '__main__':
    main()
