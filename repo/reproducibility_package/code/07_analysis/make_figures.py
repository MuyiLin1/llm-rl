"""Regenerate every figure in the paper from the run outputs.

Usage (from a directory that contains the study/ folder of result files):
    python make_figures.py --logdir study --outdir .

Produces:
    multiplication.png     learning dynamics, seed-averaged, with the routing window
    two_channel.png        full / transfer-removed / coverage-only multipliers
    decomposition.png      A_tau vs V_tau, stacked
    routing_rate.png       cumulative LLM calls over the horizon
    eps_sweep.png          the synthetic 1/eps scaling (needs no run data)

All panels are means over the ten seeds with standard-error bands, which is
what the paper reports; nothing here reads a single seed.
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPERTS = ['small', 'base', 'large']
NICE = {'small': 'Flan-T5-small', 'base': 'Flan-T5-base', 'large': 'Flan-T5-large'}
SHADE = {'small': '#8fbce8', 'base': '#3a7fd5', 'large': '#0b2f8a'}
COND = {'full':    'results_algo1_amazonv1_v1_{m}_c100.0_pfull_s{s}.json',
        'nom':     'results_algo1_nom_v1_{m}_c100.0_pfull_nom_s{s}.json',
        'passive': 'results_algo1_passiveonly_v1_{m}_c100.0_s{s}.json'}


def load(logdir, pat, m, seeds):
    out = []
    for s in seeds:
        p = os.path.join(logdir, pat.format(m=m, s=s))
        if os.path.exists(p):
            out.append(json.load(open(p)))
    if not out:
        raise SystemExit(f"no runs found for {pat.format(m=m, s='*')} in {logdir}")
    return out


def mean_se(curves):
    a = np.array(curves) * 100
    return a.mean(0), a.std(0, ddof=1) / np.sqrt(a.shape[0])


def fig_dynamics(logdir, outdir, seeds):
    hs = {m: load(logdir, COND['full'], m, seeds) for m in EXPERTS}
    step = np.array(hs['large'][0]['step']) / 1000.0
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.axvspan(0, 4.0, color='#f0c419', alpha=.18, zorder=0)
    L = np.vstack([mean_se([h['llm'] for h in hs[m]])[0] for m in EXPERTS])
    ax.fill_between(step, L.min(0), L.max(0), color='#2ca02c', alpha=.22, zorder=1)
    ax.plot(step, L.mean(0), color='#2ca02c', lw=1.4, ls=':', zorder=2)
    ax.text(6.5, 1.5, 'LLM alone (all three experts)', fontsize=8.6, color='#1e7a1e')
    c, cs = mean_se([h['linucb_cold'] for h in hs['large']])
    ax.plot(step, c, color='#d62728', lw=2.2, ls='--', label='Cold LinUCB', zorder=3)
    ax.fill_between(step, c - cs, c + cs, color='#d62728', alpha=.15, zorder=2)
    for m in EXPERTS:
        mm, ss = mean_se([h['hybrid'] for h in hs[m]])
        mult = mm[-1] / c[-1]
        ax.plot(step, mm, color=SHADE[m], lw=2.4, zorder=4,
                label=f'Router, {NICE[m]}   ${mult:.2f}\\times$')
        ax.fill_between(step, mm - ss, mm + ss, color=SHADE[m], alpha=.16, zorder=3)
    h_, l_ = ax.get_legend_handles_labels()
    order = list(range(1, len(l_))) + [0]
    ax.legend([h_[i] for i in order], [l_[i] for i in order], fontsize=8.8,
              loc='lower right', framealpha=.96)
    ax.annotate('routing window: every LLM call happens\nhere, then the expert is switched off',
                xy=(3.6, 24.0), xytext=(7.0, 27.6), fontsize=8.4, color='#7a6000',
                arrowprops=dict(arrowstyle='->', color='#a08000', lw=1.1))
    ax.set_xlabel('Round (thousands)'); ax.set_ylabel('Cumulative accuracy (%)')
    ax.set_xlim(0, step[-1]); ax.set_ylim(0, 31.5); ax.grid(alpha=.28)
    ax.set_title('The expert is switched off early; the gap keeps widening', fontsize=11.5)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'multiplication.png'), dpi=200,
                                    bbox_inches='tight'); plt.close(fig)


def fig_two_channel(logdir, outdir, seeds):
    def mult(cond, m):
        hs = load(logdir, COND[cond], m, seeds)
        R = np.array([h['hybrid'][-1] for h in hs]); C = np.array([h['linucb_cold'][-1] for h in hs])
        per = R / C
        return R.mean() / C.mean(), per.std(ddof=1) / np.sqrt(len(per))
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    bars = [('full', 'Full Router', '#0b2f8a'), ('nom', 'Transfer removed', '#7fa8d8'),
            ('passive', 'Coverage only', '#c9d9ee')]
    w, x = 0.26, np.arange(3)
    for i, (cond, lab, col) in enumerate(bars):
        vals = [mult(cond, m) for m in EXPERTS]
        mm = [v[0] for v in vals]; se = [v[1] for v in vals]
        ax.bar(x + (i - 1) * w, mm, w, yerr=se, capsize=3, label=lab, color=col,
               edgecolor='white', error_kw=dict(lw=1.1, ecolor='0.35'))
        for xi, v, s in zip(x + (i - 1) * w, mm, se):
            ax.text(xi, v + s + 0.09, f"{v:.2f}", ha='center', fontsize=8.5, color='0.15')
    ax.axhline(1.0, color='#d62728', ls='--', lw=1.4)
    ax.text(-0.44, 1.06, 'cold LinUCB', color='#d62728', fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels([NICE[m] for m in EXPERTS], fontsize=9.5)
    ax.set_ylabel('Accuracy multiplier over cold LinUCB'); ax.set_ylim(0, 4.35)
    ax.legend(fontsize=9, loc='upper left', framealpha=.95); ax.grid(axis='y', alpha=.3)
    ax.set_title('The transfer carries every difference between equally accurate experts',
                 fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'two_channel.png'), dpi=200,
                                    bbox_inches='tight'); plt.close(fig)


def fig_decomposition(logdir, outdir, seeds):
    A, V, SD = [], [], []
    for m in EXPERTS:
        a, v = [], []
        for s in seeds:
            p = os.path.join(logdir, f'perround_algo1_amazonv1_v1_{m}_c100.0_pfull_s{s}.npz')
            if not os.path.exists(p):
                continue
            z = np.load(p); n = int(z['step']); ro = z['routed'][:n].astype(bool)
            a.append(float((z['r_router'][:n][ro] - z['r_cold'][:n][ro]).sum()))
            v.append(float((z['r_router'][:n][~ro] - z['r_cold'][:n][~ro]).sum()))
        A.append(np.mean(a)); V.append(np.mean(v)); SD.append(np.std(v, ddof=1))
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    x, w = np.arange(3), 0.5
    ax.bar(x, A, w, label="$A_\\tau$  the expert's own answers on routed rounds",
           color='#e8a33d', zorder=3)
    ax.bar(x, V, w, bottom=A, yerr=SD, capsize=4, color='#0b2f8a', zorder=2,
           label='$V_\\tau$  the value of the data it left behind',
           error_kw=dict(lw=1.1, ecolor='0.4'))
    for i, (a, v) in enumerate(zip(A, V)):
        ax.text(i, a + v * 0.5, f"{v:,.0f}", ha='center', va='center', fontsize=11,
                color='white', weight='bold', zorder=4)
        ax.text(i, a + v * 0.5 - max(V) * 0.09, f"{100 * v / (a + v):.1f}%", ha='center',
                va='center', fontsize=9, color='#cfe0ff', zorder=4)
        ax.annotate(f"$A_\\tau$={a:.0f}", xy=(i - w / 2, a), xytext=(i - 0.47, a + max(V) * 0.12),
                    fontsize=8.6, color='#96650f', zorder=4,
                    arrowprops=dict(arrowstyle='->', color='#96650f', lw=1))
    ax.set_xticks(x); ax.set_xticklabels([NICE[m] for m in EXPERTS])
    ax.set_ylabel('Correct answers gained over cold LinUCB')
    ax.set_xlim(-0.68, 2.55); ax.set_ylim(0, max(np.array(V) + np.array(SD)) * 1.18)
    ax.legend(fontsize=9, loc='upper left'); ax.grid(axis='y', alpha=.3)
    ax.set_title("Almost none of the value is the expert's answers", fontsize=11.5)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'decomposition.png'), dpi=200,
                                    bbox_inches='tight'); plt.close(fig)


def fig_routing(logdir, outdir, seeds):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for m in EXPERTS:
        cum = []
        for s in seeds:
            p = os.path.join(logdir, f'perround_algo1_amazonv1_v1_{m}_c100.0_pfull_s{s}.npz')
            if not os.path.exists(p):
                continue
            z = np.load(p); n = int(z['step'])
            cum.append(np.cumsum(z['routed'][:n].astype(np.int32))[::100])
        c = np.array(cum, dtype=float); step = np.arange(c.shape[1]) * 100 / 1000.0
        mm = c.mean(0); se = c.std(0, ddof=1) / np.sqrt(c.shape[0])
        ax.plot(step, mm, color=SHADE[m], lw=2.2, label=NICE[m])
        ax.fill_between(step, mm - se, mm + se, color=SHADE[m], alpha=.18)
    ax.axvspan(0, 4, color='#f0c419', alpha=.16)
    ax.set_xlabel('Round (thousands)'); ax.set_ylabel('Cumulative LLM calls')
    ax.set_title('The gate stops calling the expert on its own', fontsize=11.5)
    ax.legend(fontsize=9, loc='lower right'); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'routing_rate.png'), dpi=200,
                                    bbox_inches='tight'); plt.close(fig)


def fig_eps_sweep(outdir):
    """Numbers printed by 06_synthetic_validation/sweep_and_row7.py."""
    eps = np.array([0.0005, 0.002, 0.008, 0.032])
    ratio = np.array([9.77, 3.17, 1.54, 1.12])
    prod = (ratio - 1) * eps
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.9))
    ax[0].loglog(eps, ratio - 1, 'o-', color='#0b2f8a', lw=2, ms=7, label='measured')
    ax[0].loglog(eps, prod.mean() / eps, '--', color='#d62728', lw=1.6,
                 label='theory: $\\propto 1/\\varepsilon$')
    ax[0].set_xlabel('$\\varepsilon$  (probe suboptimality)'); ax[0].set_ylabel('multiplier $-\\,1$')
    ax[0].legend(fontsize=9); ax[0].grid(alpha=.3, which='both')
    ax[0].set_title('(a) the predicted $1/\\varepsilon$ scaling', fontsize=10.5)
    ax[1].semilogx(eps, prod, 'o-', color='#0b2f8a', lw=2, ms=7)
    ax[1].axhline(prod.mean(), color='0.5', ls=':', lw=1.3)
    for e, p in zip(eps, prod):
        ax[1].annotate(f"{p:.4f}", (e, p), textcoords='offset points', xytext=(0, 9),
                       ha='center', fontsize=8.3)
    ax[1].set_ylim(0, 0.0065); ax[1].set_xlabel('$\\varepsilon$  (64-fold range)')
    ax[1].set_ylabel('(multiplier $-\\,1)\\times\\varepsilon$')
    ax[1].grid(alpha=.3, which='both'); ax[1].set_title('(b) the product stays constant', fontsize=10.5)
    fig.suptitle('The synthetic instance reproduces the mechanism, not just the sign',
                 fontsize=11.5, y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'eps_sweep.png'), dpi=200,
                                    bbox_inches='tight'); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logdir', default='study')
    ap.add_argument('--outdir', default='.')
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 11)))
    ap.add_argument('--only', nargs='*', default=None,
                    help='subset of: dynamics two_channel decomposition routing eps')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    jobs = {'dynamics': lambda: fig_dynamics(a.logdir, a.outdir, a.seeds),
            'two_channel': lambda: fig_two_channel(a.logdir, a.outdir, a.seeds),
            'decomposition': lambda: fig_decomposition(a.logdir, a.outdir, a.seeds),
            'routing': lambda: fig_routing(a.logdir, a.outdir, a.seeds),
            'eps': lambda: fig_eps_sweep(a.outdir)}
    for name, fn in jobs.items():
        if a.only and name not in a.only:
            continue
        try:
            fn(); print(f"  wrote {name}")
        except SystemExit as e:
            print(f"  skipped {name}: {e}")


if __name__ == '__main__':
    main()
