"""Figure 1 (two_channel.png) from the replay runs: full / transfer removed / no response sum,
all three on the SAME routed rounds (those the full Router chose). Same style as
experiments/07_analysis/make_figures.py:fig_two_channel. Bar = ratio of seed-averaged accuracies;
error bar = standard error of the per-seed multiplier."""
import os, sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(os.path.dirname(HERE), 'out')
dest = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'two_channel.png')
E = ['small', 'base', 'large']; NICE = {e: f'Flan-T5-{e}' for e in E}; S = range(1, 11)
cold = np.array([np.load(f'{OUT}/small_s{s}.npz')['r_cold'].mean() for s in S])
def mult(k, e):
    R = np.array([np.load(f'{OUT}/{e}_s{s}.npz')[f'r_{k}'].mean() for s in S]); per = R / cold
    return R.mean() / cold.mean(), per.std(ddof=1) / np.sqrt(len(per))
fig, ax = plt.subplots(figsize=(7.6, 4.5))
bars = [('full', 'Full Router', '#0b2f8a'), ('nom', 'Transfer removed', '#7fa8d8'),
        ('passive', 'No response sum', '#c9d9ee')]
w, x = 0.26, np.arange(3)
for i, (k, lab, col) in enumerate(bars):
    vals = [mult(k, e) for e in E]; mm = [v[0] for v in vals]; se = [v[1] for v in vals]
    ax.bar(x + (i - 1) * w, mm, w, yerr=se, capsize=3, label=lab, color=col,
           edgecolor='white', error_kw=dict(lw=1.1, ecolor='0.35'))
    for xi, v, s in zip(x + (i - 1) * w, mm, se):
        ax.text(xi, v + s + 0.09, f"{v:.2f}", ha='center', fontsize=8.5, color='0.15')
ax.axhline(1.0, color='#d62728', ls='--', lw=1.4); ax.text(-0.44, 1.06, 'cold LinUCB', color='#d62728', fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels([NICE[e] for e in E], fontsize=9.5)
ax.set_ylabel('Accuracy multiplier over cold LinUCB'); ax.set_ylim(0, 4.35)
ax.legend(fontsize=9, loc='upper left', framealpha=.95, title='same routed rounds in all three', title_fontsize=8.5)
ax.grid(axis='y', alpha=.3)
ax.set_title('The transfer carries every difference between equally accurate experts', fontsize=11)
fig.tight_layout(); fig.savefig(dest, dpi=200, bbox_inches='tight'); print('wrote', dest)
