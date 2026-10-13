"""Figures and numbers for the paper from the local Phase B runs: the exploration sweep, the skew experiment and the
BERT tuned baseline. Writes exploration_sweep.png and skew.png to the repository root and prints the numbers quoted
in the text. Error bars: one standard deviation of a paired bootstrap over seeds."""
import os, sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import skew
OUT = f'{HERE}/out_local'; ROOT = os.path.abspath(f'{HERE}/../..'); SEEDS = range(1, 11)
EXPERTS = [('small', 'Flan-T5-small'), ('base', 'Flan-T5-base'), ('large', 'Flan-T5-large')]
COL = {'small': '#9ecae1', 'base': '#4292c6', 'large': '#08306b'}
rng = np.random.RandomState(0)

def acc(n): return float(np.load(f'{OUT}/{n}.npz')['R'].mean())
def settle(n): a = np.load(f'{OUT}/{n}.npz')['A']; return (a[-len(a)//5:] == 1471).mean() > 0.2
def routed(n): return float(np.load(f'{OUT}/{n}.npz')['RO'].mean())
def ratio(r, c):
    r, c = np.asarray(r), np.asarray(c); B = rng.randint(0, len(r), (2000, len(r)))
    return r.mean()/c.mean(), (r[B].mean(1)/c[B].mean(1)).std()
def saved(r, c):
    """Percent of cold LinUCB's regret the Router saves, (acc_R - acc_C) / (1 - acc_C)."""
    r, c = np.asarray(r), np.asarray(c); B = rng.randint(0, len(r), (2000, len(r)))
    f = lambda rr, cc: 100*(rr.mean(-1) - cc.mean(-1))/(1 - cc.mean(-1))
    return f(r, c), f(r[B], c[B]).std()
def tstat(d): d = np.asarray(d); return d.mean()/(d.std(ddof=1)/np.sqrt(len(d)))

YLAB = "regret saved vs. cold LinUCB (%)"
def panel(ax, xs, cold_fmt, router_fmt, xlabel, logx=False):
    for e, lab in EXPERTS:
        m, s = zip(*[saved([acc(router_fmt.format(x=x, e=e, s=sd)) for sd in SEEDS], [acc(cold_fmt.format(x=x, s=sd)) for sd in SEEDS]) for x in xs])
        ax.errorbar(xs, m, yerr=s, marker='o', capsize=3, color=COL[e], label=lab)
    ax.axhline(0, color='grey', lw=0.8, ls='--')
    if logx: ax.set_xscale('log')
    ax.set_xlabel(xlabel); ax.set_ylabel(YLAB); ax.legend(frameon=False)

# exploration sweep
XF = [0.1, 0.3, 1.0, 3.0, 10.0]
fig, ax = plt.subplots(figsize=(5.2, 3.4))
panel(ax, XF, 'bert_prod_cold_-_c0_const_a{x:g}_l0.1_s{s}', 'bert_prod_router_{e}_c100_const_a{x:g}_l0.1_s{s}',
      r'exploration scale $\alpha$ (constant)', logx=True)
fig.tight_layout(); fig.savefig(f'{ROOT}/exploration_sweep.png', dpi=200); plt.close(fig)

# skew
names = list(skew.CAPS); share = [29.8, 20, 10, 5]
fig, ax = plt.subplots(figsize=(5.2, 3.4))
for e, lab in EXPERTS:
    m, s = zip(*[saved([acc(f'{n}_prod_router_{e}_c100_fast_a0.1_l0.1_s{sd}') for sd in SEEDS],
                       [acc(f'{n}_prod_cold_-_c0_fast_a0.1_l0.1_s{sd}') for sd in SEEDS]) for n in names])
    ax.errorbar(share, m, yerr=s, marker='o', capsize=3, color=COL[e], label=lab)
    print(f'skew {e}: regret saved ' + ', '.join(f'{x:.1f}+-{y:.1f}' for x, y in zip(m, s)))
ax.axhline(0, color='grey', lw=0.8, ls='--'); ax.invert_xaxis()
ax.set_xlabel('share of contexts with the most frequent category (%)'); ax.set_ylabel(YLAB)
ax.legend(frameon=False); fig.tight_layout(); fig.savefig(f'{ROOT}/skew.png', dpi=200); plt.close(fig)

# the two panels above as one figure for the main text, drawn at the printed width so its text prints at 9pt
plt.rcParams.update({'font.size': 9})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.0, 2.5), sharey=True)
panel(a1, XF, 'bert_prod_cold_-_c0_const_a{x:g}_l0.1_s{s}', 'bert_prod_router_{e}_c100_const_a{x:g}_l0.1_s{s}',
      r'exploration scale $\alpha$ (constant)', logx=True)
for e, lab in EXPERTS:
    m, s = zip(*[saved([acc(f'{n}_prod_router_{e}_c100_fast_a0.1_l0.1_s{sd}') for sd in SEEDS],
                       [acc(f'{n}_prod_cold_-_c0_fast_a0.1_l0.1_s{sd}') for sd in SEEDS]) for n in names])
    a2.errorbar(share, m, yerr=s, marker='o', capsize=3, color=COL[e], label=lab)
a2.axhline(0, color='grey', lw=0.8, ls='--'); a2.invert_xaxis()
a2.set_xlabel('share of the dominant category (%)')
a1.set_ylabel('regret saved (%)'); a1.set_title('(a) more exploration', fontsize=9)
a2.set_title('(b) rarer dominant category', fontsize=9); a1.legend(frameon=False, fontsize=8); a2.get_legend() and a2.get_legend().remove()
fig.tight_layout(); fig.savefig(f'{ROOT}/regimes.png', dpi=300); plt.close(fig)

# numbers quoted in the text
print('== BERT tuned baseline (fast, floor 0.05, lambda 0.1; Router c=1000), seeds 1-10 ==')
cn = [f'bert_prod_cold_-_c0_fast_a0.05_l0.1_s{s}' for s in SEEDS]; c = np.array([acc(n) for n in cn])
print(f'cold {c.mean()*100:.2f} +- {c.std(ddof=1)*100:.2f}, settles {sum(map(settle, cn))}/10')
for e, lab in EXPERTS:
    rn = [f'bert_prod_router_{e}_c1000_fast_a0.05_l0.1_s{s}' for s in SEEDS]; r = np.array([acc(n) for n in rn])
    print(f'{lab}: Router {r.mean()*100:.2f} +- {r.std(ddof=1)*100:.2f}, mult {r.mean()/c.mean():.2f}, '
          f'regret ratio {(1-c.mean())/(1-r.mean()):.3f}, t {tstat(r-c):.2f}, settles {sum(map(settle, rn))}/10, '
          f'routed {np.mean([routed(n) for n in rn])*100:.2f}%')
print('\n== Regret ratios (cold / Router regret) ==')
for x in XF:
    c = np.array([acc(f'bert_prod_cold_-_c0_const_a{x:g}_l0.1_s{s}') for s in SEEDS])
    print(f'alpha {x:g}: ' + ', '.join(f"{e} {(1-c.mean())/(1-np.mean([acc(f'bert_prod_router_{e}_c100_const_a{x:g}_l0.1_s{s}') for s in SEEDS])):.3f}" for e, _ in EXPERTS))
for n in names:
    c = np.array([acc(f'{n}_prod_cold_-_c0_fast_a0.1_l0.1_s{s}') for s in SEEDS])
    print(f'{n}: ' + ', '.join(f"{e} {(1-c.mean())/(1-np.mean([acc(f'{n}_prod_router_{e}_c100_fast_a0.1_l0.1_s{s}') for s in SEEDS])):.3f}" for e, _ in EXPERTS))
