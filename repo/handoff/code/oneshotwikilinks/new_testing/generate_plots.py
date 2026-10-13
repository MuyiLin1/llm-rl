"""
generate_plots.py  –  Comprehensive figure generation for the NeurIPS paper.

Produces the following figures (saved to figures/):

  PER-MODEL INDIVIDUAL PLOTS (6 models × 4 plot types = 24 images):
    1. Learning curves       – Router vs LLM vs Cold LinUCB accuracy over batches
    2. LLM routing fraction  – % of rounds routed to LLM over time
    3. Eigenvalue growth      – Bottom-10 eigenvalues of V_warm and V_cold
    4. Feature norm tracking  – Max feature norm over time

  COMPARISON PLOTS (grouped by model family, v1 vs v3, and combined):
    5.  v1 learning curves overlay     (small/base/large on one plot)
    6.  v3 learning curves overlay     (small/base/large on one plot)
    7.  v1 routing fraction overlay
    8.  v3 routing fraction overlay
    9.  Combined routing overlay       (all 6 models)
    10. Final accuracy bar chart – v1  (Router/LLM/Cold grouped)
    11. Final accuracy bar chart – v3
    12. Final accuracy bar chart – all 6
    13. Warm-start lift bar chart      (Router − Cold LinUCB) all 6
    14. Eigenvalue comparison v1 vs v3 (bottom eigenvalue at final checkpoint)
    15. v1+v3 learning curves overlay  (all 6 on one plot, v1 dashed, v3 solid)

Usage:
    python generate_plots.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ── Style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.figsize': (8, 4.5),
    'figure.dpi': 200,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9.5,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 1.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

OUTDIR = 'figures'
os.makedirs(OUTDIR, exist_ok=True)

# ── Color palette ────────────────────────────────────────────────────
C_ROUTER  = '#2166AC'   # deep blue
C_LLM     = '#B2182B'   # deep red
C_COLD    = '#4DAF4A'   # green
C_FRAC    = '#FF7F00'   # orange for routing fraction

# Size-specific colors for overlays
SIZE_COLORS = {
    'small': '#1b9e77',
    'base':  '#d95f02',
    'large': '#7570b3',
}

# Model family styles
FAMILY_STYLES = {
    'v1': {'ls': '--', 'alpha': 0.85},
    'v3': {'ls': '-',  'alpha': 1.0},
}

# ── Data loading ─────────────────────────────────────────────────────
FILES = {
    ('v1', 'small'): 'results_algo1_small_c100.0.json',
    ('v1', 'base'):  'results_algo1_base_c100.0.json',
    ('v1', 'large'): 'results_algo1_large_c100.0.json',
    ('v3', 'small'): 'results_algo1v3_small_c100.0.json',
    ('v3', 'base'):  'results_algo1v3_base_c100.0.json',
    ('v3', 'large'): 'results_algo1v3_large_c100.0.json',
}

PRETTY_NAMES = {
    ('v1', 'small'): 'Flan-T5-Small (60M)',
    ('v1', 'base'):  'Flan-T5-Base (250M)',
    ('v1', 'large'): 'Flan-T5-Large (780M)',
    ('v3', 'small'): 'Qwen2.5-0.5B-Instruct',
    ('v3', 'base'):  'Qwen2.5-3B-Instruct',
    ('v3', 'large'): 'Qwen2.5-7B-Instruct',
}

FAMILY_NAMES = {
    'v1': 'Flan-T5 + BERT (768d)',
    'v3': 'Qwen2.5 + BGE (1024d)',
}

data = {}
for key, fname in FILES.items():
    with open(fname) as f:
        data[key] = json.load(f)
    print(f"Loaded {key}: {fname}")


def batches_to_rounds(batches):
    """Convert batch indices to approximate round numbers (batch_size=128)."""
    return np.array(batches) * 128


def pct_formatter(x, pos):
    return f'{x:.0%}' if x < 1 else f'{x:.1%}'


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1: Individual per-model plots
# ═══════════════════════════════════════════════════════════════════════

def plot_learning_curve(family, size):
    """Fig type 1: Router / LLM / Cold LinUCB learning curves."""
    d = data[(family, size)]
    batches = np.array(d['batch'])
    rounds = batches_to_rounds(batches)
    name = PRETTY_NAMES[(family, size)]

    fig, ax = plt.subplots()
    ax.plot(rounds, d['hybrid'], color=C_ROUTER, label='Router (Alg. 1)')
    ax.plot(rounds, d['llm'], color=C_LLM, label='LLM Only', linestyle='--')
    ax.plot(rounds, d['linucb_cold'], color=C_COLD, label='Cold LinUCB', linestyle=':')

    ax.set_xlabel('Round')
    ax.set_ylabel('Cumulative Average Reward (Precision@1)')
    ax.set_title(f'Learning Curves — {name}')
    ax.legend(loc='best')
    ax.set_xlim(0, rounds[-1])

    fname = f'{OUTDIR}/learning_curve_{family}_{size}.pdf'
    fig.savefig(fname)
    plt.close(fig)
    print(f'  Saved {fname}')


def plot_routing_fraction(family, size):
    """Fig type 2: LLM routing fraction over time."""
    d = data[(family, size)]
    batches = np.array(d['batch'])
    rounds = batches_to_rounds(batches)
    name = PRETTY_NAMES[(family, size)]

    fig, ax = plt.subplots()
    ax.plot(rounds, d['llm_fraction'], color=C_FRAC, linewidth=2)
    ax.set_xlabel('Round')
    ax.set_ylabel('Fraction of Rounds Routed to LLM')
    ax.set_title(f'LLM Routing Fraction — {name}')
    ax.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
    ax.set_xlim(0, rounds[-1])
    ax.set_ylim(-0.02, 1.05)

    # Mark tau if detected
    if d.get('tau_round') is not None:
        ax.axvline(d['tau_round'], color='gray', linestyle='--', alpha=0.7, linewidth=1)
        ax.text(d['tau_round'] + rounds[-1]*0.01, 0.5,
                f"τ = {d['tau_round']:,}", fontsize=9, color='gray')

    fname = f'{OUTDIR}/routing_{family}_{size}.pdf'
    fig.savefig(fname)
    plt.close(fig)
    print(f'  Saved {fname}')


def plot_eigenvalues(family, size):
    """Fig type 3: Bottom-10 eigenvalue growth (warm vs cold)."""
    d = data[(family, size)]
    batches = np.array(d['batch'])
    rounds = batches_to_rounds(batches)
    name = PRETTY_NAMES[(family, size)]

    eigs_warm = d['eigenvalues_warm_bottom10']
    eigs_cold = d['eigenvalues_cold_bottom10']

    # Extract the smallest (bottom-1) eigenvalue at each checkpoint
    min_eig_warm = [e[0] if e else np.nan for e in eigs_warm]
    min_eig_cold = [e[0] if e else np.nan for e in eigs_cold]

    fig, ax = plt.subplots()
    ax.plot(rounds, min_eig_warm, color=C_ROUTER, label='$\\lambda_{\\min}(V_{\\mathrm{warm}})$')
    ax.plot(rounds, min_eig_cold, color=C_COLD, label='$\\lambda_{\\min}(V_{\\mathrm{cold}})$', linestyle='--')
    ax.set_xlabel('Round')
    ax.set_ylabel('Smallest Eigenvalue of $V$')
    ax.set_title(f'Eigenvalue Growth — {name}')
    ax.legend(loc='best')
    ax.set_yscale('log')
    ax.set_xlim(0, rounds[-1])

    fname = f'{OUTDIR}/eigenvalues_{family}_{size}.pdf'
    fig.savefig(fname)
    plt.close(fig)
    print(f'  Saved {fname}')


def plot_feat_norm(family, size):
    """Fig type 4: Max feature norm over time."""
    d = data[(family, size)]
    batches = np.array(d['batch'])
    rounds = batches_to_rounds(batches)
    name = PRETTY_NAMES[(family, size)]

    fig, ax = plt.subplots()
    ax.plot(rounds, d['max_feat_norm'], color='#555555', linewidth=1.5)
    ax.set_xlabel('Round')
    ax.set_ylabel('Max Feature Norm $L$')
    ax.set_title(f'Feature Norm Tracking — {name}')
    ax.set_xlim(0, rounds[-1])

    fname = f'{OUTDIR}/feat_norm_{family}_{size}.pdf'
    fig.savefig(fname)
    plt.close(fig)
    print(f'  Saved {fname}')


print("\n=== Individual per-model plots ===")
for family in ['v1', 'v3']:
    for size in ['small', 'base', 'large']:
        print(f"\n{family} {size}:")
        plot_learning_curve(family, size)
        plot_routing_fraction(family, size)
        plot_eigenvalues(family, size)
        plot_feat_norm(family, size)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: Comparison plots within each model family
# ═══════════════════════════════════════════════════════════════════════

print("\n=== Comparison plots ===")

def plot_learning_curves_overlay(family):
    """Fig 5/6: Overlay small/base/large learning curves for one family."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)

    for ax, size in zip(axes, ['small', 'base', 'large']):
        d = data[(family, size)]
        rounds = batches_to_rounds(np.array(d['batch']))
        name = PRETTY_NAMES[(family, size)]

        ax.plot(rounds, d['hybrid'], color=C_ROUTER, label='Router')
        ax.plot(rounds, d['llm'], color=C_LLM, label='LLM Only', linestyle='--')
        ax.plot(rounds, d['linucb_cold'], color=C_COLD, label='Cold LinUCB', linestyle=':')

        ax.set_title(name, fontsize=11)
        ax.set_xlabel('Round')
        if ax is axes[0]:
            ax.set_ylabel('Cumulative Average Reward')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_xlim(0, rounds[-1])

    fig.suptitle(f'Learning Curves — {FAMILY_NAMES[family]}', fontsize=14, y=1.02)
    fig.tight_layout()
    fname = f'{OUTDIR}/learning_curves_{family}_all.pdf'
    fig.savefig(fname)
    plt.close(fig)
    print(f'  Saved {fname}')


def plot_routing_overlay(family):
    """Fig 7/8: Overlay routing fraction for one family."""
    fig, ax = plt.subplots()

    for size in ['small', 'base', 'large']:
        d = data[(family, size)]
        rounds = batches_to_rounds(np.array(d['batch']))
        label = PRETTY_NAMES[(family, size)].split('(')[0].strip()

        ax.plot(rounds, d['llm_fraction'], color=SIZE_COLORS[size],
                label=label, linewidth=2)

    ax.set_xlabel('Round')
    ax.set_ylabel('Fraction Routed to LLM')
    ax.set_title(f'Confidence-Gated Hand-Off — {FAMILY_NAMES[family]}')
    ax.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
    ax.set_xlim(0, rounds[-1])
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc='upper right')

    fname = f'{OUTDIR}/routing_{family}_all.pdf'
    fig.savefig(fname)
    plt.close(fig)
    print(f'  Saved {fname}')


for family in ['v1', 'v3']:
    plot_learning_curves_overlay(family)
    plot_routing_overlay(family)


# ── Fig 9: Combined routing overlay (all 6) ─────────────────────────
fig, ax = plt.subplots()
for family in ['v1', 'v3']:
    for size in ['small', 'base', 'large']:
        d = data[(family, size)]
        rounds = batches_to_rounds(np.array(d['batch']))
        ls = FAMILY_STYLES[family]['ls']
        alpha = FAMILY_STYLES[family]['alpha']
        label = f"{PRETTY_NAMES[(family, size)].split('(')[0].strip()}"
        if family == 'v1':
            label += ' (v1)'
        else:
            label += ' (v3)'
        ax.plot(rounds, d['llm_fraction'], color=SIZE_COLORS[size],
                linestyle=ls, alpha=alpha, label=label, linewidth=1.8)

ax.set_xlabel('Round')
ax.set_ylabel('Fraction Routed to LLM')
ax.set_title('Confidence-Gated Hand-Off — All Models')
ax.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
ax.set_xlim(0, rounds[-1])
ax.set_ylim(-0.02, 1.05)
ax.legend(loc='upper right', fontsize=8, ncol=2)
fname = f'{OUTDIR}/routing_all_models.pdf'
fig.savefig(fname)
plt.close(fig)
print(f'  Saved {fname}')


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3: Bar charts
# ═══════════════════════════════════════════════════════════════════════

print("\n=== Bar charts ===")

def plot_accuracy_bars(family_filter, suffix):
    """Fig 10/11/12: Grouped bar chart of final accuracy."""
    keys = [(f, s) for (f, s) in FILES.keys() if family_filter is None or f == family_filter]

    labels = [PRETTY_NAMES[k].split('(')[0].strip() if family_filter else
              f"{PRETTY_NAMES[k].split('(')[0].strip()}\n({'v1' if k[0]=='v1' else 'v3'})"
              for k in keys]
    router_acc = [data[k]['hybrid'][-1] for k in keys]
    llm_acc    = [data[k]['llm'][-1] for k in keys]
    cold_acc   = [data[k]['linucb_cold'][-1] for k in keys]

    x = np.arange(len(keys))
    w = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(keys)*2), 5))
    bars1 = ax.bar(x - w, router_acc, w, label='Router', color=C_ROUTER, zorder=3)
    bars2 = ax.bar(x,     llm_acc,    w, label='LLM Only', color=C_LLM, zorder=3)
    bars3 = ax.bar(x + w, cold_acc,   w, label='Cold LinUCB', color=C_COLD, zorder=3)

    ax.set_ylabel('Final Cumulative Accuracy')
    title_family = FAMILY_NAMES.get(family_filter, 'All Models')
    ax.set_title(f'Final Performance — {title_family}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend()
    ax.set_ylim(0, max(max(router_acc), max(llm_acc), max(cold_acc)) * 1.15)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=7.5)

    fname = f'{OUTDIR}/accuracy_bars_{suffix}.pdf'
    fig.savefig(fname)
    plt.close(fig)
    print(f'  Saved {fname}')


plot_accuracy_bars('v1', 'v1')
plot_accuracy_bars('v3', 'v3')
plot_accuracy_bars(None, 'all')


# ── Fig 13: Warm-start lift bar chart ────────────────────────────────
def plot_lift_bars():
    """Router − Cold LinUCB lift for all 6 models."""
    keys = list(FILES.keys())
    labels = [f"{PRETTY_NAMES[k].split('(')[0].strip()}\n({'v1' if k[0]=='v1' else 'v3'})"
              for k in keys]
    lifts = [data[k]['hybrid'][-1] - data[k]['linucb_cold'][-1] for k in keys]
    colors = [SIZE_COLORS[k[1]] for k in keys]
    hatches = ['///' if k[0] == 'v1' else '' for k in keys]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(keys)), lifts, color=colors, zorder=3, edgecolor='white')
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    ax.set_ylabel('Accuracy Lift (Router − Cold LinUCB)')
    ax.set_title('Warm-Start Lift Across All Models')
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(0, color='black', linewidth=0.8)

    for i, (bar, lift) in enumerate(zip(bars, lifts)):
        ax.annotate(f'{lift:+.4f}',
                   xy=(bar.get_x() + bar.get_width()/2, lift),
                   xytext=(0, 3 if lift >= 0 else -12),
                   textcoords='offset points',
                   ha='center', va='bottom' if lift >= 0 else 'top',
                   fontsize=8, fontweight='bold')

    fname = f'{OUTDIR}/warmstart_lift_all.pdf'
    fig.savefig(fname)
    plt.close(fig)
    print(f'  Saved {fname}')

plot_lift_bars()


# ── Fig 14: Eigenvalue comparison ────────────────────────────────────
def plot_eigenvalue_comparison():
    """Final bottom-10 eigenvalue spectrum for warm vs cold, all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, family in zip(axes, ['v1', 'v3']):
        for size in ['small', 'base', 'large']:
            d = data[(family, size)]
            eigs_warm = d['eigenvalues_warm_bottom10'][-1]
            eigs_cold = d['eigenvalues_cold_bottom10'][-1]
            x_pos = np.arange(10)
            label_w = f'{size} (warm)'
            label_c = f'{size} (cold)'
            ax.plot(x_pos, eigs_warm, 'o-', color=SIZE_COLORS[size],
                    label=label_w, markersize=4)
            ax.plot(x_pos, eigs_cold, 's--', color=SIZE_COLORS[size],
                    label=label_c, markersize=4, alpha=0.5)

        ax.set_xlabel('Eigenvalue Index (bottom 10)')
        ax.set_ylabel('Eigenvalue of $V$')
        ax.set_title(f'{FAMILY_NAMES[family]}')
        ax.legend(fontsize=7, ncol=2)
        ax.set_yscale('log')

    fig.suptitle('Final Design Matrix Eigenvalue Spectrum', fontsize=14, y=1.02)
    fig.tight_layout()
    fname = f'{OUTDIR}/eigenvalue_comparison.pdf'
    fig.savefig(fname)
    plt.close(fig)
    print(f'  Saved {fname}')

plot_eigenvalue_comparison()


# ── Fig 15: All 6 learning curves on one plot ────────────────────────
def plot_all_learning_curves():
    """v1 dashed, v3 solid — all on one plot for Router track."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for family in ['v1', 'v3']:
        for size in ['small', 'base', 'large']:
            d = data[(family, size)]
            rounds = batches_to_rounds(np.array(d['batch']))
            ls = FAMILY_STYLES[family]['ls']
            alpha = FAMILY_STYLES[family]['alpha']
            name = PRETTY_NAMES[(family, size)]
            tag = 'v1' if family == 'v1' else 'v3'
            ax.plot(rounds, d['hybrid'], color=SIZE_COLORS[size],
                    linestyle=ls, alpha=alpha, label=f'{name} ({tag})',
                    linewidth=1.8)

    ax.set_xlabel('Round')
    ax.set_ylabel('Cumulative Average Reward (Precision@1)')
    ax.set_title('Router Performance — All Models')
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    ax.set_xlim(0, rounds[-1])

    fname = f'{OUTDIR}/learning_curves_all_router.pdf'
    fig.savefig(fname)
    plt.close(fig)
    print(f'  Saved {fname}')

plot_all_learning_curves()


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4: Key paper figures (high-quality versions)
# ═══════════════════════════════════════════════════════════════════════

print("\n=== Key paper figures ===")

def plot_paper_v1_triptych():
    """Three-panel figure: v1 small/base/large learning curves side by side.
    This is the main result figure for the Flan-T5 experiments."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)

    for ax, size in zip(axes, ['small', 'base', 'large']):
        d = data[('v1', size)]
        rounds = batches_to_rounds(np.array(d['batch']))
        name = PRETTY_NAMES[('v1', size)]

        ax.plot(rounds, d['hybrid'], color=C_ROUTER, label='Router (Alg. 1)', linewidth=2)
        ax.plot(rounds, d['llm'], color=C_LLM, label='LLM Only', linestyle='--', linewidth=1.5)
        ax.plot(rounds, d['linucb_cold'], color=C_COLD, label='Cold LinUCB', linestyle=':', linewidth=1.5)

        # Final values annotation
        final_r = d['hybrid'][-1]
        final_l = d['llm'][-1]
        final_c = d['linucb_cold'][-1]

        ax.set_title(name, fontsize=12)
        ax.set_xlabel('Round')
        if ax is axes[0]:
            ax.set_ylabel('Cumulative Avg. Reward')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_xlim(0, rounds[-1])

        # Annotate final values in top-right
        txt = f'Router: {final_r:.4f}\nLLM: {final_l:.4f}\nCold: {final_c:.4f}'
        ax.text(0.97, 0.45, txt, transform=ax.transAxes, fontsize=7.5,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

    fig.suptitle('OneShotWikiLinks-311 — Flan-T5 + BERT (768d)', fontsize=14, y=1.02)
    fig.tight_layout()
    fname = f'{OUTDIR}/paper_v1_triptych.pdf'
    fig.savefig(fname)
    plt.close(fig)
    print(f'  Saved {fname}')


def plot_paper_v3_triptych():
    """Three-panel: v3 small/base/large."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)

    for ax, size in zip(axes, ['small', 'base', 'large']):
        d = data[('v3', size)]
        rounds = batches_to_rounds(np.array(d['batch']))
        name = PRETTY_NAMES[('v3', size)]

        ax.plot(rounds, d['hybrid'], color=C_ROUTER, label='Router (Alg. 1)', linewidth=2)
        ax.plot(rounds, d['llm'], color=C_LLM, label='LLM Only', linestyle='--', linewidth=1.5)
        ax.plot(rounds, d['linucb_cold'], color=C_COLD, label='Cold LinUCB', linestyle=':', linewidth=1.5)

        final_r = d['hybrid'][-1]
        final_l = d['llm'][-1]
        final_c = d['linucb_cold'][-1]

        ax.set_title(name, fontsize=12)
        ax.set_xlabel('Round')
        if ax is axes[0]:
            ax.set_ylabel('Cumulative Avg. Reward')
        ax.legend(loc='best', fontsize=8)
        ax.set_xlim(0, rounds[-1])

        txt = f'Router: {final_r:.4f}\nLLM: {final_l:.4f}\nCold: {final_c:.4f}'
        ax.text(0.97, 0.45, txt, transform=ax.transAxes, fontsize=7.5,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

    fig.suptitle('OneShotWikiLinks-311 — Qwen2.5 + BGE (1024d)', fontsize=14, y=1.02)
    fig.tight_layout()
    fname = f'{OUTDIR}/paper_v3_triptych.pdf'
    fig.savefig(fname)
    plt.close(fig)
    print(f'  Saved {fname}')


def plot_paper_handoff_comparison():
    """Two-panel routing comparison: v1 left, v3 right."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)

    for ax, family in zip(axes, ['v1', 'v3']):
        for size in ['small', 'base', 'large']:
            d = data[(family, size)]
            rounds = batches_to_rounds(np.array(d['batch']))
            label = PRETTY_NAMES[(family, size)].split('(')[0].strip()
            ax.plot(rounds, d['llm_fraction'], color=SIZE_COLORS[size],
                    label=label, linewidth=2)

        ax.set_xlabel('Round')
        ax.set_ylabel('Fraction Routed to LLM')
        ax.set_title(f'{FAMILY_NAMES[family]}')
        ax.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
        ax.set_xlim(0, rounds[-1])
        ax.set_ylim(-0.02, 1.05)
        ax.legend(loc='upper right')

    fig.suptitle('Confidence-Gated Hand-Off Dynamics', fontsize=14, y=1.02)
    fig.tight_layout()
    fname = f'{OUTDIR}/paper_handoff_comparison.pdf'
    fig.savefig(fname)
    plt.close(fig)
    print(f'  Saved {fname}')


def plot_paper_summary_table():
    """Final summary bar chart — all 6 models, grouped by family.
    Two subplots: v1 left, v3 right."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, family in zip(axes, ['v1', 'v3']):
        sizes = ['small', 'base', 'large']
        labels = [PRETTY_NAMES[(family, s)].split('(')[0].strip() for s in sizes]
        router = [data[(family, s)]['hybrid'][-1] for s in sizes]
        llm    = [data[(family, s)]['llm'][-1] for s in sizes]
        cold   = [data[(family, s)]['linucb_cold'][-1] for s in sizes]

        x = np.arange(3)
        w = 0.25

        b1 = ax.bar(x - w, router, w, label='Router', color=C_ROUTER, zorder=3)
        b2 = ax.bar(x,     llm,    w, label='LLM Only', color=C_LLM, zorder=3)
        b3 = ax.bar(x + w, cold,   w, label='Cold LinUCB', color=C_COLD, zorder=3)

        ax.set_ylabel('Final Cumulative Accuracy')
        ax.set_title(FAMILY_NAMES[family])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.legend(fontsize=8)
        ax.set_ylim(0, max(max(router), max(llm), max(cold)) * 1.18)

        for bars in [b1, b2, b3]:
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f'{h:.3f}',
                           xy=(bar.get_x() + bar.get_width()/2, h),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=7)

    fig.suptitle('Final Performance Across All Models', fontsize=14, y=1.02)
    fig.tight_layout()
    fname = f'{OUTDIR}/paper_final_bars.pdf'
    fig.savefig(fname)
    plt.close(fig)
    print(f'  Saved {fname}')


def plot_paper_zoom_early():
    """Zoom into early batches (0-500) to show hand-off dynamics on learning curves.
    v1 left, v3 right."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, family in zip(axes, ['v1', 'v3']):
        for size in ['small', 'base', 'large']:
            d = data[(family, size)]
            batches = np.array(d['batch'])
            rounds = batches_to_rounds(batches)

            # Zoom to first ~500 batches
            mask = batches <= 500
            r = rounds[mask]
            h = np.array(d['hybrid'])[mask]
            l = np.array(d['llm'])[mask]

            ax.plot(r, h, color=SIZE_COLORS[size], linewidth=2,
                    label=f"{PRETTY_NAMES[(family, size)].split('(')[0].strip()} Router")
            ax.plot(r, l, color=SIZE_COLORS[size], linewidth=1, linestyle='--',
                    label=f"{PRETTY_NAMES[(family, size)].split('(')[0].strip()} LLM", alpha=0.6)

        ax.set_xlabel('Round')
        ax.set_ylabel('Cumulative Avg. Reward')
        ax.set_title(f'Early Learning — {FAMILY_NAMES[family]}')
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle('Early-Phase Learning Dynamics (First 64K Rounds)', fontsize=14, y=1.02)
    fig.tight_layout()
    fname = f'{OUTDIR}/paper_early_zoom.pdf'
    fig.savefig(fname)
    plt.close(fig)
    print(f'  Saved {fname}')


plot_paper_v1_triptych()
plot_paper_v3_triptych()
plot_paper_handoff_comparison()
plot_paper_summary_table()
plot_paper_zoom_early()


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5: Summary statistics table (printed)
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "="*100)
print("SUMMARY TABLE")
print("="*100)
header = f"{'Model':<30s} {'Router':>8s} {'LLM':>8s} {'Cold':>8s} {'Lift':>8s} {'Mult':>8s} {'LLM%':>8s} {'τ_batch':>8s} {'ε_tail':>8s}"
print(header)
print("-"*100)
for (family, size), d in sorted(data.items()):
    name = PRETTY_NAMES[(family, size)]
    r = d['hybrid'][-1]
    l = d['llm'][-1]
    c = d['linucb_cold'][-1]
    lift = r - c
    mult = r / l if l > 0 else float('nan')
    frac = d['llm_fraction'][-1]
    tau = d.get('tau_batch')
    tail = d['tail_gap_cumulative'][-1]
    tail_str = f'{tail:.4f}' if tail is not None else 'N/A'
    tau_str = str(tau) if tau is not None else 'N/A'
    print(f"{name:<30s} {r:>8.4f} {l:>8.4f} {c:>8.4f} {lift:>+8.4f} {mult:>8.3f}x {frac:>7.1%} {tau_str:>8s} {tail_str:>8s}")

print("\nAll figures saved to:", os.path.abspath(OUTDIR))
print(f"Total files: {len(os.listdir(OUTDIR))}")
