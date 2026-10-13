"""
generate_plots_multiseed.py  –  Multi-seed figure generation (mean ± std).

Loads results for seeds 1, 2, 3 per experiment, computes pointwise mean
and standard deviation, and plots with shaded ±1 std bands.

Produces (saved to figures_multiseed/):
  1. v1 triptych (paper figure)    – mean learning curves with ±1 std bands
  2. v3 triptych (paper figure)    – mean learning curves with ±1 std bands
  3. Handoff comparison (2-panel)  – mean routing fraction with bands
  4. Final accuracy bars (2-panel) – mean ± std error bars
  5. Warm-start lift bars          – mean ± std error bars
  6. v1 routing overlay            – all 3 sizes, mean ± std bands
  7. v3 routing overlay            – all 3 sizes, mean ± std bands
  8. All 6 learning curves overlay – v1 dashed, v3 solid, mean ± std bands

Also prints summary tables with mean ± std for all metrics.

Usage:
    python generate_plots_multiseed.py
    python generate_plots_multiseed.py --seeds 1 2 3    # explicit seeds
"""

import argparse
import json
import os
import sys
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

OUTDIR = 'figures_multiseed'
os.makedirs(OUTDIR, exist_ok=True)

# ── Color palette ────────────────────────────────────────────────────
C_ROUTER = '#2166AC'
C_LLM    = '#B2182B'
C_COLD   = '#4DAF4A'
C_FRAC   = '#FF7F00'

SIZE_COLORS = {
    'small': '#1b9e77',
    'base':  '#d95f02',
    'large': '#7570b3',
}

FAMILY_STYLES = {
    'v1': {'ls': '--', 'alpha': 0.85},
    'v3': {'ls': '-',  'alpha': 1.0},
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

# ── Data loading ─────────────────────────────────────────────────────

def result_filename(family, size, seed, c=100.0):
    """Return the expected result filename for a given config and seed."""
    prefix = 'algo1' if family == 'v1' else 'algo1v3'
    tag = f"{prefix}_{size}_c{c}"
    if seed != 1:
        tag += f"_s{seed}"
    return f"results_{tag}.json"


def load_all_seeds(seeds):
    """Load results for all 6 configs × N seeds. Returns nested dict."""
    data = {}
    for family in ['v1', 'v3']:
        for size in ['small', 'base', 'large']:
            key = (family, size)
            seed_data = []
            for s in seeds:
                fname = result_filename(family, size, s)
                if not os.path.exists(fname):
                    print(f"  WARNING: {fname} not found, skipping seed {s} for {key}")
                    continue
                with open(fname) as f:
                    seed_data.append(json.load(f))
                print(f"  Loaded {fname}")
            if not seed_data:
                print(f"  ERROR: No data for {key}, skipping entirely")
                continue
            data[key] = seed_data
    return data


def compute_stats(seed_list, field):
    """Given a list of dicts (one per seed), extract field and compute mean/std."""
    arrays = []
    for d in seed_list:
        arr = np.array(d[field], dtype=float)
        arrays.append(arr)
    # Truncate to shortest length (should be equal, but safety)
    min_len = min(len(a) for a in arrays)
    stacked = np.stack([a[:min_len] for a in arrays], axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0)


def batches_to_rounds(batches):
    return np.array(batches) * 128


def pct_formatter(x, pos):
    return f'{x:.0%}' if x < 1 else f'{x:.1%}'


# ═══════════════════════════════════════════════════════════════════════
# Plotting helpers
# ═══════════════════════════════════════════════════════════════════════

def plot_mean_std(ax, x, mean, std, color, label, linestyle='-', linewidth=1.8, alpha_fill=0.15):
    """Plot a mean line with ±1 std shaded band."""
    ax.plot(x, mean, color=color, label=label, linestyle=linestyle, linewidth=linewidth)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=alpha_fill)


# ═══════════════════════════════════════════════════════════════════════
# Paper figures
# ═══════════════════════════════════════════════════════════════════════

def plot_triptych(data, family):
    """Three-panel learning curves (mean ± std) for one model family."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)

    for ax, size in zip(axes, ['small', 'base', 'large']):
        key = (family, size)
        if key not in data:
            ax.set_title(f'{PRETTY_NAMES[key]} (no data)')
            continue

        seeds = data[key]
        rounds = batches_to_rounds(np.array(seeds[0]['batch']))
        name = PRETTY_NAMES[key]

        h_mean, h_std = compute_stats(seeds, 'hybrid')
        l_mean, l_std = compute_stats(seeds, 'llm')
        c_mean, c_std = compute_stats(seeds, 'linucb_cold')
        rounds = rounds[:len(h_mean)]

        plot_mean_std(ax, rounds, h_mean, h_std, C_ROUTER, 'Router (Alg. 1)', linewidth=2)
        plot_mean_std(ax, rounds, l_mean, l_std, C_LLM, 'LLM Only', linestyle='--', linewidth=1.5)
        plot_mean_std(ax, rounds, c_mean, c_std, C_COLD, 'Cold LinUCB', linestyle=':', linewidth=1.5)

        ax.set_title(name, fontsize=12)
        ax.set_xlabel('Round')
        if ax is axes[0]:
            ax.set_ylabel('Cumulative Avg. Reward')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_xlim(0, rounds[-1])

        n_seeds = len(seeds)
        txt = (f'Router: {h_mean[-1]:.4f}±{h_std[-1]:.4f}\n'
               f'LLM: {l_mean[-1]:.4f}±{l_std[-1]:.4f}\n'
               f'Cold: {c_mean[-1]:.4f}±{c_std[-1]:.4f}\n'
               f'({n_seeds} seeds)')
        ax.text(0.97, 0.50, txt, transform=ax.transAxes, fontsize=7,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

    family_title = FAMILY_NAMES[family]
    fig.suptitle(f'OneShotWikiLinks-311 — {family_title}', fontsize=14, y=1.02)
    fig.tight_layout()
    fname = f'{OUTDIR}/paper_{family}_triptych.pdf'
    fig.savefig(fname)
    plt.close(fig)
    print(f'  Saved {fname}')


def plot_handoff_comparison(data):
    """Two-panel routing comparison: v1 left, v3 right (mean ± std)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)

    for ax, family in zip(axes, ['v1', 'v3']):
        for size in ['small', 'base', 'large']:
            key = (family, size)
            if key not in data:
                continue
            seeds = data[key]
            rounds = batches_to_rounds(np.array(seeds[0]['batch']))
            f_mean, f_std = compute_stats(seeds, 'llm_fraction')
            rounds = rounds[:len(f_mean)]
            label = PRETTY_NAMES[key].split('(')[0].strip()
            plot_mean_std(ax, rounds, f_mean, f_std, SIZE_COLORS[size], label, linewidth=2)

        ax.set_xlabel('Round')
        ax.set_ylabel('Fraction Routed to LLM')
        ax.set_title(FAMILY_NAMES[family])
        ax.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
        ax.set_xlim(0, rounds[-1])
        ax.set_ylim(-0.02, 1.05)
        ax.legend(loc='upper right')

    fig.suptitle('Confidence-Gated Hand-Off Dynamics (mean ± std)', fontsize=14, y=1.02)
    fig.tight_layout()
    fname = f'{OUTDIR}/paper_handoff_comparison.pdf'
    fig.savefig(fname)
    plt.close(fig)
    print(f'  Saved {fname}')


def plot_final_bars(data):
    """Two-panel bar chart: v1 left, v3 right, with error bars."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, family in zip(axes, ['v1', 'v3']):
        sizes = ['small', 'base', 'large']
        labels = [PRETTY_NAMES[(family, s)].split('(')[0].strip() for s in sizes]

        router_means, router_stds = [], []
        llm_means, llm_stds = [], []
        cold_means, cold_stds = [], []

        for s in sizes:
            key = (family, s)
            if key not in data:
                router_means.append(0); router_stds.append(0)
                llm_means.append(0); llm_stds.append(0)
                cold_means.append(0); cold_stds.append(0)
                continue
            seeds = data[key]
            finals_r = [d['hybrid'][-1] for d in seeds]
            finals_l = [d['llm'][-1] for d in seeds]
            finals_c = [d['linucb_cold'][-1] for d in seeds]
            router_means.append(np.mean(finals_r)); router_stds.append(np.std(finals_r))
            llm_means.append(np.mean(finals_l)); llm_stds.append(np.std(finals_l))
            cold_means.append(np.mean(finals_c)); cold_stds.append(np.std(finals_c))

        x = np.arange(3)
        w = 0.25

        b1 = ax.bar(x - w, router_means, w, yerr=router_stds, capsize=3,
                     label='Router', color=C_ROUTER, zorder=3)
        b2 = ax.bar(x,     llm_means,    w, yerr=llm_stds, capsize=3,
                     label='LLM Only', color=C_LLM, zorder=3)
        b3 = ax.bar(x + w, cold_means,   w, yerr=cold_stds, capsize=3,
                     label='Cold LinUCB', color=C_COLD, zorder=3)

        ax.set_ylabel('Final Cumulative Accuracy')
        ax.set_title(FAMILY_NAMES[family])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.legend(fontsize=8)
        ymax = max(max(np.array(router_means) + np.array(router_stds)),
                   max(np.array(llm_means) + np.array(llm_stds)),
                   max(np.array(cold_means) + np.array(cold_stds)))
        ax.set_ylim(0, ymax * 1.18)

        for bars, means, stds in [(b1, router_means, router_stds),
                                   (b2, llm_means, llm_stds),
                                   (b3, cold_means, cold_stds)]:
            for bar, m, s in zip(bars, means, stds):
                h = bar.get_height()
                ax.annotate(f'{m:.3f}±{s:.3f}',
                           xy=(bar.get_x() + bar.get_width()/2, h + s),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=6.5)

    fig.suptitle('Final Performance Across All Models (mean ± std)', fontsize=14, y=1.02)
    fig.tight_layout()
    fname = f'{OUTDIR}/paper_final_bars.pdf'
    fig.savefig(fname)
    plt.close(fig)
    print(f'  Saved {fname}')


def plot_lift_bars(data):
    """Warm-start lift (Router - Cold) bar chart with error bars."""
    keys = [('v1','small'),('v1','base'),('v1','large'),
            ('v3','small'),('v3','base'),('v3','large')]
    labels = [f"{PRETTY_NAMES[k].split('(')[0].strip()}\n({'Flan-T5' if k[0]=='v1' else 'Qwen2.5'})"
              for k in keys]
    lift_means, lift_stds = [], []
    for key in keys:
        if key not in data:
            lift_means.append(0); lift_stds.append(0)
            continue
        seeds = data[key]
        lifts = [d['hybrid'][-1] - d['linucb_cold'][-1] for d in seeds]
        lift_means.append(np.mean(lifts)); lift_stds.append(np.std(lifts))

    colors = [SIZE_COLORS[k[1]] for k in keys]
    hatches = ['///' if k[0] == 'v1' else '' for k in keys]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(keys)), lift_means, yerr=lift_stds, capsize=4,
                  color=colors, zorder=3, edgecolor='white')
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    ax.set_ylabel('Accuracy Lift (Router − Cold LinUCB)')
    ax.set_title('Warm-Start Lift Across All Models (mean ± std)')
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(0, color='black', linewidth=0.8)

    for i, (bar, m, s) in enumerate(zip(bars, lift_means, lift_stds)):
        ax.annotate(f'{m:+.4f}±{s:.4f}',
                   xy=(bar.get_x() + bar.get_width()/2, m + s if m >= 0 else m - s),
                   xytext=(0, 3 if m >= 0 else -12),
                   textcoords='offset points',
                   ha='center', va='bottom' if m >= 0 else 'top',
                   fontsize=7, fontweight='bold')

    fname = f'{OUTDIR}/warmstart_lift_all.pdf'
    fig.savefig(fname)
    plt.close(fig)
    print(f'  Saved {fname}')


def plot_routing_overlay(data, family):
    """Routing fraction overlay for one family (mean ± std)."""
    fig, ax = plt.subplots()

    for size in ['small', 'base', 'large']:
        key = (family, size)
        if key not in data:
            continue
        seeds = data[key]
        rounds = batches_to_rounds(np.array(seeds[0]['batch']))
        f_mean, f_std = compute_stats(seeds, 'llm_fraction')
        rounds = rounds[:len(f_mean)]
        label = PRETTY_NAMES[key].split('(')[0].strip()
        plot_mean_std(ax, rounds, f_mean, f_std, SIZE_COLORS[size], label, linewidth=2)

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


def plot_all_learning_curves(data):
    """All 6 models' Router track on one plot (v1 dashed, v3 solid), mean ± std."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for family in ['v1', 'v3']:
        for size in ['small', 'base', 'large']:
            key = (family, size)
            if key not in data:
                continue
            seeds = data[key]
            rounds = batches_to_rounds(np.array(seeds[0]['batch']))
            h_mean, h_std = compute_stats(seeds, 'hybrid')
            rounds = rounds[:len(h_mean)]
            ls = FAMILY_STYLES[family]['ls']
            name = PRETTY_NAMES[key]
            tag = 'Flan-T5' if family == 'v1' else 'Qwen2.5'
            plot_mean_std(ax, rounds, h_mean, h_std, SIZE_COLORS[size],
                         f'{name} ({tag})', linestyle=ls, linewidth=1.8,
                         alpha_fill=0.10)

    ax.set_xlabel('Round')
    ax.set_ylabel('Cumulative Average Reward (Precision@1)')
    ax.set_title('Router Performance — All Models (mean ± std)')
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    ax.set_xlim(0, rounds[-1])

    fname = f'{OUTDIR}/learning_curves_all_router.pdf'
    fig.savefig(fname)
    plt.close(fig)
    print(f'  Saved {fname}')


# ═══════════════════════════════════════════════════════════════════════
# Summary tables
# ═══════════════════════════════════════════════════════════════════════

def print_summary_tables(data):
    """Print summary tables with mean ± std for all metrics."""

    # ── v1 table ──
    print("\n" + "=" * 120)
    print("TABLE: Flan-T5 Results (mean ± std over seeds)")
    print("=" * 120)
    header = (f"{'Model':<25s} {'Router':>14s} {'LLM':>14s} {'Cold':>14s} "
              f"{'Lift':>14s} {'Mult':>14s} {'LLM%':>10s} {'τ_batch':>10s}")
    print(header)
    print("-" * 120)

    for size in ['small', 'base', 'large']:
        key = ('v1', size)
        if key not in data:
            continue
        seeds = data[key]
        name = PRETTY_NAMES[key]

        finals_r = [d['hybrid'][-1] for d in seeds]
        finals_l = [d['llm'][-1] for d in seeds]
        finals_c = [d['linucb_cold'][-1] for d in seeds]
        lifts = [r - c for r, c in zip(finals_r, finals_c)]
        mults = [r / l if l > 0 else float('nan') for r, l in zip(finals_r, finals_l)]
        fracs = [d['llm_fraction'][-1] for d in seeds]
        taus = [d.get('tau_batch') for d in seeds]
        tau_vals = [t for t in taus if t is not None]

        r_str = f"{np.mean(finals_r):.4f}±{np.std(finals_r):.4f}"
        l_str = f"{np.mean(finals_l):.4f}±{np.std(finals_l):.4f}"
        c_str = f"{np.mean(finals_c):.4f}±{np.std(finals_c):.4f}"
        lift_str = f"{np.mean(lifts):+.4f}±{np.std(lifts):.4f}"
        mult_str = f"{np.mean(mults):.3f}±{np.std(mults):.3f}x"
        frac_str = f"{np.mean(fracs):.1%}"
        tau_str = f"{np.mean(tau_vals):.0f}±{np.std(tau_vals):.0f}" if tau_vals else "—"

        print(f"{name:<25s} {r_str:>14s} {l_str:>14s} {c_str:>14s} "
              f"{lift_str:>14s} {mult_str:>14s} {frac_str:>10s} {tau_str:>10s}")

    # ── v3 table ──
    print("\n" + "=" * 120)
    print("TABLE: Qwen2.5 Results (mean ± std over seeds)")
    print("=" * 120)
    header = (f"{'Model':<25s} {'Router':>14s} {'LLM':>14s} {'Cold':>14s} "
              f"{'Lift':>14s} {'Mult':>14s} {'LLM%':>10s} {'τ_batch':>10s}")
    print(header)
    print("-" * 120)

    for size in ['small', 'base', 'large']:
        key = ('v3', size)
        if key not in data:
            continue
        seeds = data[key]
        name = PRETTY_NAMES[key]

        finals_r = [d['hybrid'][-1] for d in seeds]
        finals_l = [d['llm'][-1] for d in seeds]
        finals_c = [d['linucb_cold'][-1] for d in seeds]
        lifts = [r - c for r, c in zip(finals_r, finals_c)]
        mults = [r / l if l > 0 else float('nan') for r, l in zip(finals_r, finals_l)]
        fracs = [d['llm_fraction'][-1] for d in seeds]
        taus = [d.get('tau_batch') for d in seeds]
        tau_vals = [t for t in taus if t is not None]

        r_str = f"{np.mean(finals_r):.4f}±{np.std(finals_r):.4f}"
        l_str = f"{np.mean(finals_l):.4f}±{np.std(finals_l):.4f}"
        c_str = f"{np.mean(finals_c):.4f}±{np.std(finals_c):.4f}"
        lift_str = f"{np.mean(lifts):+.4f}±{np.std(lifts):.4f}"
        mult_str = f"{np.mean(mults):.3f}±{np.std(mults):.3f}x"
        frac_str = f"{np.mean(fracs):.1%}"
        tau_str = f"{np.mean(tau_vals):.0f}±{np.std(tau_vals):.0f}" if tau_vals else "—"

        print(f"{name:<25s} {r_str:>14s} {l_str:>14s} {c_str:>14s} "
              f"{lift_str:>14s} {mult_str:>14s} {frac_str:>10s} {tau_str:>10s}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-seed figure generation')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3],
                        help='Seeds to aggregate (default: 1 2 3)')
    args = parser.parse_args()

    print(f"Loading results for seeds: {args.seeds}")
    data = load_all_seeds(args.seeds)
    n_configs = len(data)
    print(f"\nLoaded {n_configs} configs with data.\n")

    if n_configs == 0:
        print("No data found. Exiting.")
        sys.exit(1)

    print("=== Paper figures (mean ± std) ===")
    plot_triptych(data, 'v1')
    plot_triptych(data, 'v3')
    plot_handoff_comparison(data)
    plot_final_bars(data)
    plot_lift_bars(data)
    plot_routing_overlay(data, 'v1')
    plot_routing_overlay(data, 'v3')
    plot_all_learning_curves(data)

    print_summary_tables(data)

    print(f"\nAll figures saved to: {os.path.abspath(OUTDIR)}")
    print(f"Total files: {len(os.listdir(OUTDIR))}")
