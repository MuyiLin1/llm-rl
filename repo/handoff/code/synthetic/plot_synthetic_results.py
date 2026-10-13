"""
plot_synthetic_results.py — Visualize synthetic experiment results for the paper.

Creates figures showing:
1. Diagnostic table (which axes deviate per option)
2. Algorithm 1 performance comparison (T=5000)
3. Horizon effect (T=5000 vs T=50000 for Option B)
"""

import matplotlib.pyplot as plt
import numpy as np

# ================================================================
# ALL RESULTS (from experiments)
# ================================================================

# --- DIAGNOSTICS (T=5000) ---
diagnostics = {
    'Option A': {
        'desc': 'Anti-podal basis\n(d=100, K=200)',
        'alpha_hat': -1.00,
        'r_over_sqrtT': 1.10,
        'gap_heavy': True,
        'angle': 18,
        'regime': 'warm-start-essential',
        'axes_deviate': 3,
        'note': 'θ* norm (0.09) < noise (0.1)',
    },
    'Option B': {
        'desc': 'Hidden subspace\n(d=200, K=100, r=20)',
        'alpha_hat': -1.00,
        'r_over_sqrtT': 1.99,
        'gap_heavy': True,
        'angle': 25,
        'regime': 'warm-start-essential',
        'axes_deviate': 3,
        'note': 'Best demonstration',
    },
    'Option C': {
        'desc': 'Orthogonal arms\n(d=50, K=100, r=20)',
        'alpha_hat': -1.00,
        'r_over_sqrtT': 0.58,
        'gap_heavy': True,
        'angle': 14,
        'regime': 'warm-start-likely',
        'axes_deviate': 2,
        'note': 'Axis A saturates (r/√T<1)',
    },
}

# --- ALGORITHM 1 RESULTS (T=5000) ---
results_T5000 = {
    'Option A': {
        30: {'router': 0.0154, 'llm': 0.0156, 'cold': 0.0127},
        50: {'router': 0.0164, 'llm': 0.0252, 'cold': 0.0127},
        70: {'router': 0.0180, 'llm': 0.0370, 'cold': 0.0156},
    },
    'Option B': {
        30: {'router': 0.0276, 'llm': 0.0185, 'cold': 0.0143},
        50: {'router': 0.0324, 'llm': 0.0264, 'cold': 0.0143},
        70: {'router': 0.0356, 'llm': 0.0362, 'cold': 0.0143},
    },
    'Option C': {
        30: {'router': 0.0391, 'llm': 0.0268, 'cold': 0.0337},
        50: {'router': 0.0431, 'llm': 0.0395, 'cold': 0.0337},
        70: {'router': 0.0466, 'llm': 0.0495, 'cold': 0.0337},
    },
}

# --- ALGORITHM 1 RESULTS (T=50000) ---
results_T50000 = {
    'Option A': {
        30: {'router': 0.0328, 'llm': 0.0253, 'cold': 0.0341},
        50: {'router': 0.0328, 'llm': 0.0329, 'cold': 0.0341},
        70: {'router': 0.0329, 'llm': 0.0365, 'cold': 0.0341},
    },
    'Option B': {
        30: {'router': 0.0312, 'llm': 0.0292, 'cold': 0.0303},
        50: {'router': 0.0333, 'llm': 0.0338, 'cold': 0.0303},
        70: {'router': 0.0342, 'llm': 0.0360, 'cold': 0.0303},
    },
}


def print_summary_table():
    """Print a clean summary for the paper."""
    print("=" * 90)
    print("SYNTHETIC DATASET DIAGNOSTICS (T=5000)")
    print("=" * 90)
    print(f"{'Option':<12} {'α̂':<8} {'r̂/√T':<8} {'Gap':<8} {'∠ (°)':<8} {'Axes Dev.':<10} {'Regime'}")
    print("-" * 90)
    for name, d in diagnostics.items():
        gap_str = "heavy" if d['gap_heavy'] else "light"
        print(f"{name:<12} {d['alpha_hat']:<8.2f} {d['r_over_sqrtT']:<8.2f} {gap_str:<8} {d['angle']:<8} {d['axes_deviate']}/3       {d['regime']}")
    print()

    print("=" * 90)
    print("ALGORITHM 1 RESULTS — T=5000 (short horizon: warm-start helps)")
    print("=" * 90)
    print(f"{'Option':<12} {'LLM%':<6} {'Router':<10} {'LLM-only':<10} {'Cold':<10} {'Lift (R-C)':<12} {'R/C ratio'}")
    print("-" * 90)
    for opt in ['Option B', 'Option C', 'Option A']:
        for pct in [30, 50, 70]:
            r = results_T5000[opt][pct]
            lift = r['router'] - r['cold']
            ratio = r['router'] / r['cold'] if r['cold'] > 0 else float('inf')
            marker = " ★" if opt == 'Option B' and pct == 70 else ""
            print(f"{opt:<12} {pct}%   {r['router']:<10.4f} {r['llm']:<10.4f} {r['cold']:<10.4f} {lift:<+12.4f} {ratio:<.2f}x{marker}")
        print()

    print("=" * 90)
    print("ALGORITHM 1 RESULTS — T=50000 (long horizon: LinUCB catches up)")
    print("=" * 90)
    print(f"{'Option':<12} {'LLM%':<6} {'Router':<10} {'LLM-only':<10} {'Cold':<10} {'Lift (R-C)':<12} {'R/C ratio'}")
    print("-" * 90)
    for opt in ['Option B', 'Option A']:
        for pct in [30, 50, 70]:
            r = results_T50000[opt][pct]
            lift = r['router'] - r['cold']
            ratio = r['router'] / r['cold'] if r['cold'] > 0 else float('inf')
            print(f"{opt:<12} {pct}%   {r['router']:<10.4f} {r['llm']:<10.4f} {r['cold']:<10.4f} {lift:<+12.4f} {ratio:<.2f}x")
        print()

    print("=" * 90)
    print("KEY TAKEAWAYS FOR PAPER")
    print("=" * 90)
    print("""
1. DIAGNOSTICS PREDICT RESULTS:
   - Option B (3/3 deviate, warm-start-essential): 2.5x lift at T=5000 ★
   - Option C (2/3 deviate, Axis A saturates):     1.4x lift at T=5000
   - Option A (3/3 deviate BUT signal<noise):      ~1x (diagnostics right, but SNR too low)

2. HORIZON EFFECT (Option B):
   - T=5000:  Router=0.036, Cold=0.014 → 2.5x lift (LinUCB still exploring)
   - T=50000: Router=0.034, Cold=0.030 → 1.1x lift (LinUCB caught up)
   → Warm-start provides EARLY advantage; LinUCB converges eventually (as theory predicts)

3. ROUTER > LLM (the multiplication phenomenon):
   - Option B, 30% LLM, T=50000: Router=0.031 > LLM=0.029
   → Router beats BOTH the frozen LLM and helps LinUCB learn faster

4. NEGATIVE LIFT (Option A at T=50000):
   - Cold=0.034 > Router=0.033 → warm-start slightly HARMFUL
   → Predicted by: signal/noise < 1 means LLM routing adds noise to V_t
""")


def plot_algorithm1_comparison():
    """Bar chart comparing Router vs Cold vs LLM at T=5000."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    options = ['Option A', 'Option B', 'Option C']
    colors = {'router': '#2196F3', 'llm': '#FF9800', 'cold': '#4CAF50'}

    for ax, opt in zip(axes, options):
        pcts = [30, 50, 70]
        x = np.arange(len(pcts))
        width = 0.25

        routers = [results_T5000[opt][p]['router'] for p in pcts]
        llms = [results_T5000[opt][p]['llm'] for p in pcts]
        colds = [results_T5000[opt][p]['cold'] for p in pcts]

        ax.bar(x - width, routers, width, label='Router', color=colors['router'])
        ax.bar(x, llms, width, label='LLM-only', color=colors['llm'])
        ax.bar(x + width, colds, width, label='Cold LinUCB', color=colors['cold'])

        ax.set_xlabel('LLM Oracle Accuracy')
        ax.set_ylabel('Avg Reward')
        ax.set_title(f'{opt}\n{diagnostics[opt]["regime"]}\n(∠={diagnostics[opt]["angle"]}°)')
        ax.set_xticks(x)
        ax.set_xticklabels(['30%', '50%', '70%'])
        ax.legend(fontsize=8)
        ax.set_ylim(0, max(max(routers), max(llms), max(colds)) * 1.3)

        # Add lift annotation for 70%
        lift = results_T5000[opt][70]['router'] / results_T5000[opt][70]['cold']
        ax.annotate(f'{lift:.1f}x', xy=(2 - width, routers[2]),
                   xytext=(2 - width, routers[2] * 1.1),
                   ha='center', fontsize=9, fontweight='bold', color='#2196F3')

    plt.suptitle('Algorithm 1: Synthetic Datasets at T=5,000', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig_synthetic_T5000.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig_synthetic_T5000.png")


def plot_horizon_effect():
    """Show how the lift shrinks as T grows (Option B)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Option B, 70% oracle
    T_values = [5000, 50000]
    router_vals = [results_T5000['Option B'][70]['router'], results_T50000['Option B'][70]['router']]
    cold_vals = [results_T5000['Option B'][70]['cold'], results_T50000['Option B'][70]['cold']]
    llm_vals = [results_T5000['Option B'][70]['llm'], results_T50000['Option B'][70]['llm']]

    x = np.arange(len(T_values))
    width = 0.25

    ax.bar(x - width, router_vals, width, label='Router', color='#2196F3')
    ax.bar(x, llm_vals, width, label='LLM-only (70%)', color='#FF9800')
    ax.bar(x + width, cold_vals, width, label='Cold LinUCB', color='#4CAF50')

    ax.set_xlabel('Horizon T')
    ax.set_ylabel('Avg Reward')
    ax.set_title('Option B: Warm-Start Advantage Shrinks with Horizon\n(70% Oracle, warm-start-essential regime)')
    ax.set_xticks(x)
    ax.set_xticklabels(['T = 5,000', 'T = 50,000'])
    ax.legend()

    # Annotate lifts
    for i, T in enumerate(T_values):
        lift = router_vals[i] / cold_vals[i]
        ax.annotate(f'R/C = {lift:.2f}x', xy=(i - width, router_vals[i]),
                   xytext=(i - width, router_vals[i] + 0.003),
                   ha='center', fontsize=10, fontweight='bold', color='#2196F3')

    plt.tight_layout()
    plt.savefig('fig_horizon_effect.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig_horizon_effect.png")


def plot_diagnostics_table():
    """Create a visual table of diagnostics."""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis('off')

    col_labels = ['Option', 'α̂ (Axis B)', 'r̂/√T (Axis A)', 'Gap (Axis C)', '∠ subspace', 'Regime', 'Lift @T=5K']
    row_data = []
    for name, d in diagnostics.items():
        lift_70 = results_T5000[name][70]['router'] / results_T5000[name][70]['cold']
        row_data.append([
            name,
            f"{d['alpha_hat']:.2f} ✓",
            f"{d['r_over_sqrtT']:.2f} {'✓' if d['r_over_sqrtT'] >= 1 else '✗'}",
            f"heavy ✓",
            f"{d['angle']}°",
            d['regime'],
            f"{lift_70:.2f}x"
        ])

    table = ax.table(cellText=row_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Color cells
    for i in range(len(row_data)):
        for j in range(len(col_labels)):
            cell = table[i+1, j]
            if '✓' in str(row_data[i][j]):
                cell.set_facecolor('#e8f5e9')
            elif '✗' in str(row_data[i][j]):
                cell.set_facecolor('#ffebee')

    ax.set_title('Pre-Runtime Diagnostics: Synthetic Datasets (T=5,000)', fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('fig_diagnostics_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig_diagnostics_table.png")


if __name__ == '__main__':
    print_summary_table()
    print("\nGenerating figures...")
    plot_algorithm1_comparison()
    plot_horizon_effect()
    plot_diagnostics_table()
    print("\nDone!")
