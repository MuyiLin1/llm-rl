"""
plot_results.py  (v2 directory)

Loads v2 experiment JSON files and produces comparison plots.
Auto-discovers results_algo1v2_*.json in the current directory,
or pass specific files with --files.

Generates:
  1. learning_curves.png        — cumulative reward over time
  2. final_comparison.png       — bar chart of final rewards
  3. routing_handoff.png        — cumulative LLM routing % over time
  4. per_interval_routing.png   — per-interval LLM% (shows the handoff transition)
  5. routing_stacked.png        — stacked area chart of LLM vs LinUCB picks per interval

Usage:
  python plot_results.py
  python plot_results.py --files results_algo1v2_small_c100.0_pfull.json results_algo1v2_base_c100.0_pfull.json
"""

import json
import glob
import argparse
import os
import numpy as np


def load_results(pattern="results_algo1v2_*.json"):
    files = glob.glob(pattern)
    results = {}
    for fp in sorted(files):
        with open(fp) as f:
            data = json.load(f)
        name = os.path.splitext(os.path.basename(fp))[0]
        results[name] = data
    return results


def friendly_name(raw_name):
    """Turn a filename-based key into a readable label."""
    # e.g. results_algo1v2_small_c100.0_pfull -> Flan-T5-Small (c=100)
    raw = raw_name.replace("results_algo1v2_", "").replace("results_algo1_", "")
    parts = raw.split("_")
    llm_size = parts[0].capitalize() if parts else raw
    c_val = ""
    for p in parts:
        if p.startswith("c"):
            try:
                c_val = f" (c={float(p[1:])})"
            except ValueError:
                pass
    return f"Flan-T5-{llm_size}{c_val}"


def print_diagnostics(name, data):
    config = data.get("config", {})
    print(f"\n{'='*70}")
    print(f"  {friendly_name(name)}")
    print(f"  ST: {config.get('sentence_transformer', '?')}  |  lam={config.get('lam','?')}  |  alpha_decay={config.get('alpha_decay','?')}")
    print(f"{'='*70}")

    batches = data.get("batch", [])
    if not batches:
        print("  (no data)")
        return

    print(f"  Final batch:       {batches[-1]}")
    if data.get("hybrid"):
        print(f"  Router reward:     {data['hybrid'][-1]:.4f}")
    if data.get("llm"):
        print(f"  LLM reward:        {data['llm'][-1]:.4f}")
    if data.get("linucb_cold"):
        print(f"  Cold LinUCB:       {data['linucb_cold'][-1]:.4f}")
    if data.get("llm_fraction"):
        print(f"  LLM routing frac:  {data['llm_fraction'][-1]:.1%}")
    if data.get("cum_llm_picks"):
        tl = data["cum_llm_picks"][-1]
        tu = data["cum_linucb_picks"][-1]
        tt = tl + tu
        print(f"  Total LLM picks:   {tl:,.0f}  ({tl/tt*100:.1f}%)")
        print(f"  Total LinUCB picks: {tu:,.0f}  ({tu/tt*100:.1f}%)")

    # Verdict
    if data.get("hybrid") and data.get("linucb_cold"):
        diff = data["hybrid"][-1] - data["linucb_cold"][-1]
        print(f"\n  Router − Cold LinUCB: {diff:+.4f}")
    if data.get("hybrid") and data.get("llm"):
        diff2 = data["hybrid"][-1] - data["llm"][-1]
        print(f"  Router − LLM:         {diff2:+.4f}")


def plot_curves(results, prefix=""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams.update({"font.size": 11})
    except ImportError:
        print("matplotlib not installed. Skipping plots.")
        return

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0", "#795548"]

    # ── Figure 1: Learning curves ──
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ci = 0
    cold_plotted = False
    for name, data in results.items():
        batches = data.get("batch", [])
        if not batches:
            continue
        color = colors[ci % len(colors)]
        label = friendly_name(name)

        if data.get("hybrid"):
            ax1.plot(batches, data["hybrid"], label=f"Router ({label})",
                     color=color, linewidth=2.2)
        if data.get("llm"):
            ax1.plot(batches, data["llm"], label=f"LLM ({label})",
                     color=color, linewidth=1.5, linestyle="--", alpha=0.7)
        if data.get("linucb_cold") and not cold_plotted:
            ax1.plot(batches, data["linucb_cold"], label="Cold LinUCB",
                     color="gray", linewidth=1.5, linestyle=":")
            cold_plotted = True
        ci += 1

    ax1.set_xlabel("Batch")
    ax1.set_ylabel("Cumulative Average Reward")
    ax1.set_title("v2 (all-mpnet-base-v2): Learning Curves")
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f"{prefix}learning_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    # ── Figure 2: Final comparison bar chart ──
    labels, router_vals, cold_vals, llm_vals = [], [], [], []
    for name, data in results.items():
        if not data.get("hybrid") or not data.get("linucb_cold"):
            continue
        labels.append(friendly_name(name))
        router_vals.append(data["hybrid"][-1])
        cold_vals.append(data["linucb_cold"][-1])
        llm_vals.append(data["llm"][-1] if data.get("llm") else 0)

    if labels:
        x = np.arange(len(labels))
        width = 0.25
        fig2, ax2 = plt.subplots(figsize=(max(8, 2.5 * len(labels)), 5))
        b1 = ax2.bar(x - width, router_vals, width, label="Router", color="#2196F3")
        b2 = ax2.bar(x, cold_vals, width, label="Cold LinUCB", color="#FF9800")
        b3 = ax2.bar(x + width, llm_vals, width, label="LLM Only", color="#9E9E9E")
        for bars in [b1, b2, b3]:
            for bar in bars:
                h = bar.get_height()
                ax2.annotate(f"{h:.4f}",
                             xy=(bar.get_x() + bar.get_width() / 2, h),
                             xytext=(0, 4), textcoords="offset points",
                             ha="center", va="bottom", fontsize=9)
        ax2.set_ylabel("Final Cumulative Average Reward")
        ax2.set_title("v2: Final Performance Comparison")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax2.legend()
        ax2.grid(True, axis="y", alpha=0.3)
        ax2.set_ylim(0, max(router_vals + cold_vals + llm_vals) * 1.15)
        plt.tight_layout()
        out = f"{prefix}final_comparison.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
        plt.close()

    # ── Figure 3: Cumulative LLM routing fraction (hand-off curve) ──
    routing_runs = {n: d for n, d in results.items() if d.get("llm_fraction")}
    if routing_runs:
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ci = 0
        for name, data in routing_runs.items():
            batches = data["batch"]
            fracs = data["llm_fraction"]
            ax3.plot(batches, [f * 100 for f in fracs],
                     label=friendly_name(name),
                     color=colors[ci % len(colors)], linewidth=2)
            ci += 1
        ax3.set_xlabel("Batch")
        ax3.set_ylabel("Cumulative LLM Routing %")
        ax3.set_title("Confidence-Gated Hand-Off: LLM → LinUCB (Cumulative)")
        ax3.set_ylim(-5, 105)
        ax3.axhline(y=50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        out = f"{prefix}routing_handoff.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
        plt.close()

    # ── Figure 4: Per-interval LLM routing % ──
    interval_runs = {n: d for n, d in results.items() if d.get("interval_llm_pct")}
    if interval_runs:
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        ci = 0
        for name, data in interval_runs.items():
            batches = data["batch"]
            pcts = data["interval_llm_pct"]
            ax4.plot(batches, pcts,
                     label=friendly_name(name),
                     color=colors[ci % len(colors)], linewidth=1.5, alpha=0.85)
            ci += 1
        ax4.set_xlabel("Batch")
        ax4.set_ylabel("LLM % (per 10-batch interval)")
        ax4.set_title("Per-Interval LLM Routing %  (each point = 1,280 samples)")
        ax4.set_ylim(-5, 105)
        ax4.axhline(y=50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax4.axhline(y=0, color="black", linewidth=0.5)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        out = f"{prefix}per_interval_routing.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
        plt.close()

    # ── Figure 5: Stacked area — LLM vs LinUCB picks per interval ──
    if interval_runs:
        n_runs = len(interval_runs)
        fig5, axes5 = plt.subplots(1, n_runs, figsize=(8 * n_runs, 4), squeeze=False)
        for idx, (name, data) in enumerate(interval_runs.items()):
            ax = axes5[0][idx]
            batches = np.array(data["batch"])
            llm_picks = np.array(data["interval_llm_picks"])
            linucb_picks = np.array(data["interval_linucb_picks"])

            ax.fill_between(batches, 0, llm_picks,
                            alpha=0.6, color="#E91E63", label="LLM Picks")
            ax.fill_between(batches, llm_picks, llm_picks + linucb_picks,
                            alpha=0.6, color="#2196F3", label="LinUCB Picks")
            ax.set_xlabel("Batch")
            ax.set_ylabel("Picks per Interval (of 1,280)")
            ax.set_title(f"{friendly_name(name)}: Routing per Interval")
            ax.legend(fontsize=9, loc="center right")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = f"{prefix}routing_stacked.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="*", default=None,
                        help="Specific JSON files. Default: all results_algo1v2_*.json")
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--output_prefix", default="",
                        help="Prefix for output PNG filenames (e.g. 'small_')")
    args = parser.parse_args()

    if args.files:
        results = {}
        for fp in args.files:
            with open(fp) as f:
                data = json.load(f)
            results[os.path.splitext(os.path.basename(fp))[0]] = data
    else:
        results = load_results()

    if not results:
        print("No result files found. Run an experiment first or pass --files.")
        return

    print(f"\nFound {len(results)} result file(s):")
    for name in results:
        print(f"  - {friendly_name(name)}")

    for name, data in results.items():
        print_diagnostics(name, data)

    if not args.no_plot:
        plot_curves(results, prefix=args.output_prefix)


if __name__ == "__main__":
    main()
