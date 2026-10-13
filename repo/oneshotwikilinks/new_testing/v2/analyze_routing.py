"""
Analyze per-batch LLM vs LinUCB routing from experiment JSON results.

Derives per-checkpoint (every 10 batches = 1,280 samples) how many times
the LLM was picked vs LinUCB, plus running cumulative totals.

Usage:
  python analyze_routing.py                           # print analysis for all v2 JSONs
  python analyze_routing.py --enrich                  # also write per-interval data into JSONs
  python analyze_routing.py --files f1.json f2.json   # specific files
"""

import json
import argparse
import os
import sys


def derive_per_interval(data, batch_size=128):
    """Derive per-interval routing counts from cumulative llm_fraction.

    Returns a dict with:
      - interval_llm_picks: per-interval LLM pick counts
      - interval_linucb_picks: per-interval LinUCB pick counts
      - interval_llm_pct: per-interval LLM percentage (0-100)
      - cum_llm_picks: cumulative LLM picks
      - cum_linucb_picks: cumulative LinUCB picks
    """
    batches = data["batch"]
    llm_frac = data["llm_fraction"]

    interval_llm_list = []
    interval_linucb_list = []
    interval_pct_list = []
    cum_llm_list = []
    cum_linucb_list = []

    cum_llm = 0.0
    cum_linucb = 0.0

    for i in range(len(batches)):
        b = batches[i]
        total_samples = (b + 1) * batch_size
        abs_llm = llm_frac[i] * total_samples

        if i == 0:
            interval_samples = batch_size
            interval_llm = abs_llm
        else:
            b_prev = batches[i - 1]
            total_prev = (b_prev + 1) * batch_size
            abs_llm_prev = llm_frac[i - 1] * total_prev
            interval_samples = total_samples - total_prev
            interval_llm = abs_llm - abs_llm_prev

        interval_linucb = interval_samples - interval_llm
        interval_pct = (interval_llm / interval_samples * 100) if interval_samples > 0 else 0

        cum_llm += interval_llm
        cum_linucb += interval_linucb

        interval_llm_list.append(round(interval_llm, 1))
        interval_linucb_list.append(round(interval_linucb, 1))
        interval_pct_list.append(round(interval_pct, 2))
        cum_llm_list.append(round(cum_llm, 1))
        cum_linucb_list.append(round(cum_linucb, 1))

    return {
        "interval_llm_picks": interval_llm_list,
        "interval_linucb_picks": interval_linucb_list,
        "interval_llm_pct": interval_pct_list,
        "cum_llm_picks": cum_llm_list,
        "cum_linucb_picks": cum_linucb_list,
    }


def analyze(filepath, enrich=False):
    with open(filepath) as f:
        data = json.load(f)

    config = data["config"]
    batch_size = 128

    derived = derive_per_interval(data, batch_size)

    llm_type = config.get("llm_type", "?")
    c = config.get("llm_confidence", "?")
    print(f"\n{'='*90}")
    print(f"  File: {os.path.basename(filepath)}")
    print(f"  LLM: flan-t5-{llm_type}  |  c={c}  |  ST: {config.get('sentence_transformer','?')}")
    print(f"{'='*90}\n")

    # Header
    print(f"{'Batch':>7s} | {'Interval':>10s} | {'LLM Picks':>10s} | {'LinUCB Picks':>12s} | "
          f"{'LLM %':>8s} | {'Cum LLM':>10s} | {'Cum LinUCB':>10s} | {'Cum LLM %':>10s}")
    print("-" * 100)

    batches = data["batch"]
    for i in range(len(batches)):
        b = batches[i]
        if i == 0:
            interval_samples = batch_size
        else:
            interval_samples = (b + 1) * batch_size - (batches[i-1] + 1) * batch_size

        il = derived["interval_llm_picks"][i]
        iu = derived["interval_linucb_picks"][i]
        ip = derived["interval_llm_pct"][i]
        cl = derived["cum_llm_picks"][i]
        cu = derived["cum_linucb_picks"][i]
        ct = cl + cu
        cp = (cl / ct * 100) if ct > 0 else 0

        print(f"{b:>7d} | {interval_samples:>10.0f} | {il:>10.1f} | {iu:>10.1f} | "
              f"{ip:>7.1f}% | {cl:>10.1f} | {cu:>10.1f} | {cp:>9.1f}%")

    # Final summary
    total_llm = derived["cum_llm_picks"][-1]
    total_linucb = derived["cum_linucb_picks"][-1]
    total = total_llm + total_linucb
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"  Total samples:     {total:,.0f}")
    print(f"  Total LLM picks:   {total_llm:,.0f}  ({total_llm/total*100:.2f}%)")
    print(f"  Total LinUCB picks: {total_linucb:,.0f}  ({total_linucb/total*100:.2f}%)")
    print(f"{'='*60}")

    # Key milestones
    print(f"\n  Key milestones:")
    pcts = derived["interval_llm_pct"]
    for i, p in enumerate(pcts):
        if p < 100:
            print(f"  - First handoff (LLM% < 100%): batch {batches[i]}  (LLM {p:.1f}%)")
            break
    for i, p in enumerate(pcts):
        if p < 50:
            print(f"  - Majority LinUCB (LLM% < 50%): batch {batches[i]}  (LLM {p:.1f}%)")
            break
    for i in range(len(pcts) - 1, -1, -1):
        if pcts[i] > 1.0:
            print(f"  - Last interval with LLM% > 1%: batch {batches[i]}  (LLM {pcts[i]:.1f}%)")
            break
    print()

    # Enrich the JSON with derived data
    if enrich:
        data["interval_llm_picks"] = derived["interval_llm_picks"]
        data["interval_linucb_picks"] = derived["interval_linucb_picks"]
        data["interval_llm_pct"] = derived["interval_llm_pct"]
        data["cum_llm_picks"] = derived["cum_llm_picks"]
        data["cum_linucb_picks"] = derived["cum_linucb_picks"]
        with open(filepath, "w") as f:
            json.dump(data, f)
        print(f"  >> Enriched {os.path.basename(filepath)} with per-interval routing data.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze LLM vs LinUCB routing per batch")
    parser.add_argument("--files", nargs="*", help="JSON result files to analyze")
    parser.add_argument("--enrich", action="store_true",
                        help="Write per-interval routing data back into the JSON files")
    args = parser.parse_args()

    if not args.files:
        files = sorted(f for f in os.listdir(".") if f.startswith("results_algo1v2") and f.endswith(".json"))
        if not files:
            print("No result files found. Pass file paths with --files.")
            sys.exit(1)
    else:
        files = args.files

    for f in files:
        analyze(f, enrich=args.enrich)
