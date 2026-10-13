#!/bin/bash
# run_cluster_v6.sh — Run synthetic text v6 experiments on bruinml cluster
#
# Usage (on cluster):
#   tmux new -s synth_v6
#   bash run_cluster_v6.sh
#
# This script:
#   1. Caches flan-t5-large predictions
#   2. Caches qwen-3b predictions
#   3. Caches qwen-7b predictions
#   4. Runs 3-seed c-sweep for all cached models

set -e

cd /home/bruinml/DMwithLLM/synthetic_text

echo "============================================================"
echo "  SYNTHETIC TEXT V6 — CLUSTER EXPERIMENTS"
echo "  Started: $(date)"
echo "============================================================"

# Step 1: Cache flan-t5-large
echo ""
echo "--- Step 1: Caching flan-t5-large ---"
if [ -f "v6_cache_flan-t5-large.pkl.gz" ]; then
    echo "  Cache exists, skipping."
else
    python -u run_experiment_v6.py --model flan-t5-large --num_seeds 3 2>&1 | tee run_cluster_flan_large.log
fi

# Step 2: Cache qwen-3b
echo ""
echo "--- Step 2: Caching qwen-3b ---"
if [ -f "v6_cache_qwen-3b.pkl.gz" ]; then
    echo "  Cache exists, skipping."
else
    python -u run_experiment_v6.py --model qwen-3b --num_seeds 3 2>&1 | tee run_cluster_qwen3b.log
fi

# Step 3: Cache qwen-7b
echo ""
echo "--- Step 3: Caching qwen-7b ---"
if [ -f "v6_cache_qwen-7b.pkl.gz" ]; then
    echo "  Cache exists, skipping."
else
    python -u run_experiment_v6.py --model qwen-7b --num_seeds 3 2>&1 | tee run_cluster_qwen7b.log
fi

# Step 4: Run 3-seed c-sweep on ALL available caches (in case local ones were copied over)
echo ""
echo "--- Step 4: Full 3-seed c-sweep on all cached models ---"
python -u run_experiment_v6.py --sweep_only --num_seeds 3 2>&1 | tee run_cluster_sweep_all.log

echo ""
echo "============================================================"
echo "  ALL DONE: $(date)"
echo "============================================================"
