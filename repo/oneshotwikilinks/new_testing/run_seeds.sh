#!/usr/bin/env bash
# run_seeds.sh — Run seed=2,3 experiments with max 2 concurrent jobs.
# Output prints directly to terminal. Ctrl+C kills all jobs.
#
# Usage:  cd oneshotwikilinks/new_testing && bash run_seeds.sh
#   or:   nohup bash run_seeds.sh > run_seeds_output.txt 2>&1 &
#
# Uses the dmwithllm conda python directly so no activation needed.

set -uo pipefail

PYTHON=/Users/linmuyi/miniconda3/envs/dmwithllm/bin/python
MAX_JOBS=2

cleanup() {
    echo ""
    echo "Caught interrupt — killing background jobs..."
    kill $(jobs -rp) 2>/dev/null
    wait 2>/dev/null
    exit 1
}
trap cleanup INT TERM

run_job() {
    local script="$1" llm_type="$2" seed="$3" cache="$4"
    echo "[$(date +%H:%M:%S)] START  $script --llm_type $llm_type --seed $seed"
    $PYTHON -u "$script" --llm_type "$llm_type" --seed "$seed" --llm_cache "$cache"
    echo "[$(date +%H:%M:%S)] DONE   $script --llm_type $llm_type --seed $seed"
}

wait_for_slot() {
    while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
        sleep 5
    done
}

JOBS=(
    "experiment_algorithm1.py    small 2 llm_cache_v1_small.pkl.gz"
    "experiment_algorithm1_v3.py small 2 llm_cache_v3_small.pkl.gz"
    "experiment_algorithm1.py    base  2 llm_cache_v1_base.pkl.gz"
    "experiment_algorithm1_v3.py base  2 llm_cache_v3_base.pkl.gz"
    "experiment_algorithm1.py    large 2 llm_cache_v1_large.pkl.gz"
    "experiment_algorithm1_v3.py large 2 llm_cache_v3_large.pkl.gz"
    "experiment_algorithm1.py    small 3 llm_cache_v1_small.pkl.gz"
    "experiment_algorithm1_v3.py small 3 llm_cache_v3_small.pkl.gz"
    "experiment_algorithm1.py    base  3 llm_cache_v1_base.pkl.gz"
    "experiment_algorithm1_v3.py base  3 llm_cache_v3_base.pkl.gz"
    "experiment_algorithm1.py    large 3 llm_cache_v1_large.pkl.gz"
    "experiment_algorithm1_v3.py large 3 llm_cache_v3_large.pkl.gz"
)

for job in "${JOBS[@]}"; do
    wait_for_slot
    # shellcheck disable=SC2086
    run_job $job &
done

wait
echo ""
echo "All 12 experiments complete."
