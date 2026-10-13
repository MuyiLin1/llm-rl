#!/bin/bash
# run_extended_sweep_parallel.sh — Extended c-value sweep (2 jobs in parallel, with timing)
#
# v1 Flan-T5 large: c = 25000, 50000, 100000, 500000
# v3 Qwen 3B (base): c = 100, 200, 500, 1000, 5000
# v3 Qwen 7B (large): c = 100, 200, 500, 1000, 5000
# All with: passive_full=True, seed=1, using LLM caches

PYTHON="/Users/linmuyi/miniconda3/envs/dmwithllm/bin/python"
V1_SCRIPT="experiment_algorithm1.py"
V3_SCRIPT="experiment_algorithm1_v3.py"
MAX_PARALLEL=2

COMPLETED=0
SKIPPED=0
FAILED=0

# Track background PIDs
PIDS=()

wait_for_slot() {
    # Wait until fewer than MAX_PARALLEL jobs are running
    while [ ${#PIDS[@]} -ge $MAX_PARALLEL ]; do
        local new_pids=()
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
            fi
        done
        PIDS=("${new_pids[@]}")
        if [ ${#PIDS[@]} -ge $MAX_PARALLEL ]; then
            sleep 5
        fi
    done
}

wait_all() {
    for pid in "${PIDS[@]}"; do
        wait "$pid"
    done
    PIDS=()
}

run_v1() {
    local SIZE=$1 C=$2
    local CACHE="llm_cache_v1_${SIZE}.pkl.gz"
    local OUTFILE="results_algo1_${SIZE}_c${C}.0_pfull.json"
    
    if [ -f "$OUTFILE" ]; then
        local NBATCH=$($PYTHON -c "import json; d=json.load(open('$OUTFILE')); print(len(d['batch']))" 2>/dev/null)
        if [ "$NBATCH" -gt 480 ] 2>/dev/null; then
            echo "[$(date '+%H:%M:%S')] SKIP v1 ${SIZE} c=${C} (complete, ${NBATCH} batches)"
            SKIPPED=$((SKIPPED+1))
            return
        fi
    fi
    
    wait_for_slot
    
    echo "[$(date '+%H:%M:%S')] START v1 ${SIZE} c=${C} → ${OUTFILE}"
    (
        local t_start=$(date +%s)
        $PYTHON $V1_SCRIPT \
            --llm_type $SIZE \
            --llm_confidence $C \
            --passive_full \
            --seed 1 \
            --llm_cache $CACHE
        local t_end=$(date +%s)
        local elapsed=$(( (t_end - t_start) / 60 ))
        if [ $? -eq 0 ]; then
            echo "[$(date '+%H:%M:%S')] DONE  v1 ${SIZE} c=${C} (${elapsed} min)"
        else
            echo "[$(date '+%H:%M:%S')] FAIL  v1 ${SIZE} c=${C} (${elapsed} min)"
        fi
    ) &
    PIDS+=($!)
}

run_v3() {
    local SIZE=$1 C=$2
    local CACHE="llm_cache_v3_${SIZE}.pkl.gz"
    local OUTFILE="results_algo1v3_${SIZE}_c${C}.0_pfull.json"
    
    if [ -f "$OUTFILE" ]; then
        local NBATCH=$($PYTHON -c "import json; d=json.load(open('$OUTFILE')); print(len(d['batch']))" 2>/dev/null)
        if [ "$NBATCH" -gt 480 ] 2>/dev/null; then
            echo "[$(date '+%H:%M:%S')] SKIP v3 ${SIZE} c=${C} (complete, ${NBATCH} batches)"
            SKIPPED=$((SKIPPED+1))
            return
        fi
    fi
    
    wait_for_slot
    
    echo "[$(date '+%H:%M:%S')] START v3 ${SIZE} c=${C} → ${OUTFILE}"
    (
        local t_start=$(date +%s)
        $PYTHON $V3_SCRIPT \
            --llm_type $SIZE \
            --llm_confidence $C \
            --passive_full \
            --seed 1 \
            --llm_cache $CACHE
        local t_end=$(date +%s)
        local elapsed=$(( (t_end - t_start) / 60 ))
        if [ $? -eq 0 ]; then
            echo "[$(date '+%H:%M:%S')] DONE  v3 ${SIZE} c=${C} (${elapsed} min)"
        else
            echo "[$(date '+%H:%M:%S')] FAIL  v3 ${SIZE} c=${C} (${elapsed} min)"
        fi
    ) &
    PIDS+=($!)
}

echo "=============================================="
echo "  Extended c-value sweep (parallel, max ${MAX_PARALLEL} jobs)"
echo "  v1 large: c=25k,50k,100k,500k"
echo "  v3 base+large: c=100,200,500,1k,5k"
echo "  14 experiments total"
echo "  Started: $(date)"
echo "=============================================="
echo ""

SWEEP_START=$(date +%s)

# v1 Flan-T5 large
run_v1 large 25000
run_v1 large 50000
run_v1 large 100000
run_v1 large 500000

# v3 Qwen base (3B)
run_v3 base 100
run_v3 base 200
run_v3 base 500
run_v3 base 1000
run_v3 base 5000

# v3 Qwen large (7B)
run_v3 large 100
run_v3 large 200
run_v3 large 500
run_v3 large 1000
run_v3 large 5000

# Wait for all remaining jobs
wait_all

SWEEP_END=$(date +%s)
SWEEP_ELAPSED=$(( (SWEEP_END - SWEEP_START) / 60 ))

echo ""
echo "=============================================="
echo "  Sweep complete!"
echo "  Finished: $(date)"
echo "  Total time: ${SWEEP_ELAPSED} min"
echo "  Completed: ${COMPLETED} | Skipped: ${SKIPPED} | Failed: ${FAILED}"
echo "=============================================="
