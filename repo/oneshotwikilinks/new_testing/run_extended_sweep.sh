#!/bin/bash
# run_extended_sweep.sh — Extended c-value sweep for v1 (Flan-T5) and v3 (Qwen)
#
# v1 Flan-T5 large: c = 25000, 50000, 100000, 500000
# v3 Qwen 3B (base): c = 100, 200, 500, 1000, 5000
# v3 Qwen 7B (large): c = 100, 200, 500, 1000, 5000
# All with: passive_full=True, seed=1, using LLM caches

PYTHON="/Users/linmuyi/miniconda3/envs/dmwithllm/bin/python"
V1_SCRIPT="experiment_algorithm1.py"
V3_SCRIPT="experiment_algorithm1_v3.py"

COMPLETED=0
SKIPPED=0
FAILED=0

run_v1() {
    local SIZE=$1 C=$2
    local CACHE="llm_cache_v1_${SIZE}.pkl.gz"
    local OUTFILE="results_algo1_${SIZE}_c${C}.0_pfull.json"
    if [ -f "$OUTFILE" ]; then
        local NBATCH=$($PYTHON -c "import json; d=json.load(open('$OUTFILE')); print(len(d['batch']))" 2>/dev/null)
        if [ "$NBATCH" -gt 4000 ] 2>/dev/null; then
            echo ">>> SKIP v1 ${SIZE} c=${C} (complete, ${NBATCH} batches)"
            SKIPPED=$((SKIPPED+1))
            return
        fi
    fi
    echo ""
    echo ">>> Running: v1 ${SIZE} c=${C} cache=${CACHE}"
    echo "    Output: ${OUTFILE}"
    $PYTHON $V1_SCRIPT \
        --llm_type $SIZE \
        --llm_confidence $C \
        --passive_full \
        --seed 1 \
        --llm_cache $CACHE
    if [ $? -eq 0 ]; then
        echo ">>> Done: v1 ${SIZE} c=${C}"
        COMPLETED=$((COMPLETED+1))
    else
        echo ">>> FAILED: v1 ${SIZE} c=${C}"
        FAILED=$((FAILED+1))
    fi
}

run_v3() {
    local SIZE=$1 C=$2
    local CACHE="llm_cache_v3_${SIZE}.pkl.gz"
    local OUTFILE="results_algo1v3_${SIZE}_c${C}.0_pfull.json"
    if [ -f "$OUTFILE" ]; then
        local NBATCH=$($PYTHON -c "import json; d=json.load(open('$OUTFILE')); print(len(d['batch']))" 2>/dev/null)
        if [ "$NBATCH" -gt 4000 ] 2>/dev/null; then
            echo ">>> SKIP v3 ${SIZE} c=${C} (complete, ${NBATCH} batches)"
            SKIPPED=$((SKIPPED+1))
            return
        fi
    fi
    echo ""
    echo ">>> Running: v3 ${SIZE} c=${C} cache=${CACHE}"
    echo "    Output: ${OUTFILE}"
    $PYTHON $V3_SCRIPT \
        --llm_type $SIZE \
        --llm_confidence $C \
        --passive_full \
        --seed 1 \
        --llm_cache $CACHE
    if [ $? -eq 0 ]; then
        echo ">>> Done: v3 ${SIZE} c=${C}"
        COMPLETED=$((COMPLETED+1))
    else
        echo ">>> FAILED: v3 ${SIZE} c=${C}"
        FAILED=$((FAILED+1))
    fi
}

echo "=============================================="
echo "  Extended c-value sweep"
echo "  v1 large: c=25k,50k,100k,500k"
echo "  v3 base+large: c=100,200,500,1k,5k"
echo "  14 experiments total"
echo "=============================================="

# --- v1 Flan-T5 large (high c) ---
for C in 25000 50000 100000 500000; do
    run_v1 large $C
done

# --- v3 Qwen 3B (base) extended c ---
for C in 100 200 500 1000 5000; do
    run_v3 base $C
done

# --- v3 Qwen 7B (large) extended c ---
for C in 100 200 500 1000 5000; do
    run_v3 large $C
done

echo ""
echo "=============================================="
echo "  Sweep complete!"
echo "  Completed: ${COMPLETED}"
echo "  Skipped:   ${SKIPPED}"
echo "  Failed:    ${FAILED}"
echo "=============================================="
