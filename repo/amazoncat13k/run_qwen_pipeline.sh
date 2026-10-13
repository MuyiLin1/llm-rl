#!/bin/bash
# run_qwen_pipeline.sh — Generate Qwen cache, then run Algorithm 1 experiment
#
# Usage:
#   tmux 1: bash run_qwen_pipeline.sh small large
#   tmux 2: bash run_qwen_pipeline.sh base
#
# Each LLM type: generate cache → run experiment (50K samples) → next type

set -e

cd /home/bruinml/DMwithLLM/amazoncat13k

for LLM_TYPE in "$@"; do
    echo ""
    echo "============================================================"
    echo "  PIPELINE: Qwen v3 / ${LLM_TYPE}"
    echo "  Started: $(date)"
    echo "============================================================"
    echo ""

    # Step 1: Generate Qwen cache (if not already done)
    CACHE_FILE="llm_cache_amazon_v3_${LLM_TYPE}.pkl.gz"
    if [ -f "$CACHE_FILE" ]; then
        echo "Cache ${CACHE_FILE} already exists, skipping generation."
    else
        echo "--- Step 1: Generating Qwen cache for ${LLM_TYPE} ---"
        python -u cache_llm_predictions_amazon.py \
            --model_family v3 \
            --llm_type "${LLM_TYPE}" \
            2>&1 | tee "cache_v3_${LLM_TYPE}.log"
        echo "Cache generation done: $(date)"
    fi

    # Step 2: Run Algorithm 1 experiment (50K samples)
    echo ""
    echo "--- Step 2: Running Algorithm 1 for v3/${LLM_TYPE} ---"
    python -u experiment_algorithm1.py \
        --model_family v3 \
        --llm_type "${LLM_TYPE}" \
        --llm_confidence 100 \
        --passive_full \
        --max_samples 50000 \
        2>&1 | tee "run_v3_${LLM_TYPE}.log"

    echo ""
    echo "Finished v3/${LLM_TYPE}: $(date)"
    echo "============================================================"
done

echo ""
echo "ALL DONE: $(date)"
