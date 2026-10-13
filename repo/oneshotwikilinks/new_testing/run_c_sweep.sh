#!/bin/bash
# C-value sweep for Flan-T5-base with passive_full=True
# Tests c ∈ {100, 1000, 5000, 10000} to see how longer LLM phases affect lift
#
# Output files:
#   results_algo1_base_c100.0_pfull.json
#   results_algo1_base_c1000.0_pfull.json
#   results_algo1_base_c5000.0_pfull.json
#   results_algo1_base_c10000.0_pfull.json

PYTHON=/Users/linmuyi/miniconda3/envs/dmwithllm/bin/python
SCRIPT=experiment_algorithm1.py
CACHE=llm_cache_v1_base.pkl.gz
MAX_JOBS=2

cd "$(dirname "$0")"

# Cleanup on Ctrl+C
trap 'echo "Killing background jobs..."; kill $(jobs -rp) 2>/dev/null; exit 1' INT TERM

C_VALUES=(100 1000 5000 10000)

running=0
for c in "${C_VALUES[@]}"; do
    echo "=== Starting: Flan-T5-base, c=$c, passive_full=True ==="
    $PYTHON -u $SCRIPT \
        --llm_type base \
        --llm_confidence $c \
        --passive_full \
        --llm_cache $CACHE \
        --seed 1 &
    running=$((running + 1))

    # Wait if we've hit max concurrent jobs
    if [ $running -ge $MAX_JOBS ]; then
        # Wait for any one job to finish
        while true; do
            n_running=$(jobs -rp | wc -l | tr -d ' ')
            if [ "$n_running" -lt "$MAX_JOBS" ]; then
                running=$n_running
                break
            fi
            sleep 5
        done
    fi
done

# Wait for all remaining jobs
echo "Waiting for remaining jobs to finish..."
wait
echo "=== All c-sweep runs complete ==="

# Quick summary
echo ""
echo "=== RESULTS SUMMARY ==="
for c in "${C_VALUES[@]}"; do
    f="results_algo1_base_c${c}.0_pfull.json"
    if [ -f "$f" ]; then
        $PYTHON -c "
import json
with open('$f') as fp:
    d = json.load(fp)
router = d['hybrid'][-1]
cold = d['linucb_cold'][-1]
llm = d['llm'][-1]
llm_frac = d['llm_fraction'][-1]
print(f'c={$c:6d}: Router={router*100:.3f}%  Cold={cold*100:.3f}%  LLM={llm*100:.3f}%  Lift={((router-cold)*100):+.3f}pp  LLMfrac={llm_frac*100:.1f}%')
"
    else
        echo "c=$c: FILE NOT FOUND ($f)"
    fi
done
