#!/bin/bash
# Full c-sweep across all 6 models with passive_full=True
# v1: Flan-T5 small/base/large  (experiment_algorithm1.py)
# v3: Qwen2.5 0.5B/3B/7B        (experiment_algorithm1_v3.py)
# c ∈ {100, 1000, 5000, 10000}
# Total: 6 models × 4 c values = 24 experiments (seed=1 only)

PYTHON=/Users/linmuyi/miniconda3/envs/dmwithllm/bin/python
MAX_JOBS=2

cd "$(dirname "$0")"

trap 'echo "Killing background jobs..."; kill $(jobs -rp) 2>/dev/null; exit 1' INT TERM

C_VALUES=(100 1000 5000 10000)

wait_for_slot() {
    while true; do
        n_running=$(jobs -rp | wc -l | tr -d ' ')
        if [ "$n_running" -lt "$MAX_JOBS" ]; then
            break
        fi
        sleep 5
    done
}

# ─── v1: Flan-T5 (small, base, large) ───
for size in small base large; do
    for c in "${C_VALUES[@]}"; do
        outfile="results_algo1_${size}_c${c}.0_pfull.json"
        if [ -f "$outfile" ]; then
            echo "SKIP (exists): $outfile"
            continue
        fi
        wait_for_slot
        echo "=== START: v1 Flan-T5-${size}, c=${c}, passive_full=True ==="
        $PYTHON -u experiment_algorithm1.py \
            --llm_type $size \
            --llm_confidence $c \
            --passive_full \
            --llm_cache llm_cache_v1_${size}.pkl.gz \
            --seed 1 &
    done
done

# ─── v3: Qwen2.5 (small=0.5B, base=3B, large=7B) ───
for size in small base large; do
    for c in "${C_VALUES[@]}"; do
        outfile="results_algo1v3_${size}_c${c}.0_pfull.json"
        if [ -f "$outfile" ]; then
            echo "SKIP (exists): $outfile"
            continue
        fi
        wait_for_slot
        echo "=== START: v3 Qwen2.5-${size}, c=${c}, passive_full=True ==="
        $PYTHON -u experiment_algorithm1_v3.py \
            --llm_type $size \
            --llm_confidence $c \
            --passive_full \
            --llm_cache llm_cache_v3_${size}.pkl.gz \
            --seed 1 &
    done
done

echo "Waiting for remaining jobs to finish..."
wait
echo ""
echo "=========================================="
echo "  ALL 24 EXPERIMENTS COMPLETE"
echo "=========================================="

# ─── Summary ───
echo ""
echo "=== v1 (Flan-T5) RESULTS ==="
printf "%-8s %6s  %10s  %10s  %10s  %8s  %8s\n" "Size" "c" "Router" "Cold" "LLM" "Lift" "LLMfrac"
for size in small base large; do
    for c in "${C_VALUES[@]}"; do
        f="results_algo1_${size}_c${c}.0_pfull.json"
        if [ -f "$f" ]; then
            $PYTHON -c "
import json
with open('$f') as fp:
    d = json.load(fp)
router = d['hybrid'][-1]
cold = d['linucb_cold'][-1]
llm = d['llm'][-1]
llm_frac = d['llm_fraction'][-1]
print(f'${size:8s} {$c:6d}  {router*100:10.3f}%  {cold*100:10.3f}%  {llm*100:10.3f}%  {(router-cold)*100:+7.3f}pp  {llm_frac*100:7.1f}%')
"
        fi
    done
done

echo ""
echo "=== v3 (Qwen2.5) RESULTS ==="
printf "%-8s %6s  %10s  %10s  %10s  %8s  %8s\n" "Size" "c" "Router" "Cold" "LLM" "Lift" "LLMfrac"
for size in small base large; do
    for c in "${C_VALUES[@]}"; do
        f="results_algo1v3_${size}_c${c}.0_pfull.json"
        if [ -f "$f" ]; then
            $PYTHON -c "
import json
with open('$f') as fp:
    d = json.load(fp)
router = d['hybrid'][-1]
cold = d['linucb_cold'][-1]
llm = d['llm'][-1]
llm_frac = d['llm_fraction'][-1]
print(f'${size:8s} {$c:6d}  {router*100:10.3f}%  {cold*100:10.3f}%  {llm*100:10.3f}%  {(router-cold)*100:+7.3f}pp  {llm_frac*100:7.1f}%')
"
        fi
    done
done
