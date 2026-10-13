#!/bin/bash
# run_v4_sweep.sh — SVD-reduced feature sweep (v4)
#
# Tests: 3 LLM sizes × 5 c values = 15 runs
# All with: passive_full=True, seed=1, svd_rank=50, batch_size=32
#
# With 50d features + batch_size=32, each run should be ~3-5x faster
# than the 768d experiments (smaller matrices, more batches but cheaper per batch).

PYTHON="/Users/linmuyi/miniconda3/envs/dmwithllm/bin/python"
SCRIPT="experiment_algorithm1_v4.py"
RANK=50
BS=32

echo "=============================================="
echo "  v4 SVD Sweep: rank=${RANK}, batch_size=${BS}"
echo "  15 experiments (3 sizes × 5 c values)"
echo "  Using LLM caches (no model loading needed)"
echo "=============================================="

for SIZE in small base large; do
    CACHE="llm_cache_v1_${SIZE}.pkl.gz"
    if [ ! -f "$CACHE" ]; then
        echo "ERROR: cache not found: $CACHE"
        exit 1
    fi
    for C in 10 50 100 500 1000; do
        OUTFILE="results_algo1v4_${SIZE}_c${C}.0_r${RANK}_pfull_bs${BS}.json"
        if [ -f "$OUTFILE" ]; then
            COMPLETE=$($PYTHON -c "
import json
with open('$OUTFILE') as f:
    d = json.load(f)
print('yes' if len(d['batch']) > 1000 else 'no')
" 2>/dev/null)
            if [ "$COMPLETE" = "yes" ]; then
                echo ">>> SKIP ${SIZE} c=${C} (already complete)"
                continue
            fi
        fi
        echo ""
        echo ">>> Running: ${SIZE} c=${C} rank=${RANK} bs=${BS} cache=${CACHE}"
        echo "    Output: ${OUTFILE}"
        $PYTHON $SCRIPT \
            --llm_type $SIZE \
            --llm_confidence $C \
            --passive_full \
            --svd_rank $RANK \
            --batch_size $BS \
            --seed 1 \
            --llm_cache $CACHE
        echo ">>> Done: ${SIZE} c=${C}"
    done
done

echo ""
echo "=============================================="
echo "  All v4 experiments complete!"
echo "=============================================="

# Print summary
$PYTHON -c "
import json, os

print()
print('=' * 110)
print('v4 SWEEP RESULTS (SVD-${RANK}d, batch_size=${BS}, passive_full=True)')
print('=' * 110)
print(f'{\"Model\":12s} {\"c\":>6s}  {\"Router\":>10s}  {\"Cold\":>10s}  {\"LLM\":>10s}  {\"Lift\":>8s}  {\"LLMfrac\":>8s}  {\"Batches\":>8s}')
for size, label in [('small','small(60M)'), ('base','base(250M)'), ('large','large(780M)')]:
    for c in [10, 50, 100, 500, 1000]:
        f = f'results_algo1v4_{size}_c{float(c)}_r${RANK}_pfull_bs${BS}.json'
        try:
            with open(f) as fp:
                d = json.load(fp)
            n = len(d['batch'])
            r = d['hybrid'][-1]; cold = d['linucb_cold'][-1]; llm = d['llm'][-1]; lf = d['llm_fraction'][-1]
            print(f'{label:12s} {c:6d}  {r*100:9.3f}%  {cold*100:9.3f}%  {llm*100:9.3f}%  {(r-cold)*100:+7.3f}pp  {lf*100:7.1f}%  {n:8d}')
        except:
            print(f'{label:12s} {c:6d}  --- missing or incomplete ---')
    print()
"
