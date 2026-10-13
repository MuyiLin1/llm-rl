#!/bin/bash
# Second-channel pipeline, unattended:
#   Test 3: transfer decomposition from the logs (fits theta_or, saves it)
#   Test 2: 30 linearized-reward runs, noise = oracle residual std, chained after.
set -u

# Portable paths: override these if your layout differs.
#   REPO   the repository root            (default: two levels above this script)
#   STUDY  where result files are written (default: $REPO/amazoncat13k/study)
#   PY     the python interpreter         (default: python3)
REPO=${REPO:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
STUDY=${STUDY:-"$REPO/amazoncat13k/study"}
PY=${PY:-python3}

cd "$REPO/amazoncat13k"
PY="$PY"
echo "=== Test 3 start $(date +%H:%M:%S) ==="
$PY -u transfer_decomp.py --slim amazon_v1_slim.npz --logdir study --out transfer_decomp_v1.json \
    > logs/transfer_decomp.log 2>&1
echo "=== Test 3 done $(date +%H:%M:%S) ==="
[ -f theta_or_v1.npy ] || { echo "NO ORACLE -- aborting Test 2"; exit 1; }
SIG=$($PY -c "import json;print(round(json.load(open('oracle_stats_v1.json'))['resid_std'],4))")
echo "=== Test 2: linearized rewards, sigma=$SIG, start $(date +%H:%M:%S) ==="
cd study; ln -sf ../theta_or_v1.npy . 2>/dev/null
build () { for s in 1 2 3 4 5 6 7 8 9 10; do for m in small base large; do
  grep -q '^FINAL' logs/run_linear_${m}_s${s}.log 2>/dev/null || echo "$m $s"; done; done; }
for att in 1 2 3 4 5; do
  mapfile -t T < <(build)
  echo "  sweep $att: ${#T[@]} outstanding at $(date +%H:%M:%S)"
  [ ${#T[@]} -eq 0 ] && break
  printf '%s\n' "${T[@]}" | xargs -P 10 -n 2 bash -c '
    OMP_NUM_THREADS=1 "$PY" -u experiment_algorithm1.py \
      --slim amazon_v1_slim.npz --dataset_name linear --model_family v1 --llm_type $0 --seed $1 \
      --llm_confidence 100 --passive_full --alpha_decay 0.9985 --decay_per round --device cuda \
      --max_samples 50000 --resume --reward linear --theta_or theta_or_v1.npy --noise '"$SIG"' \
      > logs/run_linear_$0_s$1.log 2>&1'
done
[ $(build | wc -l) -eq 0 ] && echo "CHANNEL2_COMPLETE $(date +%H:%M:%S)" || echo "STILL_INCOMPLETE"
