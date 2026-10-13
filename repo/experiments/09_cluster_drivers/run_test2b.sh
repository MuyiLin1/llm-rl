#!/bin/bash
# Test 2b: linearized rewards at noise levels where LinUCB can actually learn
# (sigma 0.001, 0.01; the linear signal on a correct action is ~0.0025), so the
# expert-independence prediction is tested where a warm start can matter.
set -u

# Portable paths: override these if your layout differs.
#   REPO   the repository root            (default: two levels above this script)
#   STUDY  where result files are written (default: $REPO/amazoncat13k/study)
#   PY     the python interpreter         (default: python3)
REPO=${REPO:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
STUDY=${STUDY:-"$REPO/amazoncat13k/study"}
PY=${PY:-python3}

cd "$REPO/amazoncat13k"
until grep -q "TEST4_COMPLETE\|STILL_INCOMPLETE" test4_driver.log 2>/dev/null; do sleep 120; done
echo "=== Test 2b start $(date +%H:%M:%S) ==="
cd study
build () { for sg in 0.001 0.01; do for s in 1 2 3 4 5; do for m in small base large; do
  grep -q '^FINAL' logs/run_lin${sg}_${m}_s${s}.log 2>/dev/null || echo "$m $s $sg"; done; done; done; }
for att in 1 2 3 4 5; do
  mapfile -t T < <(build); echo "  sweep $att: ${#T[@]} outstanding at $(date +%H:%M:%S)"
  [ ${#T[@]} -eq 0 ] && break
  printf '%s\n' "${T[@]}" | xargs -P 10 -n 3 bash -c '
    OMP_NUM_THREADS=1 "$PY" -u experiment_algorithm1.py \
      --slim amazon_v1_slim.npz --dataset_name lin$2 --model_family v1 --llm_type $0 --seed $1 \
      --llm_confidence 100 --passive_full --alpha_decay 0.9985 --decay_per round --device cuda \
      --max_samples 50000 --resume --reward linear --theta_or theta_or_v1.npy --noise $2 \
      > logs/run_lin$2_$0_s$1.log 2>&1'
done
[ $(build | wc -l) -eq 0 ] && echo "TEST2B_COMPLETE $(date +%H:%M:%S)" || echo "STILL_INCOMPLETE"
