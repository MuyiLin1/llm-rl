#!/bin/bash
# Test 4: real binary rewards everywhere, but routed rounds feed the learner
# <theta_or, phi> -- the expert's rounds contribute exactly M theta_or, no transfer.
# Waits for Test 2 to finish so the two don't compete for the GPU.
set -u

# Portable paths: override these if your layout differs.
#   REPO   the repository root            (default: two levels above this script)
#   STUDY  where result files are written (default: $REPO/amazoncat13k/study)
#   PY     the python interpreter         (default: python3)
REPO=${REPO:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
STUDY=${STUDY:-"$REPO/amazoncat13k/study"}
PY=${PY:-python3}

cd "$REPO/amazoncat13k"
until grep -q "CHANNEL2_COMPLETE\|STILL_INCOMPLETE" channel2_driver.log 2>/dev/null; do sleep 120; done
echo "=== Test 4 start $(date +%H:%M:%S) ==="
cd study; cp ../experiment_algorithm1.py . 2>/dev/null; ln -sf ../theta_or_v1.npy . 2>/dev/null
build () { for s in 1 2 3 4 5 6 7 8 9 10; do for m in small base large; do
  grep -q '^FINAL' logs/run_nom_${m}_s${s}.log 2>/dev/null || echo "$m $s"; done; done; }
for att in 1 2 3 4 5; do
  mapfile -t T < <(build); echo "  sweep $att: ${#T[@]} outstanding at $(date +%H:%M:%S)"
  [ ${#T[@]} -eq 0 ] && break
  printf '%s\n' "${T[@]}" | xargs -P 10 -n 2 bash -c '
    OMP_NUM_THREADS=1 "$PY" -u experiment_algorithm1.py \
      --slim amazon_v1_slim.npz --dataset_name nom --model_family v1 --llm_type $0 --seed $1 \
      --llm_confidence 100 --passive_full --alpha_decay 0.9985 --decay_per round --device cuda \
      --max_samples 50000 --resume --routed_reward linear --theta_or theta_or_v1.npy \
      > logs/run_nom_$0_s$1.log 2>&1'
done
[ $(build | wc -l) -eq 0 ] && echo "TEST4_COMPLETE $(date +%H:%M:%S)" || echo "STILL_INCOMPLETE"
