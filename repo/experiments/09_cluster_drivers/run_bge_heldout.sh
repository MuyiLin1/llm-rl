#!/bin/bash
# Review item 4: c=0.1 was SELECTED on seeds 1-3 (run_bge.sh). Evaluate it, frozen,
# on ten fresh seeds 4-13 so tuning and evaluation never share a seed.
# AmazonCat BGE-1024d x 3 Qwen experts x 10 seeds = 30 runs.
set -u

# Portable paths: override these if your layout differs.
#   REPO   the repository root            (default: two levels above this script)
#   STUDY  where result files are written (default: $REPO/amazoncat13k/study)
#   PY     the python interpreter         (default: python3)
REPO=${REPO:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
STUDY=${STUDY:-"$REPO/amazoncat13k/study"}
PY=${PY:-python3}

PAR=${PAR:-10}
cd "$STUDY"
build () {
  for s in 4 5 6 7 8 9 10 11 12 13; do for m in small base large; do
    f=logs/run_amazonv3_${m}_c0.1_s${s}.log
    grep -q '^FINAL' $f 2>/dev/null || echo "$m $s"
  done; done
}
for att in $(seq 1 6); do
  mapfile -t T < <(build)
  echo "=== sweep $att: ${#T[@]} outstanding at $(date +%H:%M:%S) ==="
  [ ${#T[@]} -eq 0 ] && { echo BGE_HELDOUT_COMPLETE; exit 0; }
  printf '%s\n' "${T[@]}" | xargs -P "$PAR" -n 2 bash -c '
    m=$0; s=$1
    OMP_NUM_THREADS=1 "$PY" -u experiment_algorithm1.py \
      --slim amazon_v3_slim.npz --dataset_name amazonv3 --model_family v3 --llm_type $m --seed $s \
      --llm_confidence 0.1 --passive_full --alpha_decay 0.9985 --decay_per round \
      --device cuda --resume --max_samples 50000 > logs/run_amazonv3_${m}_c0.1_s${s}.log 2>&1'
  echo "sweep $att done at $(date +%H:%M:%S)"
done
[ $(build | wc -l) -eq 0 ] && echo BGE_HELDOUT_COMPLETE || { echo STILL_INCOMPLETE; build; }
