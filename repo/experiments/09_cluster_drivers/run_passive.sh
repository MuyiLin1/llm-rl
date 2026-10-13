#!/bin/bash
# Step 2: the covariance-only test. Waits for the BGE sweep, then re-runs
# AmazonCat+BERT with the reward update switched OFF, so a routed round contributes
# only the covariance -- the warm start the paper's proofs actually assume.
# If the three experts still separate, coverage explains the result; if they
# collapse together, the reward vector was doing the work.
set -u

# Portable paths: override these if your layout differs.
#   REPO   the repository root            (default: two levels above this script)
#   STUDY  where result files are written (default: $REPO/amazoncat13k/study)
#   PY     the python interpreter         (default: python3)
REPO=${REPO:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
STUDY=${STUDY:-"$REPO/amazoncat13k/study"}
PY=${PY:-python3}

cd "$REPO/amazoncat13k"
while pgrep -f "bash run_bge.sh" > /dev/null; do sleep 120; done
echo "BGE sweep finished at $(date +%H:%M:%S); starting covariance-only runs"
cd study
build () {
  for s in 1 2 3 4 5 6 7 8 9 10; do for m in small base large; do
    f=logs/run_passiveonly_${m}_s${s}.log
    grep -q '^FINAL' $f 2>/dev/null || echo "$m $s"
  done; done
}
for att in $(seq 1 5); do
  mapfile -t T < <(build)
  echo "=== sweep $att: ${#T[@]} outstanding at $(date +%H:%M:%S) ==="
  [ ${#T[@]} -eq 0 ] && { echo PASSIVE_ONLY_COMPLETE; exit 0; }
  printf '%s\n' "${T[@]}" | xargs -P 12 -n 2 bash -c '
    m=$0; s=$1
    OMP_NUM_THREADS=1 "$PY" -u experiment_algorithm1.py \
      --slim amazon_v1_slim.npz --dataset_name passiveonly --model_family v1 \
      --llm_type $m --seed $s --llm_confidence 100 --alpha_decay 0.9985 \
      --decay_per round --device cuda --max_samples 50000 --resume \
      > logs/run_passiveonly_${m}_s${s}.log 2>&1'
done
r=$(build | wc -l); [ "$r" -eq 0 ] && echo PASSIVE_ONLY_COMPLETE || echo "STILL_INCOMPLETE: $r"
