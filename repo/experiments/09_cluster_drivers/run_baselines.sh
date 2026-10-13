#!/bin/bash
# Reviewer hk3d (weaknesses + Q5) and PAT both asked for these.
#   corral  : randomized log-barrier master instead of the deterministic gate,
#             same cached predictions, same observation feed.
#   offline : expert pulls taken OUTSIDE the online stream, so acquisition is not
#             charged (row 1 of the catalog). tau=1600 matches the gate's routed
#             count; tau=5000 is the value promised in the rebuttal.
# 3 experts x 10 seeds x (corral + offline@1600 + offline@5000) = 90 runs.
set -u

# Portable paths: override these if your layout differs.
#   REPO   the repository root            (default: two levels above this script)
#   STUDY  where result files are written (default: $REPO/amazoncat13k/study)
#   PY     the python interpreter         (default: python3)
REPO=${REPO:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
STUDY=${STUDY:-"$REPO/amazoncat13k/study"}
PY=${PY:-python3}

cd "$STUDY"
cp ../experiment_algorithm1.py . 2>/dev/null
build () {
  for s in 1 2 3 4 5 6 7 8 9 10; do for m in small base large; do
    grep -q '^FINAL' logs/run_corral_${m}_s${s}.log 2>/dev/null || echo "corral $m $s 0"
    for t in 1600 5000; do
      grep -q '^FINAL' logs/run_off${t}_${m}_s${s}.log 2>/dev/null || echo "off$t $m $s $t"
    done
  done; done
}
for att in $(seq 1 6); do
  mapfile -t T < <(build)
  echo "=== sweep $att: ${#T[@]} outstanding at $(date +%H:%M:%S) ==="
  [ ${#T[@]} -eq 0 ] && { echo BASELINES_COMPLETE; exit 0; }
  printf '%s\n' "${T[@]}" | xargs -P 10 -n 4 bash -c '
    tag=$0; m=$1; s=$2; tau=$3
    if [ "$tag" = corral ]; then R="--router corral"; else R="--router offline --offline_tau $tau"; fi
    OMP_NUM_THREADS=1 "$PY" -u experiment_algorithm1.py \
      --slim amazon_v1_slim.npz --dataset_name $tag --model_family v1 --llm_type $m --seed $s \
      --llm_confidence 100 --passive_full --alpha_decay 0.9985 --decay_per round --device cuda \
      --max_samples 50000 --resume $R > logs/run_${tag}_${m}_s${s}.log 2>&1'
  echo "sweep $att done at $(date +%H:%M:%S)"
done
[ $(build | wc -l) -eq 0 ] && echo BASELINES_COMPLETE || { echo "STILL_INCOMPLETE"; build; }
