#!/bin/bash
# Re-run EVERY configuration with the unified, verified runner. AmazonCat/BERT is
# included even though we already have 10 seeds of it, because those runs predate
# per-round action logging and the price measurement needs the played actions.
# 12 configurations x 10 seeds = 120 runs, all GPU (float32 scoring, float64 state),
# alpha decay 0.9985 per round, per-round logging so A/V/B is computable.
# Sweeps repeatedly until every run has written its FINAL line.
set -u

# Portable paths: override these if your layout differs.
#   REPO   the repository root            (default: two levels above this script)
#   STUDY  where result files are written (default: $REPO/amazoncat13k/study)
#   PY     the python interpreter         (default: python3)
REPO=${REPO:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
STUDY=${STUDY:-"$REPO/amazoncat13k/study"}
PY=${PY:-python3}

PY="$PY"
PAR=${PAR:-12}
SEEDS="1 2 3 4 5 6 7 8 9 10"
cd "$REPO/amazoncat13k"
mkdir -p study/logs && cd study
ln -sf ../amazon_v1_slim.npz ../amazon_v3_slim.npz ../experiment_algorithm1.py . 2>/dev/null
ln -sf "$REPO/oneshotwikilinks/new_testing/wiki_v1_slim.npz" . 2>/dev/null
ln -sf "$REPO/oneshotwikilinks/new_testing/wiki_v3_slim.npz" . 2>/dev/null

build_tasks () {
  for s in $SEEDS; do
    for m in small base large; do
      # dataset  family  slim-file           max_samples
      for spec in "amazonv1 v1 amazon_v1_slim.npz 50000" \
                  "amazonv3 v3 amazon_v3_slim.npz 50000" \
                  "wikiv1   v1 wiki_v1_slim.npz   0" \
                  "wikiv3   v3 wiki_v3_slim.npz   0"; do
        set -- $spec
        [ -f "$3" ] || continue                      # slim file not here yet
        f=logs/run_$1_${m}_s${s}.log
        grep -q '^FINAL' $f 2>/dev/null || echo "$1 $2 $3 $4 $m $s"
      done
    done
  done
}

for attempt in $(seq 1 8); do
  mapfile -t T < <(build_tasks)
  echo "=== sweep $attempt: ${#T[@]} run(s) outstanding at $(date +%H:%M:%S) ==="
  [ ${#T[@]} -eq 0 ] && { echo "ALL_RUNS_COMPLETE"; exit 0; }
  printf '%s\n' "${T[@]}" | xargs -P "$PAR" -n 6 bash -c '
    name=$0; fam=$1; slim=$2; maxs=$3; m=$4; s=$5
    extra=""; [ "$maxs" != "0" ] && extra="--max_samples $maxs"
    OMP_NUM_THREADS=1 "$PY" -u experiment_algorithm1.py \
      --slim $slim --dataset_name $name --model_family $fam --llm_type $m --seed $s \
      --llm_confidence 100 --passive_full --alpha_decay 0.9985 --decay_per round \
      --device cuda --resume $extra > logs/run_${name}_${m}_s${s}.log 2>&1'
  echo "sweep $attempt done at $(date +%H:%M:%S)"
done
r=$(build_tasks | wc -l)
[ "$r" -eq 0 ] && echo ALL_RUNS_COMPLETE || { echo "STILL_INCOMPLETE: $r"; build_tasks; }
