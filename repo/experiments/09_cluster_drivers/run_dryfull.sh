#!/bin/bash
# Full-horizon dry run: every (seed, task) pair, run to completion.
#
# 10 seeds x (1 cold + 3 experts x 2 taus) = 70 independent tasks.
# Tasks run PAR at a time; the whole set is swept repeatedly until every task has
# written a JSON with "complete": true, so a task killed by an OOM, a preemption
# or a dropped connection is simply retried rather than silently left partial.
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
SEEDS="1 2 3 4 5 6 7 8 9 10"
TAUS="1600 5000"
EXPERTS="small base large"
T_DRY=50000
PAR=${PAR:-6}
OUT=dryfull
mkdir -p $OUT logs

build_tasks () {                     # emit only tasks that are not yet complete
  for s in $SEEDS; do
    f=$OUT/cold_s$s.json
    grep -q '"complete": true' $f 2>/dev/null || echo "cold $s - -"
    for m in $EXPERTS; do
      for t in $TAUS; do
        f=$OUT/warm_${m}_tau${t}_s${s}.json
        grep -q '"complete": true' $f 2>/dev/null || echo "warm $s $m $t"
      done
    done
  done
}

for attempt in $(seq 1 6); do
  mapfile -t TASKS < <(build_tasks)
  n=${#TASKS[@]}
  echo "=== sweep $attempt: $n task(s) outstanding at $(date +%H:%M:%S) ==="
  [ "$n" -eq 0 ] && { echo "ALL_TASKS_COMPLETE"; exit 0; }
  printf '%s\n' "${TASKS[@]}" | xargs -P "$PAR" -n 4 bash -c '
    task=$0; seed=$1; llm=$2; tau=$3
    if [ "$task" = cold ]; then
      tag=cold_s$seed
      OMP_NUM_THREADS=1 "$PY" -u dry_run_task.py \
        --task cold --seed $seed --T_dry '"$T_DRY"' --device auto > logs/dry_$tag.log 2>&1
    else
      tag=warm_${llm}_tau${tau}_s$seed
      OMP_NUM_THREADS=1 "$PY" -u dry_run_task.py \
        --task warm --seed $seed --llm_type $llm --tau $tau --T_dry '"$T_DRY"' \
        --device auto > logs/dry_$tag.log 2>&1
    fi'
  echo "sweep $attempt finished at $(date +%H:%M:%S)"
done

remaining=$(build_tasks | wc -l)
if [ "$remaining" -eq 0 ]; then echo "ALL_TASKS_COMPLETE"; else
  echo "STILL_INCOMPLETE: $remaining task(s) after 6 sweeps"; build_tasks; fi
