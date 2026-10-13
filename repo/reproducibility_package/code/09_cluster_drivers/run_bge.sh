#!/bin/bash
# BGE c-sweep. At c=100 the gate never flips and all six BGE rows are degenerate
# (Router == LLM exactly, V == 0, sd == 0 over ten seeds). Smaller c raises the
# expert's width u_LLM = ||phi||/sqrt(c), so the gate starts preferring LinUCB.
# 2 configs x 3 experts x 4 c-values x 3 seeds = 72 runs to locate the useful c.
set -u

# Portable paths: override these if your layout differs.
#   REPO   the repository root            (default: two levels above this script)
#   STUDY  where result files are written (default: $REPO/amazoncat13k/study)
#   PY     the python interpreter         (default: python3)
REPO=${REPO:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
STUDY=${STUDY:-"$REPO/amazoncat13k/study"}
PY=${PY:-python3}

PAR=${PAR:-12}
cd "$STUDY"
build () {
  for s in 1 2 3; do for m in small base large; do for c in 0.01 0.1 1 10; do
    for spec in "amazonv3 v3 amazon_v3_slim.npz 50000" "wikiv3 v3 wiki_v3_slim.npz 0"; do
      set -- $spec
      f=logs/run_$1_${m}_c${c}_s${s}.log
      grep -q '^FINAL' $f 2>/dev/null || echo "$1 $2 $3 $4 $m $s $c"
    done; done; done; done
}
for att in $(seq 1 6); do
  mapfile -t T < <(build)
  echo "=== sweep $att: ${#T[@]} outstanding at $(date +%H:%M:%S) ==="
  [ ${#T[@]} -eq 0 ] && { echo BGE_SWEEP_COMPLETE; exit 0; }
  printf '%s\n' "${T[@]}" | xargs -P "$PAR" -n 7 bash -c '
    name=$0; fam=$1; slim=$2; maxs=$3; m=$4; s=$5; c=$6
    extra=""; [ "$maxs" != "0" ] && extra="--max_samples $maxs"
    OMP_NUM_THREADS=1 "$PY" -u experiment_algorithm1.py \
      --slim $slim --dataset_name $name --model_family $fam --llm_type $m --seed $s \
      --llm_confidence $c --passive_full --alpha_decay 0.9985 --decay_per round \
      --device cuda --resume $extra > logs/run_${name}_${m}_c${c}_s${s}.log 2>&1'
  echo "sweep $att done at $(date +%H:%M:%S)"
done
r=$(build | wc -l); [ "$r" -eq 0 ] && echo BGE_SWEEP_COMPLETE || echo "STILL_INCOMPLETE: $r"
