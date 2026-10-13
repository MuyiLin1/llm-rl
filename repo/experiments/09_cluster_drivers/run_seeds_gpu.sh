#!/bin/bash

# Portable paths: override these if your layout differs.
#   REPO   the repository root            (default: two levels above this script)
#   STUDY  where result files are written (default: $REPO/amazoncat13k/study)
#   PY     the python interpreter         (default: python3)
REPO=${REPO:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
STUDY=${STUDY:-"$REPO/amazoncat13k/study"}
PY=${PY:-python3}

cd "$REPO/amazoncat13k/seeds"
for s in 4 5 6 7 8 9 10; do
  for m in small base large; do
    OMP_NUM_THREADS=2 "$PY" -u experiment_algorithm1.py \
      --slim amazon_v1_slim.npz --llm_type $m --seed $s --llm_confidence 100 \
      --passive_full --alpha_decay 0.9985 --max_samples 50000 --resume --device auto \
      > logs/run_${m}_s${s}.log 2>&1
  done
done
echo SEEDS_DONE
