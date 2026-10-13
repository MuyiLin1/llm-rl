#!/bin/bash

# Portable paths: override these if your layout differs.
#   REPO   the repository root            (default: two levels above this script)
#   STUDY  where result files are written (default: $REPO/amazoncat13k/study)
#   PY     the python interpreter         (default: python3)
REPO=${REPO:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
STUDY=${STUDY:-"$REPO/amazoncat13k/study"}
PY=${PY:-python3}

cd "$REPO/amazoncat13k"
mkdir -p dryseeds
for s in 1 2 3 4 5 6 7 8 9 10; do
  OMP_NUM_THREADS=2 "$PY" -u dry_run_full.py \
    --slim amazon_v1_slim.npz --T_dry 10000 --taus 1600 --lam 0.1 --beta 1.0 \
    --seed $s --device auto --out dryseeds/dry_s$s.json > dryseeds/log_s$s.txt 2>&1
done
echo ALL_DONE
