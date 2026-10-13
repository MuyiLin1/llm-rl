#!/bin/bash
# Current state of the paper's experiments. Run: ssh bruinml 'bash ~/DMwithLLM/status.sh'

# Portable paths: override these if your layout differs.
#   REPO   the repository root            (default: two levels above this script)
#   STUDY  where result files are written (default: $REPO/amazoncat13k/study)
#   PY     the python interpreter         (default: python3)
REPO=${REPO:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
STUDY=${STUDY:-"$REPO/amazoncat13k/study"}
PY=${PY:-python3}

A="$REPO/amazoncat13k"; L=$A/study/logs
echo "================= $(date '+%a %d %b %H:%M:%S') ================="
echo "GPU:  $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader)"
echo "Load: $(uptime | sed 's/.*load average/load/')"
echo "Python runs in flight: $(ps -eo cmd | grep -c '[a]naconda3.*python -u experiment_algorithm1')"
echo
echo "===== RUNNING NOW ====="
echo "-- Step 1: BGE c-sweep (make the 6 degenerate BGE rows informative)"
tot=0; don=0
for c in 0.01 0.1 1 10; do
  n=$(ls $L/run_*v3_*_c${c}_s*.log 2>/dev/null | wc -l)
  d=$(grep -l '^FINAL' $L/run_*v3_*_c${c}_s*.log 2>/dev/null | wc -l)
  printf "     c=%-5s %2d/18 done\n" $c $d; tot=$((tot+18)); don=$((don+d))
done
printf "     TOTAL   %d/%d\n" $don $tot
tail -1 $A/bge_driver.log 2>/dev/null | sed 's/^/     /'
echo
echo "-- Step 2: covariance-only test (queued, starts when Step 1 ends)"
if pgrep -f '[r]un_passive.sh' >/dev/null; then
  d=$(grep -l '^FINAL' $L/run_passiveonly_*.log 2>/dev/null | wc -l)
  echo "     queued/active: $d/30 done"
  tail -1 $A/passive_driver.log 2>/dev/null | sed 's/^/     /'
else
  echo "     driver NOT running"
fi
echo
echo "===== ALREADY FINISHED ====="
echo "  main study (12 configs x 10 seeds): $(grep -l '^FINAL' $L/run_amazonv1_*_s*.log $L/run_amazonv3_*_s*.log $L/run_wikiv1_*_s*.log $L/run_wikiv3_*_s*.log 2>/dev/null | grep -v '_c0\|_c1' | wc -l)/120"
echo "  verification gates: both passed (port reproduces published; float32==float64)"
echo "  full-horizon dry run: 70/70"
echo "  price measurement: done -- does NOT separate the experts (open question)"
echo
echo "===== IN-FLIGHT DETAIL ====="
for f in $(grep -L '^FINAL' $L/*.log 2>/dev/null | head -8); do
  printf "  %-34s %s\n" $(basename $f .log) "$(tail -1 $f | grep -o 'Step *[0-9]*/[0-9]*' || echo starting)"
done
