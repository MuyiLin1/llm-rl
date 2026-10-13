#!/bin/bash
# Start (or resume) Phase B on this Mac. Safe to rerun: finished runs are skipped, unfinished runs resume
# from their last checkpoint (every 2,000 rounds). Usage: bash start_local.sh [workers, default 3]
H=/Users/linmuyi/code/DMwithLLM/prerun_study/phaseb
cd $H
if [ -f logs_local/driver.pid ] && kill -0 "$(cat logs_local/driver.pid)" 2>/dev/null; then
  echo "already running (pid $(cat logs_local/driver.pid)). progress: bash $H/status_local.sh"; exit 0; fi
mkdir -p logs_local out_local
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PB_WORKERS=${1:-3}
nohup caffeinate -is nice -n 10 python3 -u phaseb_local.py >> logs_local/driver.log 2>&1 < /dev/null &
sleep 3
echo "started with $PB_WORKERS workers (pid $(cat logs_local/driver.pid)). progress: bash $H/status_local.sh"
echo "stop any time with: bash $H/stop_local.sh   (resume with: bash $H/start_local.sh)"
