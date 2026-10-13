#!/bin/bash
# Start (or resume) Phase B on the cluster. Safe to rerun: finished runs are skipped, unfinished batches resume.
cd ~/DMwithLLM/phaseb
if pgrep -f "phaseb.py" > /dev/null; then echo "already running (progress: python3 ~/DMwithLLM/phaseb/status.py)"; exit 0; fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
setsid nohup ~/anaconda3/envs/dmwithllm/bin/python -u phaseb.py >> logs/driver.log 2>&1 < /dev/null &
echo "started. progress: python3 ~/DMwithLLM/phaseb/status.py"
