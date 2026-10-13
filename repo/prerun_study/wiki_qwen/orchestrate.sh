#!/bin/bash
# Waits for the Flan-T5 c=1e4 runs to finish (restarting them if they stopped early), then starts
# 3 Qwen workers. Safe to rerun: finished jobs are skipped and unfinished ones resume from checkpoints.
Q=/Users/linmuyi/code/DMwithLLM/prerun_study/wiki_qwen
F=/Users/linmuyi/code/DMwithLLM/prerun_study/wiki_c10000
nq() { ps -Ao command | grep -v '^caffeinate' | grep -c "[w]iki_qwen/run.py"; }
nf() { ps -Ao command | grep -v '^caffeinate' | grep -c "[w]iki_c10000/run.py"; }
echo "[$(date '+%m-%d %H:%M')] orchestrator started"
while [ "$(ls $F/out/*_s*.npz 2>/dev/null | grep -vc ckpt)" -lt 30 ]; do
  if [ "$(nf)" -eq 0 ]; then echo "[$(date '+%m-%d %H:%M')] Flan workers not running but unfinished: restarting them"; bash $F/start.sh; fi
  sleep 300
done
echo "[$(date '+%m-%d %H:%M')] Flan runs complete (30/30); starting Qwen workers"
cd $Q
for i in 0 1 2; do
  if [ "$(nq)" -lt 3 ]; then
    nohup caffeinate -ims python3 -u $Q/run.py w$i >> $Q/logs/worker$i.log 2>&1 &
    sleep 5
  fi
done
wait
echo "[$(date '+%m-%d %H:%M')] Qwen workers exited"
