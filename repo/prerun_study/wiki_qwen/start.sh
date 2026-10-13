#!/bin/bash
# Start (or resume) the chain: Flan-T5 c=1e4 runs, then the Qwen sweep and evaluation. Safe to rerun.
Q=/Users/linmuyi/code/DMwithLLM/prerun_study/wiki_qwen
if ps -Ao command | grep -v '^caffeinate' | grep -q "[w]iki_qwen/orchestrate.sh"; then
  echo "already running (check progress with: python3 $Q/status.py)"; exit 0; fi
nohup caffeinate -ims bash $Q/orchestrate.sh >> $Q/logs/orchestrator.log 2>&1 &
echo "started. check progress with: python3 $Q/status.py"
