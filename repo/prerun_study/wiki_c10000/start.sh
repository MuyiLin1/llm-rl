#!/bin/bash
# Starts (or resumes) the WikiLinks c=10000 runs. Safe to run again: finished runs are skipped and
# unfinished ones continue from their last checkpoint.
cd "$(dirname "$0")"
if pgrep -f "wiki_c10000/run.py" >/dev/null; then echo "already running"; exit 0; fi
for i in 0 1 2; do
  nohup caffeinate -ims python3 -u "$PWD/run.py" 3 $i >> "logs/worker$i.log" 2>&1 &
done
echo "started 3 workers; check progress with:  python3 $PWD/status.py"
