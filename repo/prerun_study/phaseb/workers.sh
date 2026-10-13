#!/bin/bash
# Switch the number of workers (progress is kept). Usage: ./workers.sh 2   (office)   ./workers.sh 4   (home)
H=/Users/linmuyi/code/DMwithLLM/prerun_study/phaseb
bash $H/stop_local.sh >/dev/null; sleep 3; bash $H/start_local.sh "${1:-3}"
