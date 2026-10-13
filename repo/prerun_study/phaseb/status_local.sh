#!/bin/bash
# Phase B progress on this Mac
H=/Users/linmuyi/code/DMwithLLM/prerun_study/phaseb
PB_LOG=$H/logs_local PB_PROC="python3 -u phaseb_local.py" python3 $H/status.py | sed "s#bash ~/DMwithLLM/phaseb/start.sh#bash $H/start_local.sh#"
