#!/bin/bash
# Stop Phase B on this Mac (driver and its workers). Progress is kept: restart with start_local.sh.
H=/Users/linmuyi/code/DMwithLLM/prerun_study/phaseb
P=$(cat $H/logs_local/driver.pid 2>/dev/null)
if [ -z "$P" ] || ! kill -0 "$P" 2>/dev/null; then echo "not running"; exit 0; fi
pkill -TERM -P "$P"; kill -TERM "$P"; sleep 3
pkill -KILL -P "$P" 2>/dev/null; kill -KILL "$P" 2>/dev/null
pkill -f "caffeinate -is nice -n 10 python3 -u phaseb_local.py" 2>/dev/null
echo "stopped. resume with: bash $H/start_local.sh"
