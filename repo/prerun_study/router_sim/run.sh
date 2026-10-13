#!/usr/bin/env bash
# Launch the Router sweep detached.  Usage:
#   ./run.sh                       # full grid into results/full
#   ./run.sh results/full --resume # resume a crashed / interrupted run
#   ./run.sh results/quick --quick # smoke test
#   PYTHON=/path/to/python ./run.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"
OUT="${1:-$HERE/results/full}"
shift || true
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac
if ! "$PY" -c "import numpy" 2>/dev/null; then
  echo "ERROR: '$PY' cannot import numpy (pip install numpy, or set PYTHON=...)" >&2; exit 1
fi
if [ -f "$OUT/pid" ] && kill -0 "$(cat "$OUT/pid")" 2>/dev/null; then
  echo "A sweep is already running in $OUT (pid $(cat "$OUT/pid"))." >&2; exit 1
fi
if [ -f "$OUT/results.csv" ] && [[ " $* " != *" --resume "* ]]; then
  echo "$OUT/results.csv exists. Re-run with --resume to continue it, or pick another out dir." >&2; exit 1
fi
mkdir -p "$OUT"
nohup "$PY" "$HERE/run_sweep.py" --out "$OUT" "$@" >> "$OUT/stdout.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT/pid"
echo "started pid $PID -> $OUT"
echo "status:    $PY $HERE/status.py --out $OUT            (add --watch 30 to refresh)"
echo "summarize: $PY $HERE/summarize.py --out $OUT"
