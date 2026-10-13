#!/bin/bash
# Charger and battery check (no admin rights needed).
#   power_check.sh       one reading (jumps around as the load changes, updates about once a minute)
#   power_check.sh 10    watch for 10 minutes and report the average; use this to judge the trend
ioreg_read() {
  ioreg -rw0 -c AppleSmartBattery | python3 -c '
import re, sys
s = sys.stdin.read()
g = lambda k: (re.search(r"\"%s\" = ([-0-9A-Za-z]+)" % k, s) or [None, "0"])[1]
ad = re.search(r"\"AdapterDetails\" = \{([^}]*)\}", s); ad = ad.group(1) if ad else ""
name = (re.search(r"\"Name\"=\"([^\"]*)\"", ad) or [None, "none"])[1]
watts = (re.search(r"\"Watts\"=(\d+)", ad) or [None, "?"])[1]
amp = int(g("InstantAmperage")); amp = amp - 2**64 if amp > 2**63 else amp
print(name, "|", watts, "|", g("CurrentCapacity"), "|", g("AppleRawCurrentCapacity"), "|", round(amp/1000*int(g("Voltage"))/1000, 1))
'
}
state() { python3 -c "f=$1; print('charging' if f>0.5 else ('DRAINING: the Mac uses more than the charger gives' if f<-0.5 else 'holding steady'))"; }

IFS='|' read -r NAME W PCT MAH FLOW <<< "$(ioreg_read)"
echo "Charger:          ${NAME% }"
echo "Charger delivers: ${W// /} W (what the charger and cable agreed on)"
echo "Battery:          ${PCT// /}%"
if [ -z "$1" ]; then
  echo "Battery flow now: ${FLOW// /} W  ($(state ${FLOW// /}))  -- one reading; run with a number of minutes for the trend"
  exit 0
fi
END=$(( $(date +%s) + $1*60 )); SUM=0; N=0; MAH0=${MAH// /}
echo "Watching for $1 minutes..."
while [ "$(date +%s)" -lt "$END" ]; do
  IFS='|' read -r _ _ _ MAH FLOW <<< "$(ioreg_read)"
  SUM=$(python3 -c "print($SUM + ${FLOW// /})"); N=$((N+1)); sleep 30
done
AVG=$(python3 -c "print(round($SUM/$N, 1))"); D=$(( ${MAH// /} - MAH0 ))
echo "Average battery flow over $1 min: $AVG W  ($(state $AVG)); battery charge changed by $D mAh"
