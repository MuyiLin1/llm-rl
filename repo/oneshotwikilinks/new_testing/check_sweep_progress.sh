#!/bin/bash
# Quick progress checker for extended sweep
cd /Users/linmuyi/code/DMwithLLM/oneshotwikilinks/new_testing

TOTAL=486  # checkpoints every 10 batches, 4860 batches / 10 = 486 entries

echo "=== Extended Sweep Progress ==="
echo ""

for f in results_algo1_large_c{25000,50000,100000,500000}.0_pfull.json \
         results_algo1v3_base_c{100,200,500,1000,5000}.0_pfull.json \
         results_algo1v3_large_c{100,200,500,1000,5000}.0_pfull.json; do
    if [ -f "$f" ]; then
        python3 -c "
import json
d=json.load(open('$f'))
n=len(d['batch'])
pct=100*n/$TOTAL
status='DONE' if n>4000 else f'{n}/$TOTAL ({pct:.0f}%)'
print(f'  {\"$f\":<50s} {status}  hybrid={d[\"hybrid\"][-1]:.4f} llm_frac={d[\"llm_fraction\"][-1]:.4f}')
" 2>/dev/null
    else
        echo "  $f  NOT STARTED"
    fi
done

echo ""
echo "Running processes:"
ps aux | grep "experiment_algorithm1" | grep -v grep | awk '{print "  "$11" "$12" "$13" "$14" "$15" "$16}'
