#!/bin/bash
# Self-contained check that the pipeline works, with no real data and no GPU.
# Builds a tiny synthetic dataset in the real .npz schema, runs the actual
# experiment runner on it, and verifies the decomposition identity.
# Expected wall-clock: well under a minute.
set -eu
cd "$(dirname "$0")"
PY=${PY:-python3}

echo "== 1/4  building fixture =="
$PY make_fixture.py --actions 800 --n 2000 --n_keep 1500 --dim 128 --out fixture_slim.npz

echo
echo "== 2/4  running the experiment runner (CPU, no torch required) =="
$PY ../03_main_experiment/experiment_algorithm1.py --slim fixture_slim.npz --dataset_name smoke \
    --model_family v1 --llm_type large --seed 1 --llm_confidence 0.1 \
    --passive_full --alpha_decay 0.9985 --decay_per round \
    --max_samples 1500 --device cpu | tee smoke_run.log | tail -5

echo
echo "== 3/4  checking outputs and the A+V identity =="
$PY - <<'PYEOF'
import json, sys, numpy as np
tag = "smoke_v1_large_c0.1_pfull_s1"
h = json.load(open(f"results_algo1_{tag}.json"))
z = np.load(f"perround_algo1_{tag}.npz")
n = int(z['step']); ro = z['routed'][:n].astype(bool)
A = float((z['r_router'][:n][ro]  - z['r_cold'][:n][ro]).sum())
V = float((z['r_router'][:n][~ro] - z['r_cold'][:n][~ro]).sum())
tot = float(z['r_router'][:n].sum() - z['r_cold'][:n].sum())
ok = True
def check(cond, msg):
    global ok
    print(("  PASS  " if cond else "  FAIL  ") + msg); ok &= bool(cond)
check(h['step'][-1] == 1500, "run reached the full horizon")
check(abs(A + V - tot) < 1e-9, f"A+V identity holds (A={A:.0f}, V={V:.0f}, total={tot:.0f})")
check(V > 0, "V_tau is positive: the routed data helped after handoff")
check(abs(A) < abs(V), "A_tau is small beside V_tau, as on the real data")
mult = h['hybrid'][-1] / h['linucb_cold'][-1]
check(mult > 1.0, f"Router beats cold LinUCB ({mult:.2f}x accuracy multiplier)")
sys.exit(0 if ok else 1)
PYEOF

echo
echo "== 4/4  theory self-check (no data needed) =="
$PY ../06_synthetic_validation/arrowhead_check.py

echo
echo "SMOKE TEST PASSED"
