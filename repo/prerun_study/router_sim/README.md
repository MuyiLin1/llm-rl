# router_sim: the Router on the transfer-separation instance

Simulates the Confidence-Gated Router (v8.tex, appendix "Confidence-Gated Meta-Algorithm", m=1,
V^(1) = c I) on the instance of Theorem `thm:transfer-separation` ("Equal coverage, unequal value"),
next to cold LinUCB and the fixed-window hybrids H_tau. It tests whether the theorem's A-vs-B separation
survives when the window is chosen by the gate instead of being fixed:

- does Router-A settle on D at least as often as H_tau-A (and the bound 1-exp(-8(p-omega)h))?
- does Router-B turn into expert B (it routes whenever LinUCB proposes a fresh arm, whose width L/sqrt(lam) is maximal)?
- cold/Router-A vs cold/Router-B multipliers, scripted vs adaptive window, coverage-only (`_noreward`) variants;
- f > p-q: pseudo-regret vs realized reward (can the ranking flip?);
- sanity: c = lam/(2L^2) never routes, so Router == cold (adaptive) / == H_tau (scripted) seed by seed;
- large radius omega = 2 > p (valid-radius regime).

Simulator conventions mirror `prerun_study/transfer_thm/stronger.py` exactly (diagonal LinUCB as scalar
counts, fresh index omega = beta L/sqrt(lam), tie -> fresh arm with tolerance 1e-12).

## Files
- `sim.py` - vectorised simulator (numpy, state arrays [configs, seeds]) + scalar reference; `--check`, `--bench`.
- `run_sweep.py` - grid runner: one CSV row per (cell, policy, c), appended + fsynced per cell; `progress.json`, `run.log`.
- `run.sh` - detached launcher (nohup), writes `pid`; `PYTHON=... ./run.sh` to choose the interpreter.
- `status.py` - RUNNING / FINISHED / DIED-INCOMPLETE, progress, ETA, log tail, live table; `--watch N`.
- `summarize.py` - writes `summary.md` (the questions above) and `table.tex` (booktabs drafts for App. F).

## Commands (from this directory)
```
python3 sim.py --check                         # vectorised == scalar, every action and metric (~10 s)
python3 run_sweep.py --quick --out results/quick   # smoke test (~3 s)
./run.sh                                       # full grid, detached, into results/full
python3 status.py --out results/full --watch 30
./run.sh results/full --resume                 # after a crash / reboot: continues, skipping finished cells
python3 summarize.py --out results/full        # -> results/full/summary.md, results/full/table.tex
python3 summarize.py --out results/full --c 10 --T-table 2000   # other headline c / table horizon
```

## Grid and runtime
Settings (lam=0.1, L=2): (p,q,omega) = (0.3,0.05,0.135), (0.5,0.1,0.225), (0.2,0,0.09) with f=0; plus
f>p-q: p=0.3, q=0.1, f=0.25, omega=0.135, lam=1; each also with omega=2 (`*_bigR`). k in {1,2,4,8},
T in {2000 (500 seeds), 20000 (200 seeds)}, designs scripted/adaptive -> 128 cells. Every cell runs
cold, hybrid_{A,B}[_noreward] and router_{A,B}[_noreward] at c in {lam/(2L^2) (never), 0.05, 1, 10, 100, 1000}
on the same type sequences (29 policies). Measured on an M4 Max (single core): ~2 s per T=2000 cell,
~15-18 s per T=20000 cell, ~20 min total.

## Outputs (in the out dir)
`results.csv` (means and s.e. of regret, reward, routed rounds, last routed round, routed after tau,
settle rates (second half / after tau), fraction of second-half rounds on D, n_D, n_R, routed right answers,
routed D/R/fresh plays, acquisition regret on routed rounds; plus cold/policy ratio, paired difference
to the same-expert hybrid, bound, identical_to_cold / identical_to_hyb flags), `progress.json`, `run.log`,
`stdout.log`, `pid`, and after summarize `summary.md`, `table.tex`.
