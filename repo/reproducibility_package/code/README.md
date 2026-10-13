# Experiments: code, data and exact reproduction instructions

This folder is a **self-contained copy** of every script behind the numbers in
the paper, organised by what each one does. Nothing here is the "live" copy —
these are duplicates. The originals are still in `../amazoncat13k/`,
`../llm_rl_from_claude_LOOK_AT_THIS/validation_suite/` and (for WikiLinks) in
git history. Nothing was deleted to build this folder.

If you only read one section, read **[Seeds](#4-seeds-the-important-part)**.

---

## 1. What the experiments are trying to show

The paper asks a narrow question: *when does warm-starting a linear contextual
bandit (LinUCB) with a frozen expert (an LLM) actually help, and where does the
help come from?*

The answer has two halves, and the experiments are split the same way.

**Half one — it helps through coverage.** A warm start hands LinUCB two things:
a design matrix `M` (which directions of feature space have been observed) and
a response sum `c` (what rewards came back). The theory says the design matrix
is what shrinks LinUCB's confidence widths, and the size of the effect is a
*ratio of prices*, where a price is "regret paid per unit of coverage bought".

**Half two — but on real data, the rewards matter too.** Under a perfectly
linear reward model, the response sum carries no extra information (given `M`,
its distribution is fixed). Real rewards are not linear — Precision@1 is an
indicator, and no linear function reproduces an indicator. The leftover piece
is called the **transfer**, and on AmazonCat it turns out to be what separates
three LLMs that have nearly identical accuracy but deliver 1.71x, 2.72x and
3.29x improvements.

So: the experiments measure (a) the size of the improvement, (b) which of the
two channels carries it, and (c) whether the pre-runtime theory predicts (a)
before you spend any money on LLM calls.

### Two numbers for one effect

The paper reports every result on **two scales**, and they are easy to confuse:

```
accuracy multiplier = Router accuracy / max(LLM accuracy, cold accuracy)
regret ratio        = (1 - max(LLM acc, cold acc)) / (1 - Router accuracy)
```

The accuracy multiplier is what the literature calls "reward multiplication".
The regret ratio is what the theorems bound. Every context has a correct
action, so regret per round is `1 - accuracy`. When accuracies are low the two
scales differ a lot: Flan-T5-large goes from 8.39% to 27.64% accuracy, which is
a **3.29x accuracy multiplier** but a **1.27 regret ratio** (21% less regret).
Neither is wrong; they measure different things. `A`, `V` and `B` are
differences of regret in correct answers, so they are the same on both scales.

Both use the ratio of *seed-averaged* accuracies, not the average of per-seed
ratios; the two differ noticeably (1.71 vs 1.93 for Flan-T5-small) because seed
variance is large. `07_analysis/regret_scale_all_conditions.py` prints both
scales for every condition, with paired t-tests on the regret scale.

---

## 2. Thirty-second check that everything works

Before touching the real data, run the smoke test. It builds a tiny synthetic
dataset in the same `.npz` schema, runs the **actual** experiment runner on it,
and verifies the decomposition identity. No GPU, no torch, no downloads:

```bash
pip install -r requirements.txt
bash 00_smoke_test/run_smoke_test.sh
```

It should end with `SMOKE TEST PASSED`. The fixture is deliberately built so
the run reproduces the real signature ($A_\tau$ tiny, $V_\tau$ large, Router
above cold LinUCB), which means a pass exercises the mechanism and not just
the imports.

## 3. Quick start: reproduce the headline number

The headline is AmazonCat-13K + BERT-768d + Flan-T5-large = **3.29x**.

You need one file, `amazon_v1_slim.npz` (228 MB) — see [DATA.md](DATA.md) for
the download and its SHA-256. It is self-contained: context embeddings, action
embeddings, labels, and the three LLMs' cached predictions. You do **not** need
a GPU, an LLM, or the 6.9 GB source dataset to reproduce any online result.
`torch` is optional; without it the runner uses the NumPy/CPU path, which was
verified to give results identical to the GPU path.

```bash
# one seed, ~15 min on CPU
python 03_main_experiment/experiment_algorithm1.py \
    --slim amazon_v1_slim.npz --dataset_name amazonv1 \
    --model_family v1 --llm_type large --seed 1 \
    --llm_confidence 100 --passive_full \
    --alpha_decay 0.9985 --decay_per round \
    --max_samples 50000
```

The run prints a `FINAL` line with Router / LLM / cold-LinUCB accuracies and
then the `A / V / B` decomposition. To get the paper's table, run seeds 1
through 10 and average. `09_cluster_drivers/run_study.sh` does all 12
configurations x 10 seeds with retries.

---

## 4. Folder guide

| Folder | What it does | Run it when |
|---|---|---|
| `00_smoke_test/` | tiny synthetic fixture + end-to-end check | first, always |
| `01_data_preparation/` | Turns the raw datasets into embeddings, then condenses them into the small `.npz` the runner reads | Only if rebuilding from raw data |
| `02_llm_caching/` | Runs the LLMs once over the dataset and caches their predictions | Only if rebuilding caches |
| `03_main_experiment/` | **`experiment_algorithm1.py` — the single runner for every online result in the paper** | Always |
| `04_two_channel/` | Separates the coverage channel from the transfer channel | For the two-channel results |
| `05_pre_runtime_diagnostics/` | Predicts the multiplier *before* running anything | For the diagnostics table |
| `06_synthetic_validation/` | Simulates the theory's constructed instances and checks each proved claim numerically | To verify the theory |
| `07_analysis/` | Aggregates result JSONs into the paper's tables | After runs finish |
| `08_reference_implementations/` | Older standalone implementations, kept for reference | Never, for reproduction |
| `09_cluster_drivers/` | The shell scripts that actually launched every sweep, with the exact flags and seeds | To rerun a whole sweep |

### One runner, every configuration

`experiment_algorithm1.py` is deliberately the only online program. Both
datasets, both feature spaces, all three routers and all the ablations go
through it, selected by flags. This is why cold LinUCB is *bit-identical*
across arms — there is only one implementation of it. The flags that matter:

| Flag | Meaning |
|---|---|
| `--slim FILE` | the condensed dataset |
| `--dataset_name TAG` | output filename tag only; does not change behaviour |
| `--model_family {v1,v3}` | `v1` = BERT-768d + Flan-T5, `v3` = BGE-1024d + Qwen2.5 |
| `--llm_type {small,base,large}` | which of the three experts |
| `--seed S` | **the only randomness knob** (see below) |
| `--llm_confidence c` | the gate's confidence constant; expert width is `‖φ‖/√c` |
| `--router {gate,corral,offline}` | `gate` = Algorithm 1; the others are the baselines |
| `--offline_tau N` | with `--router offline`, how many free held-out expert pulls |
| `--passive_full` | update **both** `V` and `b` on routed rounds (what the paper analyses) |
| `--reward {binary,linear}` | `linear` replaces Precision@1 with a genuinely linear reward |
| `--alpha_decay`, `--decay_per` | exploration decay; the paper uses `0.9985` per **round** |
| `--device {auto,cpu,cuda}` | GPU float32 scoring was verified to give results identical to CPU float64 |
| `--resume` | restart from the last checkpoint |

---

## 5. Seeds: the important part

### There is exactly one seed per run

Every run is fully determined by `--seed S` together with its flags. The paper
uses **S = 1, 2, ..., 10** for every headline table, and **S = 1, 2, 3** for the
confidence sweeps (`tab:bge-csweep`). The BGE value `c = 0.1` was *chosen* on
seeds 1–3, so it is evaluated separately on the untouched seeds **4–13**
(`run_bge_heldout.sh`); tuning and evaluation never share a seed.

### What the seed actually controls

In `03_main_experiment/experiment_algorithm1.py`:

```python
np.random.seed(seed)                  # line 226
...
perm = np.random.permutation(N)       # line 257  <-- the context stream order
```

The seed's **dominant** effect is the order in which contexts arrive. With
13,330 actions and only 50,000 rounds, which arms get explored first matters
enormously — cold LinUCB alone ranges from 5.1% to 11.1% accuracy across the
ten seeds. That is why everything is reported as mean ± sd over ten seeds and
why every comparison is a **paired** t-test on shared seeds rather than an
unpaired one.

Three further generators are derived from the same seed with fixed offsets, so
that they never draw from the same stream:

| Generator | Offset | Controls | Line |
|---|---|---|---|
| global legacy RNG | `seed` | the context stream permutation | 226, 257 |
| `default_rng(seed + 1000)` | +1000 | which held-out contexts the offline arm pulls | 436 |
| `default_rng(seed + 2000)` | +2000 | CORRAL's randomised arm choice | 452 |
| `default_rng(seed + 3000)` | +3000 | reward noise under `--reward linear` | 348 |

### Why runs are comparable across routers

The stream permutation is drawn **before** any router-specific generator is
touched. So for a given seed, gate / CORRAL / offline all see the *same
contexts in the same order*, and the cold-LinUCB track inside each run is
identical. This is verified, not assumed: cold LinUCB's final accuracy agrees
to all printed digits across all four arms on every one of the ten seeds
(`07_analysis/aggregate_baselines_accuracy.py` prints this check first).

### What the seed does *not* control

The **LLM predictions are not random and not seeded.** They are generated once
with greedy decoding (`temperature=0`, `do_sample=False`) and cached to disk,
so every seed and every router sees exactly the same expert. An expert's
held-out accuracy (5.20%, 3.61%, 5.27% for Flan-T5 small/base/large) is
therefore a fixed property of the cache, with no error bar.

The pre-runtime diagnostics use their own fixed `seed=42` for subsampling
(`compute_diagnostics.py`), unrelated to the experiment seeds.

### Finding the exact seed of any run

The seed is recorded in three places, so it can always be recovered:

1. **In the filename**, as the `_s{seed}` suffix:
   `results_algo1_amazonv1_v1_large_c100.0_pfull_s7.json` is seed 7.
2. **Inside the JSON**, under `config.seed`, along with every other flag:
   ```json
   {"config": {"seed": 1, "router": "corral", "llm_type": "base",
               "alpha_decay": 0.9985, "lam": 0.1, "llm_confidence": 100.0,
               "passive_full": true, "offline_tau": null, ...}}
   ```
3. **In `runs_manifest.csv`** in this folder — all **432 runs** behind the
   paper, one row each, with the full config and the final accuracies.
   Regenerate it with `07_analysis/build_runs_manifest.py`.

### Replicating one specific run

Take any row of `runs_manifest.csv` and map the columns onto flags. For
`corral_v1_base_c100.0_pfull_corral_s3`:

```bash
python 03_main_experiment/experiment_algorithm1.py \
    --slim amazon_v1_slim.npz --dataset_name corral \
    --model_family v1 --llm_type base --seed 3 \
    --router corral --llm_confidence 100 --passive_full \
    --alpha_decay 0.9985 --decay_per round --max_samples 50000
```

It should reproduce that row's `router_acc` and `cold_acc` exactly on the same
hardware class. Small last-digit differences between CPU float64 and GPU
float32 are possible in principle; in practice the two agreed exactly on the
configurations we checked, including the `A / V / B` decomposition.

### Reproducing a whole sweep

`09_cluster_drivers/` holds the scripts that actually produced the paper's
runs, with the seeds hard-coded (`SEEDS="1 2 3 4 5 6 7 8 9 10"`). They are
idempotent: each sweeps up to 6–8 times, skipping any run whose log already
contains a `FINAL` line, so an interrupted sweep can simply be rerun. Adjust
`PY=` (the interpreter path) and the `cd` target before use.

| Script | Sweep | Runs |
|---|---|---|
| `run_study.sh` | all 12 main configurations | 120 |
| `run_baselines.sh` | CORRAL + offline warm start | 90 |
| `run_bge.sh` | confidence sweep on BGE features, seeds 1–3 (selects `c`) | 72 |
| `run_bge_heldout.sh` | BGE at the selected `c=0.1`, fresh seeds 4–13 | 30 |
| `run_channel2.sh`, `run_test4.sh`, `run_test2b.sh` | two-channel ablations | varies |
| `run_passive.sh` | covariance-only (`b` not updated) ablation | 30 |
| `run_dryfull.sh`, `run_dryseeds.sh` | pre-runtime dry run | varies |
| `run_qwen_pipeline.sh` | Qwen caching + runs | varies |
| `status.sh` | progress report; safe to run any time | — |

---

## 6. Which script produces which table

| Paper object | Produced by |
|---|---|
| `tab:amazon-results` (12-config table) | `03_main_experiment/experiment_algorithm1.py` via `run_study.sh` |
| `tab:amazon-decomp` (A / V / B) | same runs; the runner prints `A`, `V`, `B` at the end |
| `tab:two-channel` | `run_channel2.sh`, `run_test4.sh` + `04_two_channel/three_conditions.py`, `linear_regret.py` |
| transfer / MRR numbers | `04_two_channel/transfer_decomp.py` |
| price-ratio predictions (1.73, 1.59, 1.58) | `04_two_channel/measure_price.py` |
| `tab:bge-csweep` | `run_bge.sh` |
| `tab:wiki-results`, `tab:wiki-csweep` | same runner, `--slim wiki_v1_slim.npz --dataset_name wikiv1` |
| `tab:baselines` (CORRAL + offline) | `run_baselines.sh` + the three `07_analysis/aggregate_baselines_*.py` |
| `tab:synthetic` (catalog validation) | every script in `06_synthetic_validation/` |
| every figure in the paper | `07_analysis/make_figures.py --logdir study` |
| regret ratios in every results table | `07_analysis/regret_scale_all_conditions.py`, `regret_scale_spans.py` |
| BGE held-out evaluation (`c=0.1`, seeds 4–13) | `09_cluster_drivers/run_bge_heldout.sh` |
| `tab:diagnostics` (pre-runtime) | `05_pre_runtime_diagnostics/` (see caveats) |
| `fig:amazon-dynamics` | `07_analysis/plot_amazon_dynamics.py` |

The three baseline aggregators are worth naming separately because they encode
the checks, not just the numbers:

- `aggregate_baselines_accuracy.py` — per-seed accuracies, and **first** the
  cold-LinUCB identity check across arms.
- `aggregate_baselines_paired_tests.py` — paired t-tests (t\* = 2.26 at n = 10),
  query counts, and both multiplier conventions side by side.
- `aggregate_baselines_decomposition.py` — `A / V / B` per arm from the
  per-round logs.

### The synthetic checks

These need no data and no GPU. Each one simulates an instance the theory
constructs and compares against the proved claim:

```bash
cd 06_synthetic_validation
python arrowhead_check.py     # the matrix-inverse lemma, 200 random trials
python canonical_check.py     # the canonical instance and the one-pass bound
python template_sim.py        # the row-5 template: all four parts of thm:price
python sweep_and_row7.py      # the 1/eps scaling, plus row 7
python check_row5p.py         # row 5': saving collected only on the tail
python extra_rows.py          # rows 4 and 8
```

They print pass/fail against explicit bounds. The most informative is the
`sweep_and_row7.py` epsilon sweep: the theory predicts the multiplier exceeds
one by order `1/eps`, and over a 64-fold range of `eps` the product
`(ratio - 1) * eps` stays between 0.0044 and 0.0039. That tests the mechanism,
not just the sign.

---

## 7. Data artifacts

The online experiments read only the condensed `.npz` files. Everything else is
needed only to rebuild those.

| File | Size | Contents |
|---|---|---|
| `amazon_v1_slim.npz` | 228 MB | 60,000 x 768 contexts, 13,330 x 768 actions, labels, 3 LLM caches |
| `amazon_v3_slim.npz` | 303 MB | same, 1024-d (BGE) |
| `wiki_v1_slim.npz` | 1.9 GB | 622,000 x 768 contexts, 311 x 768 actions |
| `wiki_v3_slim.npz` | 2.6 GB | same, 1024-d |
| `mydataset.0.pickle.gz` | 6.9 GB | AmazonCat source, BERT embeddings |
| `mydataset.0.v3.pickle.gz` | 9.7 GB | AmazonCat source, BGE embeddings |
| `llm_cache_amazon_v1_*.pkl.gz` | ~325 MB each | full-dataset Flan-T5 predictions |
| `theta_or_v1.npy` | 6 KB | the ridge oracle, fitted on held-out contexts |

**On the 60,000 contexts.** AmazonCat's slim file holds 60,000 contexts but the
online horizon is `T = 50,000`. The extra 10,000 are a held-out pool used for
two things that must never touch the online stream: the offline arm's free
expert pulls, and fitting the ridge oracle `theta_or`. WikiLinks uses its full
622,000 (`--max_samples 0`).

Each online run writes three files, all tagged with the seed:

- `results_algo1_<tag>_..._s<seed>.json` — accuracy curves at 1,000-round
  checkpoints, plus the full config.
- `perround_algo1_..._s<seed>.npz` — per-round rewards and actions for the
  router, cold LinUCB and the LLM. **This is what `A / V / B` and all the
  post-hoc analysis are computed from.**
- `checkpoint_algo1_..._s<seed>.npz` — `V⁻¹` and `b` for `--resume`.

---

## 8. Caveats and known gaps

Stated plainly so nobody loses time rediscovering them.

1. **The WikiLinks LLM caching script was never committed to git.** It lives
   untracked in `../oneshotwikilinks/new_testing/cache_llm_predictions.py` and
   is copied here as `02_llm_caching/cache_llm_predictions_wiki.py`. It uses the
   same greedy decoding (`temperature=0`, `do_sample=False`) as the AmazonCat
   version. Commit it if the repository is to be the source of truth.

2. **`05_pre_runtime_diagnostics/compute_diagnostics.py` targets an older
   version of the paper.** Its docstring refers to "the 3 theoretical axes",
   a framework the current paper replaced with the price-ratio theorem and the
   catalog. The geometric quantities it computes (effective rank, subspace
   angle, coherence) are still the ones the diagnostics table reports, but the
   pass/fail thresholds it prints are from the old framework and should be
   ignored.

3. **`07_analysis/plot_amazon_dynamics.py` is superseded.** It drew a
   single-seed version of the dynamics figure. `make_figures.py` now draws the
   seed-averaged figures the paper uses; the old plotter is kept for reference
   with its filenames corrected. `aggregate_results.py` had an `OBS` table of
   stale 3-seed multipliers, now updated to the ten-seed values.

4. **`pre_runtime_estimates.py` cannot use the full action set.** It must hold
   every candidate's features in memory, so `build_npz_for_estimates.py` cuts
   the action set to 128 candidates per context. In that reduced world cold
   LinUCB reaches ~65% instead of ~10%, so it is not starved, and the estimate
   reports "no gain" for experts that multiply by 2-5x in reality.
   `dry_run_task.py` exists to work over the full action set instead. The
   paper's appendix discusses this honestly, including that the dry run
   underestimates by roughly 30x.

5. **`08_reference_implementations/` is not used by any current result.** It
   holds the pre-unification standalone implementations, including the
   WikiLinks scripts recovered from git (they had been deleted from the working
   tree). `corral_clip.py` is worth keeping because the CORRAL baseline inside
   `experiment_algorithm1.py` was written to mirror it.

6. **Absolute accuracies are low by design.** Roughly 8% for cold LinUCB on
   AmazonCat is not a bug — it is Precision@1 among 13,330 actions after 50,000
   rounds. The paper's claims are all about *ratios* between policies on the
   same stream, which is why the paired design matters more than the absolute
   level.
