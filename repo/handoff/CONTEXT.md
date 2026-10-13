# Project Handoff — LLM + Bandit Warm-Start Multiplication

> **Snapshot date:** 2026-09-06
> **Paper snapshot:** [`context.tex`](context.tex) (NeurIPS 2026 submission draft, working title around
> "reward multiplication as a regime indicator in LLM–bandit hybrids").
> **What this folder is:** a self-contained copy of every `.py` file we have been running,
> preserving the original directory layout under [`code/`](code/), plus this document explaining
> what each file does and the reasoning behind it.

---

## 1. The One-Paragraph Version

We study a **meta-algorithm ("Router")** that, on each round, deterministically chooses between a
**frozen LLM** and an **adaptive LinUCB** learner based on which one is more *geometrically confident*.
The central research question: **when does combining the two beat the better of either one alone**
(a "multiplication" effect, Router / max(LLM, Cold LinUCB) > 1)? Our thesis is that multiplication is
**not a generic property** of the meta-algorithm — it is a **regime indicator** governed by pre-computable
dataset diagnostics. We validate this on two real datasets (OneShotWikiLinks, AmazonCat-13K) and a family
of synthetic datasets. The headline positive result is **AmazonCat BERT-768d + Flan-T5-large → 2.34×**;
the headline negative result is **all WikiLinks configs → ≤1.05× regardless of hyperparameters**.

---

## 2. Core Concepts You Need Before Reading Code

### 2.1 Algorithm 1 — the "Router" (confidence-gated routing)

Every experiment script implements the same switchboard:

```
LLM uncertainty:     u_LLM    = ||phi(x, a_LLM)|| / sqrt(c)      # c = confidence parameter
LinUCB uncertainty:  u_LinUCB = sqrt(phi^T V^{-1} phi)

if u_LLM < u_LinUCB:  play the LLM's action
else:                 play LinUCB's action

# "Passive warm-start": update LinUCB sufficient statistics (V, b) on EVERY round
# from whichever action was actually played, regardless of who was routed.
V <- V + phi phi^T          # covariance always updated
b <- b + r * phi            # reward vector updated (passive_full mode)
```

Three policies are tracked on the same data stream for comparison:
1. **Router** — Algorithm 1 above (the warm-started hybrid).
2. **LLM only** — the frozen model on every round (static baseline).
3. **Cold LinUCB** — an independent LinUCB from `lambda*I`, gets no help from the LLM.

**Multiplication** = `Router / max(LLM, Cold)`. That is the number we chase.

Key hyperparameters: `c` (LLM confidence; swept over orders of magnitude), `lambda` (ridge init),
`alpha_t` (LinUCB exploration, decayed), `passive_full` (whether `b` is updated on LLM rounds too).

Features are **element-wise joint features**: `phi(x, a) = x ⊙ z_a` (context embedding times action
embedding), so the LinUCB dimension `d` equals the embedding dimension (768 for BERT, 1024 for BGE).

### 2.2 The three diagnostic axes (why a regime is what it is)

Computed **before** running any algorithm (see `compute_diagnostics.py` /
`pre_runtime_estimates.py`):

- **Axis A — Effective rank ratio `r̂/√T`.** Effective rank of the feature covariance over the horizon.
  Small ⇒ LinUCB learns fast on its own ⇒ "LinUCB-suffices."
- **Axis B — Feature coherence `α̂`.** How much information one arm pull shares with other arms.
  Low `α̂` (< ~0.1) ⇒ nearly orthogonal arms ⇒ each arm must be learned independently ⇒ cold LinUCB starves.
- **Axis C — Action-gap distribution.** Heavy tail of small gaps ⇒ hard to distinguish top actions.
- **(4th) Subspace mismatch angle.** Principal angle between LLM-induced design subspace and the
  reward-relevant context subspace. Large angle ⇒ LLM's exploration is wasted on irrelevant directions.

**Known open tension (documented, not yet resolved in the paper):** Axis A `r̂/√T` is tiny for *every*
dataset we tested (including AmazonCat, which *does* multiply), so Axis A alone mis-predicts. The current
working explanation: `r̂` measures the *context-only* covariance rank, but with K≈13k orthogonal arms the
true learning dimension is ~`K·r̂`. Coherence (Axis B) is the axis that actually separates the regimes.
Candidate fixes discussed: (i) redefine `r̂` on the *joint* feature matrix; (ii) make LinUCB-suffices
require A **and** B to saturate; (iii) reorder the decision tree to check B first.

---

## 3. Directory Map

| Directory | Role |
|-----------|------|
| [`code/oneshotwikilinks/new_testing/`](code/oneshotwikilinks/new_testing/) | **Real dataset #1** (311 entities, T=622k). Negative-result testbed: predicted & confirmed "LinUCB-suffices." Flan-T5 + Qwen families. |
| [`code/amazoncat13k/`](code/amazoncat13k/) | **Real dataset #2** (13,330 categories, T=50k). The stress test: BERT-768d → 2.34× multiplication; BGE-1024d → no multiplication. |
| [`code/synthetic_text/`](code/synthetic_text/) | Iterative attempts (v2–v8) to build a **controlled text dataset** that reproduces multiplication with *real* LLMs. |
| [`code/synthetic_clean/`](code/synthetic_clean/) | Fully **artificial** synthetic (no real LLM; simulated `θ*+noise` expert). Fast sanity check that the mechanism works. |
| [`code/synthetic/`](code/synthetic/) | Earlier purely-numeric synthetic (standard-basis arms) + diagnostics. |
| [`code/llm_rl_from_claude_LOOK_AT_THIS/`](code/llm_rl_from_claude_LOOK_AT_THIS/) | Theory-validation suite: numerically checks the paper's lemmas/propositions (arrowhead inverse, three-way separation, net-value rule) against brute-force simulation. |

The naming convention `v1/v2/v3…` almost always means **different embedding spaces or model families**,
not bug-fix versions:
- **v1** = BERT / `bert-base-nli-mean-tokens` (768d) + **Flan-T5** (mask-filling).
- **v2** = `all-mpnet-base-v2` (768d).
- **v3** = `BAAI/bge-large-en-v1.5` (1024d) + **Qwen2.5-Instruct** (chat).

---

## 4. File-by-File Context

### 4.1 `code/oneshotwikilinks/new_testing/` — WikiLinks (real dataset #1)

**Data preparation**
- [`make_data.py`](code/oneshotwikilinks/new_testing/make_data.py) — Builds the original WikiLinks
  pickle: context + entity embeddings (BERT 768d) from the Kaggle oneshotwikilinks dump.
- [`make_data_v2.py`](code/oneshotwikilinks/new_testing/make_data_v2.py) — Re-embeds everything with
  `all-mpnet-base-v2` (768d).
- [`make_data_v3.py`](code/oneshotwikilinks/new_testing/make_data_v3.py) — Re-embeds with
  `bge-large-en-v1.5` (1024d); this is the space Qwen outputs are matched against.
- [`cache_llm_predictions.py`](code/oneshotwikilinks/new_testing/cache_llm_predictions.py) —
  Runs the LLM **once** over the whole dataset and caches its argmax prediction per sample. This is the
  trick that makes the bandit sweep pure-numpy: after caching, no GPU is needed and `c`/`alpha`/`lambda`
  sweeps take seconds.

**Experiment runners (Algorithm 1)** — same routing logic, different embedding/model stack:
- [`experiment_algorithm1.py`](code/oneshotwikilinks/new_testing/experiment_algorithm1.py) — v1:
  BERT-768d + Flan-T5 (mask-filling). The canonical reference implementation of the Router.
- [`experiment_algorithm1_v2.py`](code/oneshotwikilinks/new_testing/experiment_algorithm1_v2.py) — v2:
  `all-mpnet-base-v2`.
- [`experiment_algorithm1_v3.py`](code/oneshotwikilinks/new_testing/experiment_algorithm1_v3.py) — v3:
  BGE-1024d + Qwen2.5-Instruct (0.5B/3B/7B).
- [`experiment_algorithm1_v4.py`](code/oneshotwikilinks/new_testing/experiment_algorithm1_v4.py) —
  **SVD-reduced features** experiment: project actions+context into a rank-`d` SVD basis (default 50) to
  test whether a lower-dimensional feature space changes the regime. Batch size 32 to match Chen et al.

**Diagnostics & analysis**
- [`compute_diagnostics.py`](code/oneshotwikilinks/new_testing/compute_diagnostics.py) — Computes the
  three axes (coherence, effective-rank ratio, gap distribution) from the dataset.
- [`oracle_diagnostic.py`](code/oneshotwikilinks/new_testing/oracle_diagnostic.py) — Measures the
  **oracle ceiling**: fraction of samples where LLM *or* Cold LinUCB is correct — the theoretical max the
  Router could reach. Uses caches, so no GPU.
- [`diagnose_router.py`](code/oneshotwikilinks/new_testing/diagnose_router.py) — Explains *why* the
  WikiLinks Router lift is negligible: when the advantage peaks, how quickly the LLM is phased out,
  windowed accuracy.
- [`diag_v3.py`](code/oneshotwikilinks/new_testing/diag_v3.py) — Quick numerical-stability check on the
  v3 (BGE) pickle (NaN/Inf/range of `Xs`).
- [`check_progress.py`](code/oneshotwikilinks/new_testing/check_progress.py) — Convenience script to
  poll long-running v4 result JSONs and print % complete.

**Plots**
- [`generate_plots.py`](code/oneshotwikilinks/new_testing/generate_plots.py) — All paper figures
  (learning curves, routing fraction, eigenvalue growth, feature-norm tracking) per model.
- [`generate_plots_multiseed.py`](code/oneshotwikilinks/new_testing/generate_plots_multiseed.py) —
  Mean ± std versions across seeds 1–3 with shaded bands (the actual paper figures).
- [`v2/analyze_routing.py`](code/oneshotwikilinks/new_testing/v2/analyze_routing.py) — Per-batch
  LLM-vs-LinUCB routing breakdown from result JSONs.
- [`v2/plot_results.py`](code/oneshotwikilinks/new_testing/v2/plot_results.py) — Plot helper for the v2
  result set.

### 4.2 `code/amazoncat13k/` — AmazonCat-13K (real dataset #2, the positive result)

**Data preparation**
- [`make_data.py`](code/amazoncat13k/make_data.py) — Parse AmazonCat-13K (titles/contents/labels) and
  embed with BERT-768d.
- [`make_data_fast.py`](code/amazoncat13k/make_data_fast.py) — Batch-encode all texts at once to
  saturate CPU; much faster rebuild of the same data.
- [`make_data_v3.py`](code/amazoncat13k/make_data_v3.py) — Re-embed with BGE-1024d for the Qwen
  comparison arm (the same 13,330 categories in a different feature space).
- [`save_dataset.py`](code/amazoncat13k/save_dataset.py) — Assemble cached `.npy` shards into one
  pickle for the experiment loader.

**Experiment runner (Algorithm 1)**
- [`experiment_algorithm1.py`](code/amazoncat13k/experiment_algorithm1.py) — The AmazonCat Router. Same
  logic as WikiLinks but: multi-label reward (`r=1` if chosen action ∈ true label set), 13,330 actions,
  Sherman–Morrison rank-1 inverse updates for O(d²) per step, and it avoids materializing the full (K,d)
  joint-feature matrix. **This is the script that produced the 2.34× headline.** Supports checkpoint
  resume and `--passive_full`.
- [`run_qwen_experiment.py`](code/amazoncat13k/run_qwen_experiment.py) — Driver for the BGE-1024d + Qwen
  arm (the within-dataset embedding contrast that flips the regime back to "LinUCB-suffices").

**LLM caching**
- [`cache_llm_predictions_amazon.py`](code/amazoncat13k/cache_llm_predictions_amazon.py) — Pre-compute
  and cache Flan-T5 predictions for all AmazonCat samples (AmazonCat prompt format).

**Diagnostics**
- [`compute_diagnostics.py`](code/amazoncat13k/compute_diagnostics.py) — Three-axis diagnostics adapted
  for 13,330 entities (computes the huge cosine matrix in memory-friendly blocks).
- [`build_npz_for_estimates.py`](code/amazoncat13k/build_npz_for_estimates.py) — Build the `.npz`
  (joint features `X`, true index `y`, expert index) that `pre_runtime_estimates.py` consumes. Uses a
  hard-negative candidate set per context (true label + LLM proposal + high-similarity negatives) because
  the full 13,330-action tensor won't fit in memory.
- [`pre_runtime_estimates.py`](code/amazoncat13k/pre_runtime_estimates.py) — The paper's **net-value
  rule**: estimates cold vs warm potential `Phi`, expert cost `C_E(tau)`, and predicts the multiplier
  before running the full experiment. Fills the diagnostic-table TODOs.

**Baselines / references (from the original Chen et al. codebase)**
- [`bandit.py`](code/amazoncat13k/bandit.py) — Pure bandit baseline.
- [`corral_clip.py`](code/amazoncat13k/corral_clip.py) — CORRAL master with probability clipping (the
  randomized meta-algorithm we contrast the deterministic Router against).
- [`llm_agent.py`](code/amazoncat13k/llm_agent.py) — LLM-as-decision-maker baseline.

**Performance / plots**
- [`benchmark_step.py`](code/amazoncat13k/benchmark_step.py) — Micro-benchmark of a single Router step
  (the 13,330-action UCB scoring is the bottleneck).
- [`plot_amazon_dynamics.py`](code/amazoncat13k/plot_amazon_dynamics.py) — The phase-transition learning
  curves (Router vs Cold around t≈10k–15k where the 2× gap opens).

### 4.3 `code/synthetic_text/` — controlled text dataset attempts (v2→v8)

This is where most of the recent iteration happened. **Goal:** build a *text* dataset (so real
Flan-T5/Qwen can classify it) whose structure forces the warm-start-essential regime. The recurring
struggle: with too few categories (K=52) LinUCB is not starved (T/K≈96), and with discriminative
bag-of-words features LinUCB solves the task almost instantly — so multiplication stays ≈1.0×.

**Dataset generators** (each encodes a different hypothesis about what triggers multiplication):
- [`generate_synthetic_text_dataset.py`](code/synthetic_text/generate_synthetic_text_dataset.py) — v1:
  200 diverse categories, ~50k docs, template text with category vocab. First attempt at "low coherence
  via semantic diversity."
- [`generate_dataset_v2.py`](code/synthetic_text/generate_dataset_v2.py) — 30+ keywords/category,
  category name never appears in text, confuser keywords for a heavy gap tail.
- [`generate_dataset_v3.py`](code/synthetic_text/generate_dataset_v3.py) — The 52-category / d=150 base
  used by most later runs; defines `CATEGORIES`, `CONFUSER_PAIRS`, templates reused downstream.
- [`generate_dataset_v4.py`](code/synthetic_text/generate_dataset_v4.py) — Complex `θ*`, higher d, more
  categories so LinUCB genuinely needs many samples; also fixes constant-`phi`-norm (which made routing
  all-or-nothing).
- [`generate_dataset_v5.py`](code/synthetic_text/generate_dataset_v5.py) — "Both methods are bad but make
  *different* errors" hypothesis: 52 categories, d=300 with noise so LinUCB converges slowly.
- [`generate_dataset_v6.py`](code/synthetic_text/generate_dataset_v6.py) — **Category-dependent
  complementarity**: split categories into a LinUCB-friendly group (clean features, confusing text) and an
  LLM-friendly group (noisy features, distinctive text), sharing one projected space so passive V-updates
  transfer. Best real-LLM multiplication achieved here (~1.09× Flan-T5-large).
- [`generate_dataset_v7.py`](code/synthetic_text/generate_dataset_v7.py) — Planted-model "quick fix" to
  pass all three axes simultaneously (orthogonal QR arm features → α̂≈0, high-rank doc features → large r̂,
  confusable categories → gap tail). Outcome: axes passed but **no multiplication** (Router degenerates to
  0% or 100% LLM) — a key negative datapoint.
- [`generate_dataset_v8.py`](code/synthetic_text/generate_dataset_v8.py) — **Large-K** push (aiming K=200,
  actually ~124) so T/K shrinks toward the starved regime, with 10 broad domains × 20 categories.
- [`embed_features_v8.py`](code/synthetic_text/embed_features_v8.py) — Replaces bag-of-words features
  with `all-MiniLM-L6-v2` (384d) sentence embeddings — continuous, overlapping, noisy — to stop LinUCB
  from trivially solving the task. This was the response to "Cold LinUCB hits 99% because BoW features are
  perfectly separable."

**Experiment runners**
- [`run_text_experiment.py`](code/synthetic_text/run_text_experiment.py) — v3 pipeline: LLM classifies
  raw text into 52 categories → cache → subspace diagnostic → Algorithm 1.
- [`run_experiment_v4.py`](code/synthetic_text/run_experiment_v4.py) — v4 pipeline with c-sweep.
- [`run_experiment_v5.py`](code/synthetic_text/run_experiment_v5.py) — v5 pipeline; supports
  `--all_local`, `--sweep_only`, and cluster models (Qwen).
- [`run_experiment_v6.py`](code/synthetic_text/run_experiment_v6.py) — v6 pipeline; supports different
  document orderings (`random`, `llm_first`, `interleaved`, `cluster`) to test whether front-loading LLM
  rounds strengthens the warm-start. **This is the current best-understood runner** — reuse its structure.
- [`run_experiment_v8.py`](code/synthetic_text/run_experiment_v8.py) — v8 runner with an explicit
  two-step SSH workflow: `--mode predict` (cache LLM once) then `--mode experiment --c_sweep`.

**Analysis / helpers**
- [`c_sweep.py`](code/synthetic_text/c_sweep.py) — Re-run *only* the routing sweep over `c` from existing
  caches (~2s per model per `c`). Use this to characterize multiplication vs `c`.
- [`deep_diagnostic.py`](code/synthetic_text/deep_diagnostic.py) — Why multiplication is weak:
  per-document LLM/LinUCB complementarity, LinUCB convergence speed, `phi`-norm geometry, routing
  transition dynamics.
- [`quick_llm_test_v4.py`](code/synthetic_text/quick_llm_test_v4.py) — Sample a few docs to estimate LLM
  accuracy before committing to a full (slow) cache run.

### 4.4 `code/synthetic_clean/` — fully artificial mechanism check

- [`experiment_synthetic.py`](code/synthetic_clean/experiment_synthetic.py) — **No real LLM.** Generates
  `θ*`, random unit-norm arm features (α̂≈1/√d), Gaussian contexts, and a simulated "LLM" =
  `(1-noise)·θ* + noise·random`. Runs the exact Algorithm 1 routing + c-sweep. Purpose: prove the
  mechanism produces multiplication in minutes (we observed ~1.2× at c≈5 with K=1000, T/K=5). Caveat: a
  reviewer may object that the expert is not a real model — this is a sanity check, not a paper result.
  Parameters worth remembering: large `K` relative to `T` (starved), `--noise` controls expert quality.

### 4.5 `code/synthetic/` — earlier numeric synthetic + diagnostics

- [`generate_and_diagnose.py`](code/synthetic/generate_and_diagnose.py) — Builds three synthetic datasets
  designed to make **all three axes deviate** (e.g. standard-basis orthogonal arms, isotropic context,
  small-gap `θ*`), then runs diagnostics + Algorithm 1. Good reference for how the axes are supposed to
  move together.
- [`plot_synthetic_results.py`](code/synthetic/plot_synthetic_results.py) — Diagnostic table, Algorithm 1
  comparison, and the T=5000 vs T=50000 horizon-effect figure.

### 4.6 `code/llm_rl_from_claude_LOOK_AT_THIS/` — theory validation suite

Pure-math checks that the paper's analytical claims are correct (no data, just linear algebra + tiny
simulations). Independent of the empirical pipeline.
- [`pre_runtime_estimates.py`](code/llm_rl_from_claude_LOOK_AT_THIS/pre_runtime_estimates.py) — Reference
  copy of the net-value-rule estimator (same intent as the AmazonCat one).
- [`validation_suite/arrowhead_check.py`](code/llm_rl_from_claude_LOOK_AT_THIS/validation_suite/arrowhead_check.py)
  — Verifies the closed-form "arrowhead matrix inverse" lemma against a dense numpy inverse.
- [`validation_suite/canonical_check.py`](code/llm_rl_from_claude_LOOK_AT_THIS/validation_suite/canonical_check.py)
  — Verifies the three-way separation / one-pass propositions on the canonical worst-case instance.
- [`validation_suite/diagnostic_check.py`](code/llm_rl_from_claude_LOOK_AT_THIS/validation_suite/diagnostic_check.py)
  — Checks the pre-runtime net-value rule `N(tau)` against the true multiplier on template instances.
- [`validation_suite/template_sim.py`](code/llm_rl_from_claude_LOOK_AT_THIS/validation_suite/template_sim.py)
  — Shared LinUCB simulation harness used by the checks above.
- [`validation_suite/sweep_and_row7.py`](code/llm_rl_from_claude_LOOK_AT_THIS/validation_suite/sweep_and_row7.py)
  — Epsilon/price-ratio scaling sweep and the "row-7 nuisance-probe" variant where the hybrid should be
  provably trapped after handoff.

---

## 5. Where We Left Off / Open Threads

1. **Synthetic-with-real-LLM is unfinished.** v6 gives a mild ~1.09×; v7 passes the diagnostics but shows
   **no** multiplication; v8 (large-K + sentence embeddings) was the in-progress attempt to get a clean,
   real-LLM positive result. The blocker is the tension between "features informative enough for the LLM
   to classify" and "features hard enough that Cold LinUCB is genuinely starved." Sentence embeddings
   ([`embed_features_v8.py`](code/synthetic_text/embed_features_v8.py)) are the current best lever; the
   other lever is pushing K well past 124 so T/K is small.

2. **Axis A contradiction (paper-level).** `r̂/√T` is saturated for *every* dataset, including the one
   that multiplies (AmazonCat). Needs an editorial/theoretical resolution before submission — see §2.2.
   Coherence (Axis B) is the axis that actually separates regimes; the diagnostic table was recently
   reordered to A, B, C, subspace to match the decision-tree flow, but the A-mispredicts issue remains.

3. **The paper only firmly needs the two real datasets.** WikiLinks (negative, robust across a full `c`
   sweep) + AmazonCat (positive, 2.34×, with a within-dataset embedding flip) already tell the regime
   story. The synthetic section is corroborating evidence; if it can't be made clean in time, it can be
   scoped down.

---

## 6. How to Reproduce (quick reference)

The universal pattern is **cache the LLM once, then sweep the bandit for free**:

```bash
# --- WikiLinks (v1 = BERT+Flan-T5) ---
cd code/oneshotwikilinks/new_testing
python make_data.py                                  # build embeddings pickle
python cache_llm_predictions.py --llm_type large     # one-time GPU pass
python experiment_algorithm1.py --llm_type large --llm_confidence 100

# --- AmazonCat (BERT+Flan-T5, the 2.34x result) ---
cd code/amazoncat13k
python cache_llm_predictions_amazon.py --llm_type large
python -u experiment_algorithm1.py --llm_type large --llm_confidence 100 --passive_full

# --- Synthetic text v8 (real LLM, SSH two-step) ---
cd code/synthetic_text
python run_experiment_v8.py --mode predict    --model flan-t5-large
python run_experiment_v8.py --mode experiment --model flan-t5-large --T 10000 --c_sweep

# --- Fully artificial mechanism check (no GPU, ~minutes) ---
cd code/synthetic_clean
python experiment_synthetic.py --d 200 --K 1000 --N 20000 --T 5000 --c_sweep --seeds 3
```

Notes:
- **GPU** is only needed for the LLM caching / inference step. Every bandit sweep after that is CPU numpy.
- `--passive_full` (update `b` on LLM rounds too) matters for the AmazonCat warm-start; keep it on to
  reproduce 2.34×.
- Result JSONs record `hybrid` (Router), `llm`, `linucb_cold`, and `llm_fraction` over steps — feed these
  to the `generate_plots*.py` / `analyze_routing.py` scripts.
- Data files (`*.pkl.gz`, `mydataset.*`, `llm_cache_*`, result JSONs) are **not** copied here — only code.
  Regenerate them with the `make_data*` / `cache_*` scripts, or copy them from the working tree / SSH box.
