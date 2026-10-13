# DMwithLLM: code export

This folder is a copy of all the code, logs, figures, notes and paper sources in the
`DMwithLLM` repository, taken on 2026-10-12. Use it to read and analyse the project without the
97 GB working copy. Nothing in the original repository was moved or changed.

**Not included:** the bulk data (cached LLM predictions, embeddings, datasets, per-run arrays,
about 56 GB) and the 44 GB of git history. Every left-out data file is listed with its size, and
with its array names and shapes for `.npz` and `.npy` files, in `_meta/DATA_MANIFEST.tsv`. So you
can see what each script reads and writes even though the data itself is not here.

---

## 1. Layout of this export

```
export/
├── README.md                  <- this file
├── _build_export.py           rebuilds this folder from the repo (python3 export/_build_export.py)
├── repo/                      the working tree, same paths as the repo, data files removed
├── _from_git_head/            files committed in git but deleted from the working tree
│   └── oneshotwikilinks/      the original (upstream) WikiLinks scripts, see section 6
└── _meta/
    ├── DATA_MANIFEST.tsv      every left-out data file: path, bytes, arrays inside (npz/npy)
    ├── FILES_INCLUDED.tsv     every file copied into repo/: path, bytes
    ├── git_log.txt            full commit history with file stats
    ├── git_status.txt         tracked files changed since the last commit
    └── git_uncommitted_changes.diff   uncommitted edits to amazoncat13k/*.py
```

What goes into `repo/`: every file **except** `.gz .npz .npy .pkl .pickle .zip`, files over 50 MB
(only the Miniconda installer), `.pyc` / `__pycache__`, `.DS_Store`, `.pid`, and two symlinks in
`prerun_study/phaseb/` that point at the slim datasets. The copy has 1,239 files and is about 44 MB.

---

## 2. What the project is

The project studies an **LLM + contextual bandit hybrid**. A frozen LLM (the "expert") and a
LinUCB learner share one stream of decisions. The **Router** (the "confidence-gated meta-algorithm",
called Algorithm 1 in the code) chooses on each round which of the two acts:

```
u_LLM    = ||phi(x, a_LLM)|| / sqrt(c)        # c = c_LLM, the confidence parameter
u_LinUCB = sqrt(phi^T V^{-1} phi)
play the LLM's action if u_LLM < u_LinUCB, else LinUCB's action
LinUCB's statistics (V, b) are updated on every round from whatever was played ("passive_full")
```

Every experiment runs three policies on the same stream:

- the **Router**;
- the **LLM alone**;
- **cold LinUCB**, which starts from λI and gets no help from the LLM.

The question is **when the Router beats the better of the other two, and why**.

**Datasets**
- **AmazonCat-13K**: 13,330 categories; reward is Precision@1. This is the main positive result.
- **OneShotWikiLinks**: 311 entities. Here the Router gives little or no gain.
- **Synthetic datasets** (`synthetic*`) and small theorem-instance simulators (`prerun_study/transfer_thm`, `prerun_study/router_sim`).

**Features**
- `v1` means BERT-768d.
- `v3` means BGE-1024d.
- φ(x, a) = x ⊙ z_a, where x is the context embedding and z_a is the category embedding.

**Experts:** Flan-T5 small, base and large. Some runs use Qwen2.5 (0.5B, 3B, 7B).

**Paper**
- It moved from NeurIPS 2026 (`v5`, `v6`, `v7.tex`) to **ALT 2027**. `repo/v8.tex` is the current working file.
- `repo/alt_submission/main.tex` is an Oct 9 snapshot of the ALT version, kept with the class files.
- The ALT deadline is Oct 12, 2026.

### Concepts you need before reading the code

**Two scales for one effect.** Keep these apart; mixing them up is the most common mistake.

- *Accuracy multiplier* = Router accuracy / max(LLM accuracy, cold accuracy). This is the headline number.
- *Regret ratio* = (1 − max(LLM, cold)) / (1 − Router). This is what the theorems bound.
- Example: Flan-T5-large on AmazonCat goes from 8.39% to 27.64% accuracy. That is a 3.29× accuracy multiplier but only a 1.27 regret ratio.
- Both ratios use seed-averaged accuracies, not averages of per-seed ratios.

**Coverage vs transfer.** A warm start gives LinUCB two things:

- a design matrix V, called **coverage**;
- a response sum b. The part of b that a linear model cannot reproduce is called the **transfer**.

The theory says coverage drives the gain under linear rewards. On AmazonCat the transfer is what
separates three experts that have nearly the same accuracy.

**Settling / category 1471.**
- On AmazonCat, category 1471 is correct on about 30% of contexts.
- The multiplier depends mostly on whether LinUCB "settles" (locks onto) 1471 after a few of the LLM's right answers.
- Settled runs: 2, 8 and 9 of 10 for small, base and large. Router accuracy: 14.37%, 22.80% and 27.64%.
- Theorem 2 in `v8.tex` is the settling theorem.

**Seeds.** Seeds 1–10 are evaluation seeds. Seeds 11–20 are tuning seeds, used to pick c and the
LinUCB settings. One seed sets the context order for all three policies, so runs are paired.

**Default configuration.** T = 50,000, λ = 0.1, α_t = max(0.1, 0.9985^t), c_LLM = 100. On
WikiLinks/BERT the selected c is 10,000.

---

## 3. Folder guide (`repo/`)

The repo built up over time, and several folders are **copies** of others. The table marks which
folder holds the live code for each part. For other files, prefer the newest modification date.

| Folder | What it is | Live or copy |
|---|---|---|
| `amazoncat13k/` | The original AmazonCat pipeline: `make_data*.py` (embed texts), `cache_llm_predictions_amazon.py` (query the LLMs), `make_slim_npz.py` (condense into the runner's `.npz`), **`experiment_algorithm1.py`** (the main runner: Router, LLM and cold tracks), plus diagnostics (`compute_diagnostics.py`, `pre_runtime_estimates.py`, `dry_run_*.py`, `measure_price.py`, `transfer_decomp.py`, `three_conditions.py`). `results_algo1_*.json` hold the per-config results. `AmazonCat-13K.raw/Yf.txt` holds the category names. | Live. Has uncommitted edits, see `_meta/git_uncommitted_changes.diff`. |
| `oneshotwikilinks/new_testing/` | The WikiLinks version of the same pipeline, with runner versions `experiment_algorithm1{,_v2,_v3,_v4}.py`, c-sweep shell drivers, `logs/`, `figures*/` and `results_algo1*.json` (filename gives the expert, c, seed and feature version). `CHANGES_TO_ALGORITHM1.txt` and `current_project.txt` are working notes. | Live (WikiLinks) |
| `prerun_study/` | **The most recent and largest body of work** (Sept–Oct). See section 4. | Live |
| `experiments/` | A curated, numbered copy of the scripts behind the paper's numbers: `00_smoke_test` … `09_cluster_drivers`, plus `README.md` (reproduction guide, seeds, which script makes which table), `DATA.md` and `runs_manifest.csv`. **Read this README second.** | Copy, Sept 22 |
| `reproducibility_package/` | The same curated code (`code/`) plus `figures/`, `paper/` (v6) and `data/README_DATA.md` (the slim `.npz` schema). The data files are in the manifest. | Copy, Sept 22 |
| `handoff/` | Sept 6 snapshot: `CONTEXT.md` (a long explanation of every script and the reasoning behind it), `context.tex` (the paper at that time), and `code/` (a copy of every `.py` file then). | Copy, Sept 6 |
| `overlap_reply/` | Reply to `teammate-overlap-analysis.md`: how much 1471's feature varies and overlaps with other categories, and the learner state at the hand-off round. Holds `README.md` with the results tables, `scripts/` and `results/`. | Copy of `prerun_study/overlap` + `replay` |
| `llm_rl_from_claude_LOOK_AT_THIS/` | An external audit of an earlier draft (`audit_report.md`) and the `validation_suite/` of small simulators that check the theory (arrowhead inverse, template instance, row-7 sweep). | Reference |
| `synthetic/`, `synthetic_clean/`, `synthetic_text/` | Synthetic datasets. `synthetic_text/` went through generator versions v2–v8 (here vN = dataset version, not the feature version). Its `*_results*.json` and `run_cluster_*.log` are the outcomes. | Earlier exploration |
| `alt_submission/` | ALT 2027 LaTeX bundle: `main.tex`, the class files and the figures it uses. | Oct 9 snapshot |
| `figures_to_upload/` | Figures exported for the NeurIPS version. | Old |
| `glp-design-workflow/` | Empty. | — |

Top-level files in `repo/`:

**Paper sources and PDFs**
- Papers: `v5.tex` → `v6.tex` → `v7.tex` (NeurIPS) → **`v8.tex`** (ALT, current).
- Older papers: `paper.tex`, `context.tex`.
- Backup: `v8_backup_oct10_pre_settling.tex`.
- Bibliography: `references.bib`.
- Built PDFs: `LLM_RL-*.pdf`, `alt_preview.pdf`.
- `llm_rl_reviews.pdf` holds the reviews of the earlier submission.

**Notes**
- `NEXT_STEPS.md`: the Oct 4–6 plan and results so far (Phase B tuned baselines, exploration sweep, skew).
- `experiment_plan.md`, `project_proposal_*.md`: earlier plans.
- `teammate-overlap-analysis.md`: a request for feature-overlap measurements.

**Figures**
- `*.png` at the top level are the paper figures: `multiplication`, `two_channel`, `regimes`, `skew`, `exploration_sweep`, `decomposition`, …

**Other**
- `README.md` is the original upstream README (OneShotWikiLinks / AmazonCat, Corral baselines).
- `requirements.txt`, `check_row5p.py` and a few stray logs.

---

## 4. `prerun_study/` in more detail

This folder holds most of the recent analysis. The scripts share one simulator.

**The simulator and its checks**
- `sim.py` (and `sim_v1.py`) is a lean re-implementation of the runner's cold and Router tracks.
- `verify_sim.py` checks it against the cluster runner. The `wiki_*` docstrings say it matches action for action.

**Pre-run rehearsals and ablations**
- `rehearse*.py`, `build_fresh.py`, `build_placebo.py`, `analyze_*.py`.
- Outputs are in `reh_out/` (per-run `.npz`, manifest only) and `reh*_w*.log`.

**Basin and seed analysis**
- `basin_*.py`, `seed_*.py`, with `basins_online*.json` and `seeding_online.json`.
- These test whether which category LinUCB locks onto explains the multiplier.

**`replay/`**
- Re-runs with interventions: `inject50k*.py` and `hide.py` (inject 1471 answers, with and without their reward).
- `out/` holds the per-round logs (manifest only).

**`overlap/`**
- Measurements for `teammate-overlap-analysis.md`. The results are written up in `overlap_reply/README.md`.

**`phaseb/`**
- The tuned-baseline study (Tier 1: tune cold LinUCB; Tier 2: add a per-category feature; Tier 3: rerun the Router on the tuned learner), plus the exploration sweep and the skew experiment.
- Driver: `phaseb.py`. Simulators: `pb_sim.py` and `pb_batch.py`.
- Summaries: `analyze_local_out.txt`, `analyze_tiers_out.txt` and `make_figs_out.txt`. `NEXT_STEPS.md` explains them.
- `out_local/` holds one JSON per run, and `logs_local/` the driver logs.

**`wiki_c10000/`, `wiki_qwen/`**
- WikiLinks evaluation runs at the selected c, for Flan-T5/BERT and for Qwen/BGE.
- `logs/*.claim` and `*.progress` are the workers' job-lock and progress files.

**`transfer_thm/`**
- Small exact simulators for the transfer-separation theorem (`stronger.py`, `sim_full.py`, …) and `theorem_draft.tex`.

**`router_sim/`**
- The Router on the separation-theorem instance, for Appendix F. It has its own `README.md`, and `results/full/summary.md` holds the outcome.

**`restructure/`, `make_alt.py`, `*_apply.py`, `v6.before_*.tex`, `v7.before_*.tex`, `v8.before_*.tex`**
- Scripts that edited the paper, and dated backups of it before each edit.
- These are useful for tracing how the text changed, not for the experiments.

---

## 5. Where to start reading

1. This file, then `repo/experiments/README.md`, for the method, seeds and the script behind each table.
2. `repo/amazoncat13k/experiment_algorithm1.py`, the reference runner. Then `repo/prerun_study/sim.py`, the faster copy the recent studies use.
3. `repo/NEXT_STEPS.md` and `repo/prerun_study/phaseb/analyze_local_out.txt`, for the newest results.
4. `repo/v8.tex`, for the claims the code supports. Search it for `\todo` and `TODO` to find open items.
5. `repo/handoff/CONTEXT.md`, for the history and the reasoning behind older scripts. It dates from Sept 6, so some numbers in it are superseded. For example, it gives 2.34× where the current number is 3.29×.

---

## 6. Caveats for anyone analysing the code

**Data is not here.** Scripts that load data will fail on this copy.

- Most of the recent scripts load a "slim" `.npz` with the keys `contexts`, `Z` (category embeddings), `y_flat`/`y_offsets` (correct labels, CSR layout) and `llm_small|base|large` (the LLM's action per context).
- The shapes are in `_meta/DATA_MANIFEST.tsv`. Example: AmazonCat v1 has `contexts[60000, 768]` and `Z[13330, 768]`.
- The schema is documented in `repo/reproducibility_package/data/README_DATA.md`.

**Hard-coded paths.** 41 scripts contain `/Users/linmuyi/...` paths, for example `DATA = '/Users/linmuyi/code/DMwithLLM/reproducibility_package/data/wiki_v3_slim.npz'`. About 20 more refer to the `bruinml` cluster. Fix these before running anything.

**Duplicates.**
- The same script often exists in 2–4 places: `amazoncat13k/`, `experiments/`, `reproducibility_package/code/`, `handoff/code/`, `overlap_reply/scripts/`.
- The copies can differ. The live versions are in `amazoncat13k/`, `oneshotwikilinks/new_testing/` and `prerun_study/`.

**Versioned filenames.**
- `v1`/`v3` in data and result names is the feature version (BERT / BGE).
- `experiment_algorithm1_v2..v4.py` and `synthetic_text/*_v2..v8.py` are code or dataset versions.
- `v5..v8.tex` are paper versions.

**Deleted WikiLinks code.**
- The original upstream scripts (`bandit.py`, `corral_*.py`, `*_decay.py`, `llm_agent.py`, `make_data.py`, …) were deleted from `oneshotwikilinks/` but are still in git HEAD.
- They are restored in `_from_git_head/oneshotwikilinks/`. They are the Corral and decay baselines described in `repo/README.md`.

**Git tracks almost nothing.** Only 23 files are under version control. Nearly all the work is untracked, so `_meta/git_log.txt` says little about how the project developed. File modification dates are kept by the copy, so `ls -lt` is a better history.

---

## 7. Rebuilding

```bash
cd /Users/linmuyi/code/DMwithLLM
python3 export/_build_export.py
```

The script runs in a few seconds. It wipes everything in `export/` except `README.md` and the
script itself, then rebuilds `repo/`, `_from_git_head/` and `_meta/`. To include more file types,
change `DATA_EXT` or `MAX_BYTES` at the top of the script.
