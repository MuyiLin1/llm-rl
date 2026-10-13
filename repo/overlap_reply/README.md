# Reply: feature-overlap measurements on AmazonCat (BERT-768d)

This answers `teammate-overlap-analysis.md`. Items 0–5 follow the order of its "What to send back" list. Everything was run on the default BERT configuration of Table 3: T = 50,000, λ = 0.1, α_t = max(0.1, 0.9985^t), c_LLM = 100, seeds 1–10, Flan-T5-small, -base and -large. All three parts are done, including the optional Part C.

**Short version.**
- 1471's feature is far from constant: cos(u_t, ū) has mean 0.61 and 5th percentile 0.43.
- Its overlap with other categories is about 0.4–0.6 for any category, frequent or random, and it comes mostly from the category vectors themselves.
- At the handoff, no run is in the theorem's "settles" range. That holds even with the scale fixed and with ω replaced by its 99th percentile.
- Only if the state is taken at the last routed round, which is often thousands of rounds after routing has mostly stopped, do 11 of the 19 settled runs (and none of the 11 unsettled runs) get h > 0.
- Part C: the oracle reward is essentially 0 (mean 0.0045), so the new condition behaves like "reward hidden".

The details are below. Item 5 lists everything surprising.

**Folder contents**

| path | what |
|---|---|
| `results/partA.json` | A1, A2, A3, the G2 list, and cold LinUCB's settle category per seed |
| `results/partB.csv`, `results/partB_step4.csv` | Part B at τ = last routed round (your definition); 30 rows each |
| `results/partB_p99.csv`, `results/partB_step4_p99.csv` | The same at τ₉₉ = the round by which 99% of routed rounds are done (extra; see 5.4) |
| `results/partC.json` | Part C: all five conditions, tests, sanity checks and the oracle rewards fed |
| `results/ubar.npy` | ū from A1 (unit length, 768-d) |
| `states/{expert}_s{seed}.npz` | The 30 learner states at round τ + 1: `Vinv`, `theta`, `tau`, `alpha_tau`, `settled`, `n_1471_correct_routed` (4.7 MB each) |
| `scripts/` | Copies of every script used (item 4) |

---

## 0. Code map

The runner is `amazoncat13k/experiment_algorithm1.py`. Line numbers below refer to that file unless another file is named.

| What | Where | Answers |
|---|---|---|
| Context embedding x_t | `amazoncat13k/make_data.py` → `makeData` encodes the title and the content separately with `SentenceTransformer('bert-base-nli-mean-tokens')` and stores `Xs = [title ‖ content]` (1,536-d). `get_context_embeddings` (l. 134) and `amazoncat13k/make_slim_npz.py` keep the **first 768 coordinates**, so x_t is the BERT embedding of the **product title**. | **Not L2-normalized**: ‖x_t‖ ranges from 13.6 to 17.6 (mean 16.1) over the 60,000 kept contexts. **No bias or constant coordinate** (no coordinate has zero variance). NaN/inf are set to 0. |
| Category vectors z_a | `make_data.py` → `getCategories`: `model.encode([label])`, the same BERT model applied to the **category name**. Loaded by `get_entity_embeddings` (l. 120). 13,330 categories. | **Not normalized**: ‖z_a‖ ranges from 14.0 to 17.5 (mean 16.7); ‖z_1471‖ = 17.05. No zero vectors. |
| Joint feature φ(x, a) | `learnOnline`: `x_cold = cx * Z[cold_choice]` (l. 515), `x_llm = cx * Z[llm_choice]`, `x_linucb = cx * Z[linucb_choice]` (l. 525–526). The scorer `make_scorer` (l. 187) computes `Z @ (cx * theta) + alpha * sqrt(var)`. | Confirmed: **φ(x, a) = x ⊙ z_a, with no extra terms** in the default configuration. |
| LinUCB state | Router learner: `V_inv_warm`, `b_warm`. Cold learner: `V_inv_cold`, `b_cold` (l. 271, 275). θ̂ = `V_inv_warm @ b_warm` is recomputed every round. | **V⁻¹ is kept directly**: initialized to (1/λ)I and updated by Sherman–Morrison in `sm_update` (l. 64). V itself is never formed. |
| Exploration scale | l. 489: `current_alpha = max(0.1, linucb_alpha * alpha_decay ** step_i)`, with defaults `linucb_alpha = 1.0`, `alpha_decay = 0.9985`, `lam = 0.1` (l. 230). | Confirmed. `step_i` is 0-based, so round t (1-based) uses 0.9985^(t−1). The floor 0.1 is reached at round 1,535; every τ in Part B is later, so α_τ = 0.1 in all 30 runs. |
| Router decision | l. 529–533: `u_llm = ‖φ(x, a_LLM)‖ / sqrt(c)`, `u_linucb = sqrt(φ(x, a_UCB)ᵀ V⁻¹ φ(x, a_UCB))`; route to the expert iff `u_llm < u_linucb`. | There is no α in the width comparison. On a routed round, V is updated with the expert's feature, and b gets the observed reward when the run uses `--passive_full`. The Table 3 Router does use it: the Router "feeds every observation to LinUCB". |
| "Settle" flag | `prerun_study/replay/paper/stats.py` → `settle(a, cat=1471)` | A run **settles on 1471 if 1471 is more than 20% of the Router's plays in the last fifth of the run** (rounds 40,001–50,000). For "settles on something else" (G3), I used the same rule with the most-played category of that last fifth. |
| Oracle θ_or | `amazoncat13k/transfer_decomp.py` → `fit_oracle(C, Z, ys, lam=1.0)`. It is called by `prerun_study/replay/replay.py` → `get_theta` and saved as `prerun_study/replay/theta_or_v1.npy`. | It is a ridge regression (λ = 1) of the binary reward on φ(x, a) over **all** (context, category) pairs of the 10,000 held-out contexts 50,000–59,999, which are outside the online stream. ‖θ_or‖ = 0.0031. This is the vector behind the transfer-removed curve. |

**How Part B was run.** I did not add a hook to the original runs. The per-round actions, rewards and routing flags of the Table 3 Router runs are already logged (`prerun_study/replay/out/{expert}_s{seed}.npz`). These logs come from `prerun_study/sim.py`, a re-implementation of the runner's cold and Router tracks with the same semantics. Its unmodified runs reproduce Table 3 exactly: Router accuracy 14.37 / 22.80 / 27.64% and 2 / 8 / 9 of 10 settled for small / base / large.

From these logs I rebuilt V = λI + Σ_{t≤τ} φ_t φ_tᵀ and b = Σ_{t≤τ} r_t φ_t. I checked each rebuild by re-scoring round τ + 1 and comparing the result with the logged action: **30 of 30 match**. The states in `states/` are therefore the exact learner state at τ + 1.

---

## 1. Part A (seed 1 stream, 5,000 sampled contexts)

Of the 5,000 sampled contexts, 30.1% have 1471 as a correct label. u_t = φ(x_t, 1471).

### A1. How much 1471's feature changes across contexts

| contexts | n | cos(u_t, ū) mean | sd | p5 | min | ‖u_t‖ mean | sd | min | max |
|---|---|---|---|---|---|---|---|---|---|
| all | 5,000 | 0.606 | 0.097 | 0.426 | 0.072 | 10.80 | 0.93 | 8.69 | 14.54 |
| 1471 correct | 1,503 | 0.601 | 0.100 | 0.421 | 0.072 | 11.05 | 0.89 | 8.92 | 14.54 |
| 1471 not correct | 3,497 | 0.609 | 0.095 | 0.430 | 0.145 | 10.69 | 0.92 | 8.69 | 14.35 |

1471's feature is far from fixed: a typical u_t is about 53° away from the mean direction. The norm is stable, with a coefficient of variation of 9%. Whether 1471 is a correct label makes almost no difference.

### A2. How much 1471's feature overlaps with other categories

cos(φ(x_t, 1471), φ(x_t, b)), pooled over the 5,000 contexts and the categories in each group:

| group | categories | pairs | mean | median | p95 | max |
|---|---|---|---|---|---|---|
| G1 pooled | 3 | 15,000 | 0.532 | 0.534 | 0.700 | 0.815 |
| G1: 7961 | 1 | 5,000 | 0.602 | 0.592 | 0.734 | 0.815 |
| G1: 7892 | 1 | 5,000 | 0.448 | 0.440 | 0.598 | 0.707 |
| G1: 7891 | 1 | 5,000 | 0.547 | 0.539 | 0.690 | 0.787 |
| G2 pooled | 20 | 100,000 | 0.400 | 0.413 | 0.606 | 0.787 |
| G3: 7892 (only member) | 1 | 5,000 | 0.448 | 0.440 | 0.598 | 0.707 |
| G4 pooled (1,000 random) | 1,000 | 5,000,000 | 0.501 | 0.506 | 0.681 | 0.904 |

**G2:** the 20 categories that cold LinUCB plays most often, excluding 1471. The counts are pooled over seeds 1–10, i.e. 500,000 plays in total.

| rank | category | plays | rank | category | plays |
|---|---|---|---|---|---|
| 1 | 7892 | 64,569 | 11 | 7889 | 1,409 |
| 2 | 6502 | 16,629 | 12 | 2372 | 1,348 |
| 3 | 9202 | 13,884 | 13 | 6845 | 1,344 |
| 4 | 9905 | 9,091 | 14 | 2347 | 1,325 |
| 5 | 2421 | 6,494 | 15 | 2319 | 1,321 |
| 6 | 1755 | 2,498 | 16 | 7821 | 1,287 |
| 7 | 379 | 2,361 | 17 | 13006 | 1,253 |
| 8 | 7891 | 2,194 | 18 | 9784 | 1,240 |
| 9 | 7452 | 1,981 | 19 | 486 | 1,226 |
| 10 | 9913 | 1,538 | 20 | 7907 | 1,114 |

**G3:** cold LinUCB settles on **7892** (the 11.3% category) in seeds 1, 4, 5, 7, 8 and 10. In seeds 2, 3, 6 and 9 no category reaches 20% of the last fifth. G3 is therefore the single category 7892, which is also in G1. Cold LinUCB never settles on 1471 (0/10, as in Table 3).

### A3. Is the overlap structural?

| category | cos(z_1471, z_b) | mean cos(φ(x, 1471), φ(x, b)) (A2) |
|---|---|---|
| 7961 (G1) | 0.534 | 0.602 |
| 7892 (G1, G3) | 0.372 | 0.448 |
| 7891 (G1) | 0.482 | 0.547 |
| G2, 20 categories: mean / p95 / max | 0.334 / 0.485 / 0.547 | 0.400 (pooled mean) |
| G4, 1,000 random: mean / p95 / max | 0.433 / 0.569 / 0.825 | 0.501 (pooled mean) |
| all 13,329 other categories: mean / p95 / max | 0.437 / 0.583 / 0.895 | — |

**Yes, the overlap is structural.** The BERT category-name vectors all share a large common component: 1471's mean cosine with an arbitrary category is 0.44. The x² weighting in Σ_i x_i² z_{1471,i} z_{b,i} adds only about +0.07 on top. The weight is spread across coordinates: the 10 largest coordinates of E[x²] carry 4.6% of its mass.

---

## 2. Part B (30 runs, state at τ + 1)

**Setup.** Step 2 used **all 13,329 competitors b ≠ 1471**, with no top-500 shortcut, for 200 contexts per run: 2.67 million (context, b) pairs per run. Following the runner's scorer, the widths vᵀV⁻¹v are computed in float32. All other quantities are float64.

### 2a. Step 3 exactly as specified (u = ū, unit length)

| expert | seed | τ | α_τ | n_1471_correct_routed | settled | N | μ̂ | ρ_max | ρ_99 | frac_ρ≥1 | ω | ω_99 | h | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| small | 1 | 2179 | 0.1 | 0 | no | 870.5 | -0.0074 | 8.77 | 2.96 | 0.284 | 12109 | 1.32 | -1.05e+07 | n/a |
| small | 2 | 2335 | 0.1 | 0 | yes | 966.5 | -0.0082 | 8.87 | 2.87 | 0.265 | 76193 | 1.18 | -7.36e+07 | n/a |
| small | 3 | 3914 | 0.1 | 0 | no | 1903.2 | -0.0099 | 9.10 | 2.71 | 0.271 | 5817 | 0.49 | -1.11e+07 | n/a |
| small | 4 | 1981 | 0.1 | 0 | no | 755.0 | -0.0056 | 9.14 | 3.02 | 0.263 | 17136 | 1.57 | -1.29e+07 | n/a |
| small | 5 | 3242 | 0.1 | 0 | no | 1588.1 | -0.0098 | 8.44 | 2.81 | 0.299 | 26016 | 0.76 | -4.13e+07 | n/a |
| small | 6 | 4117 | 0.1 | 0 | no | 2045.6 | -0.0093 | 8.65 | 2.70 | 0.268 | 7368 | 0.43 | -1.51e+07 | n/a |
| small | 7 | 4146 | 0.1 | 1 | yes | 2147.2 | -0.0070 | 8.45 | 2.64 | 0.261 | 8597 | 0.35 | -1.85e+07 | n/a |
| small | 8 | 8703 | 0.1 | 0 | no | 4329.0 | -0.0149 | 7.47 | 2.72 | 0.325 | 12931 | 0.16 | -5.60e+07 | n/a |
| small | 9 | 4382 | 0.1 | 0 | no | 2563.8 | -0.0105 | 8.86 | 2.73 | 0.291 | 4433 | 0.41 | -1.14e+07 | n/a |
| small | 10 | 4830 | 0.1 | 0 | no | 2069.0 | -0.0139 | 8.41 | 2.66 | 0.274 | 30237 | 0.62 | -6.26e+07 | n/a |
| base | 1 | 4729 | 0.1 | 2 | no | 1962.7 | -0.0145 | 8.25 | 2.66 | 0.236 | 25156 | 0.60 | -4.94e+07 | n/a |
| base | 2 | 2189 | 0.1 | 5 | yes | 828.0 | 0.0020 | 10.00 | 2.98 | 0.255 | 7159 | 1.43 | -5.93e+06 | n/a |
| base | 3 | 3914 | 0.1 | 3 | yes | 2201.2 | -0.0024 | 7.63 | 2.56 | 0.220 | 69660 | 0.48 | -1.53e+08 | n/a |
| base | 4 | 2037 | 0.1 | 5 | yes | 779.5 | 0.0057 | 9.62 | 3.04 | 0.257 | 55397 | 1.50 | -4.32e+07 | n/a |
| base | 5 | 2502 | 0.1 | 2 | no | 836.0 | -0.0060 | 9.02 | 2.70 | 0.215 | 62278 | 1.15 | -5.21e+07 | n/a |
| base | 6 | 2484 | 0.1 | 5 | yes | 4277.3 | 0.0383 | 13.03 | 3.99 | 0.328 | 7893 | 1.04 | -3.38e+07 | n/a |
| base | 7 | 2157 | 0.1 | 3 | yes | 765.1 | 0.0013 | 9.42 | 2.95 | 0.258 | 52037 | 1.64 | -3.98e+07 | n/a |
| base | 8 | 2427 | 0.1 | 4 | yes | 1959.7 | 0.0247 | 10.31 | 3.36 | 0.289 | 290160 | 1.33 | -5.69e+08 | n/a |
| base | 9 | 4447 | 0.1 | 2 | yes | 15089.2 | 0.0377 | 17.30 | 4.50 | 0.386 | 3365 | 0.55 | -5.08e+07 | n/a |
| base | 10 | 2350 | 0.1 | 5 | yes | 1457.0 | 0.0178 | 9.59 | 2.90 | 0.241 | 84076 | 1.27 | -1.23e+08 | n/a |
| large | 1 | 4139 | 0.1 | 5 | yes | 15276.8 | 0.0396 | 14.60 | 4.59 | 0.394 | 8645 | 0.67 | -1.32e+08 | n/a |
| large | 2 | 2772 | 0.1 | 4 | yes | 1976.7 | 0.0161 | 9.91 | 2.90 | 0.246 | 17086 | 1.01 | -3.38e+07 | n/a |
| large | 3 | 3914 | 0.1 | 3 | yes | 13198.6 | 0.0405 | 16.58 | 4.32 | 0.336 | 5219 | 0.76 | -6.89e+07 | n/a |
| large | 4 | 2466 | 0.1 | 5 | yes | 1056.0 | 0.0025 | 9.82 | 2.79 | 0.220 | 15121 | 1.10 | -1.60e+07 | n/a |
| large | 5 | 2238 | 0.1 | 4 | no | 909.3 | 0.0018 | 9.40 | 2.91 | 0.223 | 60768 | 1.47 | -5.53e+07 | n/a |
| large | 6 | 2673 | 0.1 | 6 | yes | 5578.3 | 0.0409 | 15.26 | 4.24 | 0.326 | 4951 | 0.85 | -2.76e+07 | n/a |
| large | 7 | 2683 | 0.1 | 2 | yes | 1304.5 | -0.0000 | 8.81 | 2.70 | 0.204 | 19447 | 0.96 | -2.54e+07 | n/a |
| large | 8 | 8703 | 0.1 | 2 | yes | 43290.9 | 0.0400 | 19.10 | 4.64 | 0.364 | 9146 | 0.34 | -3.96e+08 | n/a |
| large | 9 | 3030 | 0.1 | 4 | yes | 2836.2 | 0.0188 | 9.53 | 3.25 | 0.307 | 48054 | 0.97 | -1.36e+08 | n/a |
| large | 10 | 7470 | 0.1 | 8 | yes | 38803.0 | 0.0384 | 18.22 | 4.96 | 0.420 | 1558 | 0.48 | -6.04e+07 | n/a |

**This version is degenerate because of scale.** ū has unit length, but the real features have ‖φ(x, b)‖ ≈ 10–15 (A1). As a result, ρ = ⟨v, V⁻¹u⟩ / uᵀV⁻¹u is inflated by about ‖u_t‖ ≈ 10.8: 20–42% of pairs have ρ ≥ 1, and pairs with ρ just below 1 send w = (I − ρμ̂)/(1 − ρ) into the thousands.

### 2b. The same quantities with u at the scale of the real feature (u = m·ū, m = 10.80)

I think this is the version the theorem intends, since its settling action's feature is the actual φ. m = 10.80 is the mean ‖u_t‖ from A1. Rescaling gives N → N/m², μ̂ → m·μ̂ and ρ → ρ/m; I is unchanged. These are the `sc_*` columns in `partB.csv`.

I also add the margin and bound computed with **ω_99 in place of ω**, because ω is set by a handful of pairs out of 2.67 million.

| expert | seed | settled | N | μ̂ | ρ_max | frac_ρ≥1 | ω | ω_99 | h | bound | h (ω_99) | bound (ω_99) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| small | 1 | no | 7.5 | -0.080 | 0.81 | 0 | 0.504 | 0.168 | -4.36 | n/a | -1.85 | n/a |
| small | 2 | yes | 8.3 | -0.088 | 0.82 | 0 | 0.559 | 0.151 | -5.36 | n/a | -1.99 | n/a |
| small | 3 | no | 16.3 | -0.107 | 0.84 | 0 | 0.443 | 0.099 | -8.98 | n/a | -3.36 | n/a |
| small | 4 | no | 6.5 | -0.061 | 0.85 | 0 | 0.592 | 0.197 | -4.22 | n/a | -1.67 | n/a |
| small | 5 | no | 13.6 | -0.106 | 0.78 | 0 | 0.619 | 0.119 | -9.87 | n/a | -3.06 | n/a |
| small | 6 | no | 17.5 | -0.101 | 0.80 | 0 | 0.319 | 0.084 | -7.37 | n/a | -3.25 | n/a |
| small | 7 | yes | 18.4 | -0.076 | 0.78 | 0 | 0.547 | 0.095 | -11.46 | n/a | -3.15 | n/a |
| small | 8 | no | 37.1 | -0.161 | 0.69 | 0 | 0.580 | 0.059 | -27.49 | n/a | -8.16 | n/a |
| small | 9 | no | 22.0 | -0.113 | 0.82 | 0 | 0.605 | 0.089 | -15.80 | n/a | -4.46 | n/a |
| small | 10 | no | 17.7 | -0.150 | 0.78 | 0 | 0.476 | 0.106 | -11.11 | n/a | -4.53 | n/a |
| base | 1 | no | 16.8 | -0.156 | 0.76 | 0 | 0.863 | 0.159 | -17.14 | n/a | -5.30 | n/a |
| base | 2 | yes | 7.1 | 0.022 | 0.93 | 0 | 0.569 | 0.184 | -3.88 | n/a | -1.15 | n/a |
| base | 3 | yes | 18.9 | -0.025 | 0.71 | 0 | 0.494 | 0.097 | -9.80 | n/a | -2.30 | n/a |
| base | 4 | yes | 6.7 | 0.062 | 0.89 | 0 | 0.508 | 0.190 | -2.98 | n/a | -0.85 | n/a |
| base | 5 | no | 7.2 | -0.064 | 0.83 | 0 | 0.359 | 0.135 | -3.04 | n/a | -1.43 | n/a |
| base | 6 | yes | 36.7 | 0.414 | 1.21 | 0.00001 | 0.816 | 0.287 | -14.76 | n/a | 4.66 | 0.343 |
| base | 7 | yes | 6.6 | 0.014 | 0.87 | 0 | 0.468 | 0.178 | -2.98 | n/a | -1.07 | n/a |
| base | 8 | yes | 16.8 | 0.266 | 0.95 | 0 | 0.704 | 0.221 | -7.35 | n/a | 0.76 | 0.374 |
| base | 9 | yes | 129.4 | 0.407 | 1.60 | 0.00005 | 0.861 | 0.245 | -58.81 | n/a | 20.97 | 1.000 |
| base | 10 | yes | 12.5 | 0.192 | 0.89 | 0 | 0.456 | 0.179 | -3.30 | n/a | 0.16 | 0.144 |
| large | 1 | yes | 131.0 | 0.428 | 1.35 | 0.00004 | 0.904 | 0.271 | -62.27 | n/a | 20.51 | 0.987 |
| large | 2 | yes | 16.9 | 0.174 | 0.92 | 0 | 0.576 | 0.168 | -6.82 | n/a | 0.09 | 0.088 |
| large | 3 | yes | 113.2 | 0.438 | 1.53 | 0.00008 | 0.934 | 0.296 | -56.14 | n/a | 16.04 | 0.212 |
| large | 4 | yes | 9.1 | 0.027 | 0.91 | 0 | 0.472 | 0.172 | -4.03 | n/a | -1.31 | n/a |
| large | 5 | no | 7.8 | 0.019 | 0.87 | 0 | 0.618 | 0.202 | -4.67 | n/a | -1.42 | n/a |
| large | 6 | yes | 47.8 | 0.442 | 1.41 | 0.00004 | 0.882 | 0.278 | -21.05 | n/a | 7.80 | 0.706 |
| large | 7 | yes | 11.2 | -0.000 | 0.82 | 0 | 0.734 | 0.160 | -8.22 | n/a | -1.79 | n/a |
| large | 8 | yes | 371.2 | 0.432 | 1.77 | 0.00008 | 0.985 | 0.240 | -205.05 | n/a | 71.50 | 1.000 |
| large | 9 | yes | 24.3 | 0.203 | 0.88 | 0 | 0.482 | 0.166 | -6.78 | n/a | 0.89 | 0.611 |
| large | 10 | yes | 332.7 | 0.415 | 1.69 | 0.00014 | 1.010 | 0.251 | -197.93 | n/a | 54.48 | 1.000 |

### 2c. Step 4: per-context version (same 200 contexts, state at τ + 1)

| expert | seed | settled | μ̂_t mean | μ̂_t sd | μ̂_t p5 | width β‖φ(x_t, 1471)‖_{V⁻¹} mean | fraction of contexts where 1471 has the top index |
|---|---|---|---|---|---|---|---|
| small | 1 | no | -0.049 | 0.070 | -0.168 | 0.072 | 0.000 |
| small | 2 | yes | -0.059 | 0.057 | -0.157 | 0.068 | 0.000 |
| small | 3 | no | -0.065 | 0.059 | -0.155 | 0.048 | 0.000 |
| small | 4 | no | -0.034 | 0.059 | -0.139 | 0.077 | 0.000 |
| small | 5 | no | -0.064 | 0.080 | -0.191 | 0.053 | 0.005 |
| small | 6 | no | -0.067 | 0.045 | -0.142 | 0.046 | 0.000 |
| small | 7 | yes | -0.049 | 0.052 | -0.130 | 0.046 | 0.000 |
| small | 8 | no | -0.097 | 0.058 | -0.193 | 0.031 | 0.000 |
| small | 9 | no | -0.073 | 0.058 | -0.169 | 0.043 | 0.000 |
| small | 10 | no | -0.088 | 0.043 | -0.158 | 0.043 | 0.000 |
| base | 1 | no | -0.082 | 0.094 | -0.227 | 0.046 | 0.000 |
| base | 2 | yes | 0.017 | 0.067 | -0.078 | 0.076 | 0.005 |
| base | 3 | yes | -0.018 | 0.053 | -0.104 | 0.048 | 0.000 |
| base | 4 | yes | 0.040 | 0.080 | -0.088 | 0.079 | 0.005 |
| base | 5 | no | -0.038 | 0.051 | -0.117 | 0.068 | 0.000 |
| base | 6 | yes | 0.259 | 0.248 | -0.079 | 0.055 | 0.375 |
| base | 7 | yes | 0.004 | 0.074 | -0.102 | 0.078 | 0.005 |
| base | 8 | yes | 0.182 | 0.169 | -0.066 | 0.064 | 0.355 |
| base | 9 | yes | 0.252 | 0.286 | -0.182 | 0.033 | 0.515 |
| base | 10 | yes | 0.119 | 0.114 | -0.062 | 0.067 | 0.280 |
| large | 1 | yes | 0.244 | 0.280 | -0.121 | 0.034 | 0.465 |
| large | 2 | yes | 0.100 | 0.127 | -0.110 | 0.059 | 0.220 |
| large | 3 | yes | 0.266 | 0.306 | -0.127 | 0.037 | 0.440 |
| large | 4 | yes | 0.025 | 0.074 | -0.082 | 0.067 | 0.005 |
| large | 5 | no | 0.021 | 0.079 | -0.099 | 0.073 | 0.005 |
| large | 6 | yes | 0.228 | 0.233 | -0.055 | 0.051 | 0.335 |
| large | 7 | yes | 0.005 | 0.072 | -0.102 | 0.063 | 0.010 |
| large | 8 | yes | 0.277 | 0.286 | -0.092 | 0.021 | 0.540 |
| large | 9 | yes | 0.134 | 0.127 | -0.050 | 0.053 | 0.280 |
| large | 10 | yes | 0.274 | 0.307 | -0.178 | 0.023 | 0.555 |

Using the fixed direction is roughly fair for the mean. m·μ̂ in 2b tracks the per-context mean μ̂_t, for example 0.41 vs 0.25 for base s9 and −0.08 vs −0.05 for small s1. It hides a large spread, though: μ̂_t's 5th percentile is negative in every run. Even in the strongest settled runs, 1471 has the top index on only about half of the contexts.

---

## 3. Part C row (same format as `tab:inject50k`)

The new condition is "+75 on 1471, oracle reward" on Flan-T5-small, seeds 1–10, 50,000 rounds. On a routed changed context, the learner's response sum gets ⟨θ_or, φ(x_t, a_t)⟩ in place of the observed reward. Everything else is the existing `inject50k` code. The tests are the same as in the table: a paired t-test on Router accuracy and an exact McNemar test on settling on 1471. Multipliers and regret ratios are relative to cold LinUCB (8.40%).

| Change | Expert acc. | Router acc. | Mult. (regret ratio) | Settle on 1471 | Routed changed | p (acc. / settle) vs none | p (acc. / settle) vs +75 on 1471 |
|---|---|---|---|---|---|---|---|
| none | 5.20% | 14.37 ± 6.12% | 1.71× (1.07) | 2/10 | — | — | — |
| +75 random | 5.35% | 15.42 ± 9.41% | 1.84× (1.08) | 3/10 | 2.4 | 0.63 / 1.00 | 0.13 / 0.22 |
| +75 on 1471 | 5.35% | 21.50 ± 6.80% | 2.56× (1.17) | 7/10 | 2.6 | 0.007 / 0.063 | — |
| +75 on 1471, reward hidden | 5.35% | 15.45 ± 5.71% | 1.84× (1.08) | 3/10 | 2.6 | 0.61 / 1.00 | 0.042 / 0.13 |
| **+75 on 1471, oracle reward** | **5.35%** | **13.14 ± 5.82%** | **1.56× (1.05)** | **2/10** | **2.6** | **0.53 / 1.00** (t = −0.65) | **0.006 / 0.063** (t = −3.62) |

Your request asks for the multiplier as mean ± sd. The table's convention is mean accuracy over mean cold accuracy (1.56×). The per-seed ratio acc/cold has sd 0.56 for the oracle row (0.70 for hidden, 1.40 for "+75 on 1471").

**Sanity checks.**
- In every seed, the oracle run's actions are identical to the "+75 on 1471" run up to the first routed changed context.
- Seed 7 routes none of the changed contexts, and its whole run is identical to "+75 on 1471", as it should be.
- 26 oracle rewards were fed in total (2.6 per seed). Their mean is **0.0045**, replacing a reward of 1.

---

## 4. Scripts, commit, seeds

**Scripts** (copies in `scripts/`). They are meant to run from their original locations in the repo; the paths are hard-coded.

| script | purpose |
|---|---|
| `prerun_study/overlap/overlap_analysis.py` | Parts A and B. `python overlap_analysis.py` gives τ = last routed round; `python overlap_analysis.py p99` gives τ₉₉. |
| `prerun_study/overlap/partC.py` | Part C statistics (the same tests as `replay/paper/stats.py`) |
| `prerun_study/replay/inject50k_oracle.py` | Part C runs. `python inject50k_oracle.py 3 {0,1,2} 1 10` launches 3 workers over seeds 1–10. |
| `prerun_study/replay/inject50k.py`, `hide.py` | The existing injection test (which 75 answers change); unmodified |
| `prerun_study/sim.py` | The simulator behind all replay runs (runner semantics) |
| `prerun_study/replay/paper/stats.py` | The existing table statistics (settle rule, McNemar) |

**Commit.** The repo HEAD is `4266e588c5cf2cf39ce330a9a8faccfb1f411fa9`. However, **the experiment code used here is not committed**: `experiment_algorithm1.py`, `make_slim_npz.py`, `transfer_decomp.py`, `prerun_study/` and `reproducibility_package/` are untracked, and `make_data.py` has uncommitted changes. The commit hash therefore does not pin the code. These SHA-256 prefixes do:

| file | sha256 (first 12) |
|---|---|
| amazoncat13k/experiment_algorithm1.py | a91b1f3e7fa0 |
| amazoncat13k/make_data.py | 56b96636ef57 |
| amazoncat13k/make_slim_npz.py | 49682e876302 |
| amazoncat13k/transfer_decomp.py | a118a9f94bfb |
| prerun_study/sim.py | 22ce275e03f6 |
| prerun_study/overlap/overlap_analysis.py | 7439834519b0 |
| prerun_study/overlap/partC.py | f3c6f892c52d |
| prerun_study/replay/inject50k_oracle.py | 17dffba2c351 |
| prerun_study/replay/inject50k.py | beb428a09f94 |
| prerun_study/replay/hide.py | f9a143f1c221 |
| prerun_study/replay/theta_or_v1.npy (θ_or) | 15b51aadb590 |
| reproducibility_package/data/amazon_v1_slim.npz (data) | 5a959961a2f6 |

**Random seeds.**
- Stream order for seed s: `np.random.seed(s); np.random.permutation(50000)`, the same as the runner.
- Part A: the 5,000 contexts are `np.random.default_rng(0).choice(50000, 5000, replace=False)`, taken as positions in seed 1's stream. G4 is `np.random.default_rng(1).choice(13,329 non-1471 categories, 1000, replace=False)`.
- Part B, step 2: the 200 contexts are `np.random.default_rng(0).choice(...)` over the rounds after τ, separately for each run.
- Part C: same streams as the existing table; no new randomness.

---

## 5. Surprises

1. **At the actual handoff, no run is in the "settles" range.**
   - Routing ends early. 99% of routed rounds are done by round τ₉₉ = 1,573–1,956 in every run. After that, the Router routes only a handful of rounds, sometimes thousands of rounds later; this is why your τ ranges from 1,981 to 8,703.
   - At τ₉₉ (`partB_p99.csv`, scaled version), h ≤ 0 in all 30 runs.
   - With ω_99 in place of ω, h ≤ 0 in 29 of 30 runs. The exception is large s6, with h = 0.38 but ω_99 = 0.301 > 0.298, so the bound is still n/a.
   - At τ₉₉, μ̂ is small: the scaled μ̂ is at most 0.33. 1471 has the top index on at most 17.5% of contexts, and on ≤ 2% in 27 of 30 runs.

2. **At your τ, the state already includes LinUCB's own settling.**
   - The 11 runs with h > 0 (with ω_99) are all settled runs. None of the 11 unsettled runs has h > 0. The bound ranges from 0.09 to 1.00.
   - This separation looks good, but between τ₉₉ and τ the learner has made hundreds to thousands of its own plays. Settled runs with a late τ (large s8, s10; base s9) are exactly the ones with large N and h.
   - So at τ, the theorem's quantities partly measure settling that has already happened, rather than predicting it from the handoff.

3. **ω is above 0.298 in every run, so the bound is n/a in all 30 runs, and h ≤ 0 in all 30 (including all 19 settled runs) when the true maximum ω is used.**
   - This holds even in the scaled version. ω is 0.32–1.01 and comes from a few pairs out of 2.67 million: ω_99 is only 0.06–0.30.
   - A version of the theorem that tolerates a small fraction of competitors would change the picture at τ, but not at τ₉₉ (point 1).

4. **The literal unit-ū version is meaningless at this feature scale.**
   - With ū of unit length, ρ ≥ 1 on 20–42% of pairs and ω reaches the thousands.
   - At the real feature norm (‖u_t‖ ≈ 10.8), ρ < 1 on all but ≤ 0.014% of pairs, and ρ_max is 0.69–1.77.
   - The theorem's ρ condition is therefore nearly satisfied; ω is the binding quantity.

5. **Settling happens with almost no routed correct answers on 1471.**
   - Routed correct answers on 1471 before τ: 0–8, out of 1,372–1,625 routed rounds.
   - Flan-T5-small has **0** in 9 of 10 seeds. Its s2 run settles on 1471 with 0, and its s7 run with 1.
   - Base and large have 2–8.

6. **1471's feature is not nearly fixed.** cos(u_t, ū) has mean 0.61 and 5th percentile 0.43 (A1). The theorem's same-feature-every-round assumption is far from the data.

7. **The overlap is generic, not specific to frequent categories.** 1,000 random categories overlap with 1471 as much (mean cos 0.50) as the three frequent ones (0.53), and more than the categories cold LinUCB favors (G2, 0.40). It is inherited from the category-name embeddings: the mean z-cosine with all 13,329 other categories is 0.44 (A3).

8. **The Part C oracle reward is about zero, so "oracle" is in effect "hidden".**
   - θ_or is a ridge fit to binary rewards over all 13,330 categories, where almost all rewards are 0. Its norm is 0.003, and on the routed changed contexts it predicts 0.0045 on average for 1471.
   - The new row (1.56×, 2/10 settled) is not significantly different from "none" (p = 0.53) and is significantly below "+75 on 1471" (p = 0.006).
   - This means the result says "θ_or gives 1471 almost no credit on these contexts". It is not an independent test of the transfer.

9. **G3 is a single category.** When cold LinUCB settles on something other than 1471, it is always 7892 (6 of 10 seeds). The other 4 seeds do not settle under the 20% rule.
