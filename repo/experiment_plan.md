# Experimental plan: closing the empirical gaps in v5

*Written against `handoff.zip` (code as of the v1 submission) and `v5.tex`.*

---

## 0. Where we actually are

### What the paper now claims empirically

v5's theory is done. Its empirical section makes five claims, and the code you have supports only the first two:

| # | Claim in v5 | Status | Where in the paper |
|---|---|---|---|
| 1 | Twelve configurations; one multiplies (AmazonCat / BERT / Flan-T5-large, 2.34×) | **Have the numbers** | Tables amazon-results, wiki-results |
| 2 | WikiLinks is robust to a $c$-sweep; BGE collapses to 100% LLM at $c=100$ | **Have the numbers** (with three inconsistencies to fix) | Table wiki-csweep |
| 3 | The pre-runtime estimates $\widehat\Phi$ / $\widehat N$ predict which configuration multiplies | **Not computed.** The Predicted column is blank for exactly the three rows that matter | Table diagnostics, 3 `\todo` |
| 4 | AmazonCat's gain is carried by $V_\tau$ (the tail channel, row 5′); Flan-T5 small/base have equal accuracy but less information | **Inferred, not measured.** Nothing in the logs separates $A_\tau$ from $V_\tau$ | Sec. amazon, 1 `\todo` |
| 5 | Baselines: CORRAL routing, offline warm start, BGE $c$-sweep at small $c$, seeds for AmazonCat | **Not run.** All promised in the rebuttal | Sec. amazon, 3 `\todo` |

Everything below is about turning claims 3–5 from prose into numbers.

### What the code can and can't do today

The good news: the pipeline design is right. **LLM predictions are cached once; every bandit run afterwards is pure numpy on CPU** and takes minutes to hours, not days. The GPU step is already done for all twelve configurations (the `llm_cache_*` files). So almost nothing below needs a GPU.

The specific gaps:

- **`pre_runtime_estimates.py` already implements $\widehat\Phi$, $\widehat N$, and the dry run** exactly as v5 defines them. And `build_npz_for_estimates.py` builds its input. Neither appears to have been run for the paper — the table is blank. This is the cheapest and most valuable item on the list.
- **The experiment scripts log only running means every 1,000 steps.** They do not log the reward on each round or whether the round was routed. Without that you cannot compute $A_\tau$, $V_\tau$, $B_\tau$ from a run. One logging change unlocks claim 4.
- **Result filenames do not include the seed.** Running `--seed 2` overwrites `--seed 1`. Must fix before any multi-seed work.
- **`corral_clip.py` is Chen et al.'s original code**: SpannerGreedy plus live LLM calls plus CORRAL, on GPU. It is not a drop-in baseline for our stream. The CORRAL baseline should be implemented as a routing mode inside our own runner, using the same cached predictions — about 40 lines.
- **Paper/code mismatch on the exploration schedule.** v5 states $\alpha_t=\max(0.1,0.9985^t)$. WikiLinks code uses 0.9985 but AmazonCat code uses **0.9999**. Those are very different: 0.9985 hits the floor at $t\approx1{,}500$; 0.9999 at $t\approx23{,}000$. The paper must state both or the AmazonCat runs must be redone at 0.9985. Decide before running anything else so you don't have to run twice.

### Things in `CONTEXT.md` that are now moot

The handoff document describes the v1 framing. Three threads it lists as open are closed by the rewrite and should not be pursued:

- **"Axis A contradiction"** ($\widehat r/\sqrt T$ saturated everywhere). The axes are gone. $\widehat r$ survives only as one of four cheap proxies in a table, with no predictive claim attached.
- **"Reorder the decision tree."** There is no decision tree. The prediction is the net-value rule.
- **Synthetic-text v2–v8** (getting a *real* LLM to multiply on a constructed text dataset). The paper no longer needs it; the catalog of eleven proven instances does that job, and v7's "axes pass but no multiplication" result is now explained by Theorem general-price (passing a proxy is not a price ratio). Recommend dropping this thread. If you want a synthetic figure, Phase 8 below is far cheaper.

---

## 1. The one rule to follow

**Cache once, sweep free.** Every step below reads cached LLM labels and runs numpy. If you find yourself about to call a language model, stop — you already have what you need. The only exception would be re-embedding a dataset, which we do not need.

The artifacts you need on disk before starting (regenerate from the `make_data*` / `cache_*` scripts or copy from the SSH box):

```
oneshotwikilinks/new_testing/   mydataset.0.pickle.gz          (BERT-768d)
                                mydataset.0.v3.pickle.gz       (BGE-1024d)
                                llm_cache_{small,base,large}.*  (Flan-T5)
                                llm_cache_v3_*.*                (Qwen 0.5B/3B/7B)
amazoncat13k/                   mydataset.0.pickle.gz          (BERT-768d)
                                mydataset.0.v3.pickle.gz       (BGE-1024d)
                                llm_cache_amazon_v1_{small,base,large}.pkl.gz
                                llm_cache_amazon_v3_*.pkl.gz
```

---

## 2. Phase 0 — code housekeeping (half a day; do first)

Three changes to `experiment_algorithm1.py` in both dataset folders. Nothing else works without them.

**0.1 Put the seed in the filename.**
`tag = f"algo1_amazon_{model_family}_{llm_type}_c{llm_confidence}"` → append `_s{seed}`. Same for the checkpoint name. Same in the WikiLinks scripts.

**0.2 Log every round, not every 1,000.**
Add four arrays to `history`, appended on every step: `r_router`, `r_cold`, `r_llm` (each 0/1), and `routed` (1 if the Router deferred to the LLM this round). Save them as a compressed `.npz` alongside the JSON (50k × 4 bytes is nothing; 622k × 4 is still small). Keep the 1,000-step JSON summaries as they are so the existing plotting scripts still work.

Why: with per-round rewards and the routed flag, $A$, $V$, $B$ are one `numpy` line each (Phase 2). Without them, claim 4 stays prose.

**0.3 Add a `--router` argument** with three values: `gate` (current behaviour), `corral`, `offline`. Details in Phase 6; add the flag now so the logging format is shared.

**0.4 Decide the exploration schedule.** Either (a) change `alpha_decay` default in the AmazonCat script to 0.9985 and re-run the three BERT configurations (they're needed for seeds anyway — Phase 4), or (b) keep 0.9999 and change the paper to say so. Option (a) is cleaner. Whatever you choose, the pre-runtime estimates in Phase 1 must use the matching $\beta$.

---

## 3. Phase 1 — the $\widehat\Phi$ columns and the Predicted column (one day; highest value)

**What this settles.** The paper's central empirical claim: that a quantity computed *before* running anything predicts which configuration multiplies. Also the "equal accuracy, unequal information" paragraph, which currently rests on inference.

**What exists.** `pre_runtime_estimates.py` computes exactly v5's quantities:
- $\widehat\Phi(\lambda I)$ and $\widehat\Phi(\lambda I+(\tau/n)\widehat M_{\text{Exp}})$ — the function `phi_hat`
- $\widehat C_{\text{Exp}}(\tau)=\tau\times$ expert error rate
- $\widehat N(\tau)=(T-\tau)[\widehat\Phi_{\text{cold}}-\widehat\Phi_{\text{warm}}]-\widehat C_{\text{Exp}}(\tau)$
- the dry run (Phase 3)

`build_npz_for_estimates.py` builds its input for AmazonCat using a hard-negative candidate set of $A=128$ per context, because the full $13{,}330\times768$ tensor won't fit.

**Steps.**

1. *Align hyperparameters.* The script defaults to `--lam 0.1 --beta 2.0`. Use `--lam 0.1` (matches the experiments) and set `--beta` to the experiment's exploration coefficient. The experiments use $\alpha_t$ starting at 1.0, so **use `--beta 1.0`** and state that in the table caption. Run once more with `--beta 0.1` (the floor) as a sensitivity check; if the ordering of configurations changes between the two, say so.

2. *Build the AmazonCat inputs.* One `.npz` per Flan-T5 size:
   ```
   python build_npz_for_estimates.py --family v1 --llm_types small base large --n_held 1000 --A 128
   ```
   Use the same held-out indices for all three (the script seeds this; keep `--seed 42`). Then the Qwen sizes with `--family v3`.

3. *Build the WikiLinks inputs.* There is no builder yet; write one modelled on the AmazonCat one, but with the **full** 311-action set (no candidate subsampling needed: 1000 × 311 × 768 floats ≈ 0.9 GB in float32, fine; use `n_held=500` if memory is tight). One `.npz` per model.

4. *Run the estimates.* For AmazonCat, $T=50{,}000$ and the router routed roughly 5,000 rounds concentrated early, so:
   ```
   python pre_runtime_estimates.py amazon_v1_large.npz --T 50000 --taus 1000 2000 5000 10000 --lam 0.1 --beta 1.0
   ```
   For WikiLinks, $T=622{,}000$; the router routed under 0.3% (Flan-T5) or ~6% (Qwen), so `--taus 1000 5000 20000 40000`.

5. *Fill the table.* Add three columns to `tab:diagnostics`: $\widehat\Phi(\lambda I)$, $\Delta\widehat\Phi$ at the $\tau$ closest to the observed routed count, and $\widehat N(\tau)/((T-\tau)\widehat\Phi(\lambda I))$. Fill Predicted using v5's rule: "multiply" if that last ratio is at least $1/3$ and the expert's held-out accuracy is below the ridge oracle's; "no gain" otherwise.

**What to look for — this is the decision point for the paper.**

| Outcome | What it means | What to write |
|---|---|---|
| $\Delta\widehat\Phi$ for large ≫ small, base | The "equal accuracy, unequal information" story is **confirmed**. Large's proposals close directions the others don't. | Keep the paragraph; cite the numbers. This is the paper's best empirical hook. |
| $\Delta\widehat\Phi$ roughly equal across the three | The story is **wrong as written**. Something other than the covered directions explains 2.34× vs 1.01×. | Replace the paragraph. Candidates: the dry run (Phase 3) will tell you whether the difference is in $A_\tau$; the per-round logs (Phase 2) will tell you *when* the three diverge. Be honest about it — a reviewer who catches an unsupported mechanism story is worse than a paper that says "we do not know which term." |
| $\widehat N<0$ for large but $\widehat N\ll0$ for small/base | Consistent with v5's own caveat: $\widehat N$ misses the concurrent channel, so it can be negative on a configuration that multiplies. Relative ordering still informative. | Report it as such; lean on the dry run for the prediction. |
| WikiLinks: $\Delta\widehat\Phi\approx0$ everywhere | Expected. Confirms the framework predicts the six negatives. | One sentence. |

Whatever the outcome, **also recompute the subspace angle for each expert separately** (the `85.1°` copied across three rows and `83.1°` across three rows are flagged `TODO(check)` in the tex). `compute_diagnostics.py` computes it; it needs the per-expert Gram matrix $\widehat M_{\text{Exp}}$, which `build_npz_for_estimates.py` now gives you. If the three angles genuinely coincide, that is a *finding*: the angle proxy cannot separate the three, which is exactly why the paper needs $\widehat\Phi$.

---

## 4. Phase 2 — measure $A_\tau$, $V_\tau$, $B_\tau$ from the runs (half a day after Phase 0)

**What this settles.** v5 says AmazonCat is a row-5′ instance: $A_\tau<0$ small, $V_\tau$ large, $B_\tau$ large. Right now that is read off a learning curve. With per-round logs it becomes a measurement.

**How.** For an adaptive router, use the routed set $\mathcal T$ as v5's Remark after the decomposition prescribes (the identities hold for any routed set, not just an initial segment). With binary reward, regret is $1-r$ on every round, so:

```
A = sum over routed rounds     of (r_router - r_cold)      # expert's direct advantage
V = sum over non-routed rounds of (r_router - r_cold)      # value of transferred data
B = sum over non-routed rounds of (r_router - r_llm)       # tail advantage over the expert
```

Check: `A + V` must equal `sum(r_router) - sum(r_cold)` exactly, and `B` must equal `sum(r_router) - sum(r_llm)` minus the routed-round difference. If not, the logging is wrong.

Also compute cumulative versions over time and plot all three for the Flan-T5-large run. That figure — three curves, $V$ taking off after routing stops — would replace or accompany `fig:amazon-dynamics` and is the picture of "the saving is collected on the tail."

Do this for all twelve configurations; it costs nothing once the logs exist. For the six negatives you expect $V\approx0$ and $A\le0$; for BGE at $c=100$ you expect every round routed, so $V$ and $B$ are empty and $A<0$ — which is the "hyperparameter failure" the paper already describes.

**What to look for.** For Flan-T5-large: is $V$ actually the large term? If $A$ turns out to be large and positive instead, the configuration is row 5, not row 5′, and "Where the experiments sit" needs one word changed. Either is fine; the paper should say what happened.

---

## 5. Phase 3 — the dry run (already coded; run alongside Phase 1)

**What this settles.** $\widehat N$ only sees the tail channel. The dry run — simulate cold LinUCB and warm-started LinUCB on replayed held-out contexts — sees both channels, and produces an *implied multiplier* before any real online round. v5 proposes it as the replacement for the net-value rule.

**How.** It runs automatically inside `pre_runtime_estimates.py` (`--T_dry`, default 20,000). Set `--T_dry 20000` for AmazonCat and note the scaling. For each configuration record `R_cold`, `R_warm_tail`, and the implied multiplier.

**What to look for.** The implied multiplier should be **above 1 for Flan-T5-large and near or below 1 for small/base** on AmazonCat, and near 1 for all WikiLinks configurations. If it is, add a column "Dry-run mult." to the diagnostics table and this becomes the Predicted column's actual basis. If the dry run says large should *not* multiply, that is a serious problem with the replay assumption (contexts resampled i.i.d. from held-out may not reproduce the online stream's structure) — report it and keep $\widehat N$ as the prediction instead.

---

## 6. Phase 4 — seeds for AmazonCat BERT (compute-bound; start early and let it run)

**What this settles.** The 2.34× is a single point estimate on the one configuration that matters. Reviewer PAT flagged it; the WikiLinks table has mean ± std and the AmazonCat table does not.

**How.** After Phase 0.1 and 0.4, run seeds 1, 2, 3 for small, base, large at $c=100$ with `--passive_full`. That is nine runs of 50,000 rounds × 13,330 actions × 768d. `benchmark_step.py` tells you the cost of one step on your machine; from the handoff notes the UCB scoring over 13k actions is the bottleneck. Budget accordingly and run them in parallel if you have cores. Use `--resume`; the checkpointing exists for this reason.

Then update `tab:amazon-results` to mean ± std, matching the WikiLinks table.

**What to look for.** Is 2.34× stable? If seed variance puts small or base above 1.1× or large below 1.8×, the story softens and the paper's language should too. The seed changes the stream order, so this also tests whether the tail-channel timing (routing stops, then curves separate) is a property of the data or of one permutation.

---

## 7. Phase 5 — BGE $c$-sweep at small $c$ (one hour; promised to reviewer hk3d)

**What this settles.** At $c=100$ with unit-norm BGE features, $u^{(1)}=1/\sqrt c=0.1$ is below $u^{(0)}\ge1/\sqrt\lambda=3.16$ forever, so the Router defers to the LLM on every round. v5 calls this a hyperparameter failure and predicts that smaller $c$ recovers cold LinUCB's 39%.

**How.** `run_qwen_experiment.py --skip_experiment` is not what you want; you want the bandit sweep on cached predictions. Run the AmazonCat runner with `--model_family v3 --llm_type {0.5B,3B,7B} --llm_confidence {0.01, 0.1, 1, 10}`. Four values × three models = twelve CPU runs.

**What to look for.** As $c$ falls, the routed fraction should drop from 100% and the Router's accuracy should rise toward 39%. Report the best $c$ and the routed fraction at it. If the Router *never* recovers 39% at any $c$ — if it plateaus below cold LinUCB — then it isn't only a hyperparameter failure and the paragraph must change.

Also, while you're in sweeps: **confirm the WikiLinks sweep grid.** The tex says $\{50,100,\dots\}$ in one place and the sweep table reports $c=50$ as best for Qwen 3B/7B; one of them was edited without the other. Check the actual JSON filenames.

---

## 8. Phase 6 — the two baselines (one to two days; the theory now predicts what they should show)

Both were promised in the rebuttal. Both are now more interesting than ordinary baselines, because Theorem general-price and the decomposition remark make **testable predictions** about them.

### 6.1 CORRAL routing

**Implement** as `--router corral` in our runner (do not use `corral_clip.py`; it's a different pipeline). Maintain a probability $p_t$ of routing to the LLM; play LLM with probability $p_t$, LinUCB otherwise; after observing reward, form the importance-weighted loss $\ell_t/p_{t,i}$ for the played base and update $p_t$ by log-barrier OMD (the update in Chen et al.'s Algorithm 3 — a one-dimensional root find for the normaliser $\lambda$, since there are two bases). Clip the LinUCB probability at $p_{\min}=0.2$ as they do. Feed **every** observation to LinUCB's $(V,b)$ regardless of who chose it, exactly as the gate does.

**Prediction to test.** The decomposition holds for any routed set. CORRAL and the gate differ only in *which* rounds are routed. So: the theorem says CORRAL's multiplier is also bounded by its own price ratio, and if the two routers happen to route similar numbers of rounds, their $V$ terms should be similar and any difference should show up in $A$. Compute $A,V,B$ for CORRAL (Phase 2 logging) and set them beside the gate's.

**What to look for.** Does CORRAL also multiply on Flan-T5-large? If yes with a similar $V$, the paper's claim that the mechanism is the data and not the routing rule is confirmed. If CORRAL fails to multiply where the gate does, look at *how many* rounds CORRAL routed — with $p_{\min}=0.2$ it will route far more than the gate's 3.4%, paying acquisition regret on ~20% of 50,000 rounds at a 5%-accurate expert, and the decomposition says exactly why that kills it: $A_\tau$ goes hugely negative. Either result is a clean paragraph.

### 6.2 Offline warm start (row 1)

**Implement** as `--router offline --tau N`: take $N$ contexts from a held-out set (not the online stream), play the LLM's cached proposal on each, record $(x,r)$, and initialise LinUCB's $(V,b)$ from them. Then run plain LinUCB on the online stream with no routing. Compare tail regret to cold LinUCB.

**Prediction to test.** This is the "acquisition not charged" case. Its multiplier over cold can exceed the online one because $A_\tau$ is not paid. On AmazonCat/large, offline warm start with $N=5{,}000$ should beat the online gate; on small/base it should do roughly nothing (the data doesn't cover the right directions). Report both.

**What to look for.** If offline warm start with large's proposals *doesn't* help, then large's data isn't informative either and the 2.34× came from the concurrent channel ($A$) — cross-check against Phase 2.

---

## 9. Phase 7 — fix the table inconsistencies (an afternoon; no compute)

These are flagged `TODO(check)` in the tex. A reviewer who spots any of them stops trusting the table.

1. **`12.58%` Router with `100%` LLM routing but LLM alone is `12.43%`** (WikiLinks c-sweep, Flan-T5-large at $c=25{,}000$). If routing is truly 100% the two must be equal. Open the JSON: either the routed fraction is 99.x% (report it exactly) or the accuracy is wrong.
2. **Identical `85.1°` for AmazonCat small/base/large** and **`83.1°` for the three Qwen sizes.** Recompute from each expert's $\widehat M_{\text{Exp}}$ (Phase 1 gives you this). See the note in Phase 1 about what it means if they genuinely coincide.
3. **Sweep grid $\{100,\dots\}$ vs reported best $c=50$.** Check the filenames.
4. **Paper $\alpha_t$ decay 0.9985 vs AmazonCat code 0.9999.** Phase 0.4.

---

## 10. Phase 8 — synthetic figure from the catalog (optional; one day; no LLM)

Reviewer hk3d asked for synthetic stress tests that isolate one ingredient at a time. The catalog does this in proofs; a figure would do it in simulation and costs almost nothing because the harness exists.

`llm_rl_from_claude_LOOK_AT_THIS/validation_suite/template_sim.py` and `sweep_and_row7.py` already simulate the template. The verification script I wrote for row 5′ (`check_row5p.py`, in the outputs folder) simulates instance $\mathcal J$.

**Figure:** multiplier over cold LinUCB vs. $\varepsilon$ (probe gap) on the row-5 template, with the theoretical bracket $[m_0/(\varepsilon n), (16\lambda+1)/(\varepsilon n)]$ drawn as a band, and the row-5′ instance as a second series. Add row 8 ($\varepsilon\ge c^2/L^2$) to show the collapse to $\le2\times$. One panel; makes the price ratio visible.

This replaces the entire synthetic-text thread with something that is provably tied to the theory. If time is short, skip it — the catalog is sufficient.

---

## 11. Order and rough cost

| Phase | What | Wall time | Depends on |
|---|---|---|---|
| 0 | Housekeeping: seed in filename, per-round logs, router flag, decide $\alpha$ decay | ½ day | — |
| 1 | $\widehat\Phi$, $\widehat N$, Predicted column, recompute angles | 1 day | data on disk |
| 3 | Dry run | runs inside Phase 1 | — |
| 7 | Table inconsistencies | ½ day | nothing |
| 4 | Seeds ×3 for AmazonCat BERT | compute-bound (days of CPU, in background) | Phase 0 |
| 2 | $A,V,B$ from logs + figure | ½ day | Phase 0, and Phase 4 runs finishing |
| 5 | BGE small-$c$ sweep | 1 hour | Phase 0 |
| 6 | CORRAL + offline baselines | 1–2 days | Phase 0 |
| 8 | Synthetic figure from catalog | 1 day, optional | nothing |

**Start Phase 0 and Phase 1 the same day.** Phase 1 is the single most important result and needs no new code. Kick off Phase 4 as soon as Phase 0 is done and let it run while you do 2, 5, 6, 7.

---

## 12. What goes in the paper, by outcome

Once Phases 1–6 are in, the empirical section should be able to say each of the following with a number behind it. If any line can't be filled, that line's claim comes out of the paper.

- "$\Delta\widehat\Phi$ is [X] for Flan-T5-large and [Y], [Z] for small and base" — Table diagnostics
- "The dry run predicts multipliers of [a, b, c]; the observed are 1.01, 0.94, 2.34" — Table diagnostics
- "On the Flan-T5-large run, $A=[\ ]$, $V=[\ ]$, $B=[\ ]$; $V$ carries [%] of the saving" — Sec. amazon + figure
- "Over three seeds: $2.34\pm[\ ]\times$" — Table amazon-results
- "At $c=[\ ]$ the Router on BGE routes [%] and reaches [%], recovering cold LinUCB" — Sec. amazon
- "CORRAL routing routes [%] of rounds and achieves [ ]×; its $A$ is [ ] against the gate's [ ]" — Sec. amazon
- "Offline warm start from 5,000 of large's proposals reaches [ ]×; from small's, [ ]×" — Sec. amazon

The prose is already written for the expected outcomes. Where the outcome differs, change the prose, not the number.
