# Next steps for the ALT / AISTATS submission (as of Oct 4)

Deadline: **Oct 12**. Working file: `v8.tex`. Backup before the WikiLinks edits:
`prerun_study/v8.before_wiki_results.tex`.

This document lists everything left to do, in order: what is running, how to check it, what each
experiment is for, what result we hope for, what we do with each possible result, and the writing
changes.

## Results so far (Oct 6)

Full printouts: `prerun_study/phaseb/analyze_local_out.txt` (script `analyze_local.py`).

| Experiment | Status | Result |
|---|---|---|
| BERT Tier 1+3 (tuned baseline) | done | Tuned cold LinUCB 8.37% (default 8.39%). Router on the tuned learner: 2.07x / 2.98x / 3.03x (small / base / large), all significant. |
| Exploration sweep | done | Multiplier for Flan-T5-large: 2.47x, 3.00x, 1.66x, 1.11x, 0.72x at exploration 0.1, 0.3, 1, 3, 10. The gain disappears once exploration is large, as Remark 6 predicts. |
| Skew | done | Flan-T5-large: 4.23x, 1.86x, 1.61x, 1.44x when the top category covers 30%, 20%, 10%, 5% (regret ratio 1.22, 1.05, 1.03, 1.02). The gain shrinks with the skew, as predicted. |
| BGE Tier 1+3 | running (~6 h) | Tuning seeds: best cold 35.9% vs default 34.8%. |
| BERT Tier 2 | running (~26 h) | Must finish. |
| BGE Tier 2 | queued (~89 h) | Optional; runs last. Report whatever has finished by Oct 10. |

---

## 1. The goal in one paragraph

The paper says: an LLM helps LinUCB not because it is accurate, but because a few of its right answers
land on a category that pays off a lot (on AmazonCat, the category on 29.8% of contexts). The learner
"locks in" on that category. We call this the **transfer**.

A reviewer will attack this in three ways:
1. "Your LinUCB was badly tuned."
2. "This only works on one skewed dataset."
3. "Your price-bound theory (Theorem 1) is about a different situation from your experiments."

Each experiment below answers one of these attacks.

---

## 2. What is running and how to check it

Everything runs on this Mac through one driver. It runs the experiments in priority order and moves to
the next stage automatically.

| Command | What it does |
|---|---|
| `bash prerun_study/phaseb/status_local.sh` | Shows progress and a rough time-left estimate |
| `bash prerun_study/phaseb/stop_local.sh` | Stops everything (progress is saved every ~2,000 rounds) |
| `bash prerun_study/phaseb/start_local.sh` | Starts again where it left off (add a number for workers, e.g. `start_local.sh 4`) |
| `prerun_study/phaseb/workers.sh 2` | Switch the number of workers in one step (e.g. 2 at the office, 4 at home) |
| `prerun_study/phaseb/power_check.sh` | Shows the charger's wattage and whether the battery is charging or draining |

Results go to `prerun_study/phaseb/out_local/` (one file per run). Choices made along the way (best
settings) go to `out_local/selections.json`. The log is `logs_local/driver.log`.

If the Mac gets too slow or hot: run `stop_local.sh`, then `start_local.sh 2` to use 2 workers instead
of 3. Once the temp-prl jobs (`maze_harness.py`) finish, you can use `start_local.sh 4`.

### Order the driver runs things in (priority)
1. **BERT Tier 1+3**: tune cold LinUCB, then rerun the Router with it. **Most important.**
2. **BERT exploration sweep**: new, see experiment C.
3. **Skew experiment**: new, see experiment D (4 chains, "Skew 30%" ... "Skew 05%").
4. **BGE Tier 1+3**: the same as 1, for BGE features.
5. **BERT Tier 2**: category-term features.
6. **BGE Tier 2**: category-term features for BGE. Slowest and least important; cut it if short on time.

---

## 3. The experiments, one by one

### A. Phase B Tier 1 and Tier 3: "Was LinUCB badly tuned?" (running)

**What it does.**
- Tier 1 tries 16 exploration settings (how fast exploration shrinks, and the smallest it can get),
  then 4 values of λ, for cold LinUCB alone. It picks the best using seeds 11–20 only, then reports
  that best setting on seeds 1–10.
- Tier 3 puts that best LinUCB *inside* the Router, re-tunes the Router's gate setting `c`, and
  reports on seeds 1–10.

**The multiplier is always Router ÷ cold LinUCB with the same settings.** It is never Router ÷ the
old weak LinUCB.

**First result (already in):** on BERT, the best tuned setting (fast decay, floor 0.05) gets 11.3% on
the tuning seeds. The paper's default gets 8.3%. So tuning helps cold LinUCB a little, but it is
still far below the Router's 22–28%. Good sign so far.

**What we hope for:**
- Tuned cold LinUCB about 10–13%.
- Tuned Router still about 20–28%, so the multiplier is still about 2× or more.
- The expert ordering stays the same (Flan-T5-base and -large well above -small).

**What we do with it:**
- Good result: add a "tuned baseline" table next to Table 1 and replace the Phase B TODO in
  Section 7.1 with one or two sentences.
- Bad result (tuned cold gets close to the Router): see section 5, "If things go badly".

### B. Phase B Tier 2: "What if LinUCB could learn how popular each category is?" (queued)

**What it does.** Adds a per-category term to the features, so LinUCB can learn "category 1471 is
common" without the LLM. Tunes cold LinUCB and the Router with these features.

**What we hope for:** the Router still helps.

**What we do with it:**
- If the Router still helps: one more line in the tuned-baseline table. Strong.
- If cold LinUCB now does as well as the Router: still publishable, written as "the LLM's few right
  answers supply exactly what the linear model is missing (the transfer); a popularity term supplies
  it too." That matches our theory (the transfer is the part the linear model misses), but the
  practical claim gets smaller.

### C. Exploration sweep: "Does Theorem 1 have anything to do with the experiments?" (added, queued)

**What it does.** Runs cold LinUCB and the Router (all three Flan-T5 experts, c = 100) with a fixed
exploration level of 0.1, 0.3, 1, 3 and 10, on seeds 1–10. That is 200 runs.

**Why.** The paper has two regimes:
- **Low exploration** (what we use in practice): Theorem 3 says a few right answers can make the
  learner lock in on the right category, so the gain can be large.
- **High exploration** (what the theory wants): Theorem 1 says only coverage helps, and its gain is
  small and capped. Remark 6 says the transfer effect should fade away.

**What we hope to see.** The multiplier is large at exploration 0.1 and falls steadily toward about
1.0–1.1 as exploration goes up. One plot then shows both theorems.

**What we do with it.** A new figure in Section 7: x-axis = exploration level, y-axis = multiplier,
one line per expert. Reference it from Remark 6 and from the shortened price-bound section.

### D. Skew experiment: "Does it only work on one lucky dataset?" (built, queued)

**What it does.** We make easier and harder versions of AmazonCat by removing some contexts so that
no category covers more than a set share of the stream: 30% (natural), 20%, 10%, 5%. Then we run cold
LinUCB and the Router (three experts) on each version. That is 4 levels × 4 policies × 10 seeds =
160 runs.

**Why.** The theory says the gain comes from steering the learner onto a category that pays often.
- If the best category covers 30% of the stream, finding it is worth a lot.
- At 10%, finding it is worth much less.
- At 0.36% (WikiLinks), it is worth almost nothing, which is exactly what WikiLinks shows.

So the multiplier should shrink as the cap shrinks.

**What we hope to see.** Multiplier biggest at 30%, smaller at 20%, smaller again at 10%, near 1 at
5%. WikiLinks sits at the far end of the same curve.

**What we do with it.** A new figure: x-axis = share of the most common category, y-axis =
multiplier. One sentence: "the gain follows the frequency of the category the expert steers the
learner toward, as Theorem 3 predicts." This turns "one lucky dataset" into "a predicted, measured
relationship."

**How it was built.** `prerun_study/phaseb/skew.py`. Only the 50,000 stream contexts are used. Contexts
containing the most over-represented category are removed at random (fixed seed) until no category is
above the cap. Pool sizes: 50,000 (natural, 29.8%), 43,896 (20%), 30,794 (10%), 14,808 (5%). Every
level runs for **14,000 rounds** so the levels are comparable. That is enough: the injection test saw
the full effect within 10,000 rounds. Settings are the paper's (fast/floor 0.1, λ=0.1, c=100). On the
natural level the stream is exactly the first 14,000 rounds of the paper's stream (checked action for
action), so that level doubles as a sanity check. Each run takes about 4 minutes; the whole experiment
is about 3–4 hours.

---

## 4. Writing changes (no computer time needed, can start now)

1. **Reorder the paper so the main result comes earlier.**
   New order: decomposition (A, I, B) → coverage vs. transfer → **Theorem 3** → short price-bound
   section → experiments. The price-bound section shrinks to about one page and is introduced as "the
   other regime: when exploration is high, only coverage helps, and it is capped." Theorem 2's full
   statement and the catalog move to the appendix.
2. **Make Proposition 4 a remark.** It is a two-line fact; calling it a proposition looks like padding.
3. **Give both numbers in the headline.** Abstract and intro: "1.71–3.29× in accuracy (6.5–21% less
   regret)."
4. **Explain the toy world of Theorem 3.** Add two or three sentences: the fresh arms stand for the
   13,330 categories LinUCB keeps trying; the expert seeing the hidden type stands for the LLM reading
   the product text that LinUCB never sees.
5. **Add the Phase B results** (tuned-baseline table, replace the TODO in Section 7.1).
6. **Add the two new figures** (exploration sweep, skew) and one paragraph each in Section 7.
7. **Final pass:** update the abstract, contributions list, conclusion and limitations to match.

---

## 5. If things go badly

| Result | What it means | What we write |
|---|---|---|
| Tuned cold LinUCB gets close to the Router | The big multiplier was partly a weak-baseline effect | Report it honestly. Lead with the mechanism (transfer, lock-in, reward-hidden test), which still holds. Report multipliers against the tuned baseline. |
| Category-term LinUCB matches the Router | The LLM's value is supplying category popularity | "The transfer is the part the linear model misses; the LLM supplies it with a handful of answers; a popularity feature supplies it too." |
| Exploration sweep doesn't fall | Theorem 1's regime isn't reached by our exploration levels | Show it anyway; say the valid radius is far above the levels we can run; keep Theorem 1 as context only. |
| Skew curve is noisy or flat | Skew alone doesn't explain the size of the gain | Report the curve; keep the dominant-category analysis (Tables on settling and dose) as the main evidence. |

---

## 6. Rough schedule

| When | What |
|---|---|
| Oct 4 | BERT Tier 1+3 keeps running. Build experiment D. Start writing changes 2, 3, 4. |
| Oct 5 | BERT Tier 1+3 done → write up the tuned-baseline results. Exploration sweep and skew run. Start writing change 1 (reorder). |
| Oct 6–7 | BGE Tier 1+3 and BERT Tier 2 run. Make the two new figures. Finish the reorder. |
| Oct 8–9 | Tier 2 results in. Decide whether to keep BGE Tier 2 (cut it if not done by Oct 9). |
| Oct 10–11 | Final numbers into the paper; final read-through (abstract, intro, conclusion, limitations). |
| Oct 12 | Submit. |

Time estimates assume the temp-prl jobs finish soon. While they run, everything takes about twice as
long. `status_local.sh` prints a fresh estimate every time you run it.

---

## 7. After everything finishes: analysis checklist

1. Tuned-baseline table: for BERT and BGE, report default cold, tuned cold, default Router, tuned
   Router, the multiplier, the regret ratio, and the paired t-test against cold.
2. Check that the expert ordering (base and large above small) holds with the tuned baseline.
3. Count how often tuned cold LinUCB settles on category 1471 (it was 1 of 20 seeds before).
4. Category-term table: cold and Router with the category term.
5. Exploration figure: multiplier against exploration level, with standard errors over seeds.
6. Skew figure: multiplier against the share of the most common category, plus the WikiLinks point.
7. Update every number in the abstract, intro, Section 7 and the conclusion that changed.
