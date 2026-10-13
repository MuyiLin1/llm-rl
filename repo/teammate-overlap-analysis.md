# Request: feature-overlap measurements on AmazonCat (BERT-768d)

Hi! We are revising the theory section of our ALT paper and need a few numbers from the experiment code. Nothing here changes the paper's existing results. Part A needs no new bandit runs. Part B needs one extra hook in the existing runs (save the learner state at one round). Part C is optional and is one small rerun.

## Why we need this (short version)

Our settling theorem (Theorem 2) says: once LinUCB has seen enough rewarded plays of one action, it keeps playing that action forever, with high probability. We just generalized it so competitors' features may **overlap** the settling action's feature. It still needs the settling action to have the **same feature every round**.

On AmazonCat the feature is φ(x, a) = x ⊙ z_a (elementwise product of the BERT context embedding x and the category vector z_a). So the dominant category **1471** has a different feature on every context. We want to know how far the real data is from the theorem's assumptions:

1. **How much does 1471's feature change from context to context?** If it barely changes, the theorem is "nearly" applicable.
2. **How much does 1471's feature overlap with the other categories' features?** The theorem allows overlap, but only when a quantity we call ρ stays below 1.
3. **At the moment the Router stops routing, are the theorem's quantities (N, ρ, ω, h) actually in the "settles" range for the runs that settled?**

## Before you start: find these in the code

Please tell us the file and function names for each of these when you reply (item 0 in "What to send back").

| What | What to search for | What we need to know |
|---|---|---|
| Context embedding x_t | BERT encoder call, `768`, `embed` | Is x_t L2-normalized? Is there a bias or constant coordinate? |
| Category vectors z_a | `z_a`, `label_emb`, `category`, `13330` | How z_a is built (for example, the BERT embedding of the category name). Is it normalized? |
| Joint feature φ(x, a) | `*` or `np.multiply` between context and label embeddings, `odot` | Confirm φ(x, a) = x ⊙ z_a for the BERT configuration (no extra terms) |
| LinUCB state | `A`, `V`, `A_inv`, `b`, `theta` | Where V (Gram matrix, λI + Σ φφᵀ) and θ̂ = V⁻¹b live, and whether V⁻¹ is kept directly |
| Exploration scale | `0.9985`, `alpha` | Confirm α_t = max(0.1, 0.9985^t) and λ = 0.1 |
| Router decision | `c_llm`, `route`, width comparison | Where it decides to route a round to the LLM |
| "Settle" flag | the code behind "settle" in Table 3 | The exact rule used to say a run "settles on 1471" |
| Oracle θ_or | `theta_or`, `oracle` | The vector used in the "transfer removed" curve of Figure 1 |

Use the **default BERT configuration of Table 3**: T = 50,000, λ = 0.1, α_t = max(0.1, 0.9985^t), c_LLM = 100, seeds 1–10, experts Flan-T5-small, -base and -large.

## Part A: static feature overlap (no bandit runs needed)

Use the context stream of **seed 1** (the first 50,000 contexts). Take a random sample of **5,000 contexts** (fix the sampling seed, for example 0). For each sampled context x_t, let u_t = φ(x_t, 1471).

### A1. How much 1471's feature changes across contexts

- Let ū = (mean of u_t / ‖u_t‖), then normalize ū to unit length.
- Report the distribution of **cos(u_t, ū)**: mean, standard deviation, 5th percentile, minimum.
- Report the distribution of **‖u_t‖**: mean, standard deviation, min, max.
- Report the same two lines **only over contexts where 1471 is a correct label** (the 29.8%) and **only over contexts where it is not**.

### A2. How much 1471's feature overlaps with other categories

For each sampled context, compute cos(u_t, φ(x_t, b)) = ⟨φ(x_t,1471), φ(x_t,b)⟩ / (‖φ(x_t,1471)‖ ‖φ(x_t,b)‖) for these groups of categories b:

| Group | Which categories |
|---|---|
| G1 | The other three frequent categories: 7961, 7892, 7891 |
| G2 | The 20 categories that **cold LinUCB plays most often**, pooled over seeds 1–10 (excluding 1471). Please list them with their play counts. |
| G3 | The category cold LinUCB settles on in each seed where it settles on something else (the "11% category"). |
| G4 | 1,000 categories drawn uniformly at random (fixed seed). |

For each group, report mean, median, 95th percentile and maximum of the cosine (pooled over contexts and categories in the group). For G1 and G3, report each category separately.

### A3. Is the overlap "structural"?

Because φ(x, a) = x ⊙ z_a, the overlap between two categories on the same context is Σ_i x_i² z_{1471,i} z_{b,i}. Please also report cos(z_1471, z_b) (the plain category-vector cosine) for the G1 and G3 categories. This tells us whether the overlap comes from the category vectors themselves.

## Part B: the theorem's quantities at the handoff (needs a small hook)

**Handoff round τ** = the **last round at which the Router routes to the LLM** in that run. In the existing Router runs (seeds 1–10, each of the three Flan-T5 experts, default configuration), save at round τ + 1:

- V (768 × 768) or V⁻¹, and θ̂ (768)
- τ itself, and α_τ (the exploration scale at τ)
- whether the run settles on 1471 (the existing flag)
- the number of routed rounds with a correct answer on 1471, before τ

That is 30 saved states. Save them as `.npz` files; they are about 5 MB each.

Then, for each saved state, compute the following. Use β = 0.1 (the floor of α_t) for the radius. Also report the τ, so we can check whether α_τ is already at the floor.

**Step 1. Fixed-direction version.** Take u = ū from A1 (unit length).
- N = 1 / (uᵀ V⁻¹ u)
- μ̂ = ⟨θ̂, u⟩

**Step 2. Competitors.** Draw **200 contexts after τ** from the same run's stream (fixed seed). For each such context x and **every** category b ≠ 1471, compute, with v = φ(x, b):
- ρ = ⟨v, V⁻¹u⟩ / (uᵀ V⁻¹ u)
- I = ⟨θ̂, v⟩ + β · sqrt(vᵀ V⁻¹ v)
- the per-competitor threshold w = (I − ρ μ̂) / (1 − ρ), only when ρ < 1

A fast way to get all categories at once: compute a Cholesky factor of V⁻¹ (or solve with the Cholesky factor of V) once per state. Then for one context, build the 13,330 × 768 matrix Φ = x ⊙ Z. The ρ's are Φ (V⁻¹u)/(uᵀV⁻¹u). The norms are the row norms of Φ L, where V⁻¹ = L Lᵀ. If this is too slow, use the 500 categories with the highest index I in that context, and tell us that you did.

**Step 3. Report per run** (one row per seed × expert):

| column | meaning |
|---|---|
| seed, expert | |
| τ, α_τ | handoff round and its exploration scale |
| n_1471_correct_routed | correct routed answers on 1471 before τ |
| settled | yes/no (existing flag) |
| N | from step 1 |
| μ̂ | from step 1 |
| ρ_max, ρ_99 | largest and 99th-percentile ρ over all (context, b) pairs |
| frac_ρ≥1 | fraction of pairs with ρ ≥ 1 |
| ω | the largest w over all pairs with ρ < 1 |
| ω_99 | 99th percentile of w (in case ω is driven by a few outliers) |
| h = N(μ̂ − ω) | the margin |
| bound = 1 − exp(−2(0.298 − ω)·h / 0.25) | the theorem's settling probability, using 0.298 for μ_a and σ = 1/2; write "n/a" if h ≤ 0 or ω ≥ 0.298 |

**Step 4. Per-context version** (this checks whether using one fixed direction ū is fair). For the same 200 contexts, also compute μ̂_t = ⟨θ̂, φ(x_t, 1471)⟩ and the width β·‖φ(x_t,1471)‖_{V⁻¹}. Report, per run, the mean, standard deviation and 5th percentile of μ̂_t, and the fraction of the 200 contexts on which **1471 has the highest index of all categories**.

## Part C (optional): one rerun of the injection test

In the "+75 on 1471, reward hidden" condition (Table `tab:inject50k`, 50,000-round stream, seeds 1–10, Flan-T5-small), the learner currently gets **0** as the reward on the changed contexts. Please add one condition, **"+75 on 1471, oracle reward"**, that feeds ⟨θ_or, φ(x_t, a_t)⟩ instead of 0. Use the same θ_or as the "transfer removed" curve of Figure 1. Everything else stays the same.

Report for the new condition: the accuracy multiplier (mean ± sd), the paired p-value against "+75 on 1471" (with rewards) and against the baseline, and the number of runs that settle on 1471. Please use the same test as the existing table.

## What to send back

Please send these as one zip, or one folder plus a short message:

0. **Code map:** file and function names for each row of the "find these" table, plus answers to its questions (normalization, bias term, how z_a is built, the settle rule).
1. **Part A tables:** A1 (three lines: all, 1471 correct, 1471 not correct), A2 (four groups, plus the G2 category list with counts), A3 (category cosines).
2. **Part B table:** 30 rows as above, as CSV, plus the step 4 summary per run.
3. **Part C row** (if you do it), in the same format as `tab:inject50k`.
4. **The script(s)** you used, the git commit hash of the experiment code, and the random seeds used for sampling.
5. **Anything surprising**, for example ρ often ≥ 1, ω above 0.298, or settled runs with h ≤ 0. Surprises are useful, so please don't smooth them over.

Rough cost: Part A runs in minutes. Part B needs the 30 existing runs again, with one extra save per run, plus a short analysis. Part C is 10 runs.

Thanks!
