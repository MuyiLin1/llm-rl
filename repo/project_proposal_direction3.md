# Project Proposal: LLM-Guided Adversarial Environment Generation for Agent Testing

## 1. One-Paragraph Summary

We propose a comparative study of LLM-based vs. classical search-based methods for generating adversarial and constructive grid-world environments, given a fixed agent program. The core idea: given a program (DSL solution) that successfully solves a grid-world task (e.g., cleaning all dust in a house), can an LLM intelligently generate new environments that either (a) break the program or (b) are still solvable by the program — and does the LLM do this better, faster, or differently than classical search methods like hill climbing or random perturbation? This sits at the intersection of Unsupervised Environment Design (UED) and LLM reasoning — an intersection that, based on our literature review, is essentially unclaimed.

---

## 2. Why This Matters (Motivation)

### The Problem

When someone builds an agent — whether it's an RL policy, a heuristic program, or an LLM-generated solution — they typically test it on the environment it was designed for. But how do you know it will work on new environments? And how do you find the environments where it _breaks_?

Currently, there are two paradigms for generating test environments:

- **Classical search** (random perturbation, hill climbing, evolutionary algorithms): No semantic understanding. Blind mutations. Can find failures, but inefficiently and without understanding _why_ something broke.
- **Learned adversaries** (PAIRED, regret-guided UED): Train an RL agent to generate hard environments. Requires expensive training. Environment-specific. Not interpretable.

### The Gap

**Nobody has used LLMs for this job.** LLMs can reason about spatial layouts, understand program logic, and hypothesize about failure modes ("this program always turns left at walls, so putting the goal behind a right turn would break it"). That semantic understanding could make adversarial environment generation dramatically more efficient and interpretable. But nobody has tested this hypothesis.

### Who Needs This

- **RL researchers** who want to stress-test their trained policies before deployment
- **Program synthesis researchers** who want to measure generalization of synthesized programs
- **Robotics / embodied AI** where agents must work in varied physical environments
- **Game designers** who want to understand when their AI opponents break
- **Anyone training agents** who wants a cheap, interpretable alternative to UED methods

---

## 3. Related Work (Papers)

### 3a. Unsupervised Environment Design (UED) — The classical approach

These papers define the problem space we're inserting ourselves into. They all generate challenging environments for RL agents, but **none use LLMs**.

| Paper                                                                                | Year | Venue   | Key Idea                                                                                                       | Citations    |
| ------------------------------------------------------------------------------------ | ---- | ------- | -------------------------------------------------------------------------------------------------------------- | ------------ |
| **PAIRED** (Dennis et al.)                                                           | 2020 | NeurIPS | Train an RL adversary to generate environments that maximize regret between a protagonist and antagonist agent | Foundational |
| **Stabilizing UED with a Learned Adversary** (Mediratta et al.)                      | 2023 | CoLLAs  | Improves PAIRED's stability with better adversary training                                                     | 17           |
| **Discovering General RL Algorithms with Adversarial Env Design** (Jackson et al.)   | 2023 | NeurIPS | Uses UED to discover more general RL algorithms                                                                | 25           |
| **Adversarial Environment Design via Regret-Guided Diffusion Models** (Chung et al.) | 2024 | NeurIPS | Replaces RL adversary with a diffusion model for environment generation                                        | 6            |
| **DRED: Zero-Shot Transfer via Data-Regularised Env Design** (Garcin et al.)         | 2024 | arXiv   | Data-regularized alternative to UED for zero-shot transfer                                                     | 18           |
| **Improving Environment Novelty Quantification for UED** (Teoh et al.)               | 2024 | NeurIPS | Better novelty metrics for generated environments                                                              | 5            |
| **Refining Minimax Regret for UED** (Beukman et al.)                                 | 2024 | arXiv   | Theoretical improvements to regret-based UED                                                                   | 20           |
| **An Optimisation Framework for UED** (Monette et al.)                               | 2025 | arXiv   | Unified optimization framework for UED methods                                                                 | 6            |
| **Efficient UED through Hierarchical Policy Representation** (Li et al.)             | 2026 | arXiv   | Hierarchical approach to speed up UED                                                                          | New          |

**Our positioning:** All of these use RL-trained or evolutionary adversaries. We're testing whether an LLM (zero-shot or few-shot, no training) can match or complement these methods.

### 3b. LLM-Based Environment Generation — The cooperative version

These papers use LLMs to generate environments, but for _helping_ agents learn, not for _breaking_ them.

| Paper                                                   | Year | Venue      | Key Idea                                                                                                                                               | Citations |
| ------------------------------------------------------- | ---- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | --------- |
| **EnvGen** (Zala et al.)                                | 2024 | COLM       | LLM generates training environments for RL agents, adapts based on agent's weak skills. Agent trained with LLM-generated envs outperforms GPT-4 agent. | 39        |
| **PCGRLLM** (Baek et al.)                               | 2025 | arXiv      | LLM designs reward functions for procedural content generation RL                                                                                      | 5         |
| **Environment Generation for Autonomous Agents** (Azad) | 2024 | PhD Thesis | Comprehensive thesis on env generation for RL including both classical and LLM methods, uses MiniGrid                                                  | 1         |

**Our positioning:** EnvGen is the closest work. The critical difference is that EnvGen generates _cooperative_ environments (to help the agent learn). We generate _adversarial_ environments (to find where the agent fails). This is a fundamentally different objective with different methods and metrics.

### 3c. LLM-Based Adversarial Testing — In other domains

These papers use LLMs for adversarial testing, but in different domains (not grid-world agent testing).

| Paper                                                                              | Year | Venue           | Key Idea                                                           | Citations |
| ---------------------------------------------------------------------------------- | ---- | --------------- | ------------------------------------------------------------------ | --------- |
| **LLM-Attacker** (Mei et al.)                                                      | 2025 | IEEE T-ITS      | LLMs generate adversarial scenarios for autonomous driving testing | 44        |
| **Adversarial RL for LLM Agent Safety** (Wang et al.)                              | 2025 | arXiv           | RL trains adversary to find safety failures in LLM agents          | 3         |
| **Adversarial Testing in LLMs: Decision-Making Vulnerabilities** (Zhang et al.)    | 2025 | arXiv           | Adversarial interactions to find LLM reasoning failures            | 4         |
| **Adversarial Strategy Generation for RL Poker Agents Using LLMs** (Felter et al.) | 2024 | Stanford CS224R | LLMs generate adversarial strategies against poker RL agents       | —         |

**Our positioning:** LLM-Attacker is closest in spirit but is specific to autonomous driving scenarios. The Stanford poker paper is a class project, not a full publication. Nobody has done this for general grid-world / program synthesis agents.

### 3d. LLM Spatial Reasoning — Context for why our benchmark matters

These establish that LLMs have some (limited) spatial understanding, which is a prerequisite for our approach to work.

| Paper                               | Year | Venue         | Key Idea                                                                                | Citations |
| ----------------------------------- | ---- | ------------- | --------------------------------------------------------------------------------------- | --------- |
| **GRASP** (Tang & Kejriwal)         | 2024 | arXiv         | 16K grid-world environments testing LLM spatial planning                                | 11        |
| **TopoBench** (Maniparambil et al.) | 2026 | ICLR Workshop | Grid puzzles testing topological reasoning; even frontier LLMs solve <25% of hard cases | New       |
| **GSU** (Sidhu & Hockenmaier)       | 2026 | arXiv         | Text-only grid spatial reasoning dataset                                                | New       |
| **Mind's Eye / VoT** (Wu et al.)    | 2024 | NeurIPS       | Visualization-of-thought elicits spatial reasoning in LLMs                              | 128       |

**Our positioning:** These show LLMs can (partially) reason about grids. We leverage that capability for environment generation rather than just evaluation.

---

## 4. What We Would Do (Experiment Plan)

### Phase 1: Setup (Weeks 1-2)

- Clean up existing codebase: ensure the DSL simulator, solution evaluator, and environment perturbation code are modular and reusable
- Select 3-5 DSL programs of varying quality (the best program that solves everything, a mid-tier one, a weak one)
- Define the evaluation metrics (see Section 5)
- Build a unified benchmark runner that takes (program, environment) → reward

### Phase 2: Implement Generation Methods (Weeks 2-4)

We compare **5 methods** for generating environments given a fixed program:

| Method                  | Description                                                                    | What we already have                            |
| ----------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------- |
| **Random Perturbation** | Randomly move walls, add/remove dust markers                                   | Taro built this                                 |
| **Hill Climbing**       | Accept perturbation only if it improves the objective (break or keep solvable) | Taro built this                                 |
| **LLM One-Shot**        | Give LLM the map + program + task, ask it to generate a new environment        | Muyi built this                                 |
| **LLM Iterative**       | LLM generates, we run simulation, feed execution trace back, LLM tries again   | Muyi built the loop; needs trace feedback added |
| **Hybrid**              | LLM proposes candidate perturbation, hill climbing refines it                  | New; combines existing pieces                   |

### Phase 3: Run Experiments (Weeks 4-6)

**Experiment A — Adversarial Generation (Breaking the Agent)**

- Objective: generate environments where the program gets the _lowest_ reward
- For each of the 5 methods, run N trials (e.g., 100)
- Measure: reward of the program on the generated environment, number of attempts needed, diversity of failure modes found

**Experiment B — Constructive Generation (Maintaining Solvability)**

- Objective: generate environments where the program _still succeeds_ (reward ≥ 0.8)
- Same 5 methods, N trials
- Measure: success rate, diversity of valid environments, number of attempts

**Experiment C — Scaling Analysis**

- Fix a compute budget (e.g., 100 simulation steps OR 20 LLM calls)
- How does each method's performance change as you increase the budget?
- Does the LLM get better with more iterations? Does hill climbing plateau?

**Experiment D (Stretch) — Interpretability**

- For LLM methods: ask the LLM to _explain_ why its generated environment should break the program
- Categorize failure modes: "added wall blocking critical path," "moved dust to unreachable area," etc.
- Compare: can a human understand _why_ the LLM-generated environment is hard? Compare to hill climbing (no explanation)

### Phase 4: Analysis & Writing (Weeks 6-8)

- Compile results into comparison tables and plots
- Write the paper around the findings (see Section 7 for paper structure)

---

## 5. Metrics

| Metric                             | What it measures                                                                | How to compute                                                        |
| ---------------------------------- | ------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| **Reward**                         | How well the program performs on the generated env                              | Run simulator, get fraction of dust collected                         |
| **Success Rate**                   | % of generated environments that are valid (structurally sound, not degenerate) | Automated check: is the env well-formed?                              |
| **Sample Efficiency**              | How many attempts to find a good adversarial/constructive env                   | Count iterations                                                      |
| **Diversity**                      | How different are the generated environments from each other                    | Edit distance between grids, or count unique wall/dust configurations |
| **Cost**                           | Wall-clock time and API dollars                                                 | Time the runs; count LLM API tokens                                   |
| **Interpretability** (qualitative) | Can we understand _why_ the generated env is challenging                        | Human evaluation of LLM explanations                                  |

---

## 6. What This Would Be Used For (Applications / Impact)

1. **Robustness Testing for Any Grid-World Agent**: Given a trained/synthesized agent, automatically find environments where it fails — cheaper and more interpretable than UED
2. **Benchmark Generation at Scale**: Auto-generate diverse test suites for program synthesis methods, replacing hand-crafted benchmarks
3. **Understanding Agent Generalization**: The environments where an agent fails reveal _what it actually learned_ vs. what it memorized
4. **Practical LLM Capability Assessment**: Our results double as data on how well LLMs understand spatial layouts and program behavior — contributing to the spatial reasoning literature
5. **Curriculum Generation**: If the LLM can generate environments of calibrated difficulty, this enables curriculum learning for training agents

---

## 7. Potential Paper Structure

**Title Ideas:**

- "Can LLMs Break Your Agent? A Comparative Study of Adversarial Environment Generation"
- "LLM-Guided Adversarial Environment Design: When Language Models Meet Grid Worlds"
- "Beyond Blind Search: LLMs as Adversarial Environment Generators for Agent Testing"

**Sections:**

1. Introduction (the gap between UED and LLMs)
2. Related Work (UED, EnvGen, LLM spatial reasoning)
3. Problem Formulation (adversarial vs. constructive env generation)
4. Methods (5 generation approaches)
5. Experimental Setup (grid world, DSL programs, metrics)
6. Results (comparison tables, scaling curves, diversity analysis)
7. Analysis (when does LLM help? when does search win? failure mode categorization)
8. Discussion & Limitations
9. Conclusion

---

## 8. Potential Issues & Risks

### Technical Risks

| Risk                                                                                                       | Severity | Mitigation                                                                                                                                                                                                                          |
| ---------------------------------------------------------------------------------------------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **LLM is terrible at spatial reasoning and generates garbage environments**                                | High     | This is actually still a publishable result ("We show that LLMs cannot effectively generate adversarial environments, revealing limitations in spatial reasoning"). Negative results are valid here because nobody has tested this. |
| **Hill climbing dominates LLM on every metric**                                                            | Medium   | Focus the narrative on _different types_ of failures found. Even if LLM is slower, it may find qualitatively different failure modes that search misses. Also test the hybrid approach — maybe LLM + search > either alone.         |
| **The DSL grid world is too simple**                                                                       | Medium   | Can extend to Sokoban (push-box puzzles) which has open-source simulators and solvers. Or use multiple Karel-type tasks beyond just clean-house.                                                                                    |
| **LLM API costs are high**                                                                                 | Low      | Use smaller/cheaper models (GPT-4o-mini, Claude Haiku, Llama local). Part of the study can compare model size vs. generation quality.                                                                                               |
| **Environment validity**: LLM generates structurally invalid grids (wrong dimensions, missing walls, etc.) | Medium   | Add a validation layer that rejects malformed environments before simulation. Report rejection rate as a metric.                                                                                                                    |
| **Seed/map dependency**: Results only hold for one specific program/map                                    | Medium   | Test across 3-5 programs of varying quality and multiple base maps. Report variance.                                                                                                                                                |

### Scope Risks

| Risk                                                     | Severity | Mitigation                                                                                                                                                                  |
| -------------------------------------------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Too ambitious for summer deadline**                    | Medium   | Minimum viable paper = just Experiments A and B with 5 methods on clean-house task. Experiments C and D are stretch goals.                                                  |
| **Overlapping with EnvGen**                              | Low      | EnvGen is cooperative (help the agent). We are adversarial (break the agent) AND comparative (LLM vs. search). Clear differentiation.                                       |
| **Reviewer says "just a prompt engineering paper"**      | Medium   | Emphasize the comparative study angle. We're not claiming LLMs are the best — we're providing the first systematic comparison. Include the classical baselines prominently. |
| **Reviewer says "too narrow domain (only grid worlds)"** | Medium   | Argue that grid worlds are the standard testbed for UED (PAIRED, DRED, etc. all use MiniGrid). If time allows, add one more domain.                                         |

---

## 9. Task Breakdown (Who Does What)

| Task                                                            | Estimated Effort | Dependencies                       |
| --------------------------------------------------------------- | ---------------- | ---------------------------------- |
| Modularize existing codebase (simulator, evaluator)             | 3-5 days         | None                               |
| Implement random perturbation baseline                          | 1 day            | Already done (Taro)                |
| Implement hill climbing (adversarial + constructive objectives) | 2-3 days         | Mostly done (Taro)                 |
| Implement LLM one-shot generation                               | 1-2 days         | Mostly done (Muyi)                 |
| Add execution trace feedback to LLM iterative loop              | 3-4 days         | LLM one-shot                       |
| Implement hybrid method (LLM + hill climbing)                   | 2-3 days         | Both LLM and hill climbing working |
| Run Experiment A (adversarial)                                  | 2-3 days compute | All methods implemented            |
| Run Experiment B (constructive)                                 | 2-3 days compute | All methods implemented            |
| Run Experiment C (scaling)                                      | 2-3 days compute | Exps A & B done                    |
| Diversity / failure mode analysis                               | 3-4 days         | Exps A & B done                    |
| Write paper                                                     | 2 weeks          | All experiments done               |

**Total estimated timeline: 6-8 weeks** from start to submission-ready draft.

---

## 10. Key Differentiators (Why This Gets Accepted)

1. **Novel intersection**: First paper to use LLMs for adversarial environment generation (UED has only used RL/evolutionary methods)
2. **Fair comparison**: We don't just show LLMs — we systematically compare 5 methods head-to-head
3. **Both directions**: Adversarial AND constructive generation gives a complete picture
4. **Low compute requirement**: No RL training needed, just simulation + LLM API calls
5. **Interpretability angle**: LLM can explain _why_ it generated that environment — classical methods can't
6. **Negative results are valuable**: If LLMs can't do this, that's a meaningful finding for both the UED and LLM communities
