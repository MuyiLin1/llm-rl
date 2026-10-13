# Project Proposal: LLM-Guided Adversarial Environment Generation for Agent Testing
## Easy-Read Version (Paragraph Format)

---

### What We're Proposing

We want to study whether LLMs can intelligently generate grid-world environments that either break a given agent program or remain solvable by it, and compare that against classical methods like hill climbing and random perturbation. Right now we have a DSL program (a set of robot instructions written in a simple mini-language) that can successfully solve a "clean house" task — it navigates a grid, avoids walls, and picks up all the dust. Taro already built a hill-climbing perturbation pipeline that randomly moves walls and dust markers to create new environments, and Muyi built an LLM-based loop that prompts the model to generate new maps. The question is: which approach is better at generating useful new environments, and can combining them beat either one alone? Based on our literature search, nobody has done this comparison. This sits at an unclaimed intersection between two active research areas — Unsupervised Environment Design (UED) and LLM reasoning.

---

### Why This Matters and Who Needs It

When someone builds an agent — whether it's an RL policy, a heuristic program, or an LLM-generated solution — they typically only test it on the environment it was designed for. But that tells you nothing about where it breaks. The existing approach to finding hard environments is called Unsupervised Environment Design (UED), pioneered by a method called PAIRED. In UED, you train a separate RL agent whose sole job is to generate challenging environments. This is expensive, requires its own training loop, and gives you no insight into why the generated environment is hard. On the other side, LLMs can reason about spatial layouts and program behavior — they could potentially say "this program always turns left at walls, so putting the goal behind a right turn would break it." But nobody has actually tested whether LLMs can do this adversarial generation job. Our study would be the first to systematically compare LLM-based vs. search-based adversarial environment generation. The results matter for anyone who trains or deploys agents: RL researchers stress-testing policies, program synthesis researchers measuring generalization, robotics teams validating behavior in varied environments, and game designers testing AI opponents.

---

### What Already Exists (Related Work)

The most relevant prior work falls into four buckets. First, the UED literature (PAIRED, regret-guided diffusion, DRED, etc.) — about a dozen papers published at NeurIPS, ICLR, and elsewhere since 2020. All of these generate challenging environments using RL-trained or evolutionary adversaries. None use LLMs. Second, EnvGen (Zala et al., COLM 2024, 39 citations) is the closest paper to what we're doing — it uses an LLM to generate training environments for RL agents. But EnvGen is cooperative: the LLM generates environments to help the agent learn, not to break it. Our approach is adversarial, which is a fundamentally different objective. Third, LLM-Attacker (Mei et al., 2025, 44 citations) uses LLMs for adversarial scenario generation, but only for autonomous driving — not general grid-world agents. A Stanford class project did something similar for poker, but it was never published. Fourth, there's a large body of LLM spatial reasoning benchmarks (GRASP, TopoBench, GSU, PLUGH, Mind's Eye) that establish LLMs have some ability to reason about grids — which is a prerequisite for our approach to work. Our paper sits at the intersection of buckets one and two: we take the adversarial objective from UED and ask whether an LLM can do the job that previously required a trained RL adversary.

---

### What We Would Actually Do

We would compare five methods for generating environments given a fixed program. (1) Random perturbation: randomly move walls and dust markers. Taro already built this. (2) Hill climbing: accept a random perturbation only if it moves the reward in the desired direction. Taro already built this. (3) LLM one-shot: give the LLM the map, the program, and the task, and ask it to generate a new environment in one call. Muyi already built this. (4) LLM iterative: the LLM generates an environment, we run the simulation, capture the execution trace (where the robot went, where it crashed, where it looped), feed that trace back to the LLM, and ask it to try again. This extends what Muyi built by adding execution trace feedback, which addresses the exact problem Taro identified in our first meeting — the LLM doesn't know what went wrong. (5) Hybrid: the LLM proposes a high-level perturbation idea ("move the wall at row 3"), and hill climbing executes and refines it.

We run two main experiments. Experiment A is adversarial: for each method, try to generate environments that minimize the program's reward (break it). Experiment B is constructive: try to generate environments where the program still succeeds (reward above 0.8). For both, we measure success rate (how often does the method produce a valid, useful environment), sample efficiency (how many attempts it takes), diversity (how structurally different the generated environments are from each other), cost (LLM API calls vs. simulation steps), and — for the LLM methods — interpretability (can we understand why the generated environment is challenging). A stretch experiment would measure scaling: given a fixed compute budget, how does each method's performance improve as you give it more resources?

---

### Timeline and Effort

The whole thing should take about 6-8 weeks. Weeks 1-2 are cleanup: modularize the existing codebase so the simulator, evaluator, and perturbation methods all work as plug-and-play components. Select 3-5 DSL programs of varying quality to test against. Weeks 2-4 are implementation: most of the methods already exist in some form from Taro and Muyi's work. The main new pieces are adding execution trace feedback to the LLM loop and building the hybrid method. Weeks 4-6 are experiments: running the five methods across both adversarial and constructive objectives with enough trials to get meaningful statistics. Weeks 6-8 are analysis and writing. The minimum viable paper is just Experiments A and B on the clean-house task — that's already novel and publishable. The scaling analysis and interpretability study are stretch goals.

---

### What Could Go Wrong

The biggest risk is that the LLM is just terrible at spatial reasoning and generates garbage environments that are structurally invalid or no better than random. Here's the thing though: that's still a publishable result. If we show that LLMs cannot effectively generate adversarial environments despite being able to reason about language and code, that's a meaningful negative finding for both the UED and LLM communities. Nobody has tested this, so the result is new either way.

A second risk is that hill climbing simply dominates the LLM on every metric — it's faster, cheaper, and more effective. Even in this case, we can focus the analysis on qualitative differences: does the LLM find different types of failures than search? Does it produce more diverse environments? Does the hybrid method outperform either alone? And practically, the interpretability angle still holds — hill climbing can tell you the environment is hard but not why. The LLM can potentially explain the failure mode.

A third risk is that our grid world is too simple and reviewers say "this only works on toy domains." We mitigate this by noting that grid worlds are the standard testbed for the entire UED literature (PAIRED, DRED, etc. all use MiniGrid). If time allows, we can add a second domain like Sokoban, which has open-source simulators and solvers.

Other practical risks: LLM API costs (mitigated by using cheaper models like GPT-4o-mini or running open-weight models locally on Taro's machine), LLM generating structurally invalid grids (mitigated by adding a validation layer and reporting the rejection rate as a metric), and results being specific to one program (mitigated by testing across 3-5 programs of varying quality). A reviewer might also call this "just a prompt engineering paper," which we counter by emphasizing the systematic comparison — we're not claiming LLMs are the best, we're providing the first fair head-to-head study.

---

### Why This Gets Accepted

The paper has a clear novel contribution: it's the first to use LLMs for adversarial environment generation, filling a gap between UED (which has only used RL/evolutionary methods) and LLM-based environment generation (which has only been cooperative via EnvGen). We provide a fair comparison of five methods rather than just advocating for one. The compute requirements are low — no RL training, just simulation plus LLM API calls — so it's feasible for our team. The interpretability angle (LLMs can explain why an environment is hard) is a unique advantage that classical methods fundamentally cannot provide. And the results are informative regardless of outcome: whether LLMs are great, mediocre, or terrible at this task, the finding is new and useful to multiple communities.
