# Reproducibility package

Everything needed to reproduce the paper, in one folder: the code, the data,
the figures, and the paper source. **Every file here is a copy.** The working
copies live in `../amazoncat13k/`, `../experiments/`, `../oneshotwikilinks/`
and `../v6.tex`, and nothing was moved or deleted to build this.

```
reproducibility_package/
├── README.md          <- you are here
├── code/              every script, in the order you would run them
│   └── README.md      the detailed guide: flags, seeds, what produces what
├── data/              the condensed datasets (one file per configuration)
│   └── README_DATA.md schema, checksums, how to rebuild from source
├── figures/           the figures in the paper, and the script that makes them
└── paper/             v6.tex and references.bib
```

---

## Start here: a 30-second check

This needs no data and no GPU. It builds a tiny synthetic dataset, runs the
**real** experiment runner on it, and checks the decomposition identity:

```bash
pip install -r code/requirements.txt
bash code/00_smoke_test/run_smoke_test.sh
```

Expected last line: `SMOKE TEST PASSED`. The fixture is built so that a pass
means the mechanism works, not merely that the imports resolve — it checks that
the routed data helps after handoff ($V_\tau > 0$), that the expert's own
answers contribute almost nothing beside it, and that the Router ends above
cold LinUCB.

## Then: the headline number

AmazonCat-13K + BERT-768d + Flan-T5-large, a 3.29x accuracy multiplier over
cold LinUCB. One seed takes roughly 15 minutes on a laptop CPU:

```bash
cd code/03_main_experiment
python experiment_algorithm1.py \
    --slim ../../data/amazon_v1_slim.npz --dataset_name amazonv1 \
    --model_family v1 --llm_type large --seed 1 \
    --llm_confidence 100 --passive_full \
    --alpha_decay 0.9985 --decay_per round --max_samples 50000
```

The run prints a `FINAL` line with the three accuracies and then the
`A / V / B` decomposition. Repeat for seeds 1-10 and average to get the table.
`code/09_cluster_drivers/run_study.sh` does all twelve configurations.

**You do not need a GPU or an LLM.** The LLM predictions are cached inside the
`.npz`. `torch` is optional: without it the runner takes the NumPy/CPU path,
which was verified to give results identical to the GPU path.

---

## What the experiments show

| | |
|---|---|
| **The question** | When does warm-starting a linear bandit with a frozen LLM help, and why? |
| **The headline** | Three LLM experts of near-identical accuracy (5.20%, 3.61%, 5.27%) deliver very different gains: 1.71x, 2.72x, 3.29x. Accuracy does not predict warm-start value. |
| **The mechanism** | Over 99% of the value is not the expert's answers ($A_\tau$ = a few dozen correct answers) but the data those answers leave in the learner's design matrix ($V_\tau$ = thousands). |
| **The explanation** | A warm start acts through two channels: *coverage* (which directions were observed) and the *transfer* (what the rewards on those actions taught beyond the linear model). Under linear rewards the second is provably redundant. Real rewards are not linear, and on this data the transfer carries most of the gain and every difference between experts. |
| **The negative results** | Six of twelve configurations show no gain, and the decomposition names the failing term in each case. |

Two scales are reported throughout, and they are easy to confuse:

```
accuracy multiplier = Router accuracy / max(LLM accuracy, cold accuracy)
regret ratio        = (1 - max(LLM, cold)) / (1 - Router)
```

The accuracy multiplier is what the literature calls "reward multiplication";
the regret ratio is what the theorems bound. At 8% base accuracy, a 3.29x
accuracy multiplier is a regret ratio of 1.27, i.e. 21% less regret. Both are
reported for every row.

## Seeds

One knob, `--seed S`, which fixes the order contexts arrive. That is the
dominant source of variation: cold LinUCB alone ranges over 5.1%-11.1% on
AmazonCat depending on seed. All policies within a run share the seed, so every
comparison is **paired**.

| Block | Role |
|---|---|
| seeds 1-10 | every reported result |
| seeds 11-20 | tuning only: choosing the gate's confidence parameter `c` |

Tuning and evaluation never share a seed. The seed of any run is recoverable
three ways: from the filename (`..._s7.json`), from `config.seed` inside the
JSON, and from `code/runs_manifest.csv`, which lists every run behind the paper
with its full configuration and final accuracies.

`code/README.md` documents the derived generators (offsets +1000, +2000, +3000)
and what the seed does *not* control — the LLM predictions are greedy-decoded
and cached once, so they are identical across seeds.

---

## What is still needed from you

Nothing technical. The package is self-contained once the data files in
`data/` are present. One optional item, for release rather than for running it:

1. **A hosting link, only if the data will not ship with the submission.**
   The four `.npz` files total about 5 GB, which exceeds most submission-system
   supplement limits. If you upload them somewhere (an anonymized archive for
   review, a permanent one afterwards), put the URL in `data/README_DATA.md`,
   where the SHA-256 of each file is already recorded. If instead you hand the
   folder over directly, nothing is needed.

The original sources are recorded in `data/README_DATA.md`:
OneShotWikiLinks from <https://www.kaggle.com/generall/oneshotwikilinks>, and
AmazonCat-13K from the Extreme Classification Repository at
<http://manikvarma.org/downloads/XC/XMLRepository.html>.

The references in `paper/references.bib` are yours to edit; nothing in this
package modifies them.
