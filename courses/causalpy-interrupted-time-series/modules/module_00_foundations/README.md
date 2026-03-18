# Module 00: Foundations and Setup

## Module Overview

This module establishes the conceptual foundations for causal inference and sets up your computational environment. You will shift from correlational thinking to causal thinking, learn the formal language of potential outcomes, and gain practical skills in drawing and interpreting DAGs.

## Learning Objectives

By the end of this module, you will be able to:

1. Distinguish causal from predictive questions and explain why the distinction matters for decision-making
2. Apply the potential outcomes framework (Rubin Causal Model) to define treatment effects (ATE, ATT)
3. Recognize and articulate the fundamental problem of causal inference
4. Draw Directed Acyclic Graphs (DAGs) for realistic causal scenarios
5. Identify confounders, mediators, and colliders in a DAG
6. Apply the backdoor criterion to determine a valid adjustment set
7. Set up CausalPy, PyMC, and ArviZ and run a minimal ITS analysis

## Module Structure

```
module_00_foundations/
├── guides/
│   ├── 01_causal_vs_predictive_guide.md      # Causal vs predictive thinking
│   ├── 01_causal_vs_predictive_slides.md     # Companion slide deck (17 slides)
│   ├── 02_potential_outcomes_guide.md         # Rubin causal model + notation
│   ├── 02_potential_outcomes_slides.md        # Companion slide deck (17 slides)
│   ├── 03_dags_guide.md                       # DAGs, confounders, colliders, mediators
│   └── 03_dags_slides.md                      # Companion slide deck (17 slides)
├── notebooks/
│   ├── 01_environment_setup.ipynb             # Install and verify CausalPy stack
│   └── 02_first_causal_analysis.ipynb         # First ITS analysis end-to-end
├── exercises/
│   ├── 01_causal_thinking_exercises.py        # Self-check: causal vs correlational
│   └── 02_dag_exercises.py                    # Self-check: DAG reasoning
└── resources/                                  # Additional readings
```

## Recommended Sequence

1. Read **Guide 1** (causal vs predictive) — 20 min
2. Read **Guide 2** (potential outcomes) — 25 min
3. Read **Guide 3** (DAGs) — 25 min
4. Run **Notebook 1** (environment setup) — 10 min
5. Run **Notebook 2** (first ITS analysis) — 15 min
6. Work through **Exercise 1** (causal thinking) — 15 min
7. Work through **Exercise 2** (DAG reasoning) — 15 min

**Total estimated time: ~2 hours**

## Key Concepts

| Concept | Definition | Why It Matters |
|---------|-----------|----------------|
| Causal vs predictive | Different questions requiring different methods | Misapplying prediction to causal questions leads to wrong decisions |
| Potential outcomes | $Y(1)$ and $Y(0)$ — what would happen under each treatment | Makes the counterfactual problem formal and rigorous |
| ATE vs ATT | Population average vs effect on treated units | ITS estimates ATT — the effect on the units that actually received treatment |
| Confounder | Common cause of treatment and outcome | Creates spurious correlation; must be controlled |
| Mediator | Causal pathway variable | Should NOT be controlled (blocks the effect) |
| Collider | Caused by both treatment and outcome | Conditioning on colliders CREATES bias |
| Backdoor criterion | Identifies valid adjustment sets | Tells you exactly which variables to control for |

## Prerequisites

- Basic statistics (mean, variance, regression)
- Python 3.10+ with pip or conda
- Familiarity with pandas and matplotlib

## What's Next

**Module 01: ITS Fundamentals** — With the causal foundations in place, Module 01 introduces Interrupted Time Series formally: when to use it, what it assumes, how to specify the segmented regression model, and how to interpret the results. You will work with a real public-health dataset.
