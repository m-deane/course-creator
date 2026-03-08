# Module 01: GRPO Algorithm — Group Relative Policy Optimization

## Overview

Group Relative Policy Optimization (GRPO) is the training algorithm behind DeepSeek-R1, one of the most capable open reasoning models available. It achieves what larger models with expensive value networks cannot: stable, efficient policy learning using only a reward function and group-relative scoring.

This module gives you a complete understanding of GRPO — from intuition through mathematics to implementation. By the end you will be able to implement the core algorithm from scratch and recognize exactly what makes it different from PPO, DPO, and vanilla REINFORCE.

Module 00 introduced policy gradients and relative advantage. This module shows exactly how GRPO operationalizes those ideas into a practical training algorithm.

## Learning Objectives

After completing this module, you will be able to:

1. Explain the group sampling strategy at GRPO's core and why relative ranking eliminates the need for a critic network
2. Trace a single GRPO update step: sample G completions, score each, normalize to advantages, apply clipped surrogate loss
3. Derive the advantage formula $A_i = (r_i - \mu) / \sigma$ and explain what each term controls
4. Identify the role of the KL divergence penalty and explain what happens when it is removed or weighted too high
5. Compare GRPO against PPO, DPO, and REINFORCE on compute cost, stability, and data requirements
6. Implement advantage calculation and the clipped surrogate loss in NumPy without reference code

## Prerequisites

- Module 00 complete (policy gradients, advantage estimation)
- NumPy (array operations, broadcasting)
- Conceptual familiarity with language model log-probabilities

## Guides

| File | Topic | Time |
|------|-------|------|
| `guides/01_grpo_intuition_guide.md` | Core idea, worked example, GRPO flow diagram | 20 min |
| `guides/02_grpo_math_guide.md` | Objective function, advantage formula, clipping, KL penalty | 25 min |
| `guides/03_grpo_vs_alternatives_guide.md` | GRPO vs PPO, DPO, REINFORCE — when to use each | 20 min |

Each guide has a companion `_slides.md` for lecture delivery.

## Exercises

| File | Topic |
|------|-------|
| `exercises/01_grpo_from_scratch_exercise.py` | Implement advantage calculation, clipped loss, mini GRPO update |

## Module Map

```
Module 01: GRPO Algorithm
    │
    ├── Guide 01: Intuition
    │       Group sampling → relative scoring → diagram → worked example
    │
    ├── Guide 02: Mathematics
    │       Objective function → advantage formula → clipping → KL penalty
    │       └── Python implementation of each component
    │
    └── Guide 03: Comparisons
            GRPO vs PPO → GRPO vs DPO → GRPO vs REINFORCE
            └── Decision guide: when to use each
```

## Connections

- **Builds on:** Module 00 — Foundations (policy gradients, reward signals, advantage)
- **Leads to:** Module 02 — ART Framework (uses GRPO via Unsloth in a production training system)
- **Algorithm origin:** DeepSeekMath (2024), deployed in DeepSeek-R1 (2025)

## Time Estimate

- Guides (reading): 65 minutes total
- Slides (lecture): 50 minutes total
- Exercise: 45 minutes
- Total: approximately 2.5 hours
