# Module 1: The Bayesian Frame — Prompts as Evidence

## Overview

Every time you send a prompt, you are providing evidence to a probabilistic system. The model's response is not a lookup — it is a posterior distribution over possible answers, shaped by your evidence and the model's training priors.

This module establishes the core conceptual frame for the entire course: **P(A|C)** — your answer is always conditional on your conditions. When your conditions are weak, the model's prior (the "average case" from training) dominates. When your conditions are precise, the posterior collapses onto the answer you actually need.

This is not a metaphor. It is the mechanism.

---

## Learning Objectives

By the end of this module, you will be able to:

1. Explain why language model responses are conditional distributions, not fixed outputs
2. Apply the P(A|C) frame to diagnose why a prompt is producing generic answers
3. Distinguish between information that is "detail" and information that is "evidence"
4. Identify the missing conditions in a weak prompt and supply them
5. Predict when adding more words will fail to improve a response

---

## Module Contents

### Guides

| File | Topic | Time |
|------|-------|------|
| `guides/01_prompts_as_evidence_guide.md` | The P(A|C) frame, Bayes' theorem applied to prompting, the posterior shift mechanism | 20 min |
| `guides/01_prompts_as_evidence_slides.md` | Slide deck companion (15–18 slides) | Presentation |
| `guides/02_evidence_vs_information_guide.md` | Why detail is not evidence — constraints, timing, jurisdiction, objective functions | 15 min |
| `guides/02_evidence_vs_information_slides.md` | Slide deck companion (12–15 slides) | Presentation |

### Notebooks

| File | Topic | Time |
|------|-------|------|
| `notebooks/01_posterior_shift_simulator.ipynb` | Live Claude API calls showing how added conditions shift responses; matplotlib visualization | 15 min |
| `notebooks/02_evidence_strength_comparison.ipynb` | 5 real prompts: weak → detailed (still generic) → key condition (precise) | 15 min |

### Exercises

| File | Topic |
|------|-------|
| `exercises/01_rewrite_bad_prompts.md` | Rewrite 5 domain-spanning bad prompts by identifying and supplying missing conditions |

---

## Prerequisites

- Familiarity with sending prompts to a language model
- Basic Python (for notebooks)
- No probability theory required — the Bayesian framing is introduced from scratch

---

## Core Idea

$$P(A \mid C) \propto P(C \mid A) \cdot P(A)$$

- **P(A)** — the model's prior: what training data says is the typical answer to this type of question
- **C** — your conditions: everything in your prompt that makes this case specific
- **P(A|C)** — the posterior: the answer given your specific situation

When C is vague, P(A|C) ≈ P(A). You get the average answer.

When C is specific and discriminating, the posterior collapses onto the answer that fits your actual situation.

**The craft of prompting is the craft of choosing conditions that pin down the right posterior.**

---

## What's Next

Module 2 builds on this frame to examine **prior strength** — why certain domains (legal, medical, financial) have very strong priors that require correspondingly strong evidence to shift, while other domains are more tractable.
