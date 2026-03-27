# Module 3: The Condition Stack Framework

## Overview

Most people write prompts by dumping facts at a model and hoping for a good answer. That is Layer 5 thinking — starting with the facts. The problem: without Layers 1–4, the model has no idea what jurisdiction you're in, what objective you're optimizing, what constraints apply, or where you are in a process. It fills those gaps with its training prior, which is the statistical average of all cases, not your case.

The Condition Stack is a 6-layer protocol that forces you to specify conditions in the order that most constrains the model's posterior — from the highest-leverage conditions (jurisdiction, time, objective) down to the facts that most prompts start with. The result is not marginal improvement. It is a qualitatively different class of output.

---

## Learning Objectives

By the end of this module, you will be able to:

1. Name and define all 6 layers of the Condition Stack in order from highest to lowest leverage
2. Explain why most prompts fail at Layers 1–4, not at the facts layer
3. Apply the plug-and-play prompt template to any domain and produce a fully-specified prompt
4. Diagnose a weak prompt and identify which layers are missing
5. Demonstrate the output quality difference between a Layer-5-only prompt and a full condition stack
6. Build a Condition Stack builder tool using the Claude API

---

## Module Contents

### Guides

| File | Topic | Time |
|------|-------|------|
| `guides/01_condition_stack_guide.md` | The 6-layer framework, why Layer 5 is the mistake, full plug-and-play template | 20 min |
| `guides/01_condition_stack_slides.md` | Slide deck companion (18–20 slides) with visual stack diagram | Presentation |
| `guides/02_prompt_template_guide.md` | Template deep-dive with 3 fully worked examples: tax, code architecture, medical triage | 20 min |
| `guides/02_prompt_template_slides.md` | Slide deck companion (12–15 slides) | Presentation |

### Notebooks

| File | Topic | Time |
|------|-------|------|
| `notebooks/01_condition_stack_builder.ipynb` | Interactive Claude API tool that walks through all 6 layers, assembles the prompt, and compares raw vs stacked output | 15 min |

### Exercises

| File | Topic |
|------|-------|
| `exercises/01_apply_condition_stack.md` | Apply the condition stack to 5 domains: tax, medical, software architecture, business strategy, code generation |

---

## Prerequisites

- Module 1: The Bayesian Frame (P(A|C) — prompts as evidence)
- Module 2: Switch Variables (conditions that flip solution branches)
- Basic Python for the notebook

---

## Core Idea

$$P(A \mid C_1, C_2, C_3, C_4, C_5, C_6) \ll P(A \mid C_5)$$

The posterior over answers conditioned on all six layers is far more constrained than a posterior conditioned on facts alone. Each layer above Layer 5 is a multiplier on specificity, not an additive increment.

The layers in order of leverage:

| Layer | Name | Leverage Mechanism |
|-------|------|--------------------|
| 1 | Jurisdiction + Rule Set | Selects which legal/regulatory/domain universe applies |
| 2 | Time + Procedural Posture | Pins the temporal context and process stage |
| 3 | Objective Function | Specifies what "good" means (minimize, maximize, certify, speed) |
| 4 | Constraints | Eliminates strategies that are off the table |
| 5 | Facts | The numbers, timeline, and documents |
| 6 | Output Specification | Tells the model what form the answer should take |

---

## Why This Order

The model evaluates your prompt top-to-bottom as a conditioning sequence. A jurisdiction condition eliminates 90%+ of the model's prior distribution immediately — it collapses from "all tax law" to "California LLC tax law." An objective function eliminates another large branch ("minimize audit risk" vs "minimize current-year liability" — these lead to opposite recommendations). By the time the model reaches your facts, it is already reasoning inside a narrow, correct world.

When you start with facts, the model must fill Layers 1–4 from its prior. In high-stakes domains, that prior is built on the median case. Your case is not the median case.

---

## What's Next

Module 4 (Conditional Trees) extends this framework to multi-branch situations: when the right answer depends on a condition you cannot specify in advance, and the model must be prompted to return a decision tree instead of a single verdict.
