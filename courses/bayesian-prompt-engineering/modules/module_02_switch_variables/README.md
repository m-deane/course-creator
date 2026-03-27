# Module 2: Switch Variables — The Conditions That Actually Matter

> *The model already knows the answer. The question is which answer — and that depends entirely on which world you've placed it in.*

## Overview

Most prompts fail not because they lack detail, but because they lack the *right* detail. This module introduces **switch variables** — the small set of conditions that bifurcate the solution space. Miss one, and the model answers a different question than the one you have.

You will build a systematic way to identify, rank, and inject these high-leverage conditions before you write a single word of prompt.

## Learning Objectives

By the end of this module you will be able to:

1. Define switch variables and distinguish them from low-value descriptive detail
2. Identify the top 3-5 switch variables for any professional domain
3. Rank conditions by information gain — how much they narrow the answer space
4. Use a Claude-powered tool to surface switch variables from a domain description
5. Measure the response impact of adding or removing individual switch variables

## Contents

| File | What It Covers |
|------|----------------|
| `guides/01_switch_variables_guide.md` | Definition, categories, and domain-specific catalogs of switch variables |
| `guides/01_switch_variables_slides.md` | Slide deck companion (15 slides) with mermaid decision trees |
| `guides/02_information_gain_guide.md` | Why information gain — not volume — determines prompt quality |
| `guides/02_information_gain_slides.md` | Slide deck companion (13 slides) with entropy diagrams |
| `notebooks/01_switch_variable_identifier.ipynb` | Build a Claude-powered switch variable identifier (~15 min) |
| `exercises/01_rank_switch_variables.md` | Rank switch variables for 10 real-world prompts |

## Prerequisites

- Module 1: The Bayesian Frame (prompts as evidence, posterior narrowing)
- Basic familiarity with at least one professional domain (law, medicine, engineering, finance, or software)

## Time Estimate

| Activity | Time |
|----------|------|
| Guide 1: Switch Variables | 20 min |
| Guide 2: Information Gain | 15 min |
| Notebook 1: Identifier Tool | 15 min |
| Exercise 1: Ranking Practice | 20 min |
| **Total** | **~70 min** |

## Core Concept

A **switch variable** is a condition whose presence or absence routes the reasoning to a categorically different solution branch.

```
Without switch variables:
  Question → Model guesses context → Probably wrong world → Generic answer

With switch variables:
  Question + [jurisdiction] + [timing] + [status] → Exact world → Precise answer
```

The key skill is asking: *"What are the five conditions that, if different, would change my answer entirely?"*

## Setup

```bash
pip install anthropic matplotlib numpy
export ANTHROPIC_API_KEY="your-key-here"
```

## Next

Module 3: The Condition Stack Framework — building the full 6-layer condition specification for any prompt.
