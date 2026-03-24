# Module 6: Common Probability Mistakes and Anti-Patterns

## Overview

Bad AI answers are not random. They follow predictable patterns — each one traceable to a specific probability mistake in the prompt. This module catalogs the six most common mistakes, explains exactly why they produce bad outputs through the Bayesian lens, and provides the fix for each.

The goal is not just recognition but systematic diagnosis. By the end of this module, when Claude gives you a wrong, vague, or inconsistent answer, you will be able to identify which mistake produced it and apply the precise correction — every time.

---

## Learning Objectives

By the end of this module, you will be able to:

1. Name and explain the six common probability mistakes in prompt engineering
2. Diagnose which mistake caused a bad AI response using a structured flowchart
3. Apply the Bayesian correction for each mistake type
4. Write a "mistake taxonomy" for your own domain — mapping each error type to concrete examples
5. Use the diagnostic framework as a systematic debugging workflow

---

## Module Contents

### Guides

| File | Topic | Time |
|------|-------|------|
| `guides/01_six_mistakes_guide.md` | The 6 probability mistakes: what they are, why they happen, and how to fix them | 25 min |
| `guides/01_six_mistakes_slides.md` | Slide deck companion (18–20 slides) | Presentation |
| `guides/02_diagnostic_framework_guide.md` | Systematic flowchart for diagnosing bad AI answers | 15 min |
| `guides/02_diagnostic_framework_slides.md` | Slide deck companion (12–15 slides) | Presentation |

### Notebooks

| File | Topic | Time |
|------|-------|------|
| `notebooks/01_bad_prompt_clinic.ipynb` | Bad Prompt Clinic: diagnose, fix, and compare 6 real-world broken prompts using the Claude API | 15 min |

### Exercises

| File | Topic |
|------|-------|
| `exercises/01_mistake_taxonomy.md` | Build a mistake taxonomy for a domain of your choice |

---

## Prerequisites

- Module 2: Switch Variables (conditions that change answers)
- Module 3: The Condition Stack Framework (how conditions combine)
- Module 4: Conditional Trees (when one answer is wrong)
- An Anthropic API key (for the notebook)

---

## The Six Mistakes

| # | Mistake | Signal | Fix |
|---|---------|--------|-----|
| 1 | Confusing detail with conditions | Adding words, getting the same wrong answer | Separate information from evidence |
| 2 | Asking for one answer when you need a tree | Getting a verdict that doesn't apply to you | Prompt for conditional branches |
| 3 | Treating AI like a search engine | Keyword prompts, generic results | Specify the inference chain you need |
| 4 | No objective function | Advice that doesn't match your goal | State what you are optimizing for |
| 5 | Ignoring temporal conditions | Advice correct yesterday, wrong today | Specify when the situation applies |
| 6 | Assuming shared priors | Model confidently answers for the average case | Make your priors explicit |

---

## Core Idea

Every bad AI answer is a probability mistake you made before the model responded. The model is doing exactly what its training says to do — answering the most likely question given your evidence. When the answer is wrong, the evidence was incomplete.

The diagnostic framework treats every bad answer as a diagnostic signal: which conditions are missing? Which priors are misaligned? Which objective function was the model optimizing for instead of yours?

**The craft is conditioning, not coercion.**

---

## What's Next

Module 7 applies everything from this course to production patterns — building prompt pipelines that inject conditions dynamically, so the right conditions are always present regardless of who's asking.
