# Module 0: Foundations — How Language Models Actually Work

## Overview

Before you can engineer prompts with precision, you need to understand what you are engineering. Language models do not "understand" your question and retrieve an answer — they compute conditional probability distributions over tokens, then sample from them. This module makes that concrete, so you can reason about why prompts succeed or fail the same way a statistician reasons about data.

**Time Estimate:** 3–4 hours

---

## Learning Objectives

By completing this module, you will:

1. Explain autoregressive generation as a chain of conditional probabilities $P(t_n \mid t_1, \ldots, t_{n-1})$
2. Identify when a model is defaulting to training priors instead of responding to your actual conditions
3. Rewrite generic prompts to supply the missing conditions that shift the probability distribution toward correct outputs
4. Use the Anthropic API to observe, in real output, how a single added sentence changes model behaviour across multiple domains

---

## Module Contents

### Guides

- `01_autoregressive_generation_guide.md` — How LLMs generate text token by token as conditional probability
- `01_autoregressive_generation_slides.md` — Companion slide deck (12 slides)
- `02_prior_dominance_guide.md` — Why models produce coherent but wrong answers, and how to fix it
- `02_prior_dominance_slides.md` — Companion slide deck (13 slides)

### Notebooks

- `01_token_probability.ipynb` — See conditional probability in action with live Claude API calls

### Exercises

- `01_identify_missing_conditions.md` — Given 5 broken prompts, diagnose what conditions are missing

---

## Core Insight

Every LLM failure you have ever seen — hallucination, generic advice, wrong context assumed — is a prior dominance problem. The model filled in the blanks with the most common pattern from training data because you did not supply conditions specific enough to narrow the probability distribution.

Prompt engineering is, at its core, condition engineering: specifying the evidence that shifts $P(\text{output} \mid \text{your context})$ away from the generic prior and toward the answer that is correct for your situation.

---

## Prerequisites

- Familiarity with the concept of conditional probability (Bayes' theorem is helpful but not required)
- Basic Python (you will run API calls in the notebook)
- An Anthropic API key set as `ANTHROPIC_API_KEY` in your environment

---

## Next Steps

After completing this module, proceed to Module 1: Prompts as Bayesian Evidence — where you will learn to write prompts that systematically update the model's probability distribution.
