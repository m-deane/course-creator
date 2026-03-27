# Module 5: Applying Bayesian Prompts to AI Agents and Workflows

## Overview

Single-turn conditioning is tractable. The hard problem is **multi-step conditioning**: how do the conditions you establish in step 1 survive and propagate through step 5 of an agent workflow?

This module applies the Bayesian framework from Modules 1–4 to agentic systems. You will learn why conditions decay across agent chains, how to prevent that decay with structured handoffs, and how Claude-specific features (system prompts, prefilling, tool descriptions, structured outputs) map directly onto the condition stack layers.

The payoff: agents that reason inside the right world at every step — not just the first.

---

## Learning Objectives

By the end of this module, you will be able to:

1. Explain condition decay: why conditions set in agent step 1 often vanish by step 5
2. Design condition-aware system prompts that serve as persistent Layer 0 priors
3. Use structured JSON handoffs to pass switch variables between agents without loss
4. Identify which Claude-specific features map to which condition stack layers
5. Build a working condition-aware Claude agent that identifies missing conditions before answering
6. Build a two-agent pipeline where Agent 1 extracts conditions and Agent 2 generates conditional answers

---

## Module Contents

### Guides

| File | Topic | Time |
|------|-------|------|
| `guides/01_agent_conditioning_guide.md` | Condition propagation, decay, and structured handoffs across agent chains | 20 min |
| `guides/01_agent_conditioning_slides.md` | Slide deck: condition flow diagrams, decay visualization, prevention patterns | Presentation |
| `guides/02_claude_specific_patterns_guide.md` | Claude system prompts, prefilling, tool descriptions, structured outputs as condition layers | 20 min |
| `guides/02_claude_specific_patterns_slides.md` | Slide deck: Claude feature-to-layer mapping, multi-agent patterns | Presentation |

### Notebooks

| File | Topic | Time |
|------|-------|------|
| `notebooks/01_condition_aware_agent.ipynb` | Build an agent that identifies missing conditions and assembles a condition stack before answering | 15 min |
| `notebooks/02_multi_agent_pipeline.ipynb` | Two-agent pipeline: Condition Extractor → Conditional Answerer with JSON handoff | 15 min |

### Exercises

| File | Topic |
|------|-------|
| `exercises/01_design_agent_prompts.md` | Design condition-aware system prompts for legal research, code review, and data analysis agents |

---

## Prerequisites

- Module 2: Switch Variables — you need to know what switch variables are before passing them between agents
- Module 3: The Condition Stack Framework — the 6-layer stack is the structure being passed through pipelines
- Module 4: Conditional Trees — agents often generate conditional trees, not single answers
- An Anthropic API key set as `ANTHROPIC_API_KEY`

---

## Core Problem: Condition Decay

$$P(A \mid C_1, C_2, \ldots, C_6) \text{ (step 1)} \neq P(A \mid C_{\text{received}}) \text{ (step N)}$$

When agents chain, each step receives only what the previous step explicitly passed. Conditions not included in the handoff payload are filled from the model's training prior — which is the average case, not your case.

**The decay pattern:**

```
Step 1: Full condition stack (jurisdiction, time, objective, constraints, facts, format)
Step 2: Receives facts + partial context             -- loses jurisdiction
Step 3: Receives output from step 2                 -- loses objective
Step 4: Receives output from step 3                 -- loses constraints
Step 5: Reasons from near-prior                     -- your conditions are gone
```

**The solution:** Structured condition handoffs as first-class data, not implicit context.

---

## Why This Module Matters

Every team building AI agents discovers condition decay the same way: the agent gives a confident, fluent, completely wrong answer because the context it needed was dropped 3 steps back. This is not a model failure. It is a conditioning failure — and it is entirely preventable.

---

## What's Next

Module 6 (Common Probability Mistakes) examines the six systematic errors that appear in both single-turn and multi-agent prompts: base rate neglect, conditioning on the wrong event, collider bias, prior-data confusion, and more.
