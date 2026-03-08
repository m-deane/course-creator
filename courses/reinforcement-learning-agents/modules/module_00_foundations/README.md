# Module 00: Foundations — SFT vs RL for AI Agents

## Overview

This module establishes the conceptual foundation for the entire course: why supervised fine-tuning (SFT) is insufficient for multi-step tool-using agents, and what reinforcement learning provides instead.

By the end of this module you will understand the core failure mode that motivates RL for agents, what a reward signal is and how it drives learning, and the basic mechanics of policy optimization. Module 01 builds directly on this by introducing GRPO, the specific RL algorithm used throughout this course.

## Learning Objectives

After completing this module, you will be able to:

1. Explain why SFT teaches imitation but not success, and identify specific failure modes in multi-step agentic contexts
2. Distinguish between binary, scalar, and relative reward signals, and select the appropriate type for a given agent task
3. Define what a policy is in the RL sense and describe how policy gradient methods update it
4. Implement a basic reward function for a tool-calling agent in Python
5. Calculate relative advantages from a group of rollout scores (prerequisite for understanding GRPO)

## Prerequisites

- Python proficiency (functions, classes, NumPy basics)
- Basic familiarity with language model inference (prompt → completion)
- Conceptual understanding of gradient descent (no math required at this stage)

## Guides

| File | Topic | Time |
|------|-------|------|
| `guides/01_sft_vs_rl_guide.md` | Why imitation learning breaks for agents | 20 min |
| `guides/02_reward_signals_guide.md` | Reward functions that drive learning | 15 min |
| `guides/03_policy_optimization_basics_guide.md` | Policies, gradients, and advantage | 20 min |

Each guide has a companion `_slides.md` for lecture delivery.

## Exercises

| File | Topic |
|------|-------|
| `exercises/01_sft_vs_rl_exercise.py` | Classify scenarios, implement rewards, compute advantages |

## Module Map

```
Module 00: Foundations
    │
    ├── Guide 01: SFT vs RL
    │       What SFT does → Why it fails for agents → What RL adds
    │
    ├── Guide 02: Reward Signals
    │       Binary → Scalar → Relative rankings → Reward shaping
    │
    └── Guide 03: Policy Optimization Basics
            Policy definition → Policy gradient → Advantage estimation
            └── Sets up: Module 01 (GRPO Algorithm)
```

## Connections

- **This module leads to:** Module 01 — GRPO Algorithm (uses advantage estimation from Guide 03)
- **Related course:** `agentic-ai-llms` Module 00 (agent architecture fundamentals)

## Time Estimate

- Guides (reading): 55 minutes total
- Slides (lecture): 45 minutes total
- Exercise: 30 minutes
- Total: approximately 2 hours
