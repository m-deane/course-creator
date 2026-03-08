# Module 02: ART Framework — Agent Reinforcement Trainer

## Overview

ART (Agent Reinforcement Trainer) by OpenPipe is the open-source framework this course uses to train agents with GRPO. This module covers its architecture, installation, and the trajectory data structure that makes multi-step agent training possible.

The central design insight: split the training loop into a **client** (your agent code, anywhere) and a **backend** (vLLM + Unsloth, on GPU). This separation lets you write agent logic in any Python environment while training runs on dedicated hardware.

By the end of this module you will be able to install and configure ART, explain the client-backend split and why it matters, construct Trajectory objects from multi-turn agent runs, and wire a complete training loop.

## Learning Objectives

After completing this module, you will be able to:

1. Describe ART's client-backend architecture and the role of each component
2. Install and configure ART, vLLM, and Unsloth for a training run
3. Construct `art.Trajectory` objects from multi-turn agent conversations including tool calls
4. Explain how LoRA checkpoints are hot-swapped after each training step
5. Identify how ART differs from single-turn RL frameworks designed for chatbots
6. Write a complete `rollout` function and training loop using `art.TrainableModel`

## Prerequisites

- Module 00: SFT vs RL, reward signals
- Module 01: GRPO algorithm (group sampling, relative advantage, policy updates)
- Python async/await basics (`async def`, `await`, `asyncio`)

## Guides

| File | Topic | Time |
|------|-------|------|
| `guides/01_art_architecture_guide.md` | Client-backend split, LoRA hot-swapping, integrations | 20 min |
| `guides/02_art_installation_guide.md` | Installation, configuration, GPU requirements | 15 min |
| `guides/03_trajectories_guide.md` | Trajectory structure, tool calls, multi-turn recording | 20 min |

Each guide has a companion `_slides.md` for lecture delivery.

## Exercises

| File | Topic |
|------|-------|
| `exercises/01_art_setup_exercise.py` | Define training config, construct trajectories, validate format |

## Module Map

```
Module 02: ART Framework
    │
    ├── Guide 01: ART Architecture
    │       Client role → Backend role → LoRA hot-swap loop
    │       Integrations: LangGraph, CrewAI, ADK
    │
    ├── Guide 02: Installation & Configuration
    │       pip install → GPU setup → model selection → hyperparameters
    │
    └── Guide 03: Trajectories
            Structure → Multi-turn messages → Tool calls → GRPO input
            └── Sets up: Module 03 (RULER rewards), Module 05 (full training loop)
```

## Connections

- **Builds on:** Module 01 — GRPO Algorithm (trajectory groups are what GRPO trains on)
- **This module leads to:** Module 03 — RULER Rewards (scoring trajectories automatically)
- **This module leads to:** Module 05 — Training Loop (full orchestration)
- **Related course:** `agentic-ai-llms` — agent architecture and tool-calling patterns

## Time Estimate

- Guides (reading): 55 minutes total
- Slides (lecture): 45 minutes total
- Exercise: 25 minutes
- Total: approximately 2 hours
