# Module 05: Training Loop Deep-Dive

## Overview

The training loop is where everything in this course comes together. GRPO generates group completions. RULER scores them comparatively. The model updates and a new LoRA checkpoint loads into vLLM. Repeat until the agent reliably solves the target task.

This module examines every step of that loop in detail: what a rollout produces, what happens inside a single training step, and how checkpoint management keeps inference and training in sync during the hours-long training run.

By the end of this module you will be able to read a live training log and know exactly what is happening at each line.

## Learning Objectives

After completing this module, you will be able to:

1. Define a rollout and trace the data it produces from prompt to scored trajectory
2. Implement a complete rollout function that calls MCP tools and returns a structured trajectory
3. Describe what happens at every stage of a single GRPO training step
4. Read training logs and identify whether reward trends indicate healthy learning
5. Explain why LoRA checkpoints are preferred over full model checkpoints during training
6. Implement hot-swapping a new checkpoint into a running vLLM server
7. Write checkpoint management utilities for saving, evaluating, and resuming training runs

## Prerequisites

- Module 01: GRPO Algorithm (group sampling, relative advantage, policy update mechanics)
- Module 02: ART Framework (client/backend split, vLLM, Unsloth)
- Module 03: RULER Rewards (LLM-as-a-judge scoring, relative reward assignment)
- Module 04: MCP Integration (FastMCP tool servers, tool discovery)

## Module Contents

```
module_05_training_loop/
├── README.md                                   # This file
├── guides/
│   ├── 01_rollouts_guide.md                   # What a rollout is and how to implement one
│   ├── 01_rollouts_slides.md                  # Companion slides
│   ├── 02_training_step_guide.md              # Full training step: rollouts → RULER → GRPO → checkpoint
│   ├── 02_training_step_slides.md             # Companion slides
│   ├── 03_checkpoint_management_guide.md      # LoRA checkpoints, hot-swapping, save/resume
│   └── 03_checkpoint_management_slides.md     # Companion slides
├── exercises/
│   └── 01_training_loop_exercise.py           # Self-check: implement and instrument the training loop
├── notebooks/                                  # Placeholder for micro-notebooks (Module 06)
└── resources/
    └── additional_readings.md                 # Papers, references, and tools
```

## Estimated Time

- Guides (reading): 60 minutes total
- Slides (lecture): 45 minutes total
- Exercise: 40 minutes
- Total: approximately 2.5 hours

## Key Concepts

| Concept | What It Means |
|---------|---------------|
| Rollout | A single episode: agent receives a scenario, calls tools, produces a response |
| Trajectory | The complete record of a rollout: messages, tool calls, results, and reward |
| Training step | One iteration: N rollouts per scenario → RULER scoring → GRPO update → checkpoint |
| LoRA checkpoint | A small adapter file (~MB) that modifies the base model's behavior |
| Hot-swap | Replacing the active LoRA adapter in vLLM without restarting the server |
| Reward trend | The moving average of per-step rewards — the primary indicator of healthy training |

## The Training Loop at a Glance

```
for each training step:
    for each scenario in batch:
        run agent N times → N trajectories
    score all trajectories with RULER (relative 0–1)
    feed scores into GRPO → compute advantages → update LoRA weights
    save new checkpoint
    hot-swap checkpoint into vLLM
    log step reward, print trajectory samples
```

## Connection to the Course

```
Module 01 (GRPO)      → provides the update algorithm
Module 02 (ART)       → provides the training infrastructure
Module 03 (RULER)     → provides the reward signal
Module 04 (MCP)       → provides the tool environment
Module 05 (This)      → integrates all of the above into the full training loop
Module 06 (Text-SQL)  → applies the full loop to a real end-to-end agent
```
