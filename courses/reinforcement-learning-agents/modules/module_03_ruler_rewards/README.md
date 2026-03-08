# Module 03: RULER Automatic Rewards

## Overview

The hardest part of reinforcement learning is defining what "good" means. RULER (Reward Using LLM Evaluation Rubrics) solves this by using an LLM-as-a-judge to automatically generate reward signals — no hand-crafted reward functions required.

This module covers the full RULER mechanism: why relative scoring outperforms absolute scoring, how it integrates with GRPO, and how to build hybrid reward functions that combine programmatic checks with LLM quality judgments.

## Learning Objectives

By completing this module, you will be able to:

1. Explain why manual reward engineering fails for complex agentic tasks
2. Implement a RULER-style LLM-as-a-judge scoring function
3. Describe why relative scoring ("which is best?") is more reliable than absolute scoring ("rate 0-10")
4. Build hybrid reward functions that combine programmatic correctness checks with RULER quality scores
5. Design judge prompts that produce consistent, meaningful reward signals
6. Test and validate reward functions before using them in training

## Prerequisites

- Module 01: GRPO Algorithm (especially group sampling and relative advantage)
- Module 02: ART Framework (trajectory structure, rollout format)
- Basic Python async/await patterns
- OpenAI or Anthropic API access

## Module Contents

```
module_03_ruler_rewards/
├── README.md                           # This file
├── guides/
│   ├── 01_reward_function_challenge_guide.md     # Why reward design is hard
│   ├── 01_reward_function_challenge_slides.md    # Companion slides
│   ├── 02_ruler_mechanism_guide.md               # How RULER works
│   ├── 02_ruler_mechanism_slides.md              # Companion slides
│   ├── 03_custom_rewards_guide.md                # Hybrid reward design
│   └── 03_custom_rewards_slides.md               # Companion slides
├── exercises/
│   └── 01_ruler_exercise.py                      # Self-check: build a judge
└── resources/
    └── additional_readings.md                    # Papers and references
```

## Estimated Time

- Guides: 45-60 minutes
- Exercises: 30-45 minutes
- Total: ~90 minutes (split across 2 sessions)

## Key Concepts

| Concept | What It Means |
|---------|---------------|
| Reward hacking | Agent finds unintended ways to maximize reward without achieving the actual goal |
| LLM-as-a-judge | Using a capable LLM to evaluate the quality of another LLM's outputs |
| Relative scoring | Ranking N outputs against each other rather than scoring each independently |
| GRPO compatibility | RULER scores work directly as GRPO advantages without normalization |
| Hybrid rewards | Combining binary programmatic checks with continuous RULER quality scores |

## Connection to the Course

```
Module 01 (GRPO) → needs reward signals for each group of trajectories
Module 02 (ART)  → generates the trajectories RULER will score
Module 03 (RULER) → produces the reward signals GRPO optimizes
Module 05 (Training Loop) → integrates everything end-to-end
```

RULER sits at the center of the training loop: it is what tells GRPO which trajectories were good and which were not.
