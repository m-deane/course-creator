# Module 07: Production Considerations

## Module Overview

You have trained an RL agent. Now you need to know whether it is actually better than the alternatives, how much it costs to run, and how to deploy it reliably. This module covers the practical decisions that determine whether your trained model is worth using in production.

The central benchmark result: a 14B open-source model trained with GRPO on a single GPU for under $80 achieved 96% accuracy on the ART-E benchmark — outperforming o3, o4-mini, Gemini 2.5 Pro, and GPT-4.1, running 5x faster and costing 64x less per 1,000 queries.

That result only matters if you can measure it, reproduce it, and deploy it. This module shows you how.

---

## Learning Objectives

By the end of this module you will be able to:

1. Build a benchmark evaluation harness and measure your agent's accuracy against a held-out test set
2. Compare your trained model's cost and latency against frontier API alternatives
3. Estimate training budget requirements before committing to a full run
4. Configure vLLM for production serving of LoRA-adapted models
5. Implement a deployment pipeline with health monitoring and rollback capability
6. Decide when to train a custom RL model versus when to use a frontier API

---

## Why This Module Exists

Building the training pipeline (Modules 1–6) is only half the work. The questions that determine real-world value are:

- **Accuracy:** Is the trained model measurably better, or is the improvement within noise?
- **Cost:** Does the training investment pay off faster than just using GPT-4.1?
- **Latency:** Is the model fast enough for the use case?
- **Reliability:** Does it degrade over time, and how do you catch that?

These are engineering questions, not research questions. They have concrete, measurable answers.

---

## Guide Structure

| Guide | Topic | Key Concepts |
|-------|-------|--------------|
| 01 | Benchmarking | ART-E results, evaluation harness, accuracy measurement |
| 02 | Cost Optimization | Training budget, LoRA vs full fine-tune, when to train vs use API |
| 03 | Deployment Patterns | vLLM serving, monitoring, versioning, rollback |

---

## Prerequisites

- Completed Module 05 (Training Loop) — you have a trained checkpoint
- Completed Module 06 (Text-to-SQL Agent) — you have a concrete task to evaluate
- Familiarity with vLLM from Module 02 (ART Framework)

---

## Estimated Time

- Guide 01 (Benchmarking): 25 minutes
- Guide 02 (Cost Optimization): 20 minutes
- Guide 03 (Deployment Patterns): 25 minutes
- Exercise 01 (Production Exercise): 30 minutes

Total: ~100 minutes

---

## The Decision Framework

```
Do you have a well-defined, measurable task?
├── No  → Define the task first. RL without a clear success metric is wasted compute.
└── Yes ↓

Can you generate 1,000+ training examples with automatic reward signals?
├── No  → Use a frontier API (GPT-4.1, Gemini 2.5 Pro). Training requires data.
└── Yes ↓

Is latency or cost a constraint?
├── No  → Use a frontier API for simplicity.
└── Yes ↓

Is the task narrow enough that a 14B model can master it?
├── No  → Consider a larger base model or a different approach.
└── Yes → Train with RL. Estimated break-even: ~800 production queries vs GPT-4.1.
```

---

## Key Numbers to Know

| Metric | RL-Trained 14B (Qwen2.5) | OpenAI o3 |
|--------|---------------------------|-----------|
| Accuracy (ART-E) | 96% | ~87% |
| Latency per query | 1.1 seconds | 5.6 seconds |
| Cost per 1,000 runs | $0.85 | $55.19 |
| Training cost | ~$80 one-time | $0 |
| Break-even (vs o3) | ~1,470 queries | N/A |

---

## Next Steps After This Module

This is the final module of the core curriculum. From here:

- **Extend your agent:** Add new tools, retrain with RULER rewards (Module 03)
- **Scale your benchmark:** Increase test set size, add adversarial examples
- **Monitor in production:** Use the monitoring patterns from Guide 03
- **Portfolio project:** See `/projects/` for end-to-end capstone specifications
