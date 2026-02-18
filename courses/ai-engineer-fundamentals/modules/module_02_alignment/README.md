# Module 02: Alignment - Shaping Model Behavior

> **"Base models predict text. Aligned models are helpful assistants. The gap is everything."**

## Learning Objectives

By the end of this module, you will:
- Understand how Supervised Fine-Tuning (SFT) teaches format and instruction-following
- Implement RLHF (Reinforcement Learning from Human Feedback) to optimize preferences
- Use DPO (Direct Preference Optimization) as a simpler alternative to RLHF
- Apply Constitutional AI principles for scalable, principle-based alignment
- Choose the right alignment method for your use case

## The Core Insight

A base language model trained only on next-token prediction learns:
> "What text is likely to follow this text?"

But what you need is a model that learns:
> "What response would be most helpful to this user?"

**The uncomfortable truth:** Base models are confidently wrong, unhelpful, and sometimes harmful. Alignment transforms prediction engines into reliable assistants.

## The Alignment Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    THE ALIGNMENT JOURNEY                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  BASE MODEL (Next-token prediction)                             │
│  "Predict what text comes next"                                 │
│  └────────────┬────────────────────────────────────────────┐    │
│               │                                            │    │
│               ▼                                            │    │
│  SFT (Supervised Fine-Tuning)                              │    │
│  "Learn format and instruction-following"                  │    │
│  Input: (instruction, good_response) pairs                 │    │
│  └────────────┬────────────────────────────────────────────┤    │
│               │                                            │    │
│               ▼                                            │    │
│  PREFERENCE LEARNING                                       │    │
│  "Optimize for human preferences"                          │    │
│  ┌────────────┬────────────────────────────────────┐       │    │
│  │            │                                    │       │    │
│  │    RLHF    │         DPO                        │  CAI  │    │
│  │ (Complex)  │      (Simple)                      │(AI FB)│    │
│  │            │                                    │       │    │
│  │ RM → PPO   │  Direct optimization               │ RLAIF │    │
│  └────────────┴────────────────────────────────────┴───────┘    │
│               │                                            │    │
│               ▼                                            │    │
│  ALIGNED MODEL                                             │    │
│  "Helpful, harmless, honest assistant"                     │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Module Contents

### Guides
| Guide | Description | Time |
|-------|-------------|------|
| [01_sft_supervised_finetuning_guide.md](guides/01_sft_supervised_finetuning_guide.md) | Supervised fine-tuning for format and instructions | 15 min |
| [02_rlhf_explained_guide.md](guides/02_rlhf_explained_guide.md) | RLHF three-stage process explained | 20 min |
| [03_dpo_direct_preference_guide.md](guides/03_dpo_direct_preference_guide.md) | DPO as simpler RLHF alternative | 15 min |
| [04_constitutional_ai_guide.md](guides/04_constitutional_ai_guide.md) | Principle-based alignment with AI feedback | 15 min |
| [05_when_to_align_guide.md](guides/05_when_to_align_guide.md) | Decision framework for choosing methods | 10 min |
| [cheatsheet.md](guides/cheatsheet.md) | Quick reference for alignment methods | 5 min |

### Notebooks
| Notebook | Description | Time |
|----------|-------------|------|
| [01_sft_with_trl.ipynb](notebooks/01_sft_with_trl.ipynb) | Fine-tune with HuggingFace TRL | 15 min |
| [02_dpo_implementation.ipynb](notebooks/02_dpo_implementation.ipynb) | Implement DPO from scratch | 15 min |
| [03_alignment_comparison.ipynb](notebooks/03_alignment_comparison.ipynb) | Compare SFT vs DPO vs base | 10 min |

### Exercises
Self-check exercises to verify understanding (no grades).

### Resources
- [additional_readings.md](resources/additional_readings.md) - Key papers: InstructGPT, DPO, Constitutional AI
- [figures/](resources/figures/) - Diagrams and visual assets

## Key Concepts

### Alignment Methods Comparison

| Method | What It Does | Input Needed | Complexity | When to Use |
|--------|-------------|--------------|------------|-------------|
| **SFT** | Teaches format & instructions | (instruction, response) pairs | Low | Format learning, basic instruction-following |
| **RLHF** | Optimizes for preferences | Preference rankings (A > B) | High | Maximum quality, have compute budget |
| **DPO** | Direct preference optimization | Preference pairs (chosen, rejected) | Medium | Better than SFT, simpler than RLHF |
| **Constitutional AI** | Principle-based + AI feedback | Principles + base model | Medium | Scalable alignment, explicit values |

### The Three-Stage RLHF Process

```
Stage 1: SFT
├── Input: Demonstration data (instruction → response)
├── Method: Standard supervised learning
└── Output: Model that follows instructions

Stage 2: Reward Model
├── Input: Preference comparisons (response A > response B)
├── Method: Train classifier to predict preferences
└── Output: Reward model that scores responses

Stage 3: RL Optimization
├── Input: Reward model + SFT model
├── Method: PPO with KL constraint
└── Output: Policy optimized for human preferences
```

### Why Alignment Matters

| Without Alignment | With Alignment |
|-------------------|----------------|
| Completes patterns, not tasks | Completes tasks |
| Can be toxic or harmful | Refuses harmful requests |
| Inconsistent format | Structured, predictable output |
| Hallucinates confidently | Expresses uncertainty |
| No safety guardrails | Built-in safety policies |

## The Base vs SFT vs RLHF Distinction

**Example prompt:** "Write a Python function to calculate factorial"

**Base model response:**
```
def factorial(n):
    # TODO: implement
    pass

# Test cases
assert factorial(5) == 120

# Common interview question...
```
(Continues the pattern from training data, doesn't complete the task)

**SFT model response:**
```python
def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n-1)
```
(Follows instruction, produces working code)

**RLHF/DPO model response:**
```python
def factorial(n):
    """Calculate factorial of n using recursion.

    Args:
        n: Non-negative integer

    Returns:
        Factorial of n

    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("Factorial undefined for negative numbers")
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)
```
(Optimized for helpfulness: includes docstring, error handling, explanation)

## Prerequisites

- Module 01 understanding of Transformers (helpful but not required)
- Python proficiency
- Basic understanding of supervised learning
- Access to compute for fine-tuning (can use Colab free tier)

## Next Steps

After this module:
- **Want to evaluate alignment quality?** → Module 07: Evaluation
- **Want to optimize training efficiency?** → Module 06: Efficiency (LoRA, QLoRA)
- **Want to build aligned agents?** → Module 03: Memory Systems

## Time Estimate

- Quick path: 1 hour (notebooks only)
- Full path: 2-3 hours (guides + notebooks + exercises)
- Deep dive: 5+ hours (read all papers, implement from scratch)

## Key Takeaways

1. **SFT teaches format and instruction-following** - It's supervised learning on demonstration data
2. **RLHF optimizes for preferences** - Three stages: SFT → Reward Model → PPO
3. **DPO is the practical choice** - Simpler, more stable, comparable performance
4. **Constitutional AI scales alignment** - Use principles and AI feedback to reduce human labeling
5. **Alignment is not optional** - Base models are not production-ready assistants

---

*"The gap between a base model and ChatGPT isn't architectural. It's alignment."*
