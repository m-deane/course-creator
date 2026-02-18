# Module 06: Efficiency - Making It Actually Work

> **"If your model is 'smart but expensive', your competitors will beat you with 'slightly less smart but 10× cheaper'."**

## Learning Objectives

By the end of this module, you will:
- Understand scaling laws and compute-optimal training
- Implement LoRA and QLoRA for efficient fine-tuning
- Apply quantization for faster, cheaper inference
- Use FlashAttention and other efficiency techniques
- Design systems that balance quality, cost, and latency

## The Core Insight

In real deployments, **cost and latency are first-class metrics**.

A brilliant model that costs $1 per query or takes 30 seconds to respond won't survive in production. The winners optimize across three dimensions:

```
┌─────────────────────────────────────────────────────────────────┐
│                  THE EFFICIENCY TRIANGLE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                         QUALITY                                 │
│                            ▲                                    │
│                           /│\                                   │
│                          / │ \                                  │
│                         /  │  \                                 │
│                        /   │   \                                │
│                       /    │    \                               │
│                      /     │     \                              │
│                     /      │      \                             │
│                    /       │       \                            │
│                   ▼────────┴────────▼                           │
│               COST ◄──────────────► LATENCY                     │
│                                                                 │
│  Every technique trades between these dimensions.               │
│  Know your constraints. Optimize accordingly.                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Module Contents

### Guides
| Guide | Description | Time |
|-------|-------------|------|
| [01_scaling_laws.md](guides/01_scaling_laws.md) | Chinchilla and compute-optimal training | 15 min |
| [02_moe_architecture.md](guides/02_moe_architecture.md) | Mixture of Experts explained | 15 min |
| [03_flash_attention.md](guides/03_flash_attention.md) | IO-aware attention computation | 15 min |
| [04_quantization.md](guides/04_quantization.md) | INT8, INT4, GPTQ, AWQ | 15 min |
| [05_lora_qlora.md](guides/05_lora_qlora.md) | Parameter-efficient fine-tuning | 20 min |
| [06_distributed_training.md](guides/06_distributed_training.md) | ZeRO, FSDP, parallelism | 15 min |
| [cheatsheet.md](guides/cheatsheet.md) | Efficiency techniques decision tree | 5 min |

### Notebooks
| Notebook | Description | Time |
|----------|-------------|------|
| [01_quantization_speedup.ipynb](notebooks/01_quantization_speedup.ipynb) | Quantize and measure | 15 min |
| [02_lora_finetuning.ipynb](notebooks/02_lora_finetuning.ipynb) | Fine-tune with LoRA | 20 min |
| [03_inference_optimization.ipynb](notebooks/03_inference_optimization.ipynb) | KV cache, batching | 15 min |

## Key Concepts

### Scaling Laws: The Chinchilla Rule

```
┌─────────────────────────────────────────────────────────────────┐
│                    CHINCHILLA INSIGHT                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For compute budget C, optimal allocation:                      │
│                                                                 │
│     Parameters ∝ √C                                             │
│     Tokens     ∝ √C                                             │
│                                                                 │
│  Rule of thumb: Train on ~20 tokens per parameter               │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Model        │ Parameters │ Optimal Tokens │ Actual     │   │
│  │──────────────│────────────│────────────────│────────────│   │
│  │ GPT-3        │ 175B       │ 3.5T           │ 300B ❌    │   │
│  │ Chinchilla   │ 70B        │ 1.4T           │ 1.4T ✓     │   │
│  │ Llama 2 70B  │ 70B        │ 1.4T           │ 2T ✓       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  A well-trained smaller model beats an undertrained larger one  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Efficiency Techniques Overview

| Technique | Phase | Effect | Trade-off |
|-----------|-------|--------|-----------|
| **LoRA** | Fine-tuning | Train <1% of parameters | Slight quality loss |
| **QLoRA** | Fine-tuning | 4-bit base + LoRA | Memory ↓, Quality ≈ |
| **Quantization** | Inference | 2-4× faster, smaller | Quality ↓ slightly |
| **FlashAttention** | Both | 2-4× faster attention | Implementation complexity |
| **MoE** | Architecture | More capacity, same compute | Memory for all experts |
| **KV Caching** | Inference | Faster generation | Memory for cache |
| **Batching** | Inference | Higher throughput | Latency for first token |

### LoRA: The Key Technique

```python
# Full fine-tuning: Update ALL weights
# W' = W + ΔW  (175B parameters to update)

# LoRA: Low-rank decomposition
# W' = W + BA
# Where B: (d × r), A: (r × k), r << d, k
# Only train A and B (< 1% of parameters)

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,                      # Rank (lower = fewer params)
    lora_alpha=32,            # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which layers
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(base_model, config)
# Now only ~0.1% of parameters are trainable
```

### Quantization Quick Reference

| Format | Bits | Size Reduction | Quality Impact | Speed |
|--------|------|----------------|----------------|-------|
| FP32 | 32 | 1× (baseline) | Baseline | 1× |
| FP16/BF16 | 16 | 2× | Negligible | 1.5-2× |
| INT8 | 8 | 4× | Small | 2-3× |
| INT4 | 4 | 8× | Moderate | 3-4× |
| NF4 (QLoRA) | 4 | 8× | Small | 3-4× |

## Templates

```
templates/
├── lora_training_template.py    # Production LoRA setup
├── quantized_inference_template.py  # Serve quantized models
└── batch_inference_template.py  # High-throughput serving
```

## Prerequisites

- Module 01: Transformer Fundamentals (recommended)
- Understanding of neural network training
- Access to GPU for practical exercises

## Next Steps

After this module:
- **Need to evaluate?** → Module 07: Evaluation
- **Ready to deploy?** → Module 08: Production Systems
- **Want to fine-tune for behavior?** → Module 02: Alignment

## Cost Comparison

```
┌──────────────────────────────────────────────────────────────────┐
│                    COST PER 1M TOKENS                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Claude 3.5 Sonnet (API)        │████████████████│ $3.00       │
│  GPT-4 Turbo (API)              │████████████████████│ $10.00  │
│  Llama 70B (self-hosted)        │████│ $0.80                   │
│  Llama 70B INT4 (self-hosted)   │██│ $0.30                     │
│  Llama 7B (self-hosted)         │█│ $0.08                      │
│                                                                  │
│  * Self-hosted costs include GPU rental, vary by provider        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Time Estimate

- Quick path: 45 minutes (notebooks only)
- Full path: 2.5 hours (guides + notebooks)

---

*"The right optimization depends on your constraints: cost, latency, quality. Know which matters most."*
