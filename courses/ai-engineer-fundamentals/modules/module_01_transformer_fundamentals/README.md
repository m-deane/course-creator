# Module 01: Transformer Fundamentals

> **"Attention is learnable routing. Everything else is just plumbing."**

## Learning Objectives

By the end of this module, you will:
- Understand the attention mechanism as learnable information routing
- Know the full Transformer architecture (multi-head attention, FFN, residuals, layer norm)
- Implement attention from scratch in NumPy and PyTorch
- Understand positional encoding (sinusoidal, learned, RoPE, ALiBi)
- Recognize the three structural limitations that drive system design
- Be ready to work with pre-trained models intelligently

## The Core Insight

**The Transformer is not magic. It's three ideas:**
1. **Attention**: Learnable routing that decides which tokens influence each other
2. **Stacking**: Deep networks of attention + feed-forward blocks
3. **Parallelism**: All tokens process simultaneously (unlike RNNs)

That's it. Everything else—BERT, GPT, Claude, Llama—is variations on this theme.

## Why This Matters

You can't debug what you don't understand. When your LLM system:
- Uses too much memory → You need to know about KV caching
- Has slow inference → You need to understand the forward pass
- Struggles with long documents → You need to know context window limits
- Fails on position-dependent tasks → You need to understand positional encoding

**This module teaches you the engine so you can diagnose when it breaks.**

## Module Contents

### Guides
| Guide | Description | Time |
|-------|-------------|------|
| [01_attention_mechanism_guide.md](guides/01_attention_mechanism_guide.md) | The core idea: learnable routing with Q/K/V | 15 min |
| [02_transformer_architecture_guide.md](guides/02_transformer_architecture_guide.md) | Full architecture: multi-head attention, FFN, residuals | 20 min |
| [03_positional_encoding_guide.md](guides/03_positional_encoding_guide.md) | Teaching the model about position | 15 min |
| [04_three_limitations_guide.md](guides/04_three_limitations_guide.md) | The structural limits that drive system design | 10 min |
| [cheatsheet.md](guides/cheatsheet.md) | Quick reference: formulas, architecture, hyperparameters | 5 min |

### Notebooks
| Notebook | Description | Time |
|----------|-------------|------|
| [01_attention_from_scratch.ipynb](notebooks/01_attention_from_scratch.ipynb) | Build attention in NumPy, then PyTorch | 15 min |
| [02_full_transformer_block.ipynb](notebooks/02_full_transformer_block.ipynb) | Implement a complete Transformer layer | 15 min |
| [03_using_pretrained_models.ipynb](notebooks/03_using_pretrained_models.ipynb) | Load and inspect GPT-2, understand the structure | 10 min |

### Exercises
Self-check exercises to verify understanding (no grades).

### Resources
- [additional_readings.md](resources/additional_readings.md) - Papers and deep dives
- [figures/](resources/figures/) - Diagrams and visual assets

## Key Concepts

### 1. Attention Mechanism
**What it does:** Routes information between tokens based on learned relevance.

```
Query: "What am I looking for?"
Key:   "What do I contain?"
Value: "What information do I have?"

Attention score = how much Query attends to each Key
Output = weighted sum of Values based on scores
```

**Formula:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

### 2. Transformer Architecture
**Components (in order):**
1. Input embedding + positional encoding
2. **N × Transformer blocks**, each containing:
   - Multi-head self-attention
   - Layer normalization
   - Feed-forward network (2 layers, ReLU or GELU)
   - Residual connections around each sub-layer
3. Final layer norm
4. Output projection

**Key insight:** Residual connections preserve gradients and allow deep stacking.

### 3. Positional Encoding
**The problem:** Attention has no notion of order. "cat ate mouse" and "mouse ate cat" look identical.

**Solutions:**
- **Sinusoidal**: Fixed sin/cos patterns (original Transformer)
- **Learned**: Train position embeddings (BERT, GPT)
- **RoPE**: Rotate query/key by position (Llama, PaLM)
- **ALiBi**: Add position bias to attention scores (no embeddings)

### 4. Three Structural Limitations

These limitations are inherent to the architecture and drive system design:

| Limitation | Implication | System Solution |
|------------|-------------|-----------------|
| **Knowledge in weights** | Hard to update without retraining | RAG: retrieve at inference |
| **Limited context window** | Can't fit everything in prompt | Memory systems with hierarchy |
| **Confidently wrong** | High probability ≠ correctness | Grounding, tools, evaluation |

**Every module after this addresses these limitations.**

## The Math You Need

### Attention (Single Head)
```
Q = X W_Q    (queries)
K = X W_K    (keys)
V = X W_V    (values)

scores = QK^T / √d_k
attention_weights = softmax(scores)
output = attention_weights V
```

### Multi-Head Attention
```
Split into h heads, apply attention in parallel, concatenate, project:

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
where head_i = Attention(Q W_Q^i, K W_K^i, V W_V^i)
```

### Feed-Forward Network (per token, independently)
```
FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
       = ReLU(x W_1 + b_1) W_2 + b_2
```

Typically: `d_model` → `4 * d_model` → `d_model`

### Layer Normalization
```
LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β
```
Normalizes across the feature dimension (not batch).

## Architectural Variants

| Model | Key Difference |
|-------|----------------|
| **BERT** | Encoder-only, bidirectional attention |
| **GPT** | Decoder-only, causal (left-to-right) attention |
| **T5** | Encoder-decoder, both components |
| **Claude/Llama** | Decoder-only, modern improvements (RoPE, GQA, etc.) |

**Modern LLMs (2023+) are all decoder-only with causal attention.**

## Common Misconceptions

### "Attention is all you need"
**Reality:** You also need position encoding, layer norm, residuals, feed-forward networks, and 6+ layers stacked. The paper title is catchy but oversimplified.

### "Bigger context window is always better"
**Reality:** Attention is O(n²) in sequence length. Long context is expensive and doesn't automatically mean better retrieval. Smart memory management often beats raw context length.

### "The model 'understands' language"
**Reality:** It's next-token prediction trained on massive data. It learns correlations, not semantics. But those correlations are incredibly useful.

## Prerequisites

- Linear algebra: matrix multiplication, dot products
- Basic calculus: gradients, chain rule
- Python and NumPy
- PyTorch basics (we'll teach the specifics)

## Next Steps

After this module:
- **Module 02: Alignment** - How to shape model behavior (SFT, RLHF, DPO)
- **Module 03: Memory Systems** - RAG and context management
- **Module 06: Efficiency** - Making it fast and cheap (quantization, FlashAttention)

## Time Estimate

- **Quick path:** 30 minutes (skim guides, run notebooks)
- **Full path:** 2-3 hours (read all guides, implement from scratch)
- **Deep dive:** 5+ hours (read papers, experiment with variants)

## Setup

```bash
pip install torch numpy matplotlib transformers
```

All notebooks run in Google Colab with no additional setup.

---

*"The Transformer is the engine. Understanding it means you can debug, optimize, and extend it. That's the difference between using LLMs and engineering with them."*
