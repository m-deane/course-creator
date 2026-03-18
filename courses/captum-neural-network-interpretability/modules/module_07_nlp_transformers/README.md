# Module 07 — NLP & Transformer Interpretability

## Overview

This module covers interpretability for text models and Transformers. Key topics:
- Token-level attribution via LayerIntegratedGradients
- Why attention weights are not attribution scores ("attention is not explanation")
- Layer-wise attribution in BERT to identify which encoder layers are most important
- Subword tokenization handling and colored text visualization

## Prerequisites

- Module 02: Integrated Gradients theory
- Module 03: Layer and neuron attribution
- Basic familiarity with BERT / Transformer architecture

## Module Contents

### Guides
| File | Topic |
|------|-------|
| `guides/01_token_attribution_guide.md` | Token attribution: embedding layer, baselines, aggregation |
| `guides/01_token_attribution_slides.md` | Companion deck (14 slides) |
| `guides/02_attention_vs_attribution_guide.md` | Why attention ≠ attribution; rollout; gradient×attention |
| `guides/02_attention_vs_attribution_slides.md` | Companion deck (15 slides) |
| `guides/03_transformer_layers_guide.md` | LayerConductance, layer importance by task |
| `guides/03_transformer_layers_slides.md` | Companion deck (13 slides) |

### Notebooks
| File | Topic | Time |
|------|-------|------|
| `notebooks/01_bert_sentiment_ig.ipynb` | BERT + LayerIG: colored text visualization | 15 min |
| `notebooks/02_attention_vs_ig.ipynb` | Attention vs. IG: empirical comparison, negation | 15 min |
| `notebooks/03_layer_attribution_transformers.ipynb` | LayerConductance: token × layer heatmap | 15 min |

### Exercises
| File | Topic |
|------|-------|
| `exercises/01_nlp_interpretability_self_check.py` | Baseline quality, efficiency, negation, layer profile |

## Key Concepts

### Token Attribution
1. Text is discrete → differentiate w.r.t. token embeddings (continuous)
2. `LayerIntegratedGradients(forward_func, model.bert.embeddings)`
3. Aggregate: `attributions.sum(dim=-1)` → one score per token
4. Merge WordPiece subwords → word-level scores
5. Baseline: `[PAD]` for BERT-family

### Attention vs. Attribution
- **Attention** = information routing weights (softmax-normalized)
- **Attribution** = counterfactual impact on prediction
- These disagree on negation, adversarial inputs, and complex syntax
- Use attention for exploration; use IG for explanation

### Layer Attribution
- `LayerConductance` measures each layer's contribution to the output change
- For SST-2 (sentiment): layers 10-11 dominate
- For NER/syntax: middle layers (4-6) dominate
- Token × layer heatmap shows attribution emergence by depth

## Running the Exercises

```bash
python modules/module_07_nlp_transformers/exercises/01_nlp_interpretability_self_check.py
```

## Next Module

**Module 08:** Production interpretability pipelines — Captum Insights, FastAPI service, model debugging.
