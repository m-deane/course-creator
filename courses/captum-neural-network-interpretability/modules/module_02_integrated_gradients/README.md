# Module 02: Integrated Gradients

## Overview

Integrated Gradients (IG) is the axiomatic gold standard for gradient-based attribution. Unlike saliency maps and Input×Gradient, IG satisfies two fundamental fairness axioms — Sensitivity and Implementation Invariance — and produces attributions with a provable completeness guarantee: the sum of all attributions equals the difference in model output between the input and a reference baseline.

This module covers IG from first principles through production application, including text attribution with transformer models and variance reduction via NoiseTunnel.

## Learning Objectives

By completing this module, you will:

1. Derive the IG formula from the Fundamental Theorem of Calculus
2. Verify the completeness property (convergence delta) on real models
3. Select appropriate baselines for images, text, and tabular data
4. Apply LayerIntegratedGradients to explain transformer model predictions
5. Use NoiseTunnel to produce SmoothGrad-IG and VarGrad-IG attributions
6. Compare multiple baselines to assess attribution robustness

## Prerequisites

- Module 01 (Gradient Methods) completed
- PyTorch fluency (tensor operations, autograd)
- Familiarity with transformer tokenization (helpful for Notebook 02)

## Estimated Time

- Guides: 45 minutes (2 guides × ~22 min each)
- Notebooks: 41 minutes (14 + 14 + 13 min)
- Exercise: 25 minutes
- **Total: ~111 minutes**

## Contents

### Guides

| File | Topic | Time |
|------|-------|------|
| `guides/01_ig_theory_guide.md` | FTC derivation, completeness axiom, uniqueness theorem | 22 min |
| `guides/01_ig_theory_slides.md` | Companion slide deck (17 slides) | — |
| `guides/02_baselines_convergence_guide.md` | Baseline selection, convergence delta, LayerIG for text | 23 min |
| `guides/02_baselines_convergence_slides.md` | Companion slide deck (15 slides) | — |

### Notebooks

| File | Topic | Time |
|------|-------|------|
| `notebooks/01_ig_image_classification.ipynb` | Three baselines side-by-side on ResNet-50 | 14 min |
| `notebooks/02_ig_text_classification.ipynb` | LayerIG on DistilBERT SST-2, token attribution | 14 min |
| `notebooks/03_smoothgrad_noise_tunnel.ipynb` | SmoothGrad-IG vs VarGrad-IG, noise reduction sweep | 13 min |

### Exercises

| File | Topic | Questions |
|------|-------|-----------|
| `exercises/01_ig_self_check.py` | Theory, convergence, baselines, LayerIG, NoiseTunnel | 22 |

## Key Equations

**IG Formula:**
$$\text{IG}_i(x) = (x_i - x'_i) \cdot \int_0^1 \frac{\partial f(x' + \alpha(x - x'))}{\partial x_i} \, d\alpha$$

**Completeness Property:**
$$\sum_i \text{IG}_i(x) = f(x) - f(x')$$

**Convergence Delta (validation):**
$$\delta = \left|\sum_i \text{IG}_i^{\text{approx}} - (f(x) - f(x'))\right| < 0.05$$

**SmoothGrad-IG:**
$$\text{SG-IG}_i(x) = \frac{1}{N} \sum_{k=1}^N \text{IG}_i(x + \epsilon_k), \quad \epsilon_k \sim \mathcal{N}(0, \sigma^2)$$

## Captum API Quick Reference

```python
from captum.attr import IntegratedGradients, LayerIntegratedGradients, NoiseTunnel

# Standard IG
ig = IntegratedGradients(model)
attr, delta = ig.attribute(
    inputs, baselines=baseline, target=class_idx,
    n_steps=50, return_convergence_delta=True
)

# LayerIG for text (embedding layer)
lig = LayerIntegratedGradients(model, model.distilbert.embeddings)
attr, delta = lig.attribute(
    input_ids, baselines=zeros_like(input_ids),
    additional_forward_args=(attention_mask,),
    target=class_idx, n_steps=50,
    return_convergence_delta=True
)
token_scores = attr.sum(dim=-1).squeeze()

# SmoothGrad-IG
nt = NoiseTunnel(ig)
smooth_attr = nt.attribute(
    inputs, nt_type='smoothgrad', nt_samples=10, stdevs=0.05,
    baselines=baseline, target=class_idx, n_steps=50
)
```

## Baseline Selection Summary

| Data Type | Default Baseline | Interpretation |
|-----------|-----------------|----------------|
| Images | `torch.zeros_like(input)` | Black image — no pixel information |
| Images (alt) | Gaussian blur, σ=15 | Blurred background — local detail contribution |
| Text | `torch.zeros_like(input_ids)` | All PAD tokens — no token information |
| Tabular | `X_train.mean(axis=0)` | Typical sample from training distribution |

## Convergence Delta Guide

| Delta | Quality | Action |
|-------|---------|--------|
| < 0.01 | Excellent | None needed |
| 0.01–0.05 | Good | Acceptable for most uses |
| > 0.05 | Marginal | Increase `n_steps` |
| > 0.10 | Poor | Use `n_steps=300+` |

## Next Module

**Module 03 — Layer & Neuron Attribution:** GradCAM, Layer Conductance, and Neuron Conductance — explaining what happens inside the network, not just at the input.
