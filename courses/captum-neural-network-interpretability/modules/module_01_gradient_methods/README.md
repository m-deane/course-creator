# Module 01 — Gradient-Based Attribution Methods

## Overview

This module covers the foundational gradient-based attribution methods: Saliency, Input×Gradient, Guided Backpropagation, and Deconvolution. These are the fastest attribution methods (single backward pass) and form the basis for understanding why more principled methods like Integrated Gradients are needed.

## Learning Objectives

By the end of this module, you will be able to:

1. Explain the mathematical foundation of each gradient attribution method
2. Apply all four methods to pretrained CNNs using Captum
3. Produce and interpret side-by-side attribution comparisons
4. Run the randomization sanity check to identify architecture-dependent methods
5. Apply gradient attribution to tabular neural networks
6. Explain why gradient saturation motivates Integrated Gradients

## Contents

### Guides

| File | Topic |
|------|-------|
| `guides/01_gradient_theory_guide.md` | Theory: Saliency, Input×Gradient, Guided Backprop, Deconvolution — math and failure modes |
| `guides/01_gradient_theory_slides.md` | Companion deck with LaTeX gradient equations |
| `guides/02_captum_gradient_api_guide.md` | Captum API patterns, side-by-side comparison, sanity check implementation |
| `guides/02_captum_gradient_api_slides.md` | Companion deck with code examples |

### Notebooks

| File | Topic | Time |
|------|-------|------|
| `notebooks/01_gradient_methods_cnn.ipynb` | All four methods on ResNet-50: side-by-side visualizations + sanity check | 15 min |
| `notebooks/02_gradient_methods_tabular.ipynb` | Gradient attribution on Wine Quality tabular neural network | 14 min |
| `notebooks/03_saliency_deep_dive.ipynb` | Saturation failure, gradient noise, SmoothGrad, path to IG | 14 min |

### Exercises

| File | Topic |
|------|-------|
| `exercises/01_gradient_self_check.py` | Self-check: method properties, saturation problem, sanity check, API debugging |

## Key Concepts

**Gradient methods:**
- Saliency: $\phi_i = |\nabla_{x_i} f|$ — absolute gradient magnitude
- Input×Gradient: $\phi_i = x_i \cdot \nabla_{x_i} f$ — Taylor approximation to contribution
- Guided Backprop: modified backward pass filtering negative gradients/activations
- Deconvolution: modified backward pass filtering negative gradients only

**Axiom compliance:**

| Method | Sensitivity | Impl. Invariance |
|--------|------------|-----------------|
| Saliency | No | Yes |
| Input×Gradient | No | Yes |
| Guided Backprop | No | **No** |
| Deconvolution | No | **No** |

**The sanity check:** Guided Backprop produces similar attributions for trained and randomly initialized models — it reflects the architecture, not the learned weights.

**SmoothGrad:** `NoiseTunnel(method)` in Captum — averages attribution over 20 noisy input copies. Reduces noise but does not fix saturation.

## Prerequisites

- Module 00 completed (environment, taxonomy, first explanation)

## Connections

**Previous:** Module 00 — Foundations (taxonomy, Captum overview)

**Next:** Module 02 — Integrated Gradients & Path Methods (the axiomatic solution to gradient limitations)

## Estimated Time

- Reading guides: 50 minutes
- Notebooks: 43 minutes
- Exercise: 20 minutes
- **Total: ~2 hours**
