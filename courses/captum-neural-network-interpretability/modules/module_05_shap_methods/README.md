# Module 05 — SHAP and KernelSHAP in Captum

## Overview

SHAP (SHapley Additive exPlanations) derives from cooperative game theory and is the most theoretically grounded attribution framework in neural network interpretability. This module covers the theory of Shapley values, the KernelSHAP approximation, and gradient-based SHAP methods (GradientSHAP, DeepLIFT, DeepLIFT SHAP) available in Captum.

## Prerequisites

- Module 02: Integrated Gradients (Captum baseline/target API)
- Module 03: Layer and neuron attribution concepts
- Basic familiarity with expected values and conditional probability

## Module Contents

### Guides
| File | Topic |
|------|-------|
| `guides/01_shap_theory_guide.md` | Shapley values, four axioms, KernelSHAP algorithm |
| `guides/01_shap_theory_slides.md` | Companion slide deck (15 slides) |
| `guides/02_gradient_deep_shap_guide.md` | GradientSHAP, DeepLIFT, DeepLIFT SHAP |
| `guides/02_gradient_deep_shap_slides.md` | Companion slide deck (16 slides) |

### Notebooks
| File | Topic | Time |
|------|-------|------|
| `notebooks/01_kernelshap_vs_gradientshap.ipynb` | Benchmark: accuracy vs. speed trade-off | 15 min |
| `notebooks/02_shap_visualization.ipynb` | Beeswarm, dependence, waterfall, force plots | 15 min |
| `notebooks/03_deeplift_deep_dive.ipynb` | Saturation analysis, ResNet attribution comparison | 15 min |

### Exercises
| File | Topic |
|------|-------|
| `exercises/01_shap_self_check.py` | Efficiency verification, convergence, method agreement |

## Key Concepts

### SHAP Theory
The Shapley value for feature $i$ is the average marginal contribution across all possible feature orderings:

$$\phi_i = \sum_{S \subseteq \{1,\ldots,d\} \setminus \{i\}} \frac{|S|!(d-|S|-1)!}{d!} [v(S \cup \{i\}) - v(S)]$$

The four Shapley axioms — efficiency, symmetry, dummy, linearity — uniquely determine this formula.

### KernelSHAP
Approximates Shapley values via weighted linear regression with the SHAP kernel:

$$\pi_x(z') = \frac{d-1}{\binom{d}{|z'|}|z'|(d-|z'|)}$$

Tractable for moderate feature counts; recommended for black-box or non-differentiable models.

### GradientSHAP
Extends Integrated Gradients to an expectation over background baselines. Works with any differentiable model including Transformers. Approximate efficiency.

### DeepLIFT / DeepLIFT SHAP
Propagation-based method using activation differences instead of gradients. Handles saturation correctly. Exact efficiency guarantee. Requires standard layer types (ReLU, sigmoid, linear).

## Method Selection Quick Reference

| Model type | Recommended method |
|-----------|-------------------|
| Any black-box model | `KernelShap` |
| Standard MLP/CNN (ReLU) | `DeepLiftShap` |
| Transformer/BERT/GPT | `GradientShap` |
| Model with custom activations | `GradientShap` |
| Need exact Shapley guarantees | `KernelShap` (large n_samples) |

## Running the Exercises

```bash
# Run self-check exercises
python modules/module_05_shap_methods/exercises/01_shap_self_check.py

# Launch notebooks
jupyter notebook modules/module_05_shap_methods/notebooks/
```

## Next Module

**Module 06:** Concept-based explanations (TCAV) and influence functions (TracIn).
