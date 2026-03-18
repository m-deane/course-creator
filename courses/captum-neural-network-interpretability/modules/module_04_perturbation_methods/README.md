# Module 04: Perturbation Methods

## Overview

Perturbation methods are the gradient-free side of neural network interpretability. Instead of backpropagating to measure sensitivity, they directly observe what happens to model output when features are masked, occluded, or shuffled. This makes them applicable to any model — differentiable or not — and gives them a natural causal interpretation: "what would happen if this feature were absent?"

This module covers three methods at increasing levels of theoretical rigor:
1. **Occlusion** — sliding-window perturbation for images
2. **Feature Ablation** — generalized ablation for any feature structure
3. **Shapley Value Sampling** — game-theoretic attribution satisfying four fairness axioms

## Learning Objectives

By completing this module, you will:

1. Apply `Occlusion` to produce image heatmaps using only forward passes
2. Distinguish positive attribution (region is helpful) from negative attribution (region is confusing)
3. Apply `FeatureAblation` with superpixel masks and tabular feature groups
4. Explain the four Shapley axioms: Efficiency, Symmetry, Dummy, and Additivity
5. Apply `ShapleyValueSampling` and verify the Efficiency (completeness) property
6. Compare perturbation methods with gradient methods on the same predictions

## Prerequisites

- Module 02 (Integrated Gradients) — for comparison with IG
- Module 01 Notebook 02 (tabular gradient methods) — for context on wine quality model
- scikit-learn (StandardScaler, train_test_split) — used in Notebook 02 and 03

## Estimated Time

- Guides: 45 minutes (2 guides × ~22 min each)
- Notebooks: 42 minutes (14 + 15 + 13 min)
- Exercise: 25 minutes
- **Total: ~112 minutes**

## Contents

### Guides

| File | Topic | Time |
|------|-------|------|
| `guides/01_ablation_occlusion_guide.md` | Occlusion algorithm, FeatureAblation, superpixels, tabular FA | 22 min |
| `guides/01_ablation_occlusion_slides.md` | Companion slide deck (13 slides) | — |
| `guides/02_shapley_permutation_guide.md` | Shapley axioms, Monte Carlo sampling, Permutation Importance | 23 min |
| `guides/02_shapley_permutation_slides.md` | Companion slide deck (14 slides) | — |

### Notebooks

| File | Topic | Time |
|------|-------|------|
| `notebooks/01_occlusion_image.ipynb` | Occlusion resolution comparison, signed attribution, vs IG | 14 min |
| `notebooks/02_feature_ablation_tabular.ipynb` | Wine quality MLP, per-sample and global FA | 15 min |
| `notebooks/03_shapley_sampling.ipynb` | Shapley convergence, Efficiency verification, 3-method compare | 13 min |

### Exercises

| File | Topic | Questions |
|------|-------|-----------|
| `exercises/01_perturbation_self_check.py` | Occlusion, FA, Shapley axioms, API usage | 24 |

## Key Formulas

**Perturbation Attribution:**
$$\text{Importance}(F_i) = f(x) - f(x \text{ with } F_i \text{ replaced by baseline})$$

**Shapley Value:**
$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(n-|S|-1)!}{n!} \left[v(S \cup \{i\}) - v(S)\right]$$

**Shapley Efficiency (Completeness):**
$$\sum_i \phi_i = f(x) - f(x')$$

## Captum API Quick Reference

```python
from captum.attr import Occlusion, FeatureAblation, ShapleyValueSampling

# Occlusion (image)
occ = Occlusion(model)
attr = occ.attribute(
    input_tensor,
    strides=(3, 8, 8),
    target=class_idx,
    sliding_window_shapes=(3, 15, 15),
    baselines=0
)

# Feature Ablation (tabular or image with mask)
fa = FeatureAblation(model)
attr = fa.attribute(input_tensor, baselines=X_mean, target=class_idx)
# With superpixel mask:
attr = fa.attribute(input_tensor, baselines=0, feature_mask=mask, target=class_idx)

# Shapley Value Sampling
svs = ShapleyValueSampling(model)
attr = svs.attribute(
    input_tensor, baselines=baseline,
    target=class_idx, n_samples=200
)
# attr.sum() ≈ f(x) - f(baseline)  [Efficiency axiom]
```

## Method Comparison Summary

| | Occlusion | Feature Ablation | Shapley Values | IG |
|--|--|--|--|--|
| Model-agnostic | Yes | Yes | Yes | No |
| Axioms | None | None | 4 (Eff, Sym, Dum, Add) | 2 (Eff, Sen) |
| Forward passes | HW/stride² | n_groups | n_feat × n_samp | n_steps |
| Interactions | Partial | Partial | Full | No |
| Best for | Images | Groups/tabular | Fair tabular | Fast pixel |

## Shapley Axioms

1. **Efficiency:** sum(phi_i) = f(x) - f(baseline)
2. **Symmetry:** equal contributors get equal attribution
3. **Dummy:** irrelevant features get zero attribution
4. **Additivity:** attribution is additive across independent submodels

Shapley values are the **unique** attribution satisfying all four.

## Next

**Quick-starts:** Apply the full attribution toolkit to image, text, and tabular data in under 2 minutes each — copy-paste ready code for production use.
