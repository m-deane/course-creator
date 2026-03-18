# Module 00 — Foundations & the Interpretability Landscape

## Overview

This module establishes the conceptual foundation for the entire course. Before applying any attribution method, practitioners need to understand *why* interpretability matters, *what* the method space looks like, and *how* the Captum library organizes that space.

## Learning Objectives

By the end of this module, you will be able to:

1. Articulate why interpretability is required (not optional) for production ML systems
2. Classify any attribution method on three dimensions: intrinsic/post-hoc, local/global, model-specific/agnostic
3. Apply decision rules to choose the appropriate method for a given use case
4. Run a complete Captum attribution pipeline on a pretrained ImageNet model
5. Validate attribution quality using the completeness property

## Contents

### Guides

| File | Topic |
|------|-------|
| `guides/01_why_interpretability_guide.md` | Why interpretability matters: debugging, trust, regulation |
| `guides/01_why_interpretability_slides.md` | Companion slide deck (15 slides) |
| `guides/02_taxonomy_guide.md` | Taxonomy: intrinsic vs post-hoc, local vs global, model-specific vs agnostic |
| `guides/02_taxonomy_slides.md` | Companion deck with mermaid taxonomy diagram |
| `guides/03_captum_overview_guide.md` | Captum library architecture, API, comparison with SHAP/LIME |
| `guides/03_captum_overview_slides.md` | Companion deck |

### Notebooks

| File | Topic | Time |
|------|-------|------|
| `notebooks/01_environment_setup.ipynb` | Install and verify full stack; smoke test attribution pipeline | 10 min |
| `notebooks/02_first_explanation.ipynb` | Explain an ImageNet prediction with IG end-to-end | 12 min |

### Exercises

| File | Topic |
|------|-------|
| `exercises/01_interpretability_thinking.py` | Self-check: classify methods, match scenarios, reason about baselines |

## Prerequisites

- Python 3.9+
- PyTorch 2.0+
- Basic understanding of neural network forward pass
- Familiarity with Python (numpy, matplotlib)

## Setup

```bash
pip install captum torch torchvision transformers matplotlib seaborn Pillow
```

## Key Concepts

**Interpretability taxonomy:**
- Intrinsic vs post-hoc: is the model the explanation, or is the explanation computed afterwards?
- Local vs global: one prediction vs. model-wide behavior
- Model-specific vs agnostic: exploits architecture internals vs. only uses input-output

**Regulatory context:**
- EU AI Act (2024): mandatory for high-risk AI systems
- GDPR Article 22: right to explanation for automated decisions
- SR 11-7: Federal Reserve model risk management guidance

**Captum's unified API:**
```python
method = AnyAttributionMethod(model)
attributions = method.attribute(inputs, target=class_idx)
# attributions.shape == inputs.shape
```

## Connections

**Next:** Module 01 — Gradient-Based Attribution Methods (Saliency, Input×Gradient, Guided Backprop, IG side-by-side)

**Prerequisite for:** All subsequent modules

## Estimated Time

- Reading guides: 45 minutes
- Notebooks: 25 minutes
- Exercise: 20 minutes
- **Total: ~90 minutes**
