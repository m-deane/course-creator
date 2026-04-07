# Module 04: Explainability for Neural Forecasting Models

## Overview

Neural forecasting models like NHITS produce accurate forecasts but provide no built-in explanation of which inputs drove each prediction. This module covers three attribution methods — Integrated Gradients, Input x Gradient, and Shapley Value Sampling — that answer the question "Why did the model forecast this value?"

> **Note:** NeuralForecast does not natively support model explainability. There is no `.explain()` method in the NeuralForecast API. For interpretability, use [Captum](https://captum.ai/) with the underlying PyTorch models, or use inherently interpretable models like NHITS which provide basis function decompositions.

By the end of this module, you can:
- Use Captum to compute attributions on trained NeuralForecast models' underlying PyTorch networks
- Interpret attribution tensors to extract lag and feature importance
- Visualize attributions as heatmaps, waterfall plots, and bar charts
- Validate whether a model's learned attributions match business expectations
- Translate attribution numbers into a stakeholder-ready narrative

## Prerequisites

- Module 01: Point Forecasting with NeuralForecast (NHITS API)
- `pip install neuralforecast captum shap`

## Contents

### Guides

| File | Contents | Time |
|---|---|---|
| `guides/01_explainability_methods.md` | Three attribution methods: theory, math, code, trade-offs | 15 min |
| `guides/01_explainability_methods_slides.md` | 13-slide companion deck | — |
| `guides/02_interpreting_attributions.md` | Tensor shapes, visualization techniques, business narrative | 15 min |
| `guides/02_interpreting_attributions_slides.md` | 13-slide companion deck | — |

### Notebooks

| File | Contents | Time |
|---|---|---|
| `notebooks/01_explain_api.ipynb` | End-to-end: generate data, train NHITS, call .explain(), heatmap | 12–15 min |
| `notebooks/02_attribution_visualization.ipynb` | Compare all three methods, waterfall plot, feature importance, business narrative | 12–15 min |

### Exercises

| File | Contents |
|---|---|
| `exercises/01_explainability_exercises.py` | Self-check: verify dict keys, extract tensor shape, find dominant lag |

## Three Attribution Methods at a Glance

| Method | Passes | Additive | ReLU Safe | Use When |
|---|---|---|---|---|
| Integrated Gradients | 20–300 | Yes | Yes | Default choice |
| Input × Gradient | 1 | No | No | Latency-constrained production |
| Shapley Value Sampling | 100–1000 | Yes | Yes | Compliance audit |

## Key API Pattern

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MSE, MAE
from captum.attr import IntegratedGradients

models = [NHITS(
    h=28, input_size=56,
    futr_exog_list=["published", "is_holiday"],
    max_steps=1000, loss=MSE(), valid_loss=MAE()
)]
nf = NeuralForecast(models=models, freq="D")
nf.fit(df=train, val_size=28)

# Access underlying PyTorch model for explainability
pytorch_model = nf.models[0]
ig = IntegratedGradients(pytorch_model)
# attributions = ig.attribute(input_tensor, baselines=baseline_tensor)
# See Captum docs for full usage: https://captum.ai/
```

## Suggested Learning Path

1. Read `guides/01_explainability_methods.md`
2. Run `notebooks/01_explain_api.ipynb`
3. Read `guides/02_interpreting_attributions.md`
4. Run `notebooks/02_attribution_visualization.ipynb`
5. Complete `exercises/01_explainability_exercises.py`

## Next Module

Module 05: DLinear Models — linear models with neural feature extraction for interpretable forecasting baselines.
