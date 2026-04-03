# Module 04: Explainability for Neural Forecasting Models

## Overview

Neural forecasting models like NHITS produce accurate forecasts but provide no built-in explanation of which inputs drove each prediction. This module covers three attribution methods — Integrated Gradients, Input × Gradient, and Shapley Value Sampling — that answer the question "Why did the model forecast this value?"

By the end of this module, you can:
- Call `.explain()` on any trained NeuralForecast model and parse the attribution tensors
- Decode insample and futr_exog tensor shapes and extract lag and feature attributions
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

models = [NHITS(
    h=28, input_size=56,
    futr_exog_list=["published", "is_holiday"],
    max_steps=1000, loss=MSE(), valid_loss=MAE()
)]
nf = NeuralForecast(models=models, freq="D")
nf.fit(df=train, val_size=28)

fcsts_df, explanations = nf.explain(futr_df=futr_df, explainer="IntegratedGradients")

insample      = explanations["insample"]           # shape: (1, 28, 1, 1, 56, 2)
futr_exog     = explanations["futr_exog"]          # shape: (1, 28, 1, 1, 84, 2)
baseline_pred = explanations["baseline_predictions"]  # shape: (1, 28, 1, 1)
```

## Suggested Learning Path

1. Read `guides/01_explainability_methods.md`
2. Run `notebooks/01_explain_api.ipynb`
3. Read `guides/02_interpreting_attributions.md`
4. Run `notebooks/02_attribution_visualization.ipynb`
5. Complete `exercises/01_explainability_exercises.py`

## Next Module

Module 05: XLinear Models — linear models with neural feature extraction for interpretable forecasting baselines.
