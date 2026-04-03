# Module 5: XLinear — State-of-the-Art Architecture

**Course:** Modern Time Series Forecasting with NeuralForecast

---

## Overview

XLinear achieves state-of-the-art accuracy on long-horizon multivariate forecasting benchmarks without attention mechanisms. This module covers its four-component architecture, the role of reversible instance normalization, and practical guidance for training and tuning on the ETTm1 benchmark dataset.

**Prerequisites:** Modules 0–4 (NeuralForecast API, point forecasting, NHITS)

**Dataset:** ETTm1 (Electricity Transformer Temperature, 7 variables, 15-min intervals)

---

## Learning Objectives

After completing this module you will be able to:

1. Describe XLinear's four components (Embedding, TGM, VGM, Prediction Head) and explain the role each plays
2. Explain why RevIN (reversible instance normalization) improves accuracy on datasets with distributional shift
3. Set `n_series` correctly for multivariate XLinear training and diagnose shape errors
4. Train XLinear on ETTm1 and evaluate using MAE/MSE with utilsforecast
5. Benchmark XLinear against NHITS using `.cross_validation()` and interpret per-series results
6. Select hyperparameters (`hidden_size`, `head_dropout`, `temporal_ff`) using the priority-order tuning workflow

---

## Contents

```
module_05_xlinear/
├── guides/
│   ├── 01_xlinear_architecture.md          # Architecture deep-dive: 4 components + RevIN
│   ├── 01_xlinear_architecture_slides.md   # 15-slide Marp deck (companion to guide 01)
│   ├── 02_multivariate_forecasting.md      # n_series, exogenous features, hyperparameter tuning
│   └── 02_multivariate_forecasting_slides.md  # 14-slide Marp deck (companion to guide 02)
├── notebooks/
│   ├── 01_training_xlinear.ipynb           # Train XLinear on ETTm1, evaluate, plot (<15 min)
│   └── 02_benchmarking.ipynb              # XLinear vs. NHITS head-to-head (<15 min)
├── exercises/
│   └── 01_xlinear_exercises.py            # Self-check: shapes, MAE comparison, hidden_size ablation
└── resources/                             # (empty — add figures or paper PDFs here)
```

---

## Quick Start

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import XLinear
from datasetsforecast.long_horizon import LongHorizon

Y_df, _, _ = LongHorizon.load(directory="data", group="ETTm1")

model = XLinear(
    h=96, input_size=96, n_series=7,
    hidden_size=512, temporal_ff=256, channel_ff=21,
    head_dropout=0.5, embed_dropout=0.2,
    learning_rate=1e-4, batch_size=32, max_steps=2000,
)

nf = NeuralForecast(models=[model], freq="15min")
cv_df = nf.cross_validation(df=Y_df, val_size=11520, test_size=11520)
```

---

## Architecture Summary

```
Input (B × L × N)
    → RevIN normalize
    → Embedding Layer (linear projection + global context tokens + embed_dropout)
    → Time-wise Gating Module (TGM: MLP over time dimension, sigmoid gate)
    → Variate-wise Gating Module (VGM: MLP over channel dimension, sigmoid gate)
    → Prediction Head (FC layer + head_dropout)
    → RevIN denormalize
Output (B × H × N)
```

**Key parameters:**
- `n_series` — must equal the number of unique series in your DataFrame
- `hidden_size` — embedding dimension (largest effect on accuracy)
- `temporal_ff` — TGM MLP capacity (scale proportionally with hidden_size)
- `channel_ff` — VGM MLP capacity (must be >= n_series)
- `head_dropout` — primary regularization control (0.5 for max_steps >= 1500)

---

## Benchmark Results (ETTm1 h=96)

| Model | MAE | MSE |
|---|---|---|
| XLinear | 0.355 | 0.316 |
| NHITS | 0.380 | 0.345 |
| TiDE | 0.387 | 0.364 |
| TSMixer | 0.373 | 0.351 |
| PatchTST | 0.367 | 0.329 |

Source: datasciencewithmarco.com, XLinear review

---

## Suggested Learning Path

1. Read `guides/01_xlinear_architecture.md` — full architecture walkthrough
2. Run `notebooks/01_training_xlinear.ipynb` — get real benchmark numbers
3. Read `guides/02_multivariate_forecasting.md` — n_series, exogenous features, tuning
4. Run `notebooks/02_benchmarking.ipynb` — head-to-head XLinear vs. NHITS
5. Complete `exercises/01_xlinear_exercises.py` — verify understanding

**Slide decks** (`*_slides.md`) are companion materials for each guide — use them for review or instructor delivery.

---

## Installation

```bash
pip install neuralforecast datasetsforecast utilsforecast matplotlib pandas
```

Tested with: `neuralforecast>=1.7`, `datasetsforecast>=0.0.8`, Python 3.9+
