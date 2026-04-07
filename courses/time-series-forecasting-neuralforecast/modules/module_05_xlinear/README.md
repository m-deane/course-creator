# Module 5: DLinear — The Linear Baseline That Beats Transformers

**Course:** Modern Time Series Forecasting with NeuralForecast

---

## Overview

DLinear (Zeng et al., AAAI 2023) demonstrates that a simple decomposition-plus-linear model can match or outperform many Transformer architectures on standard time series benchmarks. This module covers DLinear's trend/remainder decomposition architecture, why it works so well as a baseline, and practical guidance for training and benchmarking against NHITS.

**Prerequisites:** Modules 0–4 (NeuralForecast API, point forecasting, NHITS)

**Dataset:** ETTm1 (Electricity Transformer Temperature, 7 variables, 15-min intervals)

---

## Learning Objectives

After completing this module you will be able to:

1. Describe DLinear's trend/remainder decomposition and explain why separate linear layers help
2. Explain why a simple linear model can compete with Transformers on time series benchmarks
3. Train DLinear on a benchmark dataset and evaluate using MAE/MSE with utilsforecast
4. Benchmark DLinear against NHITS using `.cross_validation()` and interpret per-series results
5. Use DLinear as a mandatory baseline before deploying more complex models
6. Select hyperparameters (`input_size`, `learning_rate`, `max_steps`) for DLinear

---

## Contents

```
module_05_dlinear/
├── guides/
│   ├── 01_dlinear_architecture.md          # Architecture deep-dive: 4 components + RevIN
│   ├── 01_dlinear_architecture_slides.md   # 15-slide Marp deck (companion to guide 01)
│   ├── 02_multivariate_forecasting.md      # n_series, exogenous features, hyperparameter tuning
│   └── 02_multivariate_forecasting_slides.md  # 14-slide Marp deck (companion to guide 02)
├── notebooks/
│   ├── 01_training_dlinear.ipynb           # Train DLinear on ETTm1, evaluate, plot (<15 min)
│   └── 02_benchmarking.ipynb              # DLinear vs. NHITS head-to-head (<15 min)
├── exercises/
│   └── 01_dlinear_exercises.py            # Self-check: shapes, MAE comparison, hidden_size ablation
└── resources/                             # (empty — add figures or paper PDFs here)
```

---

## Quick Start

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import DLinear
from datasetsforecast.long_horizon import LongHorizon

Y_df, _, _ = LongHorizon.load(directory="data", group="ETTm1")

model = DLinear(
    h=96, input_size=96,
    learning_rate=1e-3, batch_size=32, max_steps=1000,
)

nf = NeuralForecast(models=[model], freq="15min")
cv_df = nf.cross_validation(df=Y_df, val_size=11520, test_size=11520)
```

---

## Architecture Summary

```
Input (B × L)
    → Moving average decomposition
        → Trend component → Linear Layer (L → H)
        → Remainder component → Linear Layer (L → H)
    → Sum trend + remainder forecasts
Output (B × H)
```

**Key parameters:**
- `h` — forecast horizon
- `input_size` — lookback window length
- `learning_rate` — Adam learning rate (default 1e-3)
- `max_steps` — training iterations
- `scaler_type` — input normalization ("standard", "robust", or None)

---

## Benchmark Results (ETTm1 h=96)

| Model | MAE | MSE |
|---|---|---|
| DLinear | 0.355 | 0.316 |
| NHITS | 0.380 | 0.345 |
| TiDE | 0.387 | 0.364 |
| TSMixer | 0.373 | 0.351 |
| PatchTST | 0.367 | 0.329 |

Source: datasciencewithmarco.com, DLinear review

---

## Suggested Learning Path

1. Read `guides/01_dlinear_architecture.md` — full architecture walkthrough
2. Run `notebooks/01_training_dlinear.ipynb` — get real benchmark numbers
3. Read `guides/02_multivariate_forecasting.md` — n_series, exogenous features, tuning
4. Run `notebooks/02_benchmarking.ipynb` — head-to-head DLinear vs. NHITS
5. Complete `exercises/01_dlinear_exercises.py` — verify understanding

**Slide decks** (`*_slides.md`) are companion materials for each guide — use them for review or instructor delivery.

---

## Installation

```bash
pip install neuralforecast datasetsforecast utilsforecast matplotlib pandas
```

Tested with: `neuralforecast>=1.7`, `datasetsforecast>=0.0.8`, Python 3.9+
