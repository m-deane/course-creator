# Modern Time Series Forecasting with NeuralForecast

**Sample Paths, Explainability & State-of-the-Art Architectures**

## Course Overview

Master probabilistic time series forecasting using the neuralforecast ecosystem. Learn why marginal quantile forecasts fail for multi-period decisions, how sample paths provide the correct uncertainty framework, and how to explain neural forecast predictions to stakeholders.

## Central Insight

> **"The sum of quantiles is NOT the quantiles of the sum."**

Marginal prediction intervals treat each future timestep independently. When you need to answer questions like "How many units do I need for the whole week?" or "When should I reorder?", marginal quantiles give the **wrong answer**. Sample paths — draws from the joint forecast distribution — are the correct tool.

## Target Audience

Data scientists, ML engineers, and quantitative analysts who have basic Python and time series knowledge and want to master:
- Probabilistic forecasting with neural models
- Uncertainty quantification via sample paths
- Model explainability for stakeholder communication
- State-of-the-art architectures (DLinear)

## Course Structure

| Module | Topic | Key Concept |
|--------|-------|-------------|
| 0 | Foundations & Prerequisites | neuralforecast ecosystem |
| 1 | Point Forecasting | NHITS training and evaluation |
| 2 | Probabilistic Forecasting | Why quantiles aren't enough |
| 3 | **Sample Paths** | The correct uncertainty framework |
| 4 | Explainability | Attribution methods for neural forecasts |
| 5 | DLinear | State-of-the-art MLP architecture |
| 6 | Production Patterns | End-to-end pipelines |
| 7 | Portfolio Project | Build your own forecasting system |

## Quick Start

See `quick-starts/01_first_forecast.ipynb` — install neuralforecast and produce a forecast in under 2 minutes.

## Key Libraries

- **neuralforecast** (v1.7+) — neural forecasting models, `.fit()`, `.predict()`, `.cross_validation()`
- **datasetsforecast** — benchmark dataset loading (ETTm1, etc.)
- **utilsforecast** — evaluation metrics (MAE, MSE)
- **captum** — PyTorch interpretability backend
- **shap** — attribution visualization

## Datasets

All datasets are real — no mock or synthetic data:

1. **French Bakery Daily Sales** (Kaggle) — baguette sales, daily frequency
2. **Blog Traffic** — daily visitor counts with exogenous features (published, is_holiday)
3. **ETTm1** — 7-variable electricity transformer temperature, 15-minute intervals

## Source Articles

This course is based on:
1. "Use Sample Paths Instead of Quantiles" — minimizeregret.com
2. "Sample Paths for Uncertainty Quantification in Time Series Forecasting" — datasciencewithmarco.com
3. "Explainability for Deep Learning Models in Time Series Forecasting" — datasciencewithmarco.com
4. "Discover DLinear for State-of-the-Art Forecasting Performance" — datasciencewithmarco.com

## Directory Structure

```
time-series-forecasting-neuralforecast/
├── modules/
│   ├── module_00_foundations/
│   ├── module_01_point_forecasting/
│   ├── module_02_probabilistic_forecasting/
│   ├── module_03_sample_paths/
│   ├── module_04_explainability/
│   ├── module_05_dlinear/
│   ├── module_06_production_patterns/
│   └── module_07_portfolio_project/
├── quick-starts/          # <2 min entry-point notebooks
├── templates/             # Production-ready Python scaffolds
├── recipes/               # Copy-paste code patterns
└── projects/              # Portfolio project resources
```
