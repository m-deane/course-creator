# Module 05 — Machine Learning Extensions

## Overview

This module extends classical MIDAS regression to the high-dimensional and nonlinear settings. When the number of mixed-frequency predictors is large relative to the sample, regularization and tree-based methods outperform classical MIDAS. The module covers regularized MIDAS (Lasso, Ridge, Elastic Net, Group Lasso), random forests, gradient boosting, SHAP-based interpretation, and rigorous forecast comparison methodology.

## Learning Outcomes

After completing this module, you will be able to:

1. Formulate regularized MIDAS as a penalized regression problem and choose among Lasso, Ridge, Elastic Net, and Group Lasso
2. Engineer mixed-frequency features (flat stacking, statistical summaries, PCA embeddings) for tree-based ML models
3. Implement XGBoost and LightGBM nowcasting with proper early stopping and temporal validation
4. Apply SHAP values to produce interpretable, communication-ready ML nowcasts
5. Compare ML and MIDAS using the Diebold-Mariano test and expanding-window evaluation
6. Combine forecasts from multiple models for robust nowcasting

## Contents

### Guides

| File | Topic |
|------|-------|
| `guides/01_regularized_midas_guide.md` | Lasso, Ridge, Elastic Net, Group Lasso for MIDAS |
| `guides/01_regularized_midas_slides.md` | Slide deck companion |
| `guides/02_ml_nowcasting_guide.md` | Random forests, gradient boosting, SHAP, forecast combination |
| `guides/02_ml_nowcasting_slides.md` | Slide deck companion |

### Notebooks

| File | Description | Time |
|------|-------------|------|
| `notebooks/01_lasso_midas.ipynb` | Lasso/Ridge/ElasticNet with regularization path visualisation | 15 min |
| `notebooks/02_xgboost_vs_midas.ipynb` | XGBoost vs MIDAS with DM test and SHAP | 15 min |
| `notebooks/03_feature_engineering.ipynb` | Three feature engineering strategies compared | 12 min |

### Exercises

| File | Description |
|------|-------------|
| `exercises/01_ml_extensions_self_check.py` | Four-task self-check covering regularization, expanding-window evaluation, DM test, and forecast combination |

## Prerequisites

- Module 01: MIDAS Fundamentals (MIDAS design matrix construction)
- Module 02: Estimation and Inference (MIDAS estimation and evaluation)
- Basic familiarity with scikit-learn

## Key Concepts

**Regularized MIDAS**: Lasso selects individual lags; Group Lasso selects entire indicators; Elastic Net balances sparsity and stability.

**Feature engineering**: The critical step for ML nowcasting. Daily series (65+ lags) benefit from PCA compression; monthly series benefit from summary statistics.

**Time-series CV**: Always use expanding window or TimeSeriesSplit — never random k-fold with time series data.

**Diebold-Mariano test**: Formal test for equal predictive accuracy using Newey-West HAC standard errors.

**Forecast combination**: Equal-weight combination of MIDAS and ML forecasts almost always outperforms either alone (forecasting combination puzzle).

## Connection to Other Modules

- **Module 04 (DFMs)**: Dynamic factor models are a parametric alternative to ML for high-dimensional nowcasting
- **Module 06 (Financial)**: ML methods applied to realised volatility and commodity nowcasting
- **Module 08 (Production)**: How to productionise ML nowcasting with proper monitoring

## Quick Start

```python
# Run the self-check exercise to verify understanding
python exercises/01_ml_extensions_self_check.py

# Or open the notebooks in order:
# 1. notebooks/01_lasso_midas.ipynb
# 2. notebooks/02_xgboost_vs_midas.ipynb
# 3. notebooks/03_feature_engineering.ipynb
```
