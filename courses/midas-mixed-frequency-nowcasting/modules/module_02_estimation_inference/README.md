# Module 02: Estimation and Inference

**Mixed-Frequency Models: MIDAS Regression and Nowcasting**

## Overview

This module covers estimation and inference for MIDAS regression. By the end, you can estimate Beta MIDAS by profile NLS, select the lag order using BIC and cross-validation, and compute valid HAC standard errors with appropriate hypothesis tests.

## Learning Objectives

1. Derive and implement profile NLS for MIDAS — solve (alpha, beta) analytically for fixed theta
2. Visualize the profile SSE landscape and understand local optima
3. Select the lag order K using AIC/BIC and expanding-window cross-validation
4. Compare Beta, Almon, and U-MIDAS by information criteria
5. Compute Newey-West HAC standard errors using statsmodels
6. Test significance of the IP coefficient (t-test) and the equal-weight restriction (F-test)
7. Construct bootstrap confidence intervals for the weight function

## Directory Structure

```
module_02_estimation_inference/
├── guides/
│   ├── 01_nls_estimation_guide.md       # Profile NLS derivation and implementation
│   ├── 01_nls_estimation_slides.md      # 15-slide companion deck
│   ├── 02_model_selection_guide.md      # AIC/BIC and expanding-window CV
│   ├── 02_model_selection_slides.md     # 15-slide companion deck
│   ├── 03_inference_guide.md            # HAC standard errors and hypothesis tests
│   └── 03_inference_slides.md          # 15-slide companion deck
├── notebooks/
│   ├── 01_nls_optimization.ipynb        # Profile SSE, contour maps, optimization
│   ├── 02_model_selection.ipynb         # AIC/BIC lag selection, expanding-window CV
│   └── 03_robust_inference.ipynb        # HAC SEs, F-test, bootstrap CIs
├── exercises/
│   └── 01_estimation_self_check.py      # Self-check exercises (5 topics)
└── resources/
    (no module-specific resources; uses module_00_foundations/resources/ CSVs)
```

## Notebooks at a Glance

| Notebook | Core Concept | Key Output |
|----------|-------------|------------|
| 01_nls_optimization | Profile SSE contour map; fine optimization | theta estimates, residual diagnostics |
| 02_model_selection | AIC/BIC lag table; expanding-window RMSE | Model selection table; OOS RMSE comparison |
| 03_robust_inference | HAC SEs vs OLS SEs; F-test; bootstrap | Inference table; bootstrap CI plots |

## Key Formulas

**Profile SSE:**
$$Q_{\text{profile}}(\theta) = \min_{\alpha,\beta} \sum_{t=1}^T \left(y_t - \alpha - \beta \tilde{x}_t(\theta)\right)^2$$

Solved analytically: $\hat{\beta}(\theta) = \frac{\text{Cov}(y, \tilde{x}(\theta))}{\text{Var}(\tilde{x}(\theta))}$, $\hat{\alpha}(\theta) = \bar{y} - \hat{\beta}(\theta)\bar{\tilde{x}}$

**Information Criteria:**
$$\text{BIC}(K) = T\ln\!\left(\frac{\text{SSE}(K)}{T}\right) + k(K)\ln T$$

For restricted Beta MIDAS: $k = 4$ regardless of $K$.

**HAC Covariance:**
$$\hat{V}_{HAC} = (Z^\top Z)^{-1}\hat{S}_{NW}(Z^\top Z)^{-1}, \quad L = \lfloor 4(T/100)^{2/9}\rfloor$$

**F-Test (Equal Weights):**
$$F = \frac{(SSE_R - SSE_U)/2}{SSE_U/(T-4)} \sim F_{2,T-4}$$

## Prerequisites

- Module 00: Mixed-frequency data structures, MIDAS matrix construction
- Module 01: MIDAS equation, Beta polynomial weights, weight function shapes

## Data Requirements

Uses CSV files from `module_00_foundations/resources/`:
- `gdp_quarterly.csv` — quarterly GDP growth (100 observations)
- `industrial_production_monthly.csv` — monthly IP growth (~300 observations)

Set `USE_FRED = True` in any notebook to use live FRED data instead.

## Connections

- **Leads to:** Module 03 (Nowcasting) — apply the estimated model to real-time GDP prediction
- **Related to:** Module 04 (Dynamic Factor Models) — inference for factor-augmented MIDAS
