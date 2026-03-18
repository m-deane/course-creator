# Module 04: Dynamic Factor Models

**Mixed-Frequency Models: MIDAS Regression and Nowcasting**

## Overview

This module extends MIDAS to multi-indicator nowcasting via Dynamic Factor Models (DFMs). When many monthly indicators are available, DFMs extract a small number of common factors that summarize comovement. Factor-Augmented MIDAS (FA-MIDAS) uses the extracted factor as the MIDAS predictor, combining noise reduction (from the factor) with timing flexibility (from the Beta polynomial weight function).

## Learning Objectives

1. Extract common factors from a panel of monthly indicators via PCA
2. Select the number of factors using the Bai-Ng IC_p2 criterion
3. Interpret factor loadings as indicator-cycle correlations
4. Build a Factor-Augmented MIDAS (FA-MIDAS) model and evaluate its RMSE
5. Compare FA-MIDAS to single-indicator MIDAS and AR(1) benchmarks
6. Implement sign normalization for consistent expanding-window estimation

## Directory Structure

```
module_04_dynamic_factor_models/
├── guides/
│   ├── 01_factor_models_guide.md         # PCA factors, Bai-Ng, static DFM
│   ├── 01_factor_models_slides.md        # 13-slide companion deck
│   ├── 02_dfm_mixed_frequency_guide.md   # State-space, Kalman filter, MF-DFM
│   ├── 02_dfm_mixed_frequency_slides.md  # 13-slide companion deck
│   ├── 03_factor_augmented_midas_guide.md  # FA-MIDAS implementation
│   └── 03_factor_augmented_midas_slides.md # 13-slide companion deck
├── notebooks/
│   ├── 01_small_dfm.ipynb              # PCA extraction, scree plot, Bai-Ng
│   ├── 02_factor_augmented_midas.ipynb # FA-MIDAS estimation and RMSE comparison
│   └── 03_factor_extraction.ipynb      # Deep dive: loadings, stability, news
├── exercises/
│   └── 01_dfm_self_check.py            # Self-check exercises (5 topics)
└── resources/
    (uses module_00_foundations/resources/ CSVs)
```

## Key Formulas

**Factor model:**
$$\mathbf{x}_t = \mathbf{\Lambda}\mathbf{f}_t + \mathbf{e}_t, \quad t = 1,\ldots,T$$

**PCA estimator:** Top-$q$ eigenvectors of $\frac{1}{T}\mathbf{X}^\top\mathbf{X}$

**Bai-Ng IC_p2:**
$$IC_{p2}(q) = \log\hat{\sigma}^2(q) + q \cdot \frac{N+T}{NT}\log\min(N,T)$$

**FA-MIDAS:**
$$y_t = \alpha + \beta\sum_{j=0}^{K-1} w_j(\theta)\hat{f}_{t-j} + \varepsilon_t$$

## Sign Normalization Rule

At each expanding-window step, normalize the extracted factor so the reference indicator (IP) has a positive loading:

```python
if Lambda[ip_col, 0] < 0:
    F[:, 0] = -F[:, 0]
    Lambda[:, 0] = -Lambda[:, 0]
```

Without this, the factor can flip sign mid-sample and invalidate the regression.

## Connections

- **Builds on:** Modules 01-03 (MIDAS, estimation, nowcasting)
- **Related to:** Stock-Watson (2002) FAVAR, Giannone-Reichlin-Small (2008) nowcasting
