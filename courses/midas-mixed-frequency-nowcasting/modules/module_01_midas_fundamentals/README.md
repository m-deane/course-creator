# Module 01: MIDAS Regression Fundamentals

## Overview

This module introduces the MIDAS regression model: the equation, the weight functions that parameterize lag structure, and U-MIDAS as an unrestricted alternative. By the end, you can estimate a MIDAS model from scratch and compare it to baseline aggregation approaches.

## Learning Objectives

By the end of this module, you will be able to:

1. Write the MIDAS equation and explain the role of each component
2. Implement Beta polynomial and Almon polynomial weight functions
3. Build a MIDAS data matrix from quarterly and monthly series
4. Estimate MIDAS by NLS using scipy.optimize
5. Interpret estimated weight parameters economically
6. Compare MIDAS, U-MIDAS, and OLS-aggregate using AIC/BIC

## Structure

```
module_01_midas_fundamentals/
├── guides/
│   ├── 01_midas_equation_guide.md         # The MIDAS model: notation, derivation, worked example
│   ├── 01_midas_equation_slides.md        # 16-slide companion deck
│   ├── 02_weight_functions_guide.md       # Beta, Almon, step function families
│   ├── 02_weight_functions_slides.md      # 16-slide companion deck
│   ├── 03_umidas_guide.md                 # U-MIDAS: when unrestricted wins
│   └── 03_umidas_slides.md                # 16-slide companion deck
├── notebooks/
│   ├── 01_basic_midas_regression.ipynb    # Step-by-step MIDAS estimation (~15 min)
│   ├── 02_weight_function_comparison.ipynb # Beta vs Almon vs U-MIDAS (~15 min)
│   └── 03_lag_polynomial_visualization.ipynb # R² surface, weight evolution (~10 min)
└── exercises/
    └── 01_midas_fundamentals_self_check.py  # 4-exercise self-check
```

## Key Equations

**MIDAS model:**
$$y_t = \alpha + \beta \sum_{j=0}^{K-1} w_j(\theta) \cdot x_{mt-j} + \varepsilon_t$$

**Beta polynomial weights:**
$$w_j(\theta_1, \theta_2) = \frac{f_{\text{Beta}}\left(\frac{j+0.5}{K}; \theta_1, \theta_2\right)}{\sum_l f_{\text{Beta}}\left(\frac{l+0.5}{K}; \theta_1, \theta_2\right)}$$

**Almon polynomial weights:**
$$w_j(\theta_1, \theta_2) = \frac{e^{\theta_1 j + \theta_2 j^2}}{\sum_l e^{\theta_1 l + \theta_2 l^2}}$$

**Normalization constraint:** $\sum_{j=0}^{K-1} w_j(\theta) = 1$

## Implementation Pattern

```python
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist
import numpy as np

def beta_weights(K, t1, t2):
    x = (np.arange(K) + 0.5) / K
    raw = beta_dist.pdf(1 - x, t1, t2)
    return raw / raw.sum()

def midas_sse(params, Y, X):
    alpha, beta, t1, t2 = params
    w = beta_weights(X.shape[1], t1, t2)
    y_hat = alpha + beta * (X @ w)
    return np.sum((Y - y_hat)**2)

result = minimize(midas_sse, [0.5, 0.3, 1.0, 5.0], args=(Y, X), method='Nelder-Mead')
```

## Reading Order

1. Guide 01 (MIDAS equation) → Notebook 01 (basic MIDAS)
2. Guide 02 (weight functions) → Notebook 02 (weight comparison)
3. Guide 03 (U-MIDAS) → Notebook 03 (lag polynomial visualization)
4. Exercise (self-check)

## Data Used

Same datasets as Module 00:
- `module_00_foundations/resources/gdp_quarterly.csv`
- `module_00_foundations/resources/industrial_production_monthly.csv`

## Connection to Course Arc

This module teaches the MIDAS model itself. Module 02 provides rigorous estimation and inference. Module 03 applies MIDAS to real-time nowcasting. Module 04 extends to dynamic factor models.
