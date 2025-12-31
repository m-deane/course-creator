# Module 3: Gaussian HMMs

## Overview

Extend HMMs to continuous observations using Gaussian emission distributions. Apply to financial returns and other continuous time series.

**Time Estimate:** 8-10 hours

## Learning Objectives

By completing this module, you will:
1. Define Gaussian emission HMMs
2. Implement EM for Gaussian HMMs
3. Handle multivariate observations
4. Apply to financial data

## Contents

### Guides
- `01_gaussian_emissions.md` - Continuous observation models
- `02_em_gaussian.md` - EM for Gaussian HMMs
- `03_multivariate.md` - Multiple observation dimensions

### Notebooks
- `01_gaussian_hmm.ipynb` - Fitting Gaussian HMMs
- `02_financial_application.ipynb` - Market regime detection

## Key Concepts

### Gaussian Emission Model

Each state $k$ has a Gaussian distribution:

$$b_k(o_t) = \mathcal{N}(o_t | \mu_k, \Sigma_k)$$

For univariate:
$$b_k(o_t) = \frac{1}{\sqrt{2\pi\sigma_k^2}} \exp\left(-\frac{(o_t - \mu_k)^2}{2\sigma_k^2}\right)$$

### Parameters

| Parameter | Description | Dimension |
|-----------|-------------|-----------|
| $\pi$ | Initial state dist | $K$ |
| $A$ | Transition matrix | $K \times K$ |
| $\mu$ | State means | $K \times D$ |
| $\Sigma$ | State covariances | $K \times D \times D$ |

### M-Step Updates

$$\hat{\mu}_k = \frac{\sum_t \gamma_t(k) \cdot o_t}{\sum_t \gamma_t(k)}$$

$$\hat{\Sigma}_k = \frac{\sum_t \gamma_t(k) \cdot (o_t - \hat{\mu}_k)(o_t - \hat{\mu}_k)'}{\sum_t \gamma_t(k)}$$

### Financial Application

```python
from hmmlearn import hmm

# Prepare returns
returns = prices.pct_change().dropna().values.reshape(-1, 1)

# Fit 2-state Gaussian HMM
model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.fit(returns)

# Identify regimes
regimes = model.predict(returns)
```

## Prerequisites

- Module 0-2 completed
- Multivariate statistics
