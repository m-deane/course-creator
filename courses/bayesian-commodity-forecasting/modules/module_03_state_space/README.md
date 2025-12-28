# Module 3: Bayesian State Space Models

## Overview

State space models are the workhorses of time series forecasting. They decompose observations into latent states (trends, seasonality, cycles) and provide a natural Bayesian framework for sequential learning. This module covers the theory and implementation of state space models for commodity price dynamics.

**Time Estimate:** 8-10 hours

## Learning Objectives

By completing this module, you will:
1. **Understand** the state space formulation and its components
2. **Apply** the Kalman filter as optimal Bayesian inference for linear-Gaussian systems
3. **Implement** local level, local linear trend, and stochastic volatility models
4. **Connect** state space models to ARIMA and exponential smoothing
5. **Evaluate** model fit using posterior predictive checks

## Module Contents

### Guides
- `01_state_space_fundamentals.md` - The general state space framework
- `02_kalman_filter.md` - Optimal filtering for linear-Gaussian systems
- `03_stochastic_volatility.md` - Modeling time-varying uncertainty

### Notebooks
- `01_local_level_model.ipynb` - Random walk plus noise
- `02_local_linear_trend.ipynb` - Trend extraction for commodity prices
- `03_stochastic_volatility_pymc.ipynb` - Bayesian SV models
- `04_multi_step_forecasting.ipynb` - Producing forecasts with uncertainty

### Assessments
- `quiz.md` - State space concepts (15 questions)
- `coding_exercises.py` - Implement Kalman filter from scratch
- `mini_project_rubric.md` - State space model for chosen commodity

### Resources
- `cheatsheet.md` - State space model quick reference
- `additional_readings.md` - Durbin & Koopman selections

## Key Concepts

### The State Space Formulation

**Observation Equation:**
$$y_t = Z_t \alpha_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, H_t)$$

**State Transition Equation:**
$$\alpha_{t+1} = T_t \alpha_t + R_t \eta_t, \quad \eta_t \sim \mathcal{N}(0, Q_t)$$

Where:
- $y_t$: Observed data (commodity prices)
- $\alpha_t$: Latent state (level, trend, seasonality)
- $Z_t, T_t$: System matrices defining model structure
- $\epsilon_t, \eta_t$: Observation and state noise

### Why State Space for Commodities?

1. **Decomposition:** Separate trend from noise for clearer signals
2. **Missing data:** Handle gaps naturally (EIA releases, holidays)
3. **Stochastic volatility:** Model time-varying uncertainty (volatility clustering)
4. **Structural breaks:** Detect and adapt to regime changes
5. **Multi-step forecasts:** Natural extension for h-step ahead predictions

## Model Hierarchy

```
State Space Models
├── Local Level (Random Walk + Noise)
│   └── Simplest: level varies randomly over time
├── Local Linear Trend
│   └── Level + Trend, both stochastic
├── Basic Structural Model (BSM)
│   └── Level + Trend + Seasonality
├── Stochastic Volatility
│   └── Time-varying variance of returns
└── Dynamic Regression
    └── Time-varying coefficients on fundamentals
```

## Completion Criteria

- [ ] Kalman filter implementation passes all tests
- [ ] Quiz score ≥ 80%
- [ ] Stochastic volatility notebook completed
- [ ] Mini-project: State space model for real commodity data

## Prerequisites

- Module 1 completed (Bayesian fundamentals)
- Matrix algebra (state vectors, covariance matrices)
- Basic time series concepts (stationarity, autocorrelation)

## Time Allocation

| Activity | Time |
|----------|------|
| Reading guides | 2 hours |
| Notebook 1-2 (Local Level/Trend) | 2 hours |
| Notebook 3 (Stochastic Volatility) | 2 hours |
| Notebook 4 (Forecasting) | 1.5 hours |
| Quiz + exercises | 1 hour |
| Mini-project | 2 hours |
| **Total** | ~10 hours |

## Connections

**Builds on:** Module 1 (Bayes' theorem, conjugate Normal-Normal updates)

**Leads to:**
- Module 5: GPs as flexible non-parametric state space models
- Module 7: Regime-switching state space models
- Module 8: Dynamic regression with fundamental variables

---

*"State space models view time series through the lens of hidden dynamics. The Kalman filter reveals these dynamics optimally, one observation at a time."*
