# Module 2: Dynamic Factor Models

## Overview

This module extends static factor models by introducing dynamics: factors evolve over time following autoregressive processes. You'll learn to express DFMs in state-space form and implement the Kalman filter for factor estimation—the foundation for all subsequent methods.

**Estimated Time:** 8-10 hours
**Prerequisites:** Module 1 (static factor models)

## Learning Objectives

By completing this module, you will be able to:

1. **Formulate** dynamic factor models with factor VAR dynamics
2. **Express** DFMs in state-space representation
3. **Derive** the Kalman filter recursions
4. **Implement** Kalman filter and smoother from scratch
5. **Compute** the likelihood via prediction error decomposition
6. **Extract** filtered and smoothed factor estimates

## Module Contents

### Guides
1. `guides/01_from_static_to_dynamic.md` - Adding temporal dynamics to factors
2. `guides/02_state_space_representation.md` - Measurement and transition equations
3. `guides/03_kalman_filter_derivation.md` - Full derivation with intuition

### Notebooks
1. `notebooks/01_kalman_filter_implementation.ipynb` - From-scratch implementation

### Assessments
- `assessments/quiz_module_02.md` - Conceptual quiz
- `assessments/mini_project_kalman.md` - Implement Kalman filter for AR(2)

## Key Concepts

### Dynamic Factor Model Specification

**Measurement Equation:**
$$X_t = \Lambda F_t + e_t, \quad e_t \sim N(0, \Sigma_e)$$

**Transition Equation (Factor Dynamics):**
$$F_t = \Phi_1 F_{t-1} + \Phi_2 F_{t-2} + ... + \Phi_p F_{t-p} + \eta_t, \quad \eta_t \sim N(0, \Sigma_\eta)$$

### State-Space Form

**Measurement:** $X_t = Z \alpha_t + e_t$
**Transition:** $\alpha_t = T \alpha_{t-1} + R \eta_t$

where $\alpha_t$ is the state vector containing current and lagged factors.

### Kalman Filter

Recursive algorithm producing:
- **Filtered estimate:** $\hat{F}_{t|t} = E[F_t | X_1, ..., X_t]$
- **Prediction:** $\hat{F}_{t+1|t} = E[F_{t+1} | X_1, ..., X_t]$
- **Likelihood:** Via prediction error decomposition

### Kalman Smoother

Full-sample estimates:
- **Smoothed estimate:** $\hat{F}_{t|T} = E[F_t | X_1, ..., X_T]$
- Lower variance than filtered estimates
- Uses all available information

## Mini-Project: Kalman Filter from Scratch

**Objective:** Implement the Kalman filter for a state-space model

**Requirements:**
1. Code the prediction step
2. Code the update step
3. Compute filtered state estimates
4. Compute the log-likelihood
5. Verify against statsmodels implementation

**Deliverable:** Working Python class with test cases

## Connections to Other Modules

| Module | Connection |
|--------|------------|
| Module 1 | Static model = DFM with $\Phi = 0$ |
| Module 3 | PCA as initialization for DFM |
| Module 4 | Kalman smoother for EM algorithm |
| Module 5 | State-space handles mixed frequencies |

## Key Formulas

### Kalman Filter Recursions

**Prediction:**
$$\hat{\alpha}_{t|t-1} = T \hat{\alpha}_{t-1|t-1}$$
$$P_{t|t-1} = T P_{t-1|t-1} T' + R Q R'$$

**Update:**
$$v_t = X_t - Z \hat{\alpha}_{t|t-1}$$ (prediction error)
$$F_t = Z P_{t|t-1} Z' + H$$ (prediction error variance)
$$K_t = P_{t|t-1} Z' F_t^{-1}$$ (Kalman gain)
$$\hat{\alpha}_{t|t} = \hat{\alpha}_{t|t-1} + K_t v_t$$
$$P_{t|t} = P_{t|t-1} - K_t Z P_{t|t-1}$$

### Log-Likelihood

$$\log L = -\frac{TN}{2}\log(2\pi) - \frac{1}{2}\sum_{t=1}^T \left[\log|F_t| + v_t' F_t^{-1} v_t\right]$$

## Reading List

### Required
- Hamilton (1994). *Time Series Analysis*, Chapter 13.
- Harvey (1989). *Forecasting, Structural Time Series Models and the Kalman Filter*, Chapters 3-4.

### Recommended
- Durbin & Koopman (2012). *Time Series Analysis by State Space Methods*, 2nd ed.

## Practical Applications

After this module, you can:
1. Estimate factor dynamics (persistence, mean reversion)
2. Produce optimal forecasts of factors
3. Handle missing observations via Kalman filter
4. Compute likelihood for parameter estimation

## Next Steps

After completing this module:
1. Implement Kalman filter mini-project
2. Verify understanding with conceptual quiz
3. Proceed to Module 3: PCA Estimation

---

*"The Kalman filter is arguably the greatest discovery in statistical estimation of the 20th century." - Adapted from various sources*
