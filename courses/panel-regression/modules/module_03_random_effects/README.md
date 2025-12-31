# Module 3: Random Effects Models

## Overview

Random effects treats entity-specific effects as random draws from a distribution. When its assumptions hold, RE is more efficient than FE. Learn when to use it and how to estimate it properly.

**Time Estimate:** 6-8 hours

## Learning Objectives

By completing this module, you will:
1. Understand the random effects model and assumptions
2. Implement GLS estimation for random effects
3. Compare RE efficiency to FE and pooled OLS
4. Apply RE to appropriate research questions

## Contents

### Guides
- `01_random_effects_model.md` - The RE framework
- `02_gls_estimation.md` - Generalized Least Squares
- `03_re_assumptions.md` - When RE is valid

### Notebooks
- `01_re_implementation.ipynb` - RE in Python and R
- `02_re_vs_fe.ipynb` - Comparing estimators

## Key Concepts

### Random Effects Model

$$y_{it} = \alpha + X_{it}\beta + \mu_i + \epsilon_{it}$$

Where:
- $\mu_i \sim N(0, \sigma^2_\mu)$: Random entity effect
- $\epsilon_{it} \sim N(0, \sigma^2_\epsilon)$: Idiosyncratic error
- **Key assumption**: $Cov(\mu_i, X_{it}) = 0$

### GLS Transformation

Define $\theta = 1 - \sqrt{\frac{\sigma^2_\epsilon}{\sigma^2_\epsilon + T\sigma^2_\mu}}$

Transform:
$$y_{it} - \theta\bar{y}_i = (1-\theta)\alpha + (X_{it} - \theta\bar{X}_i)\beta + error$$

### RE vs FE Trade-offs

| Property | Fixed Effects | Random Effects |
|----------|--------------|----------------|
| $Cov(\alpha_i, X)$ | Can be ≠ 0 | Must be = 0 |
| Time-invariant X | Cannot estimate | Can estimate |
| Efficiency | Lower | Higher (if valid) |
| Generalizability | Within sample | To population |

## Prerequisites

- Module 0-2 completed
- Understanding of GLS
