# Module 5: Advanced Topics

## Overview

Address real-world complications: clustered standard errors, serial correlation, dynamic panels, and instrumental variables approaches like GMM.

**Time Estimate:** 8-10 hours

## Learning Objectives

By completing this module, you will:
1. Compute cluster-robust standard errors
2. Test and correct for serial correlation
3. Estimate dynamic panel models with lagged dependent variables
4. Apply Arellano-Bond GMM estimation

## Contents

### Guides
- `01_clustered_errors.md` - Robust inference
- `02_serial_correlation.md` - AR errors in panels
- `03_dynamic_panels.md` - Lagged dependent variables
- `04_gmm_estimation.md` - Arellano-Bond and system GMM

### Notebooks
- `01_robust_inference.ipynb` - Clustered SEs in practice
- `02_dynamic_models.ipynb` - GMM implementation

## Key Concepts

### Clustered Standard Errors

Standard errors clustered by entity account for:
- Within-entity correlation of errors
- Heteroskedasticity across entities

$$\hat{V}_{cluster} = (X'X)^{-1}\left(\sum_{i=1}^N X_i'\hat{u}_i\hat{u}_i'X_i\right)(X'X)^{-1}$$

### Dynamic Panel Model

$$y_{it} = \gamma y_{i,t-1} + X_{it}\beta + \alpha_i + \epsilon_{it}$$

**Problem**: $y_{i,t-1}$ is correlated with $\alpha_i$
**Solution**: Arellano-Bond GMM using lagged levels as instruments for differences

### Arellano-Bond Estimator

First-difference to eliminate fixed effects:
$$\Delta y_{it} = \gamma \Delta y_{i,t-1} + \Delta X_{it}\beta + \Delta\epsilon_{it}$$

Use $y_{i,t-2}, y_{i,t-3}, ...$ as instruments for $\Delta y_{i,t-1}$

### Key Tests

| Test | Purpose |
|------|---------|
| Wooldridge | Serial correlation in FE models |
| Sargan/Hansen | Overidentification (GMM validity) |
| Arellano-Bond AR(2) | No serial correlation in differences |

## Prerequisites

- Module 0-4 completed
- IV/2SLS understanding helpful
