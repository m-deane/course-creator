# Module 02: Estimation Methods for Dynamic Factor Models

## Overview

Learn the three main approaches for estimating DFM parameters: maximum likelihood (via Kalman filter), EM algorithm, and Bayesian methods. In 2 hours, you'll implement each method and understand their trade-offs.

**Time Commitment:** 2 hours
**Difficulty:** Advanced
**Prerequisites:** Modules 00-01 (State Space Models, DFMs, Kalman Filter)

## Why This Matters

Estimation methods determine:
- **Accuracy** of factor extraction
- **Computational speed** (critical for real-time systems)
- **Uncertainty quantification** (confidence intervals, posterior distributions)
- **Flexibility** for complex models (mixed frequencies, time-varying parameters)

Central banks, hedge funds, and tech companies use these methods daily for nowcasting and forecasting.

## Learning Objectives

By the end of this module, you will:

1. **Implement** maximum likelihood estimation using prediction error decomposition
2. **Code** the EM algorithm for DFM parameter estimation
3. **Apply** Bayesian methods with MCMC sampling
4. **Compare** computational efficiency across methods
5. **Diagnose** estimation quality using likelihood and diagnostics

## Module Contents

### Guides (Read First)
1. **[Maximum Likelihood](guides/01_maximum_likelihood.md)** - ML via Kalman filter (25 min)
2. **[EM Algorithm](guides/02_em_algorithm.md)** - Expectation-maximization (25 min)
3. **[Bayesian Estimation](guides/03_bayesian_estimation.md)** - MCMC and Gibbs sampling (25 min)
4. **[Cheatsheet](guides/cheatsheet.md)** - Quick reference for estimation methods

### Notebooks (Hands-On)
1. **[ML Estimation](notebooks/01_ml_estimation.ipynb)** - Maximum likelihood implementation (15 min)
2. **[EM Algorithm](notebooks/02_em_algorithm.ipynb)** - EM from scratch (15 min)

### Practice
- **[Self-Check Exercises](exercises/exercises.py)** - Test your understanding (ungraded)

### Resources
- **[Additional Readings](resources/additional_readings.md)** - Papers on DFM estimation
- **[Figures](resources/figures/)** - Visual assets

## Recommended Path

### Fast Track (1 hour)
1. Skim [Maximum Likelihood guide](guides/01_maximum_likelihood.md)
2. Run [ML Estimation notebook](notebooks/01_ml_estimation.ipynb)
3. Check [Cheatsheet](guides/cheatsheet.md)

### Deep Dive (2-3 hours)
1. Read all three guides
2. Work through both notebooks
3. Complete self-check exercises
4. Implement Bayesian version yourself

## Key Concepts

- **Prediction Error Decomposition** - Likelihood from Kalman filter innovations
- **EM Algorithm** - Iterative parameter estimation (E-step: Kalman smoother, M-step: update parameters)
- **Gibbs Sampling** - Draw parameters sequentially from conditional posteriors
- **Identification in Estimation** - Constraints needed for unique solution

## Common Questions

**Q: Which method is best?**
A: Depends on your goal:
- ML: Fast, standard errors available, industry standard
- EM: Handles missing data naturally, numerically stable
- Bayesian: Full uncertainty quantification, flexible priors

**Q: How long does estimation take?**
A: ML/EM: seconds to minutes. Bayesian MCMC: minutes to hours. Scales with N, T, r.

**Q: Can I estimate with missing data?**
A: Yes! All three methods handle missing data naturally.

## Next Steps

After this module:
- **Module 03:** Applications (nowcasting, forecasting, structural analysis)
- **Module 04:** Extensions (mixed frequency, time-varying parameters)
- **Project 2:** Build a real-time nowcasting system with parameter estimation
