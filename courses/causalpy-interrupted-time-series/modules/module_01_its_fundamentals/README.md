# Module 01: Interrupted Time Series Fundamentals

## Module Overview

This module provides a thorough grounding in ITS methodology — from the conceptual foundations of when and why ITS works, to the technical mechanics of segmented regression, to the practical CausalPy API for fitting and interpreting Bayesian ITS models.

## Learning Objectives

By the end of this module, you will be able to:

1. Determine when ITS is the appropriate causal design and when alternatives are needed
2. Enumerate and assess the core ITS assumptions for a given applied problem
3. Specify the full segmented regression model with level and slope change parameters
4. Detect and correct for autocorrelation in ITS residuals
5. Apply seasonal adjustment using month dummies and Fourier terms
6. Fit a Bayesian ITS model using CausalPy and interpret all output
7. Conduct a complete diagnostic workflow: convergence, residuals, PPC, placebo tests

## Module Structure

```
module_01_its_fundamentals/
├── guides/
│   ├── 01_its_introduction_guide.md       # When/why to use ITS, assumptions, validity threats
│   ├── 01_its_introduction_slides.md      # Companion deck (17 slides)
│   ├── 02_segmented_regression_guide.md   # Model mechanics, autocorrelation, seasonality
│   ├── 02_segmented_regression_slides.md  # Companion deck (16 slides)
│   ├── 03_causalpy_its_api_guide.md       # CausalPy API reference and complete workflow
│   └── 03_causalpy_its_api_slides.md      # Companion deck (16 slides)
├── notebooks/
│   ├── 01_its_smoking_ban.ipynb           # Full ITS workflow on AMI/smoking ban data
│   ├── 02_its_diagnostics.ipynb           # Residuals, PPC, placebo tests, autocorrelation
│   └── 03_its_seasonal_adjustments.ipynb  # Month dummies vs Fourier terms, LOO comparison
├── exercises/
│   └── 01_its_self_check.py               # Appropriateness assessment, output interpretation
└── resources/
```

## Recommended Sequence

1. Read **Guide 1** (ITS Introduction) — 25 min
2. Read **Guide 2** (Segmented Regression) — 30 min
3. Read **Guide 3** (CausalPy API) — 20 min
4. Run **Notebook 1** (Smoking Ban ITS) — 15 min
5. Run **Notebook 2** (ITS Diagnostics) — 15 min
6. Run **Notebook 3** (Seasonal Adjustments) — 15 min
7. Work through **Exercise 1** (Self-Check) — 20 min

**Total estimated time: ~2.5 hours**

## Core Model Reference

The standard ITS segmented regression:

$$Y_t = \alpha + \beta_1 t + \beta_2 D_t + \beta_3 (t - t^*) D_t + \varepsilon_t$$

| Parameter | Interpretation |
|-----------|---------------|
| $\alpha$ | Baseline level at $t=0$ |
| $\beta_1$ | Pre-intervention monthly trend |
| $\beta_2$ | **Level change** at $t^*$ (immediate effect) |
| $\beta_3$ | **Slope change** after $t^*$ (trend change) |

## Validity Checklist

Before reporting ITS results, verify:

- [ ] Pre-period is sufficiently long (12+ observations, ideally 24+)
- [ ] Intervention timing was exogenous (not triggered by the outcome)
- [ ] No major concurrent events at $t^*$
- [ ] No evidence of anticipation effects before $t^*$
- [ ] Seasonal patterns are controlled for if present
- [ ] MCMC has converged (R-hat < 1.01, ESS > 400)
- [ ] Residuals show no systematic patterns
- [ ] Placebo tests show no spurious effects in the pre-period
- [ ] Posterior predictive check passes

## Prerequisites

- Module 00 (causal foundations, potential outcomes, DAGs)
- Module 00 Notebook 01 (environment setup)

## What's Next

**Module 02: Bayesian ITS with PyMC** — Build ITS models from scratch using PyMC, understand how CausalPy constructs the model internally, specify informative priors based on domain knowledge, and conduct prior sensitivity analysis.
