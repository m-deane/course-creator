# Module 02: Bayesian ITS with PyMC

## Overview

This module develops a deep understanding of the Bayesian machinery underlying CausalPy's
interrupted time series analysis. You will build ITS models from scratch in PyMC, understand
how priors affect causal conclusions, and verify model adequacy through posterior predictive checks.

By the end of this module you can specify, fit, diagnose, and report a fully Bayesian ITS
analysis — and defend every modeling choice to a skeptical reviewer.

---

## Learning Objectives

After completing this module you will be able to:

1. Explain why Bayesian inference provides advantages over OLS for ITS (uncertainty propagation, small-sample behavior, prior incorporation)
2. Build the exact PyMC model that CausalPy constructs, manually, from a design matrix
3. Specify weakly informative priors scaled to the pre-intervention outcome distribution
4. Run prior predictive checks and adjust priors based on plausibility
5. Conduct a systematic prior sensitivity analysis and classify results as prior-robust or prior-sensitive
6. Generate posterior predictive samples and compute Bayesian p-values for multiple test statistics
7. Use LOO cross-validation to compare seasonal vs. naive ITS specifications
8. Interpret ArviZ convergence diagnostics (R-hat, ESS, divergences) and apply remedies

---

## Module Structure

```
module_02_bayesian_its/
├── guides/
│   ├── 01_bayesian_its_guide.md           Bayesian ITS theory and advantages
│   ├── 01_bayesian_its_slides.md          Companion Marp slide deck
│   ├── 02_causalpy_pymc_internals_guide.md  CausalPy architecture walkthrough
│   ├── 02_causalpy_pymc_internals_slides.md Companion Marp slide deck
│   ├── 03_prior_specification_guide.md    Prior choice protocol and sensitivity analysis
│   └── 03_prior_specification_slides.md   Companion Marp slide deck
├── notebooks/
│   ├── 01_its_from_scratch_pymc.ipynb     Build the ITS PyMC model manually
│   ├── 02_prior_sensitivity.ipynb         Systematic prior sensitivity analysis
│   └── 03_posterior_predictive_checks.ipynb  PPC workflow with ArviZ
├── exercises/
│   └── 01_bayesian_its_self_check.py      Self-check on priors, PPCs, and convergence
└── README.md
```

---

## Recommended Sequence

| Step | Material | Time |
|------|----------|------|
| 1 | Guide 01: Bayesian ITS | 20 min |
| 2 | Slides 01: Bayesian ITS | 10 min |
| 3 | Guide 02: CausalPy Internals | 20 min |
| 4 | Notebook 01: ITS from Scratch | 15 min |
| 5 | Guide 03: Prior Specification | 20 min |
| 6 | Slides 03: Prior Specification | 10 min |
| 7 | Notebook 02: Prior Sensitivity | 15 min |
| 8 | Notebook 03: Posterior Predictive Checks | 15 min |
| 9 | Self-Check Exercises | 20 min |

Total estimated time: 2.5 hours

---

## Prerequisites

- Module 00: Foundations (potential outcomes, DAGs)
- Module 01: ITS Fundamentals (segmented regression, CausalPy API)
- Python: NumPy, Pandas, Matplotlib at an intermediate level
- Probability: comfortable with Normal distributions and Bayes' theorem

---

## Key Concepts

### The Bayesian ITS Model

$$Y_t \sim \mathcal{N}(\mu_t, \sigma^2)$$
$$\mu_t = \alpha + \beta_1 t + \beta_2 \cdot \text{treated}_t + \beta_3 \cdot t_{\text{post},t}$$

With weakly informative priors scaled to the pre-intervention outcome:

$$\alpha \sim \mathcal{N}(\bar{Y}_{\text{pre}}, 2\sigma_Y)$$
$$\beta_1 \sim \mathcal{N}(0, \sigma_Y / \sqrt{n_{\text{pre}}})$$
$$\beta_2 \sim \mathcal{N}(0, \sigma_Y)$$
$$\beta_3 \sim \mathcal{N}(0, 0.1\sigma_Y)$$
$$\sigma \sim \text{HalfNormal}(\sigma_Y)$$

### Prior Predictive Check

Before fitting, sample from the prior and verify the implied outcome trajectories are
plausible. If trajectories show negative hospital visits, impossible trends, or effects
larger than the full outcome range, tighten the relevant prior.

### Convergence Criteria

| Diagnostic | Pass Threshold |
|-----------|---------------|
| R-hat | < 1.01 for all parameters |
| Bulk ESS | > 400 per chain |
| Tail ESS | > 400 per chain |
| Divergences | 0 (or < 0.1% with target_accept = 0.95) |

### Posterior Predictive Check

For each test statistic $T$:
$$\text{Bayesian p-value} = P(T(y_{\text{rep}}) \geq T(y_{\text{obs}}) \mid y_{\text{obs}})$$

Values in [0.05, 0.95] indicate the model captures that statistic. Flagged statistics
(< 0.05 or > 0.95) indicate systematic misspecification in that dimension.

---

## What's Next

**Module 03 — Synthetic Control Methods**

When ITS is not valid (no long pre-intervention series, concurrent events, anticipation effects),
synthetic control builds a weighted comparison from donor units. Module 03 covers the mechanics,
inference via permutation tests, and implementation with CausalPy.
