# Module 03: Synthetic Control Methods

## Overview

Synthetic control constructs a counterfactual for a treated unit by forming a weighted combination
of untreated donor units. The weights are chosen so the weighted donor average closely matches the
treated unit's pre-intervention trajectory. Any post-intervention divergence is the estimated
causal effect.

This module covers the full synthetic control workflow: weight optimization, inference via
permutation tests, and Bayesian uncertainty quantification using CausalPy.

---

## Learning Objectives

After completing this module you will be able to:

1. Explain when synthetic control is preferred over ITS (concurrent events, panel data available)
2. Formulate the SC weight optimization problem and solve it using scipy
3. Assess pre-intervention fit quality using the RMSPE diagnostic
4. Run in-space placebo tests and compute the permutation p-value
5. Run in-time placebo tests to validate the pre-period
6. Interpret donor weights (sparsity, posterior uncertainty with CausalPy)
7. Compute the counterfactual gap with Bayesian uncertainty using `cp.SyntheticControl`
8. Combine Bayesian uncertainty quantification with permutation inference for a complete report

---

## Module Structure

```
module_03_synthetic_control/
├── guides/
│   ├── 01_synthetic_control_guide.md      SC theory, assumptions, formal setup
│   ├── 01_synthetic_control_slides.md     Companion Marp slide deck
│   ├── 02_inference_placebo_guide.md      Permutation inference and placebo tests
│   └── 02_inference_placebo_slides.md     Companion Marp slide deck
├── notebooks/
│   ├── 01_synthetic_control_basics.ipynb  Weight optimization from scratch with scipy
│   ├── 02_placebo_tests.ipynb             In-space and in-time placebo tests
│   └── 03_causalpy_synthetic_control.ipynb  Bayesian SC with CausalPy
├── exercises/
│   └── 01_sc_self_check.py               Self-check on SC concepts and implementation
└── README.md
```

---

## Recommended Sequence

| Step | Material | Time |
|------|----------|------|
| 1 | Guide 01: Synthetic Control | 20 min |
| 2 | Slides 01: Synthetic Control | 10 min |
| 3 | Notebook 01: SC Basics | 15 min |
| 4 | Guide 02: Inference and Placebo Tests | 20 min |
| 5 | Slides 02: Inference | 10 min |
| 6 | Notebook 02: Placebo Tests | 15 min |
| 7 | Notebook 03: CausalPy SC | 15 min |
| 8 | Self-Check Exercises | 20 min |

Total estimated time: 2 hours

---

## Prerequisites

- Module 00: Potential outcomes (SUTVA, ATT), DAGs
- Module 01: ITS fundamentals (to understand when SC is preferred)
- Python: NumPy, Pandas, matplotlib at an intermediate level
- scipy: basic familiarity with scipy.optimize.minimize

---

## Key Concepts

### The Synthetic Control Estimator

Choose donor weights $\mathbf{w}$ with $w_j \geq 0$, $\sum_j w_j = 1$ to minimize pre-intervention discrepancy:

$$\mathbf{w}^* = \arg\min_{\mathbf{w}} \left\| \mathbf{X}_1 - \mathbf{X}_0 \mathbf{w} \right\|_V$$

Counterfactual: $\widehat{Y}_{1t}(0) = \sum_j w_j^* Y_{jt}$ for $t > T_0$

Causal effect: $\hat{\alpha}_t = Y_{1t} - \widehat{Y}_{1t}(0)$

### Pre-Period Fit Diagnostic

$$\text{RMSPE}_{\text{pre}} = \sqrt{\frac{1}{T_0} \sum_{t=1}^{T_0} \left(Y_{1t} - \widehat{Y}_{1t}(0)\right)^2}$$

Rule of thumb: RMSPE should be below 20% of the treated unit's pre-period standard deviation.

### Permutation P-Value

$$\text{RMSPE Ratio}_j = \frac{\text{RMSPE}_{\text{post}}(j)}{\text{RMSPE}_{\text{pre}}(j)}$$

$$p = \frac{|\{j : \text{Ratio}(j) \geq \text{Ratio}(\text{treated})\}|}{J + 1}$$

Minimum achievable: $1 / (J + 1)$ where $J$ is the number of donors.

### When to Use Synthetic Control vs ITS

| Use SC when... | Use ITS when... |
|----------------|-----------------|
| Panel data available (multiple donors) | Single unit only |
| Concurrent event threat | No concurrent events |
| Treated unit is non-outlier among donors | No comparable donors |
| Post-intervention period is short (≥1 period) | Long post-period needed |

---

## What's Next

You have completed all four modules of the course. See the **quick-starts** directory for
ready-to-run templates covering ITS, synthetic control, and the full CausalPy API.
