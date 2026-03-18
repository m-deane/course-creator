# Module 04: Difference-in-Differences

## Overview

Difference-in-differences (DiD) is the most widely used quasi-experimental design in applied causal inference. It identifies causal effects by comparing before-after changes in a treated group against the same changes in a control group, relying on the parallel trends assumption.

This module covers the canonical two-period DiD, modern staggered adoption estimators, event study visualisation, and CausalPy's Bayesian DiD implementation.

## Contents

### Guides

| File | Topic |
|------|-------|
| `guides/01_did_fundamentals_guide.md` | Parallel trends, two-period DiD, TWFE, identification |
| `guides/01_did_fundamentals_slides.md` | Slide deck companion |
| `guides/02_staggered_did_guide.md` | Staggered adoption, Callaway-Sant'Anna, Sun-Abraham, event studies |
| `guides/02_staggered_did_slides.md` | Slide deck companion |
| `guides/03_causalpy_did_api_guide.md` | CausalPy DifferenceInDifferences class, Bayesian DiD, priors |
| `guides/03_causalpy_did_api_slides.md` | Slide deck companion |

### Notebooks

| File | Topic | Time |
|------|-------|------|
| `notebooks/01_did_labour_economics.ipynb` | Card & Krueger minimum wage DiD | 15 min |
| `notebooks/02_staggered_did.ipynb` | Staggered adoption, TWFE bias, CS estimator | 15 min |
| `notebooks/03_event_study_plots.ipynb` | Event study construction and pre-trend testing | 15 min |

### Exercises

| File | Topic |
|------|-------|
| `exercises/01_did_self_check.py` | Manual DiD, regression DiD, parallel trends, TWFE |

## Prerequisites

- Module 01: ITS Fundamentals (parallel trends intuition)
- Module 03: Synthetic Control (counterfactual thinking)
- Basic panel data concepts (fixed effects)
- Python: pandas, statsmodels, numpy

## Learning Outcomes

After completing this module, you can:

1. Derive and compute the DiD estimator from first principles
2. Explain why parallel trends is the identifying assumption and assess its plausibility
3. Explain why standard TWFE fails under staggered adoption with heterogeneous effects
4. Implement the Callaway-Sant'Anna group-time ATT framework
5. Construct and interpret event study plots with pre-trend tests
6. Run Bayesian DiD using CausalPy's `DifferenceInDifferences` class
7. Choose between DiD and other causal designs based on the research context

## Key Concepts

**Parallel Trends:** The central identifying assumption. Absent treatment, the average outcome of the treated group would have followed the same trend as the control group. Cannot be tested directly but can be assessed using pre-treatment data.

**ATT:** DiD estimates the Average Treatment Effect on the Treated — the causal effect for units that were actually treated, not the general population.

**TWFE Bias:** Standard two-way fixed effects DiD can produce biased estimates under staggered adoption when treatment effects vary across cohorts or over event time. Modern estimators (Callaway-Sant'Anna, Sun-Abraham) avoid this.

**Event Study:** A regression that estimates separate treatment effects for each period relative to treatment onset. The pre-period coefficients test parallel trends; the post-period coefficients show treatment effect dynamics.

## Navigation

- Previous: [Module 03 — Synthetic Control](../module_03_synthetic_control/README.md)
- Next: [Module 05 — Regression Discontinuity](../module_05_regression_discontinuity/README.md)
