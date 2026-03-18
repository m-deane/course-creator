# Module 05: Regression Discontinuity Designs

## Overview

Regression discontinuity (RDD) exploits sharp threshold rules that determine treatment assignment. When a policy assigns treatment to units above (or below) a cutoff on a running variable, units just above and just below the cutoff provide an as-good-as-random comparison. RDD produces estimates with very high internal validity but limited external validity: the effect is local to the cutoff.

## Contents

### Guides

| File | Topic |
|------|-------|
| `guides/01_rdd_fundamentals_guide.md` | Sharp vs fuzzy RDD, continuity assumption, LATE, threats to validity |
| `guides/01_rdd_fundamentals_slides.md` | Slide deck companion (15 slides) |
| `guides/02_bandwidth_selection_guide.md` | IK optimal bandwidth, sensitivity analysis, polynomial order |
| `guides/02_bandwidth_selection_slides.md` | Slide deck companion (15 slides) |

### Notebooks

| File | Topic | Time |
|------|-------|------|
| `notebooks/01_sharp_rdd.ipynb` | Scholarship cutoff RDD: estimation, density test, balance check | 15 min |
| `notebooks/02_rdd_sensitivity.ipynb` | Bandwidth sensitivity, polynomial comparison, donut RDD, placebo test | 15 min |
| `notebooks/03_causalpy_rdd.ipynb` | CausalPy RegressionDiscontinuity: election infrastructure RDD | 15 min |

### Exercises

| File | Topic |
|------|-------|
| `exercises/01_rdd_self_check.py` | RDD identification, local linear estimation, balance testing, fuzzy vs sharp |

## Prerequisites

- Module 04: DiD (comparison group thinking)
- Basic linear regression
- Python: numpy, pandas, statsmodels

## Learning Outcomes

After completing this module, you can:

1. Determine whether a given policy setting supports an RDD design
2. Distinguish sharp from fuzzy RDD and know which estimator to use
3. Implement local linear RDD estimation with appropriate bandwidth
4. Run the complete set of validity diagnostics: density test, covariate balance, bandwidth sensitivity, placebo cutoffs, donut RDD
5. Use CausalPy's `RegressionDiscontinuity` class for Bayesian RDD
6. Correctly interpret the LATE at the cutoff and communicate its external validity limitations

## Key Concepts

**Running Variable:** The continuous variable that determines treatment assignment through comparison to a threshold.

**Continuity Assumption:** Potential outcomes are continuous in the running variable at the cutoff. Any jump in observed outcomes at the cutoff is attributable to the treatment.

**LATE at Cutoff:** RDD identifies the Local Average Treatment Effect for units at the margin of the threshold — not the ATE for the full population.

**Local Linear Regression:** The preferred estimator. Fits a line on each side of the cutoff within a bandwidth. More robust to boundary bias than global polynomials.

**IK Bandwidth:** The Imbens-Kalyanaraman MSE-optimal bandwidth. The default starting point for bandwidth choice.

**Donut RDD:** Excludes observations closest to the cutoff to test robustness to local manipulation.

## Navigation

- Previous: [Module 04 — Difference-in-Differences](../module_04_difference_in_differences/README.md)
- Next: [Module 06 — Instrumental Variables](../module_06_instrumental_variables/README.md)
