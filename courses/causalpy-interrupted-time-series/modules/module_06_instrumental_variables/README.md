# Module 06: Instrumental Variables and Advanced Designs

## Overview

Instrumental variables (IV) solves endogeneity by exploiting exogenous variation: a variable Z that shifts the treatment X without directly affecting the outcome Y. IV estimates the Local Average Treatment Effect for compliers — those who change their treatment status because of the instrument. This module covers IV fundamentals, weak instrument diagnostics, and how to combine IV with other causal designs.

## Contents

### Guides

| File | Topic |
|------|-------|
| `guides/01_iv_fundamentals_guide.md` | Endogeneity, relevance, exclusion, Wald estimator, LATE, 2SLS |
| `guides/01_iv_fundamentals_slides.md` | Slide deck companion (15 slides with DAGs) |
| `guides/02_advanced_designs_guide.md` | Weak instruments, multiple instruments, ITS+DiD, Fuzzy RDD, design selection |
| `guides/02_advanced_designs_slides.md` | Slide deck companion (15 slides with decision flowchart) |

### Notebooks

| File | Topic | Time |
|------|-------|------|
| `notebooks/01_iv_estimation.ipynb` | Card (1995) returns to education: OLS bias, Wald IV, 2SLS | 15 min |
| `notebooks/02_weak_instruments.ipynb` | Weak instrument simulation: bias, coverage, AR inference | 15 min |
| `notebooks/03_combined_designs.ipynb` | ITS+DiD, fuzzy RDD as IV, design selection guide | 15 min |

### Exercises

| File | Topic |
|------|-------|
| `exercises/01_iv_self_check.py` | Instrument validity, Wald estimator, F-statistic, exclusion violation |

## Prerequisites

- Module 04: DiD (comparison group reasoning)
- Module 05: RDD (fuzzy RDD connects to IV)
- Linear regression (OLS mechanics)
- Python: numpy, pandas, statsmodels

## Learning Outcomes

After completing this module, you can:

1. Explain why OLS fails under endogeneity and what IV solves
2. State and defend the two IV conditions (relevance and exclusion)
3. Compute the Wald estimator from group means
4. Run 2SLS with controls using statsmodels
5. Diagnose weak instruments using the first stage F-statistic
6. Implement ITS+DiD for settings with comparison groups and time series data
7. Recognise fuzzy RDD as a local IV design
8. Apply the design selection flowchart to choose the appropriate estimator

## Key Concepts

**Endogeneity:** When the treatment variable X is correlated with the error term in the outcome equation. Causes OLS to be biased.

**Instrument:** A variable Z that is relevant (predicts X) and satisfies the exclusion restriction (affects Y only through X). Provides exogenous variation in the endogenous treatment.

**Wald Estimator:** The simplest IV formula: reduced form effect / first stage effect. Equals the causal effect under the two IV conditions.

**LATE:** Local Average Treatment Effect. IV identifies the causal effect for compliers — units whose treatment status is determined by the instrument.

**Weak Instruments:** First stage F < 10 means IV is biased toward OLS and confidence intervals are miscalibrated. Always report the first stage F-statistic.

**ITS+DiD:** Combining interrupted time series with a control group. Removes common time trends while preserving the ITS structure. The most robust observational design in this course.

## Navigation

- Previous: [Module 05 — Regression Discontinuity](../module_05_regression_discontinuity/README.md)
- Next: [Module 07 — Production Pipelines](../module_07_production_pipelines/README.md)
