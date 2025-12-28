# Module 1: Bayesian Fundamentals for Time Series

## Overview

This module establishes the Bayesian framework that underpins all subsequent modeling. You'll learn to think probabilistically about parameters, update beliefs with data, and implement basic Bayesian models in PyMC.

**Time Estimate:** 6-8 hours

## Learning Objectives

By completing this module, you will:
1. **Understand** Bayes' theorem and its components (prior, likelihood, posterior)
2. **Apply** conjugate priors for analytical posterior derivation
3. **Implement** Bayesian regression models in PyMC
4. **Analyze** the effect of prior choice on posterior inference
5. **Interpret** posterior distributions and credible intervals

## Module Contents

### Guides
- `01_bayes_theorem.md` - The foundation of Bayesian inference
- `02_conjugate_priors.md` - When analytical solutions exist
- `03_bayesian_regression.md` - From frequentist to Bayesian regression

### Notebooks
- `01_bayes_theorem_interactive.ipynb` - Visualizing belief updating
- `02_conjugate_priors_examples.ipynb` - Working through conjugate families
- `03_bayesian_regression_pymc.ipynb` - Your first PyMC regression model
- `04_prior_sensitivity_analysis.ipynb` - How priors affect inference

### Assessments
- `quiz.md` - Conceptual understanding (15 questions)
- `coding_exercises.py` - Auto-graded implementation tasks
- `peer_review_rubric.md` - Template for reviewing peer work

### Resources
- `cheatsheet.md` - Quick reference for common distributions and conjugate pairs
- `additional_readings.md` - Papers and book chapters

## Key Concepts

### The Bayesian Workflow

```
Prior Knowledge → Prior Distribution → p(θ)
                          ↓
                    Observed Data → Likelihood → p(y|θ)
                          ↓
                    Bayes' Theorem
                          ↓
                 Posterior Distribution → p(θ|y)
                          ↓
              Decisions / Predictions
```

### Why Bayesian for Commodities?

1. **Uncertainty Quantification:** We get full distributions, not just point estimates
2. **Prior Integration:** Domain knowledge (storage costs, seasonality) can be encoded
3. **Small Samples:** Fundamental data is sparse; Bayesian methods handle this well
4. **Coherent Updates:** As new data arrives, we update beliefs systematically

## Completion Criteria

- [ ] All notebook exercises completed and tests passing
- [ ] Quiz score ≥ 80%
- [ ] Prior sensitivity analysis notebook demonstrates understanding
- [ ] Peer review completed (if assigned)

## Prerequisites

- Module 0 completed (diagnostic quiz ≥ 70%)
- Probability concepts fresh in mind
- Python environment verified

## Time Allocation

| Activity | Time |
|----------|------|
| Reading guides | 1.5 hours |
| Notebooks 1-2 | 2 hours |
| Notebook 3 (regression) | 1.5 hours |
| Notebook 4 (sensitivity) | 1 hour |
| Quiz + exercises | 1 hour |
| **Total** | ~7 hours |

## Connections

**Builds on:** Module 0 (probability foundations, commodity market basics)

**Leads to:**
- Module 3: State space models use Bayesian updating (Kalman filter)
- Module 4: Hierarchical models extend the prior structure
- Module 6: MCMC approximates posteriors when conjugacy fails

---

*"The Bayesian approach asks: given what I've observed, what should I believe? This is precisely the question a trader asks before every decision."*
