# Module 0: Foundations & Prerequisites

## Overview

This module reviews the mathematical and statistical foundations required for dynamic factor models. Complete the diagnostic assessment to identify areas needing review, then work through the refresher materials as needed.

**Estimated Time:** 4-6 hours (varies based on background)
**Prerequisites:** Undergraduate linear algebra, probability, basic statistics

## Learning Objectives

By completing this module, you will be able to:

1. **Perform** matrix operations essential for factor analysis (eigendecomposition, SVD)
2. **Explain** time series concepts including stationarity and autocovariance
3. **Apply** Principal Component Analysis and interpret results
4. **Set up** the development environment for the course

## Module Contents

### Diagnostic Assessment
- `assessments/diagnostic_quiz.md` - Self-assessment to identify gaps

### Review Guides
1. `guides/01_matrix_algebra_review.md` - Eigendecomposition, SVD, positive definiteness
2. `guides/02_time_series_basics.md` - Stationarity, autocovariance, AR processes
3. `guides/03_pca_refresher.md` - PCA derivation, interpretation, implementation

### Interactive Notebook
- `notebooks/01_foundations_review.ipynb` - Hands-on review with exercises

## Diagnostic Assessment

Before starting the review materials, complete the diagnostic quiz to identify which topics need attention:

| Score | Recommendation |
|-------|----------------|
| 90%+ | Skip to Module 1 |
| 70-89% | Review flagged topics only |
| Below 70% | Complete full module |

## Key Concepts

### Matrix Algebra
- **Eigendecomposition:** $A = V \Lambda V^{-1}$ for symmetric matrices
- **SVD:** $X = U \Sigma V'$ - fundamental for PCA
- **Positive definiteness:** Covariance matrices must be positive semi-definite

### Time Series
- **Stationarity:** Constant mean, variance, and autocovariance structure
- **Autocovariance:** $\gamma(h) = \text{Cov}(y_t, y_{t-h})$
- **AR(1) process:** $y_t = \phi y_{t-1} + \varepsilon_t$

### Principal Components
- **Objective:** Find directions of maximum variance
- **Solution:** Eigenvectors of covariance matrix
- **Scores:** Projections onto principal directions

## Connection to Factor Models

Everything in this module directly supports factor model understanding:

| Foundation | Factor Model Application |
|------------|-------------------------|
| Eigendecomposition | Factor extraction via PCA |
| SVD | Efficient computation for large panels |
| Stationarity | Required for consistent estimation |
| Autocovariance | Dynamic factor specification |
| PCA | Primary estimation method |

## How to Use This Module

1. **Start with diagnostic** - Don't skip areas you think you know
2. **Focus on weak areas** - Time is valuable; prioritize gaps
3. **Complete the notebook** - Hands-on practice cements understanding
4. **Move on when ready** - Perfect scores not required; 80%+ is sufficient

## Resources

### Quick References
- Matrix cookbook: [matrixcookbook.com](https://www.matrixcookbook.com)
- Time series cheat sheet: Module resources folder

### Textbook References
- Strang, G. (2016). *Introduction to Linear Algebra*. Chapters 6-7.
- Hamilton, J.D. (1994). *Time Series Analysis*. Chapters 2-3.

## Next Steps

After completing this module:
1. Verify environment setup works (`resources/environment_setup.md`)
2. Ensure diagnostic score is 80%+
3. Proceed to Module 1: Static Factor Models

---

*Don't rush through foundations. Time invested here pays dividends throughout the course.*
