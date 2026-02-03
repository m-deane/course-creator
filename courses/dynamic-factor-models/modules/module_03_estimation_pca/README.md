# Module 3: Estimation I - Principal Components

## Overview

This module covers practical estimation of dynamic factor models using principal components analysis (PCA). You'll learn the Stock-Watson two-step estimator, asymptotic theory for large N and T, methods for selecting the number of factors, and techniques for handling missing data.

**Estimated Time:** 7-9 hours
**Prerequisites:** Module 1 (static factors), Module 2 (Kalman filter)

## Learning Objectives

By completing this module, you will be able to:

1. **Implement** the Stock-Watson two-step estimator for DFMs
2. **Explain** consistency conditions for large N and T asymptotics
3. **Apply** Bai-Ng information criteria to determine the number of factors
4. **Handle** missing observations using EM-PCA algorithm
5. **Compare** PCA-based and likelihood-based estimation approaches
6. **Assess** factor estimation uncertainty via bootstrap methods

## Module Contents

### Guides
1. `guides/01_stock_watson_estimator.md` - PCA estimation of DFMs
2. `guides/02_factor_number_selection.md` - IC criteria and scree plots
3. `guides/03_missing_data_handling.md` - Expectation-maximization for incomplete panels

### Notebooks
1. `notebooks/01_stock_watson_estimation.ipynb` - Stock-Watson estimator implementation
2. `notebooks/02_factor_number_selection.ipynb` - Apply IC criteria to FRED-MD

### Assessments
- `assessments/quiz_module_03.md` - Conceptual quiz on estimation theory

## Key Concepts

### Stock-Watson Two-Step Estimator

**Step 1: Extract Factors via PCA**
$$\hat{F} = \arg\min_{F} \sum_{t=1}^T \sum_{i=1}^N (X_{it} - \lambda_i' F_t)^2$$

Subject to $F'F/T = I_r$ and $\Lambda'\Lambda$ diagonal.

**Solution:** $\hat{F}$ = $\sqrt{T}$ times first $r$ eigenvectors of $XX'$

**Step 2: Estimate Factor Dynamics**
$$\hat{F}_t = \hat{\Phi}_1 \hat{F}_{t-1} + ... + \hat{\Phi}_p \hat{F}_{t-p} + \hat{\eta}_t$$

Estimate $\{\hat{\Phi}_j\}$ by OLS regression of $\hat{F}_t$ on lagged values.

### Asymptotic Theory

**Consistency Requirements:**
- $N, T \to \infty$ jointly
- $\sqrt{T}/N \to 0$ (N grows faster than T)
- Weak cross-sectional dependence in idiosyncratic errors
- Factor loadings bounded

**Convergence Rates:**
- Factors: $\|\hat{F}_t - H F_t\| = O_p(\min(N^{-1/2}, T^{-1/2}))$
- Loadings: $\|\hat{\lambda}_i - H'\lambda_i\| = O_p(\min(N^{-1/2}, T^{-1/2}))$

where $H$ is rotation matrix.

### Bai-Ng Information Criteria

Determine $r$ by minimizing:

$$IC_p(r) = \ln[V(r)] + r \cdot g(N,T)$$

where $V(r)$ is residual variance with $r$ factors.

**Penalty functions:**
- $IC_1: g(N,T) = (N+T-r)/(NT) \ln(\frac{NT}{N+T})$
- $IC_2: g(N,T) = (N+T-r)/(NT) \ln(C_{NT}^2)$, $C_{NT} = \min(\sqrt{N}, \sqrt{T})$
- $IC_3: g(N,T) = \ln(C_{NT}^2) / C_{NT}^2$

Trade-off: Larger penalties → fewer factors selected.

### EM-PCA for Missing Data

**E-Step:** Impute missing values given current factor estimates
$$\hat{X}_{it}^{miss} = \hat{\lambda}_i' \hat{F}_t$$

**M-Step:** Re-estimate factors via PCA on completed data

**Iterate** until convergence: $\|\hat{F}^{(k+1)} - \hat{F}^{(k)}\| < \epsilon$

**Advantages:**
- Leverages cross-sectional information
- More efficient than simple mean imputation
- Preserves covariance structure

## Coding Project: Complete PCA Pipeline

**Objective:** Build end-to-end factor estimation system

**Requirements:**
1. Data standardization and transformation
2. Factor number selection using IC criteria
3. Two-step estimation with user-specified lag order
4. Missing data handling with EM-PCA
5. Visualization of factors and loadings
6. Variance decomposition (common vs idiosyncratic)

**Deliverable:** Reusable Python class with documentation and tests

**Evaluation Rubric:**
- Code correctness: 40%
- Computational efficiency: 20%
- Documentation quality: 20%
- Visualization clarity: 20%

## Connections to Other Modules

| Module | Connection |
|--------|------------|
| Module 1 | PCA for static factors extended to dynamic case |
| Module 2 | Step 2 estimates factor VAR in state-space form |
| Module 4 | PCA provides initialization for ML estimation |
| Module 6 | Estimated factors used in forecasting |
| Module 7 | Factor estimates from PCA vs ML compared |

## Key Formulas

### Factor Estimation via Eigendecomposition

Sample covariance matrix:
$$\hat{\Sigma}_X = \frac{1}{T} X X'$$

Eigenvalue decomposition:
$$\hat{\Sigma}_X = V D V'$$

Factor estimates:
$$\hat{F} = \frac{1}{\sqrt{T}} V_r' X$$

where $V_r$ contains first $r$ eigenvectors.

### Loading Estimation

Given factors $\hat{F}$:
$$\hat{\lambda}_i = \frac{1}{T} \sum_{t=1}^T X_{it} \hat{F}_t$$

Equivalently: $\hat{\Lambda} = X \hat{F}' / T$

### Variance Decomposition

Total variance of variable $i$:
$$\text{Var}(X_{it}) = \hat{\lambda}_i' \text{Var}(\hat{F}_t) \hat{\lambda}_i + \hat{\sigma}_{e_i}^2$$

R-squared (proportion explained by factors):
$$R_i^2 = \frac{\hat{\lambda}_i' \hat{\lambda}_i}{\text{Var}(X_{it})}$$

## Reading List

### Required
- Stock, J.H. & Watson, M.W. (2002). "Forecasting Using Principal Components from a Large Number of Predictors." *Journal of the American Statistical Association* 97, 1167-1179.
- Bai, J. & Ng, S. (2002). "Determining the Number of Factors in Approximate Factor Models." *Econometrica* 70(1), 191-221.

### Recommended
- Stock, J.H. & Watson, M.W. (2016). "Dynamic Factor Models, Factor-Augmented Vector Autoregressions, and Structural Vector Autoregressions in Macroeconomics." *Handbook of Macroeconomics* Vol. 2, 415-525. (§3)
- Bai, J. (2003). "Inferential Theory for Factor Models of Large Dimensions." *Econometrica* 71(1), 135-171.

### Advanced Topics
- Stock, J.H. & Watson, M.W. (2011). "Dynamic Factor Models." *Oxford Handbook of Economic Forecasting*, 35-59.
- Doz, C., Giannone, D., & Reichlin, L. (2011). "A Two-Step Estimator for Large Approximate Dynamic Factor Models Based on Kalman Filtering." *Journal of Econometrics* 164(1), 188-205.

## Practical Applications

After completing this module, you can:

1. **Extract macroeconomic factors** from FRED-MD dataset (128 monthly series)
2. **Determine optimal number of factors** using data-driven criteria
3. **Handle ragged-edge data** common in real-time forecasting
4. **Decompose variance** to identify most factor-sensitive variables
5. **Bootstrap confidence intervals** for factor estimates
6. **Compare computational speed** of PCA vs likelihood methods

## Common Pitfalls

1. **Forgetting standardization:** Always standardize variables before PCA
2. **Wrong eigenvector scaling:** Use $\sqrt{T}$ times eigenvectors, not raw eigenvectors
3. **Ignoring identification:** PCA factors have arbitrary sign and rotation
4. **Over-selecting factors:** IC criteria can over-select with small T
5. **Missing data patterns:** EM-PCA assumes MAR (missing at random)

## Computational Considerations

**PCA Advantages:**
- Extremely fast (eigendecomposition is O(N³) but N typically < 500)
- No convergence issues
- No starting values needed
- Handles large T efficiently

**When to Use PCA:**
- Quick exploratory analysis
- Large T relative to N
- Initialization for likelihood methods
- Real-time forecasting (speed critical)

**When Likelihood Methods Better:**
- Small sample (T < 100)
- Complex constraints on parameters
- Need standard errors for loadings
- Irregular observation patterns

## Next Steps

After completing this module:
1. Complete the coding project (PCA estimation pipeline)
2. Verify understanding with conceptual quiz
3. Compare your results to statsmodels DynamicFactor
4. Proceed to Module 4: Maximum Likelihood Estimation

---

*"Principal components provide a remarkably simple and robust way to extract latent factors from high-dimensional data. The method's computational efficiency and asymptotic properties make it the workhorse of modern factor analysis." - Adapted from Stock & Watson*
