# Module 7: Sparse Methods & Feature Selection

## Overview

This module addresses the challenge of selecting relevant predictors in high-dimensional factor models. You'll learn modern sparse estimation techniques including targeted predictors (Bai-Ng method), LASSO/elastic net for factor models, and the three-pass regression filter. These methods improve forecast accuracy and interpretability when dealing with hundreds of potential predictors.

**Estimated Time:** 8-10 hours
**Prerequisites:** Modules 1-3, 6 (factor models, factor-augmented regression)

## Learning Objectives

By completing this module, you will be able to:

1. **Apply** high-dimensional regression techniques to factor-based prediction
2. **Implement** targeted predictor selection (Bai-Ng method)
3. **Estimate** factor models with LASSO and elastic net penalties
4. **Execute** the three-pass regression filter for variable selection
5. **Compare** sparse factor methods with traditional PCA-based approaches
6. **Interpret** selected variables in economic context

## Module Contents

### Guides
1. `guides/01_high_dimensional_regression.md` - LASSO, ridge, elastic net fundamentals
2. `guides/02_targeted_predictors.md` - Bai-Ng method for predictor selection
3. `guides/03_three_pass_filter.md` - Three-pass filter methodology

### Notebooks
1. `notebooks/01_lasso_factor_selection.ipynb` - LASSO for factor-augmented forecasting
2. `notebooks/02_targeted_predictors.ipynb` - Implement Bai-Ng method

### Assessments
- `assessments/quiz_module_07.md` - Conceptual quiz
- `assessments/mini_project_sparse.md` - Real-data variable selection

## Key Concepts

### High-Dimensional Regression Review

**The Forecasting Problem:**
$$y_{t+h} = \beta' Z_t + \varepsilon_{t+h}$$

where $Z_t$ contains:
- Lagged target: $y_{t-1}, ..., y_{t-p}$
- Factor estimates: $\hat{F}_t, \hat{F}_{t-1}, ...$
- Raw predictors: subset of $X_t$

**Challenges when dim($Z_t$) is large:**
- Overfitting (low bias, high variance)
- Unstable estimates
- Poor out-of-sample performance
- Difficult interpretation

**Sparse Regularization:**
$$\hat{\beta} = \arg\min_\beta \sum_{t=1}^T (y_{t+h} - \beta' Z_t)^2 + \lambda P(\beta)$$

Penalty options:
- **Ridge (L2):** $P(\beta) = \sum \beta_j^2$ (shrinkage, no selection)
- **LASSO (L1):** $P(\beta) = \sum |\beta_j|$ (shrinkage + selection)
- **Elastic Net:** $P(\beta) = \alpha \sum |\beta_j| + (1-\alpha) \sum \beta_j^2$ (compromise)

### Targeted Predictors (Bai-Ng Method)

**Motivation:** Not all variables in $X_t$ are relevant for predicting $y_t$. Extract factors only from variables that contain predictive information.

**Two-Step Procedure:**

**Step 1: Screen for Relevant Predictors**
For each variable $X_{it}$, test:
$$y_{t+h} = \alpha_i + \beta_i X_{it} + \varepsilon_{it}$$

Select variable $i$ if:
$$|\hat{\beta}_i| / SE(\hat{\beta}_i) > c$$

where $c$ is threshold (e.g., 2.58 for 1% significance).

**Step 2: Extract Factors from Selected Variables**
$$\hat{F}_t^{targeted} = \text{PCA}(X_t^{selected})$$

**Theoretical Result (Bai-Ng 2008):**
Under regularity conditions, targeted factors achieve:
$$MSFE(\hat{F}^{targeted}) \leq MSFE(\hat{F}^{all})$$

**Practical Advantages:**
- Removes noise from irrelevant variables
- Improves factor interpretability
- Reduces computational burden
- Often improves forecast accuracy

### LASSO for Factor-Augmented Models

**Direct Approach:**
$$\hat{\beta} = \arg\min_\beta \sum_{t=1}^T \left(y_{t+h} - \beta_0 - \beta_F' \hat{F}_t - \beta_X' X_t\right)^2 + \lambda \|\beta\|_1$$

**Hybrid Factor-LASSO:**
1. Extract factors $\hat{F}_t$ from all $X_t$
2. Include both $\hat{F}_t$ and selected $X_{it}$ in regression
3. Apply LASSO penalty to select relevant predictors

**Advantages:**
- Factors capture common information
- LASSO selects idiosyncratic predictors
- Combines dimension reduction with variable selection

**Tuning Parameter Selection:**
- Cross-validation (time-series aware)
- Information criteria (AIC, BIC)
- One-standard-error rule

### Three-Pass Regression Filter

**Motivation:** Select variables that have both:
1. Strong relationship with predictors (first-stage power)
2. Predictive power for target (reduced-form relevance)

**The Three Passes:**

**Pass 1: Variable-by-Variable Predictive Regressions**
$$y_{t+h} = \alpha_i + \gamma_i X_{it} + e_{it}, \quad i=1,...,N$$
Compute $\hat{\gamma}_i$ for each predictor.

**Pass 2: Principal Components of Predictive Slopes**
$$\tilde{F}_t = \text{PCA}(\hat{\Gamma})$$
where $\hat{\Gamma} = [\hat{\gamma}_1, ..., \hat{\gamma}_N]'$ is $N \times T$ matrix of fitted values.

**Pass 3: Final Forecast Regression**
$$y_{t+h} = \beta' \tilde{F}_t + \varepsilon_{t+h}$$

**Theoretical Foundation (Kelly-Pruitt 2015):**
- Asymptotically equivalent to infeasible regression on true factors
- Adapts to unknown sparsity
- Robust to weak factors

**Comparison with Standard PCA:**
| Method | Extracts factors from | Selected by |
|--------|----------------------|-------------|
| Standard PCA | $X_t$ | Variance explained in $X$ |
| 3-Pass Filter | $\hat{\gamma}_i X_{it}$ | Covariance with $y$ |

## Implementation Details

### Targeted Predictors Algorithm

```
Input: Panel X (N × T), target y (T × 1), horizon h
Output: Targeted factors F_targeted

1. Screen variables:
   For i = 1 to N:
       Regress y_{t+h} on X_{it}
       Compute t-statistic: t_i = β̂_i / SE(β̂_i)
       If |t_i| > threshold:
           Add i to selected set S

2. Extract factors:
   F_targeted = PCA(X^S, r)
   where X^S contains only selected columns

3. Forecast regression:
   y_{t+h} = α + β' F_targeted + ε
```

### Three-Pass Regression Filter Algorithm

```
Input: Panel X (N × T), target y (T × 1), horizon h, # factors K₂
Output: Forecast ŷ_{t+h}

Pass 1:
For i = 1 to N:
    γ̂ᵢ = (y_{t+h} ~ X_{it})
    Γ̂[i,t] = γ̂ᵢ X_{it}

Pass 2:
[F̃, Λ̃] = PCA(Γ̂, K₂)

Pass 3:
β̂ = (y_{t+h} ~ F̃_t)
ŷ_{t+h} = β̂' F̃_t
```

## Practical Workflow

### Variable Selection Pipeline

**1. Data Preparation**
- Stationarity transformations
- Outlier treatment
- Handle missing values

**2. Candidate Predictor Pool**
- Raw variables: $X_{it}$
- Lags: $X_{it-j}$
- Transformations: $\Delta X_{it}$, growth rates
- Factors: $\hat{F}_t$ from PCA

**3. Apply Multiple Methods**
- Targeted predictors
- LASSO with CV
- Three-pass filter
- Elastic net

**4. Forecast Evaluation**
- Rolling window out-of-sample
- Multiple horizons (h = 1, 3, 6, 12)
- Statistical tests (Diebold-Mariano)

**5. Model Interpretation**
- Which variables selected most often?
- Economic interpretation
- Stability across time

## Connections to Other Modules

| Module | Connection |
|--------|------------|
| Module 3 | PCA vs sparse PCA for factor extraction |
| Module 6 | Improve diffusion index forecasting with selection |
| Module 8 | Machine learning methods as alternative to LASSO |

## Mini-Project: Inflation Forecasting with Sparse Methods

**Objective:** Compare sparse variable selection methods for forecasting inflation

**Dataset:** FRED-MD (124 macroeconomic variables)

**Requirements:**
1. Implement three methods:
   - Targeted predictors (Bai-Ng)
   - LASSO factor-augmented
   - Three-pass regression filter
2. Rolling window evaluation (10-year initial window)
3. Forecast horizons: 1, 3, 6, 12 months
4. Benchmark: AR(p) model
5. Report RMSE, MAE, selected variables

**Deliverable:**
- Code (well-documented)
- Results table with forecast metrics
- Interpretation: which variables matter, economic insights

**Evaluation Rubric:**
- Implementation correctness (40%)
- Proper cross-validation (20%)
- Forecast comparison (20%)
- Economic interpretation (20%)

## Reading List

### Required
- Bai, J. & Ng, S. (2008). "Forecasting economic time series using targeted predictors." *Journal of Econometrics*, 146(2), 304-317.
- Bai, J. & Ng, S. (2009). "Boosting diffusion indices." *JASA*, 104(486), 607-619.
- Kelly, B. & Pruitt, S. (2015). "The three-pass regression filter: A new approach to forecasting using many predictors." *Journal of Econometrics*, 186(2), 294-316.

### Recommended
- Tibshirani, R. (1996). "Regression shrinkage and selection via the lasso." *JRSS-B*, 58(1), 267-288.
- Zou, H. & Hastie, T. (2005). "Regularization and variable selection via the elastic net." *JRSS-B*, 67(2), 301-320.
- Friedman, J., Hastie, T. & Tibshirani, R. (2010). "Regularization paths for generalized linear models via coordinate descent." *JSS*, 33(1).

### Implementation References
- Pedregosa, F. et al. (2011). "Scikit-learn: Machine learning in Python." *JMLR*, 12, 2825-2830.
- For Python LASSO: `sklearn.linear_model.LassoCV`, `sklearn.linear_model.ElasticNetCV`

## Practical Applications

After this module, you can:
1. Select most predictive variables from large panels
2. Build sparse factor models with improved interpretability
3. Combine dimension reduction with variable selection
4. Implement production-ready forecasting systems
5. Explain which economic variables drive predictions

## Assessment Strategy

**Formative:**
- Notebook exercises (implementation of each method)
- Conceptual quiz on sparse methods

**Summative:**
- Mini-project: inflation forecasting comparison
- Peer review of economic interpretations

## Advanced Extensions

**For interested students:**
1. Adaptive LASSO for factor models
2. Group LASSO for variable blocks
3. Spike-and-slab priors (Bayesian variable selection)
4. Stability selection for inference

## Next Steps

After completing this module:
1. Complete all method implementations in notebooks
2. Execute mini-project with real data
3. Verify understanding with conceptual quiz
4. Proceed to Module 8: Advanced Topics

---

*"Variable selection is not just about improving forecasts—it's about understanding which economic relationships matter." - Bai & Ng (2008)*
