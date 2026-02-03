# Module 6: Factor-Augmented Regression

## Overview

This module introduces factor-augmented regression methods where estimated factors are used as predictors in forecasting and structural analysis. You'll learn diffusion index forecasting, Factor-Augmented VAR (FAVAR) models, structural identification using factors, and forecast combination techniques. These methods bridge factor models and traditional econometric approaches.

**Estimated Time:** 8-10 hours
**Prerequisites:** Modules 1-3 (static factors, dynamic factors, PCA estimation)

## Learning Objectives

By completing this module, you will be able to:

1. **Implement** diffusion index forecasting for macroeconomic prediction
2. **Specify** and estimate Factor-Augmented VAR (FAVAR) models
3. **Identify** structural shocks using factor-based restrictions
4. **Combine** forecasts from factor models with traditional methods
5. **Evaluate** forecast performance using proper cross-validation
6. **Interpret** factor-based impulse response functions

## Module Contents

### Guides
1. `guides/01_diffusion_index_forecasting.md` - Using factors as predictors
2. `guides/02_favar_models.md` - FAVAR model setup and identification
3. `guides/03_structural_identification.md` - Recovering structural shocks from factors

### Notebooks
1. `notebooks/01_diffusion_index_forecasting.ipynb` - Predict GDP using FRED-MD factors
2. `notebooks/02_favar_analysis.ipynb` - Estimate FAVAR and compute IRFs

### Assessments
- `assessments/quiz_module_06.md` - Conceptual quiz

## Key Concepts

### Diffusion Index Forecasting

**Basic Setup:**
$$y_{t+h} = \alpha + \beta' \hat{F}_t + \gamma' W_t + \varepsilon_{t+h}$$

where:
- $y_{t+h}$: target variable (e.g., GDP growth)
- $\hat{F}_t$: estimated factors from large panel
- $W_t$: additional control variables
- $h$: forecast horizon

**Two-Step Procedure:**
1. Extract factors from large panel: $\hat{F}_t$ via PCA
2. Regression: $y_{t+h}$ on $\hat{F}_t$ (and lags, controls)

**Generated Regressor Problem:** Standard errors must account for estimation uncertainty in $\hat{F}_t$ (typically ignored for large N).

### Factor-Augmented VAR (FAVAR)

**Model Structure:**
$$\begin{bmatrix} F_t \\ Y_t \end{bmatrix} = \Phi(L) \begin{bmatrix} F_{t-1} \\ Y_{t-1} \end{bmatrix} + \begin{bmatrix} u_{Ft} \\ u_{Yt} \end{bmatrix}$$

$$X_t = \Lambda^F F_t + \Lambda^Y Y_t + e_t$$

where:
- $F_t$: unobserved factors
- $Y_t$: observed policy/structural variables
- $X_t$: large panel of informational variables

**Advantages over Standard VAR:**
- Incorporates information from hundreds of variables
- Avoids variable selection problem
- Richer impulse response analysis

### Structural Identification

**Factor-Based Restrictions:**

1. **Long-run restrictions:** Factors identified by permanent vs transitory effects
   $$\Phi(1) = \begin{bmatrix} \Phi_{11}(1) & 0 \\ \Phi_{21}(1) & \Phi_{22}(1) \end{bmatrix}$$

2. **Sign restrictions:** Factors constrained by economic theory
   - Demand shock: GDP ↑, Inflation ↑
   - Supply shock: GDP ↑, Inflation ↓

3. **External instruments:** Use external data to identify specific shocks

**Impulse Response Functions:**
$$IRF_h = \Phi(L)^h \cdot A$$
where $A$ is structural impact matrix satisfying identification restrictions.

### Forecast Combination

**Simple Average:**
$$\hat{y}_{t+h} = \frac{1}{M} \sum_{m=1}^M \hat{y}_{t+h}^{(m)}$$

**Optimal Weights:**
$$w^* = \arg\min_w E[(y_{t+h} - \sum_m w_m \hat{y}_{t+h}^{(m)})^2]$$

**Factor-Based Combination:**
- Use factors to combine AR, VAR, and univariate forecasts
- Exploit common information across models

## Practical Implementation

### Stock-Watson Diffusion Index

**Original Implementation (2002):**
```
For target variable y (e.g., GDP, inflation):
1. Extract r factors from X using PCA
2. Estimate AR(p) with factors:
   y_{t+h} - y_t = α + Σ β_i F_{t-i} + Σ γ_j Δy_{t-j} + ε
3. Forecast h-steps ahead
4. Compare to AR benchmark without factors
```

**Key Results:**
- Factors improve forecasts, especially at medium horizons
- Greatest gains for variables with strong common components

### FAVAR Example: Monetary Policy Shocks

**Setup:**
- $Y_t$: Fed Funds rate (policy instrument)
- $F_t$: factors from ~120 macro variables
- Identification: Recursive (Cholesky) with policy ordered last

**Analysis:**
1. Estimate factors from macro panel
2. Estimate VAR with factors and Fed Funds rate
3. Identify monetary shock (innovation to policy equation)
4. Trace impact through factors to all variables

## Connections to Other Modules

| Module | Connection |
|--------|------------|
| Module 1-3 | Factor extraction methods provide $\hat{F}_t$ |
| Module 4 | EM algorithm can jointly estimate factors and VAR |
| Module 5 | Mixed-frequency factors improve forecasting |
| Module 7 | Sparse methods select most predictive factors |

## Suggested Extension Project: Factor-Based Forecasting System

**Objective:** Build complete forecasting system for macroeconomic variable

**Suggested Requirements:**
1. Extract factors from FRED-MD dataset
2. Implement rolling-window cross-validation
3. Compare factor-augmented AR vs pure AR
4. Compute forecast evaluation metrics (RMSE, MAE, DM test)
5. Visualize forecast performance over time

**Deliverable:** Report with code, tables, and interpretation

**Suggested Evaluation Criteria:**
- Code quality and documentation (30%)
- Proper cross-validation (25%)
- Forecast evaluation (25%)
- Economic interpretation (20%)

## Reading List

### Required
- Stock, J.H. & Watson, M.W. (2002). "Forecasting Using Principal Components from a Large Number of Predictors." *JASA*, 97(460).
- Stock, J.H. & Watson, M.W. (2002). "Macroeconomic Forecasting Using Diffusion Indexes." *Journal of Business & Economic Statistics*, 20(2).
- Bernanke, B.S., Boivin, J. & Eliasz, P. (2005). "Measuring the Effects of Monetary Policy: A Factor-Augmented Vector Autoregressive (FAVAR) Approach." *QJE*, 120(1).

### Recommended
- Giannone, D., Reichlin, L. & Small, D. (2008). "Nowcasting: The real-time informational content of macroeconomic data." *Journal of Monetary Economics*, 55(4).
- Banbura, M., Giannone, D. & Reichlin, L. (2010). "Large Bayesian vector auto regressions." *Journal of Applied Econometrics*, 25(1).

### Forecast Evaluation
- Diebold, F.X. & Mariano, R.S. (1995). "Comparing Predictive Accuracy." *Journal of Business & Economic Statistics*, 13(3).
- Clark, T.E. & McCracken, M.W. (2013). "Advances in Forecast Evaluation." *Handbook of Economic Forecasting*, Vol 2B.

## Practical Applications

After this module, you can:
1. Build factor-based forecasting models for GDP, inflation, unemployment
2. Conduct monetary policy analysis with FAVAR
3. Identify structural shocks in high-dimensional settings
4. Implement forecast combination schemes
5. Properly evaluate out-of-sample forecast performance

## Assessment Strategy

**Formative:**
- Coding exercises in notebooks (auto-graded)
- Conceptual quiz (20 questions)

**Summative:**
- Mini-project: factor-based forecasting system
- Peer review of project interpretations

## Next Steps

After completing this module:
1. Complete the diffusion index forecasting notebook
2. Implement FAVAR mini-project
3. Verify understanding with conceptual quiz
4. Proceed to Module 7: Sparse Methods & Feature Selection

---

*"The diffusion index approach exploits the covariance structure of large datasets to extract the information that is common to many variables." - Stock & Watson (2002)*
