# Module 7 Quiz: Sparse Methods and Regularization

## Instructions

**Time Estimate:** 50-60 minutes
**Total Points:** 100
**Attempts Allowed:** 2
**Open Resources:** Yes (course materials, documentation)

This quiz assesses your understanding of sparse estimation methods in factor models, including LASSO-based approaches, targeted predictors, principal components regression, and the three-pass regression filter.

**Question Types:**
- Multiple Choice (select the best answer)
- True/False (with brief justification)
- Short Answer (2-4 sentences)
- Computational application

---

## Part 1: High-Dimensional Challenges (15 points)

### Question 1 (5 points) - Conceptual Understanding
**Learning Objective:** Understand curse of dimensionality

In the context of factor models with many predictors (n >> T), what is the primary problem with using all variables directly in regression?

A) Computational cost becomes prohibitive
B) Multicollinearity causes unstable coefficient estimates
C) The model becomes difficult to interpret
D) Data storage requirements are excessive

**Feedback:**
- A: While computation matters, this isn't the primary statistical issue
- B: **Correct!** With n > T, the design matrix is rank-deficient and OLS is infeasible or produces infinite variance estimates
- C: Interpretation is a concern but not the fundamental problem
- D: Storage is rarely the limiting factor with modern systems

---

### Question 2 (5 points) - True/False
**Learning Objective:** Recognize regularization necessity

**Statement:** "When n > T, adding an L2 penalty (ridge regression) makes estimation feasible, but we can achieve even better prediction accuracy by also promoting sparsity through L1 penalties."

**True** / **False**

**Justification (2 sentences):**

**Expected Answer:** True. Ridge regression (L2) ensures a unique solution when n > T by shrinking coefficients, but it includes all predictors with non-zero weights. LASSO (L1) additionally performs variable selection by setting some coefficients exactly to zero, which can improve prediction by reducing variance and enhancing interpretability, especially when only a subset of predictors is truly relevant.

---

### Question 3 (5 points) - Bias-Variance Tradeoff
**Learning Objective:** Understand regularization tradeoff

You have T=100 observations and n=500 potential predictors. Three models are fit:
- OLS (infeasible, regularized with ridge, λ=0.01)
- LASSO (λ chosen by cross-validation)
- Factor model (5 principal components)

Rank these approaches from highest bias to lowest bias:

A) OLS, LASSO, Factor model
B) Factor model, LASSO, OLS
C) LASSO, Factor model, OLS
D) All have approximately equal bias

**Feedback:**
- A: OLS has lowest bias when feasible
- B: **Correct!** Factor model uses only 5 dimensions (high bias), LASSO selects subset (medium bias), ridge OLS includes all (lowest bias but highest variance)
- C: Incorrect ordering
- D: Bias varies substantially across methods

---

## Part 2: LASSO and Penalized Regression (30 points)

### Question 4 (6 points) - LASSO Formulation
**Learning Objective:** Understand LASSO optimization

The LASSO estimator solves:

$$\hat{\beta}^{\text{LASSO}} = \arg\min_{\beta} \left\{ \sum_{t=1}^T (y_t - x_t'\beta)^2 + \lambda \sum_{j=1}^n |\beta_j| \right\}$$

**Part A (3 points):** What is the role of the tuning parameter λ?

**Expected Answer:** λ controls the strength of penalization. As λ increases, more coefficients are shrunk to exactly zero (increasing sparsity). λ = 0 gives OLS (no penalty), while λ → ∞ gives β = 0 (all coefficients set to zero). It governs the bias-variance tradeoff.

**Part B (3 points):** Why does the L1 penalty produce exact zeros while the L2 penalty (ridge) only shrinks coefficients toward zero?

**Expected Answer:** The L1 penalty creates a non-differentiable "corner" at zero (absolute value function), causing the optimization solution to often hit this corner exactly. The L2 penalty (squared terms) is smooth everywhere, so the gradient descent solution rarely lands exactly at zero, only asymptotically approaches it.

---

### Question 5 (8 points) - Cross-Validation
**Learning Objective:** Apply cross-validation for tuning

**Scenario:** You are using LASSO to forecast GDP growth with 200 monthly indicators. You use k-fold cross-validation to select λ.

**Part A (4 points):** Describe the k-fold CV procedure specifically for time series data. Why can't you use random splitting?

**Expected Answer:** In time series, use **rolling** or **expanding window** CV to respect temporal ordering:
1. Split data into training (t=1,...,T-h) and validation (t=T-h+1,...,T)
2. Fit LASSO with candidate λ on training set
3. Predict h-steps ahead and compute validation error
4. Roll/expand window and repeat for multiple validation periods
5. Average validation errors and select λ with minimum error

Random splitting violates temporal structure and causes **look-ahead bias** - you'd be using future data to predict the past, giving artificially optimistic results.

**Part B (4 points):** You observe that the CV-selected λ results in only 15 out of 200 variables having non-zero coefficients. Is this necessarily a problem? Explain.

**Expected Answer:** No, this is not necessarily a problem - it's often desirable. High sparsity (15/200 variables) suggests most predictors are redundant or noisy. The LASSO is doing its job by selecting only the most informative variables, which:
1. Improves interpretation (focus on key drivers)
2. Reduces overfitting (lower variance)
3. Can improve out-of-sample forecasts if truly only few variables matter

However, verify robustness: check if selected variables are stable across CV folds and make economic sense.

---

### Question 6 (6 points) - Elastic Net
**Learning Objective:** Understand combined penalties

The elastic net combines L1 and L2 penalties:

$$\hat{\beta}^{\text{EN}} = \arg\min_{\beta} \left\{ \sum_{t=1}^T (y_t - x_t'\beta)^2 + \lambda_1 \sum_{j=1}^n |\beta_j| + \lambda_2 \sum_{j=1}^n \beta_j^2 \right\}$$

**Question:** In what situation would elastic net be preferred over pure LASSO? Provide a specific example relevant to factor models.

**Grading:**
- Situation identification (3 points)
- Relevant example (3 points)

**Expected Answer:**

**Situation:** Elastic net is preferred when:
1. Predictors are highly correlated (groups of collinear variables)
2. You want to select groups rather than individual variables
3. LASSO's selection instability is problematic

**Example:** When forecasting with highly correlated financial indicators (e.g., multiple corporate bond spreads: AAA, AA, A, BBB). Pure LASSO might arbitrarily select one spread and exclude others due to high correlation. Elastic net includes the group together with shared shrinkage, capturing the common information while maintaining some sparsity. This is more stable across samples and economically interpretable as a "credit risk" factor.

---

### Question 7 (5 points) - Multiple Choice
**Learning Objective:** Recognize LASSO properties

Which statement about LASSO is FALSE?

A) LASSO performs automatic variable selection
B) LASSO coefficient estimates are unbiased when the true model is sparse
C) LASSO can be computed efficiently even when n >> T
D) LASSO provides valid confidence intervals under standard theory

**Feedback:**
- A: True - L1 penalty sets coefficients exactly to zero
- B: True - under sparsity and irrepresentability conditions, LASSO recovers true model
- C: True - coordinate descent and pathwise algorithms scale well
- D: **FALSE - This is correct!** Standard inference fails because LASSO is non-differentiable at zero. Need specialized methods (post-selection inference, debiased LASSO) for valid confidence intervals

---

### Question 8 (5 points) - Short Answer
**Learning Objective:** Apply LASSO to factor models

**Question:** How can LASSO be used to improve factor model estimation? Describe two specific applications.

**Expected Answer (any two):**

1. **Sparse loadings:** Apply LASSO to factor loadings matrix Λ, setting small loadings to zero. This creates an interpretable factor structure where each factor relates to a subset of variables.

2. **Factor selection:** Use LASSO to select which principal components to include in forecasting regression, automatically choosing relevant factors from a larger set.

3. **Targeted predictors:** Apply LASSO to select which variables to use in constructing targeted factors for specific outcomes (preview of next section).

4. **Lag selection:** In dynamic factor models, use LASSO to select relevant lags in factor VAR, achieving parsimony in temporal dynamics.

5. **Structural identification:** Use LASSO in external instruments approach to select valid instruments from a large set of candidates.

---

## Part 3: Targeted Predictors and PCR (25 points)

### Question 9 (6 points) - Targeted Principal Components
**Learning Objective:** Understand supervised dimension reduction

Standard PCA extracts factors that explain variance in X (unsupervised). Targeted PCA extracts factors relevant for predicting y (supervised).

**Part A (3 points):** Why might standard PC factors perform poorly for prediction?

**Expected Answer:** Standard PCs maximize variance in the predictor space but may capture variation unrelated to the outcome y. The first PC might explain predictor variance dominated by measurement error, common seasonality, or variables irrelevant to y. For prediction, we need factors correlated with y, not just factors that explain X.

**Part B (3 points):** How does targeted PCA modify standard PCA?

**Expected Answer:** Targeted PCA pre-weights variables by their marginal correlation or regression coefficient with y before extracting principal components. Specifically:
- Compute $\hat{\theta}_i = \text{Cov}(X_i, y) / \text{Var}(X_i)$ for each predictor i
- Construct weighted data $\tilde{X}_i = \hat{\theta}_i X_i$
- Extract PCs from $\tilde{X}$

This ensures factors load more heavily on y-relevant predictors.

---

### Question 10 (8 points) - Principal Components Regression (PCR)
**Learning Objective:** Apply PCR methodology

PCR involves two steps:
1. Extract K principal components from X: $\hat{F} = X\hat{V}_K$ where $\hat{V}_K$ are first K eigenvectors
2. Regress y on $\hat{F}$: $y = \hat{F}\gamma + \epsilon$

**Part A (3 points):** If we use all n principal components (K=n), does PCR reduce to OLS? Why or why not?

**Expected Answer:** Only if T ≥ n. When K=n, PCR uses the full eigenvector matrix and is equivalent to OLS because PCs span the same space as original variables. However, if n > T, we can only extract at most T-1 meaningful PCs, so PCR with K=T-1 is still regularized compared to infeasible OLS.

**Part B (5 points):** You fit PCR with K=5 factors and obtain R²=0.75 on training data. Your colleague fits OLS-LASSO and achieves R²=0.82. Can you conclude LASSO is better?

**Expected Answer:** No, for several reasons:

1. **In-sample overfitting:** Higher training R² doesn't imply better out-of-sample performance. LASSO with more variables might overfit.

2. **Not comparing apples-to-apples:** Different K choices in PCR would change R². Need to tune both methods on validation data.

3. **Evaluation metric:** R² measures in-sample fit, not forecasting accuracy. Compare out-of-sample MSFE on hold-out test set.

4. **Statistical significance:** Test whether difference is statistically significant (Diebold-Mariano test).

**Proper comparison:** Use nested CV - outer loop for final test set, inner loop to tune K (PCR) and λ (LASSO), then compare out-of-sample forecasts.

---

### Question 11 (6 points) - Factor vs. LASSO
**Learning Objective:** Compare dimension reduction approaches

**Question:** Discuss one advantage and one disadvantage of PC factors compared to LASSO for high-dimensional forecasting.

**Expected Answer:**

**Advantage of PC factors:**
- **Stability:** PCs are more stable across samples because they use all data rather than selecting variables. LASSO selection can be sensitive to small perturbations when variables are correlated.
- **Dense information:** PCs aggregate information from all variables (possibly relevant when many weak predictors exist), while LASSO might exclude relevant variables.
- **Collinearity handling:** PCs orthogonalize variables by construction, while LASSO struggles with highly correlated groups.

**Disadvantage of PC factors:**
- **Lack of interpretability:** PCs are linear combinations of all variables with no economic interpretation. LASSO provides sparse models identifying specific predictors.
- **Suboptimal for prediction:** Unsupervised PCs might capture irrelevant variation. LASSO directly optimizes prediction (though targeted PCs address this).
- **Fixed dimension:** Must choose K, and optimal K varies with forecasting horizon and outcome.

(Any one well-articulated advantage and disadvantage receives full credit)

---

### Question 12 (5 points) - True/False
**Learning Objective:** Understand PC properties

**Statement:** "The first principal component is always the best single predictor for forecasting any outcome variable y."

**True** / **False**

**Justification (2 sentences):**

**Expected Answer:** False. The first PC explains the most variance in the predictor set X, but this variance may be unrelated to y. A later PC (or a single original variable with high correlation to y) could be a better predictor. This is why targeted PCA and supervised methods were developed.

---

## Part 4: Three-Pass Regression Filter (30 points)

### Question 13 (8 points) - Three-Pass Methodology
**Learning Objective:** Understand three-pass procedure

The three-pass regression filter (3PRF) of Kelly and Pruitt (2015) consists of:

**Pass 1:** PCA on full predictor matrix → extract factors $\hat{F}$
**Pass 2:** For each predictor $X_i$, regress on factors: $X_i = \hat{F}\lambda_i + e_i$, obtain residuals $\hat{e}_i$
**Pass 3:** Regress outcome y on residuals $\hat{e}$: $y = \hat{e}'\gamma + u$

**Part A (4 points):** What is the economic intuition behind Pass 2 and Pass 3?

**Expected Answer:**

**Pass 2:** Decomposes each predictor into:
- Common component (explained by factors): $\hat{F}\hat{\lambda}_i$ - captures systematic variation
- Idiosyncratic component (residual): $\hat{e}_i$ - captures predictor-specific information

**Pass 3:** Identifies which idiosyncratic components are useful for predicting y beyond the common factors. This finds variables with unique predictive content not captured by the factor structure, essentially asking: "After accounting for common variation, do any individual predictors still matter?"

**Part B (4 points):** How does 3PRF differ from standard PCR? What problem does it solve?

**Expected Answer:** PCR uses only the common factors from Pass 1. 3PRF adds idiosyncratic components in Pass 3, capturing predictor-specific information that might be lost in factor extraction. This solves the problem of **weak predictors**: variables with low loadings on common factors but strong predictive power for y. PCR would miss these; 3PRF recovers them through idiosyncratic components.

---

### Question 14 (10 points) - Computational Application
**Learning Objective:** Apply three-pass filter

**Scenario:** You have y (GDP growth) and X (100 monthly indicators). You apply the three-pass filter.

**Given:**
- Pass 1 extracts K=5 factors explaining 65% of X variance
- Pass 2 produces 100 residual series
- Pass 3 LASSO (on the 100 residuals) selects 8 non-zero coefficients

**Part A (3 points):** What does it mean that only 8 residuals receive non-zero weights?

**Expected Answer:** Only 8 of the 100 predictors have idiosyncratic information (beyond common factors) useful for forecasting GDP. The other 92 variables' predictive content is fully captured by the 5 common factors. These 8 variables have specific relationships with GDP not explained by aggregate economic conditions.

**Part B (3 points):** The selected variables include: Oil prices, consumer sentiment, housing starts, and 5 regional employment indicators. Provide an economic interpretation.

**Expected Answer:** These variables have predictor-specific dynamics not captured by national aggregates:
- **Oil prices:** Global supply/demand shocks affecting inflation and production costs
- **Consumer sentiment:** Expectational/behavioral component not reflected in hard data
- **Housing starts:** Leading indicator with unique sector-specific cycle
- **Regional employment:** Geographic heterogeneity not captured by national factors (some regions lead/lag overall cycle)

**Part C (4 points):** How would you construct the final forecast? Write the prediction equation.

**Expected Answer:**

The three-pass forecast combines factors and selected idiosyncratic components:

$$\hat{y}_{t+h} = \alpha + \hat{F}_t'\beta + \hat{e}_t'\hat{\gamma}$$

where:
- $\hat{F}_t$ = K=5 common factors from Pass 1
- $\hat{e}_t$ = stacked vector of 8 selected residuals from Pass 2
- $\beta$ = coefficients from regressing y on factors (or could be implicitly embedded)
- $\hat{\gamma}$ = LASSO coefficients from Pass 3 (only 8 non-zero elements)

Alternatively, can combine in one step after extracting factors and residuals:

$$\hat{y}_{t+h} = \alpha + \sum_{k=1}^5 \beta_k \hat{F}_{k,t} + \sum_{i \in S} \gamma_i \hat{e}_{i,t}$$

where S is the set of 8 selected variables.

---

### Question 15 (6 points) - Extensions and Variations
**Learning Objective:** Recognize 3PRF variations

**Question:** Kelly and Pruitt (2015) show that 3PRF can be modified to target specific forecast horizons or outcomes. Describe how you would adapt 3PRF to:

1. Forecast at horizon h=4 quarters instead of h=1
2. Forecast a different variable (inflation instead of GDP) using the same predictors

**Grading:** 3 points per modification

**Expected Answer:**

**1. Different horizon (h=4):**

- **Pass 1:** Unchanged - still extract factors from X
- **Pass 2:** Unchanged - still decompose X into factors and residuals
- **Pass 3 modification:** Regress $y_{t+4}$ (4-quarter ahead GDP) on residuals:

  $$y_{t+4} = \hat{e}_t'\gamma^{(h=4)} + u_t$$

The key insight is that different horizons require different idiosyncratic components - variables useful for nowcasting may differ from those useful for longer horizons. Re-run Pass 3 LASSO with shifted outcome.

**2. Different outcome (inflation):**

- **Pass 1:** Could be unchanged (same X factors) or re-extract targeted to inflation
- **Pass 2:** If factors unchanged, residuals also unchanged
- **Pass 3 modification:** Regress inflation on residuals:

  $$\pi_{t+h} = \hat{e}_t'\gamma^{(\pi)} + u_t$$

Different outcomes will select different idiosyncratic components. For inflation, commodity prices and wage indicators' residuals might receive high weights, while financial indicators might not.

**Best practice:** For very different outcomes, consider re-doing Pass 1 with targeted PCA specific to that outcome.

---

### Question 16 (6 points) - Critical Analysis
**Learning Objective:** Evaluate 3PRF assumptions

**Question:** What is one key assumption of the three-pass filter, and what happens if it's violated? Provide an example.

**Expected Answer:**

**Key assumption:** The factor structure is **pervasive** - common factors explain most of the variation in X, and idiosyncratic components are truly "idiosyncratic" (weakly correlated across variables).

**Violation:** If there are **multiple factor structures** (e.g., separate factors for real and financial variables that don't load on a common set), Pass 1 might miss some factors, and their information would be incorrectly attributed to idiosyncratic components in Pass 2.

**Example:** Suppose there are 3 "real economy" factors affecting 50 variables and 3 "financial" factors affecting the other 50 variables, but little overlap. If Pass 1 extracts only K=3 factors, it might capture only the real factors. The financial factors would appear as correlated "idiosyncratic" components, violating the weak correlation assumption.

**Consequence:** Pass 3 might incorrectly select many financial variables because their shared factor structure is in the residuals. Solution: Extract more factors in Pass 1 or use group-specific factor models.

(Alternative valid answers: generated regressor problem, forward-looking bias if y is revised, or assumption of linearity)

---

## Bonus Question (5 points)

### Question 17 - Research Application
**Learning Objective:** Connect to empirical research

**Question:** Recent research combines machine learning (random forests, neural networks) with factor models for economic forecasting. How might you integrate LASSO or 3PRF with a random forest? What would be the advantage?

**Expected Answer (examples):**

**Integration approaches:**

1. **Two-stage:** Use LASSO/3PRF to select variables, then feed selected variables into random forest. Reduces dimensionality for RF while maintaining nonlinear modeling flexibility.

2. **Factor-augmented RF:** Extract factors and targeted residuals via 3PRF, then use as inputs to RF along with original variables. RF can automatically determine which inputs matter and capture nonlinearities.

3. **Ensemble:** Average forecasts from LASSO (linear), 3PRF (factor-based linear), and RF (nonlinear). Combines different modeling philosophies.

**Advantages:**

- **Dimension reduction + nonlinearity:** LASSO/3PRF handles high dimensionality better than RF (which struggles with many irrelevant features), while RF captures nonlinear relationships LASSO misses.

- **Interpretability + accuracy:** LASSO/3PRF provides interpretable variable importance, RF improves forecast accuracy.

- **Regularization:** Pre-screening with LASSO reduces overfitting risk in RF by limiting feature space.

**Challenge:** Harder to interpret final model; computationally intensive; requires careful validation to avoid overfitting.

---

## Quiz Completion

**Total Points: 105 (100 + 5 bonus)**

### Learning Objectives Coverage

- ✓ High-dimensional challenges and curse of dimensionality
- ✓ LASSO formulation, optimization, and cross-validation
- ✓ Targeted predictors and supervised dimension reduction
- ✓ Principal components regression (PCR)
- ✓ Three-pass regression filter methodology
- ✓ Comparison of sparse methods vs. factor approaches
- ✓ Applications to economic forecasting

### Submission Instructions

1. Save your answers in a single document (PDF or markdown)
2. Show all mathematical derivations
3. For computational questions, include code if requested
4. Submit via course platform by due date

### Grading Scale

- 90-100: Excellent understanding of sparse methods
- 80-89: Good grasp with minor gaps
- 70-79: Adequate understanding, review regularization
- Below 70: Significant gaps, attend office hours

### Recommended Review Materials

If you scored below 80 on any section:
- **Part 1 (High-dimensional):** Review curse of dimensionality, bias-variance tradeoff
- **Part 2 (LASSO):** Re-read Tibshirani (1996), Hastie et al. textbook chapters
- **Part 3 (Targeted predictors):** Study Bai & Ng (2008), Kelly & Pruitt (2013)
- **Part 4 (Three-pass):** Work through Kelly & Pruitt (2015) step-by-step
- **Integration:** Review empirical papers combining methods

### Computational Resources

- R packages: `glmnet` (LASSO), `stats` (PCA), `threePGSGF` (three-pass filter)
- Python packages: `scikit-learn` (LASSO, PCA), `statsmodels` (regression)
- Practice datasets: FRED-MD, FRED-QD (mixed-frequency macro data)

**After submission, review the detailed answer key and work through practice problems.**
