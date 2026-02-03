# Quiz: Module 5 - Advanced Topics

**Course:** Panel Data Econometrics
**Module:** 5 - Advanced Topics
**Time Limit:** 30 minutes
**Total Points:** 100
**Passing Score:** 70%

## Instructions

This quiz assesses your understanding of clustered standard errors, dynamic panels, GMM estimation, and advanced panel diagnostics. You have 2 attempts. Show calculations where requested.

---

## Part A: Clustered Standard Errors and Robust Inference (25 points)

### Question 1: Why Cluster by Entity? (5 points)

In panel data, why do we typically cluster standard errors by entity?

**A)** To account for correlation of errors within entities over time

**B)** To correct for heteroskedasticity

**C)** To eliminate entity fixed effects

**D)** To improve coefficient estimates

---

### Question 2: Clustered SE Formula (5 points)

The cluster-robust variance estimator is:

$$\hat{V}_{cluster} = (X'X)^{-1}\left(\sum_{i=1}^N X_i'\hat{u}_i\hat{u}_i'X_i\right)(X'X)^{-1}$$

where $\hat{u}_i$ is the vector of residuals for entity $i$. What does $\hat{u}_i\hat{u}_i'$ represent?

**A)** The sum of squared residuals for entity $i$

**B)** The $T \times T$ outer product of residuals for entity $i$, allowing arbitrary correlation within entity

**C)** The cross-product of residuals across entities

**D)** The variance of residuals for entity $i$

---

### Question 3: When Are Clustered SEs Needed? (5 points)

Clustered standard errors are **most critical** when:

**A)** The number of clusters is very large ($N > 1000$)

**B)** There is substantial within-cluster (entity) correlation in errors

**C)** The panel is perfectly balanced

**D)** Entity fixed effects are included

---

### Question 4: Multi-Way Clustering (5 points)

Two-way clustering (by entity and time) accounts for:

**A)** Correlation within entities over time AND correlation across entities at the same time

**B)** Only correlation within entities

**C)** Only correlation across entities

**D)** Heteroskedasticity but not correlation

---

### Question 5: Cluster Number Rule of Thumb (5 points)

Asymptotic theory for clustered standard errors requires:

**A)** Large $T$ (many time periods)

**B)** Large $N$ (many clusters)

**C)** Both large $N$ and large $T$

**D)** Only that $NT$ is large

---

## Part B: Dynamic Panel Models (40 points)

### Question 6: Dynamic Panel Specification (8 points)

A dynamic panel model includes:

$$y_{it} = \gamma y_{i,t-1} + X_{it}\beta + \alpha_i + \epsilon_{it}$$

Why does this create a problem for standard fixed effects estimation?

**A)** $y_{i,t-1}$ is perfectly collinear with $\alpha_i$

**B)** $y_{i,t-1}$ is correlated with $\alpha_i$, creating endogeneity even after within transformation

**C)** Dynamic models require time fixed effects

**D)** The model is not identified

---

### Question 7: Nickell Bias (8 points)

The **Nickell bias** in fixed effects estimation of dynamic panels:

**A)** Is zero when $T$ is large

**B)** Causes $\hat{\gamma}$ to be biased downward (toward zero)

**C)** Increases with larger $T$

**D)** Only occurs in unbalanced panels

---

### Question 8: First-Difference Transformation (8 points)

To eliminate fixed effects in a dynamic model, we can first-difference:

$$\Delta y_{it} = \gamma \Delta y_{i,t-1} + \Delta X_{it}\beta + \Delta\epsilon_{it}$$

Why is OLS on this still inconsistent?

**A)** $\Delta y_{i,t-1} = y_{i,t-1} - y_{i,t-2}$ contains $y_{i,t-1}$, which correlates with $\Delta\epsilon_{it} = \epsilon_{it} - \epsilon_{i,t-1}$

**B)** First-differencing creates serial correlation

**C)** The intercept is eliminated

**D)** It's actually consistent; this is a valid approach

---

### Question 9: Arellano-Bond Instruments (8 points)

The Arellano-Bond difference GMM estimator uses which instruments for $\Delta y_{i,t-1}$?

**A)** $\Delta y_{i,t-2}, \Delta y_{i,t-3}, ...$

**B)** $y_{i,t-2}, y_{i,t-3}, ...$

**C)** $X_{i,t-1}, X_{i,t-2}, ...$

**D)** $\alpha_i$

---

### Question 10: System GMM (8 points)

System GMM (Arellano-Bover/Blundell-Bond) combines:

**A)** Equations in differences and equations in levels, using different instrument sets for each

**B)** FE and RE estimators

**C)** Within and between estimators

**D)** OLS and 2SLS

---

## Part C: GMM Diagnostics and Tests (35 points)

### Question 11: Sargan/Hansen Test (7 points)

The Sargan or Hansen test of overidentifying restrictions tests:

**A)** Whether the instruments are valid (uncorrelated with errors)

**B)** Whether there are enough instruments

**C)** Whether the model is dynamically stable

**D)** Whether fixed effects should be included

---

### Question 12: Arellano-Bond AR(2) Test (7 points)

The Arellano-Bond test for second-order serial correlation tests:

$$H_0: E[\Delta\epsilon_{it} \cdot \Delta\epsilon_{i,t-2}] = 0$$

Why do we test for AR(2) in differences rather than AR(1)?

**A)** AR(1) in differences is expected by construction, but AR(2) indicates misspecification

**B)** AR(2) is easier to test statistically

**C)** AR(1) doesn't matter for consistency

**D)** The model automatically corrects for AR(1)

---

### Question 13: Instrument Proliferation (7 points)

Using too many instruments in GMM can cause:

**A)** Overfit the endogenous variables and weaken the Sargan/Hansen test

**B)** Perfect identification

**C)** Improved efficiency

**D)** Faster computation

---

### Question 14: Difference vs System GMM (7 points)

System GMM is often preferred over difference GMM because:

**A)** Difference GMM has weak instruments when variables are persistent

**B)** System GMM doesn't require instruments

**C)** System GMM works better with unbalanced panels

**D)** Difference GMM cannot handle lagged dependent variables

---

### Question 15: Dynamic Panel Assumptions (7 points)

For Arellano-Bond GMM to be consistent, we need:

**A)** $E[\epsilon_{it} | y_{i,t-2}, y_{i,t-3}, ..., y_{i1}] = 0$ (sequential exogeneity)

**B)** $E[\epsilon_{it} | X_{it}] = 0$ only

**C)** No serial correlation in levels

**D)** Perfect balance in the panel

---

## Calculation and Application Problems

### Question 16: Clustered Standard Errors Impact (15 points)

A researcher estimates a wage equation with $N = 50$ workers, $T = 10$ years.

**OLS standard errors (incorrect):** $SE_{OLS}(\hat{\beta}) = 0.20$

**Clustered standard errors (by worker):** $SE_{cluster}(\hat{\beta}) = 0.45$

**Part A (5 points):** What is the ratio $SE_{cluster}/SE_{OLS}$? What does this large ratio suggest about within-worker correlation?

**Part B (5 points):** The coefficient estimate is $\hat{\beta} = 0.60$. Calculate the t-statistic using:
- OLS standard errors
- Clustered standard errors

**Part C (5 points):** At the 5% level (critical value ≈ 2.0), would you reject $H_0: \beta = 0$ using:
- OLS SEs?
- Clustered SEs?

What does this illustrate about the importance of proper standard errors?

---

### Question 17: Dynamic Panel Instruments (20 points)

Consider a dynamic panel with $T = 5$ periods: $t = 1, 2, 3, 4, 5$.

The first-differenced equation for period $t = 4$ is:

$$\Delta y_{i4} = \gamma \Delta y_{i3} + \Delta\epsilon_{i4}$$

where $\Delta y_{i3} = y_{i3} - y_{i2}$ and $\Delta\epsilon_{i4} = \epsilon_{i4} - \epsilon_{i3}$.

**Part A (7 points):** Which lagged levels of $y$ are valid instruments for $\Delta y_{i3}$? Consider:
- $y_{i2}$: Valid or invalid? Why?
- $y_{i1}$: Valid or invalid? Why?

**Part B (7 points):** For the differenced equation in period $t = 5$, list all valid instruments from lagged levels of $y$.

**Part C (6 points):** Explain why having more time periods increases the number of instruments in Arellano-Bond GMM. Is this always beneficial?

---

## Answer Key and Explanations

### Question 1: Answer A (5 points)

**Correct Answer:** A) To account for correlation of errors within entities over time

**Explanation:**

**Within-entity correlation** is the primary motivation:

$$Corr(\epsilon_{it}, \epsilon_{is}) \neq 0 \text{ for } t \neq s$$

**Sources:**
- Persistent entity-specific shocks
- Omitted time-varying entity characteristics
- Serial correlation in unobservables

**Consequence if ignored:**
- Standard errors are too small (overconfident)
- t-statistics are inflated
- False rejections of null hypotheses

**Why other answers are incomplete:**
- B: Robust SEs address heteroskedasticity; clustering addresses correlation
- C: FE removes $\alpha_i$; clustering fixes inference
- D: Clustering affects SEs, not point estimates

**Key Concept:** Cluster by entity to account for within-entity error correlation.

---

### Question 2: Answer B (5 points)

**Correct Answer:** B) The $T \times T$ outer product of residuals for entity $i$, allowing arbitrary correlation within entity

**Explanation:**

For entity $i$, let $\hat{u}_i = [\hat{\epsilon}_{i1}, \hat{\epsilon}_{i2}, ..., \hat{\epsilon}_{iT}]'$ be the $T \times 1$ vector of residuals.

$$\hat{u}_i\hat{u}_i' = \begin{bmatrix} \hat{\epsilon}_{i1} \\ \hat{\epsilon}_{i2} \\ \vdots \\ \hat{\epsilon}_{iT} \end{bmatrix} [\hat{\epsilon}_{i1}, \hat{\epsilon}_{i2}, ..., \hat{\epsilon}_{iT}] = \begin{bmatrix} \hat{\epsilon}_{i1}\hat{\epsilon}_{i1} & \hat{\epsilon}_{i1}\hat{\epsilon}_{i2} & \cdots \\ \hat{\epsilon}_{i2}\hat{\epsilon}_{i1} & \hat{\epsilon}_{i2}\hat{\epsilon}_{i2} & \cdots \\ \vdots & \vdots & \ddots \end{bmatrix}$$

**This allows:**
- Diagonal: Variance in each period
- Off-diagonal: Correlation across periods within entity $i$
- No structure imposed on correlation pattern

**Key Concept:** Cluster-robust variance uses outer product to allow arbitrary within-cluster correlation.

---

### Question 3: Answer B (5 points)

**Correct Answer:** B) There is substantial within-cluster (entity) correlation in errors

**Explanation:**

Clustering matters most when:
- **High intraclass correlation**: $\rho = Corr(\epsilon_{it}, \epsilon_{is})$ is large
- Errors are strongly persistent within entities
- Many observations per cluster

**Impact:**
- If $\rho \approx 0$: Clustered SEs ≈ Standard SEs
- If $\rho$ is high: Clustered SEs >> Standard SEs

**Formula for SE inflation:**

$$SE_{cluster} \approx SE_{standard} \sqrt{1 + (T-1)\rho}$$

With $T=10$ and $\rho=0.5$: $SE_{cluster} \approx 2.3 \times SE_{standard}$

**Why other answers are wrong:**
- A: Large $N$ is needed for asymptotics, but clustering is about correlation
- C: Balance doesn't determine need for clustering
- D: FE doesn't eliminate need for clustering

**Key Concept:** High within-cluster correlation makes clustering essential.

---

### Question 4: Answer A (5 points)

**Correct Answer:** A) Correlation within entities over time AND correlation across entities at the same time

**Explanation:**

**Two-way clustering** accounts for:

1. **Entity clustering:** $Corr(\epsilon_{it}, \epsilon_{is}) \neq 0$ (same entity, different times)

2. **Time clustering:** $Corr(\epsilon_{it}, \epsilon_{jt}) \neq 0$ (different entities, same time)

**When needed:**
- Common shocks affect all entities at same time (macro shocks, policy changes)
- Spatial correlation across entities
- Network effects

**Formula:** Cameron-Gelbach-Miller (2011) two-way clustering

**Example:** State-year panel:
- Cluster by state (within-state correlation over time)
- Cluster by year (cross-state correlation in same year)

**Key Concept:** Two-way clustering addresses both serial and cross-sectional dependence.

---

### Question 5: Answer B (5 points)

**Correct Answer:** B) Large $N$ (many clusters)

**Explanation:**

**Asymptotic requirement:** $N \to \infty$ (number of clusters)

**Intuition:**
- Clustering treats each cluster as one independent observation
- Need many independent clusters for CLT to apply
- $T$ within cluster can be small or large

**Rule of thumb:**
- $N < 30$: Cluster bootstrap or wild cluster bootstrap
- $30 \leq N < 50$: Use with caution
- $N \geq 50$: Standard clustered SEs generally reliable

**Common mistake:** Having large $NT$ but small $N$
- Example: $N=10$ firms, $T=100$ years → Only 10 clusters (problematic)

**Why other answers are wrong:**
- A: Large $T$ not required (but more power with larger $T$)
- C: Only $N$ needs to be large
- D: $NT$ could be large with small $N$ (insufficient)

**Key Concept:** Cluster asymptotics require many clusters, not many observations per cluster.

---

### Question 6: Answer B (8 points)

**Correct Answer:** B) $y_{i,t-1}$ is correlated with $\alpha_i$, creating endogeneity even after within transformation

**Explanation:**

**The problem:**

$$y_{i,t-1} = \gamma y_{i,t-2} + X_{i,t-1}\beta + \alpha_i + \epsilon_{i,t-1}$$

Since $y_{i,t-1}$ depends on $\alpha_i$:

$$Cov(y_{i,t-1}, \alpha_i) \neq 0$$

**After within transformation:**

$$\tilde{y}_{it} = \gamma \tilde{y}_{i,t-1} + \tilde{X}_{it}\beta + \tilde{\epsilon}_{it}$$

where $\tilde{y}_{i,t-1} = y_{i,t-1} - \bar{y}_i$ and $\bar{y}_i = \frac{1}{T}\sum_{t=1}^T y_{it}$.

**Problem persists:**
- $\bar{y}_i$ includes $y_{i,t-1}$ itself and other values
- $\tilde{y}_{i,t-1}$ correlates with $\tilde{\epsilon}_{it}$ through the demeaning
- Mechanical correlation from transformation

**Why other answers are wrong:**
- A: Not perfectly collinear (both vary)
- C: Time FE don't solve the endogeneity
- D: Model is identified with proper instruments

**Key Concept:** Lagged dependent variable plus FE creates endogeneity via demeaning.

---

### Question 7: Answer B (8 points)

**Correct Answer:** B) Causes $\hat{\gamma}$ to be biased downward (toward zero)

**Explanation:**

**Nickell (1981) bias:** FE estimator of $\gamma$ in dynamic panels is biased toward zero.

**Magnitude:**

$$Bias(\hat{\gamma}_{FE}) \approx -\frac{1 + \gamma}{T-1}$$

**Properties:**
- **Direction:** Negative (downward bias)
- **Magnitude:** Decreases with $T$: $O(1/T)$
- **For large $T$:** Bias → 0

**Example:**
- True $\gamma = 0.8$, $T = 5$: $Bias \approx -0.45$ → $E[\hat{\gamma}] \approx 0.35$
- True $\gamma = 0.8$, $T = 20$: $Bias \approx -0.09$ → $E[\hat{\gamma}] \approx 0.71$

**Why other answers are wrong:**
- A: Bias is $O(1/T)$, so it decreases with $T$ but isn't exactly zero until $T \to \infty$
- C: Bias decreases (not increases) with $T$
- D: Occurs in balanced and unbalanced panels

**Key Concept:** Nickell bias is $O(1/T)$ and biases $\gamma$ downward.

---

### Question 8: Answer A (8 points)

**Correct Answer:** A) $\Delta y_{i,t-1} = y_{i,t-1} - y_{i,t-2}$ contains $y_{i,t-1}$, which correlates with $\Delta\epsilon_{it} = \epsilon_{it} - \epsilon_{i,t-1}$

**Explanation:**

**The endogeneity:**

$$\Delta y_{i,t-1} = y_{i,t-1} - y_{i,t-2}$$
$$\Delta\epsilon_{it} = \epsilon_{it} - \epsilon_{i,t-1}$$

**Correlation:**

$$Cov(\Delta y_{i,t-1}, \Delta\epsilon_{it}) = Cov(y_{i,t-1} - y_{i,t-2}, \epsilon_{it} - \epsilon_{i,t-1})$$

The term $y_{i,t-1}$ was determined by $\epsilon_{i,t-1}$ (from the original equation), so:

$$Cov(y_{i,t-1}, \epsilon_{i,t-1}) \neq 0$$

**Result:** $\Delta y_{i,t-1}$ and $\Delta\epsilon_{it}$ are correlated → OLS is inconsistent.

**Solution:** Use instrumental variables (Arellano-Bond approach).

**Why other answers are wrong:**
- B: First-differencing doesn't create serial correlation (reveals it)
- C: Intercept elimination doesn't create inconsistency
- D: FALSE - OLS on first-differences is inconsistent

**Key Concept:** Mechanical correlation between $\Delta y_{i,t-1}$ and $\Delta\epsilon_{it}$ requires IV.

---

### Question 9: Answer B (8 points)

**Correct Answer:** B) $y_{i,t-2}, y_{i,t-3}, ...$

**Explanation:**

**Instrument validity:** Need instruments uncorrelated with $\Delta\epsilon_{it} = \epsilon_{it} - \epsilon_{i,t-1}$.

**Valid instruments:** Lags of $y$ from period $t-2$ and earlier:
- $y_{i,t-2}$ was determined before $\epsilon_{i,t-1}$ and $\epsilon_{it}$
- Under no serial correlation: $E[y_{i,t-2} \cdot \epsilon_{it}] = 0$ and $E[y_{i,t-2} \cdot \epsilon_{i,t-1}] = 0$

**Invalid instruments:**
- $y_{i,t-1}$: Correlated with $\epsilon_{i,t-1}$ (part of $\Delta\epsilon_{it}$)
- $\Delta y_{i,t-2}$: Can be used but less efficient than levels

**Why other answers are wrong:**
- A: First-differences can be instruments but are typically weaker
- C: Lags of $X$ can also be instruments if predetermined, but question asks about $y$
- D: $\alpha_i$ is eliminated by differencing

**Key Concept:** Use levels lagged 2+ periods as instruments for differenced lagged dependent variable.

---

### Question 10: Answer A (8 points)

**Correct Answer:** A) Equations in differences and equations in levels, using different instrument sets for each

**Explanation:**

**System GMM** (Blundell-Bond 1998) combines:

1. **Difference equations:**
   $$\Delta y_{it} = \gamma \Delta y_{i,t-1} + \Delta X_{it}\beta + \Delta\epsilon_{it}$$
   - Instruments: Lagged levels ($y_{i,t-2}, y_{i,t-3}, ...$)

2. **Level equations:**
   $$y_{it} = \gamma y_{i,t-1} + X_{it}\beta + \alpha_i + \epsilon_{it}$$
   - Instruments: Lagged differences ($\Delta y_{i,t-1}, \Delta y_{i,t-2}, ...$)

**Additional assumption for level equations:**

$$E[\alpha_i \cdot \Delta y_{i,t-1}] = 0$$

(Stationarity condition)

**Advantage:** More instruments, especially useful when series are persistent (difference GMM has weak instruments).

**Why other answers are wrong:**
- B, C: Not combinations of FE/RE or within/between
- D: Not about OLS and 2SLS combination

**Key Concept:** System GMM adds level equations with lagged differences as instruments.

---

### Question 11: Answer A (7 points)

**Correct Answer:** A) Whether the instruments are valid (uncorrelated with errors)

**Explanation:**

**Sargan/Hansen test** of overidentifying restrictions:

$$H_0: E[Z_i' \epsilon_i] = 0 \text{ (all instruments valid)}$$

**When applicable:**
- Model is overidentified: more instruments than endogenous variables
- Tests the **validity** of the extra (overidentifying) instruments

**Test statistic:**
- Sargan: $J = n \cdot J_n \sim \chi^2_r$
- Hansen: Robust version
- $r$ = number of overidentifying restrictions

**Interpretation:**
- **Fail to reject:** Instruments appear valid
- **Reject:** At least some instruments are invalid (correlated with errors)

**Why other answers are wrong:**
- B: Rank condition checks if enough instruments; Sargan checks if they're valid
- C: Dynamic stability is different (check eigenvalues of $\gamma$)
- D: This is separate from FE inclusion

**Key Concept:** Sargan/Hansen tests overidentifying restrictions (instrument validity).

---

### Question 12: Answer A (7 points)

**Correct Answer:** A) AR(1) in differences is expected by construction, but AR(2) indicates misspecification

**Explanation:**

**Why AR(1) in differences is expected:**

Even if $\epsilon_{it}$ has no serial correlation in levels:

$$\Delta\epsilon_{it} = \epsilon_{it} - \epsilon_{i,t-1}$$
$$\Delta\epsilon_{i,t-1} = \epsilon_{i,t-1} - \epsilon_{i,t-2}$$

These share $\epsilon_{i,t-1}$ → mechanically correlated.

**AR(2) test:**

$$E[\Delta\epsilon_{it} \cdot \Delta\epsilon_{i,t-2}] = E[(\epsilon_{it} - \epsilon_{i,t-1}) \cdot (\epsilon_{i,t-2} - \epsilon_{i,t-3})]$$

If no serial correlation in levels, this should be zero.

**Interpretation:**
- **Fail to reject AR(2):** No serial correlation (good)
- **Reject AR(2):** Serial correlation in levels → instruments invalid

**Why other answers are wrong:**
- B, C, D: Not about statistical ease, importance, or automatic correction

**Key Concept:** Test AR(2) in differences to detect AR(1) in levels.

---

### Question 13: Answer A (7 points)

**Correct Answer:** A) Overfit the endogenous variables and weaken the Sargan/Hansen test

**Explanation:**

**Instrument proliferation problem:**

With many instruments (especially in long panels with many lags):

1. **Overfitting:**
   - Instruments fit endogenous variables too well
   - Biases estimates toward OLS (inconsistent)
   - "Too many instruments" problem

2. **Weak Sargan/Hansen test:**
   - Test loses power
   - Fails to reject even when instruments are invalid
   - False sense of security

**Rule of thumb:**
- Number of instruments should be ≤ number of clusters
- Collapse or limit instrument lags

**Solutions:**
- Limit lag depth
- Collapse instrument matrix
- Use subset of available instruments

**Why other answers are wrong:**
- B: Overidentification, not perfect identification
- C: Actually reduces efficiency
- D: Computational cost increases

**Key Concept:** Too many instruments cause overfitting and weak specification tests.

---

### Question 14: Answer A (7 points)

**Correct Answer:** A) Difference GMM has weak instruments when variables are persistent

**Explanation:**

**Weak instrument problem in difference GMM:**

When $y_{it}$ is highly persistent (AR coefficient close to 1):
- Lagged levels are weak predictors of first-differences
- $Corr(y_{i,t-2}, \Delta y_{i,t-1})$ is small
- Large standard errors, imprecise estimates

**Why system GMM helps:**
- Adds level equations with lagged differences as instruments
- Lagged differences are better predictors when series is persistent
- More efficient estimation

**Example:**
- If $y_{it} \approx y_{i,t-1}$ (random walk): $\Delta y_{i,t-1} \approx 0$, hard to predict with $y_{i,t-2}$
- But $y_{i,t-1}$ easier to predict with $\Delta y_{i,t-2}$

**Why other answers are wrong:**
- B: System GMM still requires instruments
- C: Both work with unbalanced panels
- D: Difference GMM can handle lagged DV (that's its purpose)

**Key Concept:** System GMM overcomes weak instrument problem when series are persistent.

---

### Question 15: Answer A (7 points)

**Correct Answer:** A) $E[\epsilon_{it} | y_{i,t-2}, y_{i,t-3}, ..., y_{i1}] = 0$ (sequential exogeneity)

**Explanation:**

**Sequential exogeneity assumption:**

$$E[\epsilon_{it} | y_{i,t-j}] = 0 \text{ for all } j \geq 2$$

**Interpretation:**
- Errors are uncorrelated with sufficiently lagged $y$
- No serial correlation in $\epsilon_{it}$
- Past values of $y$ are valid instruments

**Weaker than strict exogeneity:** Allows feedback from $y$ to future $X$, but not to future $\epsilon$.

**Violated by:**
- Serial correlation in errors
- Measurement error in $y$

**Why other answers are wrong:**
- B: Too weak (need lagged $y$ exogeneity)
- C: We allow serial correlation after first-differencing (test AR(2))
- D: Works with unbalanced panels

**Key Concept:** GMM requires sequential exogeneity of errors with respect to instruments.

---

### Question 16: Solution (15 points)

**Part A: SE Ratio and Interpretation (5 points)**

**SE Ratio:**

$$\frac{SE_{cluster}}{SE_{OLS}} = \frac{0.45}{0.20} = 2.25$$

**Interpretation:**

The clustered standard errors are **2.25 times larger** than OLS standard errors.

Using the approximation:

$$SE_{cluster} \approx SE_{OLS} \sqrt{1 + (T-1)\rho}$$

$$2.25 \approx \sqrt{1 + 9\rho}$$

$$5.06 \approx 1 + 9\rho$$

$$\rho \approx 0.45$$

**Conclusion:** This suggests approximately **45% intraclass correlation** in errors within workers. Errors are highly correlated within workers over time, making clustering essential.

**Scoring:**
- 2 points for ratio calculation
- 3 points for interpretation

---

**Part B: T-statistics (5 points)**

**Using OLS standard errors:**

$$t_{OLS} = \frac{\hat{\beta}}{SE_{OLS}} = \frac{0.60}{0.20} = 3.0$$

**Using clustered standard errors:**

$$t_{cluster} = \frac{\hat{\beta}}{SE_{cluster}} = \frac{0.60}{0.45} = 1.33$$

**Difference:** t-statistic is **2.25 times smaller** with proper clustering.

**Scoring:**
- 2.5 points for OLS t-statistic
- 2.5 points for clustered t-statistic

---

**Part C: Hypothesis Testing (5 points)**

**Decision at 5% level (critical value ≈ 2.0):**

**Using OLS SEs:**
- $t_{OLS} = 3.0 > 2.0$
- **Reject $H_0: \beta = 0$**
- Conclude significant effect

**Using clustered SEs:**
- $t_{cluster} = 1.33 < 2.0$
- **Fail to reject $H_0: \beta = 0$**
- Conclude no significant effect

**Illustration:**

This demonstrates **dramatic impact** of proper standard errors:
- OLS SEs lead to false rejection (Type I error)
- Ignoring within-worker correlation overestimates precision
- Inference completely changes with correct SEs
- This is why clustering is critical in panel data

**Real-world implication:** A researcher using OLS SEs would incorrectly conclude a significant relationship. Proper clustering reveals the estimate is not statistically significant.

**Scoring:**
- 2 points for OLS decision
- 2 points for clustered decision
- 1 point for interpretation

**Key Concept:** Ignoring clustering dramatically inflates t-statistics and leads to false significance.

---

### Question 17: Solution (20 points)

**Part A: Valid Instruments for Period 4 (7 points)**

For the equation in period $t=4$:

$$\Delta y_{i4} = \gamma \Delta y_{i3} + \Delta\epsilon_{i4}$$

where $\Delta y_{i3} = y_{i3} - y_{i2}$ and $\Delta\epsilon_{i4} = \epsilon_{i4} - \epsilon_{i3}$.

**Instrument: $y_{i2}$**

**Analysis:**

$$Cov(y_{i2}, \Delta\epsilon_{i4}) = Cov(y_{i2}, \epsilon_{i4} - \epsilon_{i3})$$

- $Cov(y_{i2}, \epsilon_{i4}) = 0$ ✓ (no serial correlation assumption)
- $Cov(y_{i2}, \epsilon_{i3})$: **Problem!**

Since $y_{i2}$ depends on $\epsilon_{i2}$, and we're evaluating:

Actually, under no serial correlation: $E[\epsilon_{i3}|y_{i2}] = 0$ since $y_{i2}$ was determined in period 2.

**Verdict: $y_{i2}$ is VALID** ✓

**Instrument: $y_{i1}$**

**Analysis:**

$$Cov(y_{i1}, \Delta\epsilon_{i4}) = Cov(y_{i1}, \epsilon_{i4} - \epsilon_{i3}) = 0$$

Under no serial correlation, $y_{i1}$ is predetermined relative to $\epsilon_{i3}$ and $\epsilon_{i4}$.

**Verdict: $y_{i1}$ is VALID** ✓

**Summary for period 4:**
- Valid instruments: $y_{i2}, y_{i1}$
- Invalid: $y_{i3}$ (contains $\epsilon_{i3}$ which is part of $\Delta\epsilon_{i4}$)

**Scoring:**
- 3.5 points for $y_{i2}$ analysis
- 3.5 points for $y_{i1}$ analysis

---

**Part B: Instruments for Period 5 (7 points)**

For the equation in period $t=5$:

$$\Delta y_{i5} = \gamma \Delta y_{i4} + \Delta\epsilon_{i5}$$

where $\Delta\epsilon_{i5} = \epsilon_{i5} - \epsilon_{i4}$.

**Valid instruments:**

Need $Cov(y_{is}, \Delta\epsilon_{i5}) = Cov(y_{is}, \epsilon_{i5} - \epsilon_{i4}) = 0$

**Analysis:**
- **$y_{i3}$:** VALID ✓ (determined before $\epsilon_{i4}$ and $\epsilon_{i5}$)
- **$y_{i2}$:** VALID ✓ (even earlier)
- **$y_{i1}$:** VALID ✓ (even earlier)
- **$y_{i4}$:** INVALID ✗ (contains $\epsilon_{i4}$)

**Valid instrument set for period 5:** $\{y_{i3}, y_{i2}, y_{i1}\}$

**Scoring:**
- 2 points for each correct instrument identification
- 1 point for explaining why $y_{i4}$ is invalid

---

**Part C: Instrument Growth and Proliferation (6 points)**

**Pattern of instruments:**

- Period $t=3$: $\{y_{i1}\}$ (1 instrument)
- Period $t=4$: $\{y_{i2}, y_{i1}\}$ (2 instruments)
- Period $t=5$: $\{y_{i3}, y_{i2}, y_{i1}\}$ (3 instruments)
- Period $t$: $\{y_{i,t-2}, y_{i,t-3}, ..., y_{i1}\}$ (t-2 instruments)

**Total instruments for $T$ periods:** $\sum_{t=3}^T (t-2) = 1 + 2 + ... + (T-2) = \frac{(T-2)(T-1)}{2}$

**Growth rate:** Quadratic in $T$!

Example:
- $T=5$: 6 instruments
- $T=10$: 36 instruments
- $T=20$: 171 instruments

**Is this always beneficial? NO.**

**Problems:**
1. **Instrument proliferation:** Too many instruments relative to clusters
2. **Overfitting:** Instruments fit endogenous variables too well
3. **Weak Sargan test:** Loss of power to detect invalid instruments
4. **Computation:** Large matrices to invert

**Solutions:**
- **Limit lag depth:** Use only $y_{i,t-2}$ and $y_{i,t-3}$ instead of all lags
- **Collapse instruments:** Combine lags into fewer moment conditions
- **Rule of thumb:** Keep instruments ≤ number of clusters

**Scoring:**
- 2 points for explaining instrument growth pattern
- 2 points for identifying it's not always beneficial
- 2 points for explaining proliferation problem

**Key Concept:** Instruments grow quadratically with $T$, creating proliferation problems.

---

## Scoring Rubric

| Points | Grade | Interpretation |
|--------|-------|----------------|
| 90-100 | A     | Excellent mastery of advanced topics |
| 80-89  | B     | Good understanding |
| 70-79  | C     | Satisfactory |
| 60-69  | D     | Needs review |
| 0-59   | F     | Significant gaps |

## Learning Objectives Assessed

- **LO1:** Understand and apply clustered standard errors (Q1-Q5, Q16)
- **LO2:** Recognize dynamic panel bias and solutions (Q6-Q10, Q17)
- **LO3:** Apply GMM diagnostics (Q11-Q15)
- **LO4:** Implement Arellano-Bond estimation (Q9, Q17)

## Study Resources

If you scored below 70%, review:
- Module 5 Guide: Clustered Errors
- Module 5 Guide: Dynamic Panels
- Module 5 Guide: GMM Estimation
- Module 5 Notebook: Dynamic Models

**Time to Review:** 4-5 hours

## Common Mistakes

1. **Confusing clustering with robustness** (clustering is for correlation, not heteroskedasticity alone)
2. **Wrong asymptotic assumption** (need large $N$, not large $T$, for clustering)
3. **Misunderstanding Nickell bias** (it's downward, decreases with $T$)
4. **Invalid instruments in dynamic panels** ($y_{i,t-1}$ invalid for $\Delta\epsilon_{it}$)
5. **AR(1) vs AR(2) testing** (test AR(2) in differences to detect problems)
6. **Instrument proliferation** (more instruments not always better)
7. **Difference vs System GMM** (system better when persistent series)

## Advanced Topics Summary

**Key Takeaways:**
1. **Always cluster by entity** in panel data (unless $\rho \approx 0$)
2. **Dynamic panels require IV/GMM** (FE is biased)
3. **Test instrument validity** (Sargan/Hansen, AR tests)
4. **Limit instruments** to avoid proliferation
5. **Prefer system GMM** for persistent series
