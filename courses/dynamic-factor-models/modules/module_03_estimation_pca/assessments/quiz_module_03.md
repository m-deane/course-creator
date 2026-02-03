# Module 3 Quiz: PCA Estimation Methods

**Time Estimate:** 50-65 minutes
**Total Points:** 100
**Passing Score:** 70%

## Instructions

This quiz assesses your understanding of principal components estimation for factor models, including the Stock-Watson two-step estimator, Bai-Ng information criteria, and handling missing data. Answer all questions. You have 2 attempts, and your highest score will be recorded.

---

## Part A: Conceptual Understanding (40 points)

### Question 1 (4 points) - Foundation

The Stock-Watson two-step estimator consists of:

A) Step 1: Estimate loadings via OLS; Step 2: Estimate factors via PCA
B) Step 1: Extract factors via PCA; Step 2: Estimate factor VAR via OLS
C) Step 1: Kalman filter; Step 2: Maximum likelihood
D) Step 1: Choose number of factors; Step 2: Estimate parameters

**Correct Answer:** B

**Feedback:**
- A: This reverses the order. We extract factors first (PCA), then estimate dynamics.
- B: **Correct**. Step 1 uses PCA on the covariance matrix to extract factors $\hat{F}$. Step 2 regresses $\hat{F}_t$ on lags $\hat{F}_{t-1}, ..., \hat{F}_{t-p}$ using OLS to estimate the VAR.
- C: Kalman filter and maximum likelihood are alternative methods, not the Stock-Watson approach.
- D: Too vague; doesn't specify the actual estimation methods.

---

### Question 2 (4 points) - Core

Why is the Stock-Watson PCA approach computationally efficient compared to maximum likelihood?

A) It requires fewer observations
B) It avoids iterative optimization—both steps have closed-form solutions
C) It uses fewer factors
D) It doesn't require standardization

**Correct Answer:** B

**Feedback:**
- A: Both methods require the same data; sample size requirements are similar.
- B: **Correct**. PCA solves an eigenvalue problem (closed-form), and OLS has a closed-form solution. ML requires iterative optimization of the likelihood function via numerical methods (BFGS, EM), which is much slower.
- C: The number of factors is a modeling choice, independent of estimation method.
- D: Both methods typically standardize variables for proper scaling.

---

### Question 3 (4 points) - Core

Under what conditions are PCA estimates of factors **consistent** (converge to true factors)?

A) Only if factors are Gaussian and $\Sigma_e$ is diagonal (exact factor model)
B) As long as $T \to \infty$, regardless of $N$
C) As $N, T \to \infty$ jointly, even with approximate factor structure (weak cross-correlation in $e_t$)
D) Only if all variables load equally on all factors

**Correct Answer:** C

**Feedback:**
- A: Consistency holds more generally; Gaussianity and exact factor structure are not required.
- B: Need both $N$ and $T$ large. With fixed $N$, factor space cannot be reliably identified.
- C: **Correct**. Bai (2003) shows PCA is consistent under large $N$ and $T$ asymptotics with approximate factors—idiosyncratic errors can be weakly correlated as long as they don't dominate. The convergence rate is $\min(\sqrt{N}, \sqrt{T})$.
- D: Equal loadings are not required; factors just need to be pervasive (affecting a non-negligible fraction of variables).

---

### Question 4 (4 points) - Core

The PCA normalization imposes $\frac{1}{T}\hat{F}'\hat{F} = I_r$. What does this achieve?

A) Makes factors uncorrelated with each other and sets their sample variance to 1
B) Ensures factors are Gaussian
C) Guarantees stationarity
D) Identifies factor signs uniquely

**Correct Answer:** A

**Feedback:**
- A: **Correct**. $F'F/T = I_r$ means factors have unit sample variance and zero sample covariance (orthonormal). This is the PCA normalization that identifies factor scale.
- B: Normalization doesn't impose distributional assumptions; factors can have any distribution.
- C: Stationarity is a time-series property of the VAR, not ensured by normalization.
- D: Sign identification requires additional steps (reference variable normalization); PCA normalization alone leaves signs arbitrary.

---

### Question 5 (4 points) - Advanced

In the Bai-Ng information criteria for selecting the number of factors, the penalty term grows with:

A) $N$ only
B) $T$ only
C) Both $N$ and $T$
D) The residual variance

**Correct Answer:** C

**Feedback:**
- A: If the penalty only grew with $N$, it wouldn't account for sample size $T$.
- B: If only $T$, it wouldn't account for cross-sectional dimension.
- C: **Correct**. Bai-Ng criteria have penalty terms like $g(N,T) = \frac{N+T}{NT}\log(\text{something})$, which depend on both dimensions. This accounts for the double asymptotics ($N, T \to \infty$).
- D: Residual variance is what the criterion minimizes, not what the penalty depends on.

---

### Question 6 (4 points) - Core

A scree plot shows eigenvalues: $[42.3, 18.7, 9.2, 2.8, 2.5, 2.1, 1.9, 1.8, ...]$ with remaining values around 1.5. Using the scree plot, you would select:

A) $r = 1$ (largest eigenvalue)
B) $r = 3$ (clear elbow before the 4th eigenvalue)
C) $r = 8$ (all shown eigenvalues)
D) $r = 5$ (median of reasonable range)

**Correct Answer:** B

**Feedback:**
- A: This ignores clearly important factors 2 and 3.
- B: **Correct**. The eigenvalues show a sharp drop after the 3rd: [42.3, 18.7, 9.2] are large, then [2.8, 2.5, ...] are much smaller and flat. The "elbow" indicates $r = 3$ pervasive factors.
- C: Including all 8 would capture noise; factors 4-8 are close to the remaining eigenvalues (~1.5-2).
- D: There's no statistical basis for choosing the median; use the elbow.

---

### Question 7 (4 points) - Advanced

When estimating a factor model on FRED-MD ($N = 127$ monthly series, $T = 500$), you apply three Bai-Ng criteria and get: IC1 selects $r = 5$, IC2 selects $r = 4$, IC3 selects $r = 6$. What should you do?

A) Average to $r = 5$
B) Choose the smallest ($r = 4$) to avoid overfitting
C) Report all three and assess robustness across $r = 4, 5, 6$
D) Reject the factor model since criteria disagree

**Correct Answer:** C

**Feedback:**
- A: Averaging makes no statistical sense; you can't have 5 factors.
- B: While parsimony is valuable, automatically choosing the smallest is too mechanical.
- C: **Correct**. Disagreement of 1-2 factors is common and indicates uncertainty about the number of weak factors. Best practice: estimate models for $r = 4, 5, 6$ and check if results (forecasts, interpretations) are robust.
- D: Mild disagreement doesn't invalidate the factor structure; the criteria all suggest 4-6 factors, not wildly different numbers.

---

### Question 8 (4 points) - Core

You standardize variables before PCA (mean 0, variance 1). After estimation, you find Factor 1 explains 35% of variance. What does this mean?

A) Factor 1 loads on 35% of the variables
B) Factor 1's eigenvalue is 35% of the trace of the correlation matrix
C) 35% of time-series variation is explained
D) The first loading is 0.35

**Correct Answer:** B

**Feedback:**
- A: Variance explained doesn't directly translate to variable count; all variables may load, just weakly for most.
- B: **Correct**. For standardized data, total variance = $N$ (sum of $N$ unit variances). Factor 1 eigenvalue $d_1 = 0.35N$ means it explains $d_1/N = 35\%$ of total variance, which is $35\%$ of the trace of the correlation matrix.
- C: Variance explained refers to cross-sectional variance at each $t$, not time-series variation over $t$.
- D: Variance explained is not the same as a loading value; it's derived from the eigenvalue.

---

### Question 9 (4 points) - Advanced

In the presence of missing data, the "EM algorithm for PCA" alternates between:

A) E-step: Impute missing $X$ using current factors; M-step: Re-estimate factors via PCA on imputed data
B) E-step: Estimate factor variances; M-step: Estimate loadings
C) E-step: Kalman filtering; M-step: Kalman smoothing
D) E-step: Select factors; M-step: Estimate VAR

**Correct Answer:** A

**Feedback:**
- A: **Correct**. E-step fills in missing observations using $\hat{X}_{it} = \hat{\lambda}_i'\hat{F}_t$ from current estimates. M-step performs PCA on the completed data matrix to update factors and loadings.
- B: This is part of the M-step for likelihood-based EM, but not specific to PCA with missing data.
- C: Kalman filter/smoother are for state-space models, not the PCA-EM algorithm.
- D: Factor selection and VAR estimation are separate steps, not the EM iteration structure.

---

### Question 10 (4 points) - Foundation

Why must variables be **standardized** before applying PCA in factor models?

A) To ensure factors are Gaussian
B) To prevent variables with larger variances from dominating the principal components
C) To make loadings interpretable as correlations
D) To satisfy the exact factor model assumption

**Correct Answer:** B

**Feedback:**
- A: Standardization doesn't affect distributional properties.
- B: **Correct**. PCA finds directions of maximum variance. If variable $i$ has variance 1000 and variable $j$ has variance 1, the first PC will be dominated by variable $i$. Standardizing ensures all variables are weighted equally.
- C: Loadings in PCA aren't automatically correlations even after standardizing (that requires specific normalization).
- D: Exact vs approximate factor structure is independent of standardization.

---

## Part B: Mathematical Application (30 points)

### Question 11 (6 points) - Core

You compute PCA on standardized data ($N = 50$) and obtain eigenvalues: $[22.5, 10.3, 7.1, 3.2, 2.8, ...]$. If you extract $r = 3$ factors, what proportion of total variance is explained?

A) 22.5%
B) 40.0%
C) 79.8%
D) 90.0%

**Correct Answer:** C

**Feedback:**
- A: This is only Factor 1's contribution.
- B: This would be $(22.5 + 10.3)/50$, missing Factor 3.
- C: **Correct**. Total variance for standardized data is $N = 50$. Top 3 eigenvalues sum to $22.5 + 10.3 + 7.1 = 39.9$. Proportion: $39.9 / 50 = 0.798 = 79.8\%$.
- D: Would require eigenvalues summing to 45, which they don't.

---

### Question 12 (6 points) - Core

For a factor model estimated via PCA, loading estimates are $\hat{\Lambda} = X'\hat{F}/T$. If $\hat{F}$ satisfies $\hat{F}'\hat{F}/T = I_r$, what is $\hat{\Lambda}' \hat{\Lambda}$?

A) $I_r$
B) Diagonal matrix with eigenvalues on diagonal (scaled by $T$)
C) $\Sigma_X$ (sample covariance)
D) Cannot determine without knowing $N$

**Correct Answer:** B

**Feedback:**
- A: This would be true if $\hat{\Lambda}$ were orthonormal, but loadings are not generally orthogonal.
- B: **Correct**. From PCA theory: $\hat{\Lambda} = \sqrt{T} V_r D_r^{1/2}$ where $V_r$ are eigenvectors (orthonormal) and $D_r = \text{diag}(d_1, ..., d_r)$ are eigenvalues. So $\hat{\Lambda}'\hat{\Lambda} = T \cdot D_r$.
- C: $\Sigma_X = \hat{\Lambda}\hat{\Lambda}' + \hat{\Sigma}_e$ includes idiosyncratic variance, not just loadings squared.
- D: The dimension $N$ affects the size, but the structure is determined.

---

### Question 13 (6 points) - Advanced

You estimate a VAR(1) on extracted factors: $\hat{F}_t = \hat{\Phi}\hat{F}_{t-1} + \hat{\eta}_t$. Using $T = 200$ observations and $r = 3$ factors, how many parameters are estimated in the VAR?

A) 3 (one per factor)
B) 9 ($r^2$ for $\Phi$)
C) 12 ($r^2$ for $\Phi$ plus $r$ intercepts)
D) 15 ($r^2$ for $\Phi$ plus $r(r+1)/2$ for innovation covariance)

**Correct Answer:** C

**Feedback:**
- A: Each factor has its own equation, but each equation has multiple coefficients.
- B: Close, but forgets the intercept terms.
- C: **Correct**. Each of $r = 3$ equations has $r + 1 = 4$ coefficients (intercept + 3 lagged factors). Total: $3 \times 4 = 12$ parameters. (Or: $r$ intercepts + $r^2$ slope coefficients = $3 + 9 = 12$.)
- D: The innovation covariance is not typically estimated separately in the Stock-Watson approach (just use residual covariance).

---

### Question 14 (6 points) - Advanced

For large $N$ and $T$, PCA factor estimates converge at rate $\min(\sqrt{N}, \sqrt{T})$. If $N = 100$ and $T = 400$, what is the effective convergence rate?

A) $\sqrt{100} = 10$
B) $\sqrt{400} = 20$
C) $\sqrt{100 \times 400} = 200$
D) $\min(10, 20) = 10$

**Correct Answer:** D

**Feedback:**
- A: This is $\sqrt{N}$, but need to compare with $\sqrt{T}$.
- B: This is $\sqrt{T}$, but need to take the minimum.
- C: This incorrectly multiplies $N$ and $T$.
- D: **Correct**. The convergence rate is $\min(\sqrt{N}, \sqrt{T}) = \min(\sqrt{100}, \sqrt{400}) = \min(10, 20) = 10$. Here $N$ is the binding constraint.

---

### Question 15 (6 points) - Core

You extract 4 factors via PCA and estimate a VAR(2). To forecast 1 period ahead, you need:

A) Only $\hat{F}_T$ (most recent factors)
B) $\hat{F}_T$ and $\hat{F}_{T-1}$ (last two periods)
C) All $\hat{F}_1, ..., \hat{F}_T$
D) The VAR coefficients only

**Correct Answer:** B

**Feedback:**
- A: VAR(2) requires two lags, not just one.
- B: **Correct**. VAR(2) forecast: $\hat{F}_{T+1} = \hat{\Phi}_1 \hat{F}_T + \hat{\Phi}_2 \hat{F}_{T-1}$. Need the last two observations.
- C: The entire history was used to estimate $\hat{\Phi}$, but only the last $p = 2$ observations are needed for forecasting.
- D: Need both coefficients and recent factor values.

---

## Part C: Practical Application (30 points)

### Question 16 (6 points) - Core

You estimate a factor model on a panel with 20% missing values scattered randomly across variables and time. You compare:
- Method A: Listwise deletion (drop any time period with any missing value)
- Method B: EM algorithm for PCA (iteratively impute missing values)

Which is preferred and why?

A) Method A, because it avoids bias from imputation
B) Method B, because it uses all available data and is more efficient
C) They are equivalent if missingness is random
D) Neither; you must collect complete data

**Correct Answer:** B

**Feedback:**
- A: Listwise deletion with scattered missingness will discard most/all observations, creating huge efficiency loss or making estimation impossible.
- B: **Correct**. With random missingness, EM algorithm (or Kalman filter for state-space) provides consistent estimates while using all available information. Listwise deletion throws away useful data.
- C: Even with random missingness, efficiency differs substantially—EM is much better.
- D: In modern applications, missing data is ubiquitous (ragged edges, publication delays). Methods that handle missingness are essential.

---

### Question 17 (6 points) - Core

You estimate factors on data from 1980-2010, then apply the same loadings to 2010-2020 data to extract "out-of-sample factors." A colleague claims this is invalid. Is the colleague correct?

A) Yes, factors must be re-estimated each period
B) No, once loadings are estimated, they can be applied to new data
C) Yes, because factor identification will differ across samples
D) No, but you should re-standardize the new data first

**Correct Answer:** D

**Feedback:**
- A: Too strong. Loadings can be used out-of-sample if the structure is stable.
- B: Close, but misses the standardization issue.
- C: Identification is set by the original estimation; applying the same loadings maintains consistent identification.
- D: **Correct**. You can project new data onto the original factor space using $\hat{F}_{\text{new}} = X_{\text{new}}\hat{\Lambda}(\hat{\Lambda}'\hat{\Lambda})^{-1}$. However, you must standardize the new data using the *same* means and standard deviations from the training period for consistency.

---

### Question 18 (6 points) - Advanced

You compare PCA factors extracted from two subperiods (1980-2000 and 2000-2020). The first factor's loadings are strongly correlated (0.85), but the second factor's loadings are weakly correlated (0.30). What does this suggest?

A) The factor model is misspecified
B) The first factor is more stable/persistent; the second factor may have changed or rotated across periods
C) Sample size is too small
D) PCA failed in one of the periods

**Correct Answer:** B

**Feedback:**
- A: Instability in one factor doesn't necessarily invalidate the entire model.
- B: **Correct**. High correlation (0.85) for Factor 1 suggests it represents a stable structural feature (e.g., business cycle). Low correlation (0.30) for Factor 2 suggests either: structural change, rotation of secondary factors, or Factor 2 being less economically meaningful.
- C: With 20 years each, sample size should be adequate (T ~ 240 monthly observations).
- D: Low correlation doesn't mean PCA "failed"—both extractions are valid, but the economic structure may have shifted.

---

### Question 19 (6 points) - Core

You extract 5 factors from $N = 100$ variables. The first factor has loadings ranging from -0.05 to 0.08 (all very small). The eigenvalue is 3.2. What should you conclude?

A) This factor is economically unimportant and possibly noise
B) This factor is highly significant because the eigenvalue is > 1
C) All variables load equally, indicating a balanced factor
D) The loadings need to be rescaled

**Correct Answer:** A

**Feedback:**
- A: **Correct**. Eigenvalue 3.2 out of 100 variables (3.2%) is quite small. Combined with uniformly weak loadings (all < 0.1), this suggests the factor captures little meaningful variation—likely noise rather than a pervasive factor.
- B: The threshold "eigenvalue > 1" is for factor analysis with unstandardized data, not directly applicable here. With $N = 100$ standardized variables, an eigenvalue of 3.2 is small.
- C: Balanced loadings would be ~0.1 if equally weighted, but here they're close to zero, indicating the factor explains almost nothing for any variable.
- D: Rescaling doesn't change the fundamental issue: this factor has negligible explanatory power.

---

### Question 20 (6 points) - Advanced

You estimate a Stock-Watson model and compute 10-step-ahead factor forecasts. The forecast standard errors grow approximately **linearly** with horizon. What does this indicate about the factor dynamics?

A) Factors are stationary and persistent
B) Factors follow a random walk (non-stationary)
C) The VAR is misspecified
D) Forecast errors are autocorrelated

**Correct Answer:** B

**Feedback:**
- A: For stationary VAR, forecast uncertainty converges to a finite limit (unconditional variance), not growing linearly.
- B: **Correct**. Random walk: $F_t = F_{t-1} + \eta_t$. Forecast variance at horizon $h$: $\text{Var}(F_{T+h} - \hat{F}_{T+h}) = h \cdot \sigma_\eta^2$, which grows linearly. This indicates a unit root (non-stationarity).
- C: While possible, linear growth is specifically diagnostic of unit root, not general misspecification.
- D: Autocorrelated forecast errors would violate Kalman filter optimality but doesn't explain the linear growth pattern.

---

## Part D: Short Answer (Bonus: 10 points)

### Question 21 (5 points) - Advanced

Briefly explain why PCA on the $T \times T$ matrix $XX'$ versus the $N \times N$ matrix $X'X$ yields the same factors when $T = N$, but different computational costs when $T \neq N$.

**Your Answer:** _(Student provides written explanation)_

**Sample Answer:**

Both approaches extract the same factor space (span of top $r$ eigenvectors) because they correspond to the singular value decomposition (SVD) of $X$. Specifically:
- $X'X = V D^2 V'$ (eigendecomposition, $N \times N$)
- $XX' = U D^2 U'$ (eigendecomposition, $T \times T$)
- Related by: $F = UD$ and $\Lambda = VD$

When $T < N$ (typical in panels), computing $XX'$ is cheaper ($T^3$ vs $N^3$ operations). When $T > N$, $X'X$ is cheaper. The factor estimates $\hat{F}$ are identical up to normalization.

**Rubric:**
- Mentions SVD/dual representations (2 pts)
- Explains computational cost depends on $\min(T, N)$ (2 pts)
- Correctly identifies which is cheaper when (1 pt)

---

### Question 22 (5 points) - Core

When would you prefer maximum likelihood estimation over PCA for a factor model? List two specific scenarios.

**Your Answer:** _(Student provides written explanation)_

**Sample Answer:**

1. **Small sample size:** When $N$ and $T$ are both small (e.g., $N = 20$, $T = 50$), MLE can be more efficient because it properly accounts for finite-sample uncertainty. PCA asymptotics require large $N$ and $T$.

2. **Non-diagonal $\Sigma_e$:** If idiosyncratic errors are correlated (approximate factor model) and you want to model this structure explicitly, likelihood-based methods allow specifying and estimating $\Sigma_e$ with off-diagonal elements.

3. **Inference:** MLE provides standard errors and hypothesis tests via the information matrix. PCA requires bootstrap or other methods for standard errors.

(Any two valid scenarios)

**Rubric:**
- Each valid scenario with explanation (2.5 pts × 2)
- Partial credit for correct scenario without full explanation (1 pt)

---

## Answer Key Summary

| Question | Answer | Difficulty | Topic |
|----------|--------|------------|-------|
| 1 | B | Foundation | Stock-Watson method |
| 2 | B | Core | Computational efficiency |
| 3 | C | Core | PCA consistency |
| 4 | A | Core | PCA normalization |
| 5 | C | Advanced | Bai-Ng penalties |
| 6 | B | Core | Scree plot |
| 7 | C | Advanced | Multiple criteria |
| 8 | B | Core | Variance explained |
| 9 | A | Advanced | EM algorithm |
| 10 | B | Foundation | Standardization |
| 11 | C | Core | Variance calculation |
| 12 | B | Core | Loading structure |
| 13 | C | Advanced | VAR parameters |
| 14 | D | Advanced | Convergence rate |
| 15 | B | Core | VAR forecasting |
| 16 | B | Core | Missing data |
| 17 | D | Core | Out-of-sample factors |
| 18 | B | Advanced | Structural stability |
| 19 | A | Core | Factor importance |
| 20 | B | Advanced | Non-stationarity |
| 21 | - | Advanced | PCA computation |
| 22 | - | Core | ML vs PCA tradeoff |

**Scoring Distribution:**
- Part A (Conceptual): 40 points
- Part B (Mathematical): 30 points
- Part C (Practical): 30 points
- Part D (Bonus Short Answer): 10 points

**Difficulty Distribution:**
- Foundation: 8 points (2 questions)
- Core: 62 points (13 questions)
- Advanced: 40 points (7 questions)
