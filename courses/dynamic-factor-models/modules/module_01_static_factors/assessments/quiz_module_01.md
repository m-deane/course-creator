# Module 1 Quiz: Static Factor Models

**Time Estimate:** 45-60 minutes
**Total Points:** 100
**Passing Score:** 70%

## Instructions

This quiz assesses your understanding of static factor models, including model specification, identification problems, and the distinction between exact and approximate factor models. Answer all questions. You have 2 attempts for this quiz, and your highest score will be recorded.

---

## Part A: Conceptual Understanding (40 points)

### Question 1 (4 points) - Foundation

Consider a static factor model: $X_t = \Lambda F_t + e_t$ with $N = 50$ variables and $r = 3$ factors. How many observations ($T$) would be needed at minimum to estimate this model?

A) $T \geq 3$
B) $T \geq 50$
C) $T \geq 100$
D) The number of factors must be chosen first

**Correct Answer:** A

**Feedback:**
- A: Correct. Technically, you need $T \geq r$ for the sample covariance matrix to have rank $r$. However, in practice, you want $T$ much larger (ideally $T > 100$) for stable estimates.
- B: This confuses the number of variables with the required sample size. While having $T \approx N$ is desirable for PCA consistency, it's not a strict minimum.
- C: This is a practical guideline (rule of thumb: $T > 10r$ at minimum), but not the theoretical minimum.
- D: The number of factors is a model choice, but basic estimation is possible with very small $T$ (though highly unreliable).

---

### Question 2 (4 points) - Core

Which assumption distinguishes an **exact** factor model from an **approximate** factor model?

A) Factors must be orthogonal in exact models
B) Loadings must have constant magnitude across variables
C) Idiosyncratic errors are uncorrelated across variables ($\Sigma_e$ is diagonal)
D) Factors must be Gaussian distributed

**Correct Answer:** C

**Feedback:**
- A: Factor orthogonality is a normalization choice, not a defining feature of exact vs approximate models.
- B: Loading magnitudes are unrestricted in both exact and approximate models.
- C: **Correct**. Exact models require $\Sigma_e = \text{diag}(\psi_1^2, ..., \psi_N^2)$—all cross-variable correlations come through factors. Approximate models allow weak off-diagonal elements in $\Sigma_e$.
- D: Gaussianity is a distributional assumption often made for both exact and approximate models, but not the distinguishing feature.

---

### Question 3 (4 points) - Core

In a factor model, the **communality** $h_i^2$ for variable $i$ represents:

A) The correlation between variable $i$ and all other variables
B) The proportion of variance in variable $i$ explained by common factors
C) The loading of variable $i$ on the first factor
D) The total variance of variable $i$

**Correct Answer:** B

**Feedback:**
- A: This describes the sum of pairwise correlations, not communality.
- B: **Correct**. Communality $h_i^2 = \sum_{j=1}^r \lambda_{ij}^2$ represents the share of $\text{Var}(X_i)$ coming from factors, with $(1 - h_i^2)$ being the idiosyncratic share.
- C: This is just one component of the communality ($\lambda_{i1}^2$), not the full measure.
- D: This is the total variance, not the factor-explained portion.

---

### Question 4 (4 points) - Core

Why do factor models help with the "curse of dimensionality" in forecasting with many predictors?

A) They eliminate the need for time series data
B) They reduce $N$ variables to $r \ll N$ factors, drastically reducing parameters
C) They make all variables orthogonal, simplifying estimation
D) They automatically select the most important variables

**Correct Answer:** B

**Feedback:**
- A: Factor models still require time series observations; they don't eliminate the temporal dimension.
- B: **Correct**. A VAR with $N$ variables has $O(N^2)$ parameters per lag. By extracting $r$ factors, a factor VAR has only $O(r^2)$ parameters—a massive reduction when $r \ll N$.
- C: Factor models don't make observed variables orthogonal; they extract uncorrelated factors, but $X$ remains correlated.
- D: Factor models use all variables (weighted by loadings), not a subset. Variable selection is a different approach.

---

### Question 5 (4 points) - Advanced

In the Chamberlain-Rothschild framework for approximate factor models, a factor is **pervasive** if:

A) It affects all variables equally
B) It has the largest eigenvalue
C) The average squared loading remains bounded away from zero as $N \to \infty$
D) It is stationary over time

**Correct Answer:** C

**Feedback:**
- A: Pervasive doesn't mean equal loadings—some variables can load weakly or not at all. It means affecting a non-negligible *fraction* of variables.
- B: Ordering by eigenvalues is related but not the definition of pervasiveness.
- C: **Correct**. Formally, $\lim_{N\to\infty} \frac{1}{N}\sum_{i=1}^N \lambda_{ij}^2 = c_j > 0$. The factor's total contribution grows proportionally with $N$, not just affecting a fixed number of variables.
- D: Stationarity is a time-series property, unrelated to pervasiveness across variables.

---

### Question 6 (4 points) - Core

You estimate a factor model with $N = 100$ variables and extract $r = 5$ factors. The first 5 eigenvalues are: $[28.3, 12.7, 8.4, 5.1, 3.2]$ and the 6th is $1.8$. The total variance is 100 (standardized data). What proportion of variance do the 5 factors explain?

A) 28.3%
B) 57.7%
C) 75.4%
D) 100%

**Correct Answer:** B

**Feedback:**
- A: This is only the first factor's contribution.
- B: **Correct**. $(28.3 + 12.7 + 8.4 + 5.1 + 3.2) / 100 = 57.7 / 100 = 57.7\%$
- C: This would require eigenvalues summing to 75.4, which they don't.
- D: The factors don't explain all variance—$42.3\%$ remains idiosyncratic.

---

### Question 7 (4 points) - Advanced

Consider the implied covariance structure: $\Sigma_X = \Lambda \Lambda' + \Sigma_e$. If we have $N = 127$ variables and extract $r = 8$ factors with diagonal $\Sigma_e$, how many free parameters are in the covariance model?

A) 1,016 + 8 + 127 = 1,151
B) 1,016 + 36 + 127 = 1,179
C) $(127 \times 128)/2 = 8,128$
D) $127 \times 8 = 1,016$

**Correct Answer:** B

**Feedback:**
- A: Close, but undercounts factor covariance. If factors are correlated, need $r(r+1)/2 = 36$ parameters, not just $r = 8$.
- B: **Correct**. Loadings: $N \times r = 127 \times 8 = 1,016$. Factor covariance (symmetric): $r(r+1)/2 = 8 \times 9 / 2 = 36$. Idiosyncratic variances: $N = 127$. Total: $1,016 + 36 + 127 = 1,179$.
- C: This is the number of unique elements in $\Sigma_X$ (what we're trying to model), not the parameters.
- D: This counts only loadings, ignoring factor and error covariances.

---

### Question 8 (4 points) - Core

A researcher claims: "I found a 10-factor model explains 95% of variance in my 20-variable panel. This proves the factors capture all important information." What is the main concern with this claim?

A) 10 factors are always too many
B) With $r = 10$ and $N = 20$, the model is near-saturated and likely overfitting
C) 95% is too high—factors should explain less variance
D) Static models can never explain more than 80% of variance

**Correct Answer:** B

**Feedback:**
- A: Whether 10 factors are "too many" depends on context, not an absolute rule.
- B: **Correct**. With 10 factors for 20 variables, the model is very flexible ($(10 \times 11)/2 + 20 \times 10 + 20 = 275$ parameters vs. $(20 \times 21)/2 = 210$ covariance elements). High $R^2$ likely reflects overfitting, not genuine factor structure.
- C: High variance explanation can be legitimate with a true factor structure; the issue here is overparameterization.
- D: There's no such theoretical limit—exact factor models can explain 100% if $N = r$ and $\Sigma_e = 0$.

---

### Question 9 (4 points) - Advanced

You estimate factors for two subperiods (1980-2000 and 2000-2020) separately using PCA. The economic interpretation of "Factor 1" appears completely different across periods. Which is the most likely explanation?

A) The factor model is mis-specified
B) There was structural change in the economy
C) Sign and rotation indeterminacy across separate estimations
D) The number of factors changed between periods

**Correct Answer:** C

**Feedback:**
- A: While possible, this is not the most likely explanation for factors looking different across periods.
- B: Structural change could occur, but the question states the *interpretation* differs, not necessarily the data patterns.
- C: **Correct**. PCA factors are only identified up to rotation and sign flips. Estimating factors separately for two periods gives two arbitrary normalizations, making factors incomparable without explicit alignment (e.g., Procrustes rotation).
- D: If $r$ changed, you'd have different dimensionality, but the question implies same number of factors with different meanings.

---

### Question 10 (4 points) - Foundation

In the factor model $X_t = \Lambda F_t + e_t$, what does $\lambda_{i3} = 0.85$ mean?

A) Variable $i$ and Factor 3 have correlation 0.85
B) A one-unit increase in Factor 3 increases variable $i$ by 0.85 units (holding other factors fixed)
C) Variable $i$ explains 85% of Factor 3's variance
D) Factor 3 has 85% probability of affecting variable $i$

**Correct Answer:** B

**Feedback:**
- A: The correlation depends on factor and variable variances, not just the loading. With standardized $F$ and $X$, the correlation is $\lambda_{i3} / \sqrt{\text{Var}(X_i)}$.
- B: **Correct**. The loading $\lambda_{i3}$ is the marginal effect of Factor 3 on variable $i$, holding other factors constant.
- C: This reverses the direction—factors explain variable variance, not vice versa.
- D: Loadings are not probabilities; they're regression-like coefficients.

---

## Part B: Mathematical Application (30 points)

### Question 11 (6 points) - Core

Given a 2-factor model with loadings:
$$\Lambda = \begin{bmatrix} 0.8 & 0.1 \\ 0.6 & 0.7 \\ 0.3 & 0.9 \end{bmatrix}$$

and factor covariance $\Sigma_F = I_2$, what is the implied covariance between variables 1 and 2?

A) 0.55
B) 0.48
C) 0.90
D) Cannot determine without $\Sigma_e$

**Correct Answer:** A

**Feedback:**
- A: **Correct**. $\text{Cov}(X_1, X_2) = \lambda_1' \Sigma_F \lambda_2 = [0.8, 0.1] \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} [0.6, 0.7]' = 0.8 \times 0.6 + 0.1 \times 0.7 = 0.48 + 0.07 = 0.55$
- B: This computes only the first factor's contribution (0.8 × 0.6), missing the second factor.
- C: This incorrectly adds rather than multiplies corresponding loadings.
- D: For the factor-driven covariance, we don't need $\Sigma_e$ (which contributes zero for $i \neq j$ in exact models).

---

### Question 12 (6 points) - Core

You observe eigenvalues from PCA on $N = 50$ variables: $[15.2, 8.3, 6.1, 2.4, 2.1, 1.9, 1.8, ...]$ (remaining values < 1.5). Using the "elbow" criterion, how many factors would you select?

A) 1
B) 3
C) 5
D) 7

**Correct Answer:** B

**Feedback:**
- A: This ignores clearly important factors 2 and 3.
- B: **Correct**. The eigenvalues show a clear drop after the 3rd factor: [15.2, 8.3, 6.1] are substantially larger than [2.4, 2.1, ...]. The "elbow" appears at $r = 3$.
- C: Factors 4 and 5 (2.4, 2.1) are much smaller and close to remaining values—likely not pervasive.
- D: Including 7 factors would capture noise; eigenvalues flatten after position 3.

---

### Question 13 (6 points) - Advanced

In an approximate factor model with $N = 200$, the top eigenvalue is 45.3. As $N$ increases (holding true parameters fixed), this eigenvalue will:

A) Stay approximately constant
B) Grow proportionally with $N$ (roughly linear)
C) Decrease toward zero
D) Converge to the number of factors $r$

**Correct Answer:** B

**Feedback:**
- A: This would be true for idiosyncratic eigenvalues, not factor eigenvalues.
- B: **Correct**. For pervasive factors, eigenvalues scale as $O(N)$ because the factor contributes to a non-negligible fraction of all $N$ variables. Specifically, $\lambda_j \approx N \cdot c_j$ where $c_j$ is the average squared loading.
- C: Eigenvalues don't shrink; they grow with $N$ for factors.
- D: The number of factors is fixed; eigenvalue magnitude grows with $N$.

---

### Question 14 (6 points) - Advanced

Consider the factor model identification problem. If $X = \Lambda F + e$ and we apply rotation $H$, we get $X = \tilde{\Lambda}\tilde{F} + e$ where $\tilde{\Lambda} = \Lambda H$ and $\tilde{F} = H^{-1}F$. To uniquely identify factors using PCA normalization, how many independent constraints are needed for $r = 3$ factors?

A) 3 (one per factor)
B) 6 (orthogonality constraints)
C) 9 (full rotation matrix)
D) 12 (loadings and factors)

**Correct Answer:** B

**Feedback:**
- A: Just normalizing scale (one constraint per factor) leaves rotation freedom.
- B: **Correct**. PCA normalization imposes $F'F/T = I_r$ (orthonormal factors), which is $r(r+1)/2 = 3 \times 4 / 2 = 6$ constraints. This identifies factors up to sign flips.
- C: An invertible $r \times r$ rotation matrix has $r^2$ elements, but constraints don't need to equal degrees of freedom exactly.
- D: This overestimates; we need to restrict the $r \times r$ rotation matrix, not all loadings.

---

### Question 15 (6 points) - Core

For a single-factor model ($r = 1$), if all variables have loading $\lambda_i = 0.7$ (standardized) and idiosyncratic variance $\psi_i^2 = 0.51$, what is each variable's $R^2$ (proportion of variance from the factor)?

A) 0.49
B) 0.51
C) 0.70
D) 1.00

**Correct Answer:** A

**Feedback:**
- A: **Correct**. $R^2 = \frac{\lambda_i^2}{\lambda_i^2 + \psi_i^2} = \frac{0.7^2}{0.7^2 + 0.51} = \frac{0.49}{1.00} = 0.49$ or 49%.
- B: This is the idiosyncratic share (1 - $R^2$), not $R^2$.
- C: This confuses the loading magnitude with $R^2$; need to square: $0.7^2 = 0.49$.
- D: This would require perfect fit ($\psi_i^2 = 0$), which contradicts the problem.

---

## Part C: Practical Interpretation (30 points)

### Question 16 (6 points) - Core

A researcher extracts 3 factors from 50 macroeconomic indicators. Factor 1 loads heavily (>0.6) on: GDP, industrial production, employment, hours worked. Factor 2 loads heavily on: CPI, PPI, wages, import prices. Factor 3 loads heavily on: interest rates, credit spreads, stock returns. What would be appropriate labels?

A) Factor 1: Nominal, Factor 2: Real, Factor 3: Financial
B) Factor 1: Real Activity, Factor 2: Inflation, Factor 3: Financial Conditions
C) Factor 1: Supply, Factor 2: Demand, Factor 3: Monetary Policy
D) Cannot determine without time series plots

**Correct Answer:** B

**Feedback:**
- A: This reverses real and nominal factors.
- B: **Correct**. The loading patterns clearly indicate: Factor 1 captures real economic activity (GDP, production, employment), Factor 2 captures price/inflation dynamics, Factor 3 captures financial market conditions.
- C: Supply/demand interpretation would require structural identification (e.g., sign restrictions), which loading magnitudes alone don't provide.
- D: While time series plots help validate, the loading patterns already strongly suggest economic interpretation.

---

### Question 17 (6 points) - Advanced

You estimate a factor model on FRED-MD (N=127 monthly indicators) and extract $r = 5$ factors. Factors explain 65% of total variance. A colleague says: "35% unexplained means the model is missing important information." What is the best response?

A) Agree—you should extract more factors until 90%+ variance is explained
B) The 35% is mostly idiosyncratic noise specific to individual variables, not systematic information
C) This suggests the data violates the factor model assumptions
D) You need more time periods to reduce the unexplained variance

**Correct Answer:** B

**Feedback:**
- A: Extracting too many factors captures idiosyncratic noise rather than common factors. With $N = 127$, even 5 factors explaining 65% suggests strong common structure.
- B: **Correct**. The 35% unexplained variance consists of variable-specific measurement error, local shocks, and non-systematic variation—exactly what the idiosyncratic component $e_t$ is designed to capture. The factors already captured the "important" (pervasive) co-movement.
- C: 65% explained variance with 5 factors in a 127-variable panel is excellent and consistent with factor structure. Many macro panels show similar proportions.
- D: Sample size affects estimation precision, not the population variance decomposition.

---

### Question 18 (6 points) - Core

You apply PCA to a panel of stock returns (N=500 stocks). The first factor explains 18% of variance, and remaining factors each explain <5%. What does this suggest about the data structure?

A) The factor model is mis-specified
B) A single dominant factor (likely market factor) drives most co-movement
C) Stocks are approximately uncorrelated
D) You need to standardize the data first

**Correct Answer:** B

**Feedback:**
- A: Finding one dominant factor is perfectly consistent with factor model structure.
- B: **Correct**. This pattern is classic for stock returns: one strong factor (market/systematic risk) explains modest variance (15-25%), while remaining co-movement is weaker. This matches CAPM and APT predictions.
- C: If stocks were uncorrelated, the first factor would explain approximately $1/N \approx 0.2\%$, not 18%.
- D: Standardization affects factor *loadings* but not the relative importance pattern across factors.

---

### Question 19 (6 points) - Advanced

A factor model estimated on quarterly data (1960-2020, T=240) shows factor loadings on GDP growth: $\lambda_{\text{GDP}, 1} = 0.92$, $\lambda_{\text{GDP}, 2} = 0.05$, $\lambda_{\text{GDP}, 3} = -0.03$. In the next sample (2020-2023, T=12), the estimates are: $\lambda_{\text{GDP}, 1} = 0.88$, $\lambda_{\text{GDP}, 2} = 0.09$, $\lambda_{\text{GDP}, 3} = -0.07$. This suggests:

A) Structural break—GDP loading changed significantly
B) Normal sampling variability given smaller second sample
C) The factor model is unstable and should not be used
D) Sign normalization was inconsistent across samples

**Correct Answer:** B

**Feedback:**
- A: The changes are quite small (0.92 → 0.88) and well within sampling error given the much smaller second sample ($T = 12$).
- B: **Correct**. With only 12 observations, standard errors are large. The changes (±0.04 to ±0.06) are minor and likely just sampling variation, not structural change.
- C: Small changes across samples don't indicate instability; this is expected sampling variation.
- D: Sign normalization issues would flip signs entirely (e.g., 0.92 → -0.92), not cause small magnitude changes.

---

### Question 20 (6 points) - Core

You estimate a factor model and find that variable 23 (housing starts) has very low communality ($h_{23}^2 = 0.08$). What does this imply?

A) Housing starts should be removed from the dataset
B) Housing starts are driven primarily by idiosyncratic factors, not the common factors
C) The factor model is invalid for this dataset
D) Housing starts are more important than other variables

**Correct Answer:** B

**Feedback:**
- A: Low communality doesn't mean the variable should be removed—it still contributes information about what the factors *don't* explain.
- B: **Correct**. Communality $h_{23}^2 = 0.08$ means only 8% of housing starts variance comes from common factors, while 92% is idiosyncratic. Housing starts have unique dynamics not captured by macro factors.
- C: Some variables having low communality is normal and doesn't invalidate the model for other variables.
- D: Low communality means the variable is *less* connected to common factors, not more important.

---

## Answer Key Summary

| Question | Answer | Difficulty | Topic |
|----------|--------|------------|-------|
| 1 | A | Foundation | Model specification |
| 2 | C | Core | Exact vs approximate |
| 3 | B | Core | Communality |
| 4 | B | Core | Dimensionality reduction |
| 5 | C | Advanced | Pervasive factors |
| 6 | B | Core | Variance explained |
| 7 | B | Advanced | Parameter counting |
| 8 | B | Core | Overfitting |
| 9 | C | Advanced | Identification |
| 10 | B | Foundation | Loading interpretation |
| 11 | A | Core | Covariance calculation |
| 12 | B | Core | Scree plot |
| 13 | B | Advanced | Eigenvalue asymptotics |
| 14 | B | Advanced | Identification constraints |
| 15 | A | Core | R-squared calculation |
| 16 | B | Core | Economic interpretation |
| 17 | B | Advanced | Unexplained variance |
| 18 | B | Core | Single factor dominance |
| 19 | B | Advanced | Sampling variability |
| 20 | B | Core | Low communality |

**Scoring Distribution:**
- Part A (Conceptual): 40 points
- Part B (Mathematical): 30 points
- Part C (Practical): 30 points

**Difficulty Distribution:**
- Foundation: 8 points (2 questions)
- Core: 64 points (16 questions)
- Advanced: 28 points (7 questions)
