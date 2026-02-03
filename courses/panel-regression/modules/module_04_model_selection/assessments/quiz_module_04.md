# Quiz: Module 4 - Model Selection and Diagnostics

**Course:** Panel Data Econometrics
**Module:** 4 - Model Selection and Diagnostics
**Time Limit:** 30 minutes
**Total Points:** 100
**Passing Score:** 70%

## Instructions

This quiz assesses your understanding of the Hausman test, specification tests, and model selection strategies for panel data. You have 2 attempts. Show all calculations where requested.

---

## Part A: Hausman Test (40 points)

### Question 1: Hausman Test Purpose (8 points)

The Hausman test for panel data primarily tests:

**A)** Whether the panel is balanced or unbalanced

**B)** Whether entity-specific effects are statistically significant

**C)** Whether random effects assumptions hold (specifically $E[\mu_i | X_{it}] = 0$)

**D)** Whether time fixed effects should be included

---

### Question 2: Hausman Test Null Hypothesis (8 points)

The null hypothesis of the Hausman test is:

**A)** $H_0$: Fixed effects and random effects estimates are identical

**B)** $H_0$: Entity effects are uncorrelated with regressors (RE is consistent)

**C)** $H_0$: There are no entity-specific effects

**D)** $H_0$: The panel has no serial correlation

---

### Question 3: Hausman Test Statistic (8 points)

The Hausman test statistic is:

$$H = (\hat{\beta}_{FE} - \hat{\beta}_{RE})' [Var(\hat{\beta}_{FE}) - Var(\hat{\beta}_{RE})]^{-1} (\hat{\beta}_{FE} - \hat{\beta}_{RE})$$

Under the null hypothesis, this statistic follows:

**A)** $H \sim N(0, 1)$

**B)** $H \sim t_{NT-N-k}$

**C)** $H \sim \chi^2_k$ where $k$ is the number of time-varying regressors

**D)** $H \sim F_{k, NT-N-k}$

---

### Question 4: Hausman Test Interpretation (8 points)

You run a Hausman test and obtain $\chi^2(3) = 12.5$ with $p$-value = 0.006. What should you conclude?

**A)** Fail to reject $H_0$; use random effects

**B)** Reject $H_0$; evidence that RE assumptions are violated; use fixed effects

**C)** Reject $H_0$; use pooled OLS

**D)** The test is inconclusive; use both FE and RE

---

### Question 5: Hausman Test Limitations (8 points)

Which of the following is a known limitation of the Hausman test?

**A)** It can only be used with balanced panels

**B)** The test statistic can be negative in finite samples when using cluster-robust standard errors

**C)** It requires normality of errors

**D)** It cannot distinguish between FE and RE when $T > 10$

---

## Part B: Specification Tests (35 points)

### Question 6: Testing for Entity Fixed Effects (7 points)

To test whether entity fixed effects are needed (vs. pooled OLS), you use an F-test with:

$$F = \frac{(RSS_{pooled} - RSS_{FE})/(N-1)}{RSS_{FE}/(NT-N-k)}$$

What is the null hypothesis?

**A)** $H_0$: All entity effects $\alpha_i$ are equal (no entity effects needed)

**B)** $H_0$: All entity effects are correlated with regressors

**C)** $H_0$: Random effects should be used

**D)** $H_0$: The model is misspecified

---

### Question 7: Breusch-Pagan LM Test (7 points)

The Breusch-Pagan Lagrange Multiplier test for random effects tests:

**A)** $H_0: \sigma^2_\mu = 0$ (no random effects; use pooled OLS)

**B)** $H_0: \sigma^2_\epsilon = 0$ (no idiosyncratic variation)

**C)** $H_0:$ Random effects equals fixed effects

**D)** $H_0:$ No serial correlation

---

### Question 8: Model Selection Flowchart (7 points)

According to the standard decision flowchart, if:
1. The F-test rejects pooled OLS (entity effects are significant)
2. The Hausman test fails to reject (p-value = 0.40)

Which model should you use?

**A)** Pooled OLS

**B)** Fixed Effects

**C)** Random Effects

**D)** First-differences

---

### Question 9: Testing for Time Fixed Effects (7 points)

To test whether time fixed effects are needed in addition to entity fixed effects, you would:

**A)** Run a Hausman test comparing one-way and two-way FE

**B)** Run an F-test comparing RSS from one-way FE to two-way FE models

**C)** Check if $\sigma^2_\mu > 0$

**D)** Use the Breusch-Pagan test

---

### Question 10: Robust Hausman Test (7 points)

When using cluster-robust standard errors, the classical Hausman test may fail. An alternative is the **robust Hausman test** or:

**A)** Using bootstrap standard errors

**B)** Regression-based Hausman test (auxiliary regression approach)

**C)** Increasing the sample size

**D)** Using ML estimation instead of GLS

---

## Part C: Diagnostics and Practical Considerations (25 points)

### Question 11: Serial Correlation in Fixed Effects (5 points)

The Wooldridge test for serial correlation in fixed effects models tests:

**A)** $H_0$: No first-order serial correlation in the idiosyncratic errors

**B)** $H_0$: Entity effects are uncorrelated with regressors

**C)** $H_0$: Errors are homoskedastic

**D)** $H_0$: The model includes sufficient lags

---

### Question 12: Heteroskedasticity Testing (5 points)

A modified Wald test in fixed effects models can test for:

**A)** Groupwise heteroskedasticity (different $\sigma^2$ for each entity)

**B)** Time effects

**C)** Multicollinearity

**D)** Endogeneity

---

### Question 13: Model Selection Practical Advice (5 points)

In practice, many researchers default to fixed effects because:

**A)** FE is always more efficient than RE

**B)** FE doesn't require the potentially unrealistic assumption that entity effects are uncorrelated with regressors

**C)** FE can estimate time-invariant effects

**D)** FE is computationally faster

---

### Question 14: Residual Analysis (5 points)

After estimating a fixed effects model, you plot residuals over time and notice a clear upward trend. This suggests:

**A)** Omitted time trend or time effects

**B)** Entity effects are not properly controlled

**C)** Heteroskedasticity

**D)** The model is correctly specified

---

### Question 15: Cross-Sectional Dependence (5 points)

Cross-sectional dependence in panel data means:

**A)** Observations for different entities at the same time are correlated

**B)** Observations for the same entity over time are correlated

**C)** The panel is unbalanced

**D)** There is measurement error

---

## Calculation Problems (Show All Work)

### Question 16: F-Test for Entity Effects (15 points)

You estimate:
- **Pooled OLS:** $RSS_{pooled} = 5000$
- **Fixed Effects (Entity FE):** $RSS_{FE} = 3000$

With $N = 50$ entities, $T = 10$ periods, $k = 3$ regressors (including intercept).

**Part A (5 points):** Calculate the numerator degrees of freedom.

**Part B (5 points):** Calculate the denominator degrees of freedom.

**Part C (5 points):** Calculate the F-statistic and determine if you reject $H_0$ at the 5% level (critical value $\approx 1.4$).

---

### Question 17: Hausman Test Calculation (20 points)

You estimate the effect of training hours on productivity with:

**Fixed Effects:**
- $\hat{\beta}_{FE} = 2.5$
- $SE(\hat{\beta}_{FE}) = 0.5$

**Random Effects:**
- $\hat{\beta}_{RE} = 3.2$
- $SE(\hat{\beta}_{RE}) = 0.4$

For a single regressor, the Hausman statistic simplifies to:

$$H = \frac{(\hat{\beta}_{FE} - \hat{\beta}_{RE})^2}{Var(\hat{\beta}_{FE}) - Var(\hat{\beta}_{RE})}$$

**Part A (8 points):** Calculate the variance of each estimator and their difference.

**Part B (7 points):** Calculate the Hausman test statistic.

**Part C (5 points):** With critical value $\chi^2_1(0.05) = 3.84$, do you reject the null hypothesis? What is your conclusion about which model to use?

---

## Answer Key and Explanations

### Question 1: Answer C (8 points)

**Correct Answer:** C) Whether random effects assumptions hold (specifically $E[\mu_i | X_{it}] = 0$)

**Explanation:**

The Hausman test specifically examines whether the **random effects orthogonality assumption** is satisfied:

$$H_0: E[\mu_i | X_{it}] = 0 \iff Cov(\mu_i, X_{it}) = 0$$

**Logic:**
- If $H_0$ is true: Both FE and RE are consistent, but RE is more efficient
- If $H_0$ is false: FE is consistent, RE is biased
- Test compares FE and RE estimates to detect violation

**Why other answers are wrong:**
- A: Balance is not tested by Hausman
- B: That's the F-test for entity effects
- D: That's a separate test for time effects

**Key Concept:** Hausman test is fundamentally a specification test for the RE orthogonality assumption.

---

### Question 2: Answer B (8 points)

**Correct Answer:** B) $H_0$: Entity effects are uncorrelated with regressors (RE is consistent)

**Explanation:**

**Formal null hypothesis:**

$$H_0: E[\mu_i | X_{i1}, ..., X_{iT}] = 0$$

**Implications under $H_0$:**
- RE is consistent and efficient
- FE is consistent but inefficient
- Both estimators converge to the true $\beta$ (though FE is noisier)

**Alternative hypothesis:**

$$H_1: E[\mu_i | X_{it}] \neq 0 \text{ for some } t$$

Under $H_1$: RE is inconsistent, FE remains consistent.

**Why other answers are incorrect:**
- A: Estimates won't be exactly identical even under $H_0$ (sampling variation)
- C: That's tested by F-test or Breusch-Pagan
- D: That's tested by Wooldridge or Breusch-Godfrey tests

**Key Concept:** $H_0$ is that RE assumptions are valid (orthogonality).

---

### Question 3: Answer C (8 points)

**Correct Answer:** C) $H \sim \chi^2_k$ where $k$ is the number of time-varying regressors

**Explanation:**

Under $H_0$:

$$H = (\hat{\beta}_{FE} - \hat{\beta}_{RE})' [Var(\hat{\beta}_{FE}) - Var(\hat{\beta}_{RE})]^{-1} (\hat{\beta}_{FE} - \hat{\beta}_{RE}) \sim \chi^2_k$$

**Key points:**
- $k$ = number of time-varying coefficients being tested
- Time-invariant variables drop from FE, so only test time-varying ones
- This is a Wald test based on difference between two consistent estimators

**Degrees of freedom:** Equal to the number of restrictions being tested (number of coefficients that differ).

**Why other answers are wrong:**
- A: Not a standard normal (it's a quadratic form)
- B: Not a t-distribution (it's a chi-square)
- D: Not an F-distribution (would need to divide by k)

**Key Concept:** Hausman statistic follows chi-square distribution with df = # of coefficients tested.

---

### Question 4: Answer B (8 points)

**Correct Answer:** B) Reject $H_0$; evidence that RE assumptions are violated; use fixed effects

**Explanation:**

**Test result:**
- $\chi^2(3) = 12.5$
- $p$-value = 0.006 < 0.05
- **Reject $H_0$** at 5% level

**Interpretation:**
- Significant evidence that $E[\mu_i | X_{it}] \neq 0$
- RE estimates are likely biased and inconsistent
- FE and RE estimates differ significantly
- **Use Fixed Effects**

**Practical implication:**
- Entity-specific effects correlate with regressors
- Cannot trust RE estimates
- Sacrifice efficiency for consistency

**Why other answers are wrong:**
- A: We reject, not fail to reject
- C: Test is about FE vs RE, not pooled OLS
- D: Test gives clear guidance (use FE)

**Key Concept:** Rejecting Hausman test → use Fixed Effects.

---

### Question 5: Answer B (8 points)

**Correct Answer:** B) The test statistic can be negative in finite samples when using cluster-robust standard errors

**Explanation:**

**Known issue:** The covariance matrix $Var(\hat{\beta}_{FE}) - Var(\hat{\beta}_{RE})$ should be positive semi-definite in theory, but with robust covariance estimators:

$$\hat{Var}(\hat{\beta}_{FE}) - \hat{Var}(\hat{\beta}_{RE})$$

may not be positive definite in finite samples.

**Consequence:**
- Test statistic can be negative (impossible for a chi-square)
- Matrix inversion can fail

**Solutions:**
- Regression-based Hausman test
- Bootstrap approaches
- Use non-robust SEs for Hausman test only

**Why other answers are wrong:**
- A: Works with unbalanced panels
- C: Asymptotic test, doesn't require normality
- D: Can always distinguish if there are differences

**Key Concept:** Hausman test has finite-sample issues with robust SEs.

---

### Question 6: Answer A (7 points)

**Correct Answer:** A) $H_0$: All entity effects $\alpha_i$ are equal (no entity effects needed)

**Explanation:**

**F-test for entity fixed effects:**

$$H_0: \alpha_1 = \alpha_2 = ... = \alpha_N$$

If true, pooled OLS is appropriate (more efficient, doesn't lose N-1 degrees of freedom).

**Test statistic:**

$$F = \frac{(RSS_{pooled} - RSS_{FE})/(N-1)}{RSS_{FE}/(NT-N-k)} \sim F_{N-1, NT-N-k}$$

**Decision:**
- Reject $H_0$ → Entity effects differ significantly → Use FE or RE
- Fail to reject → Pooled OLS is appropriate

**Why other answers are wrong:**
- B: Correlation is tested by Hausman
- C: This test doesn't directly inform RE vs FE choice
- D: This is about entity effects specifically

**Key Concept:** F-test determines if entity effects are statistically significant.

---

### Question 7: Answer A (7 points)

**Correct Answer:** A) $H_0: \sigma^2_\mu = 0$ (no random effects; use pooled OLS)

**Explanation:**

**Breusch-Pagan LM test** for random effects:

$$H_0: \sigma^2_\mu = 0$$

**Interpretation:**
- If $\sigma^2_\mu = 0$: No entity-specific variance component
- All entity differences are captured by observables
- Pooled OLS is appropriate

**Test statistic:**

$$LM = \frac{NT}{2(T-1)} \left[\frac{\sum_i (\sum_t \hat{\epsilon}_{it})^2}{\sum_i \sum_t \hat{\epsilon}^2_{it}} - 1\right]^2 \sim \chi^2_1$$

**Decision:**
- Reject $H_0$ → Entity effects exist → Use RE or FE
- Fail to reject → Use pooled OLS

**Why other answers are wrong:**
- B: $\sigma^2_\epsilon = 0$ would mean no idiosyncratic variation (unrealistic)
- C: That's the Hausman test
- D: That's a different test (Wooldridge, Breusch-Godfrey)

**Key Concept:** Breusch-Pagan tests if random entity effects are present.

---

### Question 8: Answer C (7 points)

**Correct Answer:** C) Random Effects

**Explanation:**

**Decision logic:**

1. **F-test rejects pooled OLS:**
   - Entity effects are significant
   - Need to account for entity heterogeneity
   - Choose between FE and RE

2. **Hausman test fails to reject (p = 0.40):**
   - No evidence against $H_0: E[\mu_i | X_{it}] = 0$
   - RE assumptions appear valid
   - RE is consistent and more efficient than FE
   - **Use Random Effects**

**Why other answers are wrong:**
- A: F-test ruled out pooled OLS
- B: Would use FE only if Hausman rejects
- D: First-differences not indicated by these tests

**Key Concept:** When Hausman fails to reject, prefer RE for its efficiency.

---

### Question 9: Answer B (7 points)

**Correct Answer:** B) Run an F-test comparing RSS from one-way FE to two-way FE models

**Explanation:**

**Test for time fixed effects:**

$$H_0: \lambda_1 = \lambda_2 = ... = \lambda_T \text{ (no time effects needed)}$$

$$F = \frac{(RSS_{one-way} - RSS_{two-way})/(T-1)}{RSS_{two-way}/(NT-N-T+1-k)} \sim F_{T-1, NT-N-T+1-k}$$

**Decision:**
- Reject $H_0$ → Include time fixed effects
- Fail to reject → One-way (entity) FE sufficient

**Why other answers are wrong:**
- A: Hausman is for FE vs RE, not for time effects
- C: $\sigma^2_\mu$ is about entity effects, not time
- D: Breusch-Pagan is for random effects vs pooled

**Key Concept:** F-test determines if time fixed effects significantly improve model fit.

---

### Question 10: Answer B (7 points)

**Correct Answer:** B) Regression-based Hausman test (auxiliary regression approach)

**Explanation:**

**Problem:** Classical Hausman test can fail with cluster-robust SEs (non-positive definite variance matrix).

**Solution - Regression-based approach:**

1. Demean data to get within-transformed variables: $\tilde{X}_{it}$, $\tilde{y}_{it}$
2. Run auxiliary regression:
   $$y_{it} = X_{it}\beta_1 + \tilde{X}_{it}\beta_2 + error$$
3. Test $H_0: \beta_2 = 0$ using robust standard errors
4. If reject → FE and RE differ → use FE

**Advantages:**
- Works with robust/clustered SEs
- Easy to implement
- Same asymptotic properties as classical Hausman

**Why other answers are less preferred:**
- A: Bootstrap is computationally intensive
- C: Doesn't fix the covariance matrix issue
- D: Estimation method doesn't resolve the testing problem

**Key Concept:** Regression-based Hausman is robust to clustering.

---

### Question 11: Answer A (5 points)

**Correct Answer:** A) $H_0$: No first-order serial correlation in the idiosyncratic errors

**Explanation:**

**Wooldridge test** for serial correlation in FE models:

$$H_0: Corr(\epsilon_{it}, \epsilon_{i,t-1}) = 0$$

**Procedure:**
1. Estimate FE model, get residuals $\hat{\epsilon}_{it}$
2. Regress $\Delta \hat{\epsilon}_{it}$ on $\Delta \hat{\epsilon}_{i,t-1}$
3. Test if coefficient is significantly different from -0.5

**Importance:**
- Serial correlation affects standard errors
- Need clustered or robust SEs if present
- Common in panel data

**Why other answers are wrong:**
- B: That's the Hausman test
- C: That's heteroskedasticity tests
- D: Not about model specification

**Key Concept:** Wooldridge test detects AR(1) errors in FE models.

---

### Question 12: Answer A (5 points)

**Correct Answer:** A) Groupwise heteroskedasticity (different $\sigma^2$ for each entity)

**Explanation:**

**Modified Wald test** in FE:

$$H_0: \sigma^2_1 = \sigma^2_2 = ... = \sigma^2_N$$

Tests whether error variance differs across entities.

**Implication if rejected:**
- Heteroskedasticity across entities
- Standard errors are incorrect
- Use robust/clustered standard errors

**Common in practice:** Different entities (firms, countries) often have different error variances.

**Why other answers are wrong:**
- B: Time effects tested with F-test
- C: Multicollinearity tested with VIF
- D: Endogeneity requires IV or Hausman-type tests

**Key Concept:** Modified Wald detects entity-specific heteroskedasticity.

---

### Question 13: Answer B (5 points)

**Correct Answer:** B) FE doesn't require the potentially unrealistic assumption that entity effects are uncorrelated with regressors

**Explanation:**

**Practical reality:**
- In observational data, $Cov(\mu_i, X_{it}) \neq 0$ is very common
- Unobserved characteristics usually correlate with treatments
- RE assumption is often implausible

**Conservative approach:**
- Use FE to be "safe"
- Tolerate efficiency loss to avoid bias
- "When in doubt, use FE"

**When RE is preferred:**
- Randomized experiments
- Need to estimate time-invariant effects
- Hausman strongly supports RE

**Why other answers are wrong:**
- A: FALSE - FE is less efficient when RE assumption holds
- C: FALSE - FE cannot estimate time-invariant effects
- D: Computational cost is negligible

**Key Concept:** FE is safer default because it doesn't require orthogonality assumption.

---

### Question 14: Answer A (5 points)

**Correct Answer:** A) Omitted time trend or time effects

**Explanation:**

**Upward trend in residuals** suggests:
- Systematic time pattern not captured by model
- Could be:
  - Linear or nonlinear time trend
  - Time-specific effects (business cycle, policy changes)
  - Aggregate technological progress

**Solutions:**
- Include time trend: $t$ or $t^2$
- Add time fixed effects: $\lambda_t$
- Two-way fixed effects model

**Why other answers are wrong:**
- B: Entity effects would show entity patterns, not time trends
- C: Heteroskedasticity shows in variance, not mean trend
- D: Clear trend indicates misspecification

**Key Concept:** Residual patterns reveal omitted variables or effects.

---

### Question 15: Answer A (5 points)

**Correct Answer:** A) Observations for different entities at the same time are correlated

**Explanation:**

**Cross-sectional dependence:**

$$Cov(\epsilon_{it}, \epsilon_{jt}) \neq 0 \text{ for } i \neq j \text{ (same time } t \text{)}$$

**Sources:**
- Common shocks (macroeconomic, weather)
- Spatial correlation (neighboring regions)
- Network effects
- Common unobserved factors

**Implications:**
- Standard errors are incorrect
- Need Driscoll-Kraay or other robust SEs

**Why other answers describe different concepts:**
- B: That's serial correlation (within entity over time)
- C: That's about missingness pattern
- D: That's measurement error

**Key Concept:** Cross-sectional dependence is correlation across entities at same time.

---

### Question 16: Solution (15 points)

**Part A: Numerator DF (5 points)**

The F-test for entity effects compares:
- Unrestricted model: FE with $N$ entity dummies
- Restricted model: Pooled OLS (all $\alpha_i$ equal)

**Number of restrictions:** $N - 1$ (number of entity dummies after dropping one baseline)

**Numerator degrees of freedom:** $N - 1 = 50 - 1 = 49$

**Scoring:**
- 3 points for correct answer (49)
- 2 points for correct reasoning

---

**Part B: Denominator DF (5 points)**

**Fixed effects model degrees of freedom:**

$$df_{FE} = NT - N - k$$

Where:
- $NT = 50 \times 10 = 500$ total observations
- $N = 50$ entity effects
- $k = 3$ regressors

$$df_{FE} = 500 - 50 - 3 = 447$$

**Denominator degrees of freedom:** 447

**Scoring:**
- 3 points for correct calculation
- 2 points for showing formula

---

**Part C: F-Statistic Calculation (5 points)**

$$F = \frac{(RSS_{pooled} - RSS_{FE})/(N-1)}{RSS_{FE}/(NT-N-k)}$$

**Numerator:**
$$\frac{5000 - 3000}{49} = \frac{2000}{49} = 40.82$$

**Denominator:**
$$\frac{3000}{447} = 6.71$$

**F-statistic:**
$$F = \frac{40.82}{6.71} = 6.08$$

**Decision:**
- Critical value: $F_{49, 447}(0.05) \approx 1.4$
- Calculated: $F = 6.08 > 1.4$
- **Reject $H_0$**

**Conclusion:** Entity fixed effects are statistically significant. Pooled OLS is inappropriate; use FE or RE.

**Scoring:**
- 3 points for correct F-statistic
- 2 points for correct decision and interpretation

**Key Concept:** Large F-statistic indicates entity effects are important.

---

### Question 17: Solution (20 points)

**Part A: Variance Calculation (8 points)**

**Fixed Effects variance:**
$$Var(\hat{\beta}_{FE}) = [SE(\hat{\beta}_{FE})]^2 = (0.5)^2 = 0.25$$

**Random Effects variance:**
$$Var(\hat{\beta}_{RE}) = [SE(\hat{\beta}_{RE})]^2 = (0.4)^2 = 0.16$$

**Variance difference:**
$$Var(\hat{\beta}_{FE}) - Var(\hat{\beta}_{RE}) = 0.25 - 0.16 = 0.09$$

**Note:** This must be positive (FE always has larger variance when RE assumption holds). ✓

**Scoring:**
- 3 points for FE variance
- 3 points for RE variance
- 2 points for difference

---

**Part B: Hausman Statistic (7 points)**

**Point estimate difference:**
$$\hat{\beta}_{FE} - \hat{\beta}_{RE} = 2.5 - 3.2 = -0.7$$

**Hausman statistic:**
$$H = \frac{(\hat{\beta}_{FE} - \hat{\beta}_{RE})^2}{Var(\hat{\beta}_{FE}) - Var(\hat{\beta}_{RE})} = \frac{(-0.7)^2}{0.09} = \frac{0.49}{0.09} = 5.44$$

**Scoring:**
- 4 points for correct calculation
- 3 points for showing work

---

**Part C: Hypothesis Test and Conclusion (5 points)**

**Test:**
- $H \sim \chi^2_1$ under $H_0$
- Calculated: $H = 5.44$
- Critical value: $\chi^2_1(0.05) = 3.84$
- $5.44 > 3.84$ → **Reject $H_0$**

**Conclusion:**
- Significant evidence that RE assumptions are violated
- FE and RE estimates differ significantly (2.5 vs 3.2)
- The correlation between entity effects and training hours is non-zero
- **Use Fixed Effects model**

**Interpretation of difference:**
- RE estimate (3.2) is higher than FE (2.5)
- Suggests positive correlation: $Cov(\mu_i, training_{it}) > 0$
- Firms with higher unobserved productivity provide more training
- RE is biased upward

**Scoring:**
- 2 points for correct decision (reject)
- 3 points for interpretation and model choice

**Key Concept:** Significant Hausman test → reject RE, use FE.

---

## Scoring Rubric

| Points | Grade | Interpretation |
|--------|-------|----------------|
| 90-100 | A     | Excellent mastery of model selection |
| 80-89  | B     | Good understanding |
| 70-79  | C     | Satisfactory |
| 60-69  | D     | Needs review |
| 0-59   | F     | Significant gaps |

## Learning Objectives Assessed

- **LO1:** Understand and apply Hausman test (Q1-Q5, Q17)
- **LO2:** Conduct specification tests (Q6-Q10, Q16)
- **LO3:** Diagnose model adequacy (Q11-Q15)
- **LO4:** Make informed model selection decisions (Q8, Q13, Q17)

## Study Resources

If you scored below 70%, review:
- Module 4 Guide: Hausman Test
- Module 4 Guide: Specification Tests
- Module 4 Notebook: Model Selection

**Time to Review:** 3-4 hours

## Common Mistakes

1. **Confusing Hausman null** (it's about RE orthogonality, not equality of estimates)
2. **Wrong degrees of freedom** in F-tests (N-1 for entity effects, T-1 for time)
3. **Misinterpreting test results** (reject Hausman → use FE, not RE)
4. **Ignoring practical considerations** (theoretical tests vs. realistic assumptions)
5. **Breusch-Pagan vs Hausman** confusion (BP is RE vs pooled; Hausman is RE vs FE)
