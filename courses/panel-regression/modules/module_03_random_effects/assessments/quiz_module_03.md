# Quiz: Module 3 - Random Effects Models

**Course:** Panel Data Econometrics
**Module:** 3 - Random Effects Models
**Time Limit:** 30 minutes
**Total Points:** 100
**Passing Score:** 70%

## Instructions

This quiz assesses your understanding of random effects models, GLS estimation, and the trade-offs between FE and RE. You have 2 attempts. Show all work for calculation questions.

---

## Part A: Random Effects Fundamentals (35 points)

### Question 1: Random Effects Model Specification (7 points)

The random effects model is:

$$y_{it} = \alpha + X_{it}\beta + \mu_i + \epsilon_{it}$$

Which statement about $\mu_i$ is **correct**?

**A)** $\mu_i$ is a fixed parameter to be estimated for each entity

**B)** $\mu_i$ is treated as a random variable drawn from a distribution with $E[\mu_i] = 0$

**C)** $\mu_i$ can be correlated with $X_{it}$ without biasing $\hat{\beta}$

**D)** $\mu_i$ represents time-varying entity-specific shocks

---

### Question 2: RE Key Assumption (7 points)

The **critical assumption** that distinguishes random effects from fixed effects is:

**A)** $Var(\mu_i) = \sigma^2_\mu$ (constant variance)

**B)** $E[\mu_i | X_{i1}, ..., X_{iT}] = 0$ (random effects uncorrelated with regressors)

**C)** $\mu_i \sim N(0, \sigma^2_\mu)$ (normal distribution)

**D)** The panel must be balanced

---

### Question 3: Composite Error Structure (7 points)

In the RE model, the composite error is $v_{it} = \mu_i + \epsilon_{it}$.

What is the correlation between errors for the same entity at different time periods?

**A)** $Corr(v_{it}, v_{is}) = 0$ for all $t \neq s$

**B)** $Corr(v_{it}, v_{is}) = \frac{\sigma^2_\mu}{\sigma^2_\mu + \sigma^2_\epsilon}$ for $t \neq s$

**C)** $Corr(v_{it}, v_{is}) = \frac{\sigma^2_\epsilon}{\sigma^2_\mu + \sigma^2_\epsilon}$ for $t \neq s$

**D)** $Corr(v_{it}, v_{is}) = 1$ for $t \neq s$

---

### Question 4: When to Use Random Effects (7 points)

Random effects is most appropriate when:

**A)** The entities are randomly sampled from a large population and you want to generalize

**B)** You suspect entity-specific effects correlate with regressors

**C)** You need to estimate effects of time-invariant variables

**D)** Both A and C

---

### Question 5: RE Efficiency Advantage (7 points)

When the RE assumptions hold ($E[\mu_i | X_{it}] = 0$), RE is more efficient than FE because:

**A)** RE uses both between and within variation, while FE uses only within

**B)** RE has fewer parameters to estimate

**C)** RE automatically corrects for serial correlation

**D)** RE works better with unbalanced panels

---

## Part B: GLS Estimation (35 points)

### Question 6: GLS Transformation Parameter (7 points)

The GLS transformation uses parameter:

$$\theta = 1 - \sqrt{\frac{\sigma^2_\epsilon}{\sigma^2_\epsilon + T\sigma^2_\mu}}$$

What happens to $\theta$ as $T \to \infty$ (large time dimension)?

**A)** $\theta \to 0$ (RE approaches pooled OLS)

**B)** $\theta \to 1$ (RE approaches fixed effects)

**C)** $\theta \to 0.5$

**D)** $\theta$ is undefined for large $T$

---

### Question 7: GLS Transformation Mechanics (7 points)

The RE/GLS estimator transforms data as:

$$y_{it} - \theta\bar{y}_i = \alpha(1-\theta) + (X_{it} - \theta\bar{X}_i)\beta + error$$

If $\theta = 0$, what does this become?

**A)** Fixed effects (within) estimator

**B)** Pooled OLS

**C)** Between estimator

**D)** First-differences estimator

---

### Question 8: GLS Transformation Extremes (7 points)

What happens when $\sigma^2_\mu = 0$ (no entity-specific effects)?

**A)** $\theta = 0$ and RE = Pooled OLS

**B)** $\theta = 1$ and RE = Fixed Effects

**C)** GLS cannot be estimated

**D)** RE and FE give identical estimates

---

### Question 9: Variance Component Estimation (7 points)

The variance components $\sigma^2_\mu$ and $\sigma^2_\epsilon$ are typically estimated using:

**A)** Maximum likelihood estimation

**B)** Residuals from between and within regressions

**C)** ANOVA-type decomposition of variances

**D)** All of the above are valid approaches

---

### Question 10: Intraclass Correlation (7 points)

The intraclass correlation coefficient:

$$\rho = \frac{\sigma^2_\mu}{\sigma^2_\mu + \sigma^2_\epsilon}$$

measures what proportion?

**A)** Variation explained by the model

**B)** Variation in errors due to entity-specific effects

**C)** Correlation between $X$ and $\mu_i$

**D)** Proportion of time variation in the data

---

## Part C: RE vs FE Trade-offs (30 points)

### Question 11: Time-Invariant Variables (6 points)

Which of the following can be estimated with random effects but NOT with fixed effects?

**A)** The effect of education (completed before panel) on wages

**B)** The effect of experience (varies over time) on wages

**C)** The effect of union membership (varies over time) on wages

**D)** The effect of GDP growth (varies over time) on firm investment

---

### Question 12: Consistency Comparison (6 points)

Under the assumption that $E[\mu_i | X_{it}] = 0$:

**A)** FE is consistent but RE is not

**B)** RE is consistent but FE is not

**C)** Both FE and RE are consistent, but RE is more efficient

**D)** Both are consistent and equally efficient

---

### Question 13: Bias When Assumption Violated (6 points)

If $Cov(\mu_i, X_{it}) \neq 0$ (RE assumption violated):

**A)** Both FE and RE are biased

**B)** FE remains consistent, but RE is biased and inconsistent

**C)** RE remains consistent, but FE is biased

**D)** Neither estimator is affected

---

### Question 14: Generalizability (6 points)

A researcher studies the effect of class size using 50 randomly selected schools from a state with 500 schools. Which statement is correct?

**A)** FE estimates apply only to the 50 schools in the sample

**B)** RE estimates can be generalized to the population of 500 schools (if assumptions hold)

**C)** Both FE and RE generalize equally

**D)** Neither can be generalized beyond the sample

---

### Question 15: Practical Model Choice (6 points)

In practice, researchers often prefer FE over RE because:

**A)** FE is always more efficient

**B)** The RE assumption $E[\mu_i | X_{it}] = 0$ is often implausible in observational data

**C)** FE can estimate time-invariant effects

**D)** FE requires fewer computational resources

---

## Calculation Problems (Show All Work)

### Question 16: Variance Components and Theta (15 points)

Consider a random effects model with the following variance estimates:

- $\hat{\sigma}^2_\epsilon = 16$ (idiosyncratic error variance)
- $\hat{\sigma}^2_\mu = 9$ (entity-specific variance)
- $T = 5$ (time periods)

**Part A (5 points):** Calculate the intraclass correlation $\rho$.

**Part B (5 points):** Calculate the transformation parameter $\theta$.

**Part C (5 points):** Interpret what your $\theta$ value means for how RE weighs within vs between variation.

---

### Question 17: GLS Transformation (20 points)

Consider the following panel data:

| Firm | Year | Sales ($Y_{it}$) | Advertising ($X_{it}$) |
|------|------|-----------------|----------------------|
| A    | 1    | 100             | 10                   |
| A    | 2    | 120             | 12                   |
| A    | 3    | 140             | 14                   |
| B    | 1    | 150             | 20                   |
| B    | 2    | 180             | 24                   |
| B    | 3    | 210             | 28                   |

Assume $\theta = 0.6$ has been computed from variance components.

**Part A (5 points):** Calculate the entity means $\bar{Y}_A$, $\bar{Y}_B$, $\bar{X}_A$, and $\bar{X}_B$.

**Part B (10 points):** Calculate the GLS-transformed variables:

$$Y_{it}^{GLS} = Y_{it} - \theta\bar{Y}_i$$
$$X_{it}^{GLS} = X_{it} - \theta\bar{X}_i$$

for all six observations.

**Part C (5 points):** Compare your GLS-transformed values to what you would get with within transformation ($\theta = 1$). Which transformation removes more of the entity-specific mean?

---

## Answer Key and Explanations

### Question 1: Answer B (7 points)

**Correct Answer:** B) $\mu_i$ is treated as a random variable drawn from a distribution with $E[\mu_i] = 0$

**Explanation:**

In random effects:
- $\mu_i$ is a **random draw** from a population distribution
- Typically assume $\mu_i \sim N(0, \sigma^2_\mu)$ or just $E[\mu_i] = 0$ and $Var(\mu_i) = \sigma^2_\mu$
- Entities are viewed as random samples from a larger population
- We estimate the variance $\sigma^2_\mu$, not individual $\mu_i$ values

**Contrast with FE:** FE treats $\alpha_i$ as fixed parameters to be estimated.

**Why other answers are wrong:**
- A: That's the fixed effects approach
- C: FALSE - correlation with $X_{it}$ violates RE assumption and causes bias
- D: $\mu_i$ is time-invariant (constant for entity $i$ across all $t$)

**Key Concept:** RE treats entity effects as random draws, not fixed parameters.

---

### Question 2: Answer B (7 points)

**Correct Answer:** B) $E[\mu_i | X_{i1}, ..., X_{iT}] = 0$ (random effects uncorrelated with regressors)

**Explanation:**

This is the **identifying assumption** for RE:

$$E[\mu_i | X_i] = 0 \iff Cov(\mu_i, X_{it}) = 0 \text{ for all } t$$

**Implications:**
- If violated, RE is biased and inconsistent
- FE does not require this assumption
- This is testable with the Hausman test (Module 4)

**Why other answers are less critical:**
- A: Homogeneity of variance is an efficiency concern, not consistency
- C: Normality is not required for consistency (needed for ML inference)
- D: RE works with unbalanced panels

**Key Concept:** RE assumes entity effects are uncorrelated with all regressors.

---

### Question 3: Answer B (7 points)

**Correct Answer:** B) $Corr(v_{it}, v_{is}) = \frac{\sigma^2_\mu}{\sigma^2_\mu + \sigma^2_\epsilon}$ for $t \neq s$

**Derivation:**

For the same entity at different times:

$$Cov(v_{it}, v_{is}) = Cov(\mu_i + \epsilon_{it}, \mu_i + \epsilon_{is})$$

Since $\mu_i$ is common to both and $\epsilon$ are independent:

$$Cov(v_{it}, v_{is}) = Var(\mu_i) = \sigma^2_\mu$$

The variance of the composite error:

$$Var(v_{it}) = \sigma^2_\mu + \sigma^2_\epsilon$$

Therefore:

$$Corr(v_{it}, v_{is}) = \frac{Cov(v_{it}, v_{is})}{\sqrt{Var(v_{it})Var(v_{is})}} = \frac{\sigma^2_\mu}{\sigma^2_\mu + \sigma^2_\epsilon}$$

**Interpretation:** This is the intraclass correlation $\rho$, measuring how correlated observations are within the same entity.

**Key Concept:** RE induces positive serial correlation in composite errors through $\mu_i$.

---

### Question 4: Answer D (7 points)

**Correct Answer:** D) Both A and C

**Explanation:**

Random effects is appropriate when:

1. **Random sampling (A):** Entities randomly sampled from large population, allowing generalization
   - Example: 100 schools from 1,000 in a district

2. **Time-invariant variables of interest (C):** Need to estimate effects of variables that don't change over time
   - Example: Effect of gender, race, or location on wages

**Additional consideration:** RE assumes $E[\mu_i | X_{it}] = 0$, which must be plausible.

**Why B is wrong:**
- If entity effects correlate with regressors, use FE not RE

**Key Concept:** Use RE when interested in time-invariant effects and have random sample from population.

---

### Question 5: Answer A (7 points)

**Correct Answer:** A) RE uses both between and within variation, while FE uses only within

**Explanation:**

**Efficiency gain** comes from using more variation:

- **FE:** Uses only within variation (deviations from entity means)
  - Discards all between-entity variation

- **RE/GLS:** Uses weighted combination of between and within variation
  - Optimal weighting based on variance components

When RE assumption holds:

$$Var(\hat{\beta}_{RE}) < Var(\hat{\beta}_{FE})$$

**Trade-off:** RE is more efficient but requires stronger assumption.

**Why other answers are incomplete:**
- B: Fewer parameters is related, but not the fundamental reason
- C: Both can account for serial correlation with robust SEs
- D: Both work with unbalanced panels

**Key Concept:** RE gains efficiency by exploiting both variation sources.

---

### Question 6: Answer B (7 points)

**Correct Answer:** B) $\theta \to 1$ (RE approaches fixed effects)

**Explanation:**

As $T \to \infty$:

$$\theta = 1 - \sqrt{\frac{\sigma^2_\epsilon}{\sigma^2_\epsilon + T\sigma^2_\mu}} = 1 - \sqrt{\frac{1}{1 + T\frac{\sigma^2_\mu}{\sigma^2_\epsilon}}}$$

As $T \to \infty$, the denominator $\to \infty$, so:

$$\sqrt{\frac{1}{1 + T\frac{\sigma^2_\mu}{\sigma^2_\epsilon}}} \to 0$$

Therefore: $\theta \to 1 - 0 = 1$

**Implication:** With large $T$, entity means $\bar{y}_i$ are very precise estimates, so we should subtract them out (like FE).

**Key Concept:** As $T$ grows, RE converges to FE.

---

### Question 7: Answer B (7 points)

**Correct Answer:** B) Pooled OLS

**Explanation:**

When $\theta = 0$:

$$y_{it} - 0 \cdot \bar{y}_i = \alpha(1-0) + (X_{it} - 0 \cdot \bar{X}_i)\beta + error$$

$$y_{it} = \alpha + X_{it}\beta + error$$

This is **pooled OLS** - no transformation applied.

**Interpretation:** $\theta = 0$ means no entity-specific variance ($\sigma^2_\mu = 0$), so no need to account for entity effects.

**Why other answers are wrong:**
- A: FE/within uses $\theta = 1$
- C: Between uses only entity means, not original data
- D: First-differences uses $\Delta y_{it}$

**Key Concept:** $\theta$ interpolates between pooled OLS ($\theta=0$) and FE ($\theta=1$).

---

### Question 8: Answer A (7 points)

**Correct Answer:** A) $\theta = 0$ and RE = Pooled OLS

**Explanation:**

When $\sigma^2_\mu = 0$ (no entity heterogeneity):

$$\theta = 1 - \sqrt{\frac{\sigma^2_\epsilon}{\sigma^2_\epsilon + T \cdot 0}} = 1 - \sqrt{1} = 1 - 1 = 0$$

With $\theta = 0$:
- No transformation needed
- RE estimator = Pooled OLS
- Makes sense: if no entity effects, pooled OLS is appropriate

**Why other answers are wrong:**
- B: $\theta = 1$ occurs when $\sigma^2_\epsilon = 0$ or $T \to \infty$
- C: GLS is well-defined, just reduces to OLS
- D: RE and FE differ when $\sigma^2_\mu = 0$ (FE would still remove entity means)

**Key Concept:** RE adapts to the data structure via variance components.

---

### Question 9: Answer D (7 points)

**Correct Answer:** D) All of the above are valid approaches

**Explanation:**

Multiple methods exist for estimating variance components:

1. **Maximum Likelihood (ML/REML):**
   - Assumes normality
   - Jointly estimates $\beta$, $\sigma^2_\mu$, $\sigma^2_\epsilon$

2. **Method of Moments from Between/Within:**
   - $\hat{\sigma}^2_\epsilon$ from within (FE) residuals
   - $\hat{\sigma}^2_\mu$ from between residuals minus $\sigma^2_\epsilon/T$

3. **ANOVA-type Methods:**
   - Decompose total variance into components
   - Swamy-Arora, Wallace-Hussain estimators

**Common choice:** Between-within approach is computationally simple and doesn't require normality.

**Key Concept:** Variance components can be estimated multiple ways.

---

### Question 10: Answer B (7 points)

**Correct Answer:** B) Variation in errors due to entity-specific effects

**Explanation:**

The intraclass correlation:

$$\rho = \frac{\sigma^2_\mu}{\sigma^2_\mu + \sigma^2_\epsilon} = \frac{\text{Entity variance}}{\text{Total error variance}}$$

**Interpretation:**
- $\rho$ = proportion of error variance due to persistent entity effects
- $\rho \approx 0$: Little entity heterogeneity (pooled OLS appropriate)
- $\rho \approx 1$: Most variation is between entities (strong entity effects)

**Also equals:** $Corr(v_{it}, v_{is})$ for same entity, different times.

**Why other answers are wrong:**
- A: That's $R^2$
- C: That's about assumption violation (should be zero)
- D: This is about error components, not time variation in data

**Key Concept:** $\rho$ measures importance of entity-specific effects in total error.

---

### Question 11: Answer A (6 points)

**Correct Answer:** A) The effect of education (completed before panel) on wages

**Explanation:**

**Time-invariant variables:**
- Education (if completed before panel)
- Gender, race, ethnicity
- Birth location
- Industry (if firms don't switch)

**FE limitation:** Within transformation eliminates time-invariant variables
$$\tilde{education}_i = education_i - education_i = 0$$

**RE advantage:** Can estimate effects of time-invariant variables because it uses between variation.

**Why other answers are wrong:**
- B, C, D: All are time-varying and can be estimated with FE

**Key Concept:** Use RE when time-invariant variables are of substantive interest.

---

### Question 12: Answer C (6 points)

**Correct Answer:** C) Both FE and RE are consistent, but RE is more efficient

**Explanation:**

When $E[\mu_i | X_{it}] = 0$:

**Consistency:**
- FE: Consistent (doesn't require this assumption)
- RE: Consistent (assumption is satisfied)

**Efficiency:**
- RE: Uses both between and within → more efficient
- FE: Uses only within → less efficient

$$Var(\hat{\beta}_{RE}) < Var(\hat{\beta}_{FE})$$ under RE assumption

**Trade-off:** RE gains efficiency at cost of stronger assumption.

**Key Concept:** When RE assumption holds, RE dominates FE in efficiency.

---

### Question 13: Answer B (6 points)

**Correct Answer:** B) FE remains consistent, but RE is biased and inconsistent

**Explanation:**

If $Cov(\mu_i, X_{it}) \neq 0$:

**Fixed Effects:**
- Still consistent because it eliminates $\mu_i$ via demeaning
- No assumption about $Cov(\mu_i, X_{it})$ required
- Robust to this violation

**Random Effects:**
- Biased and inconsistent
- The bias: $plim(\hat{\beta}_{RE}) = \beta + bias$
- Bias depends on $Cov(\mu_i, X_{it})$ and within/between variance

**Implication:** This is why Hausman test is important (Module 4).

**Key Concept:** FE is robust to correlation between entity effects and regressors; RE is not.

---

### Question 14: Answer B (6 points)

**Correct Answer:** B) RE estimates can be generalized to the population of 500 schools (if assumptions hold)

**Explanation:**

**Random Effects:** Treats sampled entities as random draws from population
- 50 schools represent the population of 500
- Estimates apply to population if:
  - Random sampling
  - RE assumptions hold

**Fixed Effects:** Estimates apply only to sampled entities
- 50 schools are the population of interest
- No claims about other 450 schools
- "Conditional on the sampled entities"

**Practical note:** This theoretical difference often matters less than assumption validity.

**Key Concept:** RE framework enables population inference; FE is sample-specific.

---

### Question 15: Answer B (6 points)

**Correct Answer:** B) The RE assumption $E[\mu_i | X_{it}] = 0$ is often implausible in observational data

**Explanation:**

**Practical reality:**
- Unobserved entity characteristics usually correlate with regressors
- Example: Unobserved ability affects both education and wages
- Example: Unobserved firm quality affects both investment and profitability

**Researcher preference:**
- FE is "safer" - doesn't require orthogonality assumption
- RE efficiency gain not worth risking bias
- "If in doubt, use FE"

**When RE is used:**
- Randomized experiments (by design, $E[\mu_i | X_{it}] = 0$)
- Need to estimate time-invariant effects
- Hausman test supports RE

**Why other answers are wrong:**
- A: FE is less efficient when RE assumption holds
- C: FALSE - FE cannot estimate time-invariant effects
- D: Computational cost is negligible with modern software

**Key Concept:** RE requires strong assumption often violated in observational data.

---

### Question 16: Solution (15 points)

**Part A: Intraclass Correlation (5 points)**

$$\rho = \frac{\sigma^2_\mu}{\sigma^2_\mu + \sigma^2_\epsilon} = \frac{9}{9 + 16} = \frac{9}{25} = 0.36$$

**Interpretation:** 36% of total error variance is due to persistent entity-specific effects. This indicates moderate entity heterogeneity.

**Scoring:**
- 3 points for correct calculation
- 2 points for interpretation

---

**Part B: Transformation Parameter (5 points)**

$$\theta = 1 - \sqrt{\frac{\sigma^2_\epsilon}{\sigma^2_\epsilon + T\sigma^2_\mu}}$$

$$\theta = 1 - \sqrt{\frac{16}{16 + 5(9)}} = 1 - \sqrt{\frac{16}{16 + 45}} = 1 - \sqrt{\frac{16}{61}}$$

$$\theta = 1 - \sqrt{0.2623} = 1 - 0.512 = 0.488$$

**Scoring:**
- 4 points for correct calculation
- 1 point for showing steps

---

**Part C: Interpretation (5 points)**

With $\theta = 0.488 \approx 0.5$:

**Interpretation:**
- RE removes about 49% of entity-specific means from the data
- This is roughly halfway between:
  - Pooled OLS ($\theta = 0$): removes 0% of entity means
  - Fixed Effects ($\theta = 1$): removes 100% of entity means

**Implication:**
- RE gives substantial weight to both within and between variation
- Not strongly favoring either extreme
- Entity effects are moderate in importance

**Comparison:**
- If $\theta$ were closer to 0: More weight on between variation (closer to pooled OLS)
- If $\theta$ were closer to 1: More weight on within variation (closer to FE)

**Scoring:**
- 3 points for correct interpretation of $\theta$ value
- 2 points for comparing to extremes

**Key Concept:** $\theta$ determines the blend of within and between variation in RE.

---

### Question 17: Solution (20 points)

**Part A: Entity Means (5 points)**

Calculate means for each firm:

**Firm A:**
- $\bar{Y}_A = (100 + 120 + 140)/3 = 360/3 = 120$
- $\bar{X}_A = (10 + 12 + 14)/3 = 36/3 = 12$

**Firm B:**
- $\bar{Y}_B = (150 + 180 + 210)/3 = 540/3 = 180$
- $\bar{X}_B = (20 + 24 + 28)/3 = 72/3 = 24$

**Scoring:**
- 1.25 points for each correct mean (4 total)
- 1 point for showing calculation method

---

**Part B: GLS Transformation (10 points)**

With $\theta = 0.6$, calculate:

$$Y_{it}^{GLS} = Y_{it} - 0.6 \cdot \bar{Y}_i$$
$$X_{it}^{GLS} = X_{it} - 0.6 \cdot \bar{X}_i$$

| Firm | Year | $Y_{it}$ | $\bar{Y}_i$ | $0.6\bar{Y}_i$ | $Y_{it}^{GLS}$ | $X_{it}$ | $\bar{X}_i$ | $0.6\bar{X}_i$ | $X_{it}^{GLS}$ |
|------|------|----------|------------|---------------|---------------|----------|------------|---------------|---------------|
| A    | 1    | 100      | 120        | 72            | 28            | 10       | 12         | 7.2           | 2.8           |
| A    | 2    | 120      | 120        | 72            | 48            | 12       | 12         | 7.2           | 4.8           |
| A    | 3    | 140      | 120        | 72            | 68            | 14       | 12         | 7.2           | 6.8           |
| B    | 1    | 150      | 180        | 108           | 42            | 20       | 24         | 14.4          | 5.6           |
| B    | 2    | 180      | 180        | 108           | 72            | 24       | 24         | 14.4          | 9.6           |
| B    | 3    | 210      | 180        | 108           | 102           | 28       | 24         | 14.4          | 13.6          |

**Scoring:**
- 5 points for correct $Y_{it}^{GLS}$ values
- 5 points for correct $X_{it}^{GLS}$ values

---

**Part C: Comparison to Within Transformation (5 points)**

**Within Transformation** ($\theta = 1$):

| Firm | Year | Within $Y$ | Within $X$ |
|------|------|-----------|-----------|
| A    | 1    | $100-120=-20$ | $10-12=-2$ |
| A    | 2    | $120-120=0$   | $12-12=0$  |
| A    | 3    | $140-120=20$  | $14-12=2$  |
| B    | 1    | $150-180=-30$ | $20-24=-4$ |
| B    | 2    | $180-180=0$   | $24-24=0$  |
| B    | 3    | $210-180=30$  | $28-24=4$  |

**Comparison:**

For Firm A, Year 1:
- GLS ($\theta=0.6$): $Y^{GLS} = 28$ (removed 72 from original 100)
- Within ($\theta=1$): $\tilde{Y} = -20$ (removed 120 from original 100)

**Interpretation:**
- Within transformation removes **all** entity mean (100%)
- GLS removes only **60%** of entity mean
- GLS retains more of the original variation (40% of entity mean remains)
- This allows GLS to use both within and between variation

**General pattern:**
- $\theta = 0.6$ is a partial demeaning
- Preserves some between-entity differences
- More variation retained → potentially more efficient

**Scoring:**
- 2 points for calculating within transformation examples
- 3 points for clear comparison and interpretation

**Key Concept:** GLS partial demeaning balances within and between variation; within transformation uses only within.

---

## Scoring Rubric

| Points | Grade | Interpretation |
|--------|-------|----------------|
| 90-100 | A     | Excellent understanding of RE |
| 80-89  | B     | Good grasp of concepts |
| 70-79  | C     | Satisfactory understanding |
| 60-69  | D     | Needs improvement |
| 0-59   | F     | Significant gaps |

## Learning Objectives Assessed

- **LO1:** Understand RE model and assumptions (Q1-Q5, Q11-Q15)
- **LO2:** Apply GLS transformation (Q6-Q9, Q16-Q17)
- **LO3:** Compare RE and FE trade-offs (Q10-Q15)
- **LO4:** Calculate variance components and theta (Q16-Q17)

## Study Resources

If you scored below 70%, review:
- Module 3 Guide: Random Effects Model
- Module 3 Guide: GLS Estimation
- Module 3 Notebook: RE Implementation

**Time to Review:** 3-4 hours

## Common Mistakes

1. **Confusing RE assumption** with FE (RE requires orthogonality)
2. **Theta interpretation** backwards (larger theta = more like FE, not less)
3. **Intraclass correlation** interpretation (entity variance proportion)
4. **GLS vs Within** calculations (partial vs complete demeaning)
5. **When to use RE** (need random sampling + orthogonality assumption)
