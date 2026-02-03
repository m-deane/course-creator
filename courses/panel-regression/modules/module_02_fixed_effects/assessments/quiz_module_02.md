# Quiz: Module 2 - Fixed Effects Models

**Course:** Panel Data Econometrics
**Module:** 2 - Fixed Effects Models
**Time Limit:** 30 minutes
**Total Points:** 100
**Passing Score:** 70%

## Instructions

This quiz assesses your understanding of fixed effects estimation, the within transformation, and two-way fixed effects. You have 2 attempts. Show all calculations for computational questions.

---

## Part A: Fixed Effects Fundamentals (40 points)

### Question 1: Fixed Effects Model Specification (8 points)

The fixed effects model is written as:

$$y_{it} = \alpha_i + X_{it}\beta + \epsilon_{it}$$

Which statement about $\alpha_i$ is **most accurate**?

**A)** $\alpha_i$ must be uncorrelated with $X_{it}$ for FE to be consistent

**B)** $\alpha_i$ captures all time-invariant differences between entities, and FE allows correlation with $X_{it}$

**C)** $\alpha_i$ is assumed to be drawn from a normal distribution

**D)** $\alpha_i$ must be estimated for each entity, consuming $N$ degrees of freedom

---

### Question 2: When to Use Fixed Effects (8 points)

A researcher studies the effect of class size on student test scores. She has data on students in different schools over multiple years. Which scenario **most strongly** motivates fixed effects?

**A)** Class sizes vary randomly across schools and years

**B)** School quality (unobserved) is correlated with class size decisions and affects test scores

**C)** All schools have the same average quality

**D)** The researcher wants to estimate the effect of time-invariant school characteristics

---

### Question 3: Within Transformation Mechanics (8 points)

The within transformation creates:

$$\tilde{y}_{it} = y_{it} - \bar{y}_i$$
$$\tilde{X}_{it} = X_{it} - \bar{X}_i$$

What happens to a time-invariant variable $Z_i$ (e.g., gender, location) under this transformation?

**A)** It becomes $\tilde{Z}_i = Z_i - \bar{Z}$ which still varies across $i$

**B)** It becomes zero: $\tilde{Z}_i = Z_i - Z_i = 0$

**C)** It is unaffected: $\tilde{Z}_i = Z_i$

**D)** It becomes the time average: $\tilde{Z}_i = \bar{Z}$

---

### Question 4: LSDV vs Within Estimator (8 points)

The Least Squares Dummy Variable (LSDV) approach includes entity dummies:

$$y_{it} = \sum_{j=1}^{N} \gamma_j D_{ij} + X_{it}\beta + \epsilon_{it}$$

How do the $\hat{\beta}$ coefficients from LSDV compare to within estimator $\hat{\beta}_{FE}$?

**A)** LSDV estimates are biased while within estimates are unbiased

**B)** They are numerically identical for time-varying regressors

**C)** LSDV is more efficient but less consistent

**D)** Within estimator cannot estimate effects of time-invariant variables, but LSDV can

---

### Question 5: Degrees of Freedom in Fixed Effects (8 points)

A balanced panel has $N = 100$ entities, $T = 10$ time periods, and $k = 3$ time-varying regressors. How many degrees of freedom does the fixed effects (within) estimator have?

**A)** 897

**B)** 997

**C)** 900

**D)** 1,000

---

## Part B: Two-Way Fixed Effects (30 points)

### Question 6: Two-Way FE Specification (8 points)

The two-way fixed effects model is:

$$y_{it} = \alpha_i + \lambda_t + X_{it}\beta + \epsilon_{it}$$

What does $\lambda_t$ control for?

**A)** Entity-specific time trends

**B)** Aggregate shocks or common factors affecting all entities in period $t$

**C)** The correlation between $\alpha_i$ and time

**D)** Seasonal patterns within each entity

---

### Question 7: Two-Way FE Transformation (8 points)

The two-way fixed effects transformation can be implemented by demeaning twice:

$$\ddot{y}_{it} = y_{it} - \bar{y}_i - \bar{y}_t + \bar{y}$$

where $\bar{y}_i$ is the entity mean, $\bar{y}_t$ is the time mean, and $\bar{y}$ is the grand mean.

Why is the grand mean $\bar{y}$ added back?

**A)** To ensure residuals sum to zero

**B)** To avoid over-subtracting (entity and time means both include the grand mean)

**C)** To maintain the scale of the original variable

**D)** To make the intercept interpretable

---

### Question 8: When to Include Time Fixed Effects (7 points)

A researcher studies firm investment using quarterly data from 2015-2023. When should she include time fixed effects?

**A)** Only if firms have different fiscal years

**B)** To control for aggregate business cycle effects, policy changes, or macroeconomic shocks common to all firms

**C)** Only if the panel is unbalanced

**D)** Never, because time effects reduce statistical power

---

### Question 9: Two-Way FE Degrees of Freedom (7 points)

Compared to one-way entity FE, adding time fixed effects in a two-way model:

**A)** Reduces degrees of freedom by $T - 1$

**B)** Reduces degrees of freedom by $T$

**C)** Has no effect on degrees of freedom if $T$ is small

**D)** Increases degrees of freedom by removing entity effects

---

## Part C: Fixed Effects Properties and Interpretation (30 points)

### Question 10: FE Coefficient Interpretation (6 points)

In a wage regression with worker fixed effects:

$$wage_{it} = \alpha_i + \beta \cdot experience_{it} + \epsilon_{it}$$

The estimated $\hat{\beta} = 0.05$ means:

**A)** Workers with one more year of experience earn 5% more on average

**B)** A one-year increase in experience is associated with a $0.05 increase in wages, controlling for all time-invariant worker characteristics

**C)** Experience causes a $0.05 increase in wages

**D)** Between-worker differences in experience explain 5% of wage variation

---

### Question 11: What Fixed Effects Control For (6 points)

Worker fixed effects in a wage equation control for which of the following?

**A)** Worker ability, family background, and other time-invariant characteristics

**B)** Current job characteristics that vary over time

**C)** Macroeconomic conditions

**D)** Random measurement error in wages

---

### Question 12: Fixed Effects Assumptions (6 points)

Which assumption is **required** for fixed effects to produce consistent estimates of $\beta$?

**A)** $\alpha_i$ is uncorrelated with $X_{it}$

**B)** $\epsilon_{it}$ is uncorrelated with $X_{it}$ (strict exogeneity conditional on $\alpha_i$)

**C)** $\alpha_i$ is normally distributed

**D)** The panel must be balanced

---

### Question 13: Standard Errors in FE Models (6 points)

Why should you typically use **clustered standard errors** (clustered by entity) in fixed effects models?

**A)** To account for correlation of errors within entities over time

**B)** To adjust for the loss of degrees of freedom from entity dummies

**C)** To make standard errors robust to outliers

**D)** To correct for measurement error in the dependent variable

---

### Question 14: FE vs Pooled OLS Comparison (6 points)

Compared to pooled OLS, fixed effects estimators typically have:

**A)** Smaller standard errors because they use within variation efficiently

**B)** Larger standard errors because they use less variation and lose degrees of freedom

**C)** Identical standard errors if $\alpha_i$ is uncorrelated with $X_{it}$

**D)** Standard errors that depend on whether the panel is balanced

---

## Calculation Problems (Show All Work)

### Question 15: Within Transformation Calculation (12 points)

Consider the following panel data on firm profits and R&D spending:

| Firm | Year | Profit ($Y_{it}$) | R&D ($X_{it}$) |
|------|------|------------------|---------------|
| A    | 1    | 50               | 5             |
| A    | 2    | 60               | 7             |
| A    | 3    | 70               | 9             |
| B    | 1    | 100              | 10            |
| B    | 2    | 120              | 14            |
| B    | 3    | 140              | 18            |

**Part A (4 points):** Calculate the entity means $\bar{Y}_A$, $\bar{Y}_B$, $\bar{X}_A$, and $\bar{X}_B$.

**Part B (4 points):** Calculate the within-transformed variables $\tilde{Y}_{it}$ and $\tilde{X}_{it}$ for all six observations.

**Part C (4 points):** Using your within-transformed data, calculate the fixed effects estimator:

$$\hat{\beta}_{FE} = \frac{\sum_{i=1}^N \sum_{t=1}^T \tilde{X}_{it}\tilde{Y}_{it}}{\sum_{i=1}^N \sum_{t=1}^T \tilde{X}_{it}^2}$$

---

### Question 16: Two-Way Fixed Effects Setup (8 points)

Consider a balanced panel with 3 firms observed over 2 years:

| Firm | Year | Sales ($Y_{it}$) |
|------|------|-----------------|
| 1    | 2020 | 100             |
| 1    | 2021 | 110             |
| 2    | 2020 | 150             |
| 2    | 2021 | 165             |
| 3    | 2020 | 80              |
| 3    | 2021 | 85              |

**Part A (4 points):** Calculate the entity means ($\bar{Y}_i$), time means ($\bar{Y}_t$), and grand mean ($\bar{Y}$).

**Part B (4 points):** Calculate the two-way demeaned values:

$$\ddot{Y}_{it} = Y_{it} - \bar{Y}_i - \bar{Y}_t + \bar{Y}$$

---

## Answer Key and Explanations

### Question 1: Answer B (8 points)

**Correct Answer:** B) $\alpha_i$ captures all time-invariant differences between entities, and FE allows correlation with $X_{it}$

**Explanation:**

This is the key advantage of fixed effects:
- $\alpha_i$ represents all time-invariant unobserved heterogeneity
- Can include ability, preferences, location, institutional factors, etc.
- FE is consistent even if $Cov(\alpha_i, X_{it}) \neq 0$
- This addresses a major source of endogeneity (omitted variables bias)

**Why other answers are wrong:**
- A: FALSE - FE allows correlation (this is the point!)
- C: No distributional assumption needed (unlike random effects)
- D: Degrees of freedom are consumed, but $\alpha_i$ need not be explicitly estimated in within transformation

**Key Concept:** Fixed effects controls for unobserved heterogeneity by allowing arbitrary correlation with regressors.

---

### Question 2: Answer B (8 points)

**Correct Answer:** B) School quality (unobserved) is correlated with class size decisions and affects test scores

**Explanation:**

This is a classic endogeneity problem that FE addresses:
- Better schools may have resources for smaller classes
- Better schools also have better teachers, facilities, etc.
- Omitting school quality creates bias: $Cov(classsize, quality) \neq 0$
- School FE absorbs all time-invariant school quality

**Why other answers don't motivate FE:**
- A: Random variation means pooled OLS is unbiased
- C: No unobserved heterogeneity to control for
- D: FE cannot estimate effects of time-invariant variables

**Key Concept:** Use FE when unobserved entity characteristics correlate with treatment.

---

### Question 3: Answer B (8 points)

**Correct Answer:** B) It becomes zero: $\tilde{Z}_i = Z_i - Z_i = 0$

**Explanation:**

For time-invariant $Z_i$ (constant across all $t$ for entity $i$):

$$\bar{Z}_i = \frac{1}{T}\sum_{t=1}^T Z_i = \frac{1}{T}(T \cdot Z_i) = Z_i$$

Therefore:
$$\tilde{Z}_i = Z_i - \bar{Z}_i = Z_i - Z_i = 0$$

**Implication:** Time-invariant variables are perfectly collinear with entity fixed effects and drop out. You cannot estimate their effects with FE.

**Examples:** Gender, race, country, industry (if firms don't switch), permanent location.

**Key Concept:** Within transformation eliminates time-invariant variables.

---

### Question 4: Answer B (8 points)

**Correct Answer:** B) They are numerically identical for time-varying regressors

**Explanation:**

For time-varying regressors, LSDV and within estimator are algebraically equivalent:

**LSDV:**
$$y_{it} = \gamma_1 D_{i1} + \gamma_2 D_{i2} + \cdots + \gamma_N D_{iN} + X_{it}\beta + \epsilon_{it}$$

**Within:**
$$(y_{it} - \bar{y}_i) = (X_{it} - \bar{X}_i)\beta + (\epsilon_{it} - \bar{\epsilon}_i)$$

Both give identical $\hat{\beta}$. LSDV also estimates $\hat{\alpha}_i = \hat{\gamma}_i$, which within doesn't explicitly report.

**Why other answers are wrong:**
- A: Both are unbiased under same assumptions
- C: Efficiency is identical
- D: Neither can estimate time-invariant effects (they drop out in LSDV due to multicollinearity)

**Key Concept:** LSDV and within are computationally different but statistically equivalent.

---

### Question 5: Answer A (8 points)

**Correct Answer:** A) 897

**Calculation:**

Within estimator degrees of freedom:

$$df = NT - N - k = (100 \times 10) - 100 - 3 = 1000 - 100 - 3 = 897$$

**Explanation:**
- Total observations: $NT = 1,000$
- Entity fixed effects: lose $N = 100$ DF (entity demeaning)
- Time-varying regressors: lose $k = 3$ DF
- Net DF: $1000 - 100 - 3 = 897$

**Why other answers are wrong:**
- B: Forgot to subtract $N$
- C: Only subtracted 100 but forgot the 3 regressors
- D: That's total observations, not DF

**Key Concept:** FE loses $N$ degrees of freedom compared to pooled OLS.

---

### Question 6: Answer B (8 points)

**Correct Answer:** B) Aggregate shocks or common factors affecting all entities in period $t$

**Explanation:**

Time fixed effects ($\lambda_t$) capture:
- Business cycles and macroeconomic conditions
- Policy changes affecting all entities
- Technological shocks
- Seasonal patterns common across entities
- Any time-varying factors that affect all entities equally

**Example:** In a firm panel, $\lambda_t$ controls for GDP growth, interest rates, regulatory changes.

**Why other answers are less accurate:**
- A: Entity-specific trends require interaction terms $(\alpha_i \times t)$
- C: Not about correlation but common time effects
- D: Seasonality within entity is different; this is aggregate seasonality

**Key Concept:** Time FE removes aggregate time-varying confounds.

---

### Question 7: Answer B (8 points)

**Correct Answer:** B) To avoid over-subtracting (entity and time means both include the grand mean)

**Explanation:**

Consider the double-demeaning process:

$$y_{it} - \bar{y}_i - \bar{y}_t = (y_{it} - \bar{y}) - (\bar{y}_i - \bar{y}) - (\bar{y}_t - \bar{y})$$

If we just subtract $\bar{y}_i$ and $\bar{y}_t$:
- $\bar{y}_i$ includes the grand mean
- $\bar{y}_t$ includes the grand mean
- We'd subtract $\bar{y}$ twice!

Adding back $\bar{y}$ corrects for this double subtraction.

**Algebraic verification:**
$$y_{it} - \bar{y}_i - \bar{y}_t + \bar{y} = y_{it} - \bar{y} - (\bar{y}_i - \bar{y}) - (\bar{y}_t - \bar{y})$$

**Key Concept:** Two-way demeaning requires adding back grand mean to avoid double-counting.

---

### Question 8: Answer B (7 points)

**Correct Answer:** B) To control for aggregate business cycle effects, policy changes, or macroeconomic shocks common to all firms

**Explanation:**

Time FE are crucial when:
- All firms experience common shocks (recessions, Fed policy, tax changes)
- Omitting these creates correlation: $Cov(X_{it}, \lambda_t) \neq 0$
- Quarterly data especially susceptible to seasonal and business cycle effects

**Example:** In Q2 2020, all firms affected by COVID-19. Time FE absorbs this common shock.

**Why other answers are wrong:**
- A: Fiscal year differences don't require time FE
- C: Time FE useful in balanced and unbalanced panels
- D: While they reduce power slightly, bias reduction usually dominates

**Key Concept:** Include time FE when common aggregate shocks affect all entities.

---

### Question 9: Answer A (7 points)

**Correct Answer:** A) Reduces degrees of freedom by $T - 1$

**Explanation:**

Two-way FE includes:
- Entity FE: $N - 1$ dummies (one is baseline)
- Time FE: $T - 1$ dummies (one is baseline)

Adding time FE to one-way entity FE model:
$$df_{one-way} = NT - N - k$$
$$df_{two-way} = NT - N - (T-1) - k = NT - N - T + 1 - k$$

**Difference:** $df_{one-way} - df_{two-way} = T - 1$

**For large $N$ small $T$:** Losing $T-1$ DF is minor.

**Key Concept:** Time FE costs $T-1$ degrees of freedom.

---

### Question 10: Answer B (6 points)

**Correct Answer:** B) A one-year increase in experience is associated with a $0.05 increase in wages, controlling for all time-invariant worker characteristics

**Explanation:**

FE estimates **within effects**: changes in $Y$ associated with changes in $X$ for the same individual:

$$\Delta wage_i = \beta \cdot \Delta experience_i$$

The coefficient compares the same worker at different experience levels, holding constant ability, education, gender, etc. (all time-invariant factors absorbed by $\alpha_i$).

**Why other answers are imprecise:**
- A: This describes between-worker comparisons (cross-sectional), not within
- C: "Causes" is too strong without additional assumptions (strict exogeneity)
- D: This describes variance decomposition, not coefficient interpretation

**Key Concept:** FE coefficients represent within-entity changes.

---

### Question 11: Answer A (6 points)

**Correct Answer:** A) Worker ability, family background, and other time-invariant characteristics

**Explanation:**

Worker fixed effects ($\alpha_i$) absorb:
- Innate ability
- Education (typically completed before panel)
- Family background
- Personality traits
- Gender, race, ethnicity
- Other permanent worker attributes

These are controlled for even if unobserved.

**Why other answers are wrong:**
- B: Time-varying job characteristics must be included explicitly
- C: Macroeconomic conditions require time fixed effects
- D: FE doesn't address classical measurement error

**Key Concept:** Entity FE control for all time-invariant entity characteristics, observed or not.

---

### Question 12: Answer B (6 points)

**Correct Answer:** B) $\epsilon_{it}$ is uncorrelated with $X_{it}$ (strict exogeneity conditional on $\alpha_i$)

**Explanation:**

**Strict exogeneity** in FE context:

$$E[\epsilon_{it} | X_{i1}, ..., X_{iT}, \alpha_i] = 0$$

This means errors are uncorrelated with regressors in all time periods, conditional on fixed effects. It's weaker than full exogeneity (allows $\alpha_i$ to correlate with $X_{it}$) but stronger than contemporaneous exogeneity.

**Violated by:** Lagged dependent variables, feedback effects.

**Why other answers are wrong:**
- A: FALSE - FE allows this correlation (that's the point!)
- C: No distributional assumptions needed for consistency
- D: FE works with unbalanced panels (with some caveats)

**Key Concept:** FE requires strict exogeneity of idiosyncratic errors, not of $\alpha_i$.

---

### Question 13: Answer A (6 points)

**Correct Answer:** A) To account for correlation of errors within entities over time

**Explanation:**

Even after removing $\alpha_i$, errors may be correlated within entity:

$$Cov(\epsilon_{it}, \epsilon_{is}) \neq 0 \text{ for } t \neq s$$

**Sources:**
- Persistent shocks to individual entities
- Serial correlation in omitted time-varying factors
- Measurement error persistence

**Clustering** by entity accounts for arbitrary within-entity correlation:

$$\hat{V}_{cluster} = (X'X)^{-1}\left(\sum_{i=1}^N X_i' \hat{u}_i \hat{u}_i' X_i\right)(X'X)^{-1}$$

**Why other answers are less accurate:**
- B: DF adjustment is separate from clustering
- C: Outlier robustness requires different methods
- D: Clustering doesn't fix measurement error

**Key Concept:** Cluster standard errors by entity to allow within-entity error correlation.

---

### Question 14: Answer B (6 points)

**Correct Answer:** B) Larger standard errors because they use less variation and lose degrees of freedom

**Explanation:**

FE typically has larger SEs because:
1. **Less variation:** Uses only within variation, discards between variation
2. **Degrees of freedom:** Loses $N$ DF
3. **Efficiency loss:** When $Cov(\alpha_i, X_{it}) = 0$, pooled OLS is more efficient

**Trade-off:** FE sacrifices precision for consistency (bias reduction).

$$Var(\hat{\beta}_{FE}) \geq Var(\hat{\beta}_{pooled})$$

**When FE has smaller SEs:** Rare, but possible if removing $\alpha_i$ substantially reduces error variance.

**Key Concept:** FE trades precision for consistency by eliminating bias from unobserved heterogeneity.

---

### Question 15: Answer and Calculations (12 points)

**Part A: Entity Means (4 points)**

Calculate means for each firm:

**Firm A:**
- $\bar{Y}_A = (50 + 60 + 70)/3 = 180/3 = 60$
- $\bar{X}_A = (5 + 7 + 9)/3 = 21/3 = 7$

**Firm B:**
- $\bar{Y}_B = (100 + 120 + 140)/3 = 360/3 = 120$
- $\bar{X}_B = (10 + 14 + 18)/3 = 42/3 = 14$

**Scoring:**
- 1 point per correct mean (4 total)

---

**Part B: Within Transformation (4 points)**

Calculate $\tilde{Y}_{it} = Y_{it} - \bar{Y}_i$ and $\tilde{X}_{it} = X_{it} - \bar{X}_i$:

| Firm | Year | $Y_{it}$ | $\bar{Y}_i$ | $\tilde{Y}_{it}$ | $X_{it}$ | $\bar{X}_i$ | $\tilde{X}_{it}$ |
|------|------|----------|------------|-----------------|----------|------------|-----------------|
| A    | 1    | 50       | 60         | -10             | 5        | 7          | -2              |
| A    | 2    | 60       | 60         | 0               | 7        | 7          | 0               |
| A    | 3    | 70       | 60         | 10              | 9        | 7          | 2               |
| B    | 1    | 100      | 120        | -20             | 10       | 14         | -4              |
| B    | 2    | 120      | 120        | 0               | 14       | 14         | 0               |
| B    | 3    | 140      | 120        | 20              | 18       | 14         | 4               |

**Verification:** Within each firm, demeaned values sum to zero. ✓

**Scoring:**
- 2 points for correct $\tilde{Y}_{it}$ values
- 2 points for correct $\tilde{X}_{it}$ values

---

**Part C: Fixed Effects Estimator (4 points)**

Calculate the FE coefficient:

$$\hat{\beta}_{FE} = \frac{\sum_{i} \sum_{t} \tilde{X}_{it}\tilde{Y}_{it}}{\sum_{i} \sum_{t} \tilde{X}_{it}^2}$$

**Numerator:** $\sum \tilde{X}_{it}\tilde{Y}_{it}$

| Firm | Year | $\tilde{X}_{it}$ | $\tilde{Y}_{it}$ | $\tilde{X}_{it} \cdot \tilde{Y}_{it}$ |
|------|------|-----------------|-----------------|-------------------------------------|
| A    | 1    | -2              | -10             | 20                                  |
| A    | 2    | 0               | 0               | 0                                   |
| A    | 3    | 2               | 10              | 20                                  |
| B    | 1    | -4              | -20             | 80                                  |
| B    | 2    | 0               | 0               | 0                                   |
| B    | 3    | 4               | 20              | 80                                  |

Sum: $20 + 0 + 20 + 80 + 0 + 80 = 200$

**Denominator:** $\sum \tilde{X}_{it}^2$

| Firm | Year | $\tilde{X}_{it}$ | $\tilde{X}_{it}^2$ |
|------|------|-----------------|-------------------|
| A    | 1    | -2              | 4                 |
| A    | 2    | 0               | 0                 |
| A    | 3    | 2               | 4                 |
| B    | 1    | -4              | 16                |
| B    | 2    | 0               | 0                 |
| B    | 3    | 4               | 16                |

Sum: $4 + 0 + 4 + 16 + 0 + 16 = 40$

**Fixed Effects Estimate:**

$$\hat{\beta}_{FE} = \frac{200}{40} = 5$$

**Interpretation:** A one-unit increase in R&D spending is associated with a 5-unit increase in profits, controlling for time-invariant firm characteristics.

**Scoring:**
- 2 points for correct numerator calculation
- 1 point for correct denominator calculation
- 1 point for correct final estimate

---

### Question 16: Answer and Calculations (8 points)

**Part A: Means Calculation (4 points)**

**Entity means:**
- $\bar{Y}_1 = (100 + 110)/2 = 105$
- $\bar{Y}_2 = (150 + 165)/2 = 157.5$
- $\bar{Y}_3 = (80 + 85)/2 = 82.5$

**Time means:**
- $\bar{Y}_{2020} = (100 + 150 + 80)/3 = 330/3 = 110$
- $\bar{Y}_{2021} = (110 + 165 + 85)/3 = 360/3 = 120$

**Grand mean:**
- $\bar{Y} = (100 + 110 + 150 + 165 + 80 + 85)/6 = 690/6 = 115$

**Scoring:**
- 1 point for entity means
- 1 point for time means
- 1 point for grand mean
- 1 point for all correct

---

**Part B: Two-Way Demeaning (4 points)**

Calculate $\ddot{Y}_{it} = Y_{it} - \bar{Y}_i - \bar{Y}_t + \bar{Y}$:

| Firm | Year | $Y_{it}$ | $\bar{Y}_i$ | $\bar{Y}_t$ | $\bar{Y}$ | $\ddot{Y}_{it}$ |
|------|------|----------|------------|------------|----------|----------------|
| 1    | 2020 | 100      | 105        | 110        | 115      | $100-105-110+115=0$ |
| 1    | 2021 | 110      | 105        | 120        | 115      | $110-105-120+115=0$ |
| 2    | 2020 | 150      | 157.5      | 110        | 115      | $150-157.5-110+115=-2.5$ |
| 2    | 2021 | 165      | 157.5      | 120        | 115      | $165-157.5-120+115=2.5$ |
| 3    | 2020 | 80       | 82.5       | 110        | 115      | $80-82.5-110+115=2.5$ |
| 3    | 2021 | 85       | 82.5       | 120        | 115      | $85-82.5-120+115=-2.5$ |

**Verification:**
- Sum across all observations: $0 + 0 - 2.5 + 2.5 + 2.5 - 2.5 = 0$ ✓
- Sum within each entity: $0$ ✓
- Sum within each time period: $0$ ✓

**Scoring:**
- 3 points for correct calculations
- 1 point for proper formula application

**Key Concept:** Two-way FE removes both entity and time-specific means.

---

## Scoring Rubric

| Points | Grade | Interpretation |
|--------|-------|----------------|
| 90-100 | A     | Excellent mastery of FE concepts |
| 80-89  | B     | Good understanding |
| 70-79  | C     | Satisfactory |
| 60-69  | D     | Needs review |
| 0-59   | F     | Significant gaps |

## Learning Objectives Assessed

- **LO1:** Understand FE model and when to use it (Q1-Q2, Q10-Q12)
- **LO2:** Apply within transformation (Q3-Q5, Q15)
- **LO3:** Implement two-way fixed effects (Q6-Q9, Q16)
- **LO4:** Interpret FE estimates correctly (Q10-Q14)

## Study Resources

If you scored below 70%, review:
- Module 2 Guide: Fixed Effects Intuition
- Module 2 Guide: Within Transformation
- Module 2 Notebook: FE Implementation

**Time to Review:** 4-5 hours
