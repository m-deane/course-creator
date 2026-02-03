# Quiz: Module 1 - Panel Data Structure

**Course:** Panel Data Econometrics
**Module:** 1 - Panel Data Structure
**Time Limit:** 30 minutes
**Total Points:** 100
**Passing Score:** 70%

## Instructions

This quiz assesses your understanding of panel data fundamentals, pooled OLS, and the between-within decomposition. You have 2 attempts. Show calculations where requested.

---

## Part A: Pooled OLS and Panel Fundamentals (35 points)

### Question 1: Pooled OLS Specification (7 points)

Consider the pooled OLS model:

$$y_{it} = \beta_0 + \beta_1 x_{it} + \epsilon_{it}$$

Which assumption is **most problematic** for pooled OLS in panel data contexts?

**A)** $E[\epsilon_{it}] = 0$

**B)** $E[\epsilon_{it}\epsilon_{js}] = 0$ for all $i, j, t, s$

**C)** $E[x_{it}\epsilon_{it}] = 0$

**D)** $Var(\epsilon_{it}) = \sigma^2$

---

### Question 2: Panel Data Advantages (7 points)

Which of the following is **NOT** an advantage of panel data over pure cross-sectional data?

**A)** Ability to control for unobserved heterogeneity across entities

**B)** Increased sample size and statistical power

**C)** Easier to satisfy the exogeneity assumption

**D)** Ability to study dynamics and causal effects over time

---

### Question 3: Pooled OLS Estimation (7 points)

A researcher estimates the effect of R&D spending on firm profits using pooled OLS with data on 200 firms over 5 years (balanced panel). How many degrees of freedom does this regression have if there are 3 regressors (including the intercept)?

**A)** 997

**B)** 1,000

**C)** 197

**D)** 795

---

### Question 4: Panel Data Types (7 points)

A researcher has data on wages for 1,000 workers over 20 years. This panel is best characterized as:

**A)** Short panel (small $T$, large $N$)

**B)** Long panel (large $T$, small $N$)

**C)** Balanced panel only if all workers are observed for all years

**D)** Both B and C could be correct depending on missingness

---

### Question 5: Pooled OLS Limitations (7 points)

Consider the wage equation:

$$wage_{it} = \beta_0 + \beta_1 educ_i + \beta_2 exper_{it} + \epsilon_{it}$$

Why might pooled OLS produce biased estimates of $\beta_2$ (the experience effect)?

**A)** Education is time-invariant

**B)** Unobserved ability affects both wage and experience accumulation

**C)** There are too many observations

**D)** Experience is perfectly collinear with time

---

## Part B: Between and Within Variation (40 points)

### Question 6: Variation Decomposition (8 points)

In panel data, any variable $x_{it}$ can be decomposed as:

$$x_{it} = \bar{x}_i + (x_{it} - \bar{x}_i)$$

What do these two components represent?

**A)** Time-series variation and cross-sectional variation

**B)** Between-entity variation and within-entity variation

**C)** Systematic variation and random variation

**D)** Explained variation and residual variation

---

### Question 7: Between Estimator Calculation (8 points)

The between estimator uses only cross-sectional variation by regressing:

$$\bar{y}_i = \beta_0 + \beta_1 \bar{x}_i + \bar{\epsilon}_i$$

If you have $N = 50$ firms and $T = 10$ years, how many observations does the between estimator use?

**A)** 500

**B)** 50

**C)** 10

**D)** 49

---

### Question 8: Within Variation Concept (8 points)

Consider firm-level data where firm A's revenue is [100, 105, 110] over three years, and firm B's revenue is [200, 205, 210].

Which statement about variation is correct?

**A)** The between variation is larger than the within variation

**B)** The within variation is the same for both firms

**C)** There is no between variation because both firms grow at the same rate

**D)** Pooled OLS cannot distinguish these patterns

---

### Question 9: Between vs Within Estimators (8 points)

A researcher studies the effect of capital stock on output. Capital varies substantially across firms (between) but changes slowly within firms over time (small within variation).

Which estimator will likely produce more precise estimates?

**A)** Within estimator, because it uses more variation

**B)** Between estimator, because most variation is between firms

**C)** Pooled OLS, because it uses all variation equally

**D)** They will all have identical standard errors

---

### Question 10: Variance Decomposition Formula (8 points)

The total variance of $x$ can be decomposed as:

$$Var_{total}(x_{it}) = Var_{between}(\bar{x}_i) + E[Var_{within}(x_{it} - \bar{x}_i)]$$

If $Var_{total} = 100$ and $Var_{between} = 70$, what is the expected within variance?

**A)** 30

**B)** 170

**C)** 70

**D)** Cannot be determined without knowing $N$ and $T$

---

## Part C: Data Structure and Manipulation (25 points)

### Question 11: Panel Data Reshaping (5 points)

You have wide-format data:

| id | y_2020 | y_2021 | y_2022 |
|----|--------|--------|--------|
| 1  | 10     | 12     | 14     |
| 2  | 20     | 22     | 24     |

In Python pandas, which function converts this to long format?

**A)** `df.pivot()`

**B)** `df.melt(id_vars=['id'])`

**C)** `df.reshape()`

**D)** `df.transpose()`

---

### Question 12: Missing Data Patterns (5 points)

Which missing data pattern is **most problematic** for causal inference in panel data?

**A)** Random missingness unrelated to outcomes or characteristics

**B)** Missingness due to survey non-response that is unrelated to the outcome

**C)** Firms with poor performance are more likely to drop out of the panel

**D)** All missing data patterns are equally problematic

---

### Question 13: Panel Data Indexing (5 points)

In a balanced panel with $N = 100$ and $T = 5$, observation 347 corresponds to which entity and time period (assuming entities are indexed 1-100 and periods 1-5, with data sorted by entity then time)?

**A)** Entity 69, Period 3

**B)** Entity 70, Period 2

**C)** Entity 69, Period 2

**D)** Entity 70, Period 3

---

### Question 14: Data Quality Issue (5 points)

A researcher has monthly panel data on retail sales for 500 stores from January 2020 to December 2023. She notices that 50 stores have missing data for April-June 2020. What is the most likely explanation, and how should she handle it?

**A)** Random measurement error; drop those stores

**B)** COVID-19 closures; use pandemic indicators or restrict analysis period

**C)** Data entry errors; impute with store averages

**D)** Intentional non-response; no adjustment needed

---

### Question 15: Demeaning Process (5 points)

The within transformation subtracts entity-specific means:

$$\tilde{y}_{it} = y_{it} - \bar{y}_i$$

If firm A has observations $y_{A,t} = [5, 7, 9]$ across three periods, what are the demeaned values?

**A)** $[-2, 0, 2]$

**B)** $[0, 2, 4]$

**C)** $[5, 7, 9]$

**D)** $[-3, -1, 1]$

---

## Calculation Problems (Partial Credit Available)

### Question 16: Between and Within Decomposition (15 points)

Consider the following panel data for three firms over two years:

| Firm | Year | Revenue ($Y_{it}$) | Advertising ($X_{it}$) |
|------|------|-------------------|----------------------|
| 1    | 1    | 100               | 10                   |
| 1    | 2    | 120               | 12                   |
| 2    | 1    | 200               | 15                   |
| 2    | 2    | 220               | 18                   |
| 3    | 1    | 150               | 8                    |
| 3    | 2    | 170               | 11                   |

**Part A (5 points):** Calculate $\bar{Y}_i$ for each firm.

**Part B (5 points):** Calculate the within-transformed values $\tilde{Y}_{it} = Y_{it} - \bar{Y}_i$ for all observations.

**Part C (5 points):** If you ran a between regression using only the firm averages ($\bar{Y}_i$ on $\bar{X}_i$), how many observations would you have, and what would $\bar{X}_i$ be for each firm?

---

## Answer Key and Explanations

### Question 1: Answer B (7 points)

**Correct Answer:** B) $E[\epsilon_{it}\epsilon_{js}] = 0$ for all $i, j, t, s$

**Explanation:**

In panel data, observations for the same entity are likely correlated over time (serial correlation) or across entities at the same time (cross-sectional correlation). The assumption that all error terms are independent is unrealistic:

- $E[\epsilon_{it}\epsilon_{is}] \neq 0$ for $t \neq s$ (serial correlation within entity)
- Individual heterogeneity creates persistent error components

This violates the standard OLS assumption of independent errors, leading to incorrect standard errors (though point estimates remain unbiased under exogeneity).

**Why other answers are less problematic:**
- A: Zero mean is typically satisfied by including an intercept
- C: Exogeneity might hold in some applications
- D: Heteroskedasticity is concerning but often easier to correct than serial correlation

**Key Concept:** Panel data structure creates natural dependence in errors, violating the i.i.d. assumption.

---

### Question 2: Answer C (7 points)

**Correct Answer:** C) Easier to satisfy the exogeneity assumption

**Explanation:**

Panel data does NOT make exogeneity easier to achieve automatically. In fact, panel data often highlights endogeneity problems by revealing unobserved heterogeneity. However, panel methods (like fixed effects) can help address some endogeneity issues.

**Why other answers ARE advantages:**
- A: TRUE - Fixed effects control for time-invariant unobservables
- B: TRUE - More observations increase power
- D: TRUE - Repeated observations enable studying dynamics

**Key Concept:** Panel data provides tools to address endogeneity but doesn't eliminate it automatically.

---

### Question 3: Answer A (7 points)

**Correct Answer:** A) 997

**Calculation:**

- Total observations: $N \times T = 200 \times 5 = 1,000$
- Parameters estimated: $k = 3$ (including intercept)
- Degrees of freedom: $df = n - k = 1,000 - 3 = 997$

Pooled OLS treats all 1,000 observations as independent (which is problematic!) and estimates 3 parameters.

**Why other answers are wrong:**
- B: That's the total number of observations, not DF
- C: That would be $N - k$, using only cross-sectional dimension
- D: Arbitrary calculation

**Key Concept:** In pooled OLS, $df = NT - k$.

---

### Question 4: Answer D (7 points)

**Correct Answer:** D) Both B and C could be correct depending on missingness

**Explanation:**

With $T = 20$ and $N = 1,000$:
- This is a **long panel** relative to typical micro panels ($T > 10$ is often considered "long")
- However, it's only balanced if all workers are observed for all years
- Attrition, entry, and missing data can make it unbalanced

Both characterizations (B and C) are relevant.

**Why other answers are incomplete:**
- A: With $T = 20$, this is not typically classified as a "short" panel
- B: Correct about long panel but incomplete
- C: Correct about balance requirement but incomplete

**Key Concept:** Panel classification depends on both $T$ relative to $N$ and completeness of observations.

---

### Question 5: Answer B (7 points)

**Correct Answer:** B) Unobserved ability affects both wage and experience accumulation

**Explanation:**

Unobserved ability creates endogeneity:
- Higher ability workers earn more (affects $wage_{it}$)
- Higher ability workers may accumulate experience differently (affects $exper_{it}$)
- This creates $Cov(exper_{it}, \epsilon_{it}) \neq 0$ if ability is in $\epsilon_{it}$

The bias: $E[\hat{\beta}_2] = \beta_2 + \frac{Cov(exper, ability)}{Var(exper)} \times effect\_of\_ability\_on\_wage$

**Why other answers are wrong:**
- A: Time-invariance of education is not the issue for estimating $\beta_2$
- C: More observations improve precision but don't eliminate bias
- D: Experience is not perfectly collinear with time (varies across workers)

**Key Concept:** Pooled OLS suffers from omitted variable bias when unobserved heterogeneity correlates with regressors.

---

### Question 6: Answer B (8 points)

**Correct Answer:** B) Between-entity variation and within-entity variation

**Explanation:**

The decomposition separates:
- $\bar{x}_i$: **Between variation** - differences in entity-specific means (across entities)
- $(x_{it} - \bar{x}_i)$: **Within variation** - deviations from entity means (over time within entity)

This is fundamental to panel data methods:
- Between estimator uses only $\bar{x}_i$
- Within estimator uses only $(x_{it} - \bar{x}_i)$
- Pooled OLS uses both

**Why other answers are less precise:**
- A: "Time-series" suggests single entity over time
- C: Not specifically about panel structure
- D: Refers to model fit, not data structure

**Key Concept:** Panel variation has two distinct sources: between entities and within entities over time.

---

### Question 7: Answer B (8 points)

**Correct Answer:** B) 50

**Explanation:**

The between estimator:
1. Computes time averages for each entity: $\bar{y}_i$, $\bar{x}_i$
2. Runs regression on these entity-level means
3. Uses $N$ observations (one per entity)

With $N = 50$ firms, you get 50 observations in the between regression, regardless of $T$.

**Why other answers are wrong:**
- A: That's the total panel observations ($NT$)
- C: That's the number of time periods
- D: That's $N - 1$ (degrees of freedom in some contexts, but not the number of observations)

**Key Concept:** Between estimator is a cross-sectional regression on entity means.

---

### Question 8: Answer B (8 points)

**Correct Answer:** B) The within variation is the same for both firms

**Explanation:**

**Within variation** (deviations from entity mean):
- Firm A: Mean = 105, deviations = [-5, 0, 5]
- Firm B: Mean = 205, deviations = [-5, 0, 5]
- Same within variation pattern!

**Between variation** (difference in means):
- Firm A mean: 105
- Firm B mean: 205
- Between difference: 100

**Why other answers are wrong:**
- A: Between variation (100) exceeds within variation range (10), TRUE, but B is more specific
- C: Between variation exists (205 vs 105)
- D: Pooled OLS uses both patterns but doesn't distinguish their sources

**Key Concept:** Firms can have identical growth patterns (within) but different levels (between).

---

### Question 9: Answer B (8 points)

**Correct Answer:** B) Between estimator, because most variation is between firms

**Explanation:**

Estimator precision depends on variation in the regressor:
- $SE(\hat{\beta}) \propto 1/\sqrt{Var(X)}$
- If capital varies mostly between firms (large $Var(\bar{X}_i)$) but little within firms (small $Var(X_{it} - \bar{X}_i)$):
  - Between estimator uses large variation → smaller SE
  - Within estimator uses small variation → larger SE

**Trade-off:** Between is more precise here, but within controls for unobserved heterogeneity (often more important than precision).

**Why other answers are wrong:**
- A: Within uses less variation in this case
- C: Pooled OLS doesn't necessarily use variation "equally"
- D: SEs differ based on which variation source is used

**Key Concept:** Precision depends on regressor variation; bias depends on unobserved heterogeneity.

---

### Question 10: Answer A (8 points)

**Correct Answer:** A) 30

**Explanation:**

The variance decomposition is additive:

$$Var_{total} = Var_{between} + Var_{within}$$

Given:
- $Var_{total} = 100$
- $Var_{between} = 70$

Therefore:
$$Var_{within} = 100 - 70 = 30$$

**Interpretation:** 70% of total variation is between entities, 30% is within entities over time.

**Why other answers are wrong:**
- B: Added instead of subtracted
- C: That's the between variance
- D: The formula is straightforward; $N$ and $T$ not needed for this calculation

**Key Concept:** Total variation decomposes additively into between and within components.

---

### Question 11: Answer B (5 points)

**Correct Answer:** B) `df.melt(id_vars=['id'])`

**Explanation:**

`pandas.melt()` converts wide to long format:

```python
df_long = df.melt(id_vars=['id'],
                  var_name='year',
                  value_name='y')
```

Result:
```
   id    year   y
0   1  y_2020  10
1   2  y_2020  20
2   1  y_2021  12
3   2  y_2021  22
4   1  y_2022  14
5   2  y_2022  24
```

**Why other answers are wrong:**
- A: `pivot()` goes from long to wide (opposite direction)
- C: Not a pandas method in this context
- D: `transpose()` flips rows/columns but doesn't restructure properly

**Key Concept:** Use `melt()` for wide→long, `pivot()` for long→wide.

---

### Question 12: Answer C (5 points)

**Correct Answer:** C) Firms with poor performance are more likely to drop out of the panel

**Explanation:**

This creates **non-random attrition** or **selective attrition**:
- Missingness is correlated with the outcome ($Y_{it}$)
- Remaining sample is not representative
- Estimates will be biased upward (survival bias)

Example: Studying firm profitability when unprofitable firms exit overstates average profitability.

**Why other answers are less problematic:**
- A: Random missingness (MAR) can often be handled with appropriate methods
- B: Survey non-response unrelated to outcome is less biasing
- D: Non-random attrition is much more problematic than random missingness

**Key Concept:** Attrition related to outcomes violates MAR and biases estimates.

---

### Question 13: Answer C (5 points)

**Correct Answer:** C) Entity 69, Period 2

**Calculation:**

In sorted panel data (entity, then time):
- Entity 1: Observations 1-5 (periods 1-5)
- Entity 2: Observations 6-10 (periods 1-5)
- ...
- Entity $i$: Observations $(i-1) \times T + 1$ to $i \times T$

For observation 347:
- Entity number: $i = \lceil 347/5 \rceil = \lceil 69.4 \rceil = 70$

Wait, let me recalculate:
- Observation 347: $(347 - 1) \div 5 = 346 \div 5 = 69.2$
- Entity: $\lfloor 346/5 \rfloor + 1 = 69 + 1 = 70$
- Period within entity: $(347 - 1) \mod 5 + 1 = 346 \mod 5 + 1 = 1 + 1 = 2$

Actually:
- Entity 69: observations 341-345
- Entity 70: observations 346-350
- Observation 347 is the 2nd observation for entity 70

Let me reconsider: $347 = (i-1) \times 5 + t$
- If $i = 70$: $(70-1) \times 5 + t = 345 + t = 347$ → $t = 2$ ✓

Hmm, the answer key should be B, not C. Let me verify once more:
- Entity 69: obs 341 (t=1), 342 (t=2), 343 (t=3), 344 (t=4), 345 (t=5)
- Entity 70: obs 346 (t=1), 347 (t=2), ...

**Corrected: Answer is B) Entity 70, Period 2**

**Key Concept:** In sorted panel data, observation number = $(i-1)T + t$.

---

### Question 14: Answer B (5 points)

**Correct Answer:** B) COVID-19 closures; use pandemic indicators or restrict analysis period

**Explanation:**

The temporal pattern (April-June 2020) strongly suggests COVID-19 related closures, which:
- Are non-random but explainable by observable event
- Affect many stores simultaneously
- Should be modeled explicitly (pandemic dummies, period indicators) rather than ignored

Dropping stores would lose power unnecessarily. Imputation would ignore real economic shutdown.

**Why other answers are problematic:**
- A: Dropping 10% of sample wastes data
- C: These are true zeros/closures, not measurement error
- D: This is clearly non-random missingness that needs addressing

**Key Concept:** Context-specific missingness should be modeled with appropriate controls.

---

### Question 15: Answer A (5 points)

**Correct Answer:** A) $[-2, 0, 2]$

**Calculation:**

1. Calculate firm mean: $\bar{y}_A = (5 + 7 + 9)/3 = 21/3 = 7$

2. Demean each observation:
   - Period 1: $\tilde{y}_{A,1} = 5 - 7 = -2$
   - Period 2: $\tilde{y}_{A,2} = 7 - 7 = 0$
   - Period 3: $\tilde{y}_{A,3} = 9 - 7 = 2$

**Properties:**
- Demeaned values sum to zero: $-2 + 0 + 2 = 0$ ✓
- Mean of demeaned values is zero

**Why other answers are wrong:**
- B: Subtracted wrong value (subtracted minimum instead of mean)
- C: These are the original values, not demeaned
- D: Incorrect calculation

**Key Concept:** Demeaning removes entity-specific means, leaving within-entity variation.

---

### Question 16: Calculation Problem (15 points)

**Part A: Entity Means (5 points)**

Calculate $\bar{Y}_i$ for each firm:

- Firm 1: $\bar{Y}_1 = (100 + 120)/2 = 110$
- Firm 2: $\bar{Y}_2 = (200 + 220)/2 = 210$
- Firm 3: $\bar{Y}_3 = (150 + 170)/2 = 160$

**Scoring:**
- 2 points for correct method
- 1 point for each correct firm mean (3 points total)

---

**Part B: Within Transformation (5 points)**

Calculate $\tilde{Y}_{it} = Y_{it} - \bar{Y}_i$:

| Firm | Year | $Y_{it}$ | $\bar{Y}_i$ | $\tilde{Y}_{it}$ |
|------|------|----------|------------|-----------------|
| 1    | 1    | 100      | 110        | -10             |
| 1    | 2    | 120      | 110        | 10              |
| 2    | 1    | 200      | 210        | -10             |
| 2    | 2    | 220      | 210        | 10              |
| 3    | 1    | 150      | 160        | -10             |
| 3    | 2    | 170      | 160        | 10              |

**Pattern:** Each firm deviates by ±10 from its mean.

**Scoring:**
- 2 points for correct transformation formula
- 0.5 points for each correct transformed value (3 points total)

---

**Part C: Between Regression Setup (5 points)**

For between regression:

**Number of observations:** $N = 3$ (one per firm)

**Firm-level advertising means:**
- Firm 1: $\bar{X}_1 = (10 + 12)/2 = 11$
- Firm 2: $\bar{X}_2 = (15 + 18)/2 = 16.5$
- Firm 3: $\bar{X}_3 = (8 + 11)/2 = 9.5$

**Between regression dataset:**
| Firm | $\bar{Y}_i$ | $\bar{X}_i$ |
|------|-----------|-----------|
| 1    | 110       | 11        |
| 2    | 210       | 16.5      |
| 3    | 160       | 9.5       |

**Scoring:**
- 2 points for correct number of observations
- 1 point for each correct $\bar{X}_i$ (3 points total)

**Key Concept:** Between estimator collapses panel to cross-section of entity means.

---

## Scoring Rubric

| Points | Grade | Interpretation |
|--------|-------|----------------|
| 90-100 | A     | Excellent understanding |
| 80-89  | B     | Good understanding |
| 70-79  | C     | Satisfactory understanding |
| 60-69  | D     | Needs improvement |
| 0-59   | F     | Significant gaps |

## Learning Objectives Assessed

- **LO1:** Understand pooled OLS and its limitations (Q1-Q5)
- **LO2:** Decompose variation into between and within components (Q6-Q10)
- **LO3:** Manipulate panel data structures (Q11-Q15)
- **LO4:** Apply between/within calculations (Q16)

## Study Resources

If you scored below 70%, review:
- Module 1 Guide: Pooled OLS
- Module 1 Guide: Between-Within Decomposition
- Module 1 Notebook: Data Preparation

**Time to Review:** 3-4 hours

## Common Mistakes

1. **Confusing degrees of freedom** in pooled vs panel contexts
2. **Mixing up between and within variation** concepts
3. **Incorrect demeaning calculations** (using wrong entity mean)
4. **Panel data indexing errors** in sorted datasets
5. **Misunderstanding balance** requirements
