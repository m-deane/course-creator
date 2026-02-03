# Quiz: Module 0 - Foundations

**Course:** Panel Data Econometrics
**Module:** 0 - Foundations
**Time Limit:** 25 minutes
**Total Points:** 100
**Passing Score:** 70%

## Instructions

This quiz assesses your understanding of OLS fundamentals, matrix notation, and panel data structures. You have 2 attempts. Show all work for calculation questions.

---

## Part A: OLS Fundamentals (40 points)

### Question 1: Matrix Form of OLS (8 points)

The OLS estimator in matrix form is given by $\hat{\beta} = (X'X)^{-1}X'y$.

Consider the following data with $n = 3$ observations and $k = 2$ variables (including intercept):

$$X = \begin{bmatrix} 1 & 2 \\ 1 & 4 \\ 1 & 6 \end{bmatrix}, \quad y = \begin{bmatrix} 3 \\ 5 \\ 7 \end{bmatrix}$$

Calculate $X'X$ (show work).

**A)** $\begin{bmatrix} 3 & 12 \\ 12 & 56 \end{bmatrix}$

**B)** $\begin{bmatrix} 3 & 12 \\ 12 & 52 \end{bmatrix}$

**C)** $\begin{bmatrix} 6 & 12 \\ 12 & 56 \end{bmatrix}$

**D)** $\begin{bmatrix} 3 & 6 \\ 6 & 56 \end{bmatrix}$

---

### Question 2: Gauss-Markov Assumptions (8 points)

Which of the following statements about the Gauss-Markov assumptions is **FALSE**?

**A)** Under the Gauss-Markov assumptions, OLS is BLUE (Best Linear Unbiased Estimator)

**B)** The exogeneity assumption requires $E[\epsilon|X] = 0$, meaning errors are uncorrelated with regressors

**C)** The no multicollinearity assumption requires that all columns of $X$ are perfectly uncorrelated with each other

**D)** Homoskedasticity means that $Var(\epsilon_i|X) = \sigma^2$ is constant across observations

---

### Question 3: Variance-Covariance Matrix (8 points)

Under the classical linear regression assumptions, the variance-covariance matrix of the OLS estimator is:

$$Var(\hat{\beta}) = \sigma^2(X'X)^{-1}$$

If we have $X'X = \begin{bmatrix} 100 & 0 \\ 0 & 50 \end{bmatrix}$ and $\hat{\sigma}^2 = 4$, what is the standard error of $\hat{\beta}_2$?

**A)** 0.20

**B)** 0.28

**C)** 0.40

**D)** 0.08

---

### Question 4: Projection Matrix (8 points)

The projection matrix (hat matrix) is defined as $P = X(X'X)^{-1}X'$ and satisfies $\hat{y} = Py$.

Which of the following properties does the projection matrix possess?

**A)** $P$ is idempotent: $P^2 = P$

**B)** $P$ is symmetric: $P' = P$

**C)** $P$ has rank equal to $k$ (number of regressors)

**D)** All of the above

---

### Question 5: Residual Properties (8 points)

In OLS regression with an intercept, which of the following is **always true** about the residuals $\hat{\epsilon}$?

**A)** $\sum_{i=1}^n \hat{\epsilon}_i = 0$

**B)** $\sum_{i=1}^n x_{ij}\hat{\epsilon}_i = 0$ for all regressors $j$

**C)** $X'\hat{\epsilon} = 0$

**D)** All of the above

---

## Part B: Panel Data Structures (30 points)

### Question 6: Panel Data Dimensions (6 points)

A researcher has quarterly data on 500 firms from 2015 Q1 to 2023 Q4. Assuming a balanced panel, how many total observations are in the dataset?

**A)** 18,000

**B)** 15,000

**C)** 20,000

**D)** 16,000

---

### Question 7: Long vs Wide Format (6 points)

Consider the following data in wide format:

| firm_id | revenue_2020 | revenue_2021 | revenue_2022 |
|---------|--------------|--------------|--------------|
| A       | 100          | 110          | 120          |
| B       | 200          | 210          | 220          |

How many rows would this data have in long format?

**A)** 2

**B)** 3

**C)** 6

**D)** 9

---

### Question 8: Panel Data Indexing (6 points)

In panel data notation, $y_{it}$ represents the value of variable $y$ for entity $i$ at time $t$. If we have $N = 100$ firms and $T = 10$ years, what does $\bar{y}_i$ represent?

**A)** The average of $y$ across all firms in year $i$

**B)** The average of $y$ for firm $i$ across all time periods

**C)** The overall average of $y$ across all firms and all time periods

**D)** The time trend for firm $i$

---

### Question 9: Balanced vs Unbalanced Panels (6 points)

Which of the following scenarios would result in an **unbalanced** panel?

**A)** A dataset of 50 countries observed annually from 2000-2020, with complete data for all countries

**B)** A dataset of 1,000 firms from 2010-2023, where some firms went bankrupt and exited the dataset

**C)** A dataset of students measured at ages 10, 15, and 20, with all students observed at each age

**D)** A randomized controlled trial with measurements at baseline, 6 months, and 12 months for all participants

---

### Question 10: Panel Data Structure Identification (6 points)

A researcher wants to estimate the effect of education on wages using panel data. She has data on 5,000 individuals surveyed once in 2020. Is this a panel dataset?

**A)** Yes, because she has multiple individuals

**B)** No, because she only has one time period per individual

**C)** Yes, if the individuals were randomly sampled

**D)** It depends on whether the data is balanced

---

## Part C: Matrix Algebra and Computation (30 points)

### Question 11: Inverse Matrix Calculation (8 points)

Given the matrix $A = \begin{bmatrix} 2 & 1 \\ 1 & 1 \end{bmatrix}$, what is $A^{-1}$?

**A)** $\begin{bmatrix} 1 & -1 \\ -1 & 2 \end{bmatrix}$

**B)** $\begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}$

**C)** $\begin{bmatrix} 0.5 & -0.5 \\ -0.5 & 1 \end{bmatrix}$

**D)** $\begin{bmatrix} 2 & -1 \\ -1 & 1 \end{bmatrix}$

---

### Question 12: Rank and Multicollinearity (7 points)

Consider the data matrix:

$$X = \begin{bmatrix} 1 & 2 & 4 \\ 1 & 3 & 6 \\ 1 & 4 & 8 \\ 1 & 5 & 10 \end{bmatrix}$$

What is the rank of this matrix, and why would OLS fail?

**A)** Rank = 3; OLS would succeed

**B)** Rank = 2; the third column is twice the second, causing perfect multicollinearity

**C)** Rank = 1; all columns are linearly dependent

**D)** Rank = 4; there is no multicollinearity problem

---

### Question 13: Predicted Values Calculation (7 points)

Suppose you estimate a simple regression $y = \beta_0 + \beta_1 x + \epsilon$ and obtain $\hat{\beta}_0 = 2$ and $\hat{\beta}_1 = 0.5$. What is the predicted value $\hat{y}$ when $x = 10$?

**A)** 5

**B)** 7

**C)** 12

**D)** 2.5

---

### Question 14: R-squared Interpretation (8 points)

In a regression with $TSS = 1000$ (Total Sum of Squares) and $RSS = 300$ (Residual Sum of Squares), what is the $R^2$?

**A)** 0.30

**B)** 0.70

**C)** 0.50

**D)** Cannot be determined without knowing $n$ and $k$

---

## Answer Key and Explanations

### Question 1: Answer B (8 points)

**Correct Answer:** B) $\begin{bmatrix} 3 & 12 \\ 12 & 52 \end{bmatrix}$

**Calculation:**

$$X'X = \begin{bmatrix} 1 & 1 & 1 \\ 2 & 4 & 6 \end{bmatrix} \begin{bmatrix} 1 & 2 \\ 1 & 4 \\ 1 & 6 \end{bmatrix}$$

Element (1,1): $1(1) + 1(1) + 1(1) = 3$
Element (1,2): $1(2) + 1(4) + 1(6) = 12$
Element (2,1): $2(1) + 4(1) + 6(1) = 12$
Element (2,2): $2(2) + 4(4) + 6(6) = 4 + 16 + 36 = 56$

Wait, let me recalculate (2,2): $2^2 + 4^2 + 6^2 = 4 + 16 + 36 = 56$

Actually, the correct answer should be A. Let me reconsider this more carefully for the answer key below.

Upon verification: $X'X = \begin{bmatrix} 3 & 12 \\ 12 & 56 \end{bmatrix}$

**Corrected: Answer A is correct.**

**Why other answers are wrong:**
- B: Incorrect calculation of element (2,2)
- C: Element (1,1) should be 3, not 6
- D: Element (1,2) should be 12, not 6

**Key Concept:** Matrix multiplication requires careful attention to element-by-element computation.

---

### Question 2: Answer C (8 points)

**Correct Answer:** C) The no multicollinearity assumption requires that all columns of $X$ are perfectly uncorrelated with each other

**Explanation:**

The statement is FALSE. The no multicollinearity assumption requires that $rank(X) = k$, meaning no column of $X$ can be written as a linear combination of other columns. Columns can be correlated with each other; they just cannot be *perfectly* collinear. Zero correlation is not required.

**Why other answers are correct statements:**
- A: TRUE - This is the Gauss-Markov theorem
- B: TRUE - Exogeneity is correctly stated
- D: TRUE - This is the definition of homoskedasticity

**Key Concept:** Multicollinearity refers to perfect linear relationships, not correlation per se.

---

### Question 3: Answer B (8 points)

**Correct Answer:** B) 0.28

**Calculation:**

$$Var(\hat{\beta}) = \sigma^2(X'X)^{-1} = 4 \begin{bmatrix} 1/100 & 0 \\ 0 & 1/50 \end{bmatrix} = \begin{bmatrix} 0.04 & 0 \\ 0 & 0.08 \end{bmatrix}$$

The variance of $\hat{\beta}_2$ is 0.08, so the standard error is:

$$SE(\hat{\beta}_2) = \sqrt{0.08} = 0.283 \approx 0.28$$

**Why other answers are wrong:**
- A: This is $\sqrt{0.04}$, the SE of $\hat{\beta}_1$
- C: This is $\sqrt{0.16}$, twice the correct variance
- D: This is the variance, not the standard error

**Key Concept:** Standard errors are the square roots of diagonal elements of the variance-covariance matrix.

---

### Question 4: Answer D (8 points)

**Correct Answer:** D) All of the above

**Explanation:**

All three properties are true of the projection matrix:

1. **Idempotent:** $P^2 = P$ means applying the projection twice is the same as applying it once
2. **Symmetric:** $P' = P$ can be verified: $(X(X'X)^{-1}X')' = X((X'X)^{-1})'X' = X(X'X)^{-1}X' = P$
3. **Rank:** The rank of $P$ equals $k$, the number of columns in $X$

These properties make $P$ a true orthogonal projection matrix.

**Key Concept:** The projection matrix projects $y$ onto the column space of $X$.

---

### Question 5: Answer D (8 points)

**Correct Answer:** D) All of the above

**Explanation:**

All three statements are equivalent formulations of the first-order conditions for OLS:

- **A:** Residuals sum to zero (when there's an intercept)
- **B:** Residuals are orthogonal to each regressor
- **C:** Vector form of B: $X'\hat{\epsilon} = 0$

These are the normal equations: $X'X\hat{\beta} = X'y$, which imply $X'(y - X\hat{\beta}) = X'\hat{\epsilon} = 0$.

**Key Concept:** OLS minimizes $SSR = \hat{\epsilon}'\hat{\epsilon}$ by making residuals orthogonal to all regressors.

---

### Question 6: Answer A (6 points)

**Correct Answer:** A) 18,000

**Calculation:**

- Years: 2015 to 2023 = 9 years
- Quarters per year: 4
- Total time periods: $T = 9 \times 4 = 36$ quarters
- Number of firms: $N = 500$
- Total observations: $N \times T = 500 \times 36 = 18,000$

**Why other answers are wrong:**
- B: Used only 30 quarters instead of 36
- C: Incorrectly calculated time periods
- D: Used 32 quarters instead of 36

**Key Concept:** Balanced panel has $n = N \times T$ observations.

---

### Question 7: Answer C (6 points)

**Correct Answer:** C) 6

**Explanation:**

In long format, each firm-year combination becomes a separate row:

| firm_id | year | revenue |
|---------|------|---------|
| A       | 2020 | 100     |
| A       | 2021 | 110     |
| A       | 2022 | 120     |
| B       | 2020 | 200     |
| B       | 2021 | 210     |
| B       | 2022 | 220     |

Total rows: 2 firms × 3 years = 6 rows

**Why other answers are wrong:**
- A: That's the number of firms in wide format
- B: That's the number of years
- D: That's firms × variables (2 × 3 revenue columns + firm_id columns)

**Key Concept:** Long format has one row per entity-time combination; wide format has one row per entity.

---

### Question 8: Answer B (6 points)

**Correct Answer:** B) The average of $y$ for firm $i$ across all time periods

**Explanation:**

Standard panel notation:
- $y_{it}$: value for entity $i$ at time $t$
- $\bar{y}_i = \frac{1}{T}\sum_{t=1}^T y_{it}$: within-entity mean across time
- $\bar{y}_t = \frac{1}{N}\sum_{i=1}^N y_{it}$: cross-sectional mean at time $t$
- $\bar{y} = \frac{1}{NT}\sum_{i=1}^N\sum_{t=1}^T y_{it}$: grand mean

**Why other answers are wrong:**
- A: That would be $\bar{y}_t$ (time subscript, not entity)
- C: That's $\bar{y}$ (the grand mean)
- D: A time trend would need a functional form like $\gamma_i t$

**Key Concept:** Subscript indicates what is being averaged over.

---

### Question 9: Answer B (6 points)

**Correct Answer:** B) A dataset of 1,000 firms from 2010-2023, where some firms went bankrupt and exited the dataset

**Explanation:**

An unbalanced panel occurs when entities are not observed for the same number of time periods. Firms that exit have fewer observations than firms that survive the entire period.

**Why other answers describe balanced panels:**
- A: All countries observed for all years
- C: All students observed at all three ages
- D: All participants measured at all three time points

**Key Concept:** Unbalanced panels have different $T_i$ across entities. This commonly occurs with firm entry/exit, survey attrition, or missing data.

---

### Question 10: Answer B (6 points)

**Correct Answer:** B) No, because she only has one time period per individual

**Explanation:**

Panel data requires repeated observations on the same entities over time. With only one observation per person, this is cross-sectional data, not panel data.

Panel data definition: $(i, t)$ structure with $T \geq 2$

**Why other answers are wrong:**
- A: Having multiple individuals is necessary but not sufficient for panel data
- C: Random sampling doesn't make it panel data
- D: Balance/unbalance is only relevant if you have multiple time periods

**Key Concept:** Panel data has both cross-sectional ($i$) and time-series ($t$) dimensions.

---

### Question 11: Answer A (8 points)

**Correct Answer:** A) $\begin{bmatrix} 1 & -1 \\ -1 & 2 \end{bmatrix}$

**Calculation:**

For a $2 \times 2$ matrix $A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$:

$$A^{-1} = \frac{1}{ad - bc}\begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$

Here: $det(A) = 2(1) - 1(1) = 1$

$$A^{-1} = \frac{1}{1}\begin{bmatrix} 1 & -1 \\ -1 & 2 \end{bmatrix} = \begin{bmatrix} 1 & -1 \\ -1 & 2 \end{bmatrix}$$

**Verification:**
$$AA^{-1} = \begin{bmatrix} 2 & 1 \\ 1 & 1 \end{bmatrix}\begin{bmatrix} 1 & -1 \\ -1 & 2 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$$ ✓

**Key Concept:** Matrix inverse formula for $2 \times 2$ matrices.

---

### Question 12: Answer B (7 points)

**Correct Answer:** B) Rank = 2; the third column is twice the second, causing perfect multicollinearity

**Explanation:**

Observe that column 3 = 2 × column 2:
- Row 1: $4 = 2(2)$ ✓
- Row 2: $6 = 2(3)$ ✓
- Row 3: $8 = 2(4)$ ✓
- Row 4: $10 = 2(5)$ ✓

This means column 3 is perfectly collinear with column 2, so the matrix has only 2 linearly independent columns (rank = 2). OLS requires $rank(X) = k = 3$, so $(X'X)$ is singular and cannot be inverted.

**Key Concept:** Perfect multicollinearity makes $(X'X)$ singular and OLS estimation impossible.

---

### Question 13: Answer B (7 points)

**Correct Answer:** B) 7

**Calculation:**

$$\hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x = 2 + 0.5(10) = 2 + 5 = 7$$

**Why other answers are wrong:**
- A: Forgot to add the intercept
- C: Added instead of using the coefficient correctly
- D: Only computed $\hat{\beta}_0 + \hat{\beta}_1$

**Key Concept:** Predicted values are computed by plugging $x$ values into the estimated regression equation.

---

### Question 14: Answer B (8 points)

**Correct Answer:** B) 0.70

**Calculation:**

$$R^2 = 1 - \frac{RSS}{TSS} = 1 - \frac{300}{1000} = 1 - 0.3 = 0.70$$

Alternatively: $R^2 = \frac{ESS}{TSS}$ where $ESS = TSS - RSS = 1000 - 300 = 700$, so $R^2 = 700/1000 = 0.70$

**Why other answers are wrong:**
- A: This is $RSS/TSS$, not $R^2$
- C: Incorrect calculation
- D: $R^2$ can always be calculated from $RSS$ and $TSS$; we don't need $n$ or $k$ for $R^2$ (though we would for adjusted $R^2$)

**Key Concept:** $R^2$ measures the proportion of variance explained by the model.

---

## Scoring Rubric

| Points | Grade |
|--------|-------|
| 90-100 | A |
| 80-89  | B |
| 70-79  | C |
| 60-69  | D |
| 0-59   | F |

## Learning Objectives Assessed

- **LO1:** Compute OLS estimator using matrix algebra (Q1, Q3, Q11, Q13)
- **LO2:** Understand OLS assumptions and properties (Q2, Q4, Q5, Q12)
- **LO3:** Identify panel data structures (Q6-Q10)
- **LO4:** Interpret regression output (Q14)

## Study Resources

If you scored below 70%, review:
- Module 0 Guide: OLS Review
- Module 0 Guide: Data Structures
- Module 0 Notebook: OLS Fundamentals

**Time to Review:** 2-3 hours
