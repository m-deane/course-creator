# Diagnostic Quiz: Prerequisites Assessment

## Instructions

- Answer all 30 questions
- Time limit: 45 minutes
- Passing score: 70% (21/30)
- This quiz is for self-assessment only (not graded)
- Score below 70% indicates prerequisite review is recommended

---

## Section 1: Probability Fundamentals (10 questions)

### Question 1
If X ~ N(0, 1), what is the approximate value of P(X > 1.96)?

A) 0.025
B) 0.05
C) 0.975
D) 0.95

---

### Question 2
For a continuous random variable with PDF f(x), which statement is TRUE?

A) f(x) represents P(X = x)
B) f(x) must satisfy 0 ≤ f(x) ≤ 1 for all x
C) ∫f(x)dx over the entire domain equals 1
D) f(x) cannot exceed 1

---

### Question 3
If P(A) = 0.3 and P(B|A) = 0.6, what is P(A ∩ B)?

A) 0.18
B) 0.30
C) 0.50
D) 0.90

---

### Question 4
The Beta(1, 1) distribution is equivalent to which distribution?

A) Normal(0, 1)
B) Uniform(0, 1)
C) Exponential(1)
D) Gamma(1, 1)

---

### Question 5
If X and Y are independent, which is TRUE?

A) Cov(X, Y) = 1
B) P(X|Y) = P(X)
C) Var(X + Y) = Var(X) + Var(Y) - 2Cov(X,Y)
D) E[XY] = E[X] + E[Y]

---

### Question 6
Which distribution would you use to model a proportion (value between 0 and 1)?

A) Normal
B) Gamma
C) Beta
D) Poisson

---

### Question 7
For X ~ Gamma(α, β), what is E[X]?

A) α + β
B) α × β
C) α / β
D) β / α

---

### Question 8
What does the Law of Total Probability state for events B₁, B₂, ... that partition the sample space?

A) P(A) = ΣP(A)P(Bᵢ)
B) P(A) = ΣP(A|Bᵢ)P(Bᵢ)
C) P(A) = ΣP(Bᵢ|A)P(A)
D) P(A) = P(A|B₁) + P(A|B₂) + ...

---

### Question 9
If Var(X) = 4, what is Var(3X + 5)?

A) 12
B) 17
C) 36
D) 41

---

### Question 10
Which distribution has "fatter tails" than the Normal distribution, making it useful for modeling returns with outliers?

A) Uniform
B) Beta
C) Student-t
D) Exponential

---

## Section 2: Linear Algebra (5 questions)

### Question 11
For a 3×3 matrix A, what does det(A) = 0 imply?

A) A has an inverse
B) A is the identity matrix
C) A is singular (not invertible)
D) A is positive definite

---

### Question 12
If A is a 3×2 matrix and B is a 2×4 matrix, what is the dimension of AB?

A) 2×3
B) 3×4
C) 4×3
D) Not defined

---

### Question 13
What is the trace of a 3×3 identity matrix?

A) 0
B) 1
C) 3
D) 9

---

### Question 14
For symmetric matrix A, which statement is TRUE about its eigenvalues?

A) They are all complex
B) They are all real
C) They sum to zero
D) They are all positive

---

### Question 15
What does it mean for a matrix to be positive definite?

A) All elements are positive
B) The determinant is positive
C) xᵀAx > 0 for all non-zero vectors x
D) All eigenvalues are negative

---

## Section 3: Calculus (5 questions)

### Question 16
What is d/dx [log(x²)]?

A) 2/x
B) 1/x²
C) 2x
D) 1/(2x)

---

### Question 17
If f(x) = e^(-x²/2), what is f'(x)?

A) -xe^(-x²/2)
B) xe^(-x²/2)
C) -x²e^(-x²/2)
D) e^(-x²/2)/x

---

### Question 18
∫₀^∞ λe^(-λx) dx equals:

A) 0
B) 1
C) λ
D) 1/λ

---

### Question 19
For f(x, y) = x²y + xy², what is ∂f/∂x?

A) 2xy + y²
B) x² + 2xy
C) 2x + y
D) x² + y²

---

### Question 20
The gradient of f(x, y) = x² + y² at point (1, 2) is:

A) (1, 2)
B) (2, 4)
C) (2, 2)
D) (1, 4)

---

## Section 4: Statistics (5 questions)

### Question 21
Which estimator property means that as sample size increases, the estimator converges to the true parameter?

A) Unbiasedness
B) Efficiency
C) Consistency
D) Sufficiency

---

### Question 22
In Bayesian inference, what does the posterior represent?

A) P(data | parameter)
B) P(parameter | data)
C) P(data)
D) P(parameter)

---

### Question 23
What is the MLE for the mean μ of a Normal distribution given observations x₁, ..., xₙ?

A) Median of xᵢ
B) Mode of xᵢ
C) Sample mean x̄
D) Geometric mean of xᵢ

---

### Question 24
A 95% confidence interval means:

A) There's a 95% probability the true parameter is in this interval
B) 95% of the data falls within this interval
C) If we repeated sampling, 95% of such intervals would contain the true parameter
D) The parameter is within 95% of the estimate

---

### Question 25
Which describes a conjugate prior?

A) A prior that gives a posterior in the same family as the prior
B) A prior that is the inverse of the likelihood
C) A prior that is always Normal
D) A prior based on previous data

---

## Section 5: Python/Programming (5 questions)

### Question 26
What does `np.random.randn(100, 3)` return?

A) 100 random numbers from 0 to 3
B) A 100×3 array of standard normal samples
C) 3 random numbers repeated 100 times
D) A 3×100 array of uniform samples

---

### Question 27
Given a pandas DataFrame df with columns ['date', 'price', 'volume'], how do you calculate the mean price?

A) df.price.average()
B) df['price'].mean()
C) mean(df.price)
D) df.mean('price')

---

### Question 28
What does `df.groupby('category')['value'].sum()` do?

A) Filters df to the 'category' column
B) Sums all values in the 'value' column
C) Calculates sum of 'value' for each unique 'category'
D) Creates a new column called 'sum'

---

### Question 29
In NumPy, what does `A @ B` compute (for matrices A and B)?

A) Element-wise multiplication
B) Matrix multiplication
C) Concatenation
D) Tensor product

---

### Question 30
How do you select rows where column 'price' > 100 in pandas?

A) df.select(price > 100)
B) df[df['price'] > 100]
C) df.where('price', '>', 100)
D) df.filter(price > 100)

---

## Answer Key

| Q | Answer | Q | Answer | Q | Answer |
|---|--------|---|--------|---|--------|
| 1 | A | 11 | C | 21 | C |
| 2 | C | 12 | B | 22 | B |
| 3 | A | 13 | C | 23 | C |
| 4 | B | 14 | B | 24 | C |
| 5 | B | 15 | C | 25 | A |
| 6 | C | 16 | A | 26 | B |
| 7 | C | 17 | A | 27 | B |
| 8 | B | 18 | B | 28 | C |
| 9 | C | 19 | A | 29 | B |
| 10 | C | 20 | B | 30 | B |

---

## Score Interpretation

**27-30 (90-100%):** Excellent foundation - proceed directly to Module 1

**24-26 (80-89%):** Good foundation - proceed with occasional reference to review materials

**21-23 (70-79%):** Adequate foundation - may need to review specific weak areas

**18-20 (60-69%):** Some gaps - recommended review of probability and statistics fundamentals

**Below 18 (<60%):** Significant gaps - complete prerequisite review before proceeding

---

## Section-Specific Recommendations

### If you struggled with Probability (Section 1):
- Read Chapters 1-2 of McElreath's "Statistical Rethinking"
- Complete Khan Academy's probability course

### If you struggled with Linear Algebra (Section 2):
- Review matrix operations and eigenvalue decomposition
- Complete 3Blue1Brown's "Essence of Linear Algebra" series

### If you struggled with Calculus (Section 3):
- Review derivatives and integrals, especially chain rule
- Focus on partial derivatives for multivariate functions

### If you struggled with Statistics (Section 4):
- Review MLE and the concept of estimator properties
- Read introduction to Bayesian inference in McElreath Ch. 1-2

### If you struggled with Python (Section 5):
- Complete a NumPy/Pandas tutorial
- Practice with Kaggle's Python course (free)
