# Mathematical Notation Guide

## Purpose

This guide provides a comprehensive reference for mathematical notation used throughout the Bayesian Commodity Forecasting course. Refer back to this document whenever you encounter unfamiliar symbols.

---

## 1. Probability Notation

### Basic Probability

| Notation | Meaning | Example |
|----------|---------|---------|
| $P(A)$ | Probability of event A | $P(\text{price increases}) = 0.6$ |
| $P(A \cap B)$ | Joint probability of A and B | $P(\text{price up AND vol high})$ |
| $P(A \cup B)$ | Probability of A or B | $P(\text{price up OR vol high})$ |
| $P(A \| B)$ | Conditional probability | $P(\text{up} \| \text{high vol})$ |
| $P(A^c)$ or $P(\neg A)$ | Probability of not A | $P(\text{not up}) = 1 - P(\text{up})$ |

### Distributions

| Notation | Meaning | Read as |
|----------|---------|---------|
| $X \sim \mathcal{N}(\mu, \sigma^2)$ | X follows Normal distribution | "X is distributed as Normal" |
| $p(x)$ | Probability density function | "density of x" |
| $P(X = x)$ | Probability mass function | "probability X equals x" (discrete) |
| $F(x) = P(X \leq x)$ | Cumulative distribution function | "CDF of X at x" |
| $\mathbb{E}[X]$ | Expectation of X | "expected value of X" |
| $\text{Var}(X)$ | Variance of X | "variance of X" |
| $\text{Cov}(X,Y)$ | Covariance of X and Y | "covariance of X and Y" |

### Common Distributions

| Distribution | Notation | Parameters |
|--------------|----------|------------|
| **Normal** | $\mathcal{N}(\mu, \sigma^2)$ | $\mu$ = mean, $\sigma^2$ = variance |
| **Multivariate Normal** | $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ | $\boldsymbol{\mu}$ = mean vector, $\boldsymbol{\Sigma}$ = covariance matrix |
| **Beta** | $\text{Beta}(\alpha, \beta)$ | $\alpha, \beta$ = shape parameters |
| **Gamma** | $\text{Gamma}(\alpha, \beta)$ | $\alpha$ = shape, $\beta$ = rate |
| **Exponential** | $\text{Exp}(\lambda)$ | $\lambda$ = rate parameter |
| **Poisson** | $\text{Poisson}(\lambda)$ | $\lambda$ = rate parameter |
| **Binomial** | $\text{Bin}(n, p)$ | $n$ = trials, $p$ = success probability |
| **Student's t** | $t(\nu)$ | $\nu$ = degrees of freedom |

---

## 2. Bayesian Inference Notation

### Core Concepts

| Notation | Meaning | Explanation |
|----------|---------|-------------|
| $\theta$ | Parameter(s) | Unknown quantity we want to infer |
| $y$ | Data | Observed values |
| $p(\theta)$ | Prior distribution | Belief about $\theta$ before seeing data |
| $p(y \| \theta)$ | Likelihood | Probability of data given parameter |
| $p(\theta \| y)$ | Posterior distribution | Updated belief about $\theta$ after seeing data |
| $p(y)$ | Marginal likelihood | $\int p(y\|\theta)p(\theta)d\theta$ (evidence) |

### Bayes' Theorem

$$p(\theta | y) = \frac{p(y | \theta) p(\theta)}{p(y)} \propto p(y | \theta) p(\theta)$$

**Read as:** "Posterior is proportional to likelihood times prior"

### Predictive Distributions

| Notation | Meaning |
|----------|---------|
| $p(y_{new} \| y)$ | Posterior predictive distribution |
| $p(y)$ | Prior predictive distribution |

**Formula:**
$$p(y_{new} | y) = \int p(y_{new} | \theta) p(\theta | y) d\theta$$

---

## 3. Time Series Notation

### Indexing

| Notation | Meaning | Example |
|----------|---------|---------|
| $y_t$ | Observation at time t | $y_1, y_2, \ldots, y_T$ |
| $y_{1:t}$ | All observations from 1 to t | $\\{y_1, y_2, \ldots, y_t\\}$ |
| $y_{t-k}$ | Observation k periods ago | $y_{t-1}$ is previous period (lag 1) |
| $\Delta y_t$ | First difference | $y_t - y_{t-1}$ |
| $\nabla y_t$ | Alternative difference notation | Same as $\Delta y_t$ |

### State Space Models

| Notation | Meaning |
|----------|---------|
| $\alpha_t$ | Hidden state at time t |
| $\alpha_{t\|t}$ | Filtered state (using data up to t) |
| $\alpha_{t\|t-1}$ | Predicted state (using data up to t-1) |
| $\alpha_{t\|T}$ | Smoothed state (using all data 1 to T) |
| $P_t$ | State covariance matrix |
| $K_t$ | Kalman gain |

### Model Components

| Notation | Meaning | Equation |
|----------|---------|----------|
| $\epsilon_t$ | Observation noise | $y_t = \mu_t + \epsilon_t$ |
| $\eta_t$ | State innovation | $\alpha_t = \alpha_{t-1} + \eta_t$ |
| $\sigma^2$ | Variance | $\text{Var}(\epsilon_t) = \sigma^2$ |

---

## 4. Regression Notation

### Linear Regression

**Scalar form:**
$$y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \ldots + \beta_p x_{ip} + \epsilon_i$$

**Matrix form:**
$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$$

| Notation | Meaning | Dimension |
|----------|---------|-----------|
| $\mathbf{y}$ | Response vector | $n \times 1$ |
| $\mathbf{X}$ | Design matrix | $n \times p$ |
| $\boldsymbol{\beta}$ | Coefficient vector | $p \times 1$ |
| $\boldsymbol{\epsilon}$ | Error vector | $n \times 1$ |

### Estimated quantities

| Notation | Meaning |
|----------|---------|
| $\hat{\beta}$ | Estimated coefficient (frequentist) |
| $\bar{\beta}$ | Posterior mean (Bayesian) |
| $\text{SE}(\hat{\beta})$ | Standard error |
| $R^2$ | Coefficient of determination |

---

## 5. Linear Algebra Notation

### Vectors and Matrices

| Notation | Meaning | Example |
|----------|---------|---------|
| $\mathbf{x}$ | Column vector (bold lowercase) | $\mathbf{x} = \begin{pmatrix} x_1 \\\ x_2 \end{pmatrix}$ |
| $\mathbf{x}^T$ | Row vector (transpose) | $\mathbf{x}^T = (x_1, x_2)$ |
| $\mathbf{A}$ | Matrix (bold uppercase) | $\mathbf{A} = \begin{pmatrix} a_{11} & a_{12} \\\ a_{21} & a_{22} \end{pmatrix}$ |
| $\mathbf{I}$ | Identity matrix | Diagonal matrix of 1's |
| $\mathbf{A}^{-1}$ | Matrix inverse | $\mathbf{A}\mathbf{A}^{-1} = \mathbf{I}$ |

### Operations

| Notation | Meaning | Dimension |
|----------|---------|-----------|
| $\mathbf{A}\mathbf{B}$ | Matrix multiplication | $(m \times n)(n \times p) = m \times p$ |
| $\mathbf{A}^T$ | Matrix transpose | $(m \times n)^T = n \times m$ |
| $\|\mathbf{x}\|$ | Vector norm | $\sqrt{x_1^2 + x_2^2 + \ldots}$ |
| $\mathbf{x}^T \mathbf{y}$ | Dot product | $\sum_i x_i y_i$ |
| $\text{tr}(\mathbf{A})$ | Trace (sum of diagonal) | $\sum_i a_{ii}$ |
| $\|\mathbf{A}\|$ | Determinant | Scalar value |

---

## 6. Calculus Notation

### Derivatives

| Notation | Meaning | Read as |
|----------|---------|---------|
| $\frac{dy}{dx}$ | Derivative of y with respect to x | "dy dx" |
| $\frac{\partial f}{\partial x}$ | Partial derivative | "partial f partial x" |
| $\nabla f$ | Gradient vector | $\left(\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n}\right)^T$ |
| $\frac{d^2y}{dx^2}$ | Second derivative | Curvature |

### Integrals

| Notation | Meaning | Read as |
|----------|---------|---------|
| $\int f(x) dx$ | Indefinite integral | "integral of f" |
| $\int_a^b f(x) dx$ | Definite integral from a to b | "integral from a to b" |
| $\int\int f(x,y) dx dy$ | Double integral | "double integral of f" |

---

## 7. Statistical Operators

| Notation | Meaning | Formula |
|----------|---------|---------|
| $\mathbb{E}[X]$ | Expected value | $\int x p(x) dx$ (continuous) |
| $\text{Var}(X)$ | Variance | $\mathbb{E}[(X - \mathbb{E}[X])^2]$ |
| $\text{SD}(X)$ | Standard deviation | $\sqrt{\text{Var}(X)}$ |
| $\text{Cov}(X,Y)$ | Covariance | $\mathbb{E}[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])]$ |
| $\text{Corr}(X,Y)$ | Correlation | $\frac{\text{Cov}(X,Y)}{\text{SD}(X)\text{SD}(Y)}$ |

---

## 8. Special Symbols

### Greek Letters (Common in Statistics)

| Symbol | Name | Typical Use |
|--------|------|-------------|
| $\alpha$ | alpha | Significance level, state variable |
| $\beta$ | beta | Regression coefficient |
| $\gamma$ | gamma | Discount factor |
| $\delta$ | delta | Difference operator |
| $\epsilon$ | epsilon | Error term |
| $\eta$ | eta | State innovation |
| $\theta$ | theta | Generic parameter |
| $\lambda$ | lambda | Rate parameter, eigenvalue |
| $\mu$ | mu | Mean |
| $\nu$ | nu | Degrees of freedom |
| $\rho$ | rho | Correlation, AR coefficient |
| $\sigma$ | sigma | Standard deviation |
| $\Sigma$ | Sigma (capital) | Covariance matrix |
| $\tau$ | tau | Precision (1/variance) |
| $\phi$ | phi | Generic parameter |
| $\psi$ | psi | Parameter |
| $\omega$ | omega | Parameter |

### Set Notation

| Notation | Meaning |
|----------|---------|
| $\in$ | "is an element of" |
| $\subset$ | "is a subset of" |
| $\cup$ | Union (OR) |
| $\cap$ | Intersection (AND) |
| $\emptyset$ | Empty set |

### Logic Symbols

| Notation | Meaning |
|----------|---------|
| $\forall$ | "for all" |
| $\exists$ | "there exists" |
| $\implies$ | "implies" |
| $\iff$ | "if and only if" |
| $\propto$ | "is proportional to" |

---

## 9. Asymptotic Notation

| Notation | Meaning | Example |
|----------|---------|---------|
| $x \to a$ | x approaches a | $\lim_{x \to 0}$ |
| $n \to \infty$ | n approaches infinity | Large sample |
| $\mathcal{O}(n)$ | "Big O" - order of magnitude | Complexity |
| $x \approx y$ | x approximately equals y | $\pi \approx 3.14$ |
| $x \sim y$ | x is asymptotically equivalent to y | As $n \to \infty$ |

---

## 10. Commodity-Specific Notation

### Price and Return Notation

| Notation | Meaning |
|----------|---------|
| $S_t$ | Spot price at time t |
| $F_t$ | Futures price at time t |
| $r_t$ | Log return: $\log(S_t / S_{t-1})$ |
| $R_t$ | Simple return: $(S_t - S_{t-1}) / S_{t-1}$ |
| $\sigma_t$ | Volatility at time t |

### Fundamental Variables

| Notation | Meaning |
|----------|---------|
| $I_t$ | Inventory level at time t |
| $Q_t$ | Production quantity |
| $D_t$ | Demand |
| $y$ | Convenience yield |
| $u$ | Storage cost |

---

## 11. Coding Equivalents

### Python/NumPy Translations

| Math | Python | Example |
|------|--------|---------|
| $\mu$ | `mu` | `mu = 0.5` |
| $\Sigma$ | `Sigma` | `Sigma = np.eye(3)` |
| $\mathbf{x}^T \mathbf{y}$ | `x.T @ y` or `np.dot(x, y)` | Dot product |
| $\|\|\mathbf{x}\|\|$ | `np.linalg.norm(x)` | Vector norm |
| $\mathbf{A}^{-1}$ | `np.linalg.inv(A)` | Matrix inverse |
| $\log(x)$ | `np.log(x)` | Natural log |
| $e^x$ | `np.exp(x)` | Exponential |

### PyMC Distributions

| Math | PyMC | Example |
|------|------|---------|
| $X \sim \mathcal{N}(\mu, \sigma)$ | `pm.Normal('X', mu=mu, sigma=sigma)` | |
| $X \sim \text{Beta}(\alpha, \beta)$ | `pm.Beta('X', alpha=alpha, beta=beta)` | |
| $X \sim \text{Exp}(\lambda)$ | `pm.Exponential('X', lam=lam)` | |

---

## 12. Reading Mathematical Expressions

### Example 1: Bayes' Theorem

$$p(\theta | y) = \frac{p(y | \theta) p(\theta)}{p(y)}$$

**Read aloud:**
"The posterior distribution of theta given y equals the likelihood of y given theta, times the prior of theta, divided by the marginal likelihood of y."

### Example 2: Expectation

$$\mathbb{E}[X] = \int x p(x) dx$$

**Read aloud:**
"The expected value of X equals the integral of x times the density of x, with respect to x."

### Example 3: State Space Model

$$y_t = Z_t \alpha_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, H_t)$$

**Read aloud:**
"Y sub t equals Z sub t times alpha sub t, plus epsilon sub t, where epsilon sub t follows a Normal distribution with mean zero and variance H sub t."

---

## Quick Reference Card

**Print this section for easy reference:**

```
CORE FORMULAS:

Bayes' Theorem:
  p(θ|y) ∝ p(y|θ) p(θ)

Variance:
  Var(X) = E[(X - E[X])²] = E[X²] - (E[X])²

Covariance:
  Cov(X,Y) = E[(X-E[X])(Y-E[Y])]

Normal PDF:
  p(x) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))

Matrix Multiplication:
  (AB)_ij = Σ_k A_ik B_kj

Transpose:
  (AB)^T = B^T A^T
```

---

## Tips for Reading Math

1. **Start with the left side:** Understand what's being defined
2. **Identify the operation:** Is it equality (=), proportion (∝), or distribution (~)?
3. **Work through subscripts:** Time indices, matrix dimensions, etc.
4. **Look for assumptions:** "where...", "given...", "for all..."
5. **Translate to words:** Read equations aloud to yourself
6. **Draw diagrams:** Visualize what the notation represents

---

## Common Confusions

| Easy to Confuse | Distinction |
|-----------------|-------------|
| $\sigma$ vs $\Sigma$ | σ = std dev (scalar), Σ = covariance matrix |
| $\epsilon$ vs $\eta$ | ε = observation noise, η = state innovation |
| $\alpha$ (parameter) vs $\alpha$ (state) | Context-dependent! |
| $\hat{\beta}$ vs $\bar{\beta}$ | hat = frequentist estimate, bar = Bayesian mean |
| $X$ vs $\mathbf{X}$ | Regular vs bold = scalar vs vector/matrix |

---

## Resources for Learning Math Notation

- **Khan Academy:** Math notation basics (free)
- **3Blue1Brown:** Visual explanations of linear algebra and calculus
- **Brilliant.org:** Interactive math learning
- **"Mathematical Statistics and Data Analysis" by John Rice:** Comprehensive reference

---

**Tip:** Bookmark this page and refer back whenever you see unfamiliar notation. Mathematical fluency develops with practice!

---

*"Notation is a tool for thought. Once you understand the symbols, the ideas become clear."*
