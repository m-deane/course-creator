# Probability Review for Bayesian Time Series

## In Brief

This guide reviews the probability concepts essential for Bayesian inference. If these concepts feel unfamiliar, complete the prerequisite review before proceeding.

---

## 1. Probability Fundamentals

### Random Variables

A **random variable** X is a function that maps outcomes of a random experiment to real numbers.

**Discrete:** Takes countable values (e.g., number of trades)
$$P(X = x) = p(x)$$

**Continuous:** Takes any value in an interval (e.g., commodity price)
$$P(a \leq X \leq b) = \int_a^b f(x) dx$$

### Probability Mass/Density Functions

**PMF** (discrete): $p(x) = P(X = x)$, where $\sum_x p(x) = 1$

**PDF** (continuous): $f(x)$ such that $\int_{-\infty}^{\infty} f(x) dx = 1$

**Key insight:** For continuous variables, $P(X = x) = 0$ for any specific x. We work with densities and intervals.

---

## 2. Key Distributions

### Normal (Gaussian) Distribution

$$X \sim \mathcal{N}(\mu, \sigma^2)$$

$$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**Properties:**
- Mean: $\mathbb{E}[X] = \mu$
- Variance: $\text{Var}(X) = \sigma^2$
- Linear combinations remain Normal
- Central Limit Theorem: sums converge to Normal

**In commodities:** Log returns often approximately Normal; used as likelihood in many models.

### Gamma Distribution

$$X \sim \text{Gamma}(\alpha, \beta)$$

$$f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}, \quad x > 0$$

**Properties:**
- Mean: $\mathbb{E}[X] = \alpha/\beta$
- Variance: $\text{Var}(X) = \alpha/\beta^2$
- Conjugate prior for Poisson rate and Normal precision

**In commodities:** Models positive quantities like volatility, rates, durations.

### Beta Distribution

$$X \sim \text{Beta}(\alpha, \beta)$$

$$f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}, \quad x \in [0,1]$$

**Properties:**
- Mean: $\mathbb{E}[X] = \alpha/(\alpha+\beta)$
- Conjugate prior for Binomial probability
- Flexible shapes depending on parameters

**In commodities:** Models proportions, probabilities, regime weights.

### Student-t Distribution

Commodity prices routinely produce extreme moves that the Normal distribution treats as nearly impossible — crude oil can spike 10% overnight on a supply disruption, and agricultural markets shift sharply on a single USDA crop report. The Student-t's heavier tails assign meaningful probability to these regime changes, which makes it a far more honest likelihood for commodity return models than the Gaussian.

$$X \sim t_\nu(\mu, \sigma^2)$$

**Properties:**
- Heavier tails than Normal
- Approaches Normal as $\nu \to \infty$
- Robust to outliers in regression

**In commodities:** Better fits for returns with fat tails; robust inference.

### Quick Reference: Distributions in Commodity Forecasting

| Distribution | Support | Typical Use in Commodities | Conjugate For |
|---|---|---|---|
| Normal $\mathcal{N}(\mu, \sigma^2)$ | $(-\infty, \infty)$ | Log returns, forecast errors | Normal likelihood |
| Gamma $\text{Gamma}(\alpha, \beta)$ | $(0, \infty)$ | Volatility, rates, durations | Poisson rate, Normal precision |
| Beta $\text{Beta}(\alpha, \beta)$ | $[0, 1]$ | Proportions, regime weights | Binomial probability |
| Student-t $t_\nu(\mu, \sigma^2)$ | $(-\infty, \infty)$ | Fat-tailed returns, robust regression | — |

---

## 3. Joint, Marginal, and Conditional Probability

### Joint Probability

For two random variables X and Y:
$$P(X, Y) = P(X \cap Y)$$

For continuous: $f(x, y)$ is the joint density.

### Marginal Probability

Obtained by summing/integrating out the other variable:
$$p(x) = \sum_y p(x, y) \quad \text{(discrete)}$$
$$f(x) = \int f(x, y) dy \quad \text{(continuous)}$$

### Conditional Probability

$$P(X | Y) = \frac{P(X, Y)}{P(Y)}$$

**This is the foundation of Bayesian inference.**

### Independence

X and Y are independent if:
$$P(X, Y) = P(X) P(Y)$$

Equivalently: $P(X|Y) = P(X)$

---

## 4. Expectation and Variance

### Expectation

**Discrete:** $\mathbb{E}[X] = \sum_x x \cdot p(x)$

**Continuous:** $\mathbb{E}[X] = \int x \cdot f(x) dx$

**Properties:**
- Linearity: $\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$
- $\mathbb{E}[g(X)] = \int g(x) f(x) dx$

### Variance

$$\text{Var}(X) = \mathbb{E}[(X - \mu)^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

**Properties:**
- $\text{Var}(aX + b) = a^2 \text{Var}(X)$
- $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)$

### Covariance and Correlation

$$\text{Cov}(X, Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)]$$

$$\rho_{XY} = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y} \in [-1, 1]$$

---

## 5. Bayes' Theorem (Preview)

The foundation of everything in this course:

$$P(\theta | \text{data}) = \frac{P(\text{data} | \theta) \cdot P(\theta)}{P(\text{data})}$$

**Components:**
- $P(\theta | \text{data})$: **Posterior** - what we learn about parameters after seeing data
- $P(\text{data} | \theta)$: **Likelihood** - probability of data given parameters
- $P(\theta)$: **Prior** - what we believed before seeing data
- $P(\text{data})$: **Evidence** - normalizing constant

**Key insight:** Bayesian inference updates beliefs (prior → posterior) using data (likelihood).

---

## 6. Law of Total Probability

$$P(A) = \sum_i P(A | B_i) P(B_i)$$

For continuous:
$$f(x) = \int f(x | \theta) f(\theta) d\theta$$

**In Bayesian inference:** This is how we compute the evidence (marginal likelihood).

---

## 7. Common Pitfalls

### Confusing PDF value with probability
- $f(x) = 2$ does NOT mean $P(X = x) = 2$
- PDF values can exceed 1
- Only integrals give probabilities

### Forgetting conditioning context
- $P(A|B) \neq P(B|A)$ in general
- Always be clear what you're conditioning on

### Independence assumptions
- Assuming independence when variables are dependent leads to wrong conclusions
- Time series are typically NOT independent across time

### Ignoring the normalizing constant
- In Bayesian inference, we often compute unnormalized posteriors
- For MCMC, this is fine; for other purposes, normalization matters

---

## 8. Quick Reference Table

| Distribution | Notation | Parameters | Mean | Variance |
|-------------|----------|------------|------|----------|
| Normal | $\mathcal{N}(\mu, \sigma^2)$ | $\mu \in \mathbb{R}, \sigma > 0$ | $\mu$ | $\sigma^2$ |
| Gamma | $\text{Gamma}(\alpha, \beta)$ | $\alpha, \beta > 0$ | $\alpha/\beta$ | $\alpha/\beta^2$ |
| Beta | $\text{Beta}(\alpha, \beta)$ | $\alpha, \beta > 0$ | $\alpha/(\alpha+\beta)$ | $\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ |
| Student-t | $t_\nu$ | $\nu > 0$ (df) | 0 (if $\nu > 1$) | $\nu/(\nu-2)$ (if $\nu > 2$) |
| Exponential | $\text{Exp}(\lambda)$ | $\lambda > 0$ | $1/\lambda$ | $1/\lambda^2$ |
| Poisson | $\text{Pois}(\lambda)$ | $\lambda > 0$ | $\lambda$ | $\lambda$ |

---

## Practice Problems

1. If $X \sim \mathcal{N}(0, 1)$, what is $P(-1.96 < X < 1.96)$?

2. If $X \sim \text{Beta}(2, 2)$, compute $\mathbb{E}[X]$ and sketch the PDF.

3. Given $P(A) = 0.3$, $P(B|A) = 0.8$, $P(B|A^c) = 0.2$, find $P(A|B)$.

4. If $X$ and $Y$ are independent with $\text{Var}(X) = 4$ and $\text{Var}(Y) = 9$, what is $\text{Var}(2X - 3Y)$?

5. For a Gamma prior on precision $\tau \sim \text{Gamma}(1, 1)$ and Normal likelihood $x | \tau \sim \mathcal{N}(0, 1/\tau)$, write out the posterior kernel $p(\tau | x)$.

---

*Answers and detailed solutions in the notebook exercises.*
