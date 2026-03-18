# Probability Review for Bayesian Time Series

## In Brief

This guide reviews the probability concepts essential for Bayesian inference. Mastery of random variables, key distributions, and conditional probability is the prerequisite for everything in this course.

> 💡 **Key Insight:** Bayesian inference is applied conditional probability. Every model in this course is an expression of $P(\theta | \text{data}) \propto P(\text{data} | \theta) \cdot P(\theta)$. The machinery of probability—densities, expectations, conditioning—is the language we use to write that expression precisely and compute it efficiently.

---

## Formal Definition

A **probability space** is a triple $(\Omega, \mathcal{F}, P)$ where $\Omega$ is the sample space, $\mathcal{F}$ a sigma-algebra of events, and $P: \mathcal{F} \to [0,1]$ satisfies the Kolmogorov axioms: $P(\Omega) = 1$, $P(A) \geq 0$, and countable additivity.

A **random variable** $X: \Omega \to \mathbb{R}$ is a measurable function from outcomes to real numbers. Its distribution is characterized by the CDF $F(x) = P(X \leq x)$.

**Conditional probability:** For events $A$ and $B$ with $P(B) > 0$:
$$P(A | B) = \frac{P(A \cap B)}{P(B)}$$

**Bayes' theorem** follows directly:
$$P(\theta | y) = \frac{P(y | \theta) \cdot P(\theta)}{P(y)}$$

---

## Intuitive Explanation

Think of a commodity price as a random variable: before observing tomorrow's close, the price lives on a distribution shaped by supply news, positioning, and macro factors. The PDF assigns density to each possible price level — higher density where prices are more likely.

Conditional probability narrows the sample space. "What's the probability of a crude inventory draw, given that US production is rising?" is a conditional question. The answer shifts our distribution for inventory changes, which flows through to our price forecast. This conditioning chain — from fundamentals to inventory to price — is exactly what Bayesian commodity models formalize.

---

## Code Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Compare Normal vs Student-t for commodity returns
# Student-t better captures fat tails (supply shocks, geopolitical events)
np.random.seed(42)

x = np.linspace(-5, 5, 500)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: Normal vs Student-t PDFs
axes[0].plot(x, stats.norm.pdf(x), 'b-', lw=2, label='Normal(0,1)')
axes[0].plot(x, stats.t.pdf(x, df=5), 'r--', lw=2, label='t(df=5)')
axes[0].plot(x, stats.t.pdf(x, df=2), 'g:', lw=2, label='t(df=2)')
axes[0].set_title('Normal vs Student-t Tails')
axes[0].set_xlabel('Return (standardized)')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].set_xlim(-5, 5)

# Panel 2: Bayesian updating with Beta-Binomial
# "How often do weekly EIA reports show inventory draws?"
alpha_prior, beta_prior = 2, 2  # Weak prior centered at 0.5
draws = [1, 0, 1, 1, 0, 1, 1, 0, 1, 1]  # 1 = inventory draw week

theta = np.linspace(0, 1, 500)
axes[1].plot(theta, stats.beta.pdf(theta, alpha_prior, beta_prior),
             'b-', lw=2, label='Prior Beta(2,2)')

alpha_curr, beta_curr = alpha_prior, beta_prior
for i, d in enumerate(draws):
    alpha_curr += d
    beta_curr += (1 - d)

axes[1].plot(theta, stats.beta.pdf(theta, alpha_curr, beta_curr),
             'r-', lw=2, label=f'Posterior Beta({alpha_curr},{beta_curr})')
axes[1].axvline(sum(draws)/len(draws), color='green', linestyle='--',
                label=f'MLE: {sum(draws)/len(draws):.1f}')
axes[1].set_title('Bayesian Updating: Probability of Inventory Draw')
axes[1].set_xlabel('θ (probability of draw)')
axes[1].set_ylabel('Density')
axes[1].legend()

plt.tight_layout()
plt.show()

# Print posterior summaries
post = stats.beta(alpha_curr, beta_curr)
print(f"Posterior mean: {post.mean():.3f}")
print(f"95% credible interval: [{post.ppf(0.025):.3f}, {post.ppf(0.975):.3f}]")
```

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

## Connections

**Builds on:**
- High school statistics: mean, variance, histograms
- Calculus: integration, derivatives (for density functions)

**Leads to:**
- Module 1: Bayes' theorem — conditional probability applied to parameters
- Module 1: Conjugate priors — specific distribution families for analytical updating
- Module 6: MCMC — numerical methods for computing posteriors when integration is intractable

**Related to:**
- Measure theory: the formal foundation of probability
- Information theory: entropy as a measure of uncertainty

---

## Common Pitfalls

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

---

## Further Reading

1. **DeGroot & Schervish** *Probability and Statistics* (4th ed.) — Rigorous undergraduate reference, Chapters 1-5 cover everything in this guide
2. **Blitzstein & Hwang** *Introduction to Probability* — Free online, outstanding intuition with story proofs
3. **Gelman et al.** *Bayesian Data Analysis* Appendix A — Brief notation and probability summary oriented toward BDA's approach
4. **3Blue1Brown** "Bayes theorem, the geometry of changing beliefs" (YouTube) — Excellent visual intuition for conditional probability
