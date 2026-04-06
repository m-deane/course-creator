# Bayesian Regression

> **Reading time:** ~7 min | **Module:** 1 — Bayesian Fundamentals | **Prerequisites:** Module 0 Foundations


## In Brief

Bayesian regression treats regression coefficients as random variables with distributions, rather than fixed unknown values. This provides full uncertainty quantification for predictions and naturally handles regularization through priors.

<div class="callout-insight">

<strong>Insight:</strong> In frequentist regression, we get point estimates: β̂ = 2.5. In Bayesian regression, we get distributions: β ~ Normal(2.5, 0.3), telling us both the most likely value AND our uncertainty about it.

</div>

## Formal Definition

**Model:**
$$y_i = \beta_0 + \beta_1 x_{i1} + \ldots + \beta_p x_{ip} + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)$$

**Bayesian Specification:**
- **Prior:** $p(\beta, \sigma^2)$ - Our belief about parameters before seeing data
- **Likelihood:** $p(y | X, \beta, \sigma^2)$ - Probability of data given parameters
- **Posterior:** $p(\beta, \sigma^2 | y, X) \propto p(y | X, \beta, \sigma^2) p(\beta, \sigma^2)$

The posterior distribution gives us complete information about parameter uncertainty.

## Intuitive Explanation

Think of regression as drawing a line through data points.

**Frequentist approach:** "Based on this data, the best line has slope 2.5"

**Bayesian approach:** "Based on this data and my prior knowledge, I'm 95% confident the slope is between 2.0 and 3.0, with 2.5 most likely"

The Bayesian approach gives you a range of plausible lines, weighted by how likely each is.

### Analogy

Imagine predicting tomorrow's crude oil price:

- **Frequentist:** "My model predicts $75.00"
- **Bayesian:** "My model predicts $75.00 ± $3.50 (95% credible interval)"

The Bayesian answer helps you assess risk: Should I hedge? How big is my exposure? The point estimate alone doesn't tell you this.

## Mathematical Formulation

### Likelihood

With Normal errors:
$$p(y | X, \beta, \sigma^2) = \prod_{i=1}^n \mathcal{N}(y_i | X_i\beta, \sigma^2)$$

In matrix form:
$$p(y | X, \beta, \sigma^2) = \mathcal{N}(y | X\beta, \sigma^2 I)$$

### Conjugate Prior (Normal-Inverse-Gamma)

For analytical tractability:
$$\beta | \sigma^2 \sim \mathcal{N}(\mu_0, \sigma^2 \Sigma_0)$$
$$\sigma^2 \sim \text{InverseGamma}(a_0, b_0)$$

This choice makes the posterior also Normal-Inverse-Gamma (conjugacy).

### Posterior (Analytical Result)

With conjugate priors, the posterior is:
$$\beta | \sigma^2, y \sim \mathcal{N}(\mu_n, \sigma^2 \Sigma_n)$$

Where:
$$\Sigma_n = (\Sigma_0^{-1} + X^T X)^{-1}$$
$$\mu_n = \Sigma_n (\Sigma_0^{-1} \mu_0 + X^T y)$$

**Interpretation:**
- $\Sigma_n^{-1}$ combines prior precision $\Sigma_0^{-1}$ and data precision $X^T X$
- $\mu_n$ is a weighted average of prior mean $\mu_0$ and data-based estimate

### Predictive Distribution

For new data $x_{new}$:
$$p(y_{new} | x_{new}, y, X) = \int p(y_{new} | x_{new}, \beta, \sigma^2) p(\beta, \sigma^2 | y, X) d\beta d\sigma^2$$

This is a **Student's t-distribution** with interpretable parameters, giving natural prediction intervals.

## Code Implementation

### Basic Bayesian Regression with PyMC


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import pymc as pm
import numpy as np
import arviz as az

# Simulated commodity data
np.random.seed(42)
n = 100
inventory = np.random.randn(n)  # Standardized inventory levels
price = 50 + (-2.5 * inventory) + np.random.randn(n) * 3  # True β₁ = -2.5

# Bayesian regression model
with pm.Model() as model:
    # Priors
    intercept = pm.Normal('intercept', mu=50, sigma=10)
    beta_inventory = pm.Normal('beta_inventory', mu=0, sigma=5)
    sigma = pm.HalfNormal('sigma', sigma=10)

    # Linear model
    mu = intercept + beta_inventory * inventory

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=price)

    # Inference
    trace = pm.sample(2000, return_inferencedata=True)

# Posterior summary
print(az.summary(trace, var_names=['intercept', 'beta_inventory', 'sigma']))

# 95% credible intervals
print("\n95% Credible Intervals:")
print(az.hdi(trace, hdi_prob=0.95))
```

</div>
</div>

### Prediction with Uncertainty


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Posterior predictive distribution
with model:
    pm.sample_posterior_predictive(trace, extend_inferencedata=True)

# Predict for new inventory level
new_inventory = np.array([-1.0])  # Low inventory

# Extract posterior samples
intercept_samples = trace.posterior['intercept'].values.flatten()
beta_samples = trace.posterior['beta_inventory'].values.flatten()
sigma_samples = trace.posterior['sigma'].values.flatten()

# Posterior predictive samples
n_samples = len(intercept_samples)
predictions = (intercept_samples + beta_samples * new_inventory[0] +
               np.random.normal(0, sigma_samples))

# Credible interval
print(f"Predicted price for inventory={new_inventory[0]}:")
print(f"  Median: ${np.median(predictions):.2f}")
print(f"  95% CI: [${np.percentile(predictions, 2.5):.2f}, "
      f"${np.percentile(predictions, 97.5):.2f}]")
```

</div>
</div>

## Visual Representation

```
Prior Belief          +  Observed Data  =  Posterior Belief
   p(β)                   p(y | β, X)         p(β | y, X)

    │                         │                    │
    │   ┌─────┐              │                    │  ┌──┐
    │   │     │              │   ●●              │  │  │
    │  ┌┴─────┴┐             │   ●  ●●●          │ ┌┴──┴┐
────┼──┴───────┴───        ──┼───────●●──      ──┼─┴────┴─────
    │                         │     ●●              │
   β=0                      y data               β=-2.3
  (vague prior)                              (data informs us)

As n → ∞, posterior concentrates near true value
As prior gets weaker, posterior → MLE
```

## Common Pitfalls

### 1. Overly Informative Priors with Weak Data
**Problem:** Strong prior + little data → Posterior mostly reflects prior, not data
<div class="callout-warning">

<strong>Warning:</strong> **Problem:** Strong prior + little data → Posterior mostly reflects prior, not data

</div>


**Example:**

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# BAD: Very tight prior with only 10 data points
beta = pm.Normal('beta', mu=0, sigma=0.1)  # σ=0.1 very tight!
```

</div>
</div>

**Solution:** Use weakly informative priors unless you have strong domain knowledge

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# GOOD: Regularizing but not dominating
beta = pm.Normal('beta', mu=0, sigma=10)  # Allows data to speak
```

</div>
</div>

### 2. Forgetting Uncertainty in Predictions
**Problem:** Reporting posterior mean as point forecast without credible interval

**Fix:** Always report uncertainty for decisions under risk

### 3. Prior-Data Conflict
**Problem:** Data strongly contradicts prior → Model fit issues

**Diagnostic:** Posterior predictive checks show systematic bias

**Solution:**
- Check if prior is too restrictive
- Consider alternative models
- Investigate data quality

### 4. Confusing Credible Intervals with Confidence Intervals
**Bayesian (Credible):** "95% probability β is in [2.0, 3.0]" ✓
**Frequentist (Confidence):** "If we repeated this experiment, 95% of intervals would contain β"

Credible intervals are simpler to interpret!

## Connections

### Builds on:
- **Module 1.1 (Bayes' Theorem):** Posterior ∝ Likelihood × Prior
- **Module 1.2 (Conjugate Priors):** Normal-Normal conjugacy for analytical posteriors
- **Basic regression:** Understanding of linear models

### Leads to:
- **Module 3 (State Space Models):** Dynamic regression with time-varying coefficients
- **Module 4 (Hierarchical Models):** Multi-level regression with partial pooling
- **Module 8 (Fundamentals Integration):** Regression with commodity fundamental variables

### Related to:
- **Regularization:** Bayesian priors ↔ Ridge/Lasso penalties
- **Machine Learning:** Bayesian linear regression ↔ Gaussian Process regression
- **Credible intervals:** Bayesian alternative to confidence intervals

## Practice Problems

### 1. Conceptual: Prior Choice
You're modeling how crude oil inventory affects price. Which prior for β_inventory is most appropriate?

a) `Normal(0, 100000)` - Very vague
b) `Normal(-5, 2)` - Negative effect, moderate uncertainty
c) `Normal(+5, 2)` - Positive effect, moderate uncertainty

**Answer:** (b) - Economic theory says high inventory → lower prices (negative β)


<div class="callout-key">

<strong>Key Concept Summary:</strong> Bayesian regression treats regression coefficients as random variables with distributions, rather than fixed unknown values.


---

### 2. Implementation: Commodity Regression
Given data on natural gas prices and heating degree days (HDD):
- Fit a Bayesian regression: Price ~ HDD
- Report 95% credible intervals for the coefficient
- Make a prediction for HDD = 500

**Starter code:**

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
import pymc as pm

# Your code here
with pm.Model() as gas_model:
    # Define priors
    # Define likelihood
    # Sample posterior
    pass
```


---

### 3. Extension: Model Comparison
You have two competing models for oil prices:
- **Model A:** Price ~ Inventory
- **Model B:** Price ~ Inventory + Production

How would you compare them using Bayesian methods?

**Hint:** Look into WAIC (Widely Applicable Information Criterion) or LOO (Leave-One-Out) cross-validation via `arviz.compare()`.

---

### 4. Research Question: Informative Priors
Suppose you have strong prior belief that the inventory effect is β ~ N(-3, 0.5). After seeing data, your posterior is β ~ N(-1.2, 0.3).

What does this suggest? Should you:
a) Trust the posterior (data overrides prior)
b) Question the data quality
c) Investigate why prior and data differ

**Discussion:** This is a red flag requiring investigation - strong prior-data conflict often reveals model misspecification or data issues.


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Formal Definition" and why it matters in practice.

2. Given a real-world scenario involving bayesian regression, what would be your first three steps to apply the techniques from this guide?


## Further Reading

### Foundational Papers
- **Gelman et al. (2013):** *Bayesian Data Analysis* (3rd ed.) - Chapter 14-16 on regression
- **McElreath (2020):** *Statistical Rethinking* - Intuitive introduction to Bayesian regression

### Advanced Topics
- **George & McCulloch (1997):** "Stochastic search variable selection" - Bayesian variable selection
- **Park & Casella (2008):** "The Bayesian Lasso" - Connecting priors to regularization

### Applications to Commodities
- **Baumeister & Kilian (2015):** "Forecasting the real price of oil in a changing world" - Bayesian VAR for oil
- **Koop & Korobilis (2010):** "Bayesian Multivariate Time Series Methods for Empirical Macroeconomics" - Dynamic regression

### Software Documentation
- **PyMC:** https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/GLM_linear.html
- **Stan:** https://mc-stan.org/docs/stan-users-guide/linear-regression.html
- **ArviZ:** https://arviz-devs.github.io/arviz/ - Visualization and diagnostics

---

**Next:** Apply these concepts in the interactive notebook `03_bayesian_regression_pymc.ipynb`

---

## Cross-References

<a class="link-card" href="./03_bayesian_regression_slides.md">
  <div class="link-card-title">Companion Slide Deck</div>
  <div class="link-card-description">Visual presentation covering the key concepts from this guide.</div>
</a>

<a class="link-card" href="../notebooks/01_bayesian_regression_pymc.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>
