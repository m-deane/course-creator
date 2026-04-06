# Bayes' Theorem: The Foundation of Bayesian Inference

> **Reading time:** ~6 min | **Module:** 1 — Bayesian Fundamentals | **Prerequisites:** Module 0 Foundations


## In Brief

Bayes' theorem provides a principled way to update beliefs in light of new evidence. It forms the mathematical foundation for all probabilistic machine learning and is essential for building forecasting models that quantify uncertainty.

<div class="callout-insight">

<strong>Insight:</strong> **Bayesian inference inverts the usual probability question.** Instead of asking "What's the probability of seeing this data given a parameter?" we ask "What's the probability of a parameter given this data?" This inversion is exactly what we need for forecasting.

</div>

## Formal Definition

### Bayes' Theorem

$$P(\theta | y) = \frac{P(y | \theta) \cdot P(\theta)}{P(y)}$$

Where:
- $P(\theta | y)$ = **Posterior**: probability of parameter θ given data y
- $P(y | \theta)$ = **Likelihood**: probability of data y given parameter θ
- $P(\theta)$ = **Prior**: probability of parameter θ before seeing data
- $P(y)$ = **Evidence** (or marginal likelihood): probability of the data

### Proportional Form (Most Useful in Practice)

Since $P(y)$ doesn't depend on θ:

$$P(\theta | y) \propto P(y | \theta) \cdot P(\theta)$$

**Posterior ∝ Likelihood × Prior**

This is the form we use most often. The normalizing constant $P(y)$ is often intractable but unnecessary for inference.


<div class="callout-key">

<strong>Key Concept Summary:</strong> Bayes' theorem provides a principled way to update beliefs in light of new evidence.

</div>

---

## Intuitive Explanation

### The Updating Metaphor

Imagine you're a commodity trader forming beliefs about next month's crude oil inventory change:

1. **Prior ($P(\theta)$):** Before seeing any new data, you have beliefs based on:
   - Historical patterns
   - Seasonal expectations
   - Recent production trends
   - Your mental model of the market

2. **Likelihood ($P(y|\theta)$):** You observe new data (weekly EIA reports). The likelihood asks: "If the true inventory change is θ, how probable is the data I observed?"

3. **Posterior ($P(\theta|y)$):** After processing the new data, your updated beliefs. This becomes tomorrow's prior when new data arrives.

### Visual Intuition

```
Week 1:  Prior ───┬──── Data ────→ Posterior₁
                  │
Week 2:  Posterior₁ ──┬── Data ────→ Posterior₂
(as Prior)            │
                  ...
Week N:  PosteriorN-1 ─┬── Data ────→ PosteriorN
```

**Sequential updating:** Today's posterior becomes tomorrow's prior.

---

## Mathematical Formulation

### Continuous Case

For continuous parameters with densities:

$$p(\theta | y) = \frac{p(y | \theta) \cdot p(\theta)}{\int p(y | \theta') p(\theta') d\theta'}$$

The integral in the denominator is the **marginal likelihood** or **evidence**:

$$p(y) = \int p(y | \theta) p(\theta) d\theta$$

### Multiple Observations

For independent observations $y_1, ..., y_n$:

$$p(\theta | y_{1:n}) \propto p(\theta) \prod_{i=1}^{n} p(y_i | \theta)$$

### Sequential Updating

Bayes' theorem can be applied sequentially:

$$p(\theta | y_1, y_2) = \frac{p(y_2 | \theta) \cdot p(\theta | y_1)}{p(y_2 | y_1)}$$

This is crucial for time series: we update beliefs as each new observation arrives.

---

## Code Implementation

### Simple Example: Estimating a Probability


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Prior: Beta(2, 2) - slight belief that probability is around 0.5
alpha_prior, beta_prior = 2, 2

# Data: 7 successes out of 10 trials
successes, trials = 7, 10
failures = trials - successes

# Posterior: Beta(alpha + successes, beta + failures)

# This is the conjugate update for Beta-Binomial
alpha_post = alpha_prior + successes
beta_post = beta_prior + failures

# Visualize
theta = np.linspace(0, 1, 1000)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(theta, stats.beta.pdf(theta, alpha_prior, beta_prior),
        'b-', label=f'Prior: Beta({alpha_prior}, {beta_prior})', lw=2)
ax.plot(theta, stats.beta.pdf(theta, alpha_post, beta_post),
        'r-', label=f'Posterior: Beta({alpha_post}, {beta_post})', lw=2)
ax.axvline(successes/trials, color='green', linestyle='--',
           label=f'MLE: {successes/trials:.2f}')
ax.set_xlabel('θ (probability)')
ax.set_ylabel('Density')
ax.set_title('Bayesian Updating: Beta-Binomial')
ax.legend()
plt.show()

# Posterior summaries
post_dist = stats.beta(alpha_post, beta_post)
print(f"Posterior mean: {post_dist.mean():.3f}")
print(f"Posterior std: {post_dist.std():.3f}")
print(f"95% credible interval: [{post_dist.ppf(0.025):.3f}, {post_dist.ppf(0.975):.3f}]")
```

</div>
</div>

### PyMC Implementation


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
import pymc as pm
import arviz as az

# Same problem in PyMC
with pm.Model() as coin_model:
    # Prior
    theta = pm.Beta('theta', alpha=2, beta=2)

    # Likelihood
    y = pm.Binomial('y', n=10, p=theta, observed=7)

    # Sample from posterior
    trace = pm.sample(2000, tune=1000, random_seed=42)

# Posterior summary
az.summary(trace, var_names=['theta'])
```

</div>
</div>

---

## Common Pitfalls

### 1. Confusing Prior and Posterior

**Wrong:** "The prior is P(θ|y)"
**Right:** "The prior is P(θ) - our belief BEFORE seeing data"

### 2. Ignoring Prior Sensitivity

Different priors can lead to different posteriors, especially with limited data. Always perform sensitivity analysis.

### 3. Treating Posterior as Truth

The posterior is our best belief given the model and data. The model may be wrong!

### 4. Computing the Evidence

The evidence P(y) is often intractable. Fortunately, for most inference tasks (MCMC, MAP), we only need the unnormalized posterior.

---

## Connections

### Builds on
- Conditional probability: P(A|B) = P(A,B)/P(B)
- Law of total probability: P(y) = ∫P(y|θ)P(θ)dθ

### Leads to
- **Conjugate priors** (next guide): Special prior-likelihood pairs with analytical posteriors
- **MCMC** (Module 6): Approximating posteriors when closed-form is unavailable
- **Hierarchical models** (Module 4): Multi-level prior structures

---

## Practice Problems

### Problem 1: Medical Testing
A test for a disease has 95% sensitivity (P(positive|disease) = 0.95) and 90% specificity (P(negative|no disease) = 0.90). If 1% of the population has the disease, what's P(disease|positive)?

**Solution approach:** Apply Bayes' theorem with:
- P(disease) = 0.01
- P(positive|disease) = 0.95
- P(positive|no disease) = 0.10

### Problem 2: Inventory Surprise
Prior belief: inventory change is Normal(0, 10) million barrels. EIA reports -3 million (a small draw). If your likelihood is Normal(θ, 2), what's the posterior?

**Solution approach:** Use Normal-Normal conjugacy (covered in next guide).

### Problem 3: Sequential Updates
Starting with Beta(1,1) prior, you observe: success, success, failure, success. Calculate the posterior after each observation and plot the evolution.

---


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Formal Definition" and why it matters in practice.

2. Given a real-world scenario involving bayes' theorem: the foundation of bayesian inference, what would be your first three steps to apply the techniques from this guide?

</div>

## Further Reading

1. **McElreath, Ch. 2** - "Small Worlds and Large Worlds" - Excellent intuitive introduction
2. **Gelman et al., Ch. 1** - "Probability and inference" - More formal treatment
3. **Jaynes, E.T.** - "Probability Theory: The Logic of Science" - Foundational philosophy

---

## Key Takeaways

1. **Bayes' theorem inverts probability** - from P(data|parameter) to P(parameter|data)
2. **Posterior ∝ Likelihood × Prior** - the core update equation
3. **Sequential updating** - today's posterior becomes tomorrow's prior
4. **Uncertainty quantification** - we get full distributions, not just point estimates
5. **Prior sensitivity** - always check how prior choice affects conclusions

---

*"The posterior distribution represents everything we know about the parameter after seeing the data. Everything. If it's not in the posterior, we don't know it."* — Andrew Gelman

---

## Cross-References

<a class="link-card" href="./01_bayes_theorem_slides.md">
  <div class="link-card-title">Companion Slide Deck</div>
  <div class="link-card-description">Visual presentation covering the key concepts from this guide.</div>
</a>

<a class="link-card" href="../notebooks/01_bayesian_regression_pymc.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>
