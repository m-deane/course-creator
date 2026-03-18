# Conjugate Priors: Analytical Bayesian Updates

## In Brief

A **conjugate prior** is a prior distribution that, when combined with a particular likelihood, yields a posterior distribution in the same family as the prior. This allows for closed-form posterior computation without numerical methods.

## Key Insight

Conjugate priors exist for mathematical convenience, not physical reality. They are useful for:
1. Building intuition about Bayesian updating
2. Fast computation when applicable
3. Components within larger models
4. Initialization of MCMC samplers

## Intuitive Explanation

A conjugate prior is like using the same unit of measurement before and after an experiment. If you express your beliefs about a probability as "I've seen 3 successes out of 5 virtual trials" (a Beta prior), then after seeing real data your updated beliefs are still expressed in the same units — just with more virtual trials added. The form of uncertainty doesn't change, only its magnitude.

For commodity inventory forecasting: if you express prior beliefs about the weekly draw probability as Beta(6, 4) — equivalent to having seen 6 draws in 10 prior weeks — and then the EIA reports 3 more draws in the next 5 weeks, your posterior is simply Beta(9, 6). No integrals, no simulation. The Beta distribution's shape absorbed the data and spit out a new Beta.

The deeper insight: conjugacy is not a special feature of Bayesian inference — it is a property of the mathematical relationship between the exponential family distributions. Likelihoods from the exponential family always have conjugate priors. This covers the Normal, Poisson, Binomial, Gamma, and many other distributions used in commodity models.

---

## Formal Definition

A prior $p(\theta)$ is **conjugate** to a likelihood $p(y|\theta)$ if:

$$p(\theta | y) \propto p(y | \theta) \cdot p(\theta)$$

yields a posterior $p(\theta|y)$ in the same distributional family as $p(\theta)$.

---

## Major Conjugate Families

### 1. Beta-Binomial (Proportions)

**Use case:** Estimating probabilities, success rates, fill rates

**Prior:** $\theta \sim \text{Beta}(\alpha, \beta)$

**Likelihood:** $y | \theta \sim \text{Binomial}(n, \theta)$

**Posterior:** $\theta | y \sim \text{Beta}(\alpha + y, \beta + n - y)$

**Commodity application:** Probability that a crop report exceeds expectations; fill rate on limit orders.

```python
# Example: Estimating probability of inventory draw
alpha_prior, beta_prior = 2, 2  # Weak prior centered at 0.5

# Data: 8 draws out of 12 weeks
draws, weeks = 8, 12

# Posterior
alpha_post = alpha_prior + draws
beta_post = beta_prior + (weeks - draws)

print(f"Prior: Beta({alpha_prior}, {beta_prior})")
print(f"Posterior: Beta({alpha_post}, {beta_post})")
print(f"Posterior mean: {alpha_post / (alpha_post + beta_post):.3f}")
```

---

### 2. Normal-Normal (Means)

**Use case:** Estimating mean levels with known variance

**Prior:** $\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$

**Likelihood:** $y_i | \mu \sim \mathcal{N}(\mu, \sigma^2)$ (σ² known)

**Posterior:**
$$\mu | y \sim \mathcal{N}\left(\frac{\frac{\mu_0}{\sigma_0^2} + \frac{n\bar{y}}{\sigma^2}}{\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}}, \left(\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}\right)^{-1}\right)$$

**Precision form (cleaner):**

$$\tau_{\text{post}} = \tau_0 + n\tau$$
$$\mu_{\text{post}} = \frac{\tau_0 \mu_0 + n\tau \bar{y}}{\tau_{\text{post}}}$$

where $\tau = 1/\sigma^2$ is precision.

**Commodity application:** Estimating equilibrium price level; long-term mean reversion target.

```python
import numpy as np

# Prior: Mean price level around $80 with uncertainty
mu_0, sigma_0 = 80, 10
tau_0 = 1 / sigma_0**2

# Data: 20 observations with sample mean $75
n = 20
y_bar = 75
sigma = 5  # Known observation std
tau = 1 / sigma**2

# Posterior
tau_post = tau_0 + n * tau
mu_post = (tau_0 * mu_0 + n * tau * y_bar) / tau_post
sigma_post = np.sqrt(1 / tau_post)

print(f"Prior: N({mu_0}, {sigma_0}²)")
print(f"Posterior: N({mu_post:.2f}, {sigma_post:.2f}²)")
print(f"Posterior 95% CI: [{mu_post - 1.96*sigma_post:.2f}, {mu_post + 1.96*sigma_post:.2f}]")
```

**Key insight:** The posterior mean is a **precision-weighted average** of prior mean and data mean.

---

### 3. Gamma-Poisson (Rates)

**Use case:** Estimating rates, counts per unit time

**Prior:** $\lambda \sim \text{Gamma}(\alpha, \beta)$

**Likelihood:** $y_i | \lambda \sim \text{Poisson}(\lambda)$

**Posterior:** $\lambda | y \sim \text{Gamma}(\alpha + \sum y_i, \beta + n)$

**Commodity application:** Number of supply disruptions per quarter; frequency of limit moves.

```python
from scipy import stats

# Prior: Expect about 2 disruptions per year (α/β = 2)
alpha_prior, beta_prior = 2, 1

# Data: Observed 5 disruptions over 2 years
total_events = 5
time_periods = 2

# Posterior
alpha_post = alpha_prior + total_events
beta_post = beta_prior + time_periods

post_dist = stats.gamma(alpha_post, scale=1/beta_post)
print(f"Posterior mean rate: {post_dist.mean():.2f} per year")
print(f"95% CI: [{post_dist.ppf(0.025):.2f}, {post_dist.ppf(0.975):.2f}]")
```

---

### 4. Gamma-Normal (Variance/Precision)

**Use case:** Estimating variance or precision

**Prior:** $\tau \sim \text{Gamma}(\alpha, \beta)$ where $\tau = 1/\sigma^2$

**Likelihood:** $y_i | \mu, \tau \sim \mathcal{N}(\mu, 1/\tau)$ (μ known)

**Posterior:** $\tau | y \sim \text{Gamma}\left(\alpha + \frac{n}{2}, \beta + \frac{\sum(y_i - \mu)^2}{2}\right)$

**Commodity application:** Estimating volatility; variance of forecast errors.

---

### 5. Normal-Inverse-Gamma (Mean and Variance)

**Use case:** Jointly estimating mean and variance

**Prior:**
$$\sigma^2 \sim \text{Inv-Gamma}(\alpha_0, \beta_0)$$
$$\mu | \sigma^2 \sim \mathcal{N}(\mu_0, \sigma^2/\kappa_0)$$

This is the **Normal-Inverse-Gamma** prior, conjugate for the Normal likelihood with both unknown.

**Commodity application:** Jointly estimating expected return and volatility.

---

## Conjugate Prior Reference Table

| Likelihood | Parameter | Conjugate Prior | Posterior |
|------------|-----------|-----------------|-----------|
| Bernoulli/Binomial | p | Beta(α, β) | Beta(α + Σy, β + n - Σy) |
| Poisson | λ | Gamma(α, β) | Gamma(α + Σy, β + n) |
| Normal (known σ²) | μ | Normal(μ₀, σ₀²) | Normal(μₙ, σₙ²) |
| Normal (known μ) | σ² | Inv-Gamma(α, β) | Inv-Gamma(α + n/2, β + SS/2) |
| Exponential | λ | Gamma(α, β) | Gamma(α + n, β + Σy) |
| Multinomial | p | Dirichlet(α) | Dirichlet(α + counts) |

---

## Code Implementation

The code examples within each conjugate family above show single-update calculations. Here is a complete PyMC implementation that uses conjugate structure as initialization for a full posterior sampling workflow, demonstrating the connection between analytical and numerical approaches:

```python
import pymc as pm
import numpy as np
import arviz as az
from scipy import stats

# --- Analytical conjugate update (Normal-Normal) ---
# Prior on crude oil price level: N(80, 10^2)
mu_0, sigma_0 = 80.0, 10.0
tau_0 = 1.0 / sigma_0**2

# Observed weekly closes (20 weeks)
np.random.seed(42)
true_mu = 76.0
obs_sigma = 5.0
prices = np.random.normal(true_mu, obs_sigma, 20)

# Analytical posterior
tau_obs = 1.0 / obs_sigma**2
n = len(prices)
tau_post = tau_0 + n * tau_obs
mu_post = (tau_0 * mu_0 + n * tau_obs * prices.mean()) / tau_post
sigma_post = np.sqrt(1.0 / tau_post)

print("=== Analytical Conjugate Posterior ===")
print(f"Posterior mean:  {mu_post:.3f}")
print(f"Posterior std:   {sigma_post:.3f}")
print(f"95% CI:          [{mu_post - 1.96*sigma_post:.2f}, {mu_post + 1.96*sigma_post:.2f}]")

# --- Equivalent PyMC model (should match above exactly) ---
with pm.Model() as normal_normal_model:
    mu = pm.Normal('mu', mu=mu_0, sigma=sigma_0)
    y_obs = pm.Normal('y_obs', mu=mu, sigma=obs_sigma, observed=prices)
    trace = pm.sample(2000, tune=1000, random_seed=42, progressbar=False)

print("\n=== PyMC Posterior (should match analytical) ===")
print(az.summary(trace, var_names=['mu'])[['mean', 'sd', 'hdi_3%', 'hdi_97%']])
```

---

## Why Conjugate Priors Matter for Time Series

### Sequential Updating

For online learning, conjugate priors allow instant updates:

```python
class BayesianMeanEstimator:
    """Online Bayesian estimation of mean with known variance."""

    def __init__(self, prior_mean, prior_precision, obs_precision):
        self.mu = prior_mean
        self.tau = prior_precision
        self.obs_tau = obs_precision

    def update(self, y):
        """Update posterior with new observation."""
        # Precision-weighted update
        new_tau = self.tau + self.obs_tau
        new_mu = (self.tau * self.mu + self.obs_tau * y) / new_tau

        self.tau = new_tau
        self.mu = new_mu

        return self.mu, 1/np.sqrt(self.tau)

    def get_posterior(self):
        return self.mu, 1/np.sqrt(self.tau)

# Example: Online price level estimation
estimator = BayesianMeanEstimator(
    prior_mean=80,
    prior_precision=0.01,  # 1/100 = weak prior
    obs_precision=0.04     # 1/25 = σ=5 observations
)

prices = [75, 78, 72, 80, 76]
for p in prices:
    mu, std = estimator.update(p)
    print(f"After observing {p}: μ = {mu:.2f} ± {1.96*std:.2f}")
```

### Kalman Filter Connection

The Kalman filter (Module 3) is essentially conjugate Normal-Normal updating in state space form. Understanding conjugacy now makes state space models intuitive later.

---

## Choosing Prior Hyperparameters

### Weakly Informative Priors

When you have little prior knowledge:

- **Beta(1, 1):** Uniform on [0, 1]
- **Beta(0.5, 0.5):** Jeffrey's prior (less mass at 0 and 1)
- **Normal(0, 100):** Vague prior on real line
- **Gamma(0.001, 0.001):** Nearly non-informative for precision

### Encoding Domain Knowledge

**Example:** Seasonal inventory typically ranges from -5 to +10 million barrels

```python
# Prior mean: (10 + (-5)) / 2 = 2.5
# Prior std: Range/4 ≈ 3.75 (covers ~95% of expected range)
mu_0, sigma_0 = 2.5, 3.75
```

### Prior Predictive Checks

Always simulate from your prior and check if the predictions are reasonable:

```python
import numpy as np

# Prior on inventory change: N(0, 5)
prior_samples = np.random.normal(0, 5, 10000)

print(f"Prior 95% interval: [{np.percentile(prior_samples, 2.5):.1f}, "
      f"{np.percentile(prior_samples, 97.5):.1f}] million barrels")
```

If this range doesn't match domain knowledge, adjust the prior.

---

## Limitations of Conjugate Priors

1. **Restrictive:** Real-world priors may not fit conjugate forms
2. **Unrealistic:** Conjugacy may force inappropriate assumptions
3. **Multivariate:** Fewer conjugate families exist for complex models

**Solution:** Use MCMC (Module 6) for general priors. Conjugacy is a stepping stone, not a destination.

---

## Common Pitfalls

### 1. Forcing Conjugacy When Inappropriate

**Wrong:** "I'll use a Normal prior because it's conjugate"
**Right:** "Does a Normal prior represent my actual beliefs?"

### 2. Ignoring Parameterization

Different sources use different parameterizations for Gamma, Inverse-Gamma, etc. Check the rate vs. scale convention!

### 3. Zero Counts with Uniform Prior

With Beta(1,1) prior and 0 successes in 0 trials:
- Posterior is still Beta(1,1)
- The data didn't help because there was no data

---

## Connections

**Builds on:**
- Module 1 (Guide 1): Bayes' theorem — the prior-likelihood-posterior framework
- Module 0: Key distributions — Beta, Normal, Gamma are all conjugate family members

**Leads to:**
- Module 3: Kalman filter — exact Normal-Normal conjugate updates in state space form
- Module 6: MCMC — needed when conjugacy does not apply (non-conjugate priors, complex likelihoods)
- Module 4: Hierarchical models — conjugate components within larger non-conjugate structures

**Related concepts:**
- Exponential family distributions: the class of likelihoods that always have conjugate priors
- Natural parameters: the canonical parameterization that makes conjugate updates transparent

---

## Practice Problems

### Problem 1
You believe WTI volatility is around 20-40% annualized. Express this as a Gamma prior on variance (σ²). Hint: If σ ∈ [0.2, 0.4], then σ² ∈ [0.04, 0.16].

### Problem 2
Starting with Beta(5, 5) prior, update sequentially with observations: success, success, failure, success, success. Plot the posterior after each update.

### Problem 3
Derive the posterior for the Normal-Normal case from scratch using Bayes' theorem.

---

## Further Reading

1. **Murphy, K.** *Machine Learning: A Probabilistic Perspective* - Chapter 3 for conjugate families
2. **Gelman et al.** *BDA* - Appendix A for comprehensive table
3. **Bishop, C.** *Pattern Recognition and Machine Learning* - Chapter 2

---

*Conjugate priors are training wheels. They help you learn to ride, but eventually you'll want the full bicycle of MCMC.*
