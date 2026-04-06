# Bayesian Bandits Cheatsheet

> **Reading time:** ~20 min | **Module:** 02 — Bayesian Bandits | **Prerequisites:** Module 1


## Thompson Sampling Algorithm


<div class="callout-key">

**Key Concept Summary:** Interpretation: Posterior mean is precision-weighted

</div>

### Beta-Bernoulli (Binary Rewards)
<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
# Initialize
alpha = np.ones(K)  # Successes + 1
beta = np.ones(K)   # Failures + 1

# Each round
samples = np.random.beta(alpha, beta)
arm = np.argmax(samples)

# Update
if reward == 1:
    alpha[arm] += 1
else:
    beta[arm] += 1
```

</div>

### Gaussian (Continuous Rewards, Known Variance)
<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
# Initialize
mu = np.zeros(K)        # Posterior means
sigma = np.ones(K)      # Posterior std devs
sigma_data = 0.1        # Known observation noise

# Each round
samples = np.random.normal(mu, sigma)
arm = np.argmax(samples)

# Update (Normal-Normal conjugacy)
tau_post = 1/sigma[arm]**2 + 1/sigma_data**2
mu[arm] = (mu[arm]/sigma[arm]**2 + reward/sigma_data**2) / tau_post
sigma[arm] = np.sqrt(1 / tau_post)
```

</div>

### Poisson (Count Data)
<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
# Initialize
alpha = np.ones(K)      # Shape parameter
beta = np.ones(K)       # Rate parameter

# Each round
samples = np.random.gamma(alpha, 1/beta)
arm = np.argmax(samples)

# Update
alpha[arm] += reward
beta[arm] += 1
```

</div>

## Conjugate Prior Table

<div class="callout-insight">

**Insight:** The core insight of bandit algorithms is that learning and earning are not separate phases. Every observation contributes to both understanding which option is best and generating value from the best option.

</div>


| Likelihood | Parameter | Conjugate Prior | Posterior Update |
|------------|-----------|-----------------|------------------|
| Bernoulli(θ) | θ ∈ [0,1] | Beta(α, β) | Beta(α+k, β+n-k) |
| Normal(μ, σ²) | μ, known σ² | Normal(μ₀, σ₀²) | Normal(μ₁, σ₁²)* |
| Poisson(λ) | λ > 0 | Gamma(α, β) | Gamma(α+Σx, β+n) |
| Exponential(λ) | λ > 0 | Gamma(α, β) | Gamma(α+n, β+Σx) |
| Normal(μ, σ²) | both unknown | Normal-Gamma | (requires MCMC) |

*Normal-Normal: τ₁ = τ₀ + n/σ², μ₁ = (τ₀μ₀ + nμ̂/σ²)/τ₁, σ₁² = 1/τ₁

## Posterior Update Formulas

<div class="callout-warning">

**Warning:** Bandit algorithms assume the reward distributions are stationary (or slowly changing). In commodity markets, regime shifts can make a historically optimal arm suddenly suboptimal. Always implement change detection alongside your bandit.

</div>


### Beta-Bernoulli
```
Prior: θ ~ Beta(α₀, β₀)
Data: k successes in n trials
Posterior: θ ~ Beta(α₀ + k, β₀ + n - k)

Mean: α / (α + β)
Variance: αβ / [(α+β)²(α+β+1)]
Mode: (α-1) / (α+β-2) for α,β > 1
```

### Normal-Normal (Known Variance)
```
Prior: μ ~ N(μ₀, σ₀²)
Data: x₁, ..., xₙ ~ N(μ, σ²) with known σ²
Posterior: μ ~ N(μ₁, σ₁²)

Precision formulation:
τ₀ = 1/σ₀², τ = 1/σ²
τ₁ = τ₀ + n·τ
μ₁ = (τ₀·μ₀ + n·τ·x̄) / τ₁
σ₁² = 1/τ₁

Interpretation: Posterior mean is precision-weighted
average of prior mean and sample mean
```

### Gamma-Poisson
```
Prior: λ ~ Gamma(α₀, β₀)
Data: x₁, ..., xₙ ~ Poisson(λ)
Posterior: λ ~ Gamma(α₀ + Σxᵢ, β₀ + n)

Mean: α/β
Variance: α/β²
```

## Thompson Sampling vs UCB

| Feature | Thompson Sampling | UCB1 |
|---------|-------------------|------|
| **Selection** | Sample θᵢ ~ posterior, pick max | Pick max(μ̂ᵢ + √(2ln t/nᵢ)) |
| **Exploration** | Stochastic (probability matching) | Deterministic (optimism) |
| **Prior knowledge** | Yes (Bayesian priors) | No (frequentist) |
| **Delayed feedback** | Handles naturally | Struggles |
| **Non-stationary** | Discount posteriors easily | Requires modification |
| **Reproducibility** | Needs random seed | Fully deterministic |
| **Computation** | O(K) sampling | O(K) comparison |
| **Regret bound** | O(log T) | O(log T) |

## Common Prior Choices

### Uninformative Priors
```python
# Beta-Bernoulli: Uniform prior
alpha, beta = 1, 1  # Beta(1,1) = Uniform(0,1)

# Gaussian: Weak prior
mu_0, sigma_0 = 0, 10  # Wide, centered at 0

# Gamma-Poisson: Jeffreys prior
alpha, beta = 0.5, 0  # Uninformative
```

### Informative Priors
```python
# Beta with mean μ and strength n
def beta_from_mean_strength(mu, n):
    alpha = n * mu
    beta = n * (1 - mu)
    return alpha, beta

# Example: Believe θ ≈ 0.6, equivalent to 10 observations
alpha, beta = beta_from_mean_strength(0.6, 10)  # Beta(6, 4)
```

## Non-Stationary Adaptations

### Exponential Discounting
```python
gamma = 0.99  # Discount factor

# Each period, before updating
alpha *= gamma
beta *= gamma

# Then update normally
alpha[arm] += reward
beta[arm] += 1 - reward
```

### Sliding Window
```python
window_size = 100
successes = deque(maxlen=window_size)
failures = deque(maxlen=window_size)

# Update
successes.append(reward)
failures.append(1 - reward)

# Posteriors based on window
alpha = 1 + sum(successes)
beta = 1 + sum(failures)
```

### Reset on Change Detection
```python
# Detect if posterior mean shifts significantly
if abs(mu_new - mu_old) > threshold:
    alpha, beta = 1, 1  # Reset to prior
```

## Implementation Checklist

- [ ] Choose conjugate prior matching likelihood
- [ ] Initialize with weak priors (unless prior knowledge exists)
- [ ] Sample from posterior each round
- [ ] Select arm with highest sample
- [ ] Update posterior with observed reward
- [ ] Handle non-stationarity (discount or window)
- [ ] Track posterior evolution for debugging
- [ ] Set random seed for reproducibility

## Commodity Trading Applications

### Portfolio Allocation (Gaussian Returns)
```python
# 3 commodities: WTI, Gold, Corn
mu = np.array([0.0, 0.0, 0.0])       # Prior mean returns
sigma = np.array([0.05, 0.05, 0.05]) # Prior uncertainty
sigma_obs = 0.10                      # Weekly return volatility

# Each week
samples = np.random.normal(mu, sigma)
allocate_to = np.argmax(samples)

# Observe return
mu[allocate_to], sigma[allocate_to] = normal_update(
    mu[allocate_to], sigma[allocate_to], return_obs, sigma_obs
)
```

### Signal Selection (Bernoulli Wins)
```python
# 4 trading signals: Momentum, Mean-Rev, Carry, Breakout
alpha = np.ones(4)
beta = np.ones(4)

# Each trade
signals = np.random.beta(alpha, beta)
follow = np.argmax(signals)

# Trade outcome: 1 if profit, 0 if loss
if profit > 0:
    alpha[follow] += 1
else:
    beta[follow] += 1
```

## Key Intuitions

**Why sampling works:** Uncertain posteriors produce diverse samples (exploration). Confident posteriors produce concentrated samples (exploitation). No tuning needed.

**Why conjugacy matters:** Closed-form posteriors mean instant updates. No MCMC, no optimization, just arithmetic.

**Why Thompson beats UCB in practice:** Stochasticity helps in delayed feedback, non-stationarity, and contextual settings. UCB's determinism locks you into one arm until its bound shrinks.

**When to discount:** Commodity markets are non-stationary. Discount old data (γ ≈ 0.99) or use sliding windows to adapt to regime changes.

## Quick Reference: Python Libraries

```python
# Core
import numpy as np
from scipy.stats import beta, gamma, norm

# Plotting posterior evolution
import matplotlib.pyplot as plt
import seaborn as sns

# Bayesian modeling (if needed beyond conjugacy)
import pymc as pm
import arviz as az
```

## Debugging Tips

<div class="callout-danger">

**Danger:** Never deploy a bandit system without a kill switch and maximum allocation limits. An unconstrained bandit can allocate 100% of traffic/capital to a single arm, which creates catastrophic risk if the reward signal is noisy or delayed.

</div>


1. **Plot posteriors every N rounds** — See if they're concentrating
2. **Track exploration rate** — Should decay over time
3. **Check for degenerate posteriors** — Very small variance means over-confidence
4. **Compare to empirical means** — Posterior means should match sample means (eventually)
5. **Set random seed** — Reproducibility for debugging

## Further Reading

- Russo & Van Roy (2018): "Learning to Optimize via Information-Directed Sampling"
- Chapelle & Li (2011): "An Empirical Evaluation of Thompson Sampling"
- Connection to **Bayesian Commodity Forecasting** course for deeper Bayesian methods


---

## Conceptual Practice Questions

**Practice Question 1:** What is the primary tradeoff this approach makes compared to simpler alternatives?

**Practice Question 2:** Under what conditions would this approach fail or underperform?


---

## Cross-References

<a class="link-card" href="./01_thompson_sampling.md">
  <div class="link-card-title">01 Thompson Sampling</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_thompson_sampling.md">
  <div class="link-card-title">01 Thompson Sampling — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./02_posterior_updating.md">
  <div class="link-card-title">02 Posterior Updating</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_posterior_updating.md">
  <div class="link-card-title">02 Posterior Updating — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./03_thompson_vs_ucb.md">
  <div class="link-card-title">03 Thompson Vs Ucb</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_thompson_vs_ucb.md">
  <div class="link-card-title">03 Thompson Vs Ucb — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

