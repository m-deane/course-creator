# Posterior Updating with Conjugate Priors

> **Reading time:** ~16 min | **Module:** 02 — Bayesian Bandits | **Prerequisites:** Module 1


## In Brief


<div class="callout-key">

**Key Concept Summary:** Posterior updating is how Bayesian methods learn from data: combine your prior belief with new evidence using Bayes' rule to produce a posterior belief. With conjugate priors, this update has a clo...

</div>

Posterior updating is how Bayesian methods learn from data: combine your prior belief with new evidence using Bayes' rule to produce a posterior belief. With conjugate priors, this update has a closed-form solution — no MCMC needed. For Thompson Sampling, conjugate priors make posterior updates instant and exact.

> 💡 **Key Insight:** **Bayes' rule:** Posterior ∝ Likelihood × Prior

With conjugate priors, the posterior has the same functional form as the prior, just with updated parameters. This means learning is as simple as incrementing counters.

**Beta-Bernoulli example:**
- Prior: Beta(α, β)
- Observe k successes in n trials
- Posterior: Beta(α + k, β + n - k)

That's it. No integrals, no sampling, no numerical optimization. Just arithmetic.

## Visual Explanation

```
Prior Belief: Beta(2, 2)
     /\
    /  \
   /    \
  /      \
 ----------------
0.0  0.5  1.0

Observe: 3 successes, 1 failure

Posterior: Beta(2+3, 2+1) = Beta(5, 3)
        /\
       /  \
      /    \
     /      \
----/--------\----
0.0  0.5  1.0

More peaked, shifted right
(More confident, higher mean)

```

<div class="callout-insight">

**Insight:** The core insight of bandit algorithms is that learning and earning are not separate phases. Every observation contributes to both understanding which option is best and generating value from the best option.

</div>

```
After 100 observations: 58 successes, 42 failures
Posterior: Beta(60, 44)
         |█|
    -----|-----
     0.5  0.6

Very peaked, tight around true value
(High confidence, low uncertainty)
```

**Sequential updating:** Each observation updates the posterior, which becomes the prior for the next observation. Beta(2,2) → observe success → Beta(3,2) → observe failure → Beta(3,3) → ...

## Formal Definition

<div class="callout-warning">

**Warning:** Bandit algorithms assume the reward distributions are stationary (or slowly changing). In commodity markets, regime shifts can make a historically optimal arm suddenly suboptimal. Always implement change detection alongside your bandit.

</div>


**Conjugate Prior:** A prior p(θ) is conjugate to likelihood p(D|θ) if the posterior p(θ|D) has the same functional form as the prior.

**Common Conjugate Pairs:**

| Likelihood | Conjugate Prior | Posterior Update |
|------------|-----------------|------------------|
| Bernoulli(θ) | Beta(α, β) | Beta(α + k, β + n - k) |
| Normal(μ, σ²) known σ² | Normal(μ₀, τ²) | Normal(μ₁, τ₁²) with precision update |
| Poisson(λ) | Gamma(α, β) | Gamma(α + Σxᵢ, β + n) |
| Exponential(λ) | Gamma(α, β) | Gamma(α + n, β + Σxᵢ) |

**Beta-Bernoulli (most common for bandits):**

Prior: θ ~ Beta(α₀, β₀)

Likelihood: k successes in n trials with probability θ

Posterior:
```
θ | data ~ Beta(α₀ + k, β₀ + n - k)
```

Interpretation:
- α = successes + prior successes
- β = failures + prior failures
- n = α + β - 2 (for Beta(1,1) prior) = total observations

**Normal-Normal (for continuous rewards like returns):**

Prior: μ ~ Normal(μ₀, σ₀²)

Likelihood: xᵢ ~ Normal(μ, σ²) for known σ²

Posterior:
```
μ | data ~ Normal(μ₁, σ₁²)

where:
τ₀ = 1/σ₀² (prior precision)
τ = 1/σ² (data precision)
τ₁ = τ₀ + n·τ (posterior precision)

μ₁ = (τ₀·μ₀ + n·τ·x̄) / τ₁

σ₁² = 1/τ₁
```

Interpretation: Posterior mean is a precision-weighted average of prior mean and sample mean.

## Intuitive Explanation

**It's like updating your opinion when you see evidence — strong evidence moves your belief more than weak evidence.**

Imagine you believe a commodity trading signal works 50% of the time (prior: Beta(1,1) = uniform, or Beta(10,10) = confident about 50%).

**Weak prior (Beta(1,1)):** "I have no idea, could be anything."
- See 3 successes out of 5 trades → Posterior: Beta(4, 3)
- Your belief shifts strongly toward 57% (4/(4+3))

**Strong prior (Beta(10,10)):** "I'm pretty sure it's 50%."
- See 3 successes out of 5 trades → Posterior: Beta(13, 12)
- Your belief barely budges: 52% (13/(13+12))

The strong prior acts like you've already seen 18 trades (10-1 + 10-1). Five new trades don't move the needle much.

**Commodity context:** You're updating beliefs about a commodity's expected return. Each week's return is new data. With weak priors, early weeks dominate your belief. With strong priors (e.g., "I know the long-run mean is zero"), short-term data has less impact.

**Key insight:** Prior strength = effective sample size. Beta(α, β) acts like you've already seen α + β - 2 observations.

## Code Implementation

### Beta-Bernoulli Update


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

class BetaBernoulli:
    def __init__(self, alpha=1, beta_param=1):
        self.alpha = alpha
        self.beta = beta_param

    def update(self, successes, failures):
        self.alpha += successes
        self.beta += failures

    def sample(self, n=1):
        return beta.rvs(self.alpha, self.beta, size=n)

    def mean(self):
        return self.alpha / (self.alpha + self.beta)

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        x = np.linspace(0, 1, 200)
        y = beta.pdf(x, self.alpha, self.beta)
        ax.plot(x, y, label=f'Beta({self.alpha}, {self.beta})')
        ax.axvline(self.mean(), ls='--', alpha=0.5)
        return ax
```

</div>
</div>

### Normal-Normal Update


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
class NormalNormal:
    def __init__(self, mu_0=0, sigma_0=1, sigma_data=1):
        self.mu = mu_0
        self.sigma = sigma_0
        self.sigma_data = sigma_data

    def update(self, observations):
        tau_0 = 1 / self.sigma**2
        tau = 1 / self.sigma_data**2
        n = len(observations)
        x_bar = np.mean(observations)

        tau_1 = tau_0 + n * tau
        self.mu = (tau_0*self.mu + n*tau*x_bar) / tau_1
        self.sigma = np.sqrt(1 / tau_1)

    def sample(self, n=1):
        return np.random.normal(self.mu, self.sigma, size=n)
```

</div>

## Common Pitfalls

### Pitfall 1: Confusing Beta parameters with mean/variance
**Why it happens:** Beta(α, β) doesn't directly give you mean and variance.

**How to avoid:**
- Mean = α / (α + β)
- Variance = αβ / [(α + β)² (α + β + 1)]
- To get Beta with mean μ and "strength" n: α = nμ, β = n(1-μ)

**Example:** Want a prior centered at 0.6 with strength 10?
- α = 10 × 0.6 = 6
- β = 10 × 0.4 = 4
- Beta(6, 4) has mean 0.6 and is like seeing 8 observations (6+4-2)

### Pitfall 2: Forgetting to handle unknown variance in Gaussian case
**Why it happens:** Normal-Normal conjugacy assumes known observation variance σ².

**How to avoid:** If σ² is unknown, use Normal-Gamma conjugacy (prior on both μ and σ²). For Thompson Sampling, empirical Bayes (plug in sample variance) often works fine.

**Commodity example:** Commodity returns have time-varying volatility. Either use rolling window for σ̂² or model volatility explicitly.

### Pitfall 3: Accumulating too much old data in non-stationary settings
**Why it happens:** Posteriors keep adding evidence forever, even when the world changes.

**How to avoid:**
- Exponential discounting: α ← α × γ each period (e.g., γ = 0.99)
- Sliding window: only track last N observations
- Reset on change detection

**Commodity example:** A trading signal that worked in 2020 may fail in 2023. Discount old evidence or detect regime shifts.

## Connections

<div class="callout-danger">

**Danger:** Never deploy a bandit system without a kill switch and maximum allocation limits. An unconstrained bandit can allocate 100% of traffic/capital to a single arm, which creates catastrophic risk if the reward signal is noisy or delayed.



**Builds on:**
- Bayesian Commodity Forecasting: Same posterior updating framework
- Basic probability: Bayes' rule, conjugate families

**Leads to:**
- Thompson Sampling: Use posteriors to guide exploration
- Bayesian regression: Extend to contextual bandits (Module 3)
- State-space models: Time-varying parameters with Kalman filtering

**Mathematical foundations:**
- Exponential families: All conjugate priors arise from exponential family structure
- Sufficient statistics: α and β fully summarize all Bernoulli observations
- Sequential Bayes: Yesterday's posterior is today's prior



## Practice Problems

### Problem 1: Hand calculation
Start with Beta(2, 3) prior. Observe sequence: success, success, failure, success.

**Calculate:**
a) Posterior after each observation
b) Posterior mean after each observation
c) How many more failures would it take to shift the mean below 0.5?

### Problem 2: Prior strength experiment
Implement Beta-Bernoulli updating. Start with three different priors:
- Beta(1, 1) — uniform
- Beta(5, 5) — moderately informative
- Beta(20, 20) — very informative

Generate 100 observations from Bernoulli(0.6). Track posterior mean after each observation for all three priors. Plot convergence.

**Question:** How many observations until all three posteriors agree (within 0.05)?

### Problem 3: Gaussian commodity returns
You're modeling weekly returns of WTI crude oil. Assume returns ~ Normal(μ, σ²) with σ = 0.05 (from historical data).

Prior: μ ~ Normal(0, 0.02)

Observe 10 weeks of returns: [0.02, -0.01, 0.03, 0.01, -0.02, 0.04, 0.00, 0.02, -0.01, 0.03]

**Calculate:**
a) Posterior distribution of μ
b) Posterior mean and standard deviation
c) 95% credible interval for μ
d) Probability that μ > 0 (use posterior)

### Problem 4: Connection to Bayesian Commodity Forecasting
In the Bayesian Commodity Forecasting course, you learned to forecast prices using state-space models with Bayesian parameter estimation.

**Question:** How does Thompson Sampling's posterior updating differ from forecasting?

**Answer framework:**
- Forecasting goal: Predict future price
- Bandit goal: Choose best action
- Connection: Both maintain beliefs, update with data, quantify uncertainty
- Difference: Bandits act to learn (exploration), forecasts passively predict

**Extension:** Could you use commodity price forecasts as prior beliefs for a Thompson Sampling portfolio allocator? How would you convert price forecast uncertainty into reward distribution uncertainty?


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

<a class="link-card" href="./03_thompson_vs_ucb.md">
  <div class="link-card-title">03 Thompson Vs Ucb</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_thompson_vs_ucb.md">
  <div class="link-card-title">03 Thompson Vs Ucb — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./cheatsheet.md">
  <div class="link-card-title">Cheatsheet</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./cheatsheet.md">
  <div class="link-card-title">Cheatsheet — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

