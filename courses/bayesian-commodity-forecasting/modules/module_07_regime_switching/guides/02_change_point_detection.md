# Bayesian Change Point Detection

> **Reading time:** ~11 min | **Module:** 7 — Regime Switching | **Prerequisites:** Module 3 State-Space Models


## In Brief

Change point detection identifies moments when a time series undergoes structural shifts in mean, variance, or dynamics. Bayesian methods provide full posterior distributions over change point locations and parameters, enabling uncertainty quantification critical for commodity market regime identification.

<div class="callout-insight">

<strong>Insight:</strong> **Structural breaks announce themselves after they happen.** The shale revolution didn't ring a bell in 2010—it became obvious years later. Bayesian change point detection quantifies "how sure are we a regime changed?" and "when did it happen?", preventing false alarms while catching real shifts.

</div>


<div class="callout-key">

<strong>Key Concept Summary:</strong> Change point detection identifies moments when a time series undergoes structural shifts in mean, variance, or dynamics.

</div>

---

## Formal Definition

### Piecewise Constant Model

**Setup:** Time series $y_1, ..., y_T$ with $K$ change points at unknown times $\tau_1, ..., \tau_K$.

**Regime parameters:** $\theta_0, \theta_1, ..., \theta_K$ (mean/variance in each regime)

**Likelihood:**
$$y_t \sim \begin{cases}
p(y_t | \theta_0) & t \leq \tau_1 \\
p(y_t | \theta_1) & \tau_1 < t \leq \tau_2 \\
\vdots \\
p(y_t | \theta_K) & t > \tau_K
\end{cases}$$

**Priors:**
$$\tau_k \sim \text{Discrete-Uniform}(1, T)$$
$$\theta_k \sim p(\theta)$$

**Goal:** Infer posterior $p(\tau, \theta | y)$

---

### Product Partition Model

**Alternative formulation:** Number of change points $K$ is unknown.

**Prior on partition:**
$$p(\rho) = \prod_{k=1}^{K+1} g(|\rho_k|)$$

where $\rho$ is a partition of $\{1, ..., T\}$ into $K+1$ blocks, and $g(n)$ is a cohesion function (e.g., $g(n) = n!$ favors fewer, larger blocks).

**Bayesian model selection:** Compare models with different $K$ via marginal likelihoods.

---

## Intuitive Explanation

### The Story Segmentation Problem

Imagine reading a history of oil markets:

```
1970s: OPEC embargo era (high, volatile prices)
1980s: Oil glut (collapse to $10/bbl)
1990s: Stable, moderate ($20-30/bbl)
2000s: China demand surge ($50-140/bbl)
2010s: Shale revolution (increased supply, lower prices)
2020s: ESG transition (uncertain future)
```

Each era has distinct characteristics (mean price, volatility). Change point detection finds the chapter breaks.

---

## Why Change Point Detection for Commodities?

### 1. Policy Regime Changes
<div class="callout-warning">

<strong>Warning:</strong> **OPEC:** Production quota decisions alter supply dynamics

</div>


**OPEC:** Production quota decisions alter supply dynamics
**Biofuel mandates:** US ethanol requirement links corn to oil (2005+)
**Russian export ban (2010):** Wheat market regime shift

**Example:** Pre/post-shale oil volatility


<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# WTI volatility regimes
pre_shale = oil_returns['2000':'2009'].std()  # ~0.03 daily
post_shale = oil_returns['2015':'2019'].std()  # ~0.02 daily

# Shale (starting ~2010-2012) reduced volatility
```

</div>
</div>

Change point detection identifies when this shift occurred and uncertainty around timing.

---

### 2. Technology Disruptions

**Hydraulic fracturing (fracking):** Unlocked shale oil/gas (2008+)
**Precision agriculture:** Changed yield variance (2010s)
**Renewable energy:** Reduced coal/gas demand (2015+)

**Challenge:** Technology adoption is gradual. When did the regime "actually" change?

---

### 3. Geopolitical Events

**Venezuelan collapse (2016-2019):** Removed ~2M bbl/day of heavy crude
**COVID-19 (2020):** Demand destruction, negative oil prices
**Ukraine war (2022):** European natural gas crisis

**Bayesian approach:** Distinguishes temporary shocks from permanent regime changes.

---

### 4. Climate Events

**Drought cycles:** Multi-year droughts alter agricultural price dynamics
**Hurricane seasons:** Repeated Gulf of Mexico disruptions
**El Niño/La Niña:** Predictable but irregular climate patterns

**Question:** Is this year's drought a one-off or a new climate regime?

---

## Code Implementation

### Basic Mean-Shift Detection


<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

# Simulated oil price data with regime change
np.random.seed(42)
n_obs = 200

# Regime 1: High volatility (2000-2009)
regime1 = np.random.normal(70, 10, 100)

# Regime 2: Lower volatility post-shale (2010-2019)
regime2 = np.random.normal(60, 5, 100)

prices = np.concatenate([regime1, regime2])

# Bayesian change point model
with pm.Model() as changepoint_model:
    # Prior on change point location
    tau = pm.DiscreteUniform('tau', lower=10, upper=n_obs-10)

    # Regime parameters (mean and std)
    mu_pre = pm.Normal('mu_pre', mu=70, sigma=10)
    mu_post = pm.Normal('mu_post', mu=60, sigma=10)

    sigma_pre = pm.HalfNormal('sigma_pre', sigma=10)
    sigma_post = pm.HalfNormal('sigma_post', sigma=10)

    # Regime indicator (piecewise constant)
    regime = pm.math.switch(tau >= np.arange(n_obs), 0, 1)

    # Mean and variance depend on regime
    mu = pm.Deterministic('mu',
                         pm.math.switch(regime, mu_post, mu_pre))

    sigma = pm.Deterministic('sigma',
                            pm.math.switch(regime, sigma_post, sigma_pre))

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=prices)

    # Sample
    trace = pm.sample(2000, tune=2000,
                     target_accept=0.95,
                     return_inferencedata=True)

# Analyze change point
tau_samples = trace.posterior['tau'].values.flatten()
print(f"Change point: {tau_samples.mean():.0f} (true: 100)")
print(f"95% HDI: {az.hdi(trace, var_names=['tau'])['tau'].values}")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Prices with inferred change point
axes[0].plot(prices, 'ko', alpha=0.5, markersize=3)
axes[0].axvline(100, color='g', linestyle='--', label='True Change Point')
axes[0].axvline(tau_samples.mean(), color='r', linestyle='--', label='Inferred Change Point')
axes[0].fill_betweenx([prices.min(), prices.max()],
                      np.percentile(tau_samples, 2.5),
                      np.percentile(tau_samples, 97.5),
                      alpha=0.2, color='r', label='95% HDI')
axes[0].set_ylabel('Price')
axes[0].set_title('Oil Prices with Change Point Detection')
axes[0].legend()

# Change point posterior
axes[1].hist(tau_samples, bins=50, density=True, alpha=0.7)
axes[1].axvline(100, color='g', linestyle='--', linewidth=2)
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Posterior Density')
axes[1].set_title('Change Point Posterior Distribution')

plt.tight_layout()
plt.show()
```

</div>
</div>

---

## Multiple Change Points

### Unknown Number of Change Points
<div class="callout-insight">

<strong>Insight:</strong> with pm.Model() as multiple_changepoints:

</div>



<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
with pm.Model() as multiple_changepoints:
    n_obs = 300
    max_changepoints = 3  # Upper bound

    # Priors for up to 3 change points (some may not be "active")
    tau = pm.DiscreteUniform('tau', lower=20, upper=n_obs-20, shape=max_changepoints)

    # Sort change points (enforce ordering)
    tau_sorted = pm.Deterministic('tau_sorted', pm.math.sort(tau))

    # Regime means (4 potential regimes for 3 change points)
    mu_regimes = pm.Normal('mu_regimes', mu=60, sigma=15, shape=max_changepoints+1)
    sigma_regimes = pm.HalfNormal('sigma_regimes', sigma=5, shape=max_changepoints+1)

    # Determine which regime each observation belongs to
    regime_idx = (
        (np.arange(n_obs)[:, None] > tau_sorted[None, :]).sum(axis=1)
    )

    # Mean and std for each observation
    mu = mu_regimes[regime_idx]
    sigma = sigma_regimes[regime_idx]

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=oil_prices)

    # Sample (harder with discrete parameters)
    trace_multi = pm.sample(1000, tune=2000,
                           target_accept=0.95,
                           return_inferencedata=True)

# Check if all 3 change points are supported
tau_post = trace_multi.posterior['tau_sorted'].values
print("Change points (mean):", tau_post.mean(axis=(0, 1)))
print("Change points (std):", tau_post.std(axis=(0, 1)))

# Large std → change point not well-supported (may not exist)
```

</div>
</div>

---

## Variance Change Point Detection

### Detecting Volatility Regime Changes

Useful for identifying risk regime shifts (e.g., pre/post-crisis volatility).


<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
with pm.Model() as variance_changepoint:
    n_obs = 250
    returns = oil_returns[:n_obs]  # Demeaned returns

    # Change point
    tau = pm.DiscreteUniform('tau', lower=20, upper=n_obs-20)

    # Volatility regimes
    sigma_pre = pm.HalfNormal('sigma_pre', sigma=0.05)
    sigma_post = pm.HalfNormal('sigma_post', sigma=0.05)

    # Regime-dependent variance
    sigma = pm.math.switch(tau >= np.arange(n_obs), sigma_pre, sigma_post)

    # Likelihood (mean = 0 for returns)
    y_obs = pm.Normal('y_obs', mu=0, sigma=sigma, observed=returns)

    trace_var = pm.sample(2000, tune=2000,
                         target_accept=0.95,
                         return_inferencedata=True)

# Interpret: When did volatility regime change?
print(f"Volatility change point: {trace_var.posterior['tau'].mean():.0f}")
print(f"Pre-change vol: {trace_var.posterior['sigma_pre'].mean():.4f}")
print(f"Post-change vol: {trace_var.posterior['sigma_post'].mean():.4f}")
```

</div>
</div>

---

## Online Change Point Detection

### Recursive Bayesian Updating
<div class="callout-key">

<strong>Key Point:</strong> For real-time detection (new data arrives daily).

</div>


For real-time detection (new data arrives daily).

**Algorithm (Bayesian Online Change Point Detection):**

1. Maintain distribution over "run length" $r_t$ (time since last change point)
2. At each time step, either:
   - Continue current run ($r_{t+1} = r_t + 1$)
   - Start new run ($r_{t+1} = 0$) with probability $\pi(r_t)$

**Run-length posterior:**
$$p(r_t | y_{1:t}) \propto \sum_{r_{t-1}} p(r_t | r_{t-1}) p(y_t | r_t, y_{1:t-1}) p(r_{t-1} | y_{1:t-1})$$


<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np
from scipy import stats

def online_changepoint_detection(data, hazard_rate=0.01, mu0=0, kappa0=1, alpha0=1, beta0=1):
    """
    Bayesian online change point detection (Gaussian data).

    Parameters
    ----------
    data : array
        Time series observations
    hazard_rate : float
        Prior probability of change point at each time step
    mu0, kappa0, alpha0, beta0 : float
        Normal-Gamma prior parameters

    Returns
    -------
    run_length_posterior : array (T, T)
        Posterior over run length at each time
    """
    T = len(data)
    R = np.zeros((T + 1, T + 1))  # Run length distribution
    R[0, 0] = 1

    # Sufficient statistics for each run length
    mu = np.zeros(T + 1)
    kappa = np.zeros(T + 1)
    alpha = np.zeros(T + 1)
    beta = np.zeros(T + 1)

    mu[0] = mu0
    kappa[0] = kappa0
    alpha[0] = alpha0
    beta[0] = beta0

    for t in range(T):
        # Predictive probability for each run length
        df = 2 * alpha[:t+1]
        pred_mean = mu[:t+1]
        pred_std = np.sqrt(beta[:t+1] * (kappa[:t+1] + 1) / (alpha[:t+1] * kappa[:t+1]))

        pred_prob = stats.t.pdf(data[t], df, pred_mean, pred_std)

        # Growth probability (no change point)
        R[1:t+2, t+1] = R[:t+1, t] * pred_prob * (1 - hazard_rate)

        # Change point probability
        R[0, t+1] = np.sum(R[:t+1, t] * pred_prob * hazard_rate)

        # Normalize
        R[:, t+1] /= np.sum(R[:, t+1])

        # Update sufficient statistics
        mu[1:t+2] = (kappa[:t+1] * mu[:t+1] + data[t]) / (kappa[:t+1] + 1)
        kappa[1:t+2] = kappa[:t+1] + 1
        alpha[1:t+2] = alpha[:t+1] + 0.5
        beta[1:t+2] = beta[:t+1] + (kappa[:t+1] * (data[t] - mu[:t+1])**2) / (2 * (kappa[:t+1] + 1))

        # Reset for new run
        mu[0] = mu0
        kappa[0] = kappa0
        alpha[0] = alpha0
        beta[0] = beta0

    return R

# Apply to oil prices
R = online_changepoint_detection(oil_prices, hazard_rate=0.01)

# Plot run-length posterior
plt.figure(figsize=(12, 6))
plt.imshow(np.log(R + 1e-10), cmap='hot', aspect='auto', origin='lower')
plt.xlabel('Time')
plt.ylabel('Run Length')
plt.title('Online Change Point Detection (Log Posterior)')
plt.colorbar(label='Log Posterior')
plt.show()

# Detect change points (where run length resets to 0)
change_prob = R[0, :]
changepoint_threshold = 0.3
detected_changepoints = np.where(change_prob > changepoint_threshold)[0]
print(f"Detected change points: {detected_changepoints}")
```

</div>
</div>

**Advantages:**
- Real-time detection (no need to reprocess all data)
- Full posterior over change point location
- Automatic adaptation to new regimes

---

## Model Comparison: Is There a Change Point?

### Bayes Factor for Change Point Model vs. No-Change Model


<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# Model 1: No change point (constant mean/variance)
with pm.Model() as no_change:
    mu = pm.Normal('mu', 65, 10)
    sigma = pm.HalfNormal('sigma', 8)
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=prices)

    trace_no_change = pm.sample(1000, tune=1000, return_inferencedata=True)

# Model 2: Change point model (from earlier)
# (already fitted as 'changepoint_model')

# Compare via LOO
comparison = az.compare({
    'No Change': trace_no_change,
    'Change Point': trace
})
print(comparison)

# Interpretation:
# - If change point model has much lower LOO → evidence for regime change
# - If similar LOO → insufficient evidence
```

</div>
</div>

---

## Change Point Detection for Fundamentals

### Detect Structural Breaks in Inventory-Price Relationship
<div class="callout-warning">

<strong>Warning:</strong> with pm.Model() as inventory_changepoint:

</div>



<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
with pm.Model() as inventory_changepoint:
    n_obs = 150
    inventory = inventory_data[:n_obs]
    prices = price_data[:n_obs]

    # Change point in relationship
    tau = pm.DiscreteUniform('tau', lower=20, upper=n_obs-20)

    # Pre-change relationship: Price = a1 + b1 * Inventory
    a_pre = pm.Normal('a_pre', mu=80, sigma=10)
    b_pre = pm.Normal('b_pre', mu=-20, sigma=5)  # Negative (more inventory → lower price)
    sigma_pre = pm.HalfNormal('sigma_pre', sigma=5)

    # Post-change relationship (coefficient may change)
    a_post = pm.Normal('a_post', mu=80, sigma=10)
    b_post = pm.Normal('b_post', mu=-20, sigma=5)
    sigma_post = pm.HalfNormal('sigma_post', sigma=5)

    # Regime-dependent regression
    a = pm.math.switch(tau >= np.arange(n_obs), a_pre, a_post)
    b = pm.math.switch(tau >= np.arange(n_obs), b_pre, b_post)
    sigma = pm.math.switch(tau >= np.arange(n_obs), sigma_pre, sigma_post)

    mu = a + b * inventory

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=prices)

    trace_inv = pm.sample(1500, tune=1500,
                         target_accept=0.95,
                         return_inferencedata=True)

# Interpretation: When did inventory-price relationship change?
# Example: Shale revolution weakened inventory signal (more flexible supply)
```

</div>
</div>

---

## Common Pitfalls

### 1. Over-Detection (False Positives)

Every market fluctuation looks like a regime change.

**Fix:**
- Use informative prior on hazard rate (change points are rare)
- Require substantial evidence (high posterior probability)
- Model comparison (change vs. no-change)

---

### 2. Boundary Effects

Change points near start/end of sample are poorly identified.

**Fix:**
- Restrict $\tau \sim \text{DiscreteUniform}(\text{20}, T-\text{20})$
- Don't trust change points in first/last 10% of data

---

### 3. Gradual Transitions

Shale revolution wasn't a single day—it was a 5-year transition.

**Fix:** Use smooth transition models (logistic regime change):
$$\mu_t = \mu_1 + (\mu_2 - \mu_1) \cdot \frac{1}{1 + \exp(-\lambda(t - \tau))}$$

---

### 4. Confusing Shock with Regime Change

COVID-19 caused huge oil price crash (temporary shock), not permanent regime change.

**Check:** Model with/without change point. If temporary, no-change model wins.

---

## Connections

**Builds on:**
- Module 1: Bayesian model comparison
- Module 3: State space models (piecewise trends)
- Module 7: Hidden Markov models (probabilistic regime switching)

**Leads to:**
- Regime-switching forecasts (use detected change points to segment data)
- Conditional forecasts ("if regime persists" vs. "if regime changes")

**Related concepts:**
- Structural break tests (Chow test, Bai-Perron)
- Hidden Markov models (continuous regime switching)

---

## Practice Problems

### Problem 1
You suspect a change point in natural gas volatility around 2006 (when Hurricanes Katrina/Rita hit). Design a Bayesian model to test this hypothesis and quantify uncertainty.

### Problem 2
Implement online change point detection for corn prices. What hazard rate prior is appropriate? (How often do structural breaks occur in agriculture?)

### Problem 3
The WTI-Brent spread was near zero before 2010, then widened to $10-20/bbl (US shale, export restrictions). Detect the change point and quantify the spread shift.

### Problem 4
Compare three models for copper prices:
1. No change point
2. One change point (unknown location)
3. Two change points (unknown locations)

Use LOO-CV. Which is best?

### Problem 5
Design a "gradual transition" change point model for the introduction of biofuel mandates (phased in 2005-2010). How does this differ from an abrupt change point?

---


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Formal Definition" and why it matters in practice.

2. Given a real-world scenario involving bayesian change point detection, what would be your first three steps to apply the techniques from this guide?

</div>

## Further Reading

1. **Chib, S. (1998)**. "Estimation and Comparison of Multiple Change-Point Models." *Journal of Econometrics*, 86(2), 221-241.
   - Bayesian change point methods

2. **Adams, R.P. & MacKay, D.J.C. (2007)**. "Bayesian Online Changepoint Detection." arXiv:0710.3742.
   - Online detection algorithm

3. **Carvalho, C.M., et al. (2010)**. "Particle Learning and Smoothing." *Statistical Science*, 25(1), 88-106.
   - Sequential Monte Carlo for change points

4. **Pettit, L.I. (1979)**. "A Non-Parametric Approach to the Change-Point Problem." *Applied Statistics*, 28(2), 126-135.
   - Classic non-parametric test

---

*"Regimes change silently. Bayesian change point detection listens for the whisper before the shout."*

---

## Cross-References

<a class="link-card" href="./02_change_point_detection_slides.md">
  <div class="link-card-title">Companion Slide Deck</div>
  <div class="link-card-description">Visual presentation covering the key concepts from this guide.</div>
</a>

<a class="link-card" href="../notebooks/01_hmm_from_scratch.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>
