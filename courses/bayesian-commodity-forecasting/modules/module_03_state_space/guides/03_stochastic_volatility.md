# Stochastic Volatility Models for Commodities

> **Reading time:** ~9 min | **Module:** 3 — State-Space Models | **Prerequisites:** Module 2 Commodity Data


## In Brief

Stochastic volatility (SV) models treat volatility as a latent time-varying process rather than a deterministic function. This captures the clustering and persistence of volatility observed in commodity markets while providing probabilistic forecasts of future risk.

<div class="callout-insight">

<strong>Insight:</strong> **Volatility has memory.** High volatility periods cluster together (calm markets follow calm markets, chaotic markets follow chaotic markets). Stochastic volatility models this persistence through a latent AR(1) process on log-variance, enabling both backward inference (what was the volatility?) and forward prediction (what will it be?).

</div>

## Formal Definition

### Standard Stochastic Volatility Model

**Observation Equation:**
$$y_t = \exp(h_t / 2) \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, 1)$$

**Volatility Process (log-variance):**
$$h_{t+1} = \mu + \phi(h_t - \mu) + \sigma_\eta \eta_t, \quad \eta_t \sim \mathcal{N}(0, 1)$$

**Initial State:**
$$h_1 \sim \mathcal{N}\left(\mu, \frac{\sigma_\eta^2}{1 - \phi^2}\right)$$

### Parameter Interpretation

| Parameter | Range | Interpretation |
|-----------|-------|----------------|
| $\mu$ | $\mathbb{R}$ | Long-run log-variance level |
| $\phi$ | $(-1, 1)$ | Persistence (AR coefficient) |
| $\sigma_\eta$ | $\mathbb{R}^+$ | Volatility of volatility |
| $h_t$ | $\mathbb{R}$ | Log-variance at time t |
| $\sigma_t^2 = \exp(h_t)$ | $\mathbb{R}^+$ | Variance at time t |

**Stationary variance:** $\mathbb{E}[\sigma_t^2] = \exp(\mu + \sigma_\eta^2 / (2(1-\phi^2)))$


<div class="callout-key">

<strong>Key Concept Summary:</strong> Stochastic volatility (SV) models treat volatility as a latent time-varying process rather than a deterministic function.

</div>

---

## Intuitive Explanation

### Why Log-Variance?

Variance must be positive. Working with $h_t = \log(\sigma_t^2)$ ensures:
1. $\sigma_t^2 = \exp(h_t) > 0$ automatically
2. AR(1) dynamics on $h_t$ can range over $\mathbb{R}$
3. Multiplicative shocks become additive in log-space

### The Two Layers

**Layer 1: Hidden Volatility (State)**
```
Low vol → Low vol → Spike! → High vol → High vol → Decay...
  h₁=-2    h₂=-2.1   h₃=1     h₄=0.8    h₅=0.5    h₆=0.1
```

**Layer 2: Returns (Observations)**
```
Small   Small    HUGE!    Large    Large    Moderate
 y₁      y₂       y₃       y₄       y₅       y₆
```

The observations are scaled by $\exp(h_t/2)$, so large $h_t$ produces large swings in $y_t$.

---

## Why Stochastic Volatility for Commodities?

### 1. Inventory Shocks

Low inventory states produce higher volatility (supply disruptions matter more). SV models capture this without requiring inventory data.

**Example:** Crude oil volatility spiked during Hurricane Katrina (2005), Syrian civil war (2012), COVID demand crash (2020).

### 2. Weather Events

Agricultural commodities have volatility clusters around planting/harvest:
- Quiet winter months (low $h_t$)
- Volatile growing season (high $h_t$)

### 3. Regime Changes

Structural breaks in volatility (e.g., shale revolution lowering oil vol) appear as shifts in $\mu$.

### 4. Risk Management

Options pricing, VaR calculation, and position sizing require volatility forecasts with uncertainty.

---

## SV vs. GARCH

| Feature | GARCH | Stochastic Volatility |
|---------|-------|----------------------|
| Volatility process | Deterministic given past | Stochastic (latent) |
| Inference | Maximum likelihood | Bayesian (full posterior) |
| Forecasts | Point estimates | Full predictive distribution |
| Flexibility | Limited functional forms | Arbitrary prior/dynamics |
| Computation | Fast | Requires MCMC/VI |

**When to use GARCH:** High-frequency data, need fast estimation, sufficient sample size for MLE.

**When to use SV:** Need uncertainty quantification, have informative priors, want flexible specifications.

---

## Code Implementation

### Basic Stochastic Volatility Model in PyMC


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

# Simulate SV data
np.random.seed(42)
n_obs = 300
mu = -1.0
phi = 0.95
sigma_eta = 0.25

# Generate log-volatility
h = np.zeros(n_obs)
h[0] = np.random.normal(mu, sigma_eta / np.sqrt(1 - phi**2))
for t in range(1, n_obs):
    h[t] = mu + phi * (h[t-1] - mu) + sigma_eta * np.random.normal()

# Generate returns
y = np.exp(h / 2) * np.random.normal(size=n_obs)

# Demean returns (SV typically for demeaned data)
y = y - y.mean()

# Build PyMC model
with pm.Model() as sv_model:
    # Priors for SV parameters
    mu = pm.Normal('mu', mu=0, sigma=5)
    phi_raw = pm.Beta('phi_raw', alpha=20, beta=1.5)  # Concentrate near 1
    phi = pm.Deterministic('phi', 2 * phi_raw - 1)    # Map to (-1, 1)
    sigma_eta = pm.HalfNormal('sigma_eta', sigma=1)

    # Initial log-volatility (stationary distribution)
    h_init = pm.Normal('h_init',
                       mu=mu,
                       sigma=sigma_eta / pm.math.sqrt(1 - phi**2))

    # Log-volatility innovations
    eta = pm.Normal('eta', mu=0, sigma=1, shape=n_obs-1)

    # Build log-volatility path
    h = pm.Deterministic(
        'h',
        pm.math.concatenate([
            [h_init],
            h_init + pm.math.cumsum(
                mu * (1 - phi) + phi * pm.math.concatenate([[0], eta[:-1]]) + sigma_eta * eta
            )
        ])
    )

    # Observation likelihood
    y_obs = pm.Normal('y_obs',
                      mu=0,
                      sigma=pm.math.exp(h / 2),
                      observed=y)

    # Sample
    trace = pm.sample(1000, tune=2000,
                     target_accept=0.95,
                     return_inferencedata=True)

# Diagnostics
print(az.summary(trace, var_names=['mu', 'phi', 'sigma_eta']))

# Plot results
fig, axes = plt.subplots(3, 1, figsize=(12, 9))

# Volatility inference
h_post = trace.posterior['h'].mean(dim=['chain', 'draw']).values
h_hdi = az.hdi(trace, var_names=['h'])['h']

axes[0].plot(np.exp(h / 2), 'k-', alpha=0.3, label='True Volatility')
axes[0].plot(np.exp(h_post / 2), 'r-', label='Posterior Mean')
axes[0].fill_between(range(n_obs),
                      np.exp(h_hdi[:, 0] / 2),
                      np.exp(h_hdi[:, 1] / 2),
                      alpha=0.3, color='r', label='94% HDI')
axes[0].set_ylabel('Volatility (σ)')
axes[0].set_title('Stochastic Volatility Inference')
axes[0].legend()

# Returns with volatility bands
axes[1].plot(y, 'o', markersize=2, alpha=0.5)
axes[1].plot(2 * np.exp(h_post / 2), 'r--', label='+2σ band')
axes[1].plot(-2 * np.exp(h_post / 2), 'r--')
axes[1].set_ylabel('Returns')
axes[1].set_title('Returns with Inferred Volatility Bands')
axes[1].legend()

# Persistence parameter
az.plot_posterior(trace, var_names=['phi'], ax=axes[2])
axes[2].set_title('Volatility Persistence (φ)')

plt.tight_layout()
plt.show()
```

</div>
</div>

---

## Commodity-Specific Extensions

### 1. Leverage Effect
<div class="callout-warning">

<strong>Warning:</strong> Negative returns increase volatility more than positive returns (especially in energy).

</div>


Negative returns increase volatility more than positive returns (especially in energy).


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
with pm.Model() as sv_leverage:
    # Standard SV parameters
    mu = pm.Normal('mu', mu=0, sigma=5)
    phi = pm.Beta('phi', alpha=20, beta=1.5) * 2 - 1
    sigma_eta = pm.HalfNormal('sigma_eta', sigma=1)

    # Leverage parameter
    rho = pm.Uniform('rho', lower=-1, upper=0)  # Negative correlation

    h_init = pm.Normal('h_init', mu=mu,
                       sigma=sigma_eta / pm.math.sqrt(1 - phi**2))

    # Correlated innovations
    z = pm.Normal('z', mu=0, sigma=1, shape=n_obs)
    eta = rho * z[:-1] + pm.math.sqrt(1 - rho**2) * pm.Normal('eta_indep', 0, 1, shape=n_obs-1)

    # Build log-vol with leverage
    h = pt.concatenate([[h_init],
                        h_init + pm.math.cumsum(mu * (1-phi) + phi * eta + sigma_eta * eta)])

    # Returns use same innovations as vol (creates correlation)
    y_obs = pm.Normal('y_obs', mu=0, sigma=pm.math.exp(h/2) * z, observed=y)
```

</div>
</div>

### 2. Seasonal Volatility

Natural gas volatility peaks in summer (cooling) and winter (heating).


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
with pm.Model() as sv_seasonal:
    # Time indices (assume monthly data)
    months = np.arange(n_obs) % 12

    # Seasonal pattern in log-vol mean
    seasonal_mu = pm.Normal('seasonal_mu', mu=0, sigma=1, shape=12)
    mu_t = seasonal_mu[months]

    # AR(1) around seasonal mean
    phi = pm.Beta('phi', alpha=20, beta=1.5) * 2 - 1
    sigma_eta = pm.HalfNormal('sigma_eta', sigma=1)

    # ... rest similar to basic model
```

</div>
</div>

### 3. Heavy Tails (Student-t Returns)

Commodity returns have fatter tails than normal.


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
with pm.Model() as sv_student:
    mu = pm.Normal('mu', mu=0, sigma=5)
    phi = pm.Beta('phi', alpha=20, beta=1.5) * 2 - 1
    sigma_eta = pm.HalfNormal('sigma_eta', sigma=1)

    # Degrees of freedom (heavy tails)
    nu = pm.Gamma('nu', alpha=2, beta=0.1)  # Expect low nu (heavy tails)

    # Standard SV dynamics for h
    # ...

    # Student-t observations
    y_obs = pm.StudentT('y_obs', nu=nu, mu=0, sigma=pm.math.exp(h/2), observed=y)
```

</div>
</div>

---

## Forecasting Volatility

### One-Step-Ahead Forecast
<div class="callout-key">

<strong>Key Point:</strong> Given $h_t$, the predictive distribution for $h_{t+1}$:

</div>


Given $h_t$, the predictive distribution for $h_{t+1}$:

$$h_{t+1} | h_t \sim \mathcal{N}(\mu + \phi(h_t - \mu), \sigma_\eta^2)$$

Volatility forecast:
$$\sigma_{t+1}^2 | h_t \sim \text{LogNormal}(\mu + \phi(h_t - \mu), \sigma_\eta^2)$$

### Multi-Step Forecast

$$h_{t+k} | h_t \sim \mathcal{N}\left(\mu + \phi^k(h_t - \mu), \frac{\sigma_\eta^2(1 - \phi^{2k})}{1 - \phi^2}\right)$$

As $k \to \infty$, this converges to the stationary distribution.


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python

# Forecast volatility from PyMC trace
def forecast_volatility(trace, last_h, periods=20):
    """
    Generate volatility forecasts.

    Parameters
    ----------
    trace : InferenceData
        PyMC trace with mu, phi, sigma_eta
    last_h : float
        Last observed log-volatility
    periods : int
        Forecast horizon
    """
    mu_samples = trace.posterior['mu'].values.flatten()
    phi_samples = trace.posterior['phi'].values.flatten()
    sigma_eta_samples = trace.posterior['sigma_eta'].values.flatten()

    n_samples = len(mu_samples)
    h_forecast = np.zeros((n_samples, periods))

    for i in range(n_samples):
        h_t = last_h
        for t in range(periods):
            h_t = (mu_samples[i] + phi_samples[i] * (h_t - mu_samples[i]) +
                   sigma_eta_samples[i] * np.random.normal())
            h_forecast[i, t] = h_t

    # Convert to volatility
    vol_forecast = np.exp(h_forecast / 2)

    return pd.DataFrame({
        'mean': vol_forecast.mean(axis=0),
        'lower': np.percentile(vol_forecast, 2.5, axis=0),
        'upper': np.percentile(vol_forecast, 97.5, axis=0)
    })
```

</div>
</div>

---

## Model Comparison

Compare SV specifications using LOO-CV:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
with pm.Model() as sv_basic:
    # Basic SV (no leverage, no heavy tails)
    # ... build model
    trace_basic = pm.sample(1000, tune=1000, return_inferencedata=True)

with pm.Model() as sv_leverage:
    # SV with leverage
    # ... build model
    trace_leverage = pm.sample(1000, tune=1000, return_inferencedata=True)

# Compare
comparison = az.compare({
    'Basic': trace_basic,
    'Leverage': trace_leverage
})
print(comparison)
```

</div>
</div>

Lower LOO = better out-of-sample predictive performance.

---

## Common Pitfalls

### 1. Non-Centered Parameterization

Standard SV models can have poor geometry. Use non-centered parameterization:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python

# Bad: Centered
h[t] = mu + phi * (h[t-1] - mu) + sigma_eta * eta[t]

# Better: Non-centered
h_raw[t] = phi * h_raw[t-1] + eta[t]
h[t] = mu + sigma_eta / sqrt(1 - phi^2) * h_raw[t]
```

</div>
</div>

### 2. Initialization Sensitivity

Poor initial $h_1$ causes bias. Use stationary distribution:
$$h_1 \sim \mathcal{N}(\mu, \sigma_\eta^2 / (1 - \phi^2))$$

### 3. Confusing $h_t$ and $\sigma_t^2$

$h_t$ is log-variance: $h_t = \log(\sigma_t^2)$, so $\sigma_t = \exp(h_t/2)$.

### 4. Overlooking Persistence

Commodity volatility is highly persistent ($\phi > 0.9$ typical). Use strong prior: `Beta(20, 1.5)`.

---

## Connections

**Builds on:**
- Module 3: State space fundamentals (SV is a nonlinear state space model)
- Module 1: Bayesian inference for latent variables

**Leads to:**
- Module 6: Advanced MCMC for complex posteriors
- Module 7: Regime-switching volatility models
- Module 8: Incorporating fundamentals into volatility

**Related concepts:**
- GARCH models (frequentist alternative)
- Realized volatility (uses high-frequency data)
- VIX (model-free implied volatility)

---

## Practice Problems

### Problem 1
A crude oil SV model estimates $\phi = 0.96$ and $\sigma_\eta = 0.15$. If current log-volatility is $h_t = 0.5$:
1. What is the expected log-volatility in 1 week? 10 weeks?
2. What is the uncertainty (standard deviation)?
<div class="callout-insight">

<strong>Insight:</strong> A crude oil SV model estimates $\phi = 0.96$ and $\sigma_\eta = 0.15$. If current log-volatility is $h_t = 0.5$:

</div>


### Problem 2
Why does the SV model use log-variance rather than log-standard-deviation as the latent state?

**Hint:** Consider the observation equation and what parameterization makes sampling more efficient.

### Problem 3
You observe that crude oil volatility spikes immediately after large negative price moves. Which SV extension captures this? Write the model equations.

### Problem 4
Implement an SV model for natural gas prices with seasonal volatility (high in Jan/Jul, low in Apr/Oct). Use PyMC to estimate parameters.

### Problem 5
Compare the one-week-ahead volatility forecast from:
1. Simple historical volatility (std of last 20 days)
2. EWMA volatility
3. SV model

Which provides uncertainty estimates? Which adapts fastest to regime changes?

---


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Formal Definition" and why it matters in practice.

2. Given a real-world scenario involving stochastic volatility models for commodities, what would be your first three steps to apply the techniques from this guide?

</div>

## Further Reading

1. **Kim, S., Shephard, N., & Chib, S. (1998)**. "Stochastic Volatility: Likelihood Inference and Comparison with ARCH Models." *Review of Economic Studies*, 65(3), 361-393.
   - Classic reference for SV inference

2. **Kastner, G. & Frühwirth-Schnatter, S. (2014)**. "Ancillarity-Sufficiency Interweaving Strategy (ASIS) for Boosting MCMC Estimation of Stochastic Volatility Models." *Computational Statistics & Data Analysis*, 76, 408-423.
   - Advanced MCMC techniques for efficient SV sampling

3. **Shephard, N. (2005)**. *Stochastic Volatility: Selected Readings*. Oxford University Press.
   - Comprehensive collection of SV papers

4. **Prado, R. & West, M. (2010)**. *Time Series: Modeling, Computation, and Inference*. CRC Press.
   - Bayesian perspective on SV and extensions

---

*"In commodity markets, predicting volatility is often more valuable than predicting prices—because volatility is more predictable."*

---

## Cross-References

<a class="link-card" href="./03_stochastic_volatility_slides.md">
  <div class="link-card-title">Companion Slide Deck</div>
  <div class="link-card-description">Visual presentation covering the key concepts from this guide.</div>
</a>

<a class="link-card" href="../notebooks/01_local_level_model.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>
