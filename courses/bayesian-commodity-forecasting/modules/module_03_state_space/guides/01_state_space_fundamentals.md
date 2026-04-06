# State Space Fundamentals

> **Reading time:** ~6 min | **Module:** 3 — State-Space Models | **Prerequisites:** Module 2 Commodity Data


## In Brief

State space models decompose observed time series into unobserved (latent) components that evolve according to known dynamics. This framework provides a unified approach to trend extraction, forecasting, and uncertainty quantification.

<div class="callout-insight">

<strong>Insight:</strong> **Think of state space as a hidden story.** We observe the outcome (prices) but not the underlying drivers (trend, momentum, sentiment). State space models infer these hidden states from observable data.

</div>

## Formal Definition

### General Linear Gaussian State Space Model

**Observation Equation:**
$$y_t = Z_t \alpha_t + d_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, H_t)$$

**State Transition Equation:**
$$\alpha_{t+1} = T_t \alpha_t + c_t + R_t \eta_t, \quad \eta_t \sim \mathcal{N}(0, Q_t)$$

**Initial State:**
$$\alpha_1 \sim \mathcal{N}(a_1, P_1)$$

### Component Definitions

| Symbol | Dimension | Description |
|--------|-----------|-------------|
| $y_t$ | $p \times 1$ | Observations at time t |
| $\alpha_t$ | $m \times 1$ | State vector at time t |
| $Z_t$ | $p \times m$ | Observation matrix (links state to obs) |
| $T_t$ | $m \times m$ | Transition matrix (state dynamics) |
| $H_t$ | $p \times p$ | Observation noise covariance |
| $Q_t$ | $r \times r$ | State noise covariance |
| $R_t$ | $m \times r$ | State noise selection matrix |


<div class="callout-key">

<strong>Key Concept Summary:</strong> State space models decompose observed time series into unobserved (latent) components that evolve according to known dynamics.

</div>

---

## Intuitive Explanation

### The Two-Layer Model

**Layer 1: Hidden Reality (State)**
- The "true" underlying process we can't directly observe
- Examples: Equilibrium price level, volatility regime, trend direction
- Evolves according to state transition equation

**Layer 2: Noisy Observations (Data)**
- What we actually measure (market prices)
- Corrupted by observation noise (bid-ask spread, microstructure)
- Links to hidden state via observation equation

### Visual Intuition

```
Time:     t=1      t=2      t=3      t=4
          ┌───┐    ┌───┐    ┌───┐    ┌───┐
States:   │α₁│───→│α₂│───→│α₃│───→│α₄│  (Hidden)
          └─┬─┘    └─┬─┘    └─┬─┘    └─┬─┘
            │        │        │        │
            ↓        ↓        ↓        ↓
          ┌───┐    ┌───┐    ┌───┐    ┌───┐
Obs:      │y₁│    │y₂│    │y₃│    │y₄│  (Observed)
          └───┘    └───┘    └───┘    └───┘
```

---

## Common State Space Models

### 1. Local Level Model (Random Walk + Noise)
<div class="callout-warning">

<strong>Warning:</strong> The simplest state space model: a random walk observed with noise.

</div>


The simplest state space model: a random walk observed with noise.

**State:** $\mu_t$ (level)

$$y_t = \mu_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma_\epsilon^2)$$
$$\mu_{t+1} = \mu_t + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \sigma_\eta^2)$$

**State space form:**
- $\alpha_t = \mu_t$
- $Z = 1$, $T = 1$
- $H = \sigma_\epsilon^2$, $Q = \sigma_\eta^2$

**Use case:** Filtering noise from commodity prices to reveal underlying level.

**Signal-to-noise ratio:** $q = \sigma_\eta^2 / \sigma_\epsilon^2$
- $q \to 0$: Observations are mostly noise (smooth the level)
- $q \to \infty$: Level changes rapidly (track observations closely)

---

### 2. Local Linear Trend Model

Adds a stochastic trend component.

**State:** $[\mu_t, \nu_t]'$ (level and trend)

$$y_t = \mu_t + \epsilon_t$$
$$\mu_{t+1} = \mu_t + \nu_t + \eta_t$$
$$\nu_{t+1} = \nu_t + \zeta_t$$

**State space form:**
$$\alpha_t = \begin{bmatrix} \mu_t \\ \nu_t \end{bmatrix}, \quad
T = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}, \quad
Z = \begin{bmatrix} 1 & 0 \end{bmatrix}$$

**Use case:** Extracting trend direction from volatile commodity prices.

---

### 3. Basic Structural Model (BSM)

Adds seasonality to the local linear trend.

**State:** $[\mu_t, \nu_t, \gamma_{1,t}, ..., \gamma_{s-1,t}]'$

For seasonality with period $s$:
$$\gamma_t = -\sum_{j=1}^{s-1} \gamma_{t-j} + \omega_t$$

**Use case:** Agricultural commodities with harvest cycles, natural gas with heating/cooling seasons.

---

### 4. Stochastic Volatility Model

Models time-varying variance in returns.

$$y_t = \exp(h_t / 2) \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, 1)$$
$$h_{t+1} = \mu + \phi(h_t - \mu) + \sigma_\eta \eta_t$$

Where $h_t = \log(\sigma_t^2)$ is log-volatility.

**State space form:** Requires transformation (log-squared returns) for approximate linearity.

**Use case:** Capturing volatility clustering in commodity returns.

---

## The Filtering Problem

Given observations $y_1, ..., y_t$, we want:

1. **Filtering:** $p(\alpha_t | y_1, ..., y_t)$ — current state given past and present
2. **Smoothing:** $p(\alpha_t | y_1, ..., y_T)$ — state given all data
3. **Prediction:** $p(\alpha_{t+h} | y_1, ..., y_t)$ — future state given past

For linear-Gaussian models, these are all Gaussian with closed-form solutions (Kalman filter).

---

## Connection to ARIMA

Many ARIMA models have state space representations:

| ARIMA Model | State Space Equivalent |
|-------------|----------------------|
| ARIMA(0,1,0) | Local Level with $\sigma_\epsilon^2 = 0$ |
| ARIMA(0,1,1) | Local Level |
| ARIMA(0,2,2) | Local Linear Trend |
| ARIMA(p,d,q) | General state space with $m = \max(p, q+1) + d$ |

**Advantage of state space:** More flexible, handles missing data, time-varying parameters.

---

## Code Implementation

### PyMC State Space Model


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import pymc as pm
import numpy as np

# Simulated commodity price data
n_obs = 100
true_level = np.cumsum(np.random.normal(0, 0.5, n_obs)) + 80
y = true_level + np.random.normal(0, 2, n_obs)

# Local Level Model in PyMC
with pm.Model() as local_level:
    # Noise variances
    sigma_level = pm.HalfNormal('sigma_level', sigma=1)
    sigma_obs = pm.HalfNormal('sigma_obs', sigma=5)

    # Initial level
    level_init = pm.Normal('level_init', mu=80, sigma=10)

    # Level evolution (random walk)
    level_innovations = pm.Normal('level_innov', mu=0, sigma=1, shape=n_obs-1)
    level = pm.Deterministic(
        'level',
        pm.math.concatenate([[level_init],
                             level_init + pm.math.cumsum(sigma_level * level_innovations)])
    )

    # Observations
    y_obs = pm.Normal('y_obs', mu=level, sigma=sigma_obs, observed=y)

    trace = pm.sample(1000, tune=1000)
```

</div>
</div>

---

## Why Bayesian State Space?

### 1. Parameter Uncertainty
Classical Kalman filter treats variances as known. Bayesian approach estimates them with uncertainty.

### 2. Prior Information
Encode domain knowledge about level persistence, volatility ranges, seasonal patterns.

### 3. Missing Data
Natural handling of gaps (skip the update step).

### 4. Model Comparison
Compare different state space specifications using WAIC, LOO-CV.

---

## Common Pitfalls

### 1. Non-stationarity Confusion
Random walk states are non-stationary by design. This is not a bug—it's modeling persistence.

### 2. Initialization Sensitivity
Poor initial state specification can distort early inference. Use diffuse initialization or informative priors.

### 3. Overparameterization
With limited data, simpler models (local level) often outperform complex ones (BSM).

### 4. Confusing Filter and Smoother
- Filter: Uses data up to time t (for real-time forecasting)
- Smoother: Uses all data (for retrospective analysis)

---

## Connections

**Builds on:**
- Module 1: Sequential Bayesian updating
- Linear algebra: Matrix operations for state transitions

**Leads to:**
- Module 5: GPs as infinite-dimensional state space models
- Module 7: Regime-switching (hidden Markov) state space
- Module 8: Dynamic regression for fundamentals

---

## Practice Problems

### Problem 1
Write out the state space matrices $(Z, T, H, Q)$ for an AR(1) process:
$$y_t = \phi y_{t-1} + \epsilon_t$$

### Problem 2
A commodity trader claims the oil market has a "hidden equilibrium price" that the market price oscillates around. Which state space model captures this idea?

### Problem 3
You have weekly natural gas prices with occasional missing values (holidays). Explain how the Kalman filter handles missing observations.

---


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Formal Definition" and why it matters in practice.

2. Given a real-world scenario involving state space fundamentals, what would be your first three steps to apply the techniques from this guide?


## Further Reading

1. **Durbin & Koopman** *Time Series Analysis by State Space Methods* — Definitive reference
2. **Harvey, A.C.** *Forecasting, Structural Time Series Models* — Classic treatment
3. **Commandeur & Koopman** *Introduction to State Space Time Series Analysis* — Accessible intro

---

*State space models are like X-ray machines for time series: they reveal the hidden structure beneath the noisy surface.*

---

## Cross-References

<a class="link-card" href="./01_state_space_fundamentals_slides.md">
  <div class="link-card-title">Companion Slide Deck</div>
  <div class="link-card-description">Visual presentation covering the key concepts from this guide.</div>
</a>

<a class="link-card" href="../notebooks/01_local_level_model.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>
