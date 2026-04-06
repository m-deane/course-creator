# The Kalman Filter: Optimal Bayesian Updating for Linear-Gaussian Systems

> **Reading time:** ~9 min | **Module:** 3 — State-Space Models | **Prerequisites:** Module 2 Commodity Data


## In Brief

The Kalman filter is a recursive algorithm that optimally estimates the hidden state of a linear dynamical system from noisy observations. It's the Bayesian solution to state estimation when both the system dynamics and observations are linear with Gaussian noise.

<div class="callout-insight">

<strong>Insight:</strong> The Kalman filter answers: "Given all observations up to now, what's my best estimate of the current hidden state, and how confident am I?" It does this by alternating between **prediction** (using system dynamics) and **update** (incorporating new observations).

</div>

Think of it as GPS navigation: Your car's speedometer predicts where you are (prediction step), but GPS measurements correct this estimate (update step). The Kalman filter optimally combines these two sources of information.

## Formal Definition

### State Space Model

**Observation Equation:**
$$y_t = Z_t \alpha_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, H_t)$$

**State Transition Equation:**
$$\alpha_{t+1} = T_t \alpha_t + R_t \eta_t, \quad \eta_t \sim \mathcal{N}(0, Q_t)$$

Where:
- $y_t$: Observed data (e.g., commodity prices)
- $\alpha_t$: Hidden state (e.g., true price level, trend)
- $Z_t$: Observation matrix (maps state to observables)
- $T_t$: Transition matrix (state dynamics)
- $H_t$: Observation noise covariance
- $Q_t$: State noise covariance

### Kalman Filter Recursions

**Prediction Step:**
$$\alpha_{t|t-1} = T_t \alpha_{t-1|t-1}$$
$$P_{t|t-1} = T_t P_{t-1|t-1} T_t^T + R_t Q_t R_t^T$$

**Update Step:**
$$v_t = y_t - Z_t \alpha_{t|t-1}$$ (forecast error)
$$F_t = Z_t P_{t|t-1} Z_t^T + H_t$$ (forecast variance)
$$K_t = P_{t|t-1} Z_t^T F_t^{-1}$$ (Kalman gain)

$$\alpha_{t|t} = \alpha_{t|t-1} + K_t v_t$$
$$P_{t|t} = P_{t|t-1} - K_t Z_t P_{t|t-1}$$

**Interpretation:**
- $\alpha_{t|t-1}$: Predicted state before seeing $y_t$
- $\alpha_{t|t}$: Updated state after seeing $y_t$
- $P_{t|t}$: Updated state covariance (uncertainty)
- $K_t$: Optimal weight on new information

## Intuitive Explanation

### The GPS Analogy

Imagine driving with:
- **Speedometer:** Says you've traveled 100 meters (but speedometer can drift)
- **GPS:** Says you're at position 95 meters (but GPS has noise)

How do you combine them?

**Naive:** Average them → (100 + 95) / 2 = 97.5 meters

**Kalman Filter:** Weight by reliability
- If speedometer is very accurate: Trust it more
- If GPS is very accurate: Trust it more
- **Optimal weight = Kalman gain**

### Commodity Price Example

**Hidden state:** True underlying price level $\mu_t$ (evolves randomly)
**Observed:** Noisy market price $y_t = \mu_t + \epsilon_t$

**Prediction:** "Based on yesterday's level, today should be around $75"
**Observation:** "Market shows $77"
**Update:** "Combine these: likely $76, with confidence interval [$74, $78]"

The Kalman filter makes this combination optimal (minimum variance estimate).

## Mathematical Formulation

### Why the Kalman Filter is Optimal
<div class="callout-warning">

<strong>Warning:</strong> **Theorem:** Under linear-Gaussian assumptions, the Kalman filter produces:

</div>


**Theorem:** Under linear-Gaussian assumptions, the Kalman filter produces:
1. **Minimum variance** unbiased estimates
2. **Exact Bayesian posterior:** $p(\alpha_t | y_{1:t})$
3. **Sufficient statistics:** $(\alpha_{t|t}, P_{t|t})$ summarize all information

**Proof sketch:**
The posterior $p(\alpha_t | y_{1:t})$ is Gaussian (conjugacy). The Kalman filter recursively computes its mean and covariance.

### Bayesian Interpretation

**Prior:** $\alpha_{t|t-1} \sim \mathcal{N}(\hat{\alpha}_{t|t-1}, P_{t|t-1})$

**Likelihood:** $y_t | \alpha_t \sim \mathcal{N}(Z_t \alpha_t, H_t)$

**Posterior:** $\alpha_t | y_t \sim \mathcal{N}(\hat{\alpha}_{t|t}, P_{t|t})$

The update equations are just Normal-Normal conjugate updating!

### Kalman Gain Decomposition

$$K_t = P_{t|t-1} Z_t^T (Z_t P_{t|t-1} Z_t^T + H_t)^{-1}$$

**Interpretation:**
- Numerator: How uncertain is state prediction?
- Denominator: How uncertain is observation?
- If state very uncertain: $K_t$ large → trust new data more
- If observation very noisy: $K_t$ small → trust prediction more

**Extreme cases:**
- $H_t \to 0$ (perfect observations): $K_t \to Z_t^{-1}$, fully trust data
- $P_{t|t-1} \to 0$ (certain state): $K_t \to 0$, ignore new data

## Code Implementation

### Simple Local Level Model


<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    """
    Simple Kalman filter for local level model:
    y_t = α_t + ε_t,  ε_t ~ N(0, H)
    α_t = α_{t-1} + η_t,  η_t ~ N(0, Q)
    """
    def __init__(self, H, Q, alpha_0, P_0):
        self.H = H  # Observation noise variance
        self.Q = Q  # State noise variance
        self.alpha = alpha_0  # Initial state
        self.P = P_0  # Initial state variance

        # Storage
        self.filtered_states = []
        self.filtered_vars = []

    def predict(self):
        """Prediction step"""
        # α_{t|t-1} = α_{t-1|t-1} (random walk)
        alpha_pred = self.alpha
        # P_{t|t-1} = P_{t-1|t-1} + Q
        P_pred = self.P + self.Q
        return alpha_pred, P_pred

    def update(self, y):
        """Update step given observation y"""
        # Prediction
        alpha_pred, P_pred = self.predict()

        # Innovation
        v = y - alpha_pred  # Forecast error
        F = P_pred + self.H  # Forecast variance

        # Kalman gain
        K = P_pred / F

        # Updated estimates
        self.alpha = alpha_pred + K * v
        self.P = P_pred - K * P_pred

        # Store
        self.filtered_states.append(self.alpha)
        self.filtered_vars.append(self.P)

        return self.alpha, self.P

# Simulate data
np.random.seed(42)
T = 100
Q_true = 1.0  # State variance
H_true = 4.0  # Observation variance

# True states (random walk)
alpha_true = np.cumsum(np.random.randn(T) * np.sqrt(Q_true))
# Observations
y = alpha_true + np.random.randn(T) * np.sqrt(H_true)

# Run Kalman filter
kf = KalmanFilter(H=H_true, Q=Q_true, alpha_0=0, P_0=10)
for yt in y:
    kf.update(yt)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(alpha_true, label='True state', linewidth=2, color='black')
plt.plot(y, 'o', label='Observations', alpha=0.5, markersize=4)
plt.plot(kf.filtered_states, label='Filtered state', linewidth=2, color='red')

# Add uncertainty bands
states = np.array(kf.filtered_states)
stds = np.sqrt(kf.filtered_vars)
plt.fill_between(range(len(states)), states - 1.96*stds, states + 1.96*stds,
                 alpha=0.3, color='red', label='95% CI')

plt.legend()
plt.xlabel('Time')
plt.ylabel('State value')
plt.title('Kalman Filter: Local Level Model')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Mean squared error: {np.mean((np.array(kf.filtered_states) - alpha_true)**2):.3f}")
```

</div>
</div>

### Commodity Application: Filtering Oil Prices


<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import pandas as pd

# Assume we have daily oil prices
# We'll filter them to extract the underlying level

def filter_commodity_prices(prices, Q=0.5, H=2.0):
    """
    Apply Kalman filter to commodity prices
    Q: state variance (how much true level varies)
    H: observation variance (market noise)
    """
    kf = KalmanFilter(H=H, Q=Q, alpha_0=prices[0], P_0=10)

    filtered = []
    for p in prices:
        alpha, _ = kf.update(p)
        filtered.append(alpha)

    return np.array(filtered)

# Example usage
# prices = fetch_wti_prices()  # Your data here
# smooth_prices = filter_commodity_prices(prices)
```

</div>
</div>

## Visual Representation

```
Kalman Filter Cycle:

    Prior State             New Observation         Posterior State
    α_{t|t-1}, P_{t|t-1}    y_t                    α_{t|t}, P_{t|t}
         │                   │                           │
         ▼                   ▼                           ▼
    ┌────────────┐      ┌─────────┐              ┌──────────┐
    │  Predict   │      │ Observe │              │  Update  │
    │            │─────→│   v_t   │─────────────→│          │
    │ (system    │      │  F_t    │  Kalman     │ (correct  │
    │  dynamics) │      │         │   Gain K_t   │  pred.)   │
    └────────────┘      └─────────┘              └──────────┘
         │                                              │
         └──────────────────────────────────────────────┘
                    Loop for t = 1, 2, ..., T

Uncertainty Evolution:

    Predict: P increases (add Q - system noise)
      │
      ▼
    [════════════]  Wider uncertainty
      │
      ▼
    Update: P decreases (Kalman gain reduces uncertainty)
      │
      ▼
    [═══════]  Narrower uncertainty
```

## Common Pitfalls

### 1. Wrong Noise Variances (Q, H)
<div class="callout-warning">

<strong>Warning:</strong> **Problem:** If Q and H are misspecified, filter can be too sluggish or too jumpy.

</div>


**Problem:** If Q and H are misspecified, filter can be too sluggish or too jumpy.

**Example:**

<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# Too small Q, too large H → Filter stuck, doesn't adapt
kf = KalmanFilter(H=100, Q=0.01, ...)

# Too large Q, too small H → Filter chases noise
kf = KalmanFilter(H=0.01, Q=100, ...)
```

</div>
</div>

**Solution:**
- Estimate Q, H from data (maximum likelihood)
- Use robust defaults or cross-validation
- Monitor innovation sequence $v_t$ for white noise

### 2. Non-Gaussian Errors

**Problem:** Kalman filter assumes Gaussian noise. Fat-tailed errors (common in commodities!) violate this.

**Symptoms:** Poor performance during extreme events

**Solutions:**
- Use **robust Kalman filter** (e.g., Huber loss)
- Switch to **particle filter** for non-Gaussian cases
- Model fat tails explicitly (e.g., Student-t errors)

### 3. Model Misspecification

**Problem:** True system is nonlinear, but you use linear Kalman filter

**Example:** Oil prices with regime switches (not captured by linear model)

**Solutions:**
- **Extended Kalman Filter (EKF):** Linearize nonlinear system
- **Unscented Kalman Filter (UKF):** Better nonlinear approximation
- **Particle filter:** Fully nonlinear

### 4. Initialization Issues

**Problem:** Poor initial $\alpha_0$, $P_0$ can bias early estimates

**Fix:**
- Use **diffuse initialization** ($P_0 \to \infty$)
- Discard early observations ("burn-in")
- Initialize with sample mean/variance

### 5. Forgetting to Check Innovation Sequence

**Diagnostic:** If $v_t$ (forecast errors) are not white noise, model is wrong!

**Check:**

<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
innovations = y - predictions
# Should be uncorrelated, zero-mean, constant variance
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(innovations)  # Should show no significant lags
```

</div>
</div>

## Connections

### Builds on:
- **Module 1 (Bayes' Theorem):** Kalman filter = Bayesian updating
- **Module 1 (Conjugate Priors):** Normal-Normal conjugacy enables analytical solution
- **Linear algebra:** Matrix operations for state evolution

### Leads to:
- **Module 3.3 (Stochastic Volatility):** Extend to time-varying H_t
- **Module 7 (Regime Switching):** Multiple Kalman filters (one per regime)
- **Module 8 (Fundamentals):** Dynamic regression with time-varying β

### Related to:
- **ARIMA models:** Can be written as state space + Kalman filter
- **Exponential smoothing:** Special case of Kalman filter
- **Bayesian filtering:** Kalman filter is the Gaussian case

## Practice Problems

### 1. Verify Kalman Gain Formula

Show that the Kalman gain $K_t$ that minimizes the updated variance $P_{t|t}$ is:
$$K_t = P_{t|t-1} Z_t^T (Z_t P_{t|t-1} Z_t^T + H_t)^{-1}$$

**Hint:** Minimize $P_{t|t} = (1 - K_t Z_t) P_{t|t-1}$ with respect to $K_t$.


<div class="callout-key">

<strong>Key Concept Summary:</strong> The Kalman filter is a recursive algorithm that optimally estimates the hidden state of a linear dynamical system from noisy observations.

</div>

---

### 2. Implement Kalman Smoother

Extend the Kalman filter to compute smoothed estimates $\alpha_{t|T}$ using all data (not just $y_{1:t}$).

**Backward recursions:**
$$\alpha_{t|T} = \alpha_{t|t} + J_t (\alpha_{t+1|T} - \alpha_{t+1|t})$$
$$J_t = P_{t|t} T_{t+1}^T P_{t+1|t}^{-1}$$

**Task:** Code this and compare filtered vs smoothed estimates.

---

### 3. Estimate Q and H from Data

Given commodity price data, estimate optimal $Q$ and $H$ by:
a) Grid search over $(Q, H)$ pairs
b) Maximum likelihood estimation

**Hint:** Log-likelihood is sum of innovations: $\sum_t \log p(v_t | F_t)$

---

### 4. Commodity Application

Apply Kalman filter to:
- WTI crude oil prices (2020-2024)
- Compare filtered level to raw prices
- Identify periods of high innovation variance (market shocks)

**Extensions:**
- Try different $Q$, $H$ values
- Add trend component (local linear trend model)


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Formal Definition" and why it matters in practice.

2. Given a real-world scenario involving the kalman filter: optimal bayesian updating for linear-gaussian systems, what would be your first three steps to apply the techniques from this guide?

</div>

## Further Reading

### Classic References
- **Kalman (1960):** "A New Approach to Linear Filtering and Prediction Problems" - Original paper
- **Durbin & Koopman (2012):** *Time Series Analysis by State Space Methods* - Comprehensive treatment
<div class="callout-warning">

<strong>Warning:</strong> - **Kalman (1960):** "A New Approach to Linear Filtering and Prediction Problems" - Original paper

</div>


### Applied Econometrics
- **Hamilton (1994):** *Time Series Analysis* - Chapter 13 on state space models
- **Harvey (1989):** *Forecasting, Structural Time Series Models and the Kalman Filter*

### Commodity Applications
- **Schwartz & Smith (2000):** "Short-term variations and long-term dynamics in commodity prices" - Two-factor Kalman filter for oil
- **Considine & Larson (2001):** "Risk premiums on inventory assets" - Kalman filter for convenience yield

### Software
- **Python:** `statsmodels.tsa.statespace` - Full state space framework
- **Python:** `pykalman` - Simple Kalman filter library
- **R:** `dlm`, `KFAS` packages

### Advanced Topics
- **Bayesian estimation of Q, H:** Use PyMC to estimate noise parameters
- **Time-varying parameters:** $Q_t$, $H_t$ change over time
- **Multivariate Kalman filter:** Multiple correlated series

---

**Next:** Implement stochastic volatility models using the Kalman filter in `03_stochastic_volatility_pymc.ipynb`

---

## Cross-References

<a class="link-card" href="./02_kalman_filter_slides.md">
  <div class="link-card-title">Companion Slide Deck</div>
  <div class="link-card-description">Visual presentation covering the key concepts from this guide.</div>
</a>

<a class="link-card" href="../notebooks/01_local_level_model.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>
