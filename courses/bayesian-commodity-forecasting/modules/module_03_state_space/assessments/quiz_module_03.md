# Module 3 Quiz: State Space Models and Kalman Filtering

**Course:** Bayesian Commodity Forecasting
**Module:** 03 - State Space Models
**Time Limit:** 30 minutes
**Total Points:** 100 points
**Instructions:** Answer all questions. Show work for mathematical derivations.

---

## Section A: State Space Fundamentals (30 points)

### Question 1 (8 points)
Write out the general linear Gaussian state space model equations and define each component. Then express a simple AR(1) process $y_t = \phi y_{t-1} + \epsilon_t$ in state space form.

**Answer:**

**General State Space Model:**

**Observation Equation:**
$$y_t = Z_t \alpha_t + d_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, H_t)$$

**State Transition Equation:**
$$\alpha_{t+1} = T_t \alpha_t + c_t + R_t \eta_t, \quad \eta_t \sim \mathcal{N}(0, Q_t)$$

**Definitions:**
- $y_t$: Observation vector ($p \times 1$)
- $\alpha_t$: State vector ($m \times 1$)
- $Z_t$: Observation matrix ($p \times m$) — maps states to observations
- $T_t$: Transition matrix ($m \times m$) — governs state dynamics
- $d_t$: Observation intercept ($p \times 1$)
- $c_t$: State intercept ($m \times 1$)
- $H_t$: Observation noise covariance ($p \times p$)
- $Q_t$: State noise covariance ($r \times r$)
- $R_t$: State noise selection matrix ($m \times r$)

**AR(1) in State Space Form:**

For $y_t = \phi y_{t-1} + \epsilon_t$ where $\epsilon_t \sim \mathcal{N}(0, \sigma^2)$:

- **State:** $\alpha_t = y_t$ (scalar)
- **Observation equation:** $y_t = \alpha_t$ (observe state directly)
  - $Z_t = 1$, $d_t = 0$, $H_t = 0$ (no observation noise)
- **Transition equation:** $\alpha_{t+1} = \phi \alpha_t + \eta_t$
  - $T_t = \phi$, $c_t = 0$, $R_t = 1$, $Q_t = \sigma^2$

**State space representation:**
$$\begin{aligned}
y_t &= 1 \cdot \alpha_t \\
\alpha_{t+1} &= \phi \alpha_t + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \sigma^2)
\end{aligned}$$

**Scoring:**
- General equations correct: 3 points
- Definitions: 2 points
- AR(1) state space form: 3 points

---

### Question 2 (7 points)
Consider the **Local Level Model** for commodity prices:

$$y_t = \mu_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma_\epsilon^2)$$
$$\mu_{t+1} = \mu_t + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \sigma_\eta^2)$$

**(a)** What does the ratio $q = \sigma_\eta^2 / \sigma_\epsilon^2$ (signal-to-noise ratio) control? (3 points)

**(b)** For crude oil prices with high daily volatility but slowly changing fundamental value, would you expect $q$ to be large or small? (4 points)

**Answer:**

**(a)** The signal-to-noise ratio $q$ controls the **smoothness** of the extracted level:

- **$q \to 0$ (small):**
  - State noise $\sigma_\eta^2 \to 0$ → level changes very slowly
  - Observations are mostly noise
  - Filter produces **smooth** level estimate (heavy smoothing)
  - Extreme: $\sigma_\eta^2 = 0$ → constant level

- **$q \to \infty$ (large):**
  - State noise $\sigma_\eta^2$ large → level changes rapidly
  - Observations are informative about rapidly changing state
  - Filter **tracks observations closely** (minimal smoothing)
  - Extreme: $\sigma_\epsilon^2 = 0$ → level = observations

**Intuition:** $q$ is the "trust" ratio:
- Small $q$: Trust the model (smooth state), distrust data
- Large $q$: Trust the data, allow flexible state evolution

**(b)** For crude oil with high daily volatility but slow fundamental changes:
- **Expect $q$ to be SMALL**

**Reasoning:**
- **High $\sigma_\epsilon^2$:** Daily prices bounce around due to:
  - Microstructure noise
  - Intraday supply/demand fluctuations
  - Short-term speculative trading
- **Low $\sigma_\eta^2$:** Fundamental equilibrium price changes slowly:
  - Global supply/demand adjust over weeks/months
  - OPEC decisions are infrequent
  - Infrastructure constraints change gradually

**Example values:**
- Daily crude returns: $\sigma_\epsilon \approx 2-3\%$
- Fundamental drift: $\sigma_\eta \approx 0.1-0.5\%$ per day
- Ratio: $q \approx (0.3/2)^2 \approx 0.02$ (small)

**Implication:** Kalman filter would heavily smooth crude oil prices to extract underlying level.

**Scoring:**
- Part (a): 3 points (smoothness concept and extremes)
- Part (b): 4 points (2 for correct answer, 2 for reasoning)

---

### Question 3 (8 points)
The **Local Linear Trend (LLT)** model extends the local level model:

$$y_t = \mu_t + \epsilon_t$$
$$\mu_{t+1} = \mu_t + \nu_t + \eta_t$$
$$\nu_{t+1} = \nu_t + \zeta_t$$

where $\nu_t$ is the trend (slope) component.

**(a)** Write this in standard state space notation with state vector $\alpha_t = [\mu_t, \nu_t]'$. (5 points)

**(b)** How does this differ from a model with deterministic linear trend? (3 points)

**Answer:**

**(a)** State space representation:

**State vector:** $\alpha_t = \begin{bmatrix} \mu_t \\ \nu_t \end{bmatrix}$

**Observation equation:**
$$y_t = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} \mu_t \\ \nu_t \end{bmatrix} + \epsilon_t$$

$$Z = \begin{bmatrix} 1 & 0 \end{bmatrix}, \quad H = \sigma_\epsilon^2$$

**Transition equation:**
$$\begin{bmatrix} \mu_{t+1} \\ \nu_{t+1} \end{bmatrix} =
\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}
\begin{bmatrix} \mu_t \\ \nu_t \end{bmatrix} +
\begin{bmatrix} \eta_t \\ \zeta_t \end{bmatrix}$$

$$T = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}, \quad
R = I_2, \quad
Q = \begin{bmatrix} \sigma_\eta^2 & 0 \\ 0 & \sigma_\zeta^2 \end{bmatrix}$$

**Complete form:**
$$\begin{aligned}
y_t &= Z \alpha_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma_\epsilon^2) \\
\alpha_{t+1} &= T \alpha_t + R \eta_t, \quad \eta_t \sim \mathcal{N}(0, Q)
\end{aligned}$$

**(b)** Differences from deterministic linear trend:

**Deterministic:** $y_t = \alpha + \beta t + \epsilon_t$
- Trend parameters $\alpha, \beta$ are **fixed** (estimated but constant)
- Trend extrapolates linearly forever
- Cannot adapt to structural breaks

**Stochastic (LLT):**
- Trend slope $\nu_t$ **evolves over time** (random walk)
- Allows trend to change direction ($\sigma_\zeta^2 > 0$)
- **Adaptive:** Can capture gradual changes in growth rate
- If $\sigma_\zeta^2 = 0$: $\nu_t = \nu$ (constant) → equivalent to deterministic trend

**Commodity application:**
- Oil prices: Trend changes with structural shifts (shale revolution, OPEC policy)
- Deterministic trend inappropriate for regime changes
- LLT allows trend to adapt via Kalman filter updates

**Scoring:**
- Part (a): 5 points (matrices correct, proper dimensions)
- Part (b): 3 points (key differences explained)

---

### Question 4 (7 points)
Explain the difference between **filtering**, **smoothing**, and **prediction** in state space models. For a trader using a Kalman filter on live commodity prices, which would they use and why?

**Answer:**

**Three inference problems:**

**1. Filtering:** $p(\alpha_t | y_1, ..., y_t)$
- **Estimate current state** given data up to now
- Uses information: Past + Present
- **Real-time:** Updates as new data arrives
- **Output:** Filtered state estimate $\hat{\alpha}_{t|t}$ and covariance $P_{t|t}$

**2. Smoothing:** $p(\alpha_t | y_1, ..., y_T)$
- **Retrospective estimate** of state at time $t$ using ALL data (including future)
- Uses information: Past + Present + Future
- **Offline:** Requires complete dataset
- **Output:** Smoothed state estimate $\hat{\alpha}_{t|T}$ (typically more accurate than filtered)
- **Example:** Backward pass after forward filter

**3. Prediction:** $p(\alpha_{t+h} | y_1, ..., y_t)$
- **Forecast future state** $h$ steps ahead
- Uses information: Past + Present (no future data)
- **Output:** Predicted state $\hat{\alpha}_{t+h|t}$ and covariance $P_{t+h|t}$
- Uncertainty grows with horizon $h$

**For live trading:**

**A trader would use FILTERING and PREDICTION:**

1. **Filtering ($t=\text{now}$):**
   - Real-time estimate of current equilibrium price/trend
   - Example: "What is the noise-free oil price RIGHT NOW?"
   - Use case: Identify mispricing (observed vs filtered level)

2. **Prediction ($t+1, t+2, ...$):**
   - Forecast future prices for trade execution
   - Example: "What will oil price be tomorrow?"
   - Use case: Generate trading signals

**NOT smoothing because:**
- Smoothing requires future data (not available in real-time)
- Cannot wait until tomorrow to get better estimate of today's state
- Only useful for retrospective analysis, backtesting diagnostics

**Visual timeline:**
```
Past         Now (t)      Future
←-----------•----------→
Filter: Uses ←•
Smooth: Uses ←•→ (need future)
Predict:     • → (forecast from now)
```

**Practical implementation:**
```python
# Real-time trading loop
for t in range(len(prices)):
    # Filter: Current state
    filtered_state = kalman_filter.filter(y[:t+1])

    # Predict: Next period
    predicted_price = kalman_filter.forecast(steps=1)

    # Trading decision
    if observed_price < filtered_state - threshold:
        buy()  # Price below fair value
```

**Scoring:**
- Definitions of three concepts: 4 points (1 for each definition, 1 for clarity)
- Trader application: 3 points (correct choice + reasoning)

---

## Section B: Kalman Filter Mechanics (35 points)

### Question 5 (12 points)
The Kalman filter consists of two steps: **prediction** and **update**.

For the local level model with known parameters $\sigma_\epsilon^2, \sigma_\eta^2$:

**(a)** Write the prediction equations for $\hat{\mu}_{t|t-1}$ and $P_{t|t-1}$. (4 points)

**(b)** Write the update equations after observing $y_t$. (5 points)

**(c)** Interpret the Kalman gain $K_t$ intuitively. (3 points)

**Answer:**

**(a) Prediction Step:**

Given previous filtered estimate $\hat{\mu}_{t-1|t-1}$ and variance $P_{t-1|t-1}$:

**State prediction:**
$$\hat{\mu}_{t|t-1} = \hat{\mu}_{t-1|t-1}$$

(For local level: $\mu_t = \mu_{t-1} + \eta_{t-1}$, so $\mathbb{E}[\mu_t | y_1, ..., y_{t-1}] = \mu_{t-1}$)

**Variance prediction:**
$$P_{t|t-1} = P_{t-1|t-1} + \sigma_\eta^2$$

(Variance increases due to state noise)

**(b) Update Step:**

After observing $y_t$:

**1. Prediction error (innovation):**
$$\nu_t = y_t - \hat{\mu}_{t|t-1}$$

**2. Innovation variance:**
$$F_t = P_{t|t-1} + \sigma_\epsilon^2$$

**3. Kalman gain:**
$$K_t = \frac{P_{t|t-1}}{F_t} = \frac{P_{t|t-1}}{P_{t|t-1} + \sigma_\epsilon^2}$$

**4. State update:**
$$\hat{\mu}_{t|t} = \hat{\mu}_{t|t-1} + K_t \nu_t$$

**5. Variance update:**
$$P_{t|t} = (1 - K_t) P_{t|t-1}$$

**(c) Kalman Gain Interpretation:**

$$K_t = \frac{P_{t|t-1}}{P_{t|t-1} + \sigma_\epsilon^2} = \frac{\text{State uncertainty}}{\text{State uncertainty + Observation noise}}$$

**Intuition:** $K_t$ determines how much to trust the new observation:

- **$K_t \to 1$:** High state uncertainty, low observation noise
  - **Trust the data:** Large correction based on innovation $\nu_t$
  - Example: If $P_{t|t-1}$ large and $\sigma_\epsilon^2$ small

- **$K_t \to 0$:** Low state uncertainty, high observation noise
  - **Trust the model:** Small correction, ignore noisy observation
  - Example: If $P_{t|t-1}$ small and $\sigma_\epsilon^2$ large

**Update equation interpretation:**
$$\hat{\mu}_{t|t} = \underbrace{\hat{\mu}_{t|t-1}}_{\text{Prior belief}} + \underbrace{K_t}_{\text{Learning rate}} \times \underbrace{(y_t - \hat{\mu}_{t|t-1})}_{\text{Surprise}}$$

This is **optimal Bayesian updating** for Gaussian models.

**Scoring:**
- Part (a): 4 points (both equations correct)
- Part (b): 5 points (all 5 equations, 1 point each)
- Part (c): 3 points (interpretation with examples)

---

### Question 6 (8 points)
You implement a Kalman filter for WTI crude oil prices (local level model). After running the filter, you examine the **standardized innovations** (one-step-ahead prediction errors):

$$\nu_t^* = \frac{y_t - \hat{y}_{t|t-1}}{\sqrt{F_t}}$$

You observe that:
- Mean of $\nu_t^*$ is 0.15 (should be 0)
- Several $|\nu_t^*| > 3$ (should be rare)
- Strong autocorrelation in $\nu_t^*$ (should be white noise)

**(a)** What do these diagnostics suggest about your model? (4 points)

**(b)** Propose two potential fixes. (4 points)

**Answer:**

**(a)** Diagnostic interpretation:

**1. Non-zero mean (0.15):**
- **Problem:** Systematic bias in predictions
- **Causes:**
  - Model mis-specification (missing drift term)
  - Incorrect initialization
  - Persistent trend not captured by local level
- **Implication:** Model consistently under-predicts prices

**2. Extreme innovations ($|\nu_t^*| > 3$):**
- **Problem:** Fat tails / outliers
- **Causes:**
  - Observation noise $\sigma_\epsilon^2$ underestimated
  - Non-Gaussian errors (jumps, regime shifts)
  - Structural breaks (2020 oil crash, OPEC decisions)
- **Implication:** Gaussian assumption violated

**3. Autocorrelated innovations:**
- **Problem:** Model fails to capture all predictable dynamics
- **Causes:**
  - Local level too simple (missing trend, seasonality, or AR dynamics)
  - Mis-specified transition (e.g., trend is deterministic, not stochastic)
- **Implication:** Information in residuals not extracted by model

**Overall diagnosis:** **Model is too restrictive** — local level cannot capture oil price dynamics.

**(b)** Potential fixes:

**Fix 1: Use Local Linear Trend model**
- Add stochastic trend component to capture drift
- State: $[\mu_t, \nu_t]'$ (level + slope)
- Should reduce autocorrelation and bias

**Fix 2: Add robust observation model**
- Replace Gaussian observation noise with Student-t distribution:
  $$y_t = \mu_t + \epsilon_t, \quad \epsilon_t \sim t_\nu(0, \sigma_\epsilon^2)$$
- PyMC: `pm.StudentT('y_obs', nu=df, mu=level, sigma=sigma_obs, observed=y)`
- Handles outliers (fat tails)

**Fix 3: Include exogenous variables**
- Add fundamentals (storage, production) or seasonality
- Reduces autocorrelation by explaining systematic patterns

**Fix 4: Use regime-switching model**
- Allow parameters to change across regimes (volatility clusters)
- Captures structural breaks naturally

**Fix 5: Increase state noise variance**
- If $\sigma_\eta^2$ too small, model is over-confident
- Increase to allow more flexibility in level evolution

**Scoring:**
- Part (a): 4 points (interpretation of each diagnostic issue)
- Part (b): 4 points (2 points per fix with justification)

---

### Question 7 (7 points)
Consider a state space model where the observation matrix $Z_t$ varies over time. Specifically, for natural gas prices with a seasonal pattern:

$$y_t = \mu_t + S_t + \epsilon_t$$

where $S_t$ is a known seasonal component (extracted from historical data). You want to estimate the latent level $\mu_t$.

**(a)** Write this in state space form treating $S_t$ as a time-varying deterministic component. (4 points)

**(b)** How does the Kalman filter handle time-varying $Z_t$ or $d_t$? (3 points)

**Answer:**

**(a)** State space representation with known seasonality:

**State:** $\alpha_t = \mu_t$ (scalar)

**Observation equation:**
$$y_t = 1 \cdot \mu_t + S_t + \epsilon_t$$

Rearranging:
$$\underbrace{y_t - S_t}_{\text{seasonally adjusted}} = \mu_t + \epsilon_t$$

**Standard form:**
$$y_t = Z \alpha_t + d_t + \epsilon_t$$

where:
- $Z_t = 1$
- $d_t = S_t$ (time-varying deterministic term)
- $\alpha_t = \mu_t$

**Alternative:** Treat as part of observation:
$$y_t^* = y_t - S_t$$

Then: $y_t^* = \mu_t + \epsilon_t$ (standard local level)

**Transition:**
$$\mu_{t+1} = \mu_t + \eta_t$$

Standard local level dynamics.

**(b)** Handling time-varying matrices:

The Kalman filter naturally accommodates time-varying system matrices:

**Prediction step:**
$$\hat{\alpha}_{t|t-1} = T_t \hat{\alpha}_{t-1|t-1} + c_t$$
$$P_{t|t-1} = T_t P_{t-1|t-1} T_t' + R_t Q_t R_t'$$

**Update step:**
$$\nu_t = y_t - (Z_t \hat{\alpha}_{t|t-1} + d_t)$$
$$F_t = Z_t P_{t|t-1} Z_t' + H_t$$
$$K_t = P_{t|t-1} Z_t' F_t^{-1}$$
$$\hat{\alpha}_{t|t} = \hat{\alpha}_{t|t-1} + K_t \nu_t$$
$$P_{t|t} = (I - K_t Z_t) P_{t|t-1}$$

**Key points:**
1. All matrices can vary with $t$
2. Simply substitute time-specific values at each step
3. Computational cost slightly higher (can't cache matrix inverses)
4. Common use cases:
   - $d_t$: Seasonal patterns, exogenous regressors
   - $Z_t$: Missing data (set $Z_t = 0$ for missing obs)
   - $T_t$: Time-varying dynamics (though usually constant)

**Practical implementation:**
```python
# Time-varying seasonal adjustment
d_t = seasonal_pattern[t]  # Pre-computed seasonality
y_adjusted = y[t] - d_t
# Now filter y_adjusted with constant Z = 1
```

**Scoring:**
- Part (a): 4 points (correct state space formulation)
- Part (b): 3 points (explanation of time-varying handling)

---

### Question 8 (8 points)
**Missing data problem:** You have weekly natural gas storage data, but three observations are missing due to holidays.

**(a)** How does the Kalman filter handle missing observations at times $t_{\text{missing}}$? Provide the algorithm modification. (5 points)

**(b)** What happens to the state uncertainty $P_t$ during the missing period? (3 points)

**Answer:**

**(a)** Kalman filter with missing data:

**Standard algorithm:**
1. **Prediction:** $\hat{\alpha}_{t|t-1}$, $P_{t|t-1}$
2. **Update:** $\hat{\alpha}_{t|t}$, $P_{t|t}$ (using $y_t$)

**When $y_t$ is missing:**

**Modified algorithm:**

```
IF observation y_t is missing:
    # SKIP UPDATE STEP
    # State estimate = prediction (no correction)
    α̂_t|t = α̂_t|t-1  (no update)
    P_t|t = P_t|t-1    (no variance reduction)

    # Proceed to next prediction
    α̂_t+1|t = T α̂_t|t + c
    P_t+1|t = T P_t|t T' + Q
ELSE:
    # Standard update using y_t
    [normal Kalman update]
END IF
```

**Mathematical justification:**
- Bayesian: $p(\alpha_t | y_1, ..., y_{t-1})$ (no $y_t$ to condition on)
- No information from missing obs → posterior = prior

**Alternative implementation:**
- Set $H_t = \infty$ (infinite observation noise) → Kalman gain $K_t \to 0$ → no update

**Key insight:** State still evolves according to transition equation, just without observational correction.

**(b)** State uncertainty during missing data:

**Uncertainty INCREASES** over missing period:

**Mechanism:**
$$P_{t|t-1} = P_{t-1|t-1} + Q$$

Without update:
$$P_{t|t} = P_{t|t-1} = P_{t-1|t-1} + Q$$

Then:
$$P_{t+1|t} = P_{t|t} + Q = P_{t-1|t-1} + 2Q$$

**Pattern:** For $k$ consecutive missing observations:
$$P_{t+k|t+k-1} = P_{t-1|t-1} + k \cdot Q$$

Linear growth in uncertainty!

**Intuition:**
- Each time step without data, state noise accumulates
- Model becomes less confident about latent state
- When data resumes, large Kalman gain (trust new observation)

**Example: Natural gas storage**
- Normal: $P_{t|t} \approx 100$ (Bcf)²
- After 3 missing weeks: $P_{t+3|t+2} \approx 100 + 3 \times 50 = 250$ (Bcf)²
- Uncertainty increased by 2.5x

**Visual:**
```
         Data available  |  Missing data  |  Data resumes
               ↓                           ↓
P_t     -----•---------              ---------•-----
            ↓ (update)                      ↓ (large update)
            Reduces                         Reduces sharply
                  ↗↗↗ (prediction accumulates)
```

**Scoring:**
- Part (a): 5 points (algorithm + justification)
- Part (b): 3 points (increasing uncertainty + mechanism)

---

## Section C: Bayesian State Space Models (35 points)

### Question 9 (10 points)
Implement a Bayesian local level model in PyMC for commodity prices. Write the model specification that estimates the observation and state noise variances ($\sigma_\epsilon^2, \sigma_\eta^2$) from data, not assuming they are known.

Include:
- Prior distributions for parameters
- State evolution
- Observation likelihood
- Justify your prior choices for a commodity price series (e.g., crude oil $50-$100/bbl).

**Answer:**

```python
import pymc as pm
import numpy as np

# Data: daily crude oil prices
y_observed = np.array([...])  # e.g., 100 days of prices
n_obs = len(y_observed)

with pm.Model() as bayesian_local_level:
    # ================================
    # PRIORS FOR VARIANCE PARAMETERS
    # ================================

    # Observation noise standard deviation
    # Prior: σ_ε ~ HalfNormal(5)
    # Justification: Daily oil price noise typically $2-5
    # HalfNormal constrains to positive
    sigma_obs = pm.HalfNormal('sigma_obs', sigma=5)

    # State noise standard deviation
    # Prior: σ_η ~ HalfNormal(1)
    # Justification: Daily drift in equilibrium price typically < $1
    # Smaller than obs noise (q < 1 for smooth level)
    sigma_level = pm.HalfNormal('sigma_level', sigma=1)

    # ================================
    # INITIAL STATE
    # ================================

    # Prior for initial level: μ_1 ~ Normal(75, 20)
    # Justification: Oil prices often $50-100, centered at $75
    # σ=20 allows wide range
    level_init = pm.Normal('level_init', mu=75, sigma=20)

    # ================================
    # STATE EVOLUTION (RANDOM WALK)
    # ================================

    # State innovations: η_t ~ Normal(0, 1)
    # Will be scaled by sigma_level
    level_innovations = pm.Normal('level_innov',
                                   mu=0,
                                   sigma=1,
                                   shape=n_obs-1)

    # Construct full level series:
    # μ_t = μ_{t-1} + σ_η * η_t
    level = pm.Deterministic(
        'level',
        pm.math.concatenate([
            [level_init],
            level_init + pm.math.cumsum(sigma_level * level_innovations)
        ])
    )

    # ================================
    # OBSERVATION LIKELIHOOD
    # ================================

    # y_t ~ Normal(μ_t, σ_ε)
    y_obs = pm.Normal('y_obs',
                      mu=level,
                      sigma=sigma_obs,
                      observed=y_observed)

    # ================================
    # INFERENCE
    # ================================

    # Sample posterior
    trace = pm.sample(2000,
                      tune=1000,
                      return_inferencedata=True,
                      target_accept=0.95)  # High for complex state space

    # Posterior predictive checks
    ppc = pm.sample_posterior_predictive(trace,
                                         return_inferencedata=True)

# ================================
# POSTERIOR ANALYSIS
# ================================

import arviz as az

# Examine variance estimates
print(az.summary(trace, var_names=['sigma_obs', 'sigma_level']))

# Plot filtered level with uncertainty
level_mean = trace.posterior['level'].mean(dim=['chain', 'draw'])
level_hdi = az.hdi(trace, var_names=['level'])

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(y_observed, 'o', alpha=0.5, label='Observed prices')
plt.plot(level_mean, 'r-', linewidth=2, label='Filtered level (posterior mean)')
plt.fill_between(range(n_obs),
                 level_hdi['level'].sel(hdi='lower'),
                 level_hdi['level'].sel(hdi='higher'),
                 alpha=0.3, color='red', label='95% HDI')
plt.xlabel('Day')
plt.ylabel('Price ($)')
plt.title('Bayesian Local Level Model: Crude Oil Prices')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Signal-to-noise ratio
q_samples = (trace.posterior['sigma_level'] / trace.posterior['sigma_obs'])**2
print(f"Signal-to-noise ratio q: {q_samples.mean():.3f} ± {q_samples.std():.3f}")
```

**Prior Justifications:**

1. **$\sigma_\epsilon$ ~ HalfNormal(5):**
   - Crude oil daily volatility: ~2-3% of $75 = $1.5-2.25
   - HalfNormal(5) allows this range but weakly regularizes against excessive noise

2. **$\sigma_\eta$ ~ HalfNormal(1):**
   - Equilibrium price drifts slowly (fundamental changes)
   - Smaller than obs noise (expect q < 1 for smoothing)
   - HalfNormal(1) concentrates on small values

3. **$\mu_1$ ~ Normal(75, 20):**
   - Weakly informative: allows $\mu_1 \in [35, 115]$ (95% interval)
   - Centered at historical average

**Model properties:**
- Jointly estimates level and variances (full Bayesian)
- Uncertainty quantification for both states and parameters
- Can compare models using WAIC/LOO

**Scoring:**
- Correct PyMC structure: 4 points
- Prior specifications: 3 points
- Prior justifications: 3 points

---

### Question 10 (9 points)
**Model comparison problem:** You fit three state space models to weekly natural gas prices:

1. **Local Level:** $y_t = \mu_t + \epsilon_t$, $\mu_{t+1} = \mu_t + \eta_t$
2. **Local Linear Trend:** Adds trend $\nu_t$
3. **Basic Structural Model:** Adds seasonal component

Results:

| Model | WAIC | LOO | In-sample RMSE |
|-------|------|-----|---------------|
| Local Level | 1250 | 1255 | 1.85 |
| Local Linear Trend | 1220 | 1230 | 1.72 |
| Basic Structural Model | 1180 | 1190 | 1.58 |

**(a)** Which model would you select and why? (3 points)

**(b)** Why might in-sample RMSE be misleading for model selection? (3 points)

**(c)** The seasonal model has 15 more parameters than local level. How do WAIC/LOO account for this complexity? (3 points)

**Answer:**

**(a)** Model selection:

**Choose Basic Structural Model (BSM)**

**Reasoning:**
1. **Lowest WAIC and LOO:** 1180/1190 vs 1220/1230 vs 1250/1255
   - WAIC/LOO are out-of-sample predictive accuracy estimates
   - Lower is better
   - Difference of ~40 points is substantial (>10 is often meaningful)

2. **Model makes sense for natural gas:**
   - Strong annual seasonality (winter heating, summer cooling)
   - Trend captures long-term supply/demand shifts
   - BSM is theoretically appropriate

3. **WAIC ≈ LOO:**
   - Close agreement suggests stable estimation
   - If large discrepancy, would indicate outlier sensitivity

**Caveat:**
- Always validate with out-of-sample forecast evaluation
- WAIC/LOO are approximations to cross-validation

**(b)** Why in-sample RMSE is misleading:

**Problem 1: Overfitting**
- More complex models always fit training data better
- RMSE decreases with parameters even if fit is spurious
- BSM has more parameters → lower in-sample error by construction

**Problem 2: No penalty for complexity**
- RMSE: $\sqrt{\frac{1}{n}\sum (y_t - \hat{y}_t)^2}$
- Doesn't account for degrees of freedom consumed
- Can memorize noise rather than learn signal

**Problem 3: Not predictive**
- In-sample: Using data that estimated the parameters
- Out-of-sample: True test of generalization
- Model with best in-sample fit may perform poorly on new data

**Example:**
- Overfitted model: Low in-sample RMSE, high out-of-sample RMSE
- Well-regularized model: Moderate in-sample RMSE, low out-of-sample RMSE

**Better metrics:**
- WAIC/LOO: Approximate out-of-sample performance
- Cross-validation: Actual holdout evaluation
- True out-of-sample forecast period

**(c)** How WAIC/LOO account for complexity:

**WAIC (Widely Applicable Information Criterion):**
$$\text{WAIC} = -2 \times (\text{lppd} - p_{\text{WAIC}})$$

Where:
- $\text{lppd}$ = log pointwise predictive density (fit to data)
- $p_{\text{WAIC}}$ = effective number of parameters (complexity penalty)

**Effective parameters:**
$$p_{\text{WAIC}} = \sum_{i=1}^n \text{Var}_{\text{post}}(\log p(y_i | \theta))$$

**Intuition:**
- If posterior variance of log-likelihood is high → parameter is "used" (data-dependent)
- If low → parameter not learning from data (rigid prior)
- Automatically penalizes parameters that overfit

**LOO (Leave-One-Out Cross-Validation):**
$$\text{LOO} = -2 \sum_{i=1}^n \log p(y_i | y_{-i})$$

- $p(y_i | y_{-i})$ = predictive density for $y_i$ after fitting to all other data
- Exact cross-validation approximated via Pareto Smoothed Importance Sampling (PSIS)

**Why these are better than AIC/BIC for Bayesian models:**
- AIC/BIC assume simple parameter count
- WAIC/LOO use **effective parameters** (accounts for priors shrinking estimates)
- BSM with strong priors may have $p_{\text{eff}} < p$ (nominal parameters)

**Result interpretation:**
- BSM: 15 more parameters, but WAIC/LOO still prefer it
- Implies seasonal parameters genuinely improve out-of-sample prediction
- Penalty for complexity is outweighed by better fit

**Scoring:**
- Part (a): 3 points (correct choice + reasoning)
- Part (b): 3 points (overfitting issues)
- Part (c): 3 points (complexity penalty explanation)

---

### Question 11 (8 points)
Compare the classical Kalman filter (frequentist) to the Bayesian state space approach in PyMC for commodity forecasting. Discuss two advantages and one disadvantage of the Bayesian approach.

**Answer:**

**Classical Kalman Filter (Frequentist):**
- Parameters $(\sigma_\epsilon^2, \sigma_\eta^2, ...)$ estimated separately (e.g., MLE)
- Then treat as known constants in filtering
- Fast, closed-form recursions

**Bayesian State Space (PyMC):**
- Joint posterior over states AND parameters
- Parameters have uncertainty quantified by posterior distributions
- MCMC sampling required (slower)

---

**Advantage 1: Parameter Uncertainty Propagated**

**Classical:**
- Plug-in estimates: $\hat{\sigma}_\epsilon, \hat{\sigma}_\eta$
- Forecasts conditional on these point estimates
- Underestimates true predictive uncertainty

**Bayesian:**
- Full posterior: $p(\alpha_t, \sigma_\epsilon, \sigma_\eta | y_1, ..., y_t)$
- Forecasts integrate over parameter uncertainty:
  $$p(y_{t+h} | y_1, ..., y_t) = \int p(y_{t+h} | \alpha_t, \theta) p(\alpha_t, \theta | y_1, ..., y_t) d\alpha_t d\theta$$
- **More honest uncertainty quantification**

**Example:**
- If unsure whether $\sigma_\eta = 0.5$ or $1.0$, Bayesian forecast intervals are wider (correctly)

---

**Advantage 2: Incorporate Domain Knowledge via Priors**

**Classical:**
- No natural mechanism for prior information
- Estimates purely data-driven

**Bayesian:**
- Priors encode expert beliefs
- **Example:** For oil prices, might know:
  - Daily volatility typically $\sigma_\epsilon \in [2, 5]$
  - Equilibrium price drifts slowly: $\sigma_\eta < \sigma_\epsilon$
- Priors regularize, prevent overfitting with limited data

**Especially valuable when:**
- Small sample size (recent commodity data only)
- Parameter estimation uncertain (high MLE standard errors)
- Multiple plausible models (Bayesian model averaging)

---

**Disadvantage: Computational Cost**

**Classical:**
- Kalman filter: $O(n)$ operations for $n$ time points
- Real-time, scales to large datasets
- Can run in production systems easily

**Bayesian (MCMC):**
- Need to sample posterior: 1000-5000 draws typical
- Each draw requires forward-backward pass through data
- **Computational complexity:** $O(n \times S)$ where $S$ = number of samples
- Can take minutes to hours for long series

**Practical impact:**
- High-frequency trading: Classical preferred (speed critical)
- Daily/weekly forecasting: Bayesian feasible and valuable
- Research/model development: Bayesian for understanding, classical for deployment

**Mitigation:**
- Variational inference (faster, approximate)
- Pre-train model, then use classical filter with posterior mean parameters
- GPU acceleration (PyMC supports)

---

**Summary Table:**

| Aspect | Classical Kalman | Bayesian PyMC |
|--------|------------------|---------------|
| Speed | Fast ($O(n)$) | Slow ($O(nS)$) |
| Parameter uncertainty | Ignored | Propagated |
| Prior information | Cannot incorporate | Natural |
| Interpretability | Point estimates | Full distributions |
| Use case | Real-time, production | Research, decision-support |

**Scoring:**
- Advantage 1 (parameter uncertainty): 3 points
- Advantage 2 (priors): 3 points
- Disadvantage (computation): 2 points

---

### Question 12 (8 points)
You want to forecast crude oil prices 4 weeks ahead using a Bayesian local linear trend model. After obtaining the posterior distribution of the level $\mu_T$ and trend $\nu_T$ at the last observed time $T$:

**(a)** Write the formula for the 4-week ahead forecast distribution $y_{T+4}$. (4 points)

**(b)** Explain how the forecast uncertainty decomposes into different sources. (4 points)

**Answer:**

**(a)** Four-week ahead forecast distribution:

**State evolution from $T$ to $T+4$:**

$$\begin{aligned}
\mu_{T+1} &= \mu_T + \nu_T + \eta_{T+1} \\
\nu_{T+1} &= \nu_T + \zeta_{T+1} \\
\mu_{T+2} &= \mu_{T+1} + \nu_{T+1} + \eta_{T+2} \\
&= \mu_T + 2\nu_T + \sum_{i=1}^{2}\eta_{T+i} + \zeta_{T+1} \\
\mu_{T+3} &= \mu_T + 3\nu_T + \sum_{i=1}^{3}\eta_{T+i} + \sum_{j=1}^{2}\zeta_{T+j} \\
\mu_{T+4} &= \mu_T + 4\nu_T + \sum_{i=1}^{4}\eta_{T+i} + \sum_{j=1}^{3}\zeta_{T+j}
\end{aligned}$$

**Observation at $T+4$:**
$$y_{T+4} = \mu_{T+4} + \epsilon_{T+4}$$
$$y_{T+4} = \mu_T + 4\nu_T + \sum_{i=1}^{4}\eta_{T+i} + \sum_{j=1}^{3}\zeta_{T+j} + \epsilon_{T+4}$$

**Forecast distribution (conditional on $\mu_T, \nu_T$):**

**Mean:**
$$\mathbb{E}[y_{T+4} | \mu_T, \nu_T] = \mu_T + 4\nu_T$$

**Variance:**
$$\text{Var}(y_{T+4} | \mu_T, \nu_T) = 4\sigma_\eta^2 + 3\sigma_\zeta^2 + \sigma_\epsilon^2$$

**Full Bayesian forecast (integrating over parameter uncertainty):**
$$p(y_{T+4} | y_1, ..., y_T) = \int p(y_{T+4} | \mu_T, \nu_T, \theta) p(\mu_T, \nu_T, \theta | y_1, ..., y_T) d\mu_T d\nu_T d\theta$$

where $\theta = (\sigma_\eta, \sigma_\zeta, \sigma_\epsilon)$.

**In PyMC:**
```python
# Sample from posterior predictive distribution
with model:
    # Extend state forward 4 steps
    mu_forecast = mu_T + 4 * nu_T + pm.Normal('eta_future', 0, sigma_eta, shape=4).cumsum()
    # (simplified; proper implementation tracks nu evolution too)

    y_forecast = pm.Normal('y_T+4', mu=mu_forecast[-1], sigma=sigma_obs)

    forecast_samples = pm.sample_posterior_predictive(trace, var_names=['y_T+4'])
```

**(b)** Decomposition of forecast uncertainty:

**Total variance:** $\text{Var}(y_{T+4}) = \sigma_{\text{level}}^2 + \sigma_{\text{trend}}^2 + \sigma_{\text{obs}}^2 + \sigma_{\text{param}}^2$

**1. Level noise ($4\sigma_\eta^2$):**
- Accumulation of random shocks to level
- Grows linearly with horizon $h$: $h\sigma_\eta^2$
- **Source:** Stochastic drift in equilibrium price
- **Example:** OPEC policy surprises, supply disruptions

**2. Trend noise ($3\sigma_\zeta^2$):**
- Accumulation of shocks to trend (slope changes)
- Grows with horizon: $(h-1)\sigma_\zeta^2$ for $h$-step ahead
- **Source:** Changes in growth rate (structural shifts)
- **Example:** Shift from oil glut to deficit regime

**3. Observation noise ($\sigma_\epsilon^2$):**
- Idiosyncratic noise at forecast time
- **Constant** across horizons (only affects final observation)
- **Source:** Microstructure, measurement error
- **Example:** Bid-ask bounce, intraday volatility

**4. Parameter uncertainty ($\sigma_{\text{param}}^2$):**
- Uncertainty about $\sigma_\eta, \sigma_\zeta, \sigma_\epsilon$ themselves
- Bayesian approach: posterior variance of parameters contributes
- Classical approach: ignores this (underestimates uncertainty)
- **Source:** Finite sample estimation error

**Relative importance:**
```
1-step ahead:  Dominated by observation noise
Short horizon: σ_ε² largest
Long horizon:  Level/trend noise dominate (accumulate)
               σ_obs² becomes negligible fraction
```

**Graphical representation:**
```
Variance
  │
  │                    ╱ Total (all sources)
  │                  ╱
  │                ╱
  │    State noise (level+trend)
  │          ╱
  │      ╱╱
  │  ╱╱
  │╱_____________ Observation noise (constant)
  └────────────────────> Forecast horizon h
  0   1   2   3   4
```

**Practical implication:**
- **Short-term forecasts:** Reduce observation noise (better data, higher frequency)
- **Long-term forecasts:** Improve state model (add fundamentals, better dynamics)

**Scoring:**
- Part (a): 4 points (forecast formula and variance decomposition)
- Part (b): 4 points (sources of uncertainty explained)

---

## Answer Key Summary

| Question | Points | Topic |
|----------|--------|-------|
| 1 | 8 | State space definitions, AR(1) |
| 2 | 7 | Local level, signal-to-noise ratio |
| 3 | 8 | Local linear trend, state space matrices |
| 4 | 7 | Filter vs smooth vs predict |
| 5 | 12 | Kalman filter equations |
| 6 | 8 | Innovation diagnostics |
| 7 | 7 | Time-varying components |
| 8 | 8 | Missing data handling |
| 9 | 10 | PyMC implementation |
| 10 | 9 | Model comparison WAIC/LOO |
| 11 | 8 | Bayesian vs classical comparison |
| 12 | 8 | Forecast uncertainty decomposition |
| **Total** | **100** | |

---

## Grading Rubric

**A (90-100):** Mastery of state space theory, Kalman filtering mechanics, and Bayesian implementation. Can derive equations and apply to commodity contexts.

**B (80-89):** Strong understanding with minor technical gaps. May struggle with matrix notation or advanced PyMC specifications.

**C (70-79):** Basic competency in state space concepts but lacks depth in mathematical derivations or practical implementation.

**D (60-69):** Significant conceptual gaps. Difficulty with Kalman filter equations or Bayesian inference.

**F (<60):** Does not meet minimum standards for graduate-level state space modeling.

---

**Study Resources:**
- Durbin & Koopman (2012): *Time Series Analysis by State Space Methods*
- Harvey (1989): *Forecasting, Structural Time Series Models and the Kalman Filter*
- PyMC Documentation: State Space Models examples
- Module notebooks: `02_kalman_filter.ipynb`, `03_bayesian_state_space.ipynb`
