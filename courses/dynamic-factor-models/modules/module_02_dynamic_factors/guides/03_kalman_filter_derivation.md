# Kalman Filter: Derivation and Intuition

> **Reading time:** ~17 min | **Module:** Module 2: Dynamic Factors | **Prerequisites:** Modules 0-1

<div class="callout-key">

**Key Concept Summary:** The Kalman filter is the optimal recursive algorithm for estimating the state of a linear Gaussian state-space model. It produces minimum mean squared error estimates by combining model predictions with new observations, updating both state estimates and their uncertainty at each time step.

</div>

## In Brief

The Kalman filter is the optimal recursive algorithm for estimating the state of a linear Gaussian state-space model. It produces minimum mean squared error estimates by combining model predictions with new observations, updating both state estimates and their uncertainty at each time step.

<div class="callout-insight">

**Insight:** The Kalman filter is Bayesian updating in action. At each time step, you have two sources of information: (1) where the model predicts the state should be, and (2) where the new observations suggest it is. The Kalman filter optimally weights these two sources based on their relative uncertainties, producing filtered state estimates and a likelihood for the parameters. It's not just a clever algorithm—it's provably optimal under Gaussian assumptions, and the foundation for all modern dynamic factor model estimation.

</div>
---

<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## 1. The Filtering Problem

### Setup

Given state-space model:
$$\begin{align}
y_t &= Z \alpha_t + \epsilon_t, \quad \epsilon_t \sim N(0, H) \\
\alpha_t &= T \alpha_{t-1} + R\eta_t, \quad \eta_t \sim N(0, Q)
\end{align}$$

**Goal:** Compute $\hat{\alpha}_{t|t} = E[\alpha_t | y_1, ..., y_t]$ (filtered state estimate).

### Information Sets

| Notation | Meaning | What we know |
|----------|---------|--------------|
| $\hat{\alpha}_{t\|t-1}$ | Predicted state | Based on $y_1, ..., y_{t-1}$ |
| $P_{t\|t-1}$ | Prediction error variance | $\text{Var}(\alpha_t \| y_1, ..., y_{t-1})$ |
| $\hat{\alpha}_{t\|t}$ | Filtered state | Based on $y_1, ..., y_t$ |
| $P_{t\|t}$ | Filtering error variance | $\text{Var}(\alpha_t \| y_1, ..., y_t)$ |
| $\hat{\alpha}_{t\|T}$ | Smoothed state | Based on all data $y_1, ..., y_T$ |
| $P_{t\|T}$ | Smoothing error variance | $\text{Var}(\alpha_t \| y_1, ..., y_T)$ |

**Key relationship:** $P_{t|T} \leq P_{t|t} \leq P_{t|t-1}$ (more data reduces uncertainty).

---

## 2. Kalman Filter Recursions

### The Algorithm

**Initialization:**
$$\hat{\alpha}_{1|0} = a_1, \quad P_{1|0} = P_1$$

**For $t = 1, 2, ..., T$:**

**Prediction Step:**
$$\begin{align}
\hat{\alpha}_{t|t-1} &= T \hat{\alpha}_{t-1|t-1} \tag{State prediction} \\
P_{t|t-1} &= T P_{t-1|t-1} T' + R Q R' \tag{Variance prediction}
\end{align}$$

**Update Step:**
$$\begin{align}
v_t &= y_t - Z \hat{\alpha}_{t|t-1} \tag{Prediction error} \\
F_t &= Z P_{t|t-1} Z' + H \tag{Prediction error variance} \\
K_t &= P_{t|t-1} Z' F_t^{-1} \tag{Kalman gain} \\
\hat{\alpha}_{t|t} &= \hat{\alpha}_{t|t-1} + K_t v_t \tag{State update} \\
P_{t|t} &= P_{t|t-1} - K_t Z P_{t|t-1} \tag{Variance update}
\end{align}$$

### Intuition for Each Step

**Prediction:**
- "Where do I expect the state to be based on its dynamics?"
- Propagate previous estimate through transition equation
- Uncertainty increases (add $RQR'$) because of state innovations

**Prediction error:**
- "How far off was my prediction from actual observation?"
- $v_t = 0$ if observation matches prediction perfectly
- Large $|v_t|$ signals model misspecification or unusual shock

**Kalman gain:**
- "How much should I trust the new observation vs my prediction?"
- $K_t \approx 0$: Don't trust observation (large $H$, small $P_{t|t-1}$)
- $K_t \approx P_{t|t-1}Z'H^{-1}$: Trust observation (small $H$, large $P_{t|t-1}$)

**State update:**
- Bayesian posterior mean: weighted average of prediction and observation
- More weight on observation when $K_t$ is large
- More weight on prediction when $K_t$ is small

**Variance update:**
- Uncertainty decreases after observing $y_t$
- $P_{t|t} < P_{t|t-1}$ always (information never hurts)
- Reduction depends on Kalman gain magnitude

---

## 3. Derivation from First Principles

### Prediction Step (Trivial)

From transition equation:
$$E[\alpha_t | y_1, ..., y_{t-1}] = E[T\alpha_{t-1} + R\eta_t | y_1, ..., y_{t-1}]$$

Since $\eta_t$ is independent of past and has $E[\eta_t] = 0$:
$$\hat{\alpha}_{t|t-1} = T \hat{\alpha}_{t-1|t-1}$$

For variance:
$$\begin{align}
P_{t|t-1} &= E[(\alpha_t - \hat{\alpha}_{t|t-1})(\alpha_t - \hat{\alpha}_{t|t-1})' | y_1, ..., y_{t-1}] \\
&= E[(T(\alpha_{t-1} - \hat{\alpha}_{t-1|t-1}) + R\eta_t)(T(\alpha_{t-1} - \hat{\alpha}_{t-1|t-1}) + R\eta_t)' | \cdot] \\
&= T E[(\alpha_{t-1} - \hat{\alpha}_{t-1|t-1})(\alpha_{t-1} - \hat{\alpha}_{t-1|t-1})']T' + R E[\eta_t\eta_t']R' \\
&= T P_{t-1|t-1} T' + R Q R'
\end{align}$$

### Update Step (The Heart of Kalman Filter)

**Setup:** We have prediction $\hat{\alpha}_{t|t-1}$ and observe $y_t$. Want $\hat{\alpha}_{t|t}$.

**Key insight:** Under Gaussianity, posterior is also Gaussian. Just need to find mean and variance.

**Joint distribution:**
$$\begin{bmatrix} \alpha_t \\ y_t \end{bmatrix} \Bigg| y_1, ..., y_{t-1} \sim N\left( \begin{bmatrix} \hat{\alpha}_{t|t-1} \\ Z\hat{\alpha}_{t|t-1} \end{bmatrix}, \begin{bmatrix} P_{t|t-1} & P_{t|t-1}Z' \\ ZP_{t|t-1} & F_t \end{bmatrix} \right)$$

where $F_t = ZP_{t|t-1}Z' + H$ is the marginal variance of $y_t$.

**Conditional distribution:** By standard multivariate normal conditioning formula:
$$\alpha_t | y_t, y_1, ..., y_{t-1} \sim N(\hat{\alpha}_{t|t}, P_{t|t})$$

where:
$$\begin{align}
\hat{\alpha}_{t|t} &= \hat{\alpha}_{t|t-1} + P_{t|t-1}Z'F_t^{-1}(y_t - Z\hat{\alpha}_{t|t-1}) \\
&= \hat{\alpha}_{t|t-1} + K_t v_t \\
P_{t|t} &= P_{t|t-1} - P_{t|t-1}Z'F_t^{-1}ZP_{t|t-1} \\
&= P_{t|t-1} - K_t Z P_{t|t-1}
\end{align}$$

This is just Bayesian updating for Gaussian distributions.

### Alternative Derivation: Projection

**Goal:** Find $\hat{\alpha}_{t|t}$ that minimizes $E[(\alpha_t - \hat{\alpha}_{t|t})^2 | y_1, ..., y_t]$.

**Solution:** Projection theorem from Hilbert space theory gives:
$$\hat{\alpha}_{t|t} = \hat{\alpha}_{t|t-1} + K_t v_t$$

where $K_t$ is chosen to make the error $\alpha_t - \hat{\alpha}_{t|t}$ orthogonal to $v_t$.

Setting $\text{Cov}(\alpha_t - \hat{\alpha}_{t|t}, v_t) = 0$ and solving gives the Kalman gain formula.

---

## 4. Log-Likelihood Computation

### Prediction Error Decomposition

The Kalman filter produces the likelihood as a byproduct!

**Key fact:** The prediction errors $v_t$ are the innovations of the data:
$$y_t | y_1, ..., y_{t-1} \sim N(Z\hat{\alpha}_{t|t-1}, F_t)$$

So:
$$v_t | y_1, ..., y_{t-1} \sim N(0, F_t)$$

**Joint likelihood:**
$$p(y_1, ..., y_T | \theta) = p(y_1|\theta) \prod_{t=2}^T p(y_t | y_1, ..., y_{t-1}, \theta)$$

Each conditional density:
$$p(y_t | y_1, ..., y_{t-1}, \theta) = (2\pi)^{-N/2} |F_t|^{-1/2} \exp\left(-\frac{1}{2}v_t' F_t^{-1} v_t\right)$$

**Log-likelihood:**
$$\log L(\theta) = -\frac{NT}{2}\log(2\pi) - \frac{1}{2}\sum_{t=1}^T \left[\log|F_t| + v_t'F_t^{-1}v_t\right]$$

**Practical note:**
- Kalman filter loop computes $v_t$ and $F_t$ at each $t$
- Accumulate $\sum_t [\log|F_t| + v_t'F_t^{-1}v_t]$ during filtering
- Return log-likelihood at end

---

## 5. Code Implementation: From-Scratch Kalman Filter

### Basic Implementation

```python
import numpy as np
from numpy.linalg import inv, slogdet

def kalman_filter(y, Z, H, T, R, Q, a1, P1):
    """
    Kalman filter for state-space model.

    Parameters
    ----------
    y : array (T, N)
        Observations (can contain np.nan for missing values)
    Z : array (N, m)
        Measurement matrix
    H : array (N, N)
        Measurement error covariance
    T : array (m, m)
        Transition matrix
    R : array (m, r)
        Selection matrix
    Q : array (r, r)
        Innovation covariance
    a1 : array (m,)
        Initial state mean
    P1 : array (m, m)
        Initial state covariance

    Returns
    -------
    result : dict
        - 'alpha_filtered': array (T, m) - filtered states
        - 'alpha_predicted': array (T, m) - predicted states
        - 'P_filtered': array (T, m, m) - filtered covariances
        - 'P_predicted': array (T, m, m) - predicted covariances
        - 'v': array (T, N) - prediction errors
        - 'F': array (T, N, N) - prediction error covariances
        - 'K': array (T, m, N) - Kalman gains
        - 'loglik': float - log-likelihood
    """
    T_periods, N = y.shape
    m = Z.shape[1]
    r = Q.shape[0]

    # Storage
    alpha_predicted = np.zeros((T_periods, m))
    alpha_filtered = np.zeros((T_periods, m))
    P_predicted = np.zeros((T_periods, m, m))
    P_filtered = np.zeros((T_periods, m, m))
    v = np.zeros((T_periods, N))
    F = np.zeros((T_periods, N, N))
    K = np.zeros((T_periods, m, N))

    # Log-likelihood accumulator
    loglik = 0.0

    # Initialize
    alpha_filt = a1.copy()
    P_filt = P1.copy()

    for t in range(T_periods):
        # --- PREDICTION STEP ---
        alpha_pred = T @ alpha_filt
        P_pred = T @ P_filt @ T.T + R @ Q @ R.T

        # Store predictions
        alpha_predicted[t] = alpha_pred
        P_predicted[t] = P_pred

        # --- UPDATE STEP ---
        # Handle missing values
        if np.any(np.isnan(y[t])):
            # If any observation missing, skip update (state = prediction)
            alpha_filt = alpha_pred
            P_filt = P_pred
            v[t] = np.nan
            F[t] = np.nan
            K[t] = np.nan
        else:
            # Prediction error
            v_t = y[t] - Z @ alpha_pred

            # Prediction error variance
            F_t = Z @ P_pred @ Z.T + H

            # Kalman gain
            K_t = P_pred @ Z.T @ inv(F_t)

            # State update
            alpha_filt = alpha_pred + K_t @ v_t

            # Variance update
            P_filt = P_pred - K_t @ Z @ P_pred

            # Store
            v[t] = v_t
            F[t] = F_t
            K[t] = K_t

            # Log-likelihood contribution
            sign, logdet_F = slogdet(F_t)
            loglik += -0.5 * (N * np.log(2*np.pi) + logdet_F + v_t @ inv(F_t) @ v_t)

        # Store filtered state
        alpha_filtered[t] = alpha_filt
        P_filtered[t] = P_filt

    return {
        'alpha_filtered': alpha_filtered,
        'alpha_predicted': alpha_predicted,
        'P_filtered': P_filtered,
        'P_predicted': P_predicted,
        'v': v,
        'F': F,
        'K': K,
        'loglik': loglik
    }
```

### Testing the Implementation

```python
# Use state-space model from previous guide
from scipy.linalg import solve_discrete_lyapunov

# Define DFM parameters
np.random.seed(42)
N, r, p = 10, 2, 1
T_periods = 200

Lambda = np.random.randn(N, r)
Phi = np.array([[0.7, 0.1], [0.2, 0.6]])
Sigma_e = np.diag(np.random.uniform(0.1, 0.5, N))
Q = np.eye(r) * 0.5

# State-space matrices
Z = Lambda
H = Sigma_e
T = Phi
R = np.eye(r)

# Initial state
a1 = np.zeros(r)
P1 = solve_discrete_lyapunov(T, R @ Q @ R.T)

# Simulate data
def simulate_statespace(Z, H, T, R, Q, T_periods, a1, P1):
    """Simulate from state-space model."""
    N, m = Z.shape
    r = Q.shape[0]

    alpha = np.zeros((T_periods, m))
    y = np.zeros((T_periods, N))

    # Initialize
    alpha[0] = np.random.multivariate_normal(a1, P1)

    for t in range(T_periods):
        # Measurement
        epsilon = np.random.multivariate_normal(np.zeros(N), H)
        y[t] = Z @ alpha[t] + epsilon

        # Transition
        if t < T_periods - 1:
            eta = np.random.multivariate_normal(np.zeros(r), Q)
            alpha[t+1] = T @ alpha[t] + R @ eta

    return y, alpha

y, alpha_true = simulate_statespace(Z, H, T, R, Q, T_periods, a1, P1)

# Run Kalman filter
result = kalman_filter(y, Z, H, T, R, Q, a1, P1)

print(f"Log-likelihood: {result['loglik']:.2f}")
print(f"Mean prediction error: {np.nanmean(np.abs(result['v'])):.4f}")
print(f"Filtered state shape: {result['alpha_filtered'].shape}")
```

### Visualization

```python
import matplotlib.pyplot as plt

# Extract filtered factors
F_filtered = result['alpha_filtered']

# Plot true vs filtered factors
fig, axes = plt.subplots(r, 1, figsize=(12, 6), sharex=True)

for i in range(r):
    axes[i].plot(alpha_true[:, i], label='True Factor', linewidth=2, alpha=0.7)
    axes[i].plot(F_filtered[:, i], label='Filtered Estimate',
                 linewidth=1.5, alpha=0.8, linestyle='--')

    # Add uncertainty bands (±2 std)
    P_filt = result['P_filtered']
    std = np.sqrt(P_filt[:, i, i])
    axes[i].fill_between(range(T_periods),
                         F_filtered[:, i] - 2*std,
                         F_filtered[:, i] + 2*std,
                         alpha=0.2, label='95% Confidence')

    axes[i].set_ylabel(f'Factor {i+1}', fontsize=11)
    axes[i].legend(loc='upper right')
    axes[i].grid(alpha=0.3)

axes[-1].set_xlabel('Time', fontsize=11)
axes[0].set_title('True vs Filtered Factor Estimates', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# Compute correlation
for i in range(r):
    corr = np.corrcoef(alpha_true[:, i], F_filtered[:, i])[0, 1]
    print(f"Correlation (True vs Filtered) for Factor {i+1}: {corr:.4f}")
```

---

## 6. Kalman Smoother

### Motivation

Filtered estimate $\hat{\alpha}_{t|t}$ uses data up to time $t$.

Smoothed estimate $\hat{\alpha}_{t|T}$ uses all data $y_1, ..., y_T$.

**Result:** $\text{Var}(\alpha_t | y_1, ..., y_T) \leq \text{Var}(\alpha_t | y_1, ..., y_t)$

Smoothing improves estimates, especially for middle time periods.

### Smoothing Recursions (RTS Algorithm)

After running Kalman filter forward ($t = 1, ..., T$), run smoother backward ($t = T-1, ..., 1$).

**Initialization:**
$$\hat{\alpha}_{T|T} = \hat{\alpha}_{T|T}, \quad P_{T|T} = P_{T|T}$$

**For $t = T-1, T-2, ..., 1$:**
$$\begin{align}
J_t &= P_{t|t} T' P_{t+1|t}^{-1} \tag{Smoother gain} \\
\hat{\alpha}_{t|T} &= \hat{\alpha}_{t|t} + J_t(\hat{\alpha}_{t+1|T} - \hat{\alpha}_{t+1|t}) \tag{Smoothed state} \\
P_{t|T} &= P_{t|t} + J_t(P_{t+1|T} - P_{t+1|t})J_t' \tag{Smoothed variance}
\end{align}$$

**Intuition:**
- $\hat{\alpha}_{t+1|T} - \hat{\alpha}_{t+1|t}$: How much future data adjusted next period's estimate
- $J_t$: Propagate this correction backward to time $t$
- Smoothed estimate pulls filtered estimate toward where future data says it should be

### Implementation

```python
def kalman_smoother(result_filter, T):
    """
    Rauch-Tung-Striebel (RTS) smoother.

    Parameters
    ----------
    result_filter : dict
        Output from kalman_filter()
    T : array (m, m)
        Transition matrix

    Returns
    -------
    alpha_smoothed : array (T, m)
        Smoothed state estimates
    P_smoothed : array (T, m, m)
        Smoothed state covariances
    """
    T_periods, m = result_filter['alpha_filtered'].shape

    # Storage
    alpha_smoothed = np.zeros((T_periods, m))
    P_smoothed = np.zeros((T_periods, m, m))

    # Initialize at T
    alpha_smoothed[-1] = result_filter['alpha_filtered'][-1]
    P_smoothed[-1] = result_filter['P_filtered'][-1]

    # Backward recursion
    for t in range(T_periods-2, -1, -1):
        # Smoother gain
        P_pred_next = result_filter['P_predicted'][t+1]
        J_t = result_filter['P_filtered'][t] @ T.T @ inv(P_pred_next)

        # Smoothed state
        alpha_diff = alpha_smoothed[t+1] - result_filter['alpha_predicted'][t+1]
        alpha_smoothed[t] = result_filter['alpha_filtered'][t] + J_t @ alpha_diff

        # Smoothed variance
        P_diff = P_smoothed[t+1] - P_pred_next
        P_smoothed[t] = result_filter['P_filtered'][t] + J_t @ P_diff @ J_t.T

    return alpha_smoothed, P_smoothed
```

### Comparing Filtered vs Smoothed


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# Run smoother
alpha_smoothed, P_smoothed = kalman_smoother(result, T)

# Plot comparison
fig, axes = plt.subplots(r, 1, figsize=(12, 6), sharex=True)

for i in range(r):
    axes[i].plot(alpha_true[:, i], label='True', linewidth=2, color='black', alpha=0.6)
    axes[i].plot(F_filtered[:, i], label='Filtered', linewidth=1.5,
                 linestyle='--', alpha=0.8)
    axes[i].plot(alpha_smoothed[:, i], label='Smoothed', linewidth=1.5,
                 linestyle=':', alpha=0.8)

    axes[i].set_ylabel(f'Factor {i+1}', fontsize=11)
    axes[i].legend(loc='upper right')
    axes[i].grid(alpha=0.3)

axes[-1].set_xlabel('Time', fontsize=11)
axes[0].set_title('Filtered vs Smoothed Factor Estimates', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# Compare MSE
for i in range(r):
    mse_filt = np.mean((alpha_true[:, i] - F_filtered[:, i])**2)
    mse_smooth = np.mean((alpha_true[:, i] - alpha_smoothed[:, i])**2)
    print(f"Factor {i+1} MSE - Filtered: {mse_filt:.4f}, Smoothed: {mse_smooth:.4f}")
    print(f"  Improvement: {(1 - mse_smooth/mse_filt)*100:.1f}%")
```

</div>

---

## 7. Advanced Topics

### Steady-State Kalman Filter

If system is stationary and no missing data, $P_{t|t-1}$ converges to steady-state $\bar{P}$.

**Steady-state prediction covariance:** Solve discrete-time algebraic Riccati equation:
$$\bar{P} = T\bar{P}T' + RQR' - T\bar{P}Z'(Z\bar{P}Z' + H)^{-1}Z\bar{P}T'$$

**Benefits:**
- No need to update $P_t$ each iteration (computationally faster)
- Kalman gain constant: $\bar{K} = \bar{P}Z'(Z\bar{P}Z' + H)^{-1}$

**When to use:**
- Long time series ($T > 100$)
- No missing data
- Time-invariant system

### Diffuse Initialization

When initial state is unknown, use "diffuse" initialization:
$$P_1 = \kappa I_m, \quad \kappa \to \infty$$

**Practical implementation:**
- Set $\kappa = 10^7$ (large but finite)
- First few filtering iterations have high uncertainty
- Filter converges to data-driven estimates quickly

**Exact diffuse initialization (Durbin-Koopman):**
- Decompose $P_1 = P_* + \kappa P_\infty$
- Run parallel filters for finite and diffuse parts
- Combine when diffuse part becomes negligible

### Missing Data

State-space framework handles missing data naturally.

**Implementation:**
1. At time $t$, identify missing indices in $y_t$
2. Remove corresponding rows from $Z$, $H$ for that $t$
3. Run update step with reduced-dimension observation
4. Prediction step unchanged

**Example:**

<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# Mark missing observations
y_missing = y.copy()
y_missing[50:60, 3] = np.nan  # Variable 3 missing
y_missing[100:110, [5, 7]] = np.nan  # Variables 5, 7 missing

# Kalman filter handles automatically (see implementation above)
result_missing = kalman_filter(y_missing, Z, H, T, R, Q, a1, P1)
```

</div>

### Forecasting

Once Kalman filter completes, forecasting is straightforward.

**h-step ahead forecast:**
$$\hat{\alpha}_{T+h|T} = T^h \hat{\alpha}_{T|T}$$

**Forecast variance:**
$$P_{T+h|T} = T^h P_{T|T} (T^h)' + \sum_{j=0}^{h-1} T^j R Q R' (T^j)'$$

**Observation forecast:**
$$\hat{y}_{T+h|T} = Z \hat{\alpha}_{T+h|T}$$

**Implementation:**

<span class="filename">forecast_statespace.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def forecast_statespace(result_filter, Z, T, R, Q, horizons):
    """
    Forecast h steps ahead from state-space model.

    Parameters
    ----------
    result_filter : dict
        Output from kalman_filter
    Z, T, R, Q : arrays
        State-space matrices
    horizons : int
        Number of periods to forecast

    Returns
    -------
    y_forecast : array (horizons, N)
        Forecasted observations
    alpha_forecast : array (horizons, m)
        Forecasted states
    P_forecast : array (horizons, m, m)
        Forecast error variances
    """
    N, m = Z.shape
    r = Q.shape[0]

    # Start from last filtered state
    alpha_T = result_filter['alpha_filtered'][-1]
    P_T = result_filter['P_filtered'][-1]

    # Storage
    alpha_forecast = np.zeros((horizons, m))
    P_forecast = np.zeros((horizons, m, m))
    y_forecast = np.zeros((horizons, N))

    # Forecast recursion
    alpha_h = alpha_T
    P_h = P_T

    for h in range(horizons):
        # Forecast state
        alpha_h = T @ alpha_h
        P_h = T @ P_h @ T.T + R @ Q @ R.T

        # Forecast observation
        y_h = Z @ alpha_h

        # Store
        alpha_forecast[h] = alpha_h
        P_forecast[h] = P_h
        y_forecast[h] = y_h

    return y_forecast, alpha_forecast, P_forecast

# Forecast 10 periods ahead
y_fc, alpha_fc, P_fc = forecast_statespace(result, Z, T, R, Q, horizons=10)

print(f"Forecasted observations: {y_fc.shape}")
print(f"Forecasted factors: {alpha_fc.shape}")
```

</div>

---

## Common Pitfalls

### 1. Numerical Instability

**Problem:** $P_{t|t}$ not symmetric or not positive definite due to rounding errors.

**Solution:**
- Use Joseph form for covariance update:
  $$P_{t|t} = (I - K_t Z)P_{t|t-1}(I - K_t Z)' + K_t H K_t'$$
- Symmetrize: `P = (P + P.T) / 2`
- Use Cholesky decomposition for square roots

### 2. Non-Stationary Initialization

**Problem:** Setting $P_1 = 0$ or $P_1 = I$ arbitrarily.

**Solution:**
- For stationary model: use unconditional covariance (solve Lyapunov)
- For non-stationary: use diffuse initialization ($\kappa = 10^7$)
- Check eigenvalues of $T$ for stationarity

### 3. Forgetting to Check Singularity

**Problem:** $F_t$ not invertible (perfect multicollinearity or zero variance).

**Solution:**
- Use Moore-Penrose pseudo-inverse: `np.linalg.pinv(F_t)`
- Check condition number: `np.linalg.cond(F_t)`
- Add small regularization: $F_t + \epsilon I$ where $\epsilon = 10^{-6}$

### 4. Misinterpreting Filtered vs Smoothed

**Problem:** Using filtered estimates for estimation (EM algorithm needs smoothed).

**Solution:**
- Filtering: real-time estimates (forecasting, online applications)
- Smoothing: best estimates given all data (parameter estimation, structural analysis)

### 5. Ignoring Missing Data Patterns

**Problem:** Listwise deletion of missing data before Kalman filter.

**Solution:**
- Keep missing values as `np.nan`
- Let Kalman filter handle naturally via observation selection
- Maintains all information and avoids bias

---

## Connections

### Builds On
- **State-Space Representation** (Previous Guide): Framework for Kalman filter
- **Dynamic Factor Models**: Application domain
- **Bayesian Inference**: Filtering as sequential Bayesian update

### Leads To
- **EM Algorithm** (Module 4): Kalman smoother provides E-step
- **Maximum Likelihood Estimation**: Likelihood from prediction errors
- **Forecasting**: Multi-step predictions via state propagation

### Related To
- **Particle Filter**: Non-Gaussian generalization
- **Extended Kalman Filter**: Nonlinear systems (linearize locally)
- **Unscented Kalman Filter**: Better nonlinear approximation

---

## Practice Problems

### Conceptual

1. **Information Flow**
   - Why does $P_{t|t} < P_{t|t-1}$ always hold (assuming non-degenerate observation)?
   - In what case would Kalman gain $K_t = 0$? What does this mean?

2. **Steady-State Behavior**
   - If measurement noise $H$ is very large, what happens to steady-state Kalman gain?
   - If state innovations $Q = 0$, how does filter behave after initialization?

3. **Filtering vs Smoothing**
   - When would filtered and smoothed estimates be most different?
   - For which time periods does smoothing provide the most improvement?

### Mathematical

4. **Alternative Variance Update**
   - Derive the Joseph form for covariance update from the standard form.
   - Show that Joseph form is symmetric even with rounding errors.

5. **Steady-State Kalman Filter**
   - For scalar case ($m = 1, N = 1$), derive the steady-state $\bar{P}$ explicitly.
   - Show that $\bar{P}$ is the positive solution to a quadratic equation.

6. **Innovation Properties**
   - Prove that $v_t$ is uncorrelated over time: $\text{Cov}(v_t, v_s) = 0$ for $t \neq s$.
   - Show that standardized innovations $F_t^{-1/2}v_t$ are i.i.d. $N(0, I)$.

### Implementation

7. **Numerical Stability**
   ```python
   # Implement Joseph form covariance update
   # Compare numerical properties to standard form
   # Test with nearly singular F_t
   ```

8. **Diagnostic Checks**

<span class="filename">example.py</span>
</div>

   <div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
   # After running Kalman filter:
   # 1. Plot standardized innovations (should be N(0,1))
   # 2. Test for autocorrelation (Ljung-Box)
   # 3. Test for normality (QQ-plot)
   # 4. Check innovation variance equals F_t empirically
   ```

</div>

9. **Missing Data Experiment**

<span class="filename">example.py</span>
</div>

   <div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
   # Simulate data with 30% missing observations
   # Compare three approaches:
   # 1. Listwise deletion + Kalman filter
   # 2. Mean imputation + Kalman filter
   # 3. Kalman filter with missing data handling
   # Which produces best factor estimates (MSE vs true)?
   ```

</div>

### Extension

10. **Multivariate Local Level Model**
    - Implement Kalman filter for random walk plus noise:
      $$\begin{align}
      y_t &= \mu_t + \epsilon_t \\
      \mu_t &= \mu_{t-1} + \eta_t
      \end{align}$$
    - Estimate signal-to-noise ratio from data.
    - Compare to Hodrick-Prescott filter.

11. **Kalman Filter for ARIMA**
    - Express ARIMA(1,1,1) in state-space form:
      $$\Delta y_t = \phi \Delta y_{t-1} + \theta \epsilon_{t-1} + \epsilon_t$$
    - Implement Kalman filter estimation.
    - Compare to `statsmodels.tsa.arima.ARIMA`.

---

<div class="callout-insight">

**Insight:** Understanding kalman filter is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## Further Reading

### Essential
- **Durbin, J. & Koopman, S.J. (2012).** *Time Series Analysis by State Space Methods.* 2nd ed. Chapters 4-5.
  - Authoritative treatment with rigorous derivations and advanced topics.

- **Harvey, A.C. (1989).** *Forecasting, Structural Time Series Models and the Kalman Filter.* Chapters 3-4.
  - Clear exposition with economic applications.

### Recommended
- **Hamilton, J.D. (1994).** *Time Series Analysis.* Chapter 13.
  - Kalman filter in econometric context.

- **Shumway, R.H. & Stoffer, D.S. (2017).** *Time Series Analysis and Its Applications.* 4th ed. Chapter 6.
  - Practical guide with R code examples.

### Historical
- **Kalman, R.E. (1960).** "A New Approach to Linear Filtering and Prediction Problems." *Journal of Basic Engineering.*
  - Original paper introducing the Kalman filter.

### Advanced
- **Anderson, B.D.O. & Moore, J.B. (2005).** *Optimal Filtering.* Dover.
  - Rigorous mathematical treatment from engineering perspective.

- **Jazwinski, A.H. (2007).** *Stochastic Processes and Filtering Theory.* Dover.
  - Measure-theoretic foundations of optimal filtering.

---

**Next Steps:** With the Kalman filter in hand, you can now estimate dynamic factor models via maximum likelihood (Module 4) and handle mixed-frequency data (Module 5). The three guides in this module form the theoretical foundation for all advanced DFM methods.

---

## Conceptual Practice Questions

1. Walk through one iteration of the Kalman filter in your own words — what is predicted, what is updated, and why?

2. Why is the Kalman gain matrix central to the algorithm? What happens when observation noise is very large?

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.

</div>

---

## Cross-References

<a class="link-card" href="./03_kalman_filter_derivation_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_kalman_filter_implementation.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_from_static_to_dynamic.md">
  <div class="link-card-title">01 From Static To Dynamic</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_state_space_representation.md">
  <div class="link-card-title">02 State Space Representation</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

