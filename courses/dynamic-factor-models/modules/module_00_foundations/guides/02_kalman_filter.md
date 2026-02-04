# Kalman Filter

## In Brief

The Kalman filter is a recursive algorithm that optimally estimates the hidden state of a linear dynamical system from noisy observations. It's "optimal" in the sense of minimizing mean squared error when errors are Gaussian. In 40 lines of code, you get the best possible estimate of what's really happening behind the noise.

## Key Insight

The Kalman filter is just **Bayesian updating with Gaussian distributions**:
1. **Predict**: Use the state equation to forecast what happens next
2. **Update**: When new data arrives, blend your forecast with the observation (weighted by uncertainty)
3. **Repeat**: Each step refines your estimate

The magic: Gaussian distributions stay Gaussian under linear transformations, so all calculations are closed-form matrix operations. No sampling, no optimization, no iterations.

## Visual Explanation

```
KALMAN FILTER CYCLE (one time step)

         ┌─────────────────────────────────────────┐
         │  PREDICTION STEP (Time Update)          │
         │                                         │
    ┌────┼─  Given: a(t-1|t-1), P(t-1|t-1)       │
    │    │           [filtered state & covariance] │
    │    │                                         │
    │    │  Predict: a(t|t-1) = T·a(t-1|t-1)     │
    │    │          P(t|t-1) = T·P(t-1|t-1)·T'   │
    │    │                     + R·Q·R'            │
    │    │           [prior before seeing y(t)]    │
    │    └─────────────────────────────────────────┘
    │                      │
    │                      │ Observe y(t)
    │                      ▼
    │    ┌─────────────────────────────────────────┐
    │    │  UPDATE STEP (Measurement Update)       │
    │    │                                         │
    │    │  Innovation: v(t) = y(t) - Z·a(t|t-1) │
    │    │              [forecast error]           │
    │    │                                         │
    │    │  Kalman Gain: K(t) = P(t|t-1)·Z'·F⁻¹  │
    │    │               [optimal blending weight] │
    │    │                                         │
    │    │  Update: a(t|t) = a(t|t-1) + K·v(t)   │
    │    │         P(t|t) = P(t|t-1) - K·F·K'    │
    └────┼─         [posterior after seeing y(t)]  │
         │                                         │
         └─────────────────────────────────────────┘


INTUITION: Kalman Gain K(t)

    K(t) = P(t|t-1)·Z' / F(t)

    where F(t) = Z·P(t|t-1)·Z' + H  [innovation variance]

    If F is LARGE (noisy observations):
        → K is SMALL → trust model more than data

    If F is SMALL (precise observations):
        → K is LARGE → trust data more than model

    K automatically balances model vs data!
```

## Formal Definition

Given state space model:
```
α(t) = T·α(t-1) + R·η(t),    η(t) ~ N(0, Q)
y(t) = Z·α(t) + ε(t),        ε(t) ~ N(0, H)
α(0) ~ N(a₀, P₀)
```

The **Kalman filter** recursively computes:

**Prediction (Time Update):**
```
a(t|t-1) = T·a(t-1|t-1) + c                    [predicted state]
P(t|t-1) = T·P(t-1|t-1)·T' + R·Q·R'           [predicted covariance]
```

**Innovation:**
```
v(t) = y(t) - Z·a(t|t-1) - d                   [one-step-ahead forecast error]
F(t) = Z·P(t|t-1)·Z' + H                       [innovation variance]
```

**Update (Measurement Update):**
```
K(t) = P(t|t-1)·Z'·F(t)⁻¹                      [Kalman gain]
a(t|t) = a(t|t-1) + K(t)·v(t)                  [filtered state]
P(t|t) = P(t|t-1) - K(t)·F(t)·K(t)'           [filtered covariance]
```

**Initialization:**
```
a(0|0) = a₀
P(0|0) = P₀
```

**Key Properties:**
- If η, ε, α₀ are Gaussian and independent, then α(t|t) is the **minimum mean squared error** estimate
- The innovations {v(t)} are white noise if the model is correctly specified
- Filter is numerically stable and computationally efficient: O(m³) per time step

## Intuitive Explanation

**The Thermostat Analogy:**

Imagine you're trying to maintain room temperature:
- **Hidden state (α)**: Actual room temperature (can't measure perfectly)
- **Observation (y)**: Thermometer reading (noisy, fluctuates)
- **Prediction**: Based on physics, you predict how temperature evolves
- **Update**: When thermometer gives new reading, you adjust your estimate

The Kalman gain K decides: "Should I trust my physics model or the noisy thermometer?"
- If thermometer is very noisy (large H) → trust physics (small K)
- If thermometer is precise (small H) → trust measurement (large K)

**The GPS Analogy:**

Your phone's GPS combines:
- **Model prediction**: "Based on your speed/direction, you should be here"
- **GPS measurement**: "Satellite says you're here (±50m error)"
- **Kalman filter**: Blends both to show smooth, accurate position

When GPS signal is weak (large uncertainty), your location relies more on the motion model. When signal is strong, it follows GPS closely.

## Code Implementation

```python
import numpy as np

def kalman_filter(y, T, Z, R, Q, H, a0, P0):
    """
    Kalman filter for state space model.

    Returns:
        a_filt: Filtered states a(t|t)
        P_filt: Filtered covariances P(t|t)
        v: Innovations (forecast errors)
        F: Innovation variances
        loglik: Log-likelihood
    """
    n, p = y.shape
    m = T.shape[0]

    # Storage
    a_filt = np.zeros((n, m))
    v = np.zeros((n, p))
    F = np.zeros((n, p, p))
    loglik = 0.0

    # Initialize
    a = a0.copy()
    P = P0.copy()

    for t in range(n):
        # Prediction
        a_pred = T @ a
        P_pred = T @ P @ T.T + R @ Q @ R.T

        # Innovation
        if not np.isnan(y[t]).any():
            v[t] = y[t] - Z @ a_pred
            F[t] = Z @ P_pred @ Z.T + H
            F_inv = np.linalg.inv(F[t])

            # Update
            K = P_pred @ Z.T @ F_inv
            a = a_pred + K @ v[t]
            P = P_pred - K @ F[t] @ K.T

            # Log-likelihood
            loglik += -0.5 * (p * np.log(2*np.pi) +
                             np.log(np.linalg.det(F[t])) +
                             v[t] @ F_inv @ v[t])
        else:
            # Missing observation: prediction = filter
            a = a_pred
            P = P_pred

        a_filt[t] = a

    return a_filt, v, F, loglik

# Example usage: Local level model
n = 100
y = np.random.randn(n, 1).cumsum() + np.random.randn(n, 1)

T = np.array([[1.0]])
Z = np.array([[1.0]])
R = np.array([[1.0]])
Q = np.array([[0.1]])
H = np.array([[1.0]])
a0 = np.array([0.0])
P0 = np.array([[10.0]])

a_filt, v, F, loglik = kalman_filter(y, T, Z, R, Q, H, a0, P0)
print(f"Log-likelihood: {loglik:.2f}")
```

## Common Pitfalls

### 1. Numerical Instability with P(t|t)
**Problem:** P becomes non-positive definite due to rounding errors.

**Why it happens:** Update formula P(t|t) = P(t|t-1) - K·F·K' can lose symmetry.

**How to avoid:**
```python
# Use Joseph form (numerically stable)
I_KZ = np.eye(m) - K @ Z
P = I_KZ @ P_pred @ I_KZ.T + K @ H @ K.T
```

### 2. Forgetting to Handle Missing Data
**Problem:** Code crashes when y(t) has NaNs.

**Why it happens:** Inverse of F fails, or y - Z·a produces NaN.

**How to avoid:**
```python
if not np.isnan(y[t]).any():
    # Do update
else:
    # Skip update, use prediction as filter
    a = a_pred
    P = P_pred
```

### 3. Poor Initialization
**Problem:** Filter takes many iterations to converge from bad P₀.

**Why it happens:** Default P₀ = I doesn't match problem scale.

**How to avoid:**
```python
# Option 1: Use large P₀ (diffuse initialization)
P0 = 1e6 * np.eye(m)

# Option 2: Use steady-state P (if stationary)
from scipy.linalg import solve_discrete_lyapunov
P0 = solve_discrete_lyapunov(T, R @ Q @ R.T)
```

### 4. Confusing Filtered vs Smoothed Estimates
**Problem:** Using a(t|t) when you need a(t|n).

**Why it happens:** Filtered = real-time estimate; Smoothed = retrospective (uses future data).

**How to avoid:**
- Kalman filter gives a(t|t) (uses data up to t)
- Kalman smoother gives a(t|n) (uses all data) - see Module 02

### 5. Not Checking Innovation Properties
**Problem:** Model misspecification goes undetected.

**Why it happens:** Forgetting that innovations should be white noise.

**How to avoid:**
```python
# Standardized innovations
v_std = v / np.sqrt(np.diagonal(F, axis1=1, axis2=2))

# Check: should be ~ N(0,1) and uncorrelated
from scipy.stats import jarque_bera
print(f"JB test: {jarque_bera(v_std)}")

# Check autocorrelation
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(v_std)
```

### 6. Inverting F When p is Large
**Problem:** Matrix inversion is slow and numerically unstable for large p.

**Why it happens:** Direct implementation of F⁻¹.

**How to avoid:**
```python
# Use Cholesky decomposition
L = np.linalg.cholesky(F[t])
F_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(p)))

# Or use Woodbury identity when m << p
```

## Connections

### Builds on:
- **State Space Models** - The framework being estimated
- **Bayesian Inference** - Recursive Bayesian updating
- **Multivariate Normal** - Closed-form conditioning formulas
- **Linear Algebra** - Efficient matrix computations

### Leads to:
- **Kalman Smoother** - Retrospective estimation using all data
- **Maximum Likelihood** - Use filter likelihood for parameter estimation (Module 02)
- **EM Algorithm** - Expectation step uses Kalman smoother (Module 02)
- **Particle Filters** - Nonlinear/non-Gaussian extension

### Equivalent Formulations:
- **Recursive Least Squares** - Kalman filter with specific parameter settings
- **Bayes Filter** - General Bayesian filtering (KF is Gaussian special case)
- **Wiener Filter** - Frequency domain equivalent for stationary processes

### Applications:
- **Nowcasting** - Real-time GDP estimation from mixed-frequency data
- **Signal Processing** - Extract signal from noise in any domain
- **Robotics** - Sensor fusion (GPS + IMU + camera)
- **Finance** - Estimate time-varying parameters (volatility, beta)

## Practice Problems

### Conceptual Questions

1. **Why is the Kalman filter "optimal"?**
   - Hint: What does it minimize? Under what assumptions?

2. **What happens to K(t) as Q → 0?**
   - Hint: If state has no randomness, should you trust model or data?

3. **Why are innovations v(t) useful for diagnostics?**
   - Hint: What should their distribution be if model is correct?

### Implementation Challenges

4. **Add Kalman smoother:**
   - Implement backward recursion for a(t|n)
   - Compare filtered vs smoothed estimates visually

5. **Handle multivariate observations:**
   - Modify code for p > 1
   - Test on bivariate local level model

6. **Implement square-root filter:**
   - Store Cholesky factor of P instead of P
   - More numerically stable for ill-conditioned problems

### Advanced

7. **Derive the Kalman gain formula:**
   - Start from a(t|t) = argmin E[(α(t) - a)' (α(t) - a) | y₁,...,y_t]
   - Show K = P(t|t-1)·Z'·F⁻¹ minimizes MSE

8. **Prove innovation representation:**
   - Show y(t) = Z·T·a(t-1|t-1) + Z·R·η(t) + ε(t)
   - Derive E[v(t)v(s)'] = F(t)·δ_ts (white noise)

9. **Implement univariate filter:**
   - For p=1, avoid matrix inversions entirely
   - Benchmark speed vs multivariate version

## Further Reading

- Durbin & Koopman (2012), Chapter 4: The Kalman filter
- Kalman (1960): Original paper - "A New Approach to Linear Filtering and Prediction Problems"
- Harvey (1989), Chapter 3.2-3.4: Filtering and smoothing algorithms
- Welch & Bishop (2006): "An Introduction to the Kalman Filter" - excellent tutorial
- See [Additional Readings](../resources/additional_readings.md) for complete list and code examples
