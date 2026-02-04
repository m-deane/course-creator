# State Space Models

## In Brief

State space models represent any time series as a combination of hidden (latent) state dynamics and noisy observations. They provide a unified framework for filtering, smoothing, forecasting, and parameter estimation across all linear dynamic systems.

## Key Insight

Instead of modeling observations directly (like ARIMA), state space models separate the system into:
1. **State equation** - How the hidden true process evolves
2. **Observation equation** - How we measure the hidden process with noise

This separation makes it trivial to handle missing data, irregular sampling, and multivariate relationships.

## Visual Explanation

```
Time:     t-1          t           t+1

State:    α(t-1) ---> α(t) -----> α(t+1)     [Hidden - follows Markov process]
           |           |            |
          noise       noise        noise
           |           |            |
           v           v            v
Observe:  y(t-1)      y(t)        y(t+1)     [Measured - corrupted by noise]


Graphical Representation:

    ┌─────────────────────────────────────────────┐
    │         STATE EQUATION (Transition)         │
    │   α(t) = T·α(t-1) + R·η(t)                 │
    │                                             │
    │   T = transition matrix (dynamics)          │
    │   η = state disturbance ~ N(0, Q)          │
    └─────────────────────────────────────────────┘
                        │
                        ▼
    ┌─────────────────────────────────────────────┐
    │     OBSERVATION EQUATION (Measurement)       │
    │   y(t) = Z·α(t) + ε(t)                      │
    │                                             │
    │   Z = observation matrix (loading)          │
    │   ε = measurement error ~ N(0, H)          │
    └─────────────────────────────────────────────┘
```

## Formal Definition

A **state space model** consists of:

**State Equation (Transition):**
```
α(t) = T·α(t-1) + c + R·η(t),    η(t) ~ N(0, Q)
```

**Observation Equation (Measurement):**
```
y(t) = Z·α(t) + d + ε(t),        ε(t) ~ N(0, H)
```

Where:
- `α(t)` ∈ ℝ^m is the m-dimensional state vector (hidden)
- `y(t)` ∈ ℝ^p is the p-dimensional observation vector (measured)
- `T` ∈ ℝ^(m×m) is the state transition matrix
- `Z` ∈ ℝ^(p×m) is the observation matrix
- `R` ∈ ℝ^(m×r) is the state disturbance selection matrix
- `c` ∈ ℝ^m, `d` ∈ ℝ^p are optional intercept terms
- `Q` ∈ ℝ^(r×r) is the state disturbance covariance
- `H` ∈ ℝ^(p×p) is the observation error covariance

**Initial Conditions:**
```
α(0) ~ N(a(0), P(0))
```

## Intuitive Explanation

Think of tracking an airplane:
- **State (α)**: True position, velocity, acceleration (hidden, must be estimated)
- **Observation (y)**: Radar pings (noisy, what we actually measure)
- **State equation**: Physics of motion (how position evolves)
- **Observation equation**: How radar converts true position to noisy signal

The Kalman filter optimally estimates the hidden state given the noisy observations.

Another analogy: **Karaoke with a bad microphone**
- Hidden state = your actual singing (true signal)
- Observations = what the bad mic records (signal + noise)
- State dynamics = how your pitch evolves between notes
- Observation equation = mic quality (how much noise it adds)

## Code Implementation

```python
import numpy as np

class StateSpaceModel:
    """Simple state space model representation."""

    def __init__(self, T, Z, R, Q, H, a0=None, P0=None):
        """
        Parameters:
        T: (m, m) state transition matrix
        Z: (p, m) observation matrix
        R: (m, r) state disturbance selection
        Q: (r, r) state disturbance covariance
        H: (p, p) observation error covariance
        a0: (m,) initial state mean
        P0: (m, m) initial state covariance
        """
        self.T, self.Z, self.R = T, Z, R
        self.Q, self.H = Q, H
        self.m = T.shape[0]  # state dimension
        self.p = Z.shape[0]  # observation dimension

        # Initialize state
        self.a0 = a0 if a0 is not None else np.zeros(self.m)
        self.P0 = P0 if P0 is not None else np.eye(self.m)

    def simulate(self, n_periods):
        """Simulate from the model."""
        states = np.zeros((n_periods, self.m))
        observations = np.zeros((n_periods, self.p))

        # Initial state
        states[0] = np.random.multivariate_normal(self.a0, self.P0)

        for t in range(n_periods):
            # State evolution: α(t) = T·α(t-1) + R·η(t)
            if t > 0:
                eta = np.random.multivariate_normal(np.zeros(self.Q.shape[0]), self.Q)
                states[t] = self.T @ states[t-1] + self.R @ eta

            # Observation: y(t) = Z·α(t) + ε(t)
            eps = np.random.multivariate_normal(np.zeros(self.p), self.H)
            observations[t] = self.Z @ states[t] + eps

        return observations, states

# Example: Local level model (random walk + noise)
# y(t) = α(t) + ε(t),    ε ~ N(0, H)
# α(t) = α(t-1) + η(t),  η ~ N(0, Q)

T = np.array([[1.0]])           # Random walk state
Z = np.array([[1.0]])           # Direct observation
R = np.array([[1.0]])           # Identity selection
Q = np.array([[0.1]])           # State innovation variance
H = np.array([[1.0]])           # Observation noise variance

model = StateSpaceModel(T, Z, R, Q, H)
y, alpha = model.simulate(100)
```

## Common Pitfalls

### 1. Confusing State Dimension with Observation Dimension
**Problem:** Setting T, Z matrices with incompatible dimensions.

**Why it happens:** State dimension (m) and observation dimension (p) are independent. A 1D observation can have a 10D state.

**How to avoid:** Always check dimensions:
- T must be (m, m)
- Z must be (p, m)
- State covariance P is (m, m)

### 2. Non-Stationary State Dynamics
**Problem:** Eigenvalues of T outside unit circle cause explosive states.

**Why it happens:** Forgetting stability conditions when specifying T.

**How to avoid:**
```python
# Check stability
eigenvalues = np.linalg.eigvals(T)
if np.any(np.abs(eigenvalues) >= 1):
    print("Warning: Non-stationary dynamics!")
```

### 3. Forgetting R Matrix
**Problem:** Setting R = I when state dimension ≠ disturbance dimension.

**Why it happens:** Assuming one shock per state variable.

**How to avoid:** R allows for shared shocks across states or dimensionality reduction. Always specify explicitly.

### 4. Poor Initial Conditions
**Problem:** P0 too small causes slow convergence; too large causes instability.

**Why it happens:** Default P0 = I might not match your problem scale.

**How to avoid:**
```python
# For stationary processes, use steady-state covariance
from scipy.linalg import solve_discrete_lyapunov
P_inf = solve_discrete_lyapunov(T, R @ Q @ R.T)
model = StateSpaceModel(T, Z, R, Q, H, a0=np.zeros(m), P0=P_inf)
```

### 5. Identifiability Issues
**Problem:** Different parameter sets produce identical observations.

**Why it happens:** State space models are not unique (can rotate state coordinates).

**How to avoid:** Impose identification restrictions (see Module 01 on DFM identification).

## Connections

### Builds on:
- **Linear Algebra** - Matrix operations, eigendecomposition
- **Probability** - Multivariate normal distribution, conditional distributions
- **Time Series** - Stationarity, autocovariance functions
- **Markov Processes** - First-order dynamics

### Leads to:
- **Kalman Filter (Module 00)** - Optimal state estimation algorithm
- **Dynamic Factor Models (Module 01)** - High-dimensional state space models
- **Maximum Likelihood Estimation (Module 02)** - Parameter estimation via prediction error decomposition
- **Missing Data Handling** - Trivial extension of observation equation

### Equivalent Formulations:
- **ARMA Models** - Can be written in state space form
- **Structural Time Series** - Decompose series into trend, seasonal, cycle components
- **Vector Autoregression (VAR)** - Multivariate AR as state space model

## Practice Problems

### Conceptual Questions

1. **Why does state space form handle missing data naturally?**
   - Hint: What happens to the observation equation when y(t) is missing?

2. **How would you extend the local level model to include a trend?**
   - Hint: Add a second state variable for the slope.

3. **Why is the state equation called "Markov"?**
   - Hint: What information do you need to predict α(t+1)?

### Implementation Challenges

4. **Convert AR(2) to state space form:**
   - Given: y(t) = φ₁y(t-1) + φ₂y(t-2) + ε(t)
   - Specify T, Z, R, Q, H matrices

5. **Implement a local linear trend model:**
   ```
   y(t) = μ(t) + ε(t)
   μ(t) = μ(t-1) + β(t-1) + η₁(t)
   β(t) = β(t-1) + η₂(t)
   ```
   - Hint: State is α = [μ, β]'

6. **Verify your simulation:**
   - Simulate 1000 observations from local level model
   - Compute sample autocorrelation
   - Compare to theoretical ACF derived from state space parameters

### Advanced

7. **Derive steady-state Kalman gain:**
   - For stationary model, what is P(t) as t → ∞?
   - Use this to derive time-invariant Kalman gain

8. **Prove observability:**
   - Show that state α is uniquely determined from {y(t)} if rank([Z', (ZT)', (ZT²)', ...]) = m

## Further Reading

- Durbin & Koopman (2012), Chapter 2: State space models fundamentals
- Harvey (1989), Chapter 3: The state-space framework
- Kalman (1960): Original paper - surprisingly readable!
- See [Additional Readings](../resources/additional_readings.md) for complete list
