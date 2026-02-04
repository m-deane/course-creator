# Module 00 Cheatsheet: State Space Models & Kalman Filter

## Quick Reference

### State Space Model

```python
# State equation
α(t) = T·α(t-1) + R·η(t),    η ~ N(0, Q)

# Observation equation
y(t) = Z·α(t) + ε(t),        ε ~ N(0, H)

# Dimensions
α(t): (m, 1)  # state vector
y(t): (p, 1)  # observation vector
T:    (m, m)  # transition matrix
Z:    (p, m)  # observation matrix
R:    (m, r)  # selection matrix
Q:    (r, r)  # state noise covariance
H:    (p, p)  # observation noise covariance
```

### Kalman Filter Equations

```python
# PREDICT
a(t|t-1) = T @ a(t-1|t-1)
P(t|t-1) = T @ P(t-1|t-1) @ T.T + R @ Q @ R.T

# INNOVATE
v(t) = y(t) - Z @ a(t|t-1)
F(t) = Z @ P(t|t-1) @ Z.T + H

# UPDATE
K(t) = P(t|t-1) @ Z.T @ np.linalg.inv(F(t))
a(t|t) = a(t|t-1) + K(t) @ v(t)
P(t|t) = P(t|t-1) - K(t) @ F(t) @ K(t).T
```

## Common Models in State Space Form

### 1. Local Level (Random Walk + Noise)
```python
T = np.array([[1.0]])
Z = np.array([[1.0]])
R = np.array([[1.0]])
Q = np.array([[σ_η²]])
H = np.array([[σ_ε²]])
```

### 2. Local Linear Trend
```python
T = np.array([[1.0, 1.0],
              [0.0, 1.0]])
Z = np.array([[1.0, 0.0]])
R = np.eye(2)
Q = np.diag([σ_level², σ_slope²])
H = np.array([[σ_obs²]])
```

### 3. AR(1) Model
```python
# y(t) = φ·y(t-1) + ε(t)
T = np.array([[φ]])
Z = np.array([[1.0]])
R = np.array([[1.0]])
Q = np.array([[σ²/(1-φ²)]])
H = np.array([[0.0]])  # No observation noise
```

### 4. AR(p) Model
```python
# y(t) = φ₁·y(t-1) + ... + φₚ·y(t-p) + ε(t)
T = np.block([[φ],
              [np.eye(p-1), np.zeros((p-1, 1))]])
Z = np.array([[1.0] + [0.0]*(p-1)])
R = np.array([[1.0]] + [[0.0]]*(p-1))
Q = np.array([[σ²]])
H = np.array([[0.0]])
```

## Code Patterns

### Setup Model
```python
import numpy as np

class KalmanFilter:
    def __init__(self, T, Z, R, Q, H, a0, P0):
        self.T, self.Z, self.R = T, Z, R
        self.Q, self.H = Q, H
        self.a, self.P = a0.copy(), P0.copy()

    def predict(self):
        self.a = self.T @ self.a
        self.P = self.T @ self.P @ self.T.T + self.R @ self.Q @ self.R.T
        return self.a, self.P

    def update(self, y):
        v = y - self.Z @ self.a
        F = self.Z @ self.P @ self.Z.T + self.H
        K = self.P @ self.Z.T @ np.linalg.inv(F)
        self.a = self.a + K @ v
        self.P = self.P - K @ F @ K.T
        return self.a, self.P, v, F
```

### Run Filter
```python
kf = KalmanFilter(T, Z, R, Q, H, a0, P0)

filtered_states = []
innovations = []

for t in range(len(y)):
    kf.predict()
    a_filt, P_filt, v, F = kf.update(y[t])
    filtered_states.append(a_filt)
    innovations.append(v)

states = np.array(filtered_states)
```

### Handle Missing Data
```python
def update(self, y):
    if np.isnan(y).any():
        # Skip update, use prediction
        return self.a, self.P, None, None

    # Normal update
    v = y - self.Z @ self.a
    # ... rest of update code
```

### Numerically Stable Update (Joseph Form)
```python
def update_stable(self, y):
    v = y - self.Z @ self.a
    F = self.Z @ self.P @ self.Z.T + self.H
    K = self.P @ self.Z.T @ np.linalg.inv(F)

    # Joseph form covariance update
    I_KZ = np.eye(len(self.a)) - K @ self.Z
    self.P = I_KZ @ self.P @ I_KZ.T + K @ self.H @ K.T
    self.a = self.a + K @ v
    return self.a, self.P, v, F
```

## Diagnostics

### Check Innovation Properties
```python
# Should be white noise, mean zero, variance F
v_std = innovations / np.sqrt(F_diag)

# Normality
from scipy.stats import jarque_bera
stat, pval = jarque_bera(v_std)
print(f"JB test p-value: {pval:.4f}")  # Should be > 0.05

# Autocorrelation
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_stat, lb_pval = acorr_ljungbox(v_std, lags=10)
print(f"LB test p-value: {lb_pval[9]:.4f}")  # Should be > 0.05
```

### Compute Log-Likelihood
```python
loglik = 0.0
for t in range(n):
    # After computing v[t] and F[t] in filter
    loglik += -0.5 * (p * np.log(2*np.pi) +
                      np.log(np.linalg.det(F[t])) +
                      v[t].T @ np.linalg.inv(F[t]) @ v[t])
```

### Plot Results
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(10, 8))

# Observations and filtered state
axes[0].plot(y, 'o', label='Observations', alpha=0.5)
axes[0].plot(filtered_states, '-', label='Filtered State')
axes[0].legend()
axes[0].set_title('Kalman Filter Estimates')

# Innovations
axes[1].plot(innovations)
axes[1].axhline(0, color='r', linestyle='--')
axes[1].set_title('Innovations (Forecast Errors)')

# Standardized innovations
axes[2].plot(v_std)
axes[2].axhline(0, color='r', linestyle='--')
axes[2].axhline(2, color='r', linestyle=':', alpha=0.5)
axes[2].axhline(-2, color='r', linestyle=':', alpha=0.5)
axes[2].set_title('Standardized Innovations')

plt.tight_layout()
```

## Common Gotchas

### 1. Dimension Mismatches
```python
# Always check!
assert T.shape == (m, m), "T must be (m, m)"
assert Z.shape == (p, m), "Z must be (p, m)"
assert Q.shape == (r, r), "Q must be (r, r)"
assert H.shape == (p, p), "H must be (p, p)"
assert R.shape == (m, r), "R must be (m, r)"
```

### 2. Non-Positive Definite Covariances
```python
# Ensure Q, H, P are positive definite
Q = (Q + Q.T) / 2  # Force symmetry
Q += 1e-8 * np.eye(len(Q))  # Add nugget
```

### 3. Explosive Dynamics
```python
# Check stability
eigenvalues = np.linalg.eigvals(T)
max_eig = np.max(np.abs(eigenvalues))
if max_eig >= 1:
    print(f"Warning: Max eigenvalue = {max_eig:.4f} >= 1")
```

### 4. Singular F Matrix
```python
# Add small nugget to H
H = H + 1e-6 * np.eye(p)
```

## Performance Tips

### 1. Vectorize When Possible
```python
# Bad (loop)
for t in range(n):
    v[t] = y[t] - Z @ a_pred[t]

# Good (vectorized)
v = y - (Z @ a_pred.T).T
```

### 2. Exploit Structure
```python
# For diagonal H
H_inv = np.diag(1.0 / np.diag(H))  # O(p) instead of O(p³)

# For univariate (p=1)
F_inv = 1.0 / F  # Scalar division, no matrix inverse
```

### 3. Avoid Redundant Computations
```python
# Compute once, reuse
ZP = Z @ P
F = ZP @ Z.T + H
K = P @ Z.T @ np.linalg.inv(F)
# Don't recompute Z @ P multiple times!
```

## Useful Functions

```python
def simulate_state_space(T, Z, R, Q, H, n_periods, a0=None):
    """Simulate observations from state space model."""
    m = T.shape[0]
    p = Z.shape[0]

    a0 = a0 if a0 is not None else np.zeros(m)
    states = np.zeros((n_periods, m))
    obs = np.zeros((n_periods, p))

    states[0] = a0
    for t in range(n_periods):
        if t > 0:
            eta = np.random.multivariate_normal(np.zeros(Q.shape[0]), Q)
            states[t] = T @ states[t-1] + R @ eta
        eps = np.random.multivariate_normal(np.zeros(p), H)
        obs[t] = Z @ states[t] + eps

    return obs, states

def compute_steady_state_P(T, R, Q, maxiter=1000, tol=1e-6):
    """Compute steady-state covariance for stationary model."""
    from scipy.linalg import solve_discrete_lyapunov
    try:
        return solve_discrete_lyapunov(T, R @ Q @ R.T)
    except:
        # Manual iteration
        P = np.eye(T.shape[0])
        for _ in range(maxiter):
            P_new = T @ P @ T.T + R @ Q @ R.T
            if np.max(np.abs(P_new - P)) < tol:
                return P_new
            P = P_new
        print("Warning: Did not converge")
        return P
```

## Matrix Formulas

### Woodbury Identity (for large p, small m)
```python
# (A + UCV)^-1 = A^-1 - A^-1 U(C^-1 + VA^-1U)^-1 VA^-1
# Applied to F^-1 when p >> m:
# F = Z·P·Z' + H
# F^-1 = H^-1 - H^-1·Z·(P^-1 + Z'H^-1Z)^-1·Z'·H^-1
```

### Sherman-Morrison (rank-1 update)
```python
# (A + uv')^-1 = A^-1 - (A^-1 u)(v' A^-1) / (1 + v' A^-1 u)
```

### Covariance Update
```python
# Standard form (can lose positive definiteness)
P = P - K @ F @ K.T

# Joseph form (always positive definite)
I_KZ = np.eye(m) - K @ Z
P = I_KZ @ P @ I_KZ.T + K @ H @ K.T
```

## Debugging Checklist

- [ ] Check matrix dimensions (T, Z, R, Q, H)
- [ ] Verify Q, H, P₀ are symmetric and positive definite
- [ ] Check for NaNs or Infs in data
- [ ] Test with simulated data first
- [ ] Verify innovations are approximately white noise
- [ ] Check eigenvalues of T for stability
- [ ] Compare to statsmodels or KFAS (R) output
- [ ] Plot filtered states vs observations

## Resources

- **statsmodels**: `statsmodels.tsa.statespace` for production-ready implementation
- **simdkalman**: Fast Kalman filter library
- **pykalman**: Another popular implementation
- **filterpy**: Educational Kalman filter library with great docs
