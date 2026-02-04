# Dynamic Factor Model Specification

## In Brief

DFMs are written in state space form by stacking factors and their lags into the state vector. The key is converting the lag polynomial representation Λ(L)·F(t) into the standard Z·α(t) format, which allows using the Kalman filter for estimation.

## Key Insight

**Three equivalent representations:**
1. **Lag polynomial form**: X(t) = Λ(L)·F(t) + ε(t) (conceptual)
2. **Companion form**: Stack lags in state vector (for computation)
3. **State space form**: Standard Z, T matrices (for Kalman filter)

All three describe the same model. We write conceptually in lag form, estimate in state space form.

## Formal Definition

**Standard DFM:**
```
Observation equation:
X(t) = Λ(L)·F(t) + ε(t)
     = Λ₀·F(t) + Λ₁·F(t-1) + ... + Λ_p·F(t-p) + ε(t)

Factor dynamics:
F(t) = Φ₁·F(t-1) + ... + Φ_q·F(t-q) + η(t)
```

**State Space Form:**
```
X(t) = Z·α(t) + ε(t)        ε(t) ~ N(0, H)
α(t) = T·α(t-1) + R·η(t)    η(t) ~ N(0, Q)
```

Where α(t) = [F(t), F(t-1), ..., F(t-max(p,q))]'

## Code Implementation

```python
import numpy as np

def dfm_to_state_space(Lambda, Phi, p, q):
    """
    Convert DFM to state space form.

    Parameters:
    Lambda: list of (N,r) arrays [Λ₀, Λ₁, ..., Λ_p]
    Phi: list of (r,r) arrays [Φ₁, ..., Φ_q]
    p: lags in observation equation
    q: lags in factor VAR

    Returns: Z, T, R for state space model
    """
    N, r = Lambda[0].shape
    s = max(p + 1, q)  # state dimension multiplier

    # Observation matrix Z
    Z = np.zeros((N, r * s))
    for i in range(p + 1):
        Z[:, i*r:(i+1)*r] = Lambda[i]

    # Transition matrix T (companion form)
    T = np.zeros((r * s, r * s))
    for i in range(q):
        T[:r, i*r:(i+1)*r] = Phi[i]
    T[r:, :-r] = np.eye(r * (s - 1))

    # Selection matrix R
    R = np.zeros((r * s, r))
    R[:r, :] = np.eye(r)

    return Z, T, R

# Example: 2 factors, 1 lag in observation, AR(1) factors
Lambda_0 = np.random.randn(10, 2)
Lambda_1 = np.random.randn(10, 2) * 0.3
Phi_1 = np.diag([0.8, 0.6])

Z, T, R = dfm_to_state_space([Lambda_0, Lambda_1], [Phi_1], p=1, q=1)
print(f"State dimension: {T.shape[0]}")
print(f"Observation dimension: {Z.shape[0]}")
```

## Common Pitfalls

### 1. Wrong State Dimension
**Problem:** State vector too small, missing necessary lags.

**Solution:** s = max(p+1, q) where p = obs lags, q = factor AR order.

### 2. Forgetting Companion Form
**Problem:** T matrix doesn't properly stack lags.

**Solution:** Use identity blocks in bottom rows of T.

### 3. Identification Not Imposed
**Problem:** Model is unidentified, results are arbitrary.

**Solution:** See Guide 3 on identification restrictions.

## Practice Problems

1. Write state space form for N=5, r=2, p=1, q=1
2. Verify T matrix has correct eigenvalues
3. Simulate from specified DFM

## Further Reading

- Durbin & Koopman (2012), Chapter 3: Multivariate state space
- Stock & Watson (2011): DFM specification details
