# Kalman Filter: Complete Derivation

## TL;DR

The Kalman filter is the **optimal linear estimator** for state-space models with Gaussian noise. This guide provides complete mathematical derivations of the filtering and smoothing recursions, including intuition for each step.

---

## 1. Problem Setup

### 1.1 The Filtering Problem

**Given:**
- State-space model:
  ```
  α_t = T α_{t-1} + R η_t,    η_t ~ N(0, Q)
  y_t = Z α_t + ε_t,          ε_t ~ N(0, H)
  ```
- Initial distribution: α_0 ~ N(a_0, P_0)
- Observations up to time t: Y_t = {y_1, ..., y_t}

**Goal:** Compute optimal estimate of state:
```
α̂_{t|t} = E[α_t | Y_t]
P_{t|t} = Var[α_t | Y_t]
```

### 1.2 Optimality Criterion

**Definition:** The Kalman filter minimizes **mean squared error**:
```
α̂_{t|t} = argmin E[||α_t - α̂||² | Y_t]
```

For Gaussian distributions, this is the **conditional mean** (posterior mean).

### 1.3 Key Assumption

**Gaussianity:** All distributions remain Gaussian throughout:
```
α_t | Y_{t-1} ~ N(α̂_{t|t-1}, P_{t|t-1})  (prediction)
α_t | Y_t     ~ N(α̂_{t|t}, P_{t|t})      (filtering)
```

This is guaranteed when:
1. Initial state is Gaussian
2. State innovations η_t are Gaussian
3. Observation noise ε_t is Gaussian
4. Transitions are linear

---

## 2. The Kalman Filter Recursion

### 2.1 Overview

The filter consists of two steps repeated for t = 1, 2, ..., T:

**PREDICT (time update):**
```
α̂_{t|t-1} = T α̂_{t-1|t-1}
P_{t|t-1} = T P_{t-1|t-1} T' + R Q R'
```

**UPDATE (measurement update):**
```
v_t = y_t - Z α̂_{t|t-1}                    (innovation)
F_t = Z P_{t|t-1} Z' + H                    (innovation variance)
K_t = P_{t|t-1} Z' F_t⁻¹                    (Kalman gain)
α̂_{t|t} = α̂_{t|t-1} + K_t v_t              (updated state)
P_{t|t} = (I - K_t Z) P_{t|t-1}             (updated variance)
```

Let's derive each equation.

---

## 3. Prediction Step Derivation

### 3.1 State Prediction

**Goal:** Compute E[α_t | Y_{t-1}]

**Start with state equation:**
```
α_t = T α_{t-1} + R η_t
```

**Take expectation conditional on Y_{t-1}:**
```
E[α_t | Y_{t-1}] = E[T α_{t-1} + R η_t | Y_{t-1}]
                 = T E[α_{t-1} | Y_{t-1}] + R E[η_t | Y_{t-1}]
```

**Key facts:**
- η_t is independent of past: E[η_t | Y_{t-1}] = 0
- We know filtered state at t-1: E[α_{t-1} | Y_{t-1}] = α̂_{t-1|t-1}

**Result:**
```
α̂_{t|t-1} = T α̂_{t-1|t-1}
```

**Intuition:** To predict next state, apply the dynamics to current best estimate.

### 3.2 Prediction Variance

**Goal:** Compute Var[α_t | Y_{t-1}]

**Start with:**
```
α_t - α̂_{t|t-1} = T α_{t-1} + R η_t - T α̂_{t-1|t-1}
                 = T (α_{t-1} - α̂_{t-1|t-1}) + R η_t
```

**Take variance:**
```
P_{t|t-1} = Var[α_t - α̂_{t|t-1} | Y_{t-1}]
          = Var[T (α_{t-1} - α̂_{t-1|t-1}) + R η_t | Y_{t-1}]
```

**Independence:** η_t ⊥ (α_{t-1} - α̂_{t-1|t-1})

**Apply variance formula:**
```
P_{t|t-1} = T Var[α_{t-1} - α̂_{t-1|t-1} | Y_{t-1}] T' + R Var[η_t] R'
          = T P_{t-1|t-1} T' + R Q R'
```

**Intuition:** Uncertainty increases due to:
1. Propagation of previous uncertainty (T P_{t-1|t-1} T')
2. New state shocks (R Q R')

---

## 4. Update Step Derivation

### 4.1 Innovation (Prediction Error)

**Definition:**
```
v_t = y_t - E[y_t | Y_{t-1}]
```

**Compute prediction of y_t:**
```
E[y_t | Y_{t-1}] = E[Z α_t + ε_t | Y_{t-1}]
                 = Z E[α_t | Y_{t-1}] + E[ε_t | Y_{t-1}]
                 = Z α̂_{t|t-1}
```

Since ε_t is independent of past.

**Therefore:**
```
v_t = y_t - Z α̂_{t|t-1}
```

**Intuition:** How much did observation surprise us?

### 4.2 Innovation Variance

**Definition:**
```
F_t = Var[v_t | Y_{t-1}]
```

**Expand:**
```
v_t = y_t - Z α̂_{t|t-1}
    = Z α_t + ε_t - Z α̂_{t|t-1}
    = Z (α_t - α̂_{t|t-1}) + ε_t
```

**Take variance:**
```
F_t = Var[Z (α_t - α̂_{t|t-1}) + ε_t | Y_{t-1}]
```

**Independence:** (α_t - α̂_{t|t-1}) ⊥ ε_t

**Result:**
```
F_t = Z Var[α_t - α̂_{t|t-1} | Y_{t-1}] Z' + Var[ε_t]
    = Z P_{t|t-1} Z' + H
```

**Intuition:** Uncertainty in prediction comes from state uncertainty (Z P_{t|t-1} Z') plus measurement noise (H).

### 4.3 Kalman Gain (The Key Step)

**Goal:** Update state estimate using new observation y_t

**Ansatz (guess):** Linear update rule
```
α̂_{t|t} = α̂_{t|t-1} + K_t v_t
```

for some matrix K_t (the **Kalman gain**).

**Question:** What is the optimal K_t?

**Answer:** The K_t that minimizes MSE: E[||α_t - α̂_{t|t}||² | Y_t]

**Derivation:**

The updated estimate error is:
```
α_t - α̂_{t|t} = α_t - α̂_{t|t-1} - K_t v_t
               = α_t - α̂_{t|t-1} - K_t (y_t - Z α̂_{t|t-1})
               = α_t - α̂_{t|t-1} - K_t (Z α_t + ε_t - Z α̂_{t|t-1})
               = (I - K_t Z)(α_t - α̂_{t|t-1}) - K_t ε_t
```

**Updated variance:**
```
P_{t|t} = Var[α_t - α̂_{t|t} | Y_t]
        = (I - K_t Z) P_{t|t-1} (I - K_t Z)' + K_t H K_t'
```

**Minimize with respect to K_t:**

Take derivative ∂P_{t|t}/∂K_t and set to zero:
```
∂P_{t|t}/∂K_t = -2(I - K_t Z) P_{t|t-1} Z' + 2 K_t H = 0
```

**Solve for K_t:**
```
P_{t|t-1} Z' = K_t (Z P_{t|t-1} Z' + H)
K_t = P_{t|t-1} Z' (Z P_{t|t-1} Z' + H)⁻¹
K_t = P_{t|t-1} Z' F_t⁻¹
```

**This is the Kalman gain!**

**Intuition:**
- If F_t is large (high prediction uncertainty), K_t is large → trust new data more
- If P_{t|t-1} is small (confident in prediction), K_t is small → trust prediction more

### 4.4 State Update

**Formula:**
```
α̂_{t|t} = α̂_{t|t-1} + K_t v_t
```

**Expand:**
```
α̂_{t|t} = α̂_{t|t-1} + P_{t|t-1} Z' F_t⁻¹ (y_t - Z α̂_{t|t-1})
```

**Interpretation:**
1. Start with prediction α̂_{t|t-1}
2. Observe innovation v_t = y_t - Z α̂_{t|t-1}
3. Update by weighted innovation K_t v_t

**Alternative form (Joseph stabilized):**
```
α̂_{t|t} = (I - K_t Z) α̂_{t|t-1} + K_t y_t
```

Weights:
- Prediction gets weight (I - K_t Z)
- Observation gets weight K_t

### 4.5 Variance Update

**Standard form:**
```
P_{t|t} = (I - K_t Z) P_{t|t-1}
```

**Derivation:**
```
P_{t|t} = (I - K_t Z) P_{t|t-1} (I - K_t Z)' + K_t H K_t'
```

Substitute K_t = P_{t|t-1} Z' F_t⁻¹:
```
P_{t|t} = P_{t|t-1} - P_{t|t-1} Z' F_t⁻¹ Z P_{t|t-1}
```

**Joseph form (numerically stable):**
```
P_{t|t} = (I - K_t Z) P_{t|t-1} (I - K_t Z)' + K_t H K_t'
```

Always use Joseph form in practice (prevents numerical issues).

---

## 5. Initialization

### 5.1 Known Initial State

If α_0 is known:
```
α̂_{0|0} = α_0
P_{0|0} = 0  (no uncertainty)
```

### 5.2 Uncertain Initial State

If α_0 ~ N(a_0, P_0):
```
α̂_{0|0} = a_0
P_{0|0} = P_0
```

### 5.3 Diffuse Initialization

If initial state is completely unknown (P_0 → ∞):

**Exact diffuse initialization:**
- Decompose: P_0 = κ P_∞ + P_*
- P_∞ captures diffuse directions
- Run augmented Kalman filter

**Approximate diffuse initialization:**
```
P_{0|0} = κI,  κ = 10⁶  (very large)
```

Simpler but less rigorous.

**When to use:** Stationary models (long-run variance is bounded).

---

## 6. The Kalman Smoother

### 6.1 The Smoothing Problem

**Goal:** Compute E[α_t | Y_T] for t < T

Uses **future** observations y_{t+1}, ..., y_T to improve estimate.

**Result:** Smoothed estimates have lower variance:
```
P_{t|T} ≤ P_{t|t}  (better than filtered)
```

### 6.2 Fixed-Interval Smoother (Rauch-Tung-Striebel)

**Backward recursion** starting from t = T-1, T-2, ..., 1:

**Smoother gain:**
```
L_t = P_{t|t} T' P_{t+1|t}⁻¹
```

**Smoothed state:**
```
α̂_{t|T} = α̂_{t|t} + L_t (α̂_{t+1|T} - α̂_{t+1|t})
```

**Smoothed variance:**
```
P_{t|T} = P_{t|t} + L_t (P_{t+1|T} - P_{t+1|t}) L_t'
```

**Initialize:** α̂_{T|T}, P_{T|T} from filter

### 6.3 Derivation of Smoother

**Joint distribution:** (α_t, α_{t+1}) | Y_t is Gaussian

**Conditional distribution:**
```
α_t | α_{t+1}, Y_t ~ N(E[α_t | α_{t+1}, Y_t], Var[α_t | α_{t+1}, Y_t])
```

**Regression formula:**
```
E[α_t | α_{t+1}, Y_t] = E[α_t | Y_t] + Cov[α_t, α_{t+1} | Y_t] Var[α_{t+1} | Y_t]⁻¹ (α_{t+1} - E[α_{t+1} | Y_t])
```

**Identify terms:**
- E[α_t | Y_t] = α̂_{t|t}
- E[α_{t+1} | Y_t] = α̂_{t+1|t}
- Var[α_{t+1} | Y_t] = P_{t+1|t}
- Cov[α_t, α_{t+1} | Y_t] = P_{t|t} T'  (from state equation)

**Therefore:**
```
E[α_t | α_{t+1}, Y_t] = α̂_{t|t} + P_{t|t} T' P_{t+1|t}⁻¹ (α_{t+1} - α̂_{t+1|t})
```

**Taking expectation over α_{t+1} | Y_T:**
```
α̂_{t|T} = E[α_t | Y_T]
        = E[E[α_t | α_{t+1}, Y_t] | Y_T]
        = α̂_{t|t} + P_{t|t} T' P_{t+1|t}⁻¹ (α̂_{t+1|T} - α̂_{t+1|t})
        = α̂_{t|t} + L_t (α̂_{t+1|T} - α̂_{t+1|t})
```

Where L_t = P_{t|t} T' P_{t+1|t}⁻¹ is the **smoother gain**.

### 6.4 Lag-One Covariance Smoother

For EM algorithm, we also need:
```
P_{t,t-1|T} = Cov[α_t, α_{t-1} | Y_T]
```

**Recursion:**
```
P_{t,t-1|T} = P_{t|t} L_{t-1}' + L_t (P_{t+1,t|T} - T P_{t|t}) L_{t-1}'
```

**Initialize:** P_{T,T-1|T} = (I - K_T Z) T P_{T-1|T-1}

This is needed for M-step of EM algorithm when updating transition matrix T.

---

## 7. Properties of the Kalman Filter

### 7.1 Optimality

**Theorem 1 (BLUE):** Among all **linear** estimators, the Kalman filter minimizes MSE.

**Theorem 2 (MMSE):** Among **all** estimators (linear and nonlinear), the Kalman filter is optimal when noise is Gaussian.

### 7.2 Innovations Representation

**Theorem:** The innovations {v_t} are:
1. Zero mean: E[v_t | Y_{t-1}] = 0
2. Serially uncorrelated: E[v_t v_s'] = 0 for t ≠ s
3. Orthogonal to predictions: E[v_t α̂_{s|s}'] = 0 for all s < t

This is the **Wold decomposition**.

### 7.3 Steady-State Kalman Filter

If the model is time-invariant and stable, the filter converges:
```
P_{t|t-1} → P_∞  (steady-state prediction variance)
K_t → K_∞        (steady-state gain)
```

Where P_∞ solves the **Riccati equation:**
```
P_∞ = T P_∞ T' + R Q R' - T P_∞ Z' (Z P_∞ Z' + H)⁻¹ Z P_∞ T'
```

**Advantage:** Can precompute K_∞, then use:
```
α̂_{t|t} = T α̂_{t-1|t-1} + K_∞ (y_t - Z T α̂_{t-1|t-1})
```

Much faster than full Kalman filter.

---

## 8. Alternative Formulations

### 8.1 Information Filter

Works with **inverse** covariance (precision):
```
Y_t = P_{t|t}⁻¹          (information matrix)
ŷ_t = P_{t|t}⁻¹ α̂_{t|t}  (information vector)
```

**Advantages:**
- More stable when P_{t|t} is nearly singular
- Better for sparse systems

### 8.2 Square Root Filter

Works with **Cholesky factors** S_t where P_t = S_t S_t':

**Advantages:**
- Numerically stable (avoids P_t losing positive definiteness)
- Guaranteed positive definite covariance

**Disadvantage:**
- More complex implementation

### 8.3 Chandrasekhar Recursions

Exploits low-rank structure when r << N:

**Idea:** F_t changes slowly, so only update the change ΔF_t

**Advantage:** O(r²T) instead of O(r³T + N³T)

**When to use:** Large N, small r (many observations, few factors)

---

## 9. Numerical Stability

### 9.1 Common Numerical Issues

**Problem 1:** P_t becomes non-positive-definite due to rounding
**Solution:** Use Joseph form or square-root filter

**Problem 2:** F_t is nearly singular (H very small)
**Solution:** Use pseudo-inverse or regularization

**Problem 3:** Overflow/underflow in likelihood
**Solution:** Compute log-likelihood incrementally

### 9.2 Recommended Implementation

```python
def kalman_filter_stable(y, Z, T, R, Q, H, a0, P0):
    """Numerically stable Kalman filter."""
    T, N = y.shape
    r = a0.shape[0]

    # Storage
    a_pred = np.zeros((T+1, r))
    a_filt = np.zeros((T+1, r))
    P_pred = np.zeros((T+1, r, r))
    P_filt = np.zeros((T+1, r, r))

    # Initialize
    a_filt[0] = a0
    P_filt[0] = P0

    loglik = 0.0

    for t in range(T):
        # PREDICT
        a_pred[t+1] = T @ a_filt[t]
        P_pred[t+1] = T @ P_filt[t] @ T.T + R @ Q @ R.T

        # Handle missing data
        if np.any(np.isnan(y[t])):
            a_filt[t+1] = a_pred[t+1]
            P_filt[t+1] = P_pred[t+1]
            continue

        # UPDATE
        v = y[t] - Z @ a_pred[t+1]
        F = Z @ P_pred[t+1] @ Z.T + H

        # Regularize F if nearly singular
        F += 1e-10 * np.eye(N)

        K = P_pred[t+1] @ Z.T @ np.linalg.solve(F, np.eye(N))
        a_filt[t+1] = a_pred[t+1] + K @ v

        # Joseph form (stable)
        I_KZ = np.eye(r) - K @ Z
        P_filt[t+1] = I_KZ @ P_pred[t+1] @ I_KZ.T + K @ H @ K.T

        # Log-likelihood
        loglik += -0.5 * (N * np.log(2*np.pi) + np.linalg.slogdet(F)[1] + v.T @ np.linalg.solve(F, v))

    return a_filt, P_filt, loglik
```

---

## 10. Extensions

### 10.1 Missing Data

Simply skip update step when y_t is missing:
```python
if np.isnan(y_t):
    α̂_{t|t} = α̂_{t|t-1}
    P_{t|t} = P_{t|t-1}
```

Kalman filter handles this naturally!

### 10.2 Time-Varying System Matrices

Allow Z_t, T_t, etc. to change over time:
```
α_t = T_t α_{t-1} + R_t η_t
y_t = Z_t α_t + ε_t
```

Same recursions, just use time-subscripted matrices.

### 10.3 Correlated Noise

If Cov[η_t, ε_t] = S ≠ 0:

**Modified innovation variance:**
```
F_t = Z P_{t|t-1} Z' + H + Z R S + S' R' Z'
```

**Modified Kalman gain:**
```
K_t = (T P_{t|t-1} Z' + R S) F_t⁻¹
```

---

## References

### Classic Papers
- **Kalman, R.E. (1960).** "A New Approach to Linear Filtering and Prediction Problems."
- **Rauch, H.E., Tung, F. & Striebel, C.T. (1965).** "Maximum Likelihood Estimates of Linear Dynamic Systems."

### Textbooks
- **Anderson, B.D.O. & Moore, J.B. (1979).** *Optimal Filtering*.
- **Grewal, M.S. & Andrews, A.P. (2014).** *Kalman Filtering: Theory and Practice*.
- **Durbin, J. & Koopman, S.J. (2012).** *Time Series Analysis by State Space Methods*.

### Applications
- **Harvey, A.C. (1989).** *Forecasting, Structural Time Series Models and the Kalman Filter*.
- **Kim, C.-J. & Nelson, C.R. (1999).** *State-Space Models with Regime Switching*.
