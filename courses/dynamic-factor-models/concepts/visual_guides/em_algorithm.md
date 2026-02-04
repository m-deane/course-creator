# EM Algorithm for Dynamic Factor Models

```
┌─────────────────────────────────────────────────────────────────────┐
│ EM ALGORITHM (EXPECTATION-MAXIMIZATION)                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Iterative procedure for missing data problems:                   │
│                                                                     │
│   Initialize: θ⁰ = (Λ⁰, T⁰, Q⁰, H⁰)                                │
│        │                                                            │
│        ↓                                                            │
│   ┌────────────────────────────────┐                               │
│   │  E-STEP                        │                               │
│   │  Given θᵏ, estimate factors    │                               │
│   │                                │                               │
│   │  Run Kalman filter & smoother: │                               │
│   │    E[f_t | y₁:T, θᵏ]           │                               │
│   │    E[f_t f_t' | y₁:T, θᵏ]      │                               │
│   │    E[f_t f_{t-1}' | y₁:T, θᵏ]  │                               │
│   └────────────────────────────────┘                               │
│        │                                                            │
│        ↓                                                            │
│   ┌────────────────────────────────┐                               │
│   │  M-STEP                        │                               │
│   │  Given E[factors], update θ    │                               │
│   │                                │                               │
│   │  Λᵏ⁺¹ = (∑ y_t E[f_t]') · (∑ E[f_t f_t'])⁻¹                    │
│   │  Tᵏ⁺¹ = (∑ E[f_t f_{t-1}']) · (∑ E[f_{t-1} f_{t-1}'])⁻¹        │
│   │  Update Q, H via residuals     │                               │
│   └────────────────────────────────┘                               │
│        │                                                            │
│        ↓                                                            │
│   Check convergence: |θᵏ⁺¹ - θᵏ| < ε ?                              │
│        │                                                            │
│        └──────→ NO: k = k+1, repeat                                │
│                YES: Return θ̂ = θᵏ⁺¹                                 │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ TL;DR: Iteratively estimate factors (given parameters) and update  │
│        parameters (given factors) until convergence. Guaranteed    │
│        to increase likelihood each step.                           │
├─────────────────────────────────────────────────────────────────────┤
│ Code (< 15 lines):                                                  │
│                                                                     │
│   def em_dfm(y, n_factors, max_iter=100, tol=1e-6):                │
│       # Initialize with PCA                                         │
│       theta = initialize_params(y, n_factors)                       │
│       for i in range(max_iter):                                     │
│           # E-step: Kalman smoother                                 │
│           factors = kalman_smooth(y, theta)                         │
│           # M-step: Update parameters                               │
│           theta_new = update_params(y, factors)                     │
│           # Check convergence                                       │
│           if np.max(np.abs(theta_new - theta)) < tol:               │
│               break                                                 │
│           theta = theta_new                                         │
│       return theta, factors                                         │
├─────────────────────────────────────────────────────────────────────┤
│ Common Pitfall: Bad initialization leads to local optima. Always   │
│                 initialize with PCA/Stock-Watson estimates!        │
└─────────────────────────────────────────────────────────────────────┘
```

## Why EM? The Missing Data Perspective

**Key insight:** Factors are "missing data" - we don't observe them directly.

Standard MLE: Maximize L(θ | y)
Problem: L(θ | y) involves integrating out factors - no closed form!

EM insight: If we knew factors, MLE would be easy. If we knew θ, estimating factors is easy (Kalman filter). So iterate!

## The EM Guarantee

**Theorem:** Each EM iteration increases the likelihood:
```
L(θᵏ⁺¹ | y) ≥ L(θᵏ | y)
```

This monotone improvement guarantees convergence (to a local maximum).

## E-Step: Kalman Smoother

The E-step computes "sufficient statistics" for the M-step:

```python
from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother

# Run Kalman smoother
ks = KalmanSmoother(k_endog=N, k_states=r)
ks['design'] = Lambda
ks['transition'] = T
ks['obs_cov'] = H
ks['state_cov'] = Q

result = ks.smooth(data)

# Extract sufficient statistics
f_smooth = result.smoothed_state              # E[f_t | y₁:T, θᵏ]
P_smooth = result.smoothed_state_cov          # Var[f_t | y₁:T, θᵏ]
f_cov_lag = result.smoothed_state_cov_lag     # Cov[f_t, f_{t-1}]
```

Key point: We need **smoothed** (not filtered) estimates because M-step uses all T observations.

## M-Step: Closed-Form Updates

Given sufficient statistics from E-step, M-step has **closed-form solutions**:

### Update Loadings (Λ)
```python
# Λ = (∑ y_t f_t') (∑ f_t f_t')⁻¹
numerator = (data @ f_smooth.T) / T
denominator = (f_smooth @ f_smooth.T + P_smooth.sum(axis=0)) / T
Lambda_new = numerator @ np.linalg.inv(denominator)
```

### Update Transition (T)
```python
# T = (∑ f_t f_{t-1}') (∑ f_{t-1} f_{t-1}')⁻¹
numerator = f_cov_lag.sum(axis=0) / T
denominator = (f_smooth[:, :-1] @ f_smooth[:, :-1].T + P_smooth[:-1].sum(axis=0)) / T
T_new = numerator @ np.linalg.inv(denominator)
```

### Update Covariances (Q, H)
```python
# Q = Var[f_t - T f_{t-1}]
residuals_state = f_smooth[:, 1:] - T_new @ f_smooth[:, :-1]
Q_new = residuals_state @ residuals_state.T / T

# H = Var[y_t - Λ f_t]
residuals_obs = data - Lambda_new @ f_smooth
H_new = np.diag(np.diag(residuals_obs @ residuals_obs.T / T))  # Diagonal
```

## Convergence Criteria

Monitor three metrics:

1. **Parameter change:**
```python
param_change = np.max(np.abs(theta_new - theta))
if param_change < 1e-6: converged = True
```

2. **Log-likelihood change:**
```python
ll_change = np.abs(loglik_new - loglik)
if ll_change < 1e-4: converged = True
```

3. **Maximum iterations:**
```python
if iter > 100: break  # Safety valve
```

## Practical Tips

### Initialization Matters
**Good:** PCA or Stock-Watson estimates
```python
pca = PCA(n_components=r)
f_init = pca.fit_transform(data)
Lambda_init = pca.components_.T
```

**Bad:** Random initialization (often gets stuck)

### Identification Constraints
Need to impose constraints to avoid indeterminacy:

**Option 1:** Fix block of Λ
```python
Lambda[0:r, 0:r] = np.eye(r)  # First r rows = identity
```

**Option 2:** Normalize Λ'Λ
```python
# Keep Λ'Λ/T = I
```

### Speed Tips
- Use sparse matrices if H is diagonal
- Exploit structure in Kalman filter (many zeros)
- Parallel E-step if data is panel (independent cross-sections)

## EM vs. Direct Optimization

**EM Advantages:**
- Guaranteed to increase likelihood
- No step size tuning needed
- Each step is simple (closed-form)
- Numerically stable

**EM Disadvantages:**
- Can be slow near convergence
- No standard errors (need numerical Hessian or bootstrap)
- Local optima (need good initialization)

**Alternative:** Quasi-Newton on likelihood directly
- Faster near optimum
- Gives Hessian for standard errors
- But requires derivatives and can be unstable

**Best practice:** Start with EM (robust), finish with quasi-Newton (fast).

## Complete Algorithm

```python
def em_algorithm(data, n_factors, max_iter=100):
    # 1. Initialize with PCA
    theta = initialize_pca(data, n_factors)
    loglik_old = -np.inf

    for iteration in range(max_iter):
        # 2. E-step: Kalman smoother
        factors, P, P_lag = kalman_smoother(data, theta)

        # 3. M-step: Update parameters
        Lambda = update_loadings(data, factors, P)
        T = update_transition(factors, P, P_lag)
        Q = update_state_cov(factors, T, P)
        H = update_obs_cov(data, Lambda, factors, P)

        theta = {'Lambda': Lambda, 'T': T, 'Q': Q, 'H': H}

        # 4. Check convergence
        loglik = compute_loglik(data, theta)
        if abs(loglik - loglik_old) < 1e-4:
            break
        loglik_old = loglik

    return theta, factors
```

## Real-World Performance

Typical convergence:
- **Iterations:** 10-50 (depends on initialization)
- **Time:** O(T · N · r²) per iteration
- **Memory:** O(T · r²) for Kalman filter

For FRED-MD (N=127, T=500, r=3):
- ~5 seconds per EM iteration on modern CPU
- ~20 iterations to convergence
- Total: ~2 minutes

Much faster than MCMC (hours) but slower than PCA (seconds).
