# Module 02 Cheatsheet: Estimation Methods

## Three Estimation Approaches

### 1. Maximum Likelihood (ML)
```python
# Objective: Maximize log L(θ) = Σ_t log p(y_t | y₁:t-1, θ)

from scipy.optimize import minimize

def neg_loglik(params, y, Z, T, R):
    Q, H = unpack_params(params)
    _, _, _, loglik = kalman_filter(y, Z, T, R, Q, H)
    return -loglik

θ_hat = minimize(neg_loglik, θ_init, args=(y, Z, T, R))
```

**Pros:** Fast, standard errors available
**Cons:** Can get stuck in local optima

### 2. EM Algorithm
```python
# Iterate: E-step (smooth) → M-step (update params)

for iter in range(max_iter):
    # E-step: Kalman smoother
    E_alpha, E_alpha_alpha = smoother(y, θ_old)

    # M-step: Update parameters
    Λ_new = (Σ y·E_alpha') @ (Σ E_alpha_alpha)^{-1}
    Φ_new = update_Phi(E_alpha)
    Q_new, H_new = update_covariances()

    θ_old = (Λ_new, Φ_new, Q_new, H_new)
```

**Pros:** Numerically stable, handles missing data
**Cons:** Slower convergence than ML

### 3. Bayesian (Gibbs Sampling)
```python
# Sample from posterior: θ, F | y

for i in range(n_draws):
    # Sample factors
    F = ffbs(y, θ)  # Forward-filter backward-sample

    # Sample parameters
    Λ = sample_Λ(y, F)
    Φ = sample_Φ(F)
    Q = sample_Q(F, Φ)
    H = sample_H(y, F, Λ)
```

**Pros:** Full uncertainty quantification
**Cons:** Computationally expensive

## Log-Likelihood Formula

```python
loglik = 0
for t in range(T):
    v_t = y[t] - Z @ a_pred[t]  # Innovation
    F_t = Z @ P_pred[t] @ Z.T + H

    loglik += -0.5 * (p*log(2π) + log|F_t| + v_t'·F_t^{-1}·v_t)
```

## EM Update Formulas

```python
# M-step updates

# Loadings
Λ = (Σ_t y_t·α_t^s') @ (Σ_t α_t^s·α_t^s' + V_t^s)^{-1}

# Transition
Φ = (Σ_t α_t^s·α_{t-1}^s') @ (Σ_t α_{t-1}^s·α_{t-1}^s' + V_{t-1}^s)^{-1}

# State covariance
Q = (1/T) Σ_t [α_t^s·α_t^s' + V_t^s - Φ·(α_t^s·α_{t-1}^s' + V_{t,t-1}^s)]

# Observation covariance
H = (1/T) Σ_t [(y_t - Λ·α_t^s)(y_t - Λ·α_t^s)' + Λ·V_t^s·Λ']
```

Where:
- α_t^s = E[α_t | y₁:T] (smoothed state)
- V_t^s = Var[α_t | y₁:T] (smoothed covariance)

## Comparison Table

| Method | Speed | Accuracy | Uncertainty | Missing Data |
|--------|-------|----------|-------------|--------------|
| ML     | Fast  | High     | Asymptotic  | Yes          |
| EM     | Medium| High     | Asymptotic  | Excellent    |
| Bayesian| Slow | High     | Full        | Yes          |

## Initialization Strategies

```python
# PCA initialization (all methods)
from sklearn.decomposition import PCA

pca = PCA(n_components=r)
F_init = pca.fit_transform(X)
Λ_init = pca.components_.T

# VAR for transition
from statsmodels.tsa.api import VAR
var_model = VAR(F_init)
var_results = var_model.fit(maxlags=1)
Φ_init = var_results.params[1:].T

# Residual variances
Q_init = np.cov(var_results.resid.T)
H_init = np.diag(np.var(X - F_init @ Λ_init.T, axis=0))
```

## Convergence Diagnostics

```python
# ML: Check gradient norm
if np.linalg.norm(gradient) < 1e-4:
    print("ML converged")

# EM: Monitor log-likelihood
if abs(loglik - loglik_old) / abs(loglik_old) < 1e-4:
    print("EM converged")

# Bayesian: Trace plots and Gelman-Rubin statistic
from arviz import rhat
print(f"R-hat: {rhat(posterior_draws)}")
```

## Standard Errors

```python
# ML: From Hessian
from scipy.optimize import approx_fprime

hessian = approx_fprime(θ_hat, grad_loglik)
se = np.sqrt(np.diag(np.linalg.inv(-hessian)))

# Bayesian: From posterior
se_bayesian = np.std(posterior_draws, axis=0)
```

## Resources

- **statsmodels**: `DynamicFactor.fit()` uses ML
- **PyMC**: For Bayesian estimation
- **filterpy**: Educational EM implementation
