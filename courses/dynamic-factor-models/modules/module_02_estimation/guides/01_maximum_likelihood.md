# Maximum Likelihood Estimation for DFMs

## In Brief

Maximum likelihood estimates DFM parameters by maximizing the likelihood function, which is computed as a by-product of the Kalman filter. The prediction error decomposition converts the joint likelihood into a product of one-step-ahead forecast errors, making computation feasible for high-dimensional systems.

## Key Insight

**The elegant trick:** The Kalman filter gives you the likelihood for free! As you filter, you compute innovations v(t) and their covariance F(t). The log-likelihood is just:

```
log L = Σ_t [-0.5 × (log|F(t)| + v(t)'·F(t)⁻¹·v(t))]
```

No need to compute the full N×T joint density—just sum up the one-step-ahead forecast errors.

## Formal Definition

**Likelihood Function:**
```
L(θ) = p(y₁,...,y_T | θ)
     = p(y₁|θ) × p(y₂|y₁,θ) × ... × p(y_T|y₁:T-1,θ)
```

**Log-Likelihood:**
```
log L(θ) = Σ_t log p(y_t | y₁:t-1, θ)
         = Σ_t [-0.5(p·log(2π) + log|F_t| + v_t'·F_t⁻¹·v_t)]
```

Where:
- v(t) = y(t) - E[y(t)|y₁:t-1] (innovation from Kalman filter)
- F(t) = Var(v(t)) (innovation variance)
- θ = {Λ, Φ, Q, H} (all parameters)

**ML Estimator:**
```
θ̂_ML = argmax_θ log L(θ)
```

## Code Implementation

```python
import numpy as np
from scipy.optimize import minimize

def kalman_loglik(params, y, Z, T, R):
    """
    Compute negative log-likelihood for DFM.

    params: vectorized parameters [Q_vec, H_vec]
    """
    n, p = y.shape
    r = T.shape[0] // 2  # Assuming companion form

    # Unpack parameters
    Q = np.diag(params[:r])
    H = np.diag(params[r:r+p])

    # Initialize
    a = np.zeros(T.shape[0])
    P = np.eye(T.shape[0]) * 10
    loglik = 0.0

    for t in range(n):
        # Predict
        a_pred = T @ a
        P_pred = T @ P @ T.T + R @ Q @ R.T

        # Update
        if not np.isnan(y[t]).any():
            v = y[t] - Z @ a_pred
            F = Z @ P_pred @ Z.T + H
            F_inv = np.linalg.inv(F)
            K = P_pred @ Z.T @ F_inv

            a = a_pred + K @ v
            P = P_pred - K @ F @ K.T

            # Add to log-likelihood
            loglik += -0.5 * (p * np.log(2*np.pi) +
                             np.log(np.linalg.det(F)) +
                             v @ F_inv @ v)
        else:
            a = a_pred
            P = P_pred

    return -loglik  # Return negative for minimization

# ML estimation
initial_params = np.concatenate([np.ones(r), np.ones(p)])
result = minimize(
    kalman_loglik,
    initial_params,
    args=(y, Z, T, R),
    method='BFGS',
    options={'maxiter': 500}
)

Q_hat = np.diag(result.x[:r])
H_hat = np.diag(result.x[r:])
print(f"ML estimates obtained. Log-lik: {-result.fun:.2f}")
```

## Common Pitfalls

### 1. Numerical Optimization Failure
**Problem:** Optimizer doesn't converge or gets stuck.
**Solution:** Use good initial values (from PCA), impose bounds on parameters.

### 2. Non-Positive Definite Q or H
**Problem:** Optimization produces negative variances.
**Solution:** Optimize log(variance) or use constrained optimization.

### 3. Identification Not Imposed
**Problem:** Likelihood is flat in some directions.
**Solution:** Fix identification restrictions before optimization.

## Practice Problems

1. Implement ML estimation for simple local level model
2. Compare ML estimates to true parameters in simulation
3. Compute standard errors from Hessian matrix

## Further Reading

- Durbin & Koopman (2012), Chapter 7: ML estimation
- Hamilton (1994), Chapter 13: Likelihood evaluation
