# EM Algorithm for DFM Estimation

## In Brief

The Expectation-Maximization (EM) algorithm iteratively estimates DFM parameters by alternating between computing expected sufficient statistics (E-step, using Kalman smoother) and updating parameters (M-step, using closed-form formulas). It's more stable than direct ML optimization and handles missing data naturally.

## Key Insight

**The iterative approach:**
1. **E-step:** Pretend you know parameters → run Kalman smoother → get expected factors
2. **M-step:** Pretend you know factors → update parameters using regression
3. **Repeat** until convergence

Guaranteed to increase likelihood each iteration!

## Formal Definition

**E-step:** Compute E[α(t)|y₁:T, θ⁽ⁱ⁾] and E[α(t)α(s)'|y₁:T, θ⁽ⁱ⁾] using Kalman smoother.

**M-step:** Update parameters:
```
Λ⁽ⁱ⁺¹⁾ = [Σ_t y(t)·E[α(t)]'] [Σ_t E[α(t)α(t)']']⁻¹

Φ⁽ⁱ⁺¹⁾ = [Σ_t E[α(t)α(t-1)']'] [Σ_t E[α(t-1)α(t-1)']']⁻¹

Q⁽ⁱ⁺¹⁾ = (1/T)Σ_t [E[α(t)α(t)'] - Φ⁽ⁱ⁺¹⁾·E[α(t)α(t-1)']']

H⁽ⁱ⁺¹⁾ = (1/T)Σ_t [y(t)y(t)' - Λ⁽ⁱ⁺¹⁾·E[α(t)]'·y(t)']
```

## Code Implementation

```python
def em_algorithm(y, Z_init, T_init, R, Q_init, H_init, max_iter=100, tol=1e-4):
    """EM algorithm for DFM estimation."""
    # Initialize
    Z, T, Q, H = Z_init.copy(), T_init.copy(), Q_init.copy(), H_init.copy()
    loglik_old = -np.inf

    for iteration in range(max_iter):
        # E-step: Kalman smoother
        a_smooth, V_smooth, V_cross = kalman_smoother(y, Z, T, R, Q, H)

        # M-step: Update parameters
        # Update Λ (loadings)
        sum_ya = np.zeros((y.shape[1], T.shape[0]))
        sum_aa = np.zeros((T.shape[0], T.shape[0]))

        for t in range(len(y)):
            if not np.isnan(y[t]).any():
                sum_ya += np.outer(y[t], a_smooth[t])
                sum_aa += V_smooth[t] + np.outer(a_smooth[t], a_smooth[t])

        Z = sum_ya @ np.linalg.inv(sum_aa)

        # Update Q, H similarly...

        # Compute log-likelihood
        _, _, _, loglik = kalman_filter(y, Z, T, R, Q, H)

        # Check convergence
        if abs(loglik - loglik_old) < tol:
            print(f"Converged after {iteration+1} iterations")
            break

        loglik_old = loglik

    return Z, T, Q, H

# Usage
Z_hat, T_hat, Q_hat, H_hat = em_algorithm(y, Z_init, T_init, R, Q_init, H_init)
```

## Common Pitfalls

### 1. Slow Convergence
**Problem:** EM takes many iterations.
**Solution:** Use good initialization (PCA), implement acceleration schemes.

### 2. Identification Drift
**Problem:** Parameters drift during iteration.
**Solution:** Re-impose identification after each M-step.

## Practice Problems

1. Implement EM for local level model
2. Compare convergence speed to direct ML
3. Test with 30% missing data

## Further Reading

- Shumway & Stoffer (1982): "An approach to time series smoothing and forecasting"
- Watson & Engle (1983): "Alternative algorithms for estimation of dynamic factor models"
