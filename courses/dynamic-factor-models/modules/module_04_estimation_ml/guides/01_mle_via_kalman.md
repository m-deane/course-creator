# Maximum Likelihood Estimation via Kalman Filter

## In Brief

Maximum likelihood estimation of dynamic factor models uses the Kalman filter to construct the likelihood function via prediction error decomposition. The likelihood is computed recursively by evaluating one-step-ahead prediction errors and their variances at each time point, providing an exact likelihood evaluation even with latent factors.

> 💡 **Key Insight:** The genius of the Kalman filter approach is that it transforms an intractable likelihood (marginalizing over all latent factor paths) into a tractable sequential computation. Each observation contributes its prediction error - the surprise relative to what the model expected - and the likelihood measures how well the model predicts the data one step ahead.

---

## Formal Definition

For a dynamic factor model in state-space form:

**Measurement equation:**
$$X_t = Z \alpha_t + \varepsilon_t, \quad \varepsilon_t \sim N(0, H)$$

**State equation:**
$$\alpha_t = T \alpha_{t-1} + R\eta_t, \quad \eta_t \sim N(0, Q)$$

The **log-likelihood function** is:

$$\log L(\theta) = -\frac{TN}{2}\log(2\pi) - \frac{1}{2}\sum_{t=1}^T \left[\log|F_t(\theta)| + v_t(\theta)' F_t(\theta)^{-1} v_t(\theta)\right]$$

where:
- $\theta = \{Z, T, R, Q, H\}$ are the model parameters
- $v_t = X_t - Z\hat{\alpha}_{t|t-1}$ is the **prediction error** (innovation)
- $F_t = ZP_{t|t-1}Z' + H$ is the **prediction error variance**
- $\hat{\alpha}_{t|t-1}$ is the predicted state from Kalman filter
- $P_{t|t-1}$ is the predicted state covariance

---

## Intuitive Explanation

### Why Prediction Errors?

Think of the likelihood as measuring "How surprised is the model by the data?" At each time $t$:

1. **Predict**: Based on past data $(X_1, ..., X_{t-1})$ and current parameters, what do we expect $X_t$ to be?
   - Prediction: $\hat{X}_t = Z\hat{\alpha}_{t|t-1}$

2. **Compare**: How different is the actual observation from the prediction?
   - Prediction error: $v_t = X_t - \hat{X}_t$

3. **Evaluate**: How surprising is this error given the uncertainty?
   - Normalized squared error: $v_t' F_t^{-1} v_t$ (Mahalanobis distance)

4. **Accumulate**: Sum across all time points to get total likelihood

### The Decomposition

The key mathematical insight is the **prediction error decomposition** of the likelihood:

$$p(X_1, ..., X_T | \theta) = p(X_1|\theta) \prod_{t=2}^T p(X_t | X_1, ..., X_{t-1}, \theta)$$

Each term $p(X_t | X_1, ..., X_{t-1}, \theta)$ is the **one-step-ahead predictive density**, which is:

$$X_t | X_{1:t-1}, \theta \sim N(Z\hat{\alpha}_{t|t-1}, F_t)$$

This is exactly what the Kalman filter computes!

### Visual Representation

```
Time:     t=1      t=2      t=3      t=4     ...      t=T
          │        │        │        │                 │
Data:    X₁  →   X₂  →   X₃  →   X₄  →  ...  →      Xₜ
          │        │        │        │                 │
Predict:  -   →   X̂₂  →   X̂₃  →   X̂₄  →  ...  →      X̂ₜ
          │        │        │        │                 │
Error:    -   →   v₂  →   v₃  →   v₄  →  ...  →      vₜ
          │        │        │        │                 │
          └────────┴────────┴────────┴─────────────────┘

                    Likelihood = Product of
                    N(vₜ | 0, Fₜ) for all t
```

---

## Mathematical Formulation

### Step 1: State-Space Representation

Convert DFM to state-space form. For a DFM with $r$ factors and $p$ lags:

**Original DFM:**
$$X_t = \Lambda F_t + e_t$$
$$\Phi(L) F_t = \eta_t$$

**State-space form:**

Define state vector $\alpha_t = [F_t', F_{t-1}', ..., F_{t-p+1}']'$ (size $rp \times 1$)

$$Z = [\Lambda, 0, ..., 0]_{N \times rp}$$

$$T = \begin{bmatrix} \Phi_1 & \Phi_2 & \cdots & \Phi_{p-1} & \Phi_p \\ I_r & 0 & \cdots & 0 & 0 \\ 0 & I_r & \cdots & 0 & 0 \\ \vdots & \vdots & \ddots & \vdots & \vdots \\ 0 & 0 & \cdots & I_r & 0 \end{bmatrix}_{rp \times rp}$$

$$R = [I_r, 0, ..., 0]'_{rp \times r}, \quad Q = \Sigma_\eta$$

$$H = \Sigma_e = \text{diag}(\sigma_{e_1}^2, ..., \sigma_{e_N}^2)$$

### Step 2: Kalman Filter Recursions

**Initialize** at $t=0$:
$$\hat{\alpha}_{0|0} = 0$$
$$P_{0|0} = \text{solve}(P - TP T' = RQR')$$ (unconditional variance)

**For** $t = 1, ..., T$:

**Prediction step:**
$$\hat{\alpha}_{t|t-1} = T\hat{\alpha}_{t-1|t-1}$$
$$P_{t|t-1} = T P_{t-1|t-1} T' + RQR'$$

**Prediction error:**
$$v_t = X_t - Z\hat{\alpha}_{t|t-1}$$
$$F_t = ZP_{t|t-1}Z' + H$$

**Update step:**
$$K_t = P_{t|t-1}Z'F_t^{-1}$$ (Kalman gain)
$$\hat{\alpha}_{t|t} = \hat{\alpha}_{t|t-1} + K_t v_t$$
$$P_{t|t} = P_{t|t-1} - K_t F_t K_t'$$

### Step 3: Log-Likelihood Computation

Accumulate log-likelihood from prediction errors:

$$\log L(\theta) = \sum_{t=1}^T \log p(X_t | X_{1:t-1}, \theta)$$

$$= \sum_{t=1}^T \log\left[(2\pi)^{-N/2} |F_t|^{-1/2} \exp\left(-\frac{1}{2}v_t'F_t^{-1}v_t\right)\right]$$

$$= -\frac{TN}{2}\log(2\pi) - \frac{1}{2}\sum_{t=1}^T \left[\log|F_t| + v_t'F_t^{-1}v_t\right]$$

### Step 4: Maximization

To find $\hat{\theta}_{MLE}$, maximize $\log L(\theta)$ using numerical optimization:
- **Quasi-Newton methods**: BFGS, L-BFGS-B (with parameter bounds)
- **Nelder-Mead**: Derivative-free (slower but robust)
- **EM algorithm**: Iterative approach (next guide)

Gradient:
$$\frac{\partial \log L}{\partial \theta} = -\frac{1}{2}\sum_{t=1}^T \text{tr}\left[F_t^{-1}\frac{\partial F_t}{\partial \theta}\right] + \sum_{t=1}^T v_t'F_t^{-1}\frac{\partial v_t}{\partial \theta} + ...$$

Usually computed numerically via finite differences.

---

## Code Implementation

### Complete Likelihood Function

```python
import numpy as np
from scipy.linalg import solve, cholesky, solve_triangular
from scipy.optimize import minimize

def kalman_filter_likelihood(params, X, Z, T, R, constrain_fn=None):
    """
    Compute log-likelihood via Kalman filter.

    Parameters
    ----------
    params : array-like
        Parameter vector to optimize
    X : ndarray (T, N)
        Observed data
    Z, T, R : ndarray
        System matrices (can depend on params)
    constrain_fn : callable, optional
        Function to convert params to system matrices

    Returns
    -------
    neg_loglik : float
        Negative log-likelihood (for minimization)
    """
    T_obs, N = X.shape

    # Convert params to system matrices
    if constrain_fn is not None:
        Z, T_mat, R, Q, H = constrain_fn(params)
    else:
        # Assume params are already system matrices
        Q = params['Q']
        H = params['H']
        T_mat = T

    r = R.shape[1]  # Number of factors
    rp = T_mat.shape[0]  # State dimension

    # Initialize
    alpha_pred = np.zeros(rp)

    # Solve for unconditional variance: P = T P T' + R Q R'
    # Vectorize and solve linear system
    I_rp2 = np.eye(rp**2)
    T_kron = np.kron(T_mat, T_mat)
    vec_RQR = (R @ Q @ R.T).ravel()
    vec_P0 = solve(I_rp2 - T_kron, vec_RQR)
    P_pred = vec_P0.reshape(rp, rp)

    # Make symmetric
    P_pred = (P_pred + P_pred.T) / 2

    # Storage for likelihood computation
    loglik = 0.0

    for t in range(T_obs):
        # Prediction error
        v = X[t] - Z @ alpha_pred

        # Prediction error variance
        F = Z @ P_pred @ Z.T + H

        # Make symmetric and ensure positive definite
        F = (F + F.T) / 2

        # Cholesky decomposition for numerical stability
        try:
            L = cholesky(F, lower=True)
        except np.linalg.LinAlgError:
            # Add small regularization if not positive definite
            F += np.eye(N) * 1e-8
            L = cholesky(F, lower=True)

        # Log determinant via Cholesky
        log_det_F = 2 * np.sum(np.log(np.diag(L)))

        # Solve L L' x = v for x = F^{-1} v
        w = solve_triangular(L, v, lower=True)
        v_F_inv_v = w @ w

        # Accumulate log-likelihood
        loglik += -0.5 * (log_det_F + v_F_inv_v)

        # Kalman gain
        # K = P_pred Z' F^{-1}
        # Solve F^{-1} (Z P_pred') = (solve(F', Z P_pred'))'
        K = P_pred @ Z.T @ solve(F, np.eye(N))

        # Update
        alpha_filt = alpha_pred + K @ v
        P_filt = P_pred - K @ F @ K.T

        # Make symmetric
        P_filt = (P_filt + P_filt.T) / 2

        # Predict next period
        alpha_pred = T_mat @ alpha_filt
        P_pred = T_mat @ P_filt @ T_mat.T + R @ Q @ R.T

        # Make symmetric
        P_pred = (P_pred + P_pred.T) / 2

    # Add constant
    loglik += -0.5 * T_obs * N * np.log(2 * np.pi)

    return -loglik  # Return negative for minimization


def constrain_parameters(params, N, r, p):
    """
    Map unconstrained parameter vector to system matrices with identification.

    Identification restrictions:
    1. Lower triangular loadings (first r variables)
    2. Diagonal unit variance for first r factors
    3. Positive diagonal elements

    Parameters
    ----------
    params : array-like
        Unconstrained parameter vector
    N : int
        Number of variables
    r : int
        Number of factors
    p : int
        Number of lags

    Returns
    -------
    Z, T, R, Q, H : ndarray
        System matrices
    """
    idx = 0

    # Loadings: N x r
    # First r rows are lower triangular with positive diagonal
    Lambda = np.zeros((N, r))

    for i in range(r):
        for j in range(i+1):
            if i == j:
                Lambda[i, j] = np.exp(params[idx])  # Positive diagonal
            else:
                Lambda[i, j] = params[idx]
            idx += 1

    # Remaining rows are unrestricted
    n_remaining = N - r
    Lambda[r:, :] = params[idx:idx + n_remaining * r].reshape(n_remaining, r)
    idx += n_remaining * r

    # Transition matrix: r x r for each lag
    Phi = np.zeros((p, r, r))
    for lag in range(p):
        Phi[lag] = params[idx:idx + r**2].reshape(r, r)
        idx += r**2

    # State-space form
    rp = r * p
    Z = np.hstack([Lambda] + [np.zeros((N, r))] * (p-1))

    T_mat = np.zeros((rp, rp))
    T_mat[:r, :] = Phi.reshape(r, r*p)
    if p > 1:
        T_mat[r:, :rp-r] = np.eye(rp - r)

    R = np.zeros((rp, r))
    R[:r, :] = np.eye(r)

    # Factor innovation variance: identity (identified)
    Q = np.eye(r)

    # Idiosyncratic variances: diagonal
    log_sigmas = params[idx:idx + N]
    H = np.diag(np.exp(log_sigmas))

    return Z, T_mat, R, Q, H


# Example: Estimate DFM parameters
def estimate_dfm_mle(X, r=2, p=1, verbose=True):
    """
    Estimate dynamic factor model via maximum likelihood.

    Parameters
    ----------
    X : ndarray (T, N)
        Observed data (demeaned)
    r : int
        Number of factors
    p : int
        Number of lags
    verbose : bool
        Print optimization progress

    Returns
    -------
    result : dict
        MLE estimates and diagnostics
    """
    T, N = X.shape

    # Count parameters
    n_loadings = r * (r + 1) // 2 + (N - r) * r  # Lower tri + rest
    n_transition = p * r * r
    n_variances = N
    n_params = n_loadings + n_transition + n_variances

    print(f"Estimating {r}-factor DFM with {p} lag(s)")
    print(f"Total parameters: {n_params}")

    # Initialize from PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=r)
    F_init = pca.fit_transform(X)
    Lambda_init = pca.components_.T  # N x r

    # Initialize AR coefficients via OLS
    from statsmodels.tsa.api import VAR
    var_model = VAR(F_init)
    var_result = var_model.fit(p)
    Phi_init = np.array([var_result.params[i*r:(i+1)*r, :].T for i in range(1, p+1)])

    # Initialize variances
    resid_X = X - F_init @ Lambda_init.T
    sigmas_init = np.std(resid_X, axis=0)

    # Construct initial parameter vector
    params_init = []

    # Loadings (lower triangular)
    for i in range(r):
        for j in range(i+1):
            if i == j:
                params_init.append(np.log(np.abs(Lambda_init[i, j]) + 0.1))
            else:
                params_init.append(Lambda_init[i, j])

    # Remaining loadings
    params_init.extend(Lambda_init[r:, :].ravel())

    # Transition coefficients
    params_init.extend(Phi_init.ravel())

    # Log variances
    params_init.extend(np.log(sigmas_init))

    params_init = np.array(params_init)

    # Objective function
    def objective(params):
        return kalman_filter_likelihood(
            params, X, None, None, None,
            constrain_fn=lambda p: constrain_parameters(p, N, r, p)
        )

    # Optimize
    result = minimize(
        objective,
        params_init,
        method='L-BFGS-B',
        options={'maxiter': 500, 'disp': verbose}
    )

    if not result.success:
        print(f"Warning: Optimization did not converge: {result.message}")

    # Extract estimates
    Z_hat, T_hat, R_hat, Q_hat, H_hat = constrain_parameters(result.x, N, r, p)
    Lambda_hat = Z_hat[:, :r]
    Phi_hat = T_hat[:r, :r*p].reshape(r, r*p)
    sigma_e_hat = np.sqrt(np.diag(H_hat))

    # Standard errors via observed information matrix (Hessian)
    # Approximate with finite differences
    from scipy.optimize import approx_fprime

    def grad_fn(x):
        return approx_fprime(x, objective, 1e-8)

    # This is expensive - skip for large models
    if n_params < 50:
        H_numerical = np.zeros((n_params, n_params))
        grad0 = grad_fn(result.x)
        for i in range(n_params):
            eps = np.zeros(n_params)
            eps[i] = 1e-6
            grad_plus = grad_fn(result.x + eps)
            H_numerical[i, :] = (grad_plus - grad0) / 1e-6

        # Observed information
        obs_info = (H_numerical + H_numerical.T) / 2

        try:
            cov_matrix = np.linalg.inv(obs_info)
            std_errors = np.sqrt(np.diag(cov_matrix))
        except:
            std_errors = None
    else:
        std_errors = None

    return {
        'Lambda': Lambda_hat,
        'Phi': Phi_hat,
        'sigma_e': sigma_e_hat,
        'Q': Q_hat,
        'loglik': -result.fun,
        'aic': 2 * result.fun + 2 * n_params,
        'bic': 2 * result.fun + n_params * np.log(T),
        'converged': result.success,
        'n_iterations': result.nit,
        'std_errors': std_errors,
        'result': result
    }


# Test with simulated data
if __name__ == '__main__':
    np.random.seed(42)

    # Simulate DFM
    T, N, r, p = 200, 10, 2, 1

    # True parameters
    Lambda_true = np.random.randn(N, r)
    Phi_true = np.array([[0.8, 0.1], [0.1, 0.7]])
    sigma_e_true = np.ones(N) * 0.5

    # Simulate factors
    F = np.zeros((T, r))
    for t in range(1, T):
        F[t] = Phi_true @ F[t-1] + np.random.randn(r)

    # Simulate data
    X = F @ Lambda_true.T + np.random.randn(T, N) * sigma_e_true

    # Demean
    X = X - X.mean(axis=0)

    # Estimate
    result = estimate_dfm_mle(X, r=2, p=1)

    print("\n=== Estimation Results ===")
    print(f"Log-likelihood: {result['loglik']:.2f}")
    print(f"AIC: {result['aic']:.2f}")
    print(f"BIC: {result['bic']:.2f}")
    print(f"Converged: {result['converged']}")

    print("\nEstimated Phi:")
    print(result['Phi'][:, :r].round(3))
    print("\nTrue Phi:")
    print(Phi_true.round(3))
```

---

## Common Pitfalls

### 1. Numerical Instability

**Problem:** $F_t$ can become ill-conditioned, leading to inaccurate inversions.

**Solution:**
- Use Cholesky decomposition: $F_t = LL'$, then solve $L^{-1}v$ instead of $F_t^{-1}v$
- Add small regularization: $F_t \leftarrow F_t + \epsilon I$ with $\epsilon = 10^{-8}$
- Ensure $P_{t|t-1}$ stays symmetric: $P \leftarrow (P + P')/2$

### 2. Non-Stationarity

**Problem:** If factors are non-stationary, $P_{0|0}$ is undefined.

**Solution:**
- Use **diffuse initialization**: $P_{0|0} = \kappa I$ with $\kappa \to \infty$
- In practice, set $\kappa = 10^7$ for first $r$ states
- Or difference data to induce stationarity

### 3. Identification

**Problem:** Likelihood invariant to rotations without constraints.

**Solution:** Impose identification before optimization:
- Lower triangular loadings for first $r$ variables
- Fix factor scale: $Q = I_r$ or $\Lambda'\Lambda = I_r$
- Order factors by variance explained

### 4. Local Optima

**Problem:** Non-convex likelihood surface with multiple peaks.

**Solution:**
- Initialize from PCA estimates (usually near global optimum)
- Try multiple random starting values
- Use EM algorithm (guaranteed to increase likelihood)

---

## Connections

- **Builds on:** Kalman filter (Module 2), state-space representation
- **Leads to:** EM algorithm (next guide), Bayesian estimation (Guide 3)
- **Related to:** Quasi-maximum likelihood (allows for misspecification)

---

## Practice Problems

### Conceptual

1. Why does the prediction error decomposition avoid integrating over all possible factor paths?

2. Explain why $F_t$ shrinks toward $H$ as the signal-to-noise ratio decreases.

3. What happens to the likelihood if we set $H = 0$ (no measurement error)?

### Mathematical

4. Derive the gradient $\frac{\partial \log L}{\partial \Lambda}$ for a single observation.

5. Show that the log-likelihood simplifies to OLS when $F_t$ is known (no latent factors).

6. Prove that the Kalman filter provides the minimum mean-squared error predictor.

### Implementation

7. Modify the code to handle missing data by skipping unavailable observations in $v_t$.

8. Implement quasi-maximum likelihood by replacing $H$ with robust covariance estimator.

9. Compare ML estimates with PCA for different $T/N$ ratios. When does ML outperform PCA?

---

## Further Reading

- **Durbin & Koopman (2012).** *Time Series Analysis by State Space Methods*, 2nd ed., Chapter 7.
  - Comprehensive treatment of likelihood evaluation via Kalman filter

- **Hamilton (1994).** *Time Series Analysis*, Chapter 13.
  - State-space representation and maximum likelihood

- **Harvey (1989).** *Forecasting, Structural Time Series Models and the Kalman Filter*.
  - Applications to structural models and forecasting

- **Shumway & Stoffer (2017).** *Time Series Analysis and Its Applications*, 4th ed.
  - Accessible introduction with R code

- **Doz, Giannone & Reichlin (2012).** "A Quasi-Maximum Likelihood Approach for Large, Approximate Dynamic Factor Models." *Review of Economics and Statistics* 94(4), 1014-1024.
  - Two-step QML for computational efficiency with large N
