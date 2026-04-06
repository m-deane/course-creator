# EM Algorithm for Dynamic Factor Models

> **Reading time:** ~12 min | **Module:** Module 4: Estimation Ml | **Prerequisites:** Modules 0-3

<div class="callout-key">

**Key Concept Summary:** The Expectation-Maximization (EM) algorithm provides an iterative approach to maximum likelihood estimation that treats the latent factors as missing data. It alternates between computing expected sufficient statistics given current parameters (E-step via Kalman smoother) and maximizing the expec...

</div>

## In Brief

The Expectation-Maximization (EM) algorithm provides an iterative approach to maximum likelihood estimation that treats the latent factors as missing data. It alternates between computing expected sufficient statistics given current parameters (E-step via Kalman smoother) and maximizing the expected complete-data log-likelihood (M-step with closed-form updates).

<div class="callout-insight">

**Insight:** Direct maximization of the likelihood is challenging because factors are latent. The EM algorithm's brilliance is treating factors as "missing data" - if we knew the factors, ML estimation would be trivial (just regression). So we iterate: (1) pretend we know the factors by using their conditional expectations, (2) update parameters as if those expectations were true, (3) repeat until convergence. Each iteration is guaranteed to increase the likelihood.

</div>
---

<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## Intuitive Explanation

## Formal Definition

For DFM in state-space form with observed data $X_{1:T}$ and latent states $\alpha_{1:T}$:

**Complete-data log-likelihood:**
$$\log L_c(\theta; X, \alpha) = \log p(X_{1:T}, \alpha_{1:T} | \theta)$$

**EM Algorithm:**

**E-step:** Compute expected complete-data log-likelihood given current parameters $\theta^{(k)}$:
$$Q(\theta | \theta^{(k)}) = E[\log L_c(\theta; X, \alpha) | X_{1:T}, \theta^{(k)}]$$

**M-step:** Update parameters by maximizing $Q$:
$$\theta^{(k+1)} = \arg\max_\theta Q(\theta | \theta^{(k)})$$

**Convergence:** Repeat until $|\log L(\theta^{(k+1)}) - \log L(\theta^{(k)})| < \epsilon$

**Theorem:** EM guarantees $\log L(\theta^{(k+1)}) \geq \log L(\theta^{(k)})$ (non-decreasing likelihood).

---

### The Missing Data Perspective

Consider estimating $\Lambda$ when factors $F_t$ are observed:
$$X_t = \Lambda F_t + e_t$$

This is just multivariate regression! OLS gives:
$$\hat{\Lambda} = \left(\sum_t X_t F_t'\right)\left(\sum_t F_t F_t'\right)^{-1}$$

**Problem:** $F_t$ is latent.

**EM Solution:** Replace $F_t$ with conditional expectations $\hat{F}_{t|T}$ given all data:
$$\hat{\Lambda} = \left(\sum_t X_t \hat{F}_{t|T}'\right)\left(\sum_t [\hat{F}_{t|T}\hat{F}_{t|T}' + P_{t|T}]\right)^{-1}$$

The $P_{t|T}$ term corrects for uncertainty about the factors!

### Why It Works

The EM algorithm exploits a mathematical identity:

$$\log L(\theta) = Q(\theta | \theta^{(k)}) - E[\log p(\alpha | X, \theta) | X, \theta^{(k)}]$$

Maximizing $Q$ with respect to $\theta$ increases $\log L(\theta)$ because the second term depends only on $\theta^{(k)}$.

### Visual Flow

```

Iteration k:
    θ⁽ᵏ⁾ → Kalman Filter → Predictions
           ↓
    θ⁽ᵏ⁾ → Kalman Smoother → Smoothed factors & covariances
           ↓                   {α̂ₜ|ₜ, Pₜ|ₜ, Pₜ,ₜ₋₁|ₜ}
           ↓
         M-step ← Sufficient statistics
           ↓
         θ⁽ᵏ⁺¹⁾ → Check convergence
```

---

## Mathematical Formulation

### Complete-Data Log-Likelihood

For state-space model:
$$\log L_c = \log p(\alpha_1) + \sum_{t=2}^T \log p(\alpha_t | \alpha_{t-1}) + \sum_{t=1}^T \log p(X_t | \alpha_t)$$

$$= -\frac{1}{2}\log|P_1| - \frac{1}{2}(\alpha_1 - \mu_1)'P_1^{-1}(\alpha_1 - \mu_1)$$
$$- \frac{T-1}{2}\log|Q| - \frac{1}{2}\sum_{t=2}^T (\alpha_t - T\alpha_{t-1})'R'Q^{-1}R(\alpha_t - T\alpha_{t-1})$$
$$- \frac{T}{2}\log|H| - \frac{1}{2}\sum_{t=1}^T (X_t - Z\alpha_t)'H^{-1}(X_t - Z\alpha_t) + \text{const}$$

### E-Step: Sufficient Statistics

To compute $Q(\theta | \theta^{(k)})$, we need expectations of sufficient statistics:

1. **Smoothed states:**
   $$\hat{\alpha}_{t|T}^{(k)} = E[\alpha_t | X_{1:T}, \theta^{(k)}]$$

2. **Smoothed state covariances:**
   $$P_{t|T}^{(k)} = \text{Var}[\alpha_t | X_{1:T}, \theta^{(k)}]$$

3. **Smoothed lag-one covariances:**
   $$P_{t,t-1|T}^{(k)} = \text{Cov}[\alpha_t, \alpha_{t-1} | X_{1:T}, \theta^{(k)}]$$

These are computed by **Kalman smoother**.

### Kalman Smoother Recursions

After running the Kalman filter forward, run backward smoothing:

**Initialize** at $t = T$:
$$\hat{\alpha}_{T|T} = \text{from filter}$$
$$P_{T|T} = \text{from filter}$$

**For** $t = T-1, ..., 1$ (backward):
$$J_t = P_{t|t} T' P_{t+1|t}^{-1}$$
$$\hat{\alpha}_{t|T} = \hat{\alpha}_{t|t} + J_t(\hat{\alpha}_{t+1|T} - \hat{\alpha}_{t+1|t})$$
$$P_{t|T} = P_{t|t} + J_t(P_{t+1|T} - P_{t+1|t})J_t'$$

**Lag-one covariances:**
$$P_{t+1,t|T} = J_t P_{t+1|T}$$

### M-Step: Parameter Updates

Given sufficient statistics, update parameters to maximize $Q$:

#### 1. Loading Matrix $Z$

$$Z^{(k+1)} = \left(\sum_{t=1}^T X_t \hat{\alpha}_{t|T}^{(k)'}\right) \left(\sum_{t=1}^T [\hat{\alpha}_{t|T}^{(k)}\hat{\alpha}_{t|T}^{(k)'} + P_{t|T}^{(k)}]\right)^{-1}$$

With identification constraints (e.g., lower triangular for first $r$ rows):
- Update only free elements
- Keep constrained elements fixed

#### 2. Transition Matrix $T$

$$T^{(k+1)} = \left(\sum_{t=2}^T [\hat{\alpha}_{t|T}^{(k)}\hat{\alpha}_{t-1|T}^{(k)'} + P_{t,t-1|T}^{(k)}]\right) \left(\sum_{t=2}^T [\hat{\alpha}_{t-1|T}^{(k)}\hat{\alpha}_{t-1|T}^{(k)'} + P_{t-1|T}^{(k)}]\right)^{-1}$$

#### 3. State Innovation Covariance $Q$

Define:
$$S_{11}^{(k)} = \sum_{t=2}^T [\hat{\alpha}_{t|T}^{(k)}\hat{\alpha}_{t|T}^{(k)'} + P_{t|T}^{(k)}]$$
$$S_{10}^{(k)} = \sum_{t=2}^T [\hat{\alpha}_{t|T}^{(k)}\hat{\alpha}_{t-1|T}^{(k)'} + P_{t,t-1|T}^{(k)}]$$
$$S_{00}^{(k)} = \sum_{t=2}^T [\hat{\alpha}_{t-1|T}^{(k)}\hat{\alpha}_{t-1|T}^{(k)'} + P_{t-1|T}^{(k)}]$$

Then:
$$RQ^{(k+1)}R' = \frac{1}{T-1}\left[S_{11}^{(k)} - T^{(k+1)}S_{10}^{(k)'} - S_{10}^{(k)}T^{(k+1)'} + T^{(k+1)}S_{00}^{(k)}T^{(k+1)'}\right]$$

If $R = [I_r, 0]'$, extract $Q$ from top-left $r \times r$ block.

Often **constrained** to $Q = I_r$ for identification.

#### 4. Measurement Error Covariance $H$

$$H^{(k+1)} = \frac{1}{T}\sum_{t=1}^T [X_t X_t' - X_t \hat{\alpha}_{t|T}^{(k)'}Z^{(k+1)'} - Z^{(k+1)}\hat{\alpha}_{t|T}^{(k)}X_t' + Z^{(k+1)}(\hat{\alpha}_{t|T}^{(k)}\hat{\alpha}_{t|T}^{(k)'} + P_{t|T}^{(k)})Z^{(k+1)'}]$$

Often **constrained** to diagonal: $H^{(k+1)} = \text{diag}(h_1, ..., h_N)$.

### Convergence Criteria

**Relative likelihood change:**
$$\frac{|\log L(\theta^{(k+1)}) - \log L(\theta^{(k)})|}{\max(1, |\log L(\theta^{(k)})|)} < \epsilon$$

Typical: $\epsilon = 10^{-6}$

**Parameter change (alternative):**
$$\|\theta^{(k+1)} - \theta^{(k)}\| < \delta$$

**Maximum iterations:** Stop if $k > k_{\max}$ (e.g., 500) to prevent infinite loops.

---

## Code Implementation

### Complete EM Algorithm


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">emdynamicfactormodel.py</span>
</div>

```python
import numpy as np
from scipy.linalg import solve, cholesky, lstsq

class EMDynamicFactorModel:
    """
    EM algorithm for Dynamic Factor Model estimation.

    Model:
        X_t = Lambda * F_t + e_t,  e_t ~ N(0, Sigma_e)
        F_t = Phi_1 * F_{t-1} + ... + Phi_p * F_{t-p} + eta_t,  eta_t ~ N(0, Q)

    State-space form:
        X_t = Z * alpha_t + eps_t
        alpha_t = T * alpha_{t-1} + R * eta_t

    where alpha_t = [F_t', F_{t-1}', ..., F_{t-p+1}']'
    """

    def __init__(self, n_factors, n_lags=1):
        self.r = n_factors
        self.p = n_lags
        self.rp = n_factors * n_lags

    def initialize_from_pca(self, X):
        """Initialize parameters using PCA."""
        from sklearn.decomposition import PCA
        from statsmodels.tsa.api import VAR

        T, N = X.shape

        # PCA for loadings and factors
        pca = PCA(n_components=self.r)
        F_init = pca.fit_transform(X)
        Lambda_init = pca.components_.T  # N x r

        # Ensure lower triangular identification
        Lambda_init = self._enforce_identification(Lambda_init)

        # VAR for factor dynamics
        var_model = VAR(F_init)
        var_result = var_model.fit(self.p, trend='nc')

        # Extract Phi matrices
        Phi_mats = []
        for lag in range(1, self.p + 1):
            Phi_mats.append(var_result.params[(lag-1)*self.r:lag*self.r, :].T)

        # Residual covariance
        resid_factors = var_result.resid
        Q_init = np.cov(resid_factors, rowvar=False)

        # Measurement error variance
        resid_X = X - F_init @ Lambda_init.T
        Sigma_e_init = np.var(resid_X, axis=0)

        # Construct state-space matrices
        self.N = N
        self.Z = np.hstack([Lambda_init] + [np.zeros((N, self.r))] * (self.p - 1))

        self.T_mat = np.zeros((self.rp, self.rp))
        self.T_mat[:self.r, :] = np.hstack(Phi_mats)
        if self.p > 1:
            self.T_mat[self.r:, :self.rp - self.r] = np.eye(self.rp - self.r)

        self.R = np.zeros((self.rp, self.r))
        self.R[:self.r, :] = np.eye(self.r)

        self.Q = Q_init
        self.H = np.diag(Sigma_e_init)

        return self

    def _enforce_identification(self, Lambda):
        """Enforce lower triangular structure on first r rows."""
        r = self.r
        L = Lambda.copy()

        # First r x r block should be lower triangular
        for i in range(r):
            for j in range(i + 1, r):
                L[i, j] = 0

            # Positive diagonal
            if L[i, i] < 0:
                L[:, i] *= -1

        return L

    def kalman_filter(self, X):
        """Run Kalman filter forward pass."""
        T, N = X.shape
        rp = self.rp

        # Storage
        alpha_pred = np.zeros((T + 1, rp))
        alpha_filt = np.zeros((T, rp))
        P_pred = np.zeros((T + 1, rp, rp))
        P_filt = np.zeros((T, rp, rp))
        v = np.zeros((T, N))
        F_mat = np.zeros((T, N, N))
        K = np.zeros((T, rp, N))

        # Initialize at unconditional mean and variance
        alpha_pred[0] = np.zeros(rp)

        # Solve for P_0: P = T P T' + R Q R'
        I_rp2 = np.eye(rp**2)
        T_kron = np.kron(self.T_mat, self.T_mat)
        vec_RQR = (self.R @ self.Q @ self.R.T).ravel()

        try:
            vec_P0 = solve(I_rp2 - T_kron, vec_RQR)
            P_pred[0] = vec_P0.reshape(rp, rp)
        except:
            # If near unit root, use large variance
            P_pred[0] = np.eye(rp) * 1e6

        P_pred[0] = (P_pred[0] + P_pred[0].T) / 2

        loglik = 0.0

        for t in range(T):
            # Prediction error
            v[t] = X[t] - self.Z @ alpha_pred[t]

            # Prediction error variance
            F_mat[t] = self.Z @ P_pred[t] @ self.Z.T + self.H
            F_mat[t] = (F_mat[t] + F_mat[t].T) / 2

            # Kalman gain
            try:
                K[t] = P_pred[t] @ self.Z.T @ solve(F_mat[t], np.eye(N))
            except:
                K[t] = np.zeros((rp, N))

            # Update
            alpha_filt[t] = alpha_pred[t] + K[t] @ v[t]
            P_filt[t] = P_pred[t] - K[t] @ F_mat[t] @ K[t].T
            P_filt[t] = (P_filt[t] + P_filt[t].T) / 2

            # Predict next
            if t < T - 1:
                alpha_pred[t + 1] = self.T_mat @ alpha_filt[t]
                P_pred[t + 1] = self.T_mat @ P_filt[t] @ self.T_mat.T + self.R @ self.Q @ self.R.T
                P_pred[t + 1] = (P_pred[t + 1] + P_pred[t + 1].T) / 2

            # Log-likelihood contribution
            try:
                L = cholesky(F_mat[t], lower=True)
                log_det = 2 * np.sum(np.log(np.diag(L)))
                w = solve(L, v[t], lower=True)
                quad_form = w @ w
                loglik += -0.5 * (log_det + quad_form)
            except:
                pass

        loglik += -0.5 * T * N * np.log(2 * np.pi)

        return alpha_pred, alpha_filt, P_pred, P_filt, v, F_mat, K, loglik

    def kalman_smoother(self, alpha_pred, alpha_filt, P_pred, P_filt):
        """Run Kalman smoother backward pass."""
        T = alpha_filt.shape[0]
        rp = self.rp

        alpha_smooth = np.zeros((T, rp))
        P_smooth = np.zeros((T, rp, rp))
        P_smooth_lag = np.zeros((T, rp, rp))

        # Initialize at T
        alpha_smooth[-1] = alpha_filt[-1]
        P_smooth[-1] = P_filt[-1]

        # Backward recursion
        for t in range(T - 2, -1, -1):
            try:
                J = P_filt[t] @ self.T_mat.T @ solve(P_pred[t + 1], np.eye(rp))
            except:
                J = np.zeros((rp, rp))

            alpha_smooth[t] = alpha_filt[t] + J @ (alpha_smooth[t + 1] - alpha_pred[t + 1])
            P_smooth[t] = P_filt[t] + J @ (P_smooth[t + 1] - P_pred[t + 1]) @ J.T
            P_smooth[t] = (P_smooth[t] + P_smooth[t].T) / 2

            # Lag-one covariance
            P_smooth_lag[t + 1] = J @ P_smooth[t + 1]

        return alpha_smooth, P_smooth, P_smooth_lag

    def m_step(self, X, alpha_smooth, P_smooth, P_smooth_lag):
        """M-step: update parameters."""
        T, N = X.shape

        # Sufficient statistics
        S_11 = np.sum([np.outer(alpha_smooth[t], alpha_smooth[t]) + P_smooth[t]
                       for t in range(T)], axis=0)
        S_10 = np.sum([np.outer(alpha_smooth[t], alpha_smooth[t - 1]) + P_smooth_lag[t]
                       for t in range(1, T)], axis=0)
        S_00 = np.sum([np.outer(alpha_smooth[t], alpha_smooth[t]) + P_smooth[t]
                       for t in range(T - 1)], axis=0)

        X_alpha = np.sum([np.outer(X[t], alpha_smooth[t]) for t in range(T)], axis=0)

        # Update Z (loadings)
        Z_new = X_alpha @ solve(S_11, np.eye(self.rp))
        Lambda_new = Z_new[:, :self.r]

        # Enforce identification
        Lambda_new = self._enforce_identification(Lambda_new)
        Z_new = np.hstack([Lambda_new] + [np.zeros((N, self.r))] * (self.p - 1))

        # Update T (transition)
        T_new = S_10.T @ solve(S_00, np.eye(self.rp))

        # Update Q (factor innovation variance)
        S_11_trunc = S_11[:self.r, :self.r] - S_10[:self.r, :self.r].T @ solve(S_00, S_10.T)[:, :self.r]
        Q_new = S_11_trunc / (T - 1)
        Q_new = (Q_new + Q_new.T) / 2

        # Often constrain to identity for identification
        # Q_new = np.eye(self.r)

        # Update H (measurement error variance - diagonal)
        H_diag = np.zeros(N)
        for i in range(N):
            resid_var = np.sum([X[t, i]**2 - 2 * X[t, i] * Z_new[i] @ alpha_smooth[t] +
                                (Z_new[i] @ (np.outer(alpha_smooth[t], alpha_smooth[t]) +
                                             P_smooth[t]) @ Z_new[i])
                                for t in range(T)])
            H_diag[i] = resid_var / T

        H_new = np.diag(np.maximum(H_diag, 1e-6))  # Prevent negative variance

        # Update parameters
        self.Z = Z_new
        self.T_mat = T_new
        self.Q = Q_new
        self.H = H_new

    def fit(self, X, max_iter=100, tol=1e-6, verbose=True):
        """
        Fit model via EM algorithm.

        Parameters
        ----------
        X : ndarray (T, N)
            Observed data (should be demeaned)
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance for log-likelihood
        verbose : bool
            Print progress

        Returns
        -------
        self : fitted model
        """
        # Initialize
        self.initialize_from_pca(X)

        loglik_hist = []

        for iteration in range(max_iter):
            # E-step
            alpha_pred, alpha_filt, P_pred, P_filt, v, F_mat, K, loglik = self.kalman_filter(X)
            alpha_smooth, P_smooth, P_smooth_lag = self.kalman_smoother(
                alpha_pred, alpha_filt, P_pred, P_filt)

            loglik_hist.append(loglik)

            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: Log-likelihood = {loglik:.2f}")

            # Check convergence
            if iteration > 0:
                ll_change = abs(loglik_hist[-1] - loglik_hist[-2])
                rel_change = ll_change / max(1, abs(loglik_hist[-2]))

                if rel_change < tol:
                    if verbose:
                        print(f"Converged at iteration {iteration}")
                    break

            # M-step
            self.m_step(X, alpha_smooth, P_smooth, P_smooth_lag)

        self.loglik_ = loglik_hist[-1]
        self.loglik_hist_ = loglik_hist
        self.n_iter_ = iteration + 1

        # Extract parameters
        self.Lambda_ = self.Z[:, :self.r]
        self.Phi_ = self.T_mat[:self.r, :self.r * self.p]
        self.sigma_e_ = np.sqrt(np.diag(self.H))

        # Compute information criteria
        n_params = self.r * self.N + self.r**2 * self.p + self.r * (self.r + 1) // 2 + self.N
        T = X.shape[0]
        self.aic_ = -2 * self.loglik_ + 2 * n_params
        self.bic_ = -2 * self.loglik_ + n_params * np.log(T)

        return self

    def smooth_factors(self, X):
        """Extract smoothed factors given data."""
        alpha_pred, alpha_filt, P_pred, P_filt, v, F_mat, K, loglik = self.kalman_filter(X)
        alpha_smooth, P_smooth, P_smooth_lag = self.kalman_smoother(
            alpha_pred, alpha_filt, P_pred, P_filt)

        # Extract factors (first r components of state)
        F_smooth = alpha_smooth[:, :self.r]

        return F_smooth, alpha_smooth, P_smooth


# Example usage
if __name__ == '__main__':
    np.random.seed(42)

    # Simulate DFM
    T, N, r, p = 200, 10, 2, 1

    Lambda_true = np.random.randn(N, r)
    Phi_true = np.array([[0.8, 0.1], [0.1, 0.7]])
    sigma_e_true = np.ones(N) * 0.5

    F = np.zeros((T, r))
    for t in range(1, T):
        F[t] = Phi_true @ F[t - 1] + np.random.randn(r)

    X = F @ Lambda_true.T + np.random.randn(T, N) * sigma_e_true
    X = X - X.mean(axis=0)

    # Fit model
    model = EMDynamicFactorModel(n_factors=2, n_lags=1)
    model.fit(X, max_iter=100, verbose=True)

    print("\n=== Results ===")
    print(f"Log-likelihood: {model.loglik_:.2f}")
    print(f"AIC: {model.aic_:.2f}, BIC: {model.bic_:.2f}")
    print(f"Converged in {model.n_iter_} iterations")

    print("\nEstimated Phi:")
    print(model.Phi_[:, :r].round(3))
    print("\nTrue Phi:")
    print(Phi_true.round(3))

    # Extract factors
    F_hat, _, _ = model.smooth_factors(X)

    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    for i in range(r):
        axes[i].plot(F[:, i], label='True', alpha=0.7)
        axes[i].plot(F_hat[:, i], label='Estimated', alpha=0.7)
        axes[i].set_title(f'Factor {i+1}')
        axes[i].legend()
    plt.tight_layout()
    plt.show()
```

</div>
</div>

---

## Common Pitfalls

### 1. Slow Convergence

**Problem:** EM can take hundreds of iterations, especially with weak identification.

**Solutions:**
- Initialize from PCA (usually close to optimum)
- Use accelerated EM variants (ECME, SQUAREM)
- Switch to quasi-Newton after EM converges roughly

### 2. Boundary Issues

**Problem:** Variances can become negative or near-zero.

**Solutions:**
- Constrain variances: $\sigma^2 \geq \epsilon$ with $\epsilon = 10^{-6}$
- Use $\log(\sigma^2)$ parameterization in M-step
- Check positive definiteness of covariance matrices

### 3. Identification Violations

**Problem:** EM may not respect identification constraints.

**Solutions:**
- **Enforce constraints in M-step:** Zero out upper triangle, rescale factors
- **Post-process estimates:** Rotate to satisfy constraints
- **Penalty approach:** Add term to $Q$ that penalizes deviation from constraints

### 4. Local Maxima

**Problem:** EM finds local, not global, maximum.

**Solutions:**
- Run from multiple starting values
- PCA initialization usually near global maximum for factor models
- Compare likelihood across different initializations

---

## Connections

- **Builds on:** Kalman smoother (Module 2), MLE via Kalman filter (previous guide)
- **Leads to:** Bayesian estimation (next guide uses similar complete-data approach)
- **Related to:** Missing data imputation (EM treats factors as missing)

---

## Practice Problems

### Conceptual

1. Why does EM guarantee non-decreasing likelihood at each iteration?

2. Explain the role of $P_{t|T}$ in the M-step. Why not just use $\hat{\alpha}_{t|T}$?

3. How would you modify EM to handle missing observations in $X_t$?

### Mathematical

4. Derive the M-step update for $\Lambda$ from first principles (maximize $Q$ with respect to $\Lambda$).

5. Show that with known factors ($P_{t|T} = 0$), the M-step reduces to OLS.

6. Prove that EM converges (at least to a stationary point) under regularity conditions.

### Implementation

7. Implement EM with diagonal $\Phi$ (independent factor AR processes).

8. Add constraints: force some loadings to be zero (sparse loadings).

9. Compare EM convergence speed with different initializations: random, PCA, true values.

---

<div class="callout-insight">

**Insight:** Understanding em algorithm for dynamic factor models is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## Further Reading

- **Dempster, Laird & Rubin (1977).** "Maximum Likelihood from Incomplete Data via the EM Algorithm." *Journal of the Royal Statistical Society: Series B* 39(1), 1-38.
  - Original EM paper with complete theory

- **Shumway & Stoffer (1982).** "An Approach to Time Series Smoothing and Forecasting Using the EM Algorithm." *Journal of Time Series Analysis* 3(4), 253-264.
  - EM for state-space models specifically

- **Durbin & Koopman (2012).** *Time Series Analysis by State Space Methods*, 2nd ed., Chapter 7.
  - Detailed EM algorithm for state-space models

- **Watson & Engle (1983).** "Alternative Algorithms for the Estimation of Dynamic Factor, Mimic and Varying Coefficient Regression Models." *Journal of Econometrics* 23(3), 385-400.
  - Applications to factor models

- **Banbura & Modugno (2014).** "Maximum Likelihood Estimation of Factor Models on Datasets with Arbitrary Pattern of Missing Data." *Journal of Applied Econometrics* 29(1), 133-160.
  - EM with realistic missing data patterns

---

## Conceptual Practice Questions

1. In your own words, explain the difference between common factors and idiosyncratic components.

2. Why do factor models require identification restrictions? Give a concrete example.

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.


---



## Cross-References

<a class="link-card" href="./02_em_algorithm_dfm_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_em_algorithm_implementation.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_mle_via_kalman.md">
  <div class="link-card-title">01 Mle Via Kalman</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_bayesian_dfm.md">
  <div class="link-card-title">03 Bayesian Dfm</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

