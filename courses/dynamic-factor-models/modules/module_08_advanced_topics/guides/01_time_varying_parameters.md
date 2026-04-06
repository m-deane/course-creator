# Time-Varying Parameters in Dynamic Factor Models

> **Reading time:** ~16 min | **Module:** Module 8: Advanced Topics | **Prerequisites:** Modules 0-7

<div class="callout-key">

**Key Concept Summary:** Economic structure evolves—the relationship between factors and observables changes with policy regimes, technological innovation, and structural transformations. Time-varying parameter dynamic factor models (TVP-DFM) allow factor loadings and dynamics to drift smoothly or shift discretely, captu...

</div>

## In Brief

Economic structure evolves—the relationship between factors and observables changes with policy regimes, technological innovation, and structural transformations. Time-varying parameter dynamic factor models (TVP-DFM) allow factor loadings and dynamics to drift smoothly or shift discretely, capturing phenomena like the Great Moderation and evolving industry structures.

<div class="callout-insight">

**Insight:** Static parameters assume stable economic relationships indefinitely. Reality shows the factor structure governing GDP, inflation, and financial markets has changed substantially over decades. TVP-DFMs nest constant-parameter models as special cases while providing flexibility to detect and model structural evolution—critical for robust long-horizon forecasting and policy analysis.

</div>
---

## 1. Motivation for Time-Varying Parameters

### Intuitive Explanation

Consider forecasting inflation using a factor model. In the 1970s, oil prices had enormous impact on inflation (high loading). By the 2000s, energy's direct effect diminished due to efficiency improvements and monetary policy changes (lower loading). A constant-parameter model averages these periods, performing poorly in both. A TVP model adapts loadings to each era.

**Economic Examples of Parameter Change:**

<div class="flow">
<div class="flow-step mint">1. Great Moderation (19...</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">2. Evolving Industry St...</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">3. Monetary Policy Regi...</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. Globalization:</div>

</div>


1. **Great Moderation (1984-2007):** Macroeconomic volatility declined—lower variance parameters
2. **Evolving Industry Structure:** Tech sector's loading on traditional factors changed post-internet
3. **Monetary Policy Regime Shifts:** Central bank reaction functions evolve, changing factor dynamics
4. **Globalization:** Increased international co-movement altered cross-country factor loadings

### Formal Definition

A **Time-Varying Parameter Dynamic Factor Model** allows parameters $\theta_t$ (loadings $\Lambda_t$, dynamics $\Phi_t$, or variances $\Sigma_t$) to evolve over time rather than remaining fixed.

**General TVP-DFM:**
$$X_t = \Lambda_t F_t + e_t$$
$$F_t = \Phi_t F_{t-1} + \eta_t$$

where $\Lambda_t$ and/or $\Phi_t$ follow specified evolution processes.

### Mathematical Framework

Four approaches to time-variation, ordered by complexity:

**1. Discrete Regime Switching:**
$$\theta_t = \theta_{s_t}, \quad s_t \in \{1, 2, \ldots, K\}$$

Parameters jump between $K$ discrete states governed by Markov chain.

**2. Random Walk Evolution:**
$$\theta_t = \theta_{t-1} + \nu_t, \quad \nu_t \sim N(0, Q)$$

Smooth drifting parameters with permanent innovations.

**3. Stationary AR(1) Process:**
$$\theta_t = (1-\rho)\bar{\theta} + \rho \theta_{t-1} + \nu_t, \quad |\rho| < 1$$

Mean-reverting parameters fluctuating around long-run mean $\bar{\theta}$.

**4. Score-Driven Dynamics:**
$$\theta_t = \omega + A \cdot s_{t-1} + B \cdot \theta_{t-1}$$

where $s_t = S_t \nabla_\theta \log p(X_t | F_t, \theta_t)$ is scaled score, $S_t$ is scaling matrix.

<div class="callout-insight">

**Insight:** Score-driven models are adaptive: parameters update based on how surprising recent data is.

</div>

### Choosing the Right Approach

| Approach | Parameters Evolve... | Best When | Computational Cost |
|---|---|---|---|
| Discrete Regime Switching | In sudden jumps between $K$ states | Clear structural breaks (GFC, policy change) | Medium (EM or MCMC) |
| Random Walk | Smoothly, permanently | Slow structural drift, no mean reversion | Low (Kalman filter) |
| Stationary AR(1) | Smoothly, around a long-run mean | Cyclical parameter variation | Low (Kalman filter) |
| Score-Driven | Based on prediction surprise | Unknown dynamics; robust to misspecification | Medium |

### Code Implementation


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">tvpdynamicfactormodel.py</span>

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from scipy.stats import norm

class TVPDynamicFactorModel:
    """
    Time-Varying Parameter Dynamic Factor Model with random walk loadings.

    Model:
        X_t = Lambda_t * F_t + e_t
        F_t = Phi * F_{t-1} + eta_t
        Lambda_t = Lambda_{t-1} + nu_t

    Estimates factors and time-varying loadings via Kalman filtering.
    """

    def __init__(self, n_factors, tvp_type='loading', Q_scale=0.01):
        """
        Parameters
        ----------
        n_factors : int
            Number of latent factors
        tvp_type : str
            Which parameters vary: 'loading', 'dynamics', or 'both'
        Q_scale : float
            Variance of parameter innovations (controls smoothness)
        """
        self.n_factors = n_factors
        self.tvp_type = tvp_type
        self.Q_scale = Q_scale
        self.Lambda_t = None  # Time-varying loadings
        self.Phi = None       # Factor dynamics (may be time-varying)
        self.F_t = None       # Latent factors

    def fit(self, X, n_iter=10, verbose=False):
        """
        Estimate TVP-DFM using iterative Kalman filtering.

        Parameters
        ----------
        X : ndarray, shape (T, N)
            Observed data
        n_iter : int
            Number of EM iterations
        verbose : bool
            Print progress

        Returns
        -------
        self
        """
        T, N = X.shape
        r = self.n_factors

        # Initialize with static PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=r)
        F_init = pca.fit_transform(X)
        Lambda_init = pca.components_.T

        # Initialize storage for time-varying loadings
        self.Lambda_t = np.tile(Lambda_init[:, :, np.newaxis], (1, 1, T))  # (N, r, T)
        self.F_t = F_init.copy()

        # Estimate factor dynamics
        self.Phi = self._estimate_var_dynamics(self.F_t)

        # Main EM loop with TVP
        for iteration in range(n_iter):
            if verbose:
                print(f"Iteration {iteration + 1}/{n_iter}")

            # E-step: Filter factors given current Lambda_t
            self.F_t = self._kalman_filter_factors(X)

            # M-step: Update time-varying loadings
            if self.tvp_type in ['loading', 'both']:
                self._update_tvp_loadings(X)

            # M-step: Update dynamics
            if self.tvp_type in ['dynamics', 'both']:
                self.Phi = self._estimate_var_dynamics(self.F_t)
            else:
                self.Phi = self._estimate_var_dynamics(self.F_t)

        return self

    def _kalman_filter_factors(self, X):
        """
        Kalman filter for factors with time-varying loadings.

        State equation: F_t = Phi * F_{t-1} + eta_t
        Obs equation: X_t = Lambda_t * F_t + e_t
        """
        T, N = X.shape
        r = self.n_factors

        # Initialize
        F_filt = np.zeros((T, r))
        F_pred = np.zeros((T, r))
        P_filt = np.zeros((T, r, r))
        P_pred = np.zeros((T, r, r))

        # Initial conditions
        F_filt[0] = np.linalg.lstsq(self.Lambda_t[:, :, 0], X[0], rcond=None)[0]
        P_filt[0] = np.eye(r)

        Q = np.eye(r) * 0.1  # Factor innovation variance

        # Forward pass
        for t in range(1, T):
            # Prediction
            F_pred[t] = self.Phi @ F_filt[t-1]
            P_pred[t] = self.Phi @ P_filt[t-1] @ self.Phi.T + Q

            # Update with time-varying Lambda_t
            Lambda_t = self.Lambda_t[:, :, t]
            R = np.eye(N) * 0.5  # Measurement error variance

            # Innovation
            v = X[t] - Lambda_t @ F_pred[t]
            S = Lambda_t @ P_pred[t] @ Lambda_t.T + R
            K = P_pred[t] @ Lambda_t.T @ np.linalg.inv(S)

            # Update
            F_filt[t] = F_pred[t] + K @ v
            P_filt[t] = (np.eye(r) - K @ Lambda_t) @ P_pred[t]

        return F_filt

    def _update_tvp_loadings(self, X):
        """
        Update time-varying loadings using Kalman smoothing.

        For each variable i, treat lambda_i as state:
            lambda_{i,t} = lambda_{i,t-1} + nu_t
        """
        T, N = X.shape
        r = self.n_factors

        Q_lambda = np.eye(r) * self.Q_scale  # Loading innovation variance

        for i in range(N):
            # For variable i, filter its loading vector
            lambda_i_filt = np.zeros((T, r))
            P_lambda = np.zeros((T, r, r))

            # Initial condition (from PCA)
            lambda_i_filt[0] = self.Lambda_t[i, :, 0]
            P_lambda[0] = np.eye(r) * 0.1

            # Kalman filter for lambda_i
            for t in range(1, T):
                # Prediction (random walk)
                lambda_i_pred = lambda_i_filt[t-1]
                P_pred = P_lambda[t-1] + Q_lambda

                # Update using observation X_{i,t} and factor F_t
                F_t = self.F_t[t]
                y_t = X[t, i]

                # Innovation: y_t = lambda_i' * F_t + e_t
                v = y_t - lambda_i_pred @ F_t
                S = F_t.T @ P_pred @ F_t + 0.5  # R = 0.5
                K = (P_pred @ F_t) / S

                lambda_i_filt[t] = lambda_i_pred + K * v
                P_lambda[t] = P_pred - np.outer(K, K) * S

            # Store updated loadings
            self.Lambda_t[i, :, :] = lambda_i_filt.T

    def _estimate_var_dynamics(self, F):
        """
        Estimate VAR(1) dynamics for factors: F_t = Phi * F_{t-1} + eta_t
        """
        T, r = F.shape
        Y = F[1:, :]
        X = F[:-1, :]

        Phi = lstsq(X, Y, rcond=None)[0].T
        return Phi

    def get_loadings(self, t):
        """
        Get loadings at specific time t.

        Parameters
        ----------
        t : int
            Time index

        Returns
        -------
        Lambda_t : ndarray, shape (N, r)
            Loadings at time t
        """
        return self.Lambda_t[:, :, t]

    def get_all_loadings(self):
        """
        Get full array of time-varying loadings.

        Returns
        -------
        Lambda_t : ndarray, shape (N, r, T)
        """
        return self.Lambda_t

    def plot_loading_evolution(self, variable_idx, factor_idx, dates=None):
        """
        Plot evolution of a specific loading over time.

        Parameters
        ----------
        variable_idx : int
            Which variable's loading to plot
        factor_idx : int
            Which factor's loading to plot
        dates : array-like, optional
            Date labels for x-axis
        """
        T = self.Lambda_t.shape[2]
        loading_path = self.Lambda_t[variable_idx, factor_idx, :]

        if dates is None:
            dates = np.arange(T)

        plt.figure(figsize=(10, 4))
        plt.plot(dates, loading_path, linewidth=2)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.xlabel('Time')
        plt.ylabel(f'Loading')
        plt.title(f'Time-Varying Loading: Variable {variable_idx}, Factor {factor_idx}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()


# Example: Simulate data with structural break in loadings
np.random.seed(42)
T, N, r = 200, 10, 2

# Generate factors
F_true = np.zeros((T, r))
Phi_true = np.array([[0.9, 0.1], [0.0, 0.8]])
F_true[0] = np.random.randn(r)
for t in range(1, T):
    F_true[t] = Phi_true @ F_true[t-1] + np.random.randn(r) * 0.3

# Time-varying loadings with break at T/2
Lambda_pre = np.random.randn(N, r)
Lambda_post = Lambda_pre + np.random.randn(N, r) * 0.5

X_data = np.zeros((T, N))
for t in range(T):
    if t < T // 2:
        Lambda_t = Lambda_pre
    else:
        # Smooth transition
        weight = (t - T // 2) / (T // 2)
        Lambda_t = (1 - weight) * Lambda_pre + weight * Lambda_post

    X_data[t] = Lambda_t @ F_true[t] + np.random.randn(N) * 0.3

# Fit TVP-DFM
print("Fitting TVP-DFM with time-varying loadings...")
model = TVPDynamicFactorModel(n_factors=2, tvp_type='loading', Q_scale=0.01)
model.fit(X_data, n_iter=5, verbose=True)

print(f"\nEstimated factor dynamics:")
print(model.Phi)

# Plot loading evolution for first variable, first factor
fig = model.plot_loading_evolution(0, 0)
plt.show()

# Compare loadings at different time points
print(f"\nLoading for variable 0, factor 0:")
print(f"  At t=50 (pre-break): {model.get_loadings(50)[0, 0]:.3f}")
print(f"  At t=150 (post-break): {model.get_loadings(150)[0, 0]:.3f}")
```

</div>
</div>

---

## 2. Estimation Approaches for TVP-DFM

### Rolling Window Estimation

**Simplest Approach:** Re-estimate model on moving window.

**Algorithm:**
1. Choose window width $W$ (e.g., 40 quarters)
2. For $t = W, W+1, \ldots, T$:
   - Estimate static DFM on data $\{X_{t-W+1}, \ldots, X_t\}$
   - Save parameters $\hat{\theta}_t$

**Advantages:**
- Conceptually simple
- Easy to implement with existing code
- No distributional assumptions

**Disadvantages:**
- Discrete jumps as observations enter/exit window
- Bandwidth selection crucial but arbitrary
- Inefficient (discards older data completely)
- Poor real-time performance at window start

### State-Space Formulation with TVP

**Key Idea:** Augment state vector to include time-varying parameters.

**Model:**
$$X_t = H_t \alpha_t + e_t$$
$$\alpha_t = T_t \alpha_{t-1} + \eta_t$$

where augmented state $\alpha_t = [F_t', \text{vec}(\Lambda_t)']'$ includes both factors and loadings.

**Example: TVP Loadings Only**

State equation (factors + loadings):
$$\begin{bmatrix} F_t \\ \text{vec}(\Lambda_t) \end{bmatrix} = \begin{bmatrix} \Phi & 0 \\ 0 & I \end{bmatrix} \begin{bmatrix} F_{t-1} \\ \text{vec}(\Lambda_{t-1}) \end{bmatrix} + \begin{bmatrix} \eta_t \\ \nu_t \end{bmatrix}$$

Observation equation:
$$X_t = \Lambda_t F_t + e_t$$

This is nonlinear in states! Use extended Kalman filter or particle filter.

**Challenge:** High dimensionality ($N \times r$ loading parameters). Requires:
- Strong regularization (small $Q$ for loadings)
- Shrinkage priors in Bayesian context
- Sparsity constraints

### Score-Driven (GAS) Models

**Generalized Autoregressive Score Framework:**

Parameters evolve based on scaled score of predictive density:
$$\theta_{t+1} = \omega + A \cdot s_t + B \cdot \theta_t$$

where scaled score is:
$$s_t = S_t \nabla_{\theta} \log p(X_t | F_t, \theta_t)$$

Scaling matrix: $S_t = \mathcal{I}_t^{-1/2}$ (inverse square root of information).

**Intuition:**
- If data surprises model (large score), parameters adjust more
- If data fits well (small score), parameters stay stable
- Adaptive to outliers and structural breaks

**Example: Score-Driven Loading**

For Gaussian model, score for $\lambda_i$ (loading of variable $i$):
$$s_{i,t} = \frac{1}{\sigma_i^2} (X_{it} - \lambda_i' F_t) F_t$$

Update:
$$\lambda_{i,t+1} = (1 - b) \bar{\lambda}_i + a \cdot s_{i,t} + b \cdot \lambda_{i,t}$$

**Advantages:**
- Observation-driven (no latent parameter states)
- Theoretically justified
- Robust to model misspecification
- Faster than Kalman filtering for TVP

### Change-Point Detection

**Alternative:** Detect discrete breaks rather than smooth evolution.

**Bai-Perron Test for Factor Models:**

Test null of no breaks against alternative of $m$ breaks at unknown dates.

**Algorithm:**
1. For each candidate break date $\tau$:
   - Estimate pre-break model: $\hat{\theta}_1$ using $t = 1, \ldots, \tau$
   - Estimate post-break model: $\hat{\theta}_2$ using $t = \tau+1, \ldots, T$
   - Compute SSR (sum of squared residuals)

2. Choose break maximizing fit improvement:
$$\hat{\tau} = \arg\min_{\tau} \left[ SSR_1(\tau) + SSR_2(\tau) \right]$$

3. Test if $SSR(\hat{\tau}) < SSR(\text{no break}) - c \cdot \log T$ (BIC criterion)

**Multiple Breaks:** Sequential testing or global optimization.

### Code Implementation: Change-Point Detection


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">detect_loading_breaks.py</span>

```python
def detect_loading_breaks(X, F, max_breaks=3, min_segment=20):
    """
    Detect structural breaks in factor loadings using Bai-Perron approach.

    Parameters
    ----------
    X : ndarray, shape (T, N)
        Observed data
    F : ndarray, shape (T, r)
        Estimated factors
    max_breaks : int
        Maximum number of breaks to detect
    min_segment : int
        Minimum observations between breaks

    Returns
    -------
    break_dates : list
        Estimated break dates
    Lambda_regimes : list of ndarray
        Loading matrices for each regime
    """
    T, N = X.shape
    r = F.shape[1]

    def estimate_loadings_segment(X_seg, F_seg):
        """Estimate loadings on segment by OLS."""
        Lambda = lstsq(F_seg, X_seg, rcond=None)[0].T
        return Lambda

    def compute_ssr(X_seg, F_seg, Lambda):
        """Compute sum of squared residuals."""
        resid = X_seg - F_seg @ Lambda.T
        return np.sum(resid ** 2)

    break_dates = []

    for n_break in range(1, max_breaks + 1):
        best_ssr = np.inf
        best_breaks = None

        # Try all possible break combinations
        from itertools import combinations
        candidate_dates = range(min_segment, T - min_segment)

        for breaks in combinations(candidate_dates, n_break):
            breaks = [0] + list(breaks) + [T]
            ssr_total = 0

            # Compute SSR for each segment
            valid = True
            for i in range(len(breaks) - 1):
                start, end = breaks[i], breaks[i+1]
                if end - start < min_segment:
                    valid = False
                    break

                X_seg = X[start:end]
                F_seg = F[start:end]
                Lambda_seg = estimate_loadings_segment(X_seg, F_seg)
                ssr_total += compute_ssr(X_seg, F_seg, Lambda_seg)

            if valid and ssr_total < best_ssr:
                best_ssr = ssr_total
                best_breaks = breaks[1:-1]  # Exclude endpoints

        # BIC test for this number of breaks
        k_params = (n_break + 1) * N * r  # Parameters in all regimes
        bic = T * np.log(best_ssr / T) + k_params * np.log(T)

        if n_break == 1:
            bic_no_break = T * np.log(
                compute_ssr(X, F, estimate_loadings_segment(X, F)) / T
            ) + N * r * np.log(T)

            if bic < bic_no_break:
                break_dates = [best_breaks[0]]
        else:
            # Compare to previous number of breaks
            # (simplified: would track BIC across iterations)
            break_dates = list(best_breaks)

    # Estimate loadings for each regime
    breaks = [0] + break_dates + [T]
    Lambda_regimes = []

    for i in range(len(breaks) - 1):
        start, end = breaks[i], breaks[i+1]
        Lambda_seg = estimate_loadings_segment(X[start:end], F[start:end])
        Lambda_regimes.append(Lambda_seg)

    return break_dates, Lambda_regimes


# Example: Detect break in simulated data

# (Using data from previous example with known break at T/2)

# First extract factors (assuming we don't know true factors)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
F_est = pca.fit_transform(X_data)

print("\nDetecting structural breaks...")
breaks, regimes = detect_loading_breaks(X_data, F_est, max_breaks=2, min_segment=40)

print(f"Detected break dates: {breaks}")
print(f"True break at: {T // 2}")
print(f"\nNumber of regimes: {len(regimes)}")

# Compare regime loadings
if len(regimes) >= 2:
    print(f"\nLoading change for variable 0, factor 0:")
    print(f"  Regime 1: {regimes[0][0, 0]:.3f}")
    print(f"  Regime 2: {regimes[1][0, 0]:.3f}")
    print(f"  Change: {regimes[1][0, 0] - regimes[0][0, 0]:.3f}")
```

</div>
</div>

---

## 3. TVP State-Space Representation

### Full State-Space TVP-DFM

**Augmented State Vector:**
$$\alpha_t = \begin{bmatrix} F_t \\ \text{vec}(\Lambda_t) \\ \text{vec}(\Phi_t) \end{bmatrix}$$

Includes factors, loadings, and dynamics (if all vary).

**State Evolution (Random Walk TVP):**
$$\alpha_t = T \alpha_{t-1} + R \eta_t$$

where:
$$T = \begin{bmatrix}
\Phi & 0 & 0 \\
0 & I_{Nr} & 0 \\
0 & 0 & I_{r^2}
\end{bmatrix}, \quad
R = \begin{bmatrix}
I_r & 0 & 0 \\
0 & I_{Nr} & 0 \\
0 & 0 & I_{r^2}
\end{bmatrix}$$

$$\eta_t \sim N(0, Q), \quad Q = \text{blockdiag}(Q_F, Q_\Lambda, Q_\Phi)$$

**Observation Equation:**
$$X_t = H(\alpha_t) + e_t$$

where $H(\alpha_t) = \Lambda_t F_t$ is **nonlinear** in augmented state.

### Extended Kalman Filter for TVP-DFM

**Linearization:** Use first-order Taylor approximation around predicted state.

**Jacobian of Observation Equation:**
$$\frac{\partial H}{\partial \alpha} = \begin{bmatrix} \frac{\partial X}{\partial F} & \frac{\partial X}{\partial \text{vec}(\Lambda)} & 0 \end{bmatrix}$$

where:
- $\frac{\partial X}{\partial F} = \Lambda_t$ (N × r matrix)
- $\frac{\partial X}{\partial \text{vec}(\Lambda)} = F_t \otimes I_N$ (N × Nr matrix)

**EKF Algorithm:**

1. **Prediction:**
   $$\alpha_{t|t-1} = T \alpha_{t-1|t-1}$$
   $$P_{t|t-1} = T P_{t-1|t-1} T' + R Q R'$$

2. **Update:**
   - Compute Jacobian $H_t$ at $\alpha_{t|t-1}$
   - Innovation: $v_t = X_t - H(\alpha_{t|t-1})$
   - Innovation covariance: $S_t = H_t P_{t|t-1} H_t' + \Sigma_e$
   - Kalman gain: $K_t = P_{t|t-1} H_t' S_t^{-1}$
   - State update: $\alpha_{t|t} = \alpha_{t|t-1} + K_t v_t$
   - Covariance update: $P_{t|t} = (I - K_t H_t) P_{t|t-1}$

### Practical Considerations

**Dimensionality Curse:**
- $r$ factors: dimension $r$
- $N$ variables, $r$ factors: $Nr$ loadings
- Factor VAR($p$): $r^2 p$ dynamics parameters
- Total state dimension: $r + Nr + r^2 p$ can be huge!

**Solutions:**
1. **Restrict TVP:** Only allow subset of parameters to vary
   - Example: Let $\Lambda_t$ vary but fix $\Phi$
2. **Shrinkage:** Use small $Q$ matrices (strong smoothing)
3. **Bayesian Priors:** Regularize with informative priors
4. **Sparsity:** Many loadings stay at zero
5. **Factor-Specific TVP:** Allow only few factors to have time-varying loadings

---

## Common Pitfalls

### 1. Over-Parameterization
- **Mistake**: Allowing all parameters to vary with large innovation variances
- **Result**: Model chases noise, poor out-of-sample forecasts
- **Fix**: Start with subset of TVP, use cross-validation to tune $Q$ scale

### 2. Identification Issues
- **Mistake**: TVP in both loadings and factors without normalization
- **Result**: Observationally equivalent transformations (rotation ambiguity)
- **Fix**: Impose identification restrictions each period (e.g., $\Lambda_t' \Lambda_t = I$)

### 3. Confusing Breaks with Smooth Evolution
- **Mistake**: Using smooth TVP when discrete regime shift occurred
- **Result**: Slow adaptation, poor forecasts near break
- **Fix**: Test for breaks first, use TVP only if breaks are rejected

### 4. Ignoring Parameter Uncertainty
- **Mistake**: Treating filtered parameters as if known with certainty
- **Result**: Underestimated forecast uncertainty
- **Fix**: Include parameter covariance $P_{t|t}$ in forecast distributions

---

## Connections

- **Builds on:** State-space representation (Module 2), Kalman filter, EM algorithm (Module 4)
- **Leads to:** Structural break testing, regime-switching models, adaptive forecasting
- **Related to:** Bayesian DFM with time-varying parameters, stochastic volatility models

---

## Practice Problems

### Conceptual

1. Why might factor loadings change over time in macroeconomic data? Give three economic reasons.

2. Compare random walk TVP with mean-reverting TVP. When would you prefer each?

3. Explain why observation equation $X_t = \Lambda_t F_t + e_t$ is nonlinear when both $\Lambda_t$ and $F_t$ are state variables.

### Implementation

4. Modify the `TVPDynamicFactorModel` class to allow user-specified $Q_\Lambda$ for each variable (heterogeneous smoothness).

5. Implement rolling window estimation and compare loading estimates to Kalman filter approach on simulated data.

6. Create a function to compute out-of-sample forecast performance comparing static vs TVP-DFM.

### Extension

7. Derive the Kalman filter update equations when $\Phi_t$ (not $\Lambda_t$) is time-varying as a random walk.

8. Research "forgetting factor" methods in recursive least squares. How do they relate to TVP models?

9. Implement a simple score-driven loading update and compare convergence speed to Kalman filter TVP.

---

## Further Reading

- **Primiceri, G.E.** (2005). "Time varying structural vector autoregressions and monetary policy." *Review of Economic Studies*, 72(3), 821-852.
  - Influential TVP-VAR paper with random walk coefficients

- **Koop, G. & Korobilis, D.** (2014). "A new index of financial conditions." *European Economic Review*, 71, 101-116.
  - TVP factor model for financial conditions index

- **Creal, D., Koopman, S.J. & Lucas, A.** (2013). "Generalized autoregressive score models with applications." *Journal of Applied Econometrics*, 28(5), 777-795.
  - Score-driven time-varying parameters

- **Bai, J., Han, X. & Shi, Y.** (2020). "Estimation and inference of change points in high-dimensional factor models." *Journal of Business & Economic Statistics*, 38(3), 629-642.
  - Modern methods for break detection in factor models

- **Eickmeier, S., Lemke, W. & Marcellino, M.** (2015). "Classical time varying factor-augmented vector auto-regressive models—Estimation, forecasting and structural analysis." *Journal of the Royal Statistical Society: Series A*, 178(3), 493-533.
  - Comprehensive treatment of TVP-FAVAR

- **Del Negro, M. & Primiceri, G.E.** (2015). "Time varying structural vector autoregressions and monetary policy: A corrigendum." *Review of Economic Studies*, 82(4), 1342-1345.
  - Important technical corrections for TVP estimation

---

## Summary

**Key Takeaways:**

1. **Time-varying parameters capture structural evolution** in factor loadings, dynamics, or variances
2. **Multiple estimation approaches** trade off complexity and flexibility: rolling windows, state-space TVP, score-driven, change-point detection
3. **State-space TVP-DFM** augments state vector with parameters, requiring extended Kalman filter for nonlinear observation equation
4. **Regularization essential** to prevent over-fitting in high-dimensional TVP models
5. **Economic insight guides specification**: which parameters vary, smooth vs discrete changes

**Next Steps:**

The next guide explores non-Gaussian factor models—relaxing normality to handle heavy tails, asymmetry, and outliers through Student-t distributions and robust estimation.

---

## Conceptual Practice Questions

1. In your own words, explain the difference between common factors and idiosyncratic components.

2. Why do factor models require identification restrictions? Give a concrete example.

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.

</div>

---

## Cross-References

<a class="link-card" href="./01_time_varying_parameters_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_time_varying_factors.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./02_non_gaussian_factors.md">
  <div class="link-card-title">02 Non Gaussian Factors</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_ml_connections.md">
  <div class="link-card-title">03 Ml Connections</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

