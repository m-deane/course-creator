# Bayesian Estimation for Dynamic Factor Models

> **Reading time:** ~13 min | **Module:** Module 4: Estimation Ml | **Prerequisites:** Modules 0-3

<div class="callout-key">

**Key Concept Summary:** Bayesian estimation treats DFM parameters as random variables with prior distributions, combining prior information with data likelihood to obtain posterior distributions. Inference proceeds via Markov Chain Monte Carlo (MCMC), typically using a Gibbs sampler that iteratively draws factors and pa...

</div>

## In Brief

Bayesian estimation treats DFM parameters as random variables with prior distributions, combining prior information with data likelihood to obtain posterior distributions. Inference proceeds via Markov Chain Monte Carlo (MCMC), typically using a Gibbs sampler that iteratively draws factors and parameters from their conditional distributions. This provides full uncertainty quantification and naturally handles identification through informative priors.

<div class="callout-insight">

**Insight:** Frequentist MLE gives point estimates. Bayesian estimation gives distributions representing uncertainty. This matters enormously for policy: instead of "the loading is 0.8," we get "the loading is between 0.6 and 1.0 with 95% probability." Priors regularize estimates (like ridge regression), prevent overfitting, and incorporate economic theory. The posterior distribution fully characterizes what we know about parameters given the data.

</div>
---

<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## Intuitive Explanation

## Formal Definition

### Bayesian Framework

**Prior distribution:**
$$p(\theta) = p(\Lambda) p(\Phi) p(\Sigma_\eta) p(\Sigma_e)$$

**Likelihood:**
$$p(X_{1:T} | \theta) = \int p(X_{1:T} | F_{1:T}, \theta) p(F_{1:T} | \theta) dF_{1:T}$$

**Posterior distribution** (Bayes' theorem):
$$p(\theta | X_{1:T}) = \frac{p(X_{1:T} | \theta) p(\theta)}{\int p(X_{1:T} | \theta') p(\theta') d\theta'}$$

$$\propto p(X_{1:T} | \theta) p(\theta)$$

**Joint posterior including factors:**
$$p(\theta, F_{1:T} | X_{1:T}) \propto p(X_{1:T} | F_{1:T}, \theta) p(F_{1:T} | \theta) p(\theta)$$

### Standard Priors for DFM

For model:
$$X_t = \Lambda F_t + e_t, \quad e_t \sim N(0, \Sigma_e)$$
$$F_t = \Phi_1 F_{t-1} + ... + \Phi_p F_{t-p} + \eta_t, \quad \eta_t \sim N(0, \Sigma_\eta)$$

**1. Loadings (column-wise):**
$$\lambda_j \sim N(\mu_{\lambda_j}, \Sigma_{\lambda_j})$$

Typical: $\mu_{\lambda_j} = 0$, $\Sigma_{\lambda_j} = \tau^2 I_N$ (diffuse)

**2. Factor dynamics (row-wise):**
$$\phi_i \sim N(\mu_{\phi_i}, \Sigma_{\phi_i})$$

where $\phi_i$ is $i$-th row of $[\Phi_1, ..., \Phi_p]$

Typical: $\mu_{\phi_i} = [0.5, 0, ..., 0]$ (AR(1)-ish), $\Sigma_{\phi_i} = 0.5^2 I_{rp}$

**3. Idiosyncratic variances:**
$$\sigma_{e_i}^2 \sim IG(a_e, b_e)$$

Inverse-Gamma prior: $p(\sigma^2) \propto (\sigma^2)^{-(a+1)} \exp(-b/\sigma^2)$

Typical: $a_e = 2$, $b_e = 1$ (mean = 1, variance = $\infty$)

**4. Factor innovation covariance:**
$$\Sigma_\eta \sim IW(\nu_\eta, S_\eta)$$

Inverse-Wishart prior: $p(\Sigma) \propto |\Sigma|^{-(\nu+r+1)/2} \exp\left(-\frac{1}{2}\text{tr}(S\Sigma^{-1})\right)$

Typical: $\nu_\eta = r + 2$, $S_\eta = I_r$ (mean = $I_r/(r+1-r-2) = I_r$ if $\nu > r+1$)

Often **constrained** to $\Sigma_\eta = I_r$ for identification.

---

### Why Bayesian?

**1. Uncertainty Quantification**

MLE: $\hat{\Lambda}_{ML} = 0.8$ (point estimate)

Bayesian: $\Lambda | X \sim N(0.8, 0.05^2)$ (distribution)

Allows statements like "95% probability loading is between 0.7 and 0.9"

**2. Regularization**

Priors prevent overfitting by shrinking estimates toward reasonable values:
- Diffuse prior ($\tau^2 = 10$): little shrinkage, close to MLE
- Tight prior ($\tau^2 = 0.1$): strong shrinkage toward prior mean

**3. Incorporating Information**

Economic theory suggests loadings should be moderate (not 10 or 100):
$$\lambda_i \sim N(0, 2^2) \quad \text{(99.7% probability in [-6, 6])}$$

Stationarity requires eigenvalues of $\Phi$ inside unit circle:
- Truncate samples with eigenvalues > 1
- Or use prior that favors stationarity

**4. Handling Identification**

Instead of hard constraints ($\Lambda_{1:r}$ lower triangular), use soft priors:
- Put tight prior on scale: $\Sigma_\eta = I_r$ (exactly)
- Prior on loadings implicitly identifies model

### The Gibbs Sampler

Direct sampling from $p(\theta | X)$ is impossible. **Gibbs sampler** iteratively samples from conditional distributions:

**Iteration $k$:**
1. Draw $F_{1:T}^{(k)} | X, \Lambda^{(k-1)}, \Phi^{(k-1)}, \Sigma_\eta^{(k-1)}, \Sigma_e^{(k-1)}$
2. Draw $\Lambda^{(k)} | X, F_{1:T}^{(k)}, \Sigma_e^{(k-1)}$
3. Draw $\Phi^{(k)} | F_{1:T}^{(k)}, \Sigma_\eta^{(k-1)}$
4. Draw $\Sigma_e^{(k)} | X, F_{1:T}^{(k)}, \Lambda^{(k)}$
5. Draw $\Sigma_\eta^{(k)} | F_{1:T}^{(k)}, \Phi^{(k)}$

Each step samples from a simple distribution (normal, inverse-gamma, inverse-Wishart).

After burn-in, samples converge to draws from joint posterior $p(\theta, F | X)$.

### Visual Flow

```

Initialize: θ⁽⁰⁾, F⁽⁰⁾

Iteration k:
    ┌─────────────────────────────────────┐
    │ Sample F⁽ᵏ⁾ | X, θ⁽ᵏ⁻¹⁾             │ (Simulation smoother)
    │            ↓                        │
    │ Sample Λ⁽ᵏ⁾ | X, F⁽ᵏ⁾, Σₑ⁽ᵏ⁻¹⁾      │ (Multivariate normal)
    │            ↓                        │
    │ Sample Φ⁽ᵏ⁾ | F⁽ᵏ⁾, Σₙ⁽ᵏ⁻¹⁾         │ (Multivariate normal)
    │            ↓                        │
    │ Sample Σₑ⁽ᵏ⁾ | X, F⁽ᵏ⁾, Λ⁽ᵏ⁾        │ (Inverse-gamma × N)
    │            ↓                        │
    │ Sample Σₙ⁽ᵏ⁾ | F⁽ᵏ⁾, Φ⁽ᵏ⁾           │ (Inverse-Wishart)
    └─────────────────────────────────────┘

After burn-in (e.g., 5000 iterations):
    {θ⁽ᵏ⁾, F⁽ᵏ⁾}ₖ₌₅₀₀₁¹⁰⁰⁰⁰ ~ p(θ, F | X)
```

---

## Mathematical Formulation

### Gibbs Sampler Conditional Distributions

#### 1. Factors | Data, Parameters

**Draw:** $F_{1:T} | X_{1:T}, \Lambda, \Phi, \Sigma_\eta, \Sigma_e$

**Method:** Simulation smoother (Durbin & Koopman, 2002)

State-space form:
$$X_t = \Lambda F_t + e_t, \quad e_t \sim N(0, \Sigma_e)$$
$$F_t = \Phi F_{t-1} + \eta_t, \quad \eta_t \sim N(0, \Sigma_\eta)$$

**Algorithm:**
1. Run Kalman filter to get $\hat{F}_{t|t}, P_{t|t}$
2. Sample $\tilde{F}_T \sim N(\hat{F}_{T|T}, P_{T|T})$
3. For $t = T-1, ..., 1$:
   - Compute $\hat{F}_{t|t+1} = \hat{F}_{t|t} + J_t(\tilde{F}_{t+1} - \Phi \hat{F}_{t|t})$
   - Compute $P_{t|t+1} = P_{t|t} - J_t(\Phi P_{t|t} - P_{t+1|t})J_t'$
   - Sample $\tilde{F}_t \sim N(\hat{F}_{t|t+1}, P_{t|t+1})$

where $J_t = P_{t|t}\Phi' P_{t+1|t}^{-1}$

#### 2. Loadings | Data, Factors

**Draw:** $\lambda_j | X_{1:T}, F_{1:T}, \Sigma_e$ (column-wise)

**Distribution:** Conditional normal

For column $j$ of $\Lambda$:
$$X_t^{(j)} = \lambda_j F_t + e_t^{(j)}, \quad e_t^{(j)} \sim N(0, \sigma_{e_j}^2)$$

This is Bayesian linear regression:

**Prior:** $\lambda_j \sim N(\mu_0, \Sigma_0)$

**Posterior:**
$$\lambda_j | X, F \sim N(\tilde{\mu}_j, \tilde{\Sigma}_j)$$

where:
$$\tilde{\Sigma}_j = \left(\Sigma_0^{-1} + \frac{1}{\sigma_{e_j}^2}\sum_{t=1}^T F_t F_t'\right)^{-1}$$
$$\tilde{\mu}_j = \tilde{\Sigma}_j \left(\Sigma_0^{-1}\mu_0 + \frac{1}{\sigma_{e_j}^2}\sum_{t=1}^T X_t^{(j)} F_t\right)$$

With diffuse prior ($\Sigma_0 \to \infty$):
$$\tilde{\Sigma}_j = \sigma_{e_j}^2 \left(\sum_{t=1}^T F_t F_t'\right)^{-1}$$
$$\tilde{\mu}_j = \left(\sum_{t=1}^T F_t F_t'\right)^{-1} \sum_{t=1}^T X_t^{(j)} F_t$$

#### 3. Dynamics | Factors

**Draw:** $\phi_i | F_{1:T}, \Sigma_\eta$ (row-wise)

**Distribution:** Conditional normal

For row $i$ of $\Phi$:
$$F_{it} = \phi_i [F_{t-1}', ..., F_{t-p}']' + \eta_{it}, \quad \eta_{it} \sim N(0, \sigma_{\eta_i}^2)$$

**Prior:** $\phi_i \sim N(\mu_{\phi}, \Sigma_{\phi})$

**Posterior:**
$$\phi_i | F \sim N(\tilde{\mu}_i, \tilde{\Sigma}_i)$$

$$\tilde{\Sigma}_i = \left(\Sigma_{\phi}^{-1} + \frac{1}{\sigma_{\eta_i}^2}\sum_{t=p+1}^T F_{t-1:t-p} F_{t-1:t-p}'\right)^{-1}$$
$$\tilde{\mu}_i = \tilde{\Sigma}_i \left(\Sigma_{\phi}^{-1}\mu_{\phi} + \frac{1}{\sigma_{\eta_i}^2}\sum_{t=p+1}^T F_{it} F_{t-1:t-p}\right)$$

where $F_{t-1:t-p} = [F_{t-1}', ..., F_{t-p}']'$ is the lagged state.

#### 4. Idiosyncratic Variances | Data, Factors, Loadings

**Draw:** $\sigma_{e_i}^2 | X_{1:T}, F_{1:T}, \Lambda$ (element-wise)

**Distribution:** Conditional inverse-gamma

**Prior:** $\sigma_{e_i}^2 \sim IG(a_0, b_0)$

**Likelihood:** $X_t^{(i)} - \lambda_i' F_t \sim N(0, \sigma_{e_i}^2)$

**Posterior:**
$$\sigma_{e_i}^2 | X, F, \Lambda \sim IG(\tilde{a}_i, \tilde{b}_i)$$

$$\tilde{a}_i = a_0 + T/2$$
$$\tilde{b}_i = b_0 + \frac{1}{2}\sum_{t=1}^T (X_t^{(i)} - \lambda_i' F_t)^2$$

**Sampling:** If $\sigma^2 \sim IG(a, b)$, then $\sigma^2 = b / \chi^2_{2a}$

#### 5. Factor Innovation Covariance | Factors, Dynamics

**Draw:** $\Sigma_\eta | F_{1:T}, \Phi$

**Distribution:** Conditional inverse-Wishart

**Prior:** $\Sigma_\eta \sim IW(\nu_0, S_0)$

**Likelihood:** $F_t - \Phi F_{t-1} \sim N(0, \Sigma_\eta)$

**Posterior:**
$$\Sigma_\eta | F, \Phi \sim IW(\tilde{\nu}, \tilde{S})$$

$$\tilde{\nu} = \nu_0 + (T - p)$$
$$\tilde{S} = S_0 + \sum_{t=p+1}^T (F_t - \Phi F_{t-1})(F_t - \Phi F_{t-1})'$$

**Note:** Often **fixed** at $\Sigma_\eta = I_r$ for identification.

### MCMC Algorithm Summary

```

1. Initialize: Λ⁽⁰⁾, Φ⁽⁰⁾, Σₑ⁽⁰⁾, Σₙ⁽⁰⁾ (from PCA/VAR)

2. For k = 1, ..., K:
   a. Draw F⁽ᵏ⁾ ~ p(F | X, Λ⁽ᵏ⁻¹⁾, Φ⁽ᵏ⁻¹⁾, Σₑ⁽ᵏ⁻¹⁾, Σₙ⁽ᵏ⁻¹⁾) via simulation smoother

   b. For j = 1, ..., r:
      Draw λⱼ⁽ᵏ⁾ ~ N(μ̃ⱼ, Σ̃ⱼ)

   c. For i = 1, ..., r:
      Draw φᵢ⁽ᵏ⁾ ~ N(μ̃ᵢ, Σ̃ᵢ)

   d. For i = 1, ..., N:
      Draw σₑᵢ²⁽ᵏ⁾ ~ IG(ãᵢ, b̃ᵢ)

   e. Draw Σₙ⁽ᵏ⁾ ~ IW(ν̃, S̃) [or fix at I_r]

3. Discard first B iterations (burn-in, e.g., B = 5000)

4. Posterior samples: {θ⁽ᵏ⁾}ₖ₌ᵦ₊₁ᴷ ~ p(θ | X)
```

---

## Code Implementation

### Complete Gibbs Sampler


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">bayesiandfm.py</span>

```python
import numpy as np
from scipy.stats import invgamma, invwishart, multivariate_normal
from scipy.linalg import solve, cholesky

class BayesianDFM:
    """
    Bayesian Dynamic Factor Model via Gibbs sampler.

    Model:
        X_t = Lambda * F_t + e_t,  e_t ~ N(0, Sigma_e)
        F_t = Phi * F_{t-1} + eta_t,  eta_t ~ N(0, Sigma_eta)
    """

    def __init__(self, n_factors, n_lags=1):
        self.r = n_factors
        self.p = n_lags

    def initialize_priors(self, N, prior_config=None):
        """Set prior hyperparameters."""
        if prior_config is None:
            prior_config = {}

        # Loadings: diffuse normal
        self.mu_lambda = prior_config.get('mu_lambda', np.zeros(self.r))
        self.Sigma_lambda = prior_config.get('Sigma_lambda', 10 * np.eye(self.r))

        # Dynamics: weakly informative (AR(1)-ish)
        self.mu_phi = prior_config.get('mu_phi', np.zeros(self.r * self.p))
        self.mu_phi[0] = 0.5  # Slight persistence
        self.Sigma_phi = prior_config.get('Sigma_phi', np.eye(self.r * self.p))

        # Idiosyncratic variances: weakly informative IG
        self.a_e = prior_config.get('a_e', 2.0)
        self.b_e = prior_config.get('b_e', 1.0)

        # Factor innovation: fixed for identification
        self.Sigma_eta = np.eye(self.r)
        self.fix_Sigma_eta = prior_config.get('fix_Sigma_eta', True)

        if not self.fix_Sigma_eta:
            self.nu_eta = prior_config.get('nu_eta', self.r + 2)
            self.S_eta = prior_config.get('S_eta', np.eye(self.r))

    def initialize_from_pca(self, X):
        """Initialize parameters from PCA."""
        from sklearn.decomposition import PCA
        from statsmodels.tsa.api import VAR

        T, N = X.shape
        self.N = N
        self.T = T

        # PCA
        pca = PCA(n_components=self.r)
        F_init = pca.fit_transform(X)
        Lambda_init = pca.components_.T

        # VAR
        var_model = VAR(F_init)
        var_fit = var_model.fit(self.p, trend='nc')
        Phi_init = var_fit.params.T[:self.r, :]

        # Variances
        resid_X = X - F_init @ Lambda_init.T
        sigma_e_init = np.var(resid_X, axis=0)

        return Lambda_init, Phi_init, sigma_e_init, F_init

    def simulation_smoother(self, X, Lambda, Phi, Sigma_e):
        """
        Draw factors from p(F | X, theta) via simulation smoother.

        Uses the method of Durbin & Koopman (2002).
        """
        T, N = X.shape
        r = self.r

        # Run Kalman filter
        F_filt, P_filt, F_pred, P_pred = self._kalman_filter(X, Lambda, Phi, Sigma_e)

        # Backward simulation
        F_draw = np.zeros((T, r))

        # Sample at T
        try:
            L = cholesky(P_filt[T-1], lower=True)
            F_draw[T-1] = F_filt[T-1] + L @ np.random.randn(r)
        except:
            F_draw[T-1] = F_filt[T-1]

        # Backward recursion
        for t in range(T-2, -1, -1):
            try:
                J_t = P_filt[t] @ Phi.T @ solve(P_pred[t+1], np.eye(r))

                F_cond_mean = F_filt[t] + J_t @ (F_draw[t+1] - F_pred[t+1])
                P_cond = P_filt[t] - J_t @ P_pred[t+1] @ J_t.T
                P_cond = (P_cond + P_cond.T) / 2

                L = cholesky(P_cond, lower=True)
                F_draw[t] = F_cond_mean + L @ np.random.randn(r)
            except:
                F_draw[t] = F_filt[t]

        return F_draw

    def _kalman_filter(self, X, Lambda, Phi, Sigma_e):
        """Run Kalman filter (simplified for AR(1) factors)."""
        T, N = X.shape
        r = self.r

        F_filt = np.zeros((T, r))
        P_filt = np.zeros((T, r, r))
        F_pred = np.zeros((T, r))
        P_pred = np.zeros((T, r, r))

        # Initialize
        F_pred[0] = np.zeros(r)

        # Solve for stationary variance: P = Phi P Phi' + Q
        vec_Q = self.Sigma_eta.ravel()
        Phi_kron = np.kron(Phi, Phi)
        try:
            vec_P0 = solve(np.eye(r**2) - Phi_kron, vec_Q)
            P_pred[0] = vec_P0.reshape(r, r)
        except:
            P_pred[0] = np.eye(r) * 10

        for t in range(T):
            # Update
            v = X[t] - Lambda @ F_pred[t]
            F_var = Lambda @ P_pred[t] @ Lambda.T + Sigma_e

            try:
                K = P_pred[t] @ Lambda.T @ solve(F_var, np.eye(N))
                F_filt[t] = F_pred[t] + K @ v
                P_filt[t] = P_pred[t] - K @ Lambda @ P_pred[t]
                P_filt[t] = (P_filt[t] + P_filt[t].T) / 2
            except:
                F_filt[t] = F_pred[t]
                P_filt[t] = P_pred[t]

            # Predict next
            if t < T - 1:
                F_pred[t+1] = Phi @ F_filt[t]
                P_pred[t+1] = Phi @ P_filt[t] @ Phi.T + self.Sigma_eta
                P_pred[t+1] = (P_pred[t+1] + P_pred[t+1].T) / 2

        return F_filt, P_filt, F_pred, P_pred

    def draw_loadings(self, X, F, Sigma_e):
        """Draw Lambda | X, F, Sigma_e (column-wise)."""
        T = F.shape[0]
        Lambda = np.zeros((self.N, self.r))

        FF = F.T @ F
        Sigma_0_inv = solve(self.Sigma_lambda, np.eye(self.r))

        for i in range(self.N):
            # Posterior precision and mean
            Sigma_i_inv = Sigma_0_inv + FF / Sigma_e[i, i]
            Sigma_i = solve(Sigma_i_inv, np.eye(self.r))

            mu_i = Sigma_i @ (Sigma_0_inv @ self.mu_lambda + (F.T @ X[:, i]) / Sigma_e[i, i])

            # Draw
            Lambda[i, :] = multivariate_normal.rvs(mean=mu_i, cov=Sigma_i)

        return Lambda

    def draw_dynamics(self, F):
        """Draw Phi | F, Sigma_eta."""
        T = F.shape[0]

        # Construct lagged factors
        Y = F[self.p:, :]  # T-p x r
        X_lag = np.hstack([F[self.p-lag-1:T-lag-1, :] for lag in range(self.p)])  # T-p x rp

        Phi_post = np.zeros((self.r, self.r * self.p))

        Sigma_phi_inv = solve(self.Sigma_phi, np.eye(self.r * self.p))
        XX = X_lag.T @ X_lag

        for i in range(self.r):
            # Posterior
            Sigma_i_inv = Sigma_phi_inv + XX / self.Sigma_eta[i, i]
            Sigma_i = solve(Sigma_i_inv, np.eye(self.r * self.p))

            mu_i = Sigma_i @ (Sigma_phi_inv @ self.mu_phi + (X_lag.T @ Y[:, i]) / self.Sigma_eta[i, i])

            # Draw
            Phi_post[i, :] = multivariate_normal.rvs(mean=mu_i, cov=Sigma_i)

        return Phi_post[:, :self.r] if self.p == 1 else Phi_post

    def draw_sigma_e(self, X, F, Lambda):
        """Draw Sigma_e | X, F, Lambda (diagonal)."""
        T = X.shape[0]
        sigma_e = np.zeros(self.N)

        for i in range(self.N):
            resid = X[:, i] - F @ Lambda[i, :]
            SS = np.sum(resid**2)

            # Posterior IG parameters
            a_post = self.a_e + T / 2
            b_post = self.b_e + SS / 2

            # Draw
            sigma_e[i] = invgamma.rvs(a=a_post, scale=b_post)

        return np.diag(sigma_e)

    def fit(self, X, n_iter=10000, burn_in=5000, thin=5, verbose=True):
        """
        Fit Bayesian DFM via Gibbs sampler.

        Parameters
        ----------
        X : ndarray (T, N)
            Data (demeaned)
        n_iter : int
            Total MCMC iterations
        burn_in : int
            Burn-in iterations to discard
        thin : int
            Thinning interval
        verbose : bool
            Print progress

        Returns
        -------
        samples : dict
            Posterior samples for parameters
        """
        T, N = X.shape
        self.initialize_priors(N)

        # Initialize
        Lambda, Phi, sigma_e_vec, F = self.initialize_from_pca(X)
        Sigma_e = np.diag(sigma_e_vec)

        # Storage
        n_saved = (n_iter - burn_in) // thin
        samples = {
            'Lambda': np.zeros((n_saved, N, self.r)),
            'Phi': np.zeros((n_saved, self.r, self.r)),
            'sigma_e': np.zeros((n_saved, N)),
            'F': np.zeros((n_saved, T, self.r))
        }

        save_idx = 0

        for iteration in range(n_iter):
            # Gibbs steps
            F = self.simulation_smoother(X, Lambda, Phi, Sigma_e)
            Lambda = self.draw_loadings(X, F, Sigma_e)
            Phi = self.draw_dynamics(F)
            Sigma_e = self.draw_sigma_e(X, F, Lambda)

            # Save after burn-in
            if iteration >= burn_in and (iteration - burn_in) % thin == 0:
                samples['Lambda'][save_idx] = Lambda
                samples['Phi'][save_idx] = Phi
                samples['sigma_e'][save_idx] = np.diag(Sigma_e)
                samples['F'][save_idx] = F
                save_idx += 1

            if verbose and iteration % 1000 == 0:
                print(f"Iteration {iteration}/{n_iter}")

        self.samples_ = samples

        # Posterior means
        self.Lambda_post_mean_ = samples['Lambda'].mean(axis=0)
        self.Phi_post_mean_ = samples['Phi'].mean(axis=0)
        self.sigma_e_post_mean_ = samples['sigma_e'].mean(axis=0)
        self.F_post_mean_ = samples['F'].mean(axis=0)

        return samples


# Example usage
if __name__ == '__main__':
    np.random.seed(42)

    # Simulate data
    T, N, r = 200, 10, 2

    Lambda_true = np.random.randn(N, r)
    Phi_true = np.array([[0.8, 0.1], [0.1, 0.7]])
    sigma_e_true = np.ones(N) * 0.5

    F = np.zeros((T, r))
    for t in range(1, T):
        F[t] = Phi_true @ F[t-1] + np.random.randn(r)

    X = F @ Lambda_true.T + np.random.randn(T, N) * sigma_e_true
    X = X - X.mean(axis=0)

    # Fit Bayesian model
    model = BayesianDFM(n_factors=2, n_lags=1)
    samples = model.fit(X, n_iter=6000, burn_in=1000, thin=5, verbose=True)

    print("\n=== Posterior Means ===")
    print("Phi:")
    print(model.Phi_post_mean_.round(3))
    print("\nTrue Phi:")
    print(Phi_true.round(3))

    # Posterior credible intervals
    Phi_samples = samples['Phi']
    Phi_lower = np.percentile(Phi_samples, 2.5, axis=0)
    Phi_upper = np.percentile(Phi_samples, 97.5, axis=0)

    print("\n95% Credible Intervals for Phi[0,0]:")
    print(f"[{Phi_lower[0,0]:.3f}, {Phi_upper[0,0]:.3f}]")
```

</div>
</div>

---

## Common Pitfalls

### 1. Poor Mixing

**Problem:** MCMC chain gets stuck, explores posterior slowly.

**Solutions:**
- Use informative priors to constrain parameter space
- Re-parameterize (e.g., correlation instead of covariance)
- Increase burn-in period
- Use Metropolis-within-Gibbs for problematic parameters

### 2. Identification Issues

**Problem:** Posterior samples show label switching or sign flipping.

**Solutions:**
- Fix $\Sigma_\eta = I_r$ exactly
- Post-process: match signs to maximize correlation with reference
- Use ordering constraint: $\Lambda_{11} > \Lambda_{12} > ... > \Lambda_{1r}$

### 3. Convergence Failure

**Problem:** Chains haven't converged (Gelman-Rubin $\hat{R} > 1.1$).

**Solutions:**
- Run longer (more burn-in, more samples)
- Improve initialization (use MLE estimates)
- Check for coding errors in conditionals

### 4. Improper Priors

**Problem:** Diffuse priors lead to non-integrable posterior.

**Solutions:**
- Use proper priors (finite variance)
- For "non-informative," use large but finite variance ($\tau^2 = 10^4$)

---

## Connections

- **Builds on:** State-space representation, Kalman filter/smoother
- **Leads to:** Bayesian forecasting, impulse response uncertainty
- **Related to:** Hierarchical models (cross-sectional priors), shrinkage methods

---

## Practice Problems

### Conceptual

1. Why do Bayesian credible intervals have a different interpretation than frequentist confidence intervals?

2. How do priors help with identification compared to hard constraints?

3. Explain why the Gibbs sampler works: why do conditional samples converge to joint posterior?

### Mathematical

4. Derive the conditional posterior for $\lambda_i$ (column of loadings) from Bayes' theorem.

5. Show that with flat priors, the posterior mean equals the MLE.

6. Derive the inverse-Wishart posterior for $\Sigma_\eta$ given factors and dynamics.

### Implementation

7. Implement MCMC diagnostics: trace plots, Geweke test, effective sample size.

8. Run multiple chains and compute Gelman-Rubin statistic for each parameter.

9. Compare posterior intervals with ML standard errors. When do they differ most?

---

<div class="callout-insight">

**Insight:** Understanding bayesian estimation for dynamic factor models is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## Further Reading

- **Kim & Nelson (1999).** *State-Space Models with Regime Switching*, Chapter 6.
  - Bayesian estimation of state-space models

- **Carter & Kohn (1994).** "On Gibbs Sampling for State Space Models." *Biometrika* 81(3), 541-553.
  - Original simulation smoother algorithm

- **Durbin & Koopman (2002).** "A Simple and Efficient Simulation Smoother for State Space Time Series Analysis." *Biometrika* 89(3), 603-616.
  - Modern efficient simulation smoother

- **Del Negro & Otrok (2008).** "Dynamic Factor Models with Time-Varying Parameters: Measuring Changes in International Business Cycles." Federal Reserve Bank of New York Staff Report 326.
  - Extensions with time-varying parameters

- **Korobilis (2013).** "Assessing the Transmission of Monetary Policy Using Time-Varying Parameter Dynamic Factor Models." *Oxford Bulletin of Economics and Statistics* 75(2), 157-179.
  - Applications to monetary policy

- **Koop & Korobilis (2014).** "A New Index of Financial Conditions." *European Economic Review* 71, 101-116.
  - Bayesian DFM for financial conditions index

---

## Conceptual Practice Questions

1. In your own words, explain the difference between common factors and idiosyncratic components.

2. Why do factor models require identification restrictions? Give a concrete example.

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.

</div>

---

## Cross-References

<a class="link-card" href="./03_bayesian_dfm_slides.md">
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

<a class="link-card" href="./02_em_algorithm_dfm.md">
  <div class="link-card-title">02 Em Algorithm Dfm</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

