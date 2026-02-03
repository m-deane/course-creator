# Non-Gaussian Factor Models

## In Brief

Financial returns exhibit fat tails, asymmetry, and occasional extreme outliers—phenomena inconsistent with Gaussian assumptions. Non-Gaussian factor models use heavy-tailed distributions (Student-t, mixture of Gaussians) and robust estimation to capture realistic data-generating processes, improving both parameter estimates and risk assessment.

## Key Insight

Gaussian factor models minimize squared residuals, giving extreme observations enormous influence. A single outlier can distort all loadings. Non-Gaussian models downweight outliers automatically through likelihood-based mechanisms, producing stable estimates. Student-t distributions with low degrees of freedom accommodate fat tails while maintaining tractable inference via EM algorithms.

---

## 1. Limitations of Gaussian Assumptions

### Formal Definition

**Standard Gaussian Factor Model:**
$$X_t = \Lambda F_t + e_t, \quad e_t \sim N(0, \Sigma_e)$$
$$F_t \sim N(\mu_F, \Sigma_F)$$

Assumes:
- All shocks Gaussian (thin tails)
- Symmetric distributions
- Quadratic loss (squared errors)

### Intuitive Explanation

**Stylized Facts from Financial Data:**

1. **Fat Tails**: Returns more likely to exceed 3-4 standard deviations than Gaussian predicts
   - Black Monday (1987): -22.6% drop (>20 sigma event under Gaussian)
   - Flash Crash (2010): Should happen once in billions of years under normality

2. **Asymmetry**: Crashes more severe than rallies
   - Negative skewness in equity returns
   - Leverage effect (volatility increases after declines)

3. **Outliers**: Extreme observations not rare errors but genuine data features
   - Financial crises
   - Policy surprises
   - Natural disasters

**Consequences of Gaussian Misspecification:**

1. **Biased Parameters**: Outliers pull loadings away from bulk of data
2. **Poor Risk Assessment**: Underestimate tail probabilities
3. **Inefficient Estimation**: Not using optimal weighting
4. **Misleading Inference**: Confidence intervals too narrow

### Mathematical Framework

**Kurtosis (Fourth Moment):**

Gaussian: $\kappa = 3$ (excess kurtosis = 0)

Heavy-tailed: $\kappa > 3$ (excess kurtosis > 0)

**Student-t Distribution:**
$$f(x; \nu, \mu, \sigma^2) = \frac{\Gamma((\nu+1)/2)}{\Gamma(\nu/2)\sqrt{\nu \pi \sigma^2}} \left(1 + \frac{(x-\mu)^2}{\nu \sigma^2}\right)^{-(\nu+1)/2}$$

where $\nu$ = degrees of freedom controls tail thickness:
- $\nu \to \infty$: approaches Gaussian
- $\nu = 1$: Cauchy (very heavy tails, no mean)
- $\nu = 3$: Infinite fourth moment (kurtosis undefined)
- Typical financial data: $\nu \in [4, 10]$

**Kurtosis of Student-t:**
$$\kappa = 3 + \frac{6}{\nu - 4}, \quad \nu > 4$$

Example: $\nu = 5 \implies \kappa = 9$ (much heavier than Gaussian)

### Code Implementation: Detecting Non-Gaussianty

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import jarque_bera, shapiro, kstest

def test_gaussianity(data, name="Variable"):
    """
    Test whether data is consistent with Gaussian distribution.

    Parameters
    ----------
    data : array-like
        Data to test
    name : str
        Variable name for display

    Returns
    -------
    results : dict
        Test statistics and p-values
    """
    data = np.asarray(data).ravel()
    data = data[~np.isnan(data)]  # Remove NaNs

    results = {
        'mean': np.mean(data),
        'std': np.std(data, ddof=1),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data, fisher=True),  # Excess kurtosis
    }

    # Jarque-Bera test (skewness and kurtosis)
    jb_stat, jb_pval = jarque_bera(data)
    results['jarque_bera_stat'] = jb_stat
    results['jarque_bera_pval'] = jb_pval

    # Shapiro-Wilk test
    if len(data) <= 5000:  # Shapiro-Wilk limited to 5000 obs
        sw_stat, sw_pval = shapiro(data)
        results['shapiro_wilk_stat'] = sw_stat
        results['shapiro_wilk_pval'] = sw_pval

    # Kolmogorov-Smirnov test
    ks_stat, ks_pval = kstest(
        (data - np.mean(data)) / np.std(data, ddof=1),
        'norm'
    )
    results['ks_stat'] = ks_stat
    results['ks_pval'] = ks_pval

    # Print results
    print(f"\nGaussianity Tests: {name}")
    print("-" * 50)
    print(f"Skewness: {results['skewness']:.3f} (Gaussian: 0)")
    print(f"Excess Kurtosis: {results['kurtosis']:.3f} (Gaussian: 0)")
    print(f"\nJarque-Bera test: stat={jb_stat:.2f}, p-value={jb_pval:.4f}")
    if jb_pval < 0.05:
        print("  -> Reject Gaussianity at 5% level")
    else:
        print("  -> Cannot reject Gaussianity")

    if 'shapiro_wilk_pval' in results:
        print(f"Shapiro-Wilk test: stat={sw_stat:.4f}, p-value={sw_pval:.4f}")

    print(f"Kolmogorov-Smirnov test: stat={ks_stat:.4f}, p-value={ks_pval:.4f}")

    return results


def compare_distributions(data, name="Data"):
    """
    Fit both Gaussian and Student-t, compare via likelihood.

    Parameters
    ----------
    data : array-like
        Data to fit
    name : str
        Label for plots

    Returns
    -------
    comparison : dict
        Fitted parameters and log-likelihoods
    """
    data = np.asarray(data).ravel()

    # Fit Gaussian
    mu_gauss, sigma_gauss = np.mean(data), np.std(data, ddof=1)
    ll_gauss = np.sum(stats.norm.logpdf(data, mu_gauss, sigma_gauss))

    # Fit Student-t (MLE)
    params_t = stats.t.fit(data)
    df_t, mu_t, sigma_t = params_t
    ll_t = np.sum(stats.t.logpdf(data, df_t, mu_t, sigma_t))

    # Likelihood ratio test
    lr_stat = 2 * (ll_t - ll_gauss)
    # Asymptotically chi^2(1) since Student-t has 1 extra parameter (df)
    lr_pval = 1 - stats.chi2.cdf(lr_stat, df=1)

    comparison = {
        'gaussian': {'mu': mu_gauss, 'sigma': sigma_gauss, 'loglik': ll_gauss},
        'student_t': {'df': df_t, 'mu': mu_t, 'sigma': sigma_t, 'loglik': ll_t},
        'lr_stat': lr_stat,
        'lr_pval': lr_pval
    }

    print(f"\nDistribution Comparison: {name}")
    print("-" * 50)
    print(f"Gaussian: mu={mu_gauss:.3f}, sigma={sigma_gauss:.3f}, LogLik={ll_gauss:.2f}")
    print(f"Student-t: df={df_t:.2f}, mu={mu_t:.3f}, sigma={sigma_t:.3f}, LogLik={ll_t:.2f}")
    print(f"\nLikelihood Ratio Test: LR={lr_stat:.2f}, p-value={lr_pval:.4f}")

    if lr_pval < 0.05:
        print("  -> Student-t fits significantly better")
    else:
        print("  -> No significant difference")

    # QQ-plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Gaussian QQ-plot
    stats.probplot(data, dist="norm", plot=axes[0])
    axes[0].set_title(f"{name}: Gaussian QQ-Plot")

    # Student-t QQ-plot
    stats.probplot(data, dist=stats.t, sparams=(df_t,), plot=axes[1])
    axes[1].set_title(f"{name}: Student-t(df={df_t:.1f}) QQ-Plot")

    plt.tight_layout()
    plt.show()

    return comparison


# Example: Compare Gaussian vs heavy-tailed data
np.random.seed(42)
n = 1000

# Generate data: Student-t(5) with occasional outliers
data_t = stats.t.rvs(df=5, loc=0, scale=1, size=n)
data_gauss = np.random.randn(n)

print("=" * 70)
print("HEAVY-TAILED DATA (Student-t with df=5)")
print("=" * 70)
test_gaussianity(data_t, "Student-t(5) Data")
comp_t = compare_distributions(data_t, "Student-t(5) Data")

print("\n" + "=" * 70)
print("GAUSSIAN DATA")
print("=" * 70)
test_gaussianity(data_gauss, "Gaussian Data")
comp_gauss = compare_distributions(data_gauss, "Gaussian Data")
```

---

## 2. Student-t Factor Models

### Model Specification

**Student-t Factor Model:**
$$X_t | F_t \sim t_{\nu_e}(\Lambda F_t, \Sigma_e)$$
$$F_t \sim t_{\nu_F}(\mu_F, \Sigma_F)$$

where $\nu_e, \nu_F$ are degrees of freedom for idiosyncratic errors and factors.

**Scale Mixture Representation:**

Key insight: Student-t is Gaussian scale mixture.

$$X_t | F_t, w_t \sim N(\Lambda F_t, w_t^{-1} \Sigma_e)$$
$$w_t \sim \text{Gamma}(\nu_e/2, \nu_e/2)$$

where $w_t$ is latent precision (inverse variance) weight.

**Intuition:**
- Data points with low $w_t$ (high variance) are outliers
- Model automatically downweights them
- EM algorithm treats $w_t$ as missing data

### EM Algorithm for Student-t Factor Model

**E-step:** Compute posterior of latent variables $(F_t, w_t)$ given data and parameters.

For Student-t with df $\nu$:
$$E[w_t | X_t] = \frac{\nu + N}{\nu + \delta_t^2}$$

where $\delta_t^2 = (X_t - \Lambda F_t)' \Sigma_e^{-1} (X_t - \Lambda F_t)$ is squared Mahalanobis distance.

**Key Property:** Outliers (large $\delta_t^2$) get small weights $w_t \approx \nu / \delta_t^2$.

**M-step:** Update parameters using weighted likelihood.

Update loadings:
$$\Lambda^{\text{new}} = \left(\sum_t w_t X_t F_t'\right) \left(\sum_t w_t F_t F_t'\right)^{-1}$$

This is weighted least squares with weights $w_t$!

### Code Implementation: Student-t Factor Model

```python
from scipy.special import digamma, logsumexp
from scipy.optimize import minimize_scalar

class StudentTFactorModel:
    """
    Factor model with Student-t distributions for robust estimation.

    Model:
        X_t | F_t, w_t ~ N(Lambda * F_t, w_t^{-1} * Sigma_e)
        w_t ~ Gamma(nu/2, nu/2)
        F_t ~ N(0, I)

    Equivalently: X_t | F_t ~ t_nu(Lambda * F_t, Sigma_e)
    """

    def __init__(self, n_factors, df_init=5):
        """
        Parameters
        ----------
        n_factors : int
            Number of latent factors
        df_init : float
            Initial degrees of freedom for Student-t (>2)
        """
        self.n_factors = n_factors
        self.df = df_init
        self.Lambda = None
        self.Sigma_e = None
        self.F = None
        self.weights = None  # Latent precision weights

    def fit(self, X, n_iter=50, update_df=True, verbose=False):
        """
        Fit Student-t factor model via EM algorithm.

        Parameters
        ----------
        X : ndarray, shape (T, N)
            Observed data
        n_iter : int
            Number of EM iterations
        update_df : bool
            Whether to estimate degrees of freedom
        verbose : bool
            Print progress

        Returns
        -------
        self
        """
        T, N = X.shape
        r = self.n_factors

        # Initialize with PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=r)
        self.F = pca.fit_transform(X)
        self.Lambda = pca.components_.T
        residuals = X - self.F @ self.Lambda.T
        self.Sigma_e = np.diag(np.var(residuals, axis=0))

        # Initialize weights (uniform for Gaussian start)
        self.weights = np.ones(T)

        log_likelihoods = []

        for iteration in range(n_iter):
            # E-step: Update latent weights and factors
            self._e_step(X)

            # M-step: Update parameters
            self._m_step(X)

            # Update degrees of freedom
            if update_df:
                self._update_df()

            # Compute log-likelihood
            ll = self._log_likelihood(X)
            log_likelihoods.append(ll)

            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: LogLik={ll:.2f}, df={self.df:.2f}")

        return self

    def _e_step(self, X):
        """
        E-step: Compute posterior expectations of latent variables.

        Updates:
            - weights w_t
            - factors F_t
        """
        T, N = X.shape
        r = self.n_factors

        # Update weights: E[w_t | X_t]
        for t in range(T):
            residual = X[t] - self.Lambda @ self.F[t]
            # Mahalanobis distance
            delta_sq = residual @ np.linalg.inv(self.Sigma_e) @ residual

            # Posterior mean of weight
            self.weights[t] = (self.df + N) / (self.df + delta_sq)

        # Update factors: E[F_t | X_t, w_t]
        # Posterior: F_t | X_t, w_t ~ N(mu_t, V_t)
        for t in range(T):
            Sigma_e_inv = np.linalg.inv(self.Sigma_e)
            V_t_inv = np.eye(r) + self.weights[t] * self.Lambda.T @ Sigma_e_inv @ self.Lambda
            V_t = np.linalg.inv(V_t_inv)
            mu_t = self.weights[t] * V_t @ self.Lambda.T @ Sigma_e_inv @ X[t]
            self.F[t] = mu_t

    def _m_step(self, X):
        """
        M-step: Update parameters via weighted maximum likelihood.
        """
        T, N = X.shape
        r = self.n_factors

        # Update loadings (weighted least squares)
        XF = np.zeros((N, r))
        FF = np.zeros((r, r))

        for t in range(T):
            XF += self.weights[t] * np.outer(X[t], self.F[t])
            FF += self.weights[t] * np.outer(self.F[t], self.F[t])

        self.Lambda = XF @ np.linalg.inv(FF)

        # Update error covariance (weighted)
        resid_sq = np.zeros(N)
        for t in range(T):
            residual = X[t] - self.Lambda @ self.F[t]
            resid_sq += self.weights[t] * residual ** 2

        self.Sigma_e = np.diag(resid_sq / T)

    def _update_df(self):
        """
        Update degrees of freedom via profile likelihood.
        """
        T = len(self.weights)

        def neg_profile_loglik(nu):
            if nu <= 2:
                return np.inf

            # Log-likelihood of weights under Gamma(nu/2, nu/2)
            from scipy.special import gammaln
            ll = 0
            for w in self.weights:
                # Gamma density
                ll += (nu/2) * np.log(nu/2) - gammaln(nu/2)
                ll += (nu/2 - 1) * np.log(w) - (nu/2) * w

            # Add weight expectation terms
            ll += T * (digamma(nu/2) - np.log(nu/2)) * (nu/2)
            ll -= T * np.mean(self.weights + np.log(self.weights))

            return -ll

        # Optimize over reasonable range
        result = minimize_scalar(neg_profile_loglik, bounds=(2.5, 30), method='bounded')
        self.df = result.x

    def _log_likelihood(self, X):
        """
        Compute observed data log-likelihood (approximate via weights).
        """
        T, N = X.shape
        ll = 0

        for t in range(T):
            residual = X[t] - self.Lambda @ self.F[t]
            delta_sq = residual @ np.linalg.inv(self.Sigma_e) @ residual

            # Student-t log-density (multivariate)
            from scipy.special import gammaln
            ll += gammaln((self.df + N) / 2) - gammaln(self.df / 2)
            ll -= 0.5 * N * np.log(self.df * np.pi)
            ll -= 0.5 * np.log(np.linalg.det(self.Sigma_e))
            ll -= ((self.df + N) / 2) * np.log(1 + delta_sq / self.df)

        return ll

    def get_outlier_scores(self):
        """
        Return outlier scores based on weights.

        Lower weight = more outlying observation.

        Returns
        -------
        scores : ndarray, shape (T,)
            Outlier scores (1 / w_t)
        """
        return 1 / self.weights

    def plot_weights(self, dates=None):
        """
        Plot time series of observation weights.

        Low weights indicate outliers/heavy-tailed observations.
        """
        T = len(self.weights)
        if dates is None:
            dates = np.arange(T)

        plt.figure(figsize=(12, 4))
        plt.plot(dates, self.weights, linewidth=1.5)
        plt.axhline(y=1, color='r', linestyle='--', label='Gaussian weight')
        plt.xlabel('Time')
        plt.ylabel('Weight $w_t$')
        plt.title('Observation Weights (Low = Outlier)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()


# Example: Robust estimation with contaminated data
np.random.seed(42)
T, N, r = 200, 15, 3

# Generate clean factors and data
F_true = np.random.randn(T, r)
Lambda_true = np.random.randn(N, r)
X_clean = F_true @ Lambda_true.T + np.random.randn(T, N) * 0.3

# Add outliers at specific times
outlier_times = [50, 100, 150]
for t in outlier_times:
    X_clean[t] += np.random.randn(N) * 5  # Large shocks

# Fit Gaussian model (standard PCA)
print("Fitting Gaussian (PCA) model...")
pca_model = PCA(n_components=r)
F_pca = pca_model.fit_transform(X_clean)
Lambda_pca = pca_model.components_.T

# Fit Student-t model
print("Fitting Student-t model...")
t_model = StudentTFactorModel(n_factors=r, df_init=5)
t_model.fit(X_clean, n_iter=30, update_df=True, verbose=True)

print(f"\nEstimated degrees of freedom: {t_model.df:.2f}")

# Identify outliers
outlier_scores = t_model.get_outlier_scores()
detected_outliers = np.argsort(outlier_scores)[-5:]  # Top 5 outliers

print(f"\nDetected outlier times: {sorted(detected_outliers)}")
print(f"True outlier times: {outlier_times}")

# Plot weights
fig = t_model.plot_weights()
for t in outlier_times:
    plt.axvline(x=t, color='red', alpha=0.5, linestyle=':', linewidth=2)
plt.legend(['Weights', 'Gaussian weight', 'True outliers'])
plt.show()

# Compare loadings
print("\nLoading comparison (variable 0):")
print(f"True: {Lambda_true[0]}")
print(f"PCA: {Lambda_pca[0]}")
print(f"Student-t: {t_model.Lambda[0]}")
```

---

## 3. Alternative Heavy-Tailed Specifications

### Mixture of Gaussians

**Model:**
$$X_t | F_t, z_t \sim N(\Lambda F_t, \Sigma_{e,z_t})$$
$$P(z_t = k) = \pi_k, \quad \sum_{k=1}^K \pi_k = 1$$

where $z_t$ is latent mixture component.

**Interpretation:**
- $K=2$: "normal" vs "crisis" periods
- Different components have different variances
- Captures both fat tails and multimodality

**EM Algorithm:**

E-step: Compute posterior probabilities
$$\gamma_{tk} = P(z_t = k | X_t) \propto \pi_k \cdot N(X_t; \Lambda F_t, \Sigma_{e,k})$$

M-step: Update mixture weights
$$\pi_k^{\text{new}} = \frac{1}{T} \sum_t \gamma_{tk}$$

### Asymmetric Distributions

**Skew-Normal Factor Model:**

Allow asymmetry via skew-normal distribution:
$$X_t | F_t \sim \text{SN}(\Lambda F_t, \Sigma_e, \alpha)$$

where $\alpha$ is skewness parameter vector.

**Captures:**
- Negative skewness in returns
- Asymmetric responses to shocks
- Different behavior in expansions vs recessions

### Robust M-Estimation

**Huber Loss:**

Instead of maximizing Gaussian likelihood (quadratic loss), use robust loss:
$$\min_{\Lambda, F} \sum_{i,t} \rho\left(\frac{X_{it} - \lambda_i' F_t}{\sigma_i}\right)$$

where Huber loss function:
$$\rho(u) = \begin{cases}
\frac{1}{2} u^2 & |u| \leq k \\
k|u| - \frac{1}{2}k^2 & |u| > k
\end{cases}$$

**Properties:**
- Quadratic for small residuals (efficient)
- Linear for large residuals (bounded influence)
- Threshold $k$ controls robustness

---

## Common Pitfalls

### 1. Over-Estimating Tail Thickness
- **Mistake**: Fitting very low df ($\nu < 4$) to data with mild outliers
- **Result**: Model too conservative, poor fit to bulk of data
- **Fix**: Check if df estimate is boundary value; consider mixture models

### 2. Ignoring Parameter Uncertainty in Robust Models
- **Mistake**: Using Student-t but assuming parameters known with certainty
- **Result**: Understate forecast uncertainty
- **Fix**: Bootstrap or Bayesian methods for parameter distributions

### 3. Confusing Fat Tails with Heteroskedasticity
- **Mistake**: Using Student-t when variance changes over time
- **Result**: Spurious heavy tails from volatility clustering
- **Fix**: Test for GARCH effects first; combine Student-t with stochastic volatility

### 4. Not Validating Robustness Gains
- **Mistake**: Assuming Student-t always better
- **Result**: Added complexity without performance improvement
- **Fix**: Compare out-of-sample forecasts and cross-validate

---

## Connections

- **Builds on:** Standard Gaussian factor models, EM algorithm
- **Leads to:** Stochastic volatility models, mixture models, regime-switching
- **Related to:** Robust regression, quantile regression, copula models

---

## Practice Problems

### Conceptual

1. Explain why Student-t distribution with low df automatically downweights outliers in EM algorithm.

2. How does kurtosis relate to degrees of freedom in Student-t? What df gives kurtosis of 6?

3. Compare Student-t factors vs Student-t idiosyncratic errors. Which matters more for robust loading estimation?

### Implementation

4. Extend `StudentTFactorModel` to allow different degrees of freedom for each variable (heterogeneous tail thickness).

5. Implement a two-component Gaussian mixture factor model and compare to Student-t on simulated data.

6. Create diagnostic plots: (a) QQ-plot of standardized residuals, (b) histogram with fitted Student-t overlay.

### Extension

7. Derive the posterior distribution $p(w_t | X_t, F_t, \nu)$ for the precision weight in Student-t mixture representation.

8. Research "multivariate Student-t distribution." How does it differ from univariate Student-t applied to each dimension?

9. Implement a simple Huber M-estimator for factor loadings and compare convergence to EM Student-t.

---

## Further Reading

- **Zhou, G., Liu, L. & Huang, J.Z.** (2009). "Robust factor analysis." *Computational Statistics & Data Analysis*, 53(12), 4026-4037.
  - M-estimation for robust factor analysis

- **Lucas, A., Koopman, S.J. & Klaassen, F.** (2006). "Forecasting volatility with fat-tailed distributions." *European Journal of Finance*, 12(1), 1-16.
  - Student-t GARCH models

- **Peel, D. & McLachlan, G.J.** (2000). "Robust mixture modelling using the t distribution." *Statistics and Computing*, 10(4), 339-348.
  - Mixture of t-distributions for clustering

- **Mancini, L., Ronchetti, E. & Trojani, F.** (2005). "Optimal conditionally unbiased bounded-influence inference in dynamic location and scale models." *Journal of the American Statistical Association*, 100(470), 628-641.
  - Robust time series estimation theory

- **Oh, D.H. & Patton, A.J.** (2017). "Modeling dependence in high dimensions with factor copulas." *Journal of Business & Economic Statistics*, 35(1), 139-154.
  - Copula-based factor models for flexible dependence

- **Lange, K.L., Little, R.J. & Taylor, J.M.** (1989). "Robust statistical modeling using the t distribution." *Journal of the American Statistical Association*, 84(408), 881-896.
  - Classic reference on EM for Student-t models

---

## Summary

**Key Takeaways:**

1. **Financial and economic data exhibit heavy tails and asymmetry** that Gaussian models misrepresent
2. **Student-t factor models** provide robust estimation via automatic outlier downweighting through latent precision weights
3. **EM algorithm** treats Student-t as Gaussian scale mixture, leading to weighted least squares with adaptive weights
4. **Degrees of freedom parameter** controls tail thickness; low df (4-6) common in financial applications
5. **Alternative specifications** include mixture of Gaussians (multimodality), skew distributions (asymmetry), and M-estimators (bounded influence)

**Next Steps:**

The final guide explores connections between factor models and machine learning—particularly autoencoders, which provide nonlinear generalizations of PCA, and neural network approaches to factor extraction. We'll see how deep learning both extends and challenges traditional factor modeling.
