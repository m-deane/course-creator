# Hypothesis Testing and Robust Inference for MIDAS

> **Reading time:** ~13 min | **Module:** 02 — Estimation Inference | **Prerequisites:** Module 1


## In Brief


<div class="callout-key">

**Key Concept Summary:** MIDAS regression residuals frequently exhibit serial correlation and heteroscedasticity. Standard OLS standard errors are invalid; HAC (Heteroscedasticity and Autocorrelation Consistent) standard e...

</div>

MIDAS regression residuals frequently exhibit serial correlation and heteroscedasticity. Standard OLS standard errors are invalid; HAC (Heteroscedasticity and Autocorrelation Consistent) standard errors due to Newey and West (1987) provide valid inference. Additionally, we test the significance of the high-frequency regressors and the polynomial weight restriction.

## Key Insight

<div class="callout-insight">

**Insight:** The mixed-frequency approach preserves within-period dynamics that aggregation destroys. This is especially valuable when the timing of high-frequency movements carries economic information.

</div>


The HAC covariance estimator corrects for both serial correlation and heteroscedasticity in the error process without assuming a specific parametric model for the error dynamics. For MIDAS, applying HAC to the linearized model (after profiling out theta) gives valid standard errors for the regression coefficients $(\alpha, \beta)$.

---

## Why Standard Errors Fail for MIDAS

<div class="callout-warning">

**Warning:** Be cautious about extrapolating MIDAS performance from stable periods to crisis periods. The relationship between high-frequency indicators and the low-frequency target can shift dramatically during regime changes.

</div>


Three potential violations of the OLS covariance assumptions:

**1. Serial Correlation in Residuals**

Quarterly GDP growth has mild autocorrelation (AR ≈ 0.3). Even after conditioning on IP, residuals may be serially correlated. OLS standard errors assume i.i.d. errors — serial correlation makes them too small (understates uncertainty).

**2. Heteroscedasticity**

Crisis periods (2008–2009, 2020) have dramatically larger residuals than tranquil periods. Heteroscedastic errors make OLS variance estimates biased.

**3. Generated Regressors**

The weighted aggregate $\tilde{x}_t(\hat{\theta}) = X_t w(\hat{\theta})$ uses estimated weights. This is a "generated regressor" — inference must account for the estimation uncertainty in $\hat{\theta}$.

---

## HAC Standard Errors: Theory

The long-run covariance matrix of the gradient process uses the Newey-West estimator:

$$\hat{\mathbf{S}}_{NW} = \hat{\mathbf{\Gamma}}_0 + \sum_{l=1}^{L} \left(1 - \frac{l}{L+1}\right)\left(\hat{\mathbf{\Gamma}}_l + \hat{\mathbf{\Gamma}}_l^\top\right)$$

where $\hat{\mathbf{\Gamma}}_l = T^{-1} \sum_{t=l+1}^T \hat{e}_t \mathbf{z}_t \hat{e}_{t-l} \mathbf{z}_{t-l}^\top$

and $L$ is the bandwidth (typically $L = \lfloor 4(T/100)^{2/9} \rfloor$ for quarterly data).

For MIDAS, apply HAC to the linearized model at the estimated $\hat{\theta}$:

**Linearized model:** $y_t = \alpha + \beta \tilde{x}_t(\hat{\theta}) + e_t$

**HAC covariance of $(\hat{\alpha}, \hat{\beta})$:**

$$\text{Var}_{HAC}(\hat{\alpha}, \hat{\beta}) = (Z^\top Z)^{-1} \hat{S}_{NW} (Z^\top Z)^{-1}$$

where $Z_t = (1, \tilde{x}_t(\hat{\theta}))$.


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np
import statsmodels.api as sm

def midas_hac_inference(Y, X, theta1_hat, theta2_hat, beta_weights_fn, n_lags=None):
    """
    Compute HAC standard errors for MIDAS regression coefficients.

    Parameters
    ----------
    Y : np.ndarray, shape (T,)
    X : np.ndarray, shape (T, K)
    theta1_hat, theta2_hat : float
        Estimated Beta polynomial parameters.
    beta_weights_fn : callable
    n_lags : int or None
        HAC bandwidth. If None, uses automatic Newey-West rule.

    Returns
    -------
    result : dict with keys: alpha, beta, se_alpha, se_beta, t_alpha, t_beta,
             p_alpha, p_beta, conf_alpha, conf_beta
    """
    T = len(Y)
    K = X.shape[1]

    # Compute optimal weights
    w_hat = beta_weights_fn(K, theta1_hat, theta2_hat)

    # Weighted aggregate (the linearized regressor)
    x_tilde = X @ w_hat

    # OLS estimates
    Z = np.column_stack([np.ones(T), x_tilde])
    params = np.linalg.lstsq(Z, Y, rcond=None)[0]
    alpha_hat, beta_hat = params

    # Residuals
    residuals = Y - Z @ params

    # HAC standard errors via statsmodels
    if n_lags is None:
        # Newey-West automatic bandwidth rule
        n_lags = int(4 * (T / 100) ** (2/9))

    ols_model = sm.OLS(Y, Z)
    ols_result = ols_model.fit(cov_type='HAC', cov_kwds={'maxlags': n_lags})

    from scipy.stats import t as t_dist
    se_alpha = ols_result.bse[0]
    se_beta = ols_result.bse[1]

    t_alpha = alpha_hat / se_alpha
    t_beta = beta_hat / se_beta

    df = T - 2
    p_alpha = 2 * (1 - t_dist.cdf(abs(t_alpha), df))
    p_beta = 2 * (1 - t_dist.cdf(abs(t_beta), df))

    # 95% confidence intervals
    t_crit = t_dist.ppf(0.975, df)
    ci_alpha = (alpha_hat - t_crit * se_alpha, alpha_hat + t_crit * se_alpha)
    ci_beta = (beta_hat - t_crit * se_beta, beta_hat + t_crit * se_beta)

    print("MIDAS Regression Results (HAC Standard Errors)")
    print("=" * 55)
    print(f"{'Parameter':<15} {'Estimate':>10} {'HAC SE':>8} {'t-stat':>8} {'p-value':>8}")
    print("-" * 55)
    print(f"{'alpha'::<15} {alpha_hat:>10.4f} {se_alpha:>8.4f} {t_alpha:>8.3f} {p_alpha:>8.4f}")
    print(f"{'beta (IP)'::<15} {beta_hat:>10.4f} {se_beta:>8.4f} {t_beta:>8.3f} {p_beta:>8.4f}")
    print()
    print(f"HAC bandwidth: {n_lags} lags")
    print(f"N = {T}, R² = {1 - np.sum(residuals**2)/np.sum((Y-Y.mean())**2):.4f}")

    return {
        'alpha': alpha_hat, 'beta': beta_hat,
        'se_alpha': se_alpha, 'se_beta': se_beta,
        't_alpha': t_alpha, 't_beta': t_beta,
        'p_alpha': p_alpha, 'p_beta': p_beta,
        'ci_alpha': ci_alpha, 'ci_beta': ci_beta,
        'residuals': residuals,
        'n_lags_hac': n_lags
    }
```

</div>

---

## Key Hypothesis Tests

### Test 1: Significance of High-Frequency Regressor

$$H_0: \beta = 0 \quad \text{(IP has no predictive power for GDP)}$$

$$t = \frac{\hat{\beta}}{\text{se}_{HAC}(\hat{\beta})} \sim t_{T-2} \text{ approximately}$$

Reject at 5% if $|t| > 1.96$ (or exact $t_{T-2}$ critical value).

### Test 2: Joint Significance of All Regressors

$$H_0: \alpha = \beta = 0$$

Wald statistic:
$$W = \hat{\boldsymbol{\delta}}^\top \left[\widehat{\text{Var}}_{HAC}(\hat{\boldsymbol{\delta}})\right]^{-1} \hat{\boldsymbol{\delta}} \sim \chi^2_2$$

where $\hat{\boldsymbol{\delta}} = (\hat{\alpha}, \hat{\beta})^\top$.

### Test 3: Equal-Weight Restriction

$$H_0: w_j(\hat{\theta}) = 1/K \text{ for all } j \quad (\text{i.e., } \theta_1 = \theta_2 = 1)$$

F-test comparing restricted (OLS-aggregate) to unrestricted (MIDAS) SSE:

$$F = \frac{(SSE_R - SSE_U) / 2}{SSE_U / (T-4)} \sim F_{2, T-4}$$


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def test_equal_weights(Y, X, midas_result, beta_weights_fn):
    """F-test for equal-weight restriction H0: theta = (1, 1)."""
    from scipy.stats import f as f_dist
    from sklearn.linear_model import LinearRegression

    T = len(Y)
    K = X.shape[1]

    # Restricted: equal-weight OLS
    w_r = beta_weights_fn(K, 1.0, 1.0)
    xw_r = X @ w_r
    lr = LinearRegression().fit(xw_r.reshape(-1,1), Y)
    sse_r = np.sum((Y - lr.predict(xw_r.reshape(-1,1)))**2)

    # Unrestricted
    sse_u = midas_result['sse']

    r, k = 2, 4  # Restrictions, unrestricted params
    F = ((sse_r - sse_u) / r) / (sse_u / (T - k))
    p_val = 1 - f_dist.cdf(F, r, T - k)

    print(f"\nF-test: H0 = equal-weight aggregation (Beta(1,1))")
    print(f"  SSE restricted (OLS): {sse_r:.4f}")
    print(f"  SSE unrestricted:     {sse_u:.4f}")
    print(f"  F({r}, {T-k}) = {F:.4f}, p = {p_val:.4f}")
    conclusion = "Reject H0" if p_val < 0.05 else "Fail to reject H0"
    print(f"  Conclusion: {conclusion} at 5%")

    return F, p_val
```

</div>

---

## Bootstrap Standard Errors (Alternative)

For small samples or when asymptotic theory is unreliable, bootstrap inference:


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def bootstrap_midas(Y, X, beta_weights_fn, n_bootstrap=499, seed=42):
    """
    Bootstrap standard errors for MIDAS parameters.

    Uses residual resampling (fixed design bootstrap) to avoid
    bootstrap breakdown under misspecification.

    Returns
    -------
    boot_dist : np.ndarray, shape (n_bootstrap, 4)
        Bootstrap distribution of (alpha, beta, theta1, theta2).
    """
    from scipy.optimize import minimize
    from sklearn.linear_model import LinearRegression

    np.random.seed(seed)
    T = len(Y)
    K = X.shape[1]

    def estimate_one(Y_b, X_b):
        """Estimate MIDAS on one bootstrap sample."""
        def profile_sse(theta, Y, X):
            t1, t2 = theta
            if t1 <= 0 or t2 <= 0: return 1e10
            w = beta_weights_fn(X.shape[1], t1, t2)
            xw = X @ w
            xc = xw - xw.mean(); yc = Y - Y.mean()
            ss = np.dot(xc, xc)
            if ss < 1e-12: return np.sum((Y - Y.mean())**2)
            b = np.dot(yc, xc) / ss
            a = Y.mean() - b * xw.mean()
            return np.sum((Y - a - b * xw)**2)

        best_sse = np.inf; best = None
        for t1, t2 in [(1.0, 5.0), (1.5, 4.0)]:
            r = minimize(profile_sse, [t1, t2], args=(Y_b, X_b),
                         method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-6})
            if r.fun < best_sse:
                best_sse = r.fun; best = r

        t1_h, t2_h = max(best.x[0], 0.01), max(best.x[1], 0.01)
        w = beta_weights_fn(K, t1_h, t2_h)
        xw = X_b @ w
        lr = LinearRegression().fit(xw.reshape(-1,1), Y_b)
        return [lr.intercept_, lr.coef_[0], t1_h, t2_h]

    # Get residuals from original fit
    w_orig = beta_weights_fn(K, 1.5, 4.0)  # Approximate — use actual estimated params
    xw_orig = X @ w_orig
    lr_orig = LinearRegression().fit(xw_orig.reshape(-1,1), Y)
    resid_orig = Y - lr_orig.predict(xw_orig.reshape(-1,1))
    fitted_orig = lr_orig.predict(xw_orig.reshape(-1,1))

    # Residual bootstrap
    boot_dist = np.zeros((n_bootstrap, 4))
    for b in range(n_bootstrap):
        # Resample residuals and add back to fitted values
        resid_b = np.random.choice(resid_orig, size=T, replace=True)
        Y_b = fitted_orig + resid_b
        boot_dist[b] = estimate_one(Y_b, X)

        if (b + 1) % 100 == 0:
            print(f"  Bootstrap: {b+1}/{n_bootstrap}")

    return boot_dist


def bootstrap_se(boot_dist, param_names=None):
    """Summarize bootstrap distribution."""
    if param_names is None:
        param_names = ['alpha', 'beta', 'theta1', 'theta2']

    print("\nBootstrap Standard Errors:")
    print(f"{'Parameter':<12} {'Mean':>10} {'Std':>10} {'2.5%':>10} {'97.5%':>10}")
    print("-" * 55)
    for i, name in enumerate(param_names):
        vals = boot_dist[:, i]
        ci = np.percentile(vals, [2.5, 97.5])
        print(f"{name:<12} {vals.mean():>10.4f} {vals.std():>10.4f} {ci[0]:>10.4f} {ci[1]:>10.4f}")
```

</div>

---

## Common Pitfalls

**Pitfall 1: Using OLS standard errors for MIDAS.** The `LinearRegression` fitted at the optimal theta does not account for either the HAC correction or the estimation uncertainty in theta. Always compute HAC standard errors explicitly.

**Pitfall 2: Not reporting p-values for the equal-weight test.** The F-test for equal weights is the most important diagnostic for MIDAS — it answers "does the polynomial restriction buy anything over simple aggregation?" Always report it.

**Pitfall 3: Bootstrap with too few replications.** For 95% confidence intervals, at least 999 bootstrap replications are needed for stable quantile estimates. For 5th percentile estimates, even more are needed.

---

## Connections

<div class="callout-danger">

**Danger:** Never use future information when constructing the high-frequency regressor matrix. In a real-time nowcasting context, you only have data up to the current date -- using the full quarter of monthly data when nowcasting mid-quarter is a look-ahead bias that invalidates your results.

</div>


- **Builds on:** Guide 01 (NLS estimation), Guide 02 (model selection)
- **Leads to:** Module 03 (nowcasting with valid inference), Module 04 (DFM inference)
- **Related to:** HAC estimators (Newey-West 1987), bootstrap for time series

---

## Practice Problems

1. A MIDAS residual series has autocorrelation $\hat{\rho}_1 = 0.25$ at lag 1. How does this affect OLS standard errors relative to true standard errors? (Hint: derive the approximate inflation factor under AR(1) errors.)

2. Using the Newey-West bandwidth rule $L = \lfloor 4(T/100)^{2/9}\rfloor$, compute $L$ for $T = 80$, $T = 150$, and $T = 400$.

3. Why does the residual bootstrap preserve the time ordering of observations implicitly in the MIDAS context? When would the block bootstrap be preferred?


---

## Cross-References

<a class="link-card" href="./01_nls_estimation_guide.md">
  <div class="link-card-title">01 Nls Estimation</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_nls_estimation_slides.md">
  <div class="link-card-title">01 Nls Estimation — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./02_model_selection_guide.md">
  <div class="link-card-title">02 Model Selection</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_model_selection_slides.md">
  <div class="link-card-title">02 Model Selection — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

