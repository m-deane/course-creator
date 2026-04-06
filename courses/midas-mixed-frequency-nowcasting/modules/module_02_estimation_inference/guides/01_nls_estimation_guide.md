# Non-Linear Least Squares Estimation for MIDAS

> **Reading time:** ~17 min | **Module:** 02 — Estimation Inference | **Prerequisites:** Module 1


## In Brief

<div class="flow">
<div class="flow-step mint">1. Set Starting Values</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">2. NLS Optimization</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">3. Check Convergence</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. Validate Residuals</div>
</div>


<div class="callout-key">

**Key Concept Summary:** MIDAS regression with parameterized weight functions is nonlinear in parameters — the weights $w_j(\theta)$ depend nonlinearly on $\theta$. This requires non-linear least squares (NLS) rather than ...

</div>

MIDAS regression with parameterized weight functions is nonlinear in parameters — the weights $w_j(\theta)$ depend nonlinearly on $\theta$. This requires non-linear least squares (NLS) rather than OLS. Understanding NLS optimization, convergence, and the implied covariance matrix is essential for valid inference.

## Key Insight

<div class="callout-insight">

**Insight:** Convergence failures in NLS estimation are often a signal of model misspecification, not just bad starting values. If the optimizer struggles, simplify the weight function before increasing iterations.

</div>


NLS for MIDAS minimizes the sum of squared residuals over $(α, β, θ)$ jointly. Because the model is nonlinear only through $θ$, a profile likelihood approach can reduce the optimization to a 2-dimensional search over $θ$ alone, with $(α, β)$ solved analytically at each $θ$.

---

## NLS Setup

<div class="callout-warning">

**Warning:** Be cautious about extrapolating MIDAS performance from stable periods to crisis periods. The relationship between high-frequency indicators and the low-frequency target can shift dramatically during regime changes.

</div>


The MIDAS objective function is:

$$Q(α, β, θ) = \sum_{t=1}^T \left(y_t - α - β \underbrace{\sum_{j=0}^{K-1} w_j(θ) x_{mt-j}}_{\tilde{x}_t(θ)}\right)^2$$

This can be written as:

$$Q(α, β, θ) = \|\mathbf{y} - α\mathbf{1} - β\tilde{\mathbf{x}}(θ)\|^2$$

where $\tilde{\mathbf{x}}(θ) = \mathbf{X} \mathbf{w}(θ)$ is the weighted aggregate of the MIDAS regressors.

### The Profile Likelihood Approach

For fixed $θ$, $(α, β)$ are determined by OLS on the weighted aggregate:

$$(\hat{α}(θ), \hat{β}(θ)) = \arg\min_{α,β} \|y - α - β\tilde{x}(θ)\|^2$$

The solution is:

$$\hat{β}(θ) = \frac{\sum_t (y_t - \bar{y})(\tilde{x}_t(θ) - \bar{\tilde{x}}(θ))}{\sum_t (\tilde{x}_t(θ) - \bar{\tilde{x}}(θ))^2}$$

$$\hat{α}(θ) = \bar{y} - \hat{β}(θ) \bar{\tilde{x}}(θ)$$

Substituting back, the profiled objective is purely a function of $θ$:

$$Q_{\text{profile}}(θ) = \|\mathbf{M}(θ)\mathbf{y}\|^2$$

where $\mathbf{M}(θ) = \mathbf{I} - \mathbf{P}_{[\mathbf{1}, \tilde{\mathbf{x}}(θ)]}$ is the annihilator of the design matrix at $θ$.

This reduces the problem from optimizing over 4 parameters to optimizing over 2 ($θ_1, θ_2$).


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

def profile_sse(theta, Y, X, weight_fn):
    """
    Profile SSE: optimize out (alpha, beta) analytically, optimize only theta.

    Parameters
    ----------
    theta : array-like [theta1, theta2]
    Y : np.ndarray, shape (T,)
    X : np.ndarray, shape (T, K)
    weight_fn : callable(K, theta1, theta2) -> np.ndarray shape (K,)

    Returns
    -------
    profile_sse : float
    """
    t1, t2 = theta
    K = X.shape[1]

    # Compute weights
    w = weight_fn(K, t1, t2)
    if not np.all(np.isfinite(w)):
        return 1e10

    # Weighted aggregate
    x_w = X @ w  # Shape (T,)

    # OLS: analytical solution for (alpha, beta)
    x_centered = x_w - x_w.mean()
    y_centered = Y - Y.mean()

    ss_x = np.dot(x_centered, x_centered)
    if ss_x < 1e-12:
        return np.sum((Y - Y.mean())**2)  # Return total SS if no variation

    beta = np.dot(y_centered, x_centered) / ss_x
    alpha = Y.mean() - beta * x_w.mean()

    # Residuals
    residuals = Y - alpha - beta * x_w
    return np.sum(residuals**2)


def estimate_midas_profile(Y, X, weight_fn, theta0=(1.0, 5.0)):
    """
    Estimate MIDAS by profile NLS.

    Uses 2D optimization over theta, with (alpha, beta) solved analytically.
    More numerically stable than joint 4D optimization.

    Returns
    -------
    dict with alpha, beta, theta1, theta2, weights, fitted, residuals, r2
    """
    K = X.shape[1]

    # Grid search for robust starting values
    best_sse = np.inf
    best_theta = theta0

    for t1 in [0.5, 1.0, 1.5, 2.0]:
        for t2 in [1.0, 3.0, 5.0, 8.0]:
            try:
                sse = profile_sse([t1, t2], Y, X, weight_fn)
                if sse < best_sse:
                    best_sse = sse
                    best_theta = [t1, t2]
            except Exception:
                pass

    # Fine optimization from best grid point
    result = minimize(
        profile_sse,
        best_theta,
        args=(Y, X, weight_fn),
        method='Nelder-Mead',
        options={'maxiter': 20000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
    )

    theta1_hat, theta2_hat = result.x
    theta1_hat = max(theta1_hat, 0.01)
    theta2_hat = max(theta2_hat, 0.01)

    # Recover (alpha, beta) at optimal theta
    w = weight_fn(K, theta1_hat, theta2_hat)
    x_w = X @ w

    lr = LinearRegression().fit(x_w.reshape(-1, 1), Y)
    alpha_hat = lr.intercept_
    beta_hat = lr.coef_[0]

    fitted = alpha_hat + beta_hat * x_w
    residuals = Y - fitted
    r2 = 1 - np.sum(residuals**2) / np.sum((Y - Y.mean())**2)

    return {
        'alpha': alpha_hat, 'beta': beta_hat,
        'theta1': theta1_hat, 'theta2': theta2_hat,
        'weights': w, 'fitted': fitted,
        'residuals': residuals, 'r2': r2,
        'sse': result.fun, 'converged': result.success
    }
```

</div>
</div>

---

## Asymptotic Theory

Under standard regularity conditions (Gallant 1987, Wooldridge 1994), the NLS estimator $\hat{θ}_{\text{NLS}}$ satisfies:

$$\sqrt{T}\left(\hat{θ}_{\text{NLS}} - θ^*\right) \xrightarrow{d} \mathcal{N}\left(0, \sigma^2 \mathbf{Q}^{-1}\right)$$

where:
- $θ^* = (α^*, β^*, θ_1^*, θ_2^*)$ are the true parameters
- $\sigma^2 = \text{Var}(\varepsilon_t)$ is the error variance
- $\mathbf{Q} = \text{plim}(T^{-1} \nabla_θ \hat{\mathbf{f}}^\top \nabla_θ \hat{\mathbf{f}})$ is the Hessian of the objective

In practice, $\mathbf{Q}$ is estimated by the numerical Hessian at convergence.

### Standard Errors

The sandwich covariance matrix for HAC-robust inference:

$$\text{Var}(\hat{θ}) = T^{-1} \hat{\mathbf{Q}}^{-1} \hat{\mathbf{S}} \hat{\mathbf{Q}}^{-1}$$

where $\hat{\mathbf{S}}$ is the Newey-West estimator of the long-run variance of the gradient.


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import inv

def nls_covariance(theta_hat, Y, X, objective_fn, eps=1e-4):
    """
    Estimate covariance matrix of NLS estimator using numerical Hessian.

    Parameters
    ----------
    theta_hat : np.ndarray
        Estimated parameters [alpha, beta, theta1, theta2].
    Y, X : np.ndarray
        Data.
    objective_fn : callable
        The NLS objective function (returns SSE).
    eps : float
        Step size for numerical differentiation.

    Returns
    -------
    vcov : np.ndarray, shape (4, 4)
        Covariance matrix of parameter estimates.
    se : np.ndarray, shape (4,)
        Standard errors.
    """
    T = len(Y)
    k = len(theta_hat)
    H = np.zeros((k, k))  # Numerical Hessian

    f0 = objective_fn(theta_hat, Y, X)

    for i in range(k):
        for j in range(k):
            theta_pp = theta_hat.copy()
            theta_pm = theta_hat.copy()
            theta_mp = theta_hat.copy()
            theta_mm = theta_hat.copy()

            theta_pp[i] += eps; theta_pp[j] += eps
            theta_pm[i] += eps; theta_pm[j] -= eps
            theta_mp[i] -= eps; theta_mp[j] += eps
            theta_mm[i] -= eps; theta_mm[j] -= eps

            H[i, j] = (objective_fn(theta_pp, Y, X)
                       - objective_fn(theta_pm, Y, X)
                       - objective_fn(theta_mp, Y, X)
                       + objective_fn(theta_mm, Y, X)) / (4 * eps**2)

    # Estimate sigma^2 from residuals
    # (Need to evaluate residuals at theta_hat)
    sigma2 = objective_fn(theta_hat, Y, X) / (T - k)

    try:
        Q_inv = inv(H / T)
        vcov = (sigma2 / T) * Q_inv
        se = np.sqrt(np.diag(vcov))
    except np.linalg.LinAlgError:
        vcov = np.full((k, k), np.nan)
        se = np.full(k, np.nan)

    return vcov, se
```

</div>
</div>

---

## Convergence Diagnostics

### When NLS Fails to Converge

Common causes and remedies:

**1. Poor starting values:** The NLS landscape has flat regions where gradients are near zero. Solution: grid search over a coarse parameter grid before fine optimization.

**2. Near-zero theta:** Beta parameters near 0 produce degenerate weight distributions. Solution: bound $\theta_i > 0.01$ or switch to Almon polynomial with unconstrained parameters.

**3. Multicollinearity in MIDAS matrix:** Adjacent monthly observations are highly correlated, making the Hessian near-singular. Solution: increase the number of lags so the spread of lags reduces correlation, or use ridge regularization.

**4. Objective function not smooth:** Numerical issues in beta PDF evaluation at extreme parameters. Solution: use midpoint evaluation $(j+0.5)/K$ and ensure $\theta > 0$.


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
def check_convergence(result, theta_hat, Y, X, objective_fn):
    """
    Post-estimation convergence check.

    Parameters
    ----------
    result : scipy.OptimizeResult
    theta_hat : array
        Final parameter estimates.
    Y, X : data arrays

    Returns
    -------
    diagnostics : dict
    """
    K = X.shape[1]
    T = len(Y)

    # 1. Optimizer convergence flag
    converged = result.success

    # 2. Gradient norm at solution (should be near zero)
    eps = 1e-6
    k = len(theta_hat)
    grad = np.zeros(k)
    f0 = objective_fn(theta_hat, Y, X)
    for i in range(k):
        theta_p = theta_hat.copy(); theta_p[i] += eps
        grad[i] = (objective_fn(theta_p, Y, X) - f0) / eps
    grad_norm = np.linalg.norm(grad)

    # 3. Check if at boundary
    at_boundary = any(abs(theta_hat[2:] - 0.01) < 0.01)  # theta params near lower bound

    # 4. Compare to alternative starting values
    f_at_uniform = objective_fn([Y.mean(), 0.1, 1.0, 1.0], Y, X)
    improvement = (f_at_uniform - result.fun) / f_at_uniform

    diagnostics = {
        'converged': converged,
        'final_sse': result.fun,
        'gradient_norm': grad_norm,
        'at_boundary': at_boundary,
        'improvement_over_uniform': improvement,
        'gradient_ok': grad_norm < 1e-3,
    }

    print("Convergence Diagnostics:")
    print(f"  Optimizer converged: {converged}")
    print(f"  Gradient norm: {grad_norm:.6f} ({'OK' if grad_norm < 1e-3 else 'LARGE — may not be at optimum'})")
    print(f"  Parameters at boundary: {at_boundary}")
    print(f"  SSE improvement over uniform: {improvement:.1%}")

    return diagnostics
```

</div>
</div>

---

## Testing the Polynomial Restriction

A key hypothesis test: is the equal-weight restriction rejected by the data?

**Null hypothesis:** $H_0: \theta_1 = \theta_2 = 1$ (Beta(1,1) = uniform = equal-weight aggregation)

**Alternative:** $H_1: (\theta_1, \theta_2) \neq (1, 1)$

Test using a Wald-type statistic or a likelihood ratio test (F-test):

$$F = \frac{(SSE_{\text{restricted}} - SSE_{\text{unrestricted}})/r}{SSE_{\text{unrestricted}}/(T - k)}$$

where $r = 2$ (two restrictions: $\theta_1 = 1$, $\theta_2 = 1$) and $k = 4$ (full model parameters).

```python
def test_aggregation_restriction(Y, X, beta_weights_fn, midas_result):
    """
    Test the null hypothesis of equal-weight aggregation.

    H0: Beta(1,1) weights (= equal-weight aggregation)
    H1: Optimal Beta(theta1, theta2) weights

    Returns F-statistic and p-value.
    """
    from scipy.stats import f as f_dist
    from sklearn.linear_model import LinearRegression

    T = len(Y)

    # Restricted model: Beta(1,1) = equal-weight OLS
    w_restricted = beta_weights_fn(X.shape[1], 1.0, 1.0)
    x_r = X @ w_restricted
    lr = LinearRegression().fit(x_r.reshape(-1,1), Y)
    sse_restricted = np.sum((Y - lr.predict(x_r.reshape(-1,1)))**2)

    # Unrestricted model: optimal MIDAS
    sse_unrestricted = midas_result['sse']

    r = 2   # Number of restrictions (theta1 = theta2 = 1)
    k = 4   # Parameters in unrestricted model
    df1 = r
    df2 = T - k

    F_stat = ((sse_restricted - sse_unrestricted) / r) / (sse_unrestricted / df2)
    p_value = 1 - f_dist.cdf(F_stat, df1, df2)

    print(f"\nTest of Equal-Weight Restriction (H0: Beta(1,1)):")
    print(f"  SSE restricted (OLS):     {sse_restricted:.4f}")
    print(f"  SSE unrestricted (MIDAS): {sse_unrestricted:.4f}")
    print(f"  F({df1}, {df2}) = {F_stat:.4f}")
    print(f"  p-value = {p_value:.4f}")
    if p_value < 0.05:
        print(f"  Conclusion: Reject H0 at 5% — MIDAS weights significantly differ from uniform.")
    else:
        print(f"  Conclusion: Fail to reject H0 — equal-weight aggregation not significantly worse.")

    return F_stat, p_value
```

---

## Common Pitfalls

**Pitfall 1: Ignoring NLS convergence.** Always check `result.success` and the gradient norm. A converged flag with a large gradient suggests a local minimum.

**Pitfall 2: Single-start optimization.** NLS with a single starting point may converge to a local rather than global minimum. Always try multiple starting values and report the one with lowest SSE.

**Pitfall 3: OLS standard errors for NLS.** The OLS covariance formula $\sigma^2 (\mathbf{X}^\top\mathbf{X})^{-1}$ does not apply to NLS parameters $(\alpha, \beta, \theta)$. Use the sandwich form with the numerical Hessian.

**Pitfall 4: Standard errors for $\hat{\theta}$ only.** The inference on $\theta$ tells you about the weight function shape. But for forecast evaluation, inference on $\hat{\beta}$ (the scale coefficient) is typically more relevant. Ensure you extract both sets of standard errors.

---

## Connections

- **Builds on:** Module 01 (MIDAS equation and weight functions)
- **Leads to:** Guide 02 (model selection), Guide 03 (robust inference), Module 03 (nowcasting)
- **Related to:** NLS in econometrics (Wooldridge 2010 Chapter 12), method of moments

---

## Practice Problems

<div class="callout-danger">

**Danger:** Never use future information when constructing the high-frequency regressor matrix. In a real-time nowcasting context, you only have data up to the current date -- using the full quarter of monthly data when nowcasting mid-quarter is a look-ahead bias that invalidates your results.

</div>


1. Show that when the weight function is flat ($w_j = 1/K$ for all $j$), the MIDAS NLS estimator reduces to OLS on the equal-weight aggregate. In particular, show that $\nabla_\theta Q = 0$ at $\theta_1 = \theta_2 = 1$ is not generally satisfied.

2. Implement the profile NLS approach in Python and verify that it produces the same parameter estimates as joint 4D optimization. Compare convergence time for the two approaches.

3. The F-test for the equal-weight restriction has how many numerator degrees of freedom? Why?

---

## Further Reading

- Gallant, A. R. (1987). *Nonlinear Statistical Models.* Wiley.
- Wooldridge, J. M. (1994). "A simple specification test for the predictive ability of transformation models." *Review of Economics and Statistics.*
- Ghysels, E., & Qian, H. (2019). "Estimating MIDAS regressions via OLS with polynomial parameter profiling." *Econometrics and Statistics*, 9, 1–16. [Profile OLS paper]


---

## Cross-References

<a class="link-card" href="./02_model_selection_guide.md">
  <div class="link-card-title">02 Model Selection</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_model_selection_slides.md">
  <div class="link-card-title">02 Model Selection — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./03_inference_guide.md">
  <div class="link-card-title">03 Inference</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_inference_slides.md">
  <div class="link-card-title">03 Inference — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

