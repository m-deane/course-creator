---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# NLS Estimation for MIDAS

## Non-Linear Least Squares: Theory and Practice

**Mixed-Frequency Models: MIDAS Regression and Nowcasting**
Module 02 — Guide 01

<!-- Speaker notes: This guide covers the estimation machinery behind MIDAS. Students coming from linear regression need to understand why OLS can't be used directly, how NLS works, and what convergence diagnostics to check. The profile likelihood approach is the key practical insight — it reduces a 4D optimization to 2D, which is much more reliable. -->

---

## Why OLS Doesn't Work Directly

**MIDAS model:**
$$y_t = \alpha + \beta \sum_{j=0}^{K-1} w_j(\theta) x_{mt-j} + \varepsilon_t$$

**Problem:** $w_j(\theta)$ depends nonlinearly on $\theta$.

- If $\theta$ is known: OLS on $\tilde{x}_t(\theta) = \sum_j w_j(\theta) x_{mt-j}$
- If $\theta$ is unknown: minimizing SSE over $(\alpha, \beta, \theta)$ jointly requires NLS

**OLS is a special case:** When $\theta$ is fixed at $(1,1)$ (uniform weights), MIDAS = OLS.

<!-- Speaker notes: The nonlinearity is entirely in theta. Given theta, the model is linear in alpha and beta. This structure is what enables the profile likelihood approach: we can solve analytically for alpha and beta at any theta, reducing the optimization to a 2D search over theta alone. This is much more numerically stable than optimizing all 4 parameters jointly. -->

<div class="callout-key">

The key advantage of MIDAS is preserving high-frequency information that temporal aggregation destroys.

</div>

---

## The NLS Objective

$$Q(\alpha, \beta, \theta) = \sum_{t=1}^T \left(y_t - \alpha - \beta \tilde{x}_t(\theta)\right)^2$$

where $\tilde{x}_t(\theta) = \mathbf{X}_t \mathbf{w}(\theta)$

**In matrix notation:**
$$Q = \|\mathbf{y} - \alpha\mathbf{1} - \beta\mathbf{X}\mathbf{w}(\theta)\|^2$$

**NLS estimator:**
$$(\hat{\alpha}, \hat{\beta}, \hat{\theta}) = \arg\min_{\alpha, \beta, \theta} Q(\alpha, \beta, \theta)$$

<!-- Speaker notes: The NLS objective is just the familiar sum of squared residuals, but with the twist that the regressors themselves depend on unknown parameters theta. This is what makes the problem nonlinear. The optimizer must simultaneously find the regression coefficients and the shape parameters of the weight function. Using Nelder-Mead or L-BFGS-B from scipy.optimize is the standard approach. -->

<div class="callout-insight">

**Insight:** Parsimonious weight functions with 2-3 parameters can capture decay patterns that unrestricted models need 12+ parameters to approximate.

</div>

---

## The Profile Likelihood Trick

**Key observation:** For fixed $\theta$, $(α, β)$ are solved by OLS.

$$\hat{\beta}(\theta) = \frac{\sum_t (y_t - \bar{y})(\tilde{x}_t(\theta) - \bar{\tilde{x}})}{\sum_t (\tilde{x}_t(\theta) - \bar{\tilde{x}})^2}, \quad \hat{\alpha}(\theta) = \bar{y} - \hat{\beta}(\theta)\bar{\tilde{x}}(\theta)$$

**Profile objective:**

$$Q_{\text{prof}}(\theta) = \min_{\alpha, \beta} Q(\alpha, \beta, \theta)$$

**4D optimization → 2D optimization over $\theta$ only.**

<!-- Speaker notes: The profile likelihood approach is the key practical insight of this guide. By recognizing that alpha and beta can be solved analytically given theta, we reduce the 4-parameter optimization to a 2-parameter optimization. This is dramatically more reliable — 2D optimization landscapes are much easier to explore than 4D ones, especially when starting values are uncertain. The Ghysels and Qian (2019) paper formalizes this approach as "OLS with polynomial parameter profiling." -->

<div class="callout-warning">

**Warning:** Always account for the real-time data vintage when evaluating nowcast performance. Using revised data overstates accuracy.

</div>

---

## Profile NLS: Implementation

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def profile_sse(theta, Y, X, weight_fn):
    """2D objective: optimize alpha, beta analytically."""
    t1, t2 = theta
    w = weight_fn(X.shape[1], t1, t2)
    x_w = X @ w          # weighted aggregate

    # Analytical OLS
    x_c = x_w - x_w.mean()
    y_c = Y - Y.mean()
    beta = np.dot(y_c, x_c) / np.dot(x_c, x_c)
    alpha = Y.mean() - beta * x_w.mean()

    return np.sum((Y - alpha - beta * x_w)**2)

# 2D optimization — much more reliable than 4D
result = minimize(profile_sse, [1.5, 4.0], args=(Y, X, beta_weights),
                  method='Nelder-Mead',
                  options={'maxiter': 20000, 'xatol': 1e-8})
```

</div>

<!-- Speaker notes: This is the code students will use in the notebook. The key point: minimize() only receives a 2-element theta vector. The optimizer never sees alpha and beta directly — they are computed inside profile_sse at each theta evaluation. This means the optimizer is working in a much cleaner, smoother landscape. Nelder-Mead is the default because it doesn't require gradient computation, but L-BFGS-B with numerical gradients also works well. -->

<div class="callout-info">

**Info:** MIDAS models can handle any frequency ratio: monthly-to-quarterly (3:1), daily-to-monthly (~22:1), or even tick-to-daily.

</div>

---

## Asymptotic Theory

Under standard regularity conditions:

$$\sqrt{T}\left(\hat{\theta}_{\text{NLS}} - \theta^*\right) \xrightarrow{d} \mathcal{N}\left(0,\, \sigma^2 \mathbf{Q}^{-1}\right)$$

where:
- $\mathbf{Q} = \text{plim}\!\left(\frac{1}{T}\mathbf{J}^\top\mathbf{J}\right)$ — Jacobian outer product
- $\mathbf{J}_{tj} = \partial f_t(\theta^*)/\partial \theta_j$ — gradient of fitted value w.r.t. $\theta$
- $\sigma^2$ — error variance

**Estimated covariance:** $\widehat{\text{Var}}(\hat{\theta}) = \hat{\sigma}^2 (\hat{\mathbf{J}}^\top \hat{\mathbf{J}})^{-1}$

<!-- Speaker notes: This is the standard NLS asymptotic result. The Jacobian J contains the derivatives of the fitted values with respect to each parameter. For MIDAS, the derivatives with respect to theta require differentiating the Beta PDF, which is not trivial analytically — in practice we use numerical differentiation. The key assumption is that the model is correctly specified (no restriction bias), the errors are homoscedastic, and the regressor process is stationary. Module 02 Guide 03 relaxes the homoscedasticity assumption using HAC standard errors. -->

---

## Numerical Hessian for Standard Errors

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def numerical_hessian(fn, theta, eps=1e-4):
    """Compute Hessian by finite differences."""
    k = len(theta)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            tp, tm = theta.copy(), theta.copy()
            tp[i] += eps; tp[j] += eps
            tm[i] -= eps; tm[j] -= eps
            t2 = theta.copy(); t2[i] += eps; t2[j] -= eps
            t3 = theta.copy(); t3[i] -= eps; t3[j] += eps
            H[i,j] = (fn(tp) - fn(t2) - fn(t3) + fn(tm)) / (4*eps**2)
    return H

# Standard errors from Hessian
H = numerical_hessian(lambda t: profile_sse(t, Y, X, beta_weights),
                       result.x)
sigma2 = result.fun / (len(Y) - 4)  # Estimated error variance
vcov = sigma2 * np.linalg.inv(H / len(Y))
se_theta = np.sqrt(np.diag(vcov))
```

</div>

<!-- Speaker notes: The numerical Hessian approach is standard when analytical derivatives are not available. The step size eps=1e-4 works well for most MIDAS applications. Larger eps introduces numerical error; smaller eps causes cancellation error. If the Hessian is singular (non-invertible), the model may be over-parameterized or at a flat region of the objective. In that case, the standard errors will be NaN — a warning that inference is unreliable at this parameter point. -->

---

## Testing the Uniform Weight Restriction

**Null hypothesis:** $H_0: \theta = (1, 1)$ — equal-weight aggregation is adequate

**F-test:**
$$F = \frac{(SSE_R - SSE_U) / r}{SSE_U / (T-k)} \sim F_{r, T-k}$$

where $r=2$ (restrictions), $k=4$ (unrestricted params), $T$ = sample size.

```python
SSE_R = sse_equal_weight_OLS   # Restricted (uniform weights)
SSE_U = midas_result['sse']    # Unrestricted (optimal theta)
F_stat = ((SSE_R - SSE_U)/2) / (SSE_U/(T-4))
from scipy.stats import f as f_dist
p_val = 1 - f_dist.cdf(F_stat, 2, T-4)
```

<!-- Speaker notes: The F-test has r=2 degrees of freedom in the numerator because we're testing two restrictions simultaneously: theta1 = 1 AND theta2 = 1. The F distribution under H0 relies on the standard NLS asymptotic normality. In finite samples, this test can be undersized or oversized depending on the degree of nonlinearity. For a more robust test, one can use a bootstrap p-value. The test is important because it provides formal justification for using MIDAS over simple aggregation. -->

---

## Convergence Diagnostics

Always verify NLS output before reporting results:

| Check | Good | Problem |
|-------|------|---------|
| `result.success` | `True` | `False` — try different method |
| Gradient norm $\|\nabla Q\|$ | $< 10^{-3}$ | Large — not at optimum |
| $\hat{\theta}_i$ at boundary | No | Yes — consider Almon |
| SSE $<$ naive baseline | Yes | No — model diverged |
| Multiple starts agree | Same $\hat{\theta}$ | Different — local optima |

<!-- Speaker notes: This checklist should be run mechanically after every MIDAS estimation. The most common failure mode is convergence to a local minimum, which manifests as the multiple-starts check failing (different starting points give different theta estimates). When this happens, expand the grid search to cover more of the parameter space. The theta-at-boundary warning is important for Beta polynomial: if theta1 or theta2 converges to near 0.01 (the lower bound), consider switching to Almon polynomial which has no positivity constraint. -->

---

## Multiple Starting Values: Implementation

```python
def estimate_midas_robust(Y, X, weight_fn):
    """NLS with grid search for starting values."""
    theta1_grid = [0.5, 1.0, 1.5, 2.0, 3.0]
    theta2_grid = [1.0, 3.0, 5.0, 8.0]

    results = []
    for t1 in theta1_grid:
        for t2 in theta2_grid:
            r = minimize(profile_sse, [t1, t2],
                         args=(Y, X, weight_fn),
                         method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-7})
            results.append((r.fun, r))

    # Select global minimum
    best = min(results, key=lambda x: x[0])[1]
    return best
```

Grid of $5 \times 4 = 20$ starting points ensures coverage of parameter space.

<!-- Speaker notes: The 20-point grid search is a practical default. For most MIDAS applications, the objective function is smooth enough that this grid covers the relevant parameter space. The runtime cost is 20× the single-start optimization — acceptable for model development but potentially slow for real-time updating in production systems. Module 08 (production systems) discusses how to cache optimal theta estimates and update them periodically rather than re-optimizing from scratch each period. -->

---

## When NLS Fails: Troubleshooting

**Problem 1:** `result.success = False`, large gradient

**Fix:** Increase `maxiter`, try `L-BFGS-B` method, tighten bounds

**Problem 2:** $\hat{\theta}$ at lower bound (near 0.01)

**Fix:** Switch to Almon polynomial (unconstrained params)

**Problem 3:** Multiple local optima (starts disagree)

**Fix:** Use differential evolution (global optimizer):
```python
from scipy.optimize import differential_evolution
result = differential_evolution(
    profile_sse, bounds=[(0.1, 8), (0.1, 15)],
    args=(Y, X, beta_weights),
    seed=42, maxiter=1000
)
```

<!-- Speaker notes: The troubleshooting guide is practical. Problem 1 (no convergence) is usually fixed by increasing maxiter or switching to a more robust optimizer. Problem 2 (boundary convergence) means the Beta polynomial family may not fit the data well — switch to Almon. Problem 3 (multiple optima) is the most concerning because it means the R² surface has multiple peaks. Differential evolution is a global optimizer that explores the entire feasible region but is slower than gradient-based methods. Use it as a last resort when multiple starts give inconsistent results. -->

---

## Summary

$$\hat{\theta}_{\text{NLS}} = \arg\min_\theta Q_{\text{profile}}(\theta) = \arg\min_\theta \|\mathbf{M}(\theta)\mathbf{y}\|^2$$

**Estimation protocol:**
1. Grid search over $(θ_1, θ_2)$ — 20 starting points
2. Nelder-Mead fine optimization from best grid point
3. Recover $(α, β)$ by OLS at $\hat{θ}$
4. Check convergence diagnostics
5. Compute standard errors from numerical Hessian

**Next:** Guide 02 — Model selection: how many lags, which weight function family.

<!-- Speaker notes: The estimation protocol is the practical workflow. It should become routine. The profile likelihood approach (step 1-3) is more reliable than joint 4D optimization. The convergence diagnostics (step 4) should never be skipped. Standard errors (step 5) are needed for inference in Module 02 Guide 03. The next guide covers how to choose the number of lags K and whether to use Beta vs. Almon vs. U-MIDAS — the model selection problem that completes the estimation workflow. -->
