# RDD Bandwidth Selection and Sensitivity Analysis

> **Reading time:** ~7 min | **Module:** 5 — Regression Discontinuity | **Prerequisites:** Module 0 — Causal Foundations

## Learning Objectives

By the end of this guide, you will be able to:
1. Explain the bias-variance tradeoff underlying bandwidth selection in RDD
2. Implement the Imbens-Kalyanaraman (IK) MSE-optimal bandwidth
3. Conduct bandwidth sensitivity analysis and interpret the results
4. Choose between linear and higher-order polynomial fits
5. Communicate RDD robustness to a general audience

---

## 1. Why Bandwidth Matters More Than Anything Else

In regression discontinuity, the bandwidth $h$ defines which observations are used in estimation. Everything else — polynomial order, kernel choice, standard error clustering — is secondary.

The tension is fundamental:

**Narrow bandwidth (small $h$):**
- Observations are very close to the cutoff → plausible local continuity
- Few observations → high sampling variance → wide confidence intervals
- Estimates are noisy

**Wide bandwidth (large $h$):**
- More observations → lower variance → precise estimates
- Observations far from cutoff → the linear fit may miss nonlinear patterns → bias
- Estimates may be far from the true local effect

The **optimal bandwidth** minimises the sum of squared bias and variance — the mean squared error (MSE).

---

## 2. The Imbens-Kalyanaraman (IK) Optimal Bandwidth

The asymptotic MSE of the local linear RDD estimator is:

$$MSE(h) \approx \frac{\sigma^2}{n h f(c)} + \frac{h^4}{4} B^2$$

where:
- $\sigma^2$ = outcome variance near the cutoff
- $f(c)$ = density of the running variable at the cutoff
- $B$ = the second derivative of the regression function (captures curvature/bias)
- $n$ = sample size

Setting $\partial MSE / \partial h = 0$ gives the optimal bandwidth:

$$h_{IK} = \left(\frac{\sigma^2}{n f(c) B^2}\right)^{1/5}$$

This scales as $n^{-1/5}$ — bandwidth shrinks as sample size grows.

### Practical Implementation

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>

```python
import numpy as np
import pandas as pd
from rdrobust import rdrobust, rdbwselect

# Generate RDD data
np.random.seed(42)
n = 1000
x = np.random.uniform(-1, 1, n)
y = (2 + 3*x + 0.5*x**2          # control side
     + 1.5*(x >= 0)               # treatment effect at cutoff
     + 0.3*(x >= 0)*x             # slope change
     + np.random.normal(0, 0.5, n))

df = pd.DataFrame({'x': x, 'y': y, 'treated': (x >= 0).astype(int)})

# IK bandwidth selection
bw_result = rdbwselect(y=df['y'], x=df['x'], c=0, bwselect='mserd')
h_ik = bw_result.bws['h'][0]
print(f"IK optimal bandwidth: {h_ik:.4f}")
print(f"Obs within bandwidth: {(np.abs(df['x']) <= h_ik).sum()}")

# Estimate with IK bandwidth
rdd_result = rdrobust(y=df['y'], x=df['x'], c=0, h=h_ik)
print(f"Treatment effect estimate: {rdd_result.coef[0]:.3f}")
print(f"95% CI: [{rdd_result.ci[0]:.3f}, {rdd_result.ci[1]:.3f}]")
```

</div>

### Alternative Bandwidth Selectors

| Method | Description | Use When |
|--------|-------------|---------|
| `mserd` | MSE-optimal, same bandwidth each side | Default choice |
| `msetwo` | MSE-optimal, different bandwidths each side | Asymmetric density around cutoff |
| `cerrd` | Coverage error rate-optimal | Inference accuracy is priority |
| Manual | Fixed bandwidth based on domain knowledge | Substantive reasons for window |

---

## 3. Bandwidth Sensitivity Analysis

The goal of sensitivity analysis is to show that your treatment effect estimate does not depend critically on the specific bandwidth chosen.

### Generating the Sensitivity Plot

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>

```python
import matplotlib.pyplot as plt

bandwidths = np.linspace(0.05, 0.6, 30)
estimates = []
ci_lows = []
ci_highs = []

for h in bandwidths:
    try:
        result = rdrobust(y=df['y'], x=df['x'], c=0, h=h)
        estimates.append(result.coef[0])
        ci_lows.append(result.ci[0])
        ci_highs.append(result.ci[1])
    except Exception:
        estimates.append(np.nan)
        ci_lows.append(np.nan)
        ci_highs.append(np.nan)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: point estimates across bandwidths
ax = axes[0]
ax.plot(bandwidths, estimates, 'o-', color='steelblue', linewidth=2, markersize=5)
ax.fill_between(bandwidths, ci_lows, ci_highs, alpha=0.2, color='steelblue')
ax.axhline(1.5, color='green', linestyle='--', label='True effect = 1.5')
ax.axvline(h_ik, color='red', linestyle=':', linewidth=2, label=f'IK bandwidth = {h_ik:.2f}')
ax.set_xlabel('Bandwidth h')
ax.set_ylabel('Treatment Effect Estimate')
ax.set_title('Bandwidth Sensitivity Plot')
ax.legend()

# Right: number of observations vs bandwidth
ax2 = axes[1]
n_obs = [int((np.abs(df['x']) <= h).sum()) for h in bandwidths]
ax2.plot(bandwidths, n_obs, 'o-', color='darkorange', linewidth=2, markersize=5)
ax2.axvline(h_ik, color='red', linestyle=':', linewidth=2, label=f'IK bandwidth')
ax2.set_xlabel('Bandwidth h')
ax2.set_ylabel('Observations within bandwidth')
ax2.set_title('Effective Sample Size vs Bandwidth')
ax2.legend()

plt.tight_layout()
plt.show()
```

</div>

### Interpreting the Sensitivity Plot

A **stable** sensitivity plot shows:
- Estimates relatively flat across a range of bandwidths
- Confidence intervals consistently excluding zero (if the effect is real)
- No sudden jumps or discontinuities

A **fragile** plot shows:
- Estimates that change sign or magnitude sharply at some bandwidth
- Very wide confidence intervals at small bandwidths
- The result only appears at one specific bandwidth

---

## 4. Polynomial Order Selection

Beyond bandwidth, you must choose the polynomial order for the local regression.

### Gelman & Imbens (2019) Recommendation

Use **local linear** (order 1) or at most **local quadratic** (order 2). Higher-order polynomials (3, 4, 5) are not recommended because:

1. Global high-order polynomials are sensitive to observations far from the cutoff
2. They produce erratic behaviour near the boundaries (Runge's phenomenon)
3. The fit near the cutoff is dominated by distant points

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>

```python
# Compare polynomial orders
for order in [1, 2, 3, 4]:
    result = rdrobust(y=df['y'], x=df['x'], c=0, h=h_ik, p=order)
    print(f"Order p={order}: τ = {result.coef[0]:.3f}, "
          f"SE = {result.se[0]:.3f}, "
          f"95% CI = [{result.ci[0]:.3f}, {result.ci[1]:.3f}]")
```

</div>

### The Standard Approach

1. **Report local linear as primary estimate** — it's the most robust
2. **Report local quadratic as robustness check** — should give similar results
3. **Do not report higher order results** unless there's a compelling reason

---

## 5. Kernel Weighting

Kernel weights down-weight observations further from the cutoff. Common choices:

| Kernel | Formula | Notes |
|--------|---------|-------|
| Triangular | $1 - |x-c|/h$ | Optimal for boundary estimation (default) |
| Uniform | $1$ for all obs in window | Simple; used in manual implementations |
| Epanechnikov | $1 - ((x-c)/h)^2$ | Optimal for density estimation |

The triangular kernel is optimal for the boundary regression problem in RDD — it down-weights observations far from the cutoff, which is exactly what you want.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>

```python
# Compare kernels
for kernel in ['triangular', 'uniform', 'epanechnikov']:
    result = rdrobust(y=df['y'], x=df['x'], c=0, h=h_ik, kernel=kernel)
    print(f"{kernel:<15}: τ = {result.coef[0]:.3f}, SE = {result.se[0]:.3f}")
```

</div>

In practice, kernel choice rarely matters as much as bandwidth choice.

---

## 6. Uncertainty and Standard Errors

### Heteroscedasticity-Robust Standard Errors

The variance of the outcome near the cutoff may differ from the variance further away. Use HC2 or HC3 robust standard errors:

```python
# rdrobust uses HC-type robust SEs by default
result = rdrobust(y=df['y'], x=df['x'], c=0, h=h_ik, vce='hc1')
```

### Cluster-Robust Standard Errors

If observations are clustered (e.g., students within schools), cluster at the natural clustering level:

```python
result = rdrobust(y=df['y'], x=df['x'], c=0, h=h_ik,
                  cluster=df['school_id'])
```

Ignoring clustering in clustered data leads to over-rejection of the null (false positives).

---

## 7. The Donut RDD

The "donut" removes observations very close to the cutoff, testing whether results are driven by local manipulation:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>

```python
def donut_rdd(df, outcome, running_var, cutoff, bandwidth, donut_width):
    """
    Estimate RDD excluding observations within [cutoff - donut_width, cutoff + donut_width].
    If manipulation occurs, units in the donut hole are most affected.
    Excluding them tests if the estimate is robust to local manipulation.
    """
    centered = df[running_var] - cutoff
    # Exclude the donut
    mask = (np.abs(centered) > donut_width) & (np.abs(centered) <= bandwidth)
    donut_df = df[mask].copy()

    result = rdrobust(y=donut_df[outcome], x=donut_df[running_var], c=cutoff)
    return result.coef[0], result.pv[0]

# Test across different donut widths
true_h = h_ik
for donut in [0.01, 0.02, 0.05, 0.10]:
    tau, pval = donut_rdd(df, 'y', 'x', 0, true_h, donut)
    print(f"Donut width = {donut:.2f}: τ = {tau:.3f}, p = {pval:.3f}")
```

</div>

If estimates are stable across donut widths, manipulation of the running variable is unlikely to be driving results.

---

## 8. CausalPy Bandwidth Sensitivity

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>

```python
import causalpy as cp
import numpy as np

bandwidths_to_test = [0.1, 0.2, 0.3, 0.4, 0.5]
results_cp = []

for bw in bandwidths_to_test:
    result = cp.RegressionDiscontinuity(
        data=df,
        formula='y ~ 1 + x',
        running_variable_name='x',
        model=cp.pymc_models.LinearRegression(
            sample_kwargs={'draws': 1000, 'tune': 500, 'chains': 2}
        ),
        bandwidth=bw,
        epsilon=0.01
    )
    tau = result.idata.posterior['treated'].values.flatten()
    results_cp.append({
        'bandwidth': bw,
        'mean': tau.mean(),
        'hdi_3': np.percentile(tau, 3),
        'hdi_97': np.percentile(tau, 97),
    })

# Plot Bayesian bandwidth sensitivity
import matplotlib.pyplot as plt
import pandas as pd

sens_df = pd.DataFrame(results_cp)
fig, ax = plt.subplots(figsize=(9, 4))
ax.errorbar(sens_df['bandwidth'], sens_df['mean'],
            yerr=[sens_df['mean'] - sens_df['hdi_3'],
                  sens_df['hdi_97'] - sens_df['mean']],
            fmt='o-', capsize=5, color='steelblue', linewidth=2)
ax.axhline(1.5, color='green', linestyle='--', label='True effect')
ax.set_xlabel('Bandwidth')
ax.set_ylabel('Posterior Mean (94% HDI)')
ax.set_title('CausalPy RDD: Bandwidth Sensitivity')
ax.legend()
plt.show()
```

</div>

---

## 9. Summary: Best Practices

| Decision | Recommended Approach |
|----------|---------------------|
| Bandwidth | IK MSE-optimal; present full sensitivity plot |
| Polynomial | Local linear as primary; local quadratic as robustness |
| Kernel | Triangular (default in rdrobust) |
| Standard errors | HC-robust; cluster if grouped data |
| Manipulation | McCrary density test + donut RDD |
| Primary plot | RDD plot with raw data scatter + fitted lines + CI band |
| Reporting | Always report range of bandwidths, not just one |

---


## Practice Questions

### Question 1: Conceptual Check
**Question:** In your own words, explain the core concept of RDD Bandwidth Selection and Sensitivity Analysis and why it matters for practical applications. What problem does it solve that simpler approaches cannot?

### Question 2: Application
**Question:** Describe a real-world scenario where you would apply the techniques from this guide. What assumptions would you need to verify before proceeding?

## Further Reading

- Imbens & Kalyanaraman (2012), "Optimal Bandwidth Choice for the Regression Discontinuity Estimator"
- Calonico, Cattaneo & Titiunik (2014), "Robust Nonparametric Confidence Intervals for RD Designs" (CCT)
- Gelman & Imbens (2019), "Why High-Order Polynomials Should Not Be Used in Regression Discontinuity Designs"
- Cattaneo & Titiunik (2022), "Regression Discontinuity Designs" (Annual Review of Economics)

---

**Previous:** [01 — RDD Fundamentals](01_rdd_fundamentals_guide.md)
**Next:** [Module 05 Notebooks](../notebooks/)

<div class="callout-key">

<strong>Key Concept:</strong> **Previous:** [01 — RDD Fundamentals](01_rdd_fundamentals_guide.md)
**Next:** [Module 05 Notebooks](../notebooks/)

</div>



## Resources

<a class="link-card" href="../notebooks/01_sharp_rdd.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises for this topic.</div>
</a>
