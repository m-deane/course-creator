---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Hypothesis Testing and Robust Inference

## HAC Standard Errors and Significance Tests

**Mixed-Frequency Models: MIDAS Regression and Nowcasting**
Module 02 — Guide 03

<!-- Speaker notes: This guide covers inference for MIDAS. Three key topics: why standard OLS errors fail for MIDAS, how to compute HAC standard errors, and the three most important hypothesis tests (beta significance, equal-weight restriction, joint significance). Students who've seen HAC before can move quickly through the theory and focus on the MIDAS-specific implementation. -->

---

## Why Standard Errors Fail for MIDAS

**Problem 1: Serial correlation**
$$\text{Cov}(\varepsilon_t, \varepsilon_{t-k}) \neq 0$$
OLS assumes $\text{Cov} = 0$. Violation → standard errors too small.

**Problem 2: Heteroscedasticity**
$$\text{Var}(\varepsilon_t) = \sigma_t^2 \neq \sigma^2$$
Crisis periods (2020, 2008) have much larger residuals.

**Problem 3: Generated regressors**
$$\tilde{x}_t(\hat{\theta}) \text{ uses estimated weights} \Rightarrow \text{extra estimation uncertainty}$$

**Solution: HAC (Heteroscedasticity and Autocorrelation Consistent) standard errors.**

<!-- Speaker notes: These three problems are present to varying degrees in most MIDAS applications. Serial correlation is the most important — quarterly GDP growth has AR ≈ 0.3, which means residuals are likely serially correlated even after conditioning on IP. Heteroscedasticity is visible in plots of residuals vs. time (big spikes in 2008 and 2020). The generated regressors problem is more subtle but is automatically handled by the HAC estimator when applied at the optimal theta. -->

<div class="callout-key">

The key advantage of MIDAS is preserving high-frequency information that temporal aggregation destroys.

</div>

---

## The Newey-West HAC Estimator

The HAC covariance matrix at the linearized model:

$$\hat{V}_{HAC} = (\mathbf{Z}^\top\mathbf{Z})^{-1} \hat{\mathbf{S}}_{NW} (\mathbf{Z}^\top\mathbf{Z})^{-1}$$

where $\mathbf{Z}_t = (1, \tilde{x}_t(\hat{\theta}))$ and:

$$\hat{\mathbf{S}}_{NW} = \hat{\boldsymbol{\Gamma}}_0 + \sum_{l=1}^{L} \underbrace{\left(1 - \frac{l}{L+1}\right)}_{\text{Bartlett kernel}}\left(\hat{\boldsymbol{\Gamma}}_l + \hat{\boldsymbol{\Gamma}}_l^\top\right)$$

**Bandwidth rule:** $L = \lfloor 4 (T/100)^{2/9} \rfloor$

For $T = 100$: $L = 4$. For $T = 200$: $L = 5$.

<!-- Speaker notes: The Newey-West estimator is the standard HAC estimator used in econometrics. The Bartlett kernel (1 - l/(L+1)) gives declining weights to higher-order autocovariances, which is necessary for the estimator to be positive semi-definite. The bandwidth L controls how many lags of autocorrelation are accounted for. The rule 4*(T/100)^(2/9) is the standard recommendation from Andrews (1991) for quarterly data. For T=100 it gives L=4, meaning autocorrelations up to 4 quarters are included. -->

<div class="callout-insight">

**Insight:** Parsimonious weight functions with 2-3 parameters can capture decay patterns that unrestricted models need 12+ parameters to approximate.

</div>

---

## Implementation with statsmodels

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import statsmodels.api as sm
import numpy as np

def midas_hac(Y, x_tilde, L=None):
    """HAC inference for MIDAS (linearized model)."""
    T = len(Y)
    if L is None:
        L = int(4 * (T/100)**(2/9))

    Z = sm.add_constant(x_tilde)  # [1, x_tilde]
    model = sm.OLS(Y, Z)
    result = model.fit(cov_type='HAC', cov_kwds={'maxlags': L})

    print(result.summary())
    return result

# Usage
x_tilde = X @ beta_weights(K, theta1_hat, theta2_hat)
hac_result = midas_hac(Y, x_tilde)
```

</div>

<!-- Speaker notes: statsmodels makes HAC inference easy with the cov_type='HAC' option. The linearized model is just OLS of Y on [1, x_tilde] where x_tilde is computed using the estimated theta. This approach correctly gives HAC standard errors for alpha and beta. It does NOT account for the estimation uncertainty in theta itself — for that, you need the bootstrap or the full NLS covariance matrix. In practice, the standard errors for beta are the most important, and the HAC approach is adequate for these. -->

<div class="callout-warning">

**Warning:** Always account for the real-time data vintage when evaluating nowcast performance. Using revised data overstates accuracy.

</div>

---

## Key Hypothesis Tests

### Test 1: Is IP significant?

$$H_0: \beta = 0 \quad \text{vs} \quad H_1: \beta \neq 0$$

$$t = \hat{\beta} / \text{se}_{HAC}(\hat{\beta}) \sim t_{T-2} \text{ approx}$$

Typically $|\hat{\beta}| / \text{se}_{HAC} > 3$ for strong macro predictors.

### Test 2: Equal weights (aggregation)

$$H_0: \theta_1 = \theta_2 = 1 \quad (\text{Beta(1,1) = equal-weight})$$

$$F = \frac{(SSE_R - SSE_U)/2}{SSE_U/(T-4)} \sim F_{2, T-4}$$

### Test 3: AR terms needed?

$$H_0: \text{no serial correlation in residuals}$$

Ljung-Box $Q(4) \sim \chi^2_4$

<!-- Speaker notes: These three tests form the core inference checklist for any MIDAS application. Test 1 (IP significance) is the baseline — if beta is not significant, the whole MIDAS specification is questionable. Test 2 (equal weights) is the key MIDAS-specific test — it tells you whether the polynomial restriction provides any improvement over simple aggregation. Test 3 (AR terms) determines whether the MIDAS-AR extension is needed. Run all three in order before finalizing any MIDAS model. -->

<div class="callout-info">

**Info:** MIDAS models can handle any frequency ratio: monthly-to-quarterly (3:1), daily-to-monthly (~22:1), or even tick-to-daily.

</div>

---

## Example: GDP ~ IP Inference

```
MIDAS Results: GDP Growth ~ IP Growth
Sample: 2000Q1 – 2024Q4 (T=100)
Weight function: Beta(θ₁=1.42, θ₂=4.31), K=12

HAC Standard Errors (L=4 lags):
─────────────────────────────────────────────
Parameter    Estimate   HAC SE   t-stat   p
─────────────────────────────────────────────
constant      0.413      0.078    5.30   0.000 ***
beta (IP)     0.521      0.143    3.64   0.001 ***
─────────────────────────────────────────────
R² = 0.354

F-test (equal weights): F(2,96) = 4.83, p = 0.011 *
Ljung-Box Q(4) = 6.21, p = 0.184 (no AR needed)
```

*Values are illustrative — Notebook 03 computes actual results.*

<!-- Speaker notes: Walk through this example carefully. The IP coefficient is significant at 0.1% level (t=3.64) — strong evidence that IP predicts GDP. The F-test for equal weights rejects at 5% (p=0.011) — evidence that the polynomial restriction improves on simple aggregation. The Ljung-Box test does not reject (p=0.184) — no AR terms needed. These three conclusions together justify the Beta MIDAS specification over OLS-aggregate. -->

---

## Confidence Intervals for Weights

The weights $\hat{w}_j = w_j(\hat{\theta}_1, \hat{\theta}_2)$ inherit uncertainty from $\hat{\theta}$.

**Delta method approximation:**

$$\text{Var}(\hat{w}_j) \approx \left(\frac{\partial w_j}{\partial \theta}\right)^\top \text{Var}(\hat{\theta}) \left(\frac{\partial w_j}{\partial \theta}\right)$$

**Simpler: bootstrap confidence bands around weight function.**

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
# Bootstrap bands for weight function
boot_weights = np.array([
    beta_weights(K, t1, t2)
    for t1, t2 in boot_dist[:, 2:4]
])
w_lower = np.percentile(boot_weights, 2.5, axis=0)
w_upper = np.percentile(boot_weights, 97.5, axis=0)
```

</div>

<!-- Speaker notes: Confidence intervals for the weight function are less commonly reported but can be informative. The bootstrap approach is the most practical: after generating the bootstrap distribution of theta, compute the implied weight function at each bootstrap draw, then take percentiles. The resulting confidence band around the weight function shows where the data has strong vs. weak evidence about the timing of influence. If the confidence band for lag 0 is tight and clearly above the equal-weight horizontal line, that's strong evidence for front-loading. -->

---

## Visualizing Inference

```
Estimated weight function with 95% bootstrap confidence bands:

Weight
 0.25 |   ████
 0.20 |   ████████████
 0.15 |   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
 0.10 |   ░░░░░░░░░░░░░░░░░░░░░░
 0.05 |   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 0.00 +───────────────────────────────
      j=0  2  4  6  8  10
      │         │         │
     Curr Q    Q-1       Q-2

--- Equal-weight line (1/K)
███ Estimated weight (Beta polynomial)
░░░ 95% bootstrap confidence band
```

<!-- Speaker notes: This visualization shows what we're looking for: the estimated weight bars above the equal-weight line for recent lags (especially j=0,1,2), and below the line for older lags. The bootstrap confidence bands show whether the front-loading is statistically robust. If the lower band is above the equal-weight line for lag 0, we have strong evidence that the most recent month is more informative than average. This is the kind of result that can be reported as a finding: "the most recent month of the quarter carries approximately 22% of the total information weight, compared to 8% under equal weighting." -->

---

## Summary: Inference Checklist

For any MIDAS application, verify:

- [ ] HAC standard errors computed (not plain OLS)
- [ ] Beta (IP) coefficient is significant ($|t| > 1.96$)
- [ ] F-test for equal weights reported
- [ ] Ljung-Box test for residual autocorrelation
- [ ] If LB rejects: add AR term and re-test
- [ ] 95% CI for beta reported
- [ ] Bootstrap CI for weight function (optional but informative)

**Next:** Module 03 — Nowcasting with MIDAS. Apply this inference framework to real-time GDP estimation.

<!-- Speaker notes: The checklist format is useful for students to follow mechanically. Every MIDAS analysis should go through this checklist before results are reported. The most commonly skipped item is the Ljung-Box test — many practitioners forget to check for residual autocorrelation and end up with models that have serially correlated errors but invalid standard errors. Make this automatic. -->
