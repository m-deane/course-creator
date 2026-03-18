---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Segmented Regression

## The Statistical Engine of ITS

### Causal Inference with CausalPy — Module 01, Guide 2

<!-- Speaker notes: This guide goes under the hood of the ITS model. Guide 1 gave the conceptual picture; this guide provides the mathematical machinery. Students will leave understanding exactly what each parameter estimates, why autocorrelation is a problem and how to fix it, and how to choose between model specifications. This is the material that separates practitioners who use ITS correctly from those who apply it naively. -->

---

# The Full Segmented Regression

$$Y_t = \underbrace{\alpha + \beta_1 t}_{\text{pre-trend}} + \underbrace{\beta_2 D_t}_{\text{level change}} + \underbrace{\beta_3 (t - t^*) D_t}_{\text{slope change}} + \varepsilon_t$$

Two line segments, joined at $t^*$:

**Pre-intervention** ($t < t^*$):
$$Y_t = \alpha + \beta_1 t$$

**Post-intervention** ($t \geq t^*$):
$$Y_t = (\alpha + \beta_1 t^* + \beta_2) + (\beta_1 + \beta_3)(t - t^*)$$

<!-- Speaker notes: Walk through the algebra to show these two segments. The pre-period is just a simple linear trend. The post-period has a different intercept (shifted by beta_2) and a different slope (shifted by beta_3). The beauty of the parameterization: beta_2 directly gives the level change and beta_3 directly gives the slope change. The counterfactual is the pre-trend extrapolated forward: alpha + beta_1 * t. Subtracting that from the post-period gives beta_2 + beta_3 * (t - t*). -->

---

# Reading the Parameters

| Parameter | What It Measures | Example Interpretation |
|-----------|-----------------|----------------------|
| $\alpha$ | Baseline level at $t=0$ | "Productivity was 100 units at study start" |
| $\beta_1$ | Pre-intervention monthly growth | "Growing 0.8 units/month before the program" |
| $\beta_2$ | **Immediate level change** | "Program caused an instant +8 unit jump" |
| $\beta_3$ | **Additional monthly growth** | "After program, growth increased by 0.3/month" |

**Causal effect at month $k$ post-intervention:** $\hat{\tau}_{t^*+k} = \beta_2 + \beta_3 \cdot k$

<!-- Speaker notes: Each parameter has a direct, interpretable meaning in the policy evaluation context. Practitioners often want to report: (1) was there an immediate effect? (beta_2), (2) did the trend change? (beta_3), and (3) what is the cumulative impact? (sum of tau_t over post-period). In the Bayesian framework, we have full posterior distributions for all three, plus the probability that each is positive. This gives much richer reporting than a single p-value. -->

---

# Design Matrix: What Goes Into the Regression

For $n_{pre}=5$, $n_{post}=5$, $t^*=5$:

| $t$ | Intercept | $t$ | $D_t$ (treated) | $(t-t^*)D_t$ (t\_post) | $Y_t$ |
|-----|-----------|-----|-----------------|------------------------|-------|
| 0 | 1 | 0 | 0 | 0 | pre... |
| 1 | 1 | 1 | 0 | 0 | pre... |
| 4 | 1 | 4 | 0 | 0 | pre... |
| **5** | 1 | 5 | **1** | **0** | ← intervention |
| 6 | 1 | 6 | 1 | 1 | post... |
| 9 | 1 | 9 | 1 | 4 | post... |

At the intervention point: $D_t = 1$ but $t_{post} = 0$ — only the level change applies immediately.

<!-- Speaker notes: The design matrix makes concrete what the model is estimating. The key column is t_post (time since intervention): it is zero in the entire pre-period AND at the intervention point itself, then increases by 1 for each subsequent period. This means beta_2 captures the instantaneous jump at t*, and beta_3 captures the additional accumulating effect over time. If a student is confused about why t_post=0 at the intervention point, clarify: the level change happens at the moment of intervention (captured by beta_2), but the slope change needs time to accumulate. -->

---

# Model Variants: Choose Based on Theory

<div class="columns">

**Level + Slope (Full)**
$$y \sim 1 + t + D_t + t_{post}$$
Use when: both immediate and trajectory effects expected.

**Level Only**
$$y \sim 1 + t + D_t$$
Use when: one-time shock, no lasting trajectory change.

**Slope Only**
$$y \sim 1 + t + t_{post}$$
Use when: gradual diffusion effect, no immediate jump.

</div>

**Recommendation:** Start with the full model. The posterior will tell you which components are active.

<!-- Speaker notes: The choice of model variant should be driven by theory, not by which one gives the best fit. If you have a policy that was phased in gradually, a slope-only model makes more sense than a level-only model. If you have a one-time shock (a natural disaster, a one-time payment), a level-only model is more appropriate. Starting with the full model and examining the posteriors for beta_2 and beta_3 is a good strategy: if either credible interval includes zero, that component may not be supported. -->

---

# Autocorrelation: The Standard Error Problem

Standard regression assumes: $\text{Cov}(\varepsilon_t, \varepsilon_s) = 0$ for $t \neq s$

Time series reality: $\hat{\varepsilon}_t \approx \rho \hat{\varepsilon}_{t-1}$ with $\rho > 0$

**Consequences:**
- Residuals are correlated → effective sample size $< N$
- Standard errors are too small
- T-statistics are too large → too many false positives

**Durbin-Watson test:**
$$DW = \frac{\sum_{t=2}^T (\hat{\varepsilon}_t - \hat{\varepsilon}_{t-1})^2}{\sum_{t=1}^T \hat{\varepsilon}_t^2}$$

$DW \approx 2$: no autocorrelation. $DW < 1.5$: positive autocorrelation (common).

<!-- Speaker notes: Autocorrelation is endemic to time series data. Monthly economic data typically shows autocorrelation of 0.6-0.9. A DW statistic of 1.0 corresponds to rho of about 0.5. At rho=0.5, the effective sample size is roughly halved — you have 100 observations but the information content of only 50 independent ones. Standard errors computed ignoring this are roughly 1/sqrt(2) = 70% of their correct values. The corresponding t-statistics are 40% too large, dramatically inflating the false positive rate. This is not a minor correction — it changes conclusions. -->

---

# Three Solutions to Autocorrelation

**Option 1: Newey-West Standard Errors**
HAC (heteroskedasticity and autocorrelation consistent) standard errors.
- Does not change point estimates
- Adjusts standard errors post-hoc
- Simple to implement: `model.fit(cov_type='HAC')`

**Option 2: Prais-Winsten / Cochrane-Orcutt**
Transform the model to remove AR(1) correlation.
- Changes both point estimates and standard errors
- `statsmodels.regression.linear_model.GLSAR`

**Option 3: Bayesian AR(1) (CausalPy + PyMC)**
Model autocorrelation explicitly in the error structure.
- Most principled approach
- Posterior over $\rho$ propagated to all other estimates
- Recommended for this course

<!-- Speaker notes: In practice for this course, we use the Bayesian approach (Option 3) because it handles autocorrelation naturally through the PyMC model specification, and the posterior over rho gives us an explicit estimate of how much autocorrelation is present. Newey-West (Option 1) is the simplest fix and appropriate for quick analyses in a frequentist framework. Prais-Winsten (Option 2) is the gold standard for frequentist ITS but requires more implementation effort. -->

---

# Visual Autocorrelation Diagnostics

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Residuals from initial OLS fit
residuals = ols_result.resid

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(residuals, lags=20, ax=axes[0])
plot_pacf(residuals, lags=20, ax=axes[1])
axes[0].set_title("Autocorrelation Function (ACF)")
axes[1].set_title("Partial Autocorrelation Function (PACF)")
plt.tight_layout()
plt.show()
```

ACF shows correlation at each lag. PACF shows the unique contribution at each lag.

Significant spike at lag 1 in both → AR(1) process. Multiple spikes → higher-order AR.

<!-- Speaker notes: The ACF and PACF plots are the standard diagnostic for autocorrelation structure. For a first-order AR process, the ACF decays exponentially and the PACF has a single significant spike at lag 1. For a second-order AR process, the PACF has significant spikes at lags 1 and 2. These plots tell you how much autocorrelation to model. In practice, AR(1) is sufficient for most monthly ITS analyses — higher-order AR processes are rare and require more data to identify. The Bayesian approach can handle AR(p) with appropriate PyMC model specification. -->

---

# Seasonality: The Second Major Issue

Many outcomes have strong seasonal patterns:
- Hospital admissions peak in winter (respiratory illness)
- Crime rates peak in summer (outdoor activity)
- Retail sales peak in November-December (holidays)

**Problem:** If the intervention happens to occur at a seasonal peak or trough, the observed level change may partly reflect the season, not the intervention.

**Solution:** Include seasonal terms in the model formula.

<!-- Speaker notes: Seasonality is particularly dangerous when the intervention timing coincides with a seasonal pattern. A school lunch program starting in September will coincide with the return from summer break — test scores and attendance naturally improve in September. An ITS analysis without seasonal controls would attribute this September bounce to the program. Seasonal controls (month dummies or Fourier terms) remove this spurious effect. -->

---

# Seasonal Adjustment: Two Approaches

**Approach 1: Month Dummy Variables**
```python
# Add month indicators
df["calendar_month"] = df["date"].dt.month
formula = "y ~ 1 + t + treated + t_post + C(calendar_month)"
```
- Non-parametric: each month has its own fixed effect
- Requires at least one full seasonal cycle in pre-period

**Approach 2: Fourier Terms**
```python
# Add sine/cosine pairs for smooth seasonality
for k in range(1, 3):  # 2 harmonics for annual seasonality
    df[f"sin_{k}"] = np.sin(2 * np.pi * k * df["t"] / 12)
    df[f"cos_{k}"] = np.cos(2 * np.pi * k * df["t"] / 12)

formula = "y ~ 1 + t + treated + t_post + sin_1 + cos_1 + sin_2 + cos_2"
```
- Parametric: smooth, fewer parameters
- Can work with less data

<!-- Speaker notes: The choice between month dummies and Fourier terms depends on data length and the shape of the seasonal pattern. Month dummies need at least 12 pre-intervention months (one complete cycle) to estimate. With less data, Fourier terms with 1-2 harmonics work better. Month dummies allow non-smooth seasonal patterns (e.g., a sharp holiday spike), while Fourier terms force a smooth sinusoidal pattern. In most practical cases, either approach gives similar results when the seasonal pattern is smooth and the pre-period is at least 24 months. -->

---

# Model Selection: LOO-CV in ArviZ

```python
import arviz as az

# Fit models with different specifications
model_full = cp.InterruptedTimeSeries(
    data=df, formula="y ~ 1 + t + treated + t_post", ...
)
model_level = cp.InterruptedTimeSeries(
    data=df, formula="y ~ 1 + t + treated", ...
)
model_slope = cp.InterruptedTimeSeries(
    data=df, formula="y ~ 1 + t + t_post", ...
)

# Compare using LOO cross-validation
comparison = az.compare({
    "full": model_full.idata,
    "level_only": model_level.idata,
    "slope_only": model_slope.idata,
})
print(comparison)
```

Higher LOO ELPD = better predictive accuracy. Weight differences ≥ 4 are meaningful.

<!-- Speaker notes: LOO-CV (Leave-One-Out Cross-Validation) is the Bayesian model comparison tool. It estimates out-of-sample predictive accuracy by computing how well the model would have predicted each observation if that observation had been left out of training. ArviZ computes this efficiently using importance sampling (PSIS-LOO). When comparing models, focus on the ELPD differences and standard errors — differences smaller than the standard error are not meaningful. In practice, the full model is usually preferred unless there is a strong theoretical reason to prefer one of the restricted models. -->

---

# The Counterfactual in Math

The ITS counterfactual at each post-intervention time:

$$\hat{Y}_t(0) = \hat{\alpha} + \hat{\beta}_1 \cdot t \quad \text{for } t \geq t^*$$

This is the regression line from the pre-period, extrapolated forward.

**Bayesian counterfactual:** Not a single line but a distribution:

$$P(\hat{Y}_t(0) | \text{data}) \sim \mathcal{N}(\hat{\alpha} + \hat{\beta}_1 t, \text{posterior uncertainty})$$

The uncertainty grows as we extrapolate further from $t^*$.

<!-- Speaker notes: The growing uncertainty as we extrapolate further from t* is a key feature of the Bayesian approach. In a frequentist framework, the standard error of the counterfactual prediction is constant (or very similar) across all post-intervention time points. But intuitively, our uncertainty about what would have happened should grow the longer we extrapolate away from the last observed data point. The Bayesian posterior captures this naturally through the posterior uncertainty on the coefficients, which translates to growing predictive uncertainty over time. -->

---

# Putting It Together: Posterior Predictive Check

After fitting the model, generate data from the posterior and compare to observed:

```python
import arviz as az

# Posterior predictive check
with its_model.model:
    ppc = pm.sample_posterior_predictive(its_model.idata)

az.plot_ppc(
    its_model.idata,
    observed=True,
    num_pp_samples=100,
)
```

If the posterior predictive distribution does not cover the observed data well, the model is misspecified.

<!-- Speaker notes: The posterior predictive check (PPC) is the Bayesian equivalent of checking residual plots in frequentist regression. Generate many datasets from the posterior predictive distribution and overlay them on the observed data. If the model is well-specified, the observed data should look like a typical draw from the posterior predictive. Systematic misfits (e.g., the model consistently under-predicts in summer) indicate model misspecification that should be addressed before interpreting the causal estimates. -->

---

<!-- _class: lead -->

# Core Takeaway

## $\beta_2$ = immediate level change (jump at $t^*$)
## $\beta_3$ = slope change (trend acceleration/deceleration)
## Autocorrelation inflates false positives — always check and correct

<!-- Speaker notes: The three-line takeaway for segmented regression. Beta_2 and beta_3 are the quantities policymakers care about. Autocorrelation is the technical issue that makes naive standard errors unreliable. The Bayesian approach with an explicit AR(1) error structure handles both: it correctly estimates beta_2 and beta_3 while propagating uncertainty about the autocorrelation coefficient into all other parameters. -->

---

# What's Next

**Guide 3:** CausalPy ITS API Walkthrough
- `InterruptedTimeSeries` class in depth
- Formula specification
- Model objects and output attributes
- Custom PyMC models

**Notebook 1:** ITS on Smoking Ban Data
- Monthly hospital admissions data
- Full diagnostic workflow
- Interpreting and communicating results

<!-- Speaker notes: Guide 3 is the practical API guide — students will learn exactly what arguments to pass to InterruptedTimeSeries, how to read the output, and how to customize the PyMC model if needed. After Guide 3, they have everything they need for Notebook 1, which is a full end-to-end ITS analysis on a real policy dataset. The smoking ban example is a classic in the ITS literature, with a clear mechanism (reduced secondhand smoke exposure → reduced cardiac events) and enough data to demonstrate the full diagnostic workflow. -->
