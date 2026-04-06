# Segmented Regression for ITS

> **Reading time:** ~10 min | **Module:** 1 — Its Fundamentals | **Prerequisites:** Module 0 — Causal Foundations

## In Brief

Segmented regression (also called piecewise linear regression) is the statistical engine of ITS. It fits two separate regression lines to the pre- and post-intervention periods, with constraints ensuring continuity. The difference between the two fitted lines gives the level and slope change estimates.

<div class="callout-key">

<strong>Key Concept:</strong> Segmented regression (also called piecewise linear regression) is the statistical engine of ITS. It fits two separate regression lines to the pre- and post-intervention periods, with constraints ensuring continuity.

</div>

## Key Insight

The model parameterization is critical: including or excluding specific terms changes the estimand fundamentally. A level-only model estimates a one-time shock; a slope-only model estimates a gradually accumulating effect; the full model estimates both. Using the wrong parameterization produces biased results even with perfect data.

---

## The Full Segmented Regression Model

$$Y_t = \alpha + \beta_1 t + \beta_2 D_t + \beta_3 (t - t^*) D_t + \varepsilon_t$$

This can be rewritten to make the two segments explicit:

**Pre-intervention segment** ($D_t = 0$):
$$Y_t = \alpha + \beta_1 t + \varepsilon_t$$

**Post-intervention segment** ($D_t = 1$):
$$Y_t = (\alpha + \beta_2) + (\beta_1 + \beta_3)(t - t^*) + \beta_1 t^* + \varepsilon_t$$
$$= (\alpha + \beta_1 t^* + \beta_2) + (\beta_1 + \beta_3)(t - t^*) + \varepsilon_t$$

The post-intervention intercept is $\alpha + \beta_1 t^* + \beta_2$, and the post-intervention slope is $\beta_1 + \beta_3$.

**Why this parameterization?** The $\beta_2$ coefficient directly measures the level change at $t^*$ and $\beta_3$ measures the change in slope — both are directly interpretable.

---

## Constructing the Design Matrix

The formula `y ~ 1 + t + treated + t_post` in CausalPy maps to:

| Time | $t$ | $D_t$ (treated) | $(t - t^*)D_t$ (t\_post) |
|------|-----|-----------------|--------------------------|
| Pre-1 | 0 | 0 | 0 |
| Pre-2 | 1 | 0 | 0 |
| ... | ... | 0 | 0 |
| Pre-N | $t^* - 1$ | 0 | 0 |
| Post-1 | $t^*$ | 1 | 0 |
| Post-2 | $t^* + 1$ | 1 | 1 |
| Post-3 | $t^* + 2$ | 1 | 2 |
| ... | ... | 1 | ... |

Note: at $t = t^*$, the `t_post` variable is 0 (the level change applies but no slope accumulation yet). This is the standard convention.


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np
import pandas as pd

def build_its_design_matrix(n_total: int, n_pre: int) -> pd.DataFrame:
    """
    Construct the design matrix for a standard ITS segmented regression.

    Parameters
    ----------
    n_total : int
        Total number of time periods
    n_pre : int
        Number of pre-intervention periods

    Returns
    -------
    pd.DataFrame with columns: t, treated, t_post
    """
    t = np.arange(n_total)
    treated = (t >= n_pre).astype(float)
    t_post = np.maximum(t - n_pre, 0).astype(float)

    return pd.DataFrame({"t": t, "treated": treated, "t_post": t_post})

# Verify the design matrix structure
dm = build_its_design_matrix(n_total=10, n_pre=5)
print(dm)
# Expected: rows 0-4 have treated=0, t_post=0
#           row 5: treated=1, t_post=0 (intervention point)
#           rows 6-9: treated=1, t_post increasing
```

</div>

---

## Interpreting Each Parameter

### $\alpha$ (Intercept)

The estimated outcome level at $t = 0$ in the absence of treatment. This is typically not directly interpretable unless $t = 0$ has a natural meaning (e.g., the start of the study).

**Tip:** Center the time variable around the intervention point to make the intercept the estimated outcome just before the intervention. Replace $t$ with $(t - t^*)$ so that $t = 0$ corresponds to $t^*$.

### $\beta_1$ (Pre-Intervention Slope)

The monthly (or per-period) change in the outcome during the pre-intervention period. Represents the secular trend — background changes in the outcome unrelated to the intervention.

- $\beta_1 > 0$: outcome was rising before the intervention
- $\beta_1 < 0$: outcome was falling before the intervention
- $\beta_1 \approx 0$: roughly stable pre-intervention level

### $\beta_2$ (Level Change)

The immediate discontinuous jump in the outcome at $t^*$. This captures the instantaneous effect of the intervention.

- $\beta_2 > 0$: the intervention caused an immediate increase
- $\beta_2 < 0$: the intervention caused an immediate decrease
- $\beta_2 \approx 0$ with wide CI: no strong evidence of an immediate effect

In the Bayesian framework, we compute $P(\beta_2 > 0)$ directly from the posterior samples.

### $\beta_3$ (Slope Change)

The additional change in the slope of the outcome series after the intervention. The total post-intervention slope is $\beta_1 + \beta_3$.

- $\beta_3 > 0$: the post-intervention slope is steeper than the pre-intervention slope (growing effect)
- $\beta_3 < 0$: the post-intervention slope is flatter or negative (fading or reversing effect)
- $\beta_1 + \beta_3 = 0$: the post-intervention trend is flat regardless of the pre-trend

### The Time-Varying Causal Effect

At post-intervention time $k$ (where $k = t - t^*$), the causal effect is:

$$\hat{\tau}_{t^*+k} = \hat{\beta}_2 + \hat{\beta}_3 \cdot k$$

The cumulative effect over $K$ post-intervention periods is:

$$\hat{\tau}_{cumulative} = K \hat{\beta}_2 + \hat{\beta}_3 \cdot \frac{K(K-1)}{2}$$

---

## Autocorrelation: Detection and Correction

### Why Autocorrelation Occurs

In time series data, the outcome at time $t$ is influenced by the outcome at $t-1$. This creates serially correlated errors. Ignoring this leads to:

1. **Deflated standard errors**: The effective sample size is less than $N$ because consecutive observations carry redundant information.
2. **Inflated t-statistics**: More rejections of the null than the nominal significance level warrants.
3. **Incorrect credible intervals**: The Bayesian posterior is too narrow when the model misspecifies the error structure.

### Formal Test: Durbin-Watson

$$DW = \frac{\sum_{t=2}^T (\hat{\varepsilon}_t - \hat{\varepsilon}_{t-1})^2}{\sum_{t=1}^T \hat{\varepsilon}_t^2} \approx 2(1 - \hat{\rho})$$

where $\hat{\rho}$ is the estimated first-order autocorrelation. Values close to 2 indicate no autocorrelation.


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
from statsmodels.stats.stattools import durbin_watson
import statsmodels.formula.api as smf
import numpy as np

def check_autocorrelation(df: pd.DataFrame, formula: str) -> dict:
    """
    Fit OLS model and test for autocorrelation in residuals.

    Returns
    -------
    dict with 'dw_statistic', 'rho_estimate', and 'interpretation'
    """
    model = smf.ols(formula, data=df).fit()
    residuals = model.resid

    dw = durbin_watson(residuals)
    rho = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]

    if dw < 1.5:
        interpretation = f"Positive autocorrelation detected (ρ ≈ {rho:.3f})"
    elif dw > 2.5:
        interpretation = f"Negative autocorrelation detected (ρ ≈ {rho:.3f})"
    else:
        interpretation = "No significant autocorrelation"

    return {
        "dw_statistic": dw,
        "rho_estimate": rho,
        "interpretation": interpretation,
        "residuals": residuals,
    }
```

</div>

### Solution 1: Newey-West Standard Errors

The Newey-West (HAC) estimator produces consistent standard errors under autocorrelation without changing the point estimates.


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import statsmodels.formula.api as smf

# Fit OLS with Newey-West standard errors
model = smf.ols("y ~ 1 + t + treated + t_post", data=df).fit(
    cov_type="HAC",
    cov_kwds={"maxlags": 3},  # Adjust lag window for your data frequency
)
print(model.summary())
```

</div>

### Solution 2: Prais-Winsten / Cochrane-Orcutt (AR(1) Model)

Explicitly model the autocorrelation by including an AR(1) error:

$$\varepsilon_t = \rho \varepsilon_{t-1} + u_t, \quad u_t \sim \mathcal{N}(0, \sigma^2)$$

This transforms the original model to remove autocorrelation. Implemented in statsmodels as `GLSAR`.

### Solution 3: Bayesian AR(1) in CausalPy / PyMC

The cleanest approach: include an AR(1) process in the PyMC model definition. The posterior for $\rho$ captures the autocorrelation, and all other parameters are estimated conditionally on it.


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import pymc as pm
import numpy as np

def build_its_ar1_model(y, X):
    """
    ITS model with AR(1) error structure in PyMC.
    X should contain [1, t, treated, t_post] columns.
    """
    n = len(y)

    with pm.Model() as model:
        # Priors on regression coefficients
        beta = pm.Normal("beta", mu=0, sigma=10, shape=X.shape[1])

        # AR(1) autocorrelation coefficient
        rho = pm.Uniform("rho", lower=-1, upper=1)

        # Innovation standard deviation
        sigma = pm.HalfNormal("sigma", sigma=5)

        # Linear predictor (mean)
        mu = pm.math.dot(X, beta)

        # AR(1) errors via GaussianRandomWalk
        # Equivalent to: epsilon_t = rho * epsilon_{t-1} + noise
        ar_errors = pm.AR("ar_errors", rho=[rho], sigma=sigma, shape=n)

        # Likelihood
        y_obs = pm.Normal("y_obs", mu=mu + ar_errors, sigma=0.001, observed=y)

    return model
```

</div>

---

## Detecting Non-Linear Pre-Trends

A linear pre-trend assumption may be wrong if:
- The outcome shows curvature in the pre-period
- There is a seasonal component
- The trend was accelerating or decelerating

### Visual Check

Always plot the pre-period data with a linear trend overlay before running ITS. If the residuals show systematic curvature (consistently above or below the line in different parts of the pre-period), the linear trend is misspecified.

### Formal Test: Polynomial Trend

Fit a quadratic model to the pre-period:
$$Y_t = \alpha + \beta_1 t + \gamma t^2 + \varepsilon_t \quad \text{(pre-period only)}$$

If $\gamma$ is significantly different from zero, a linear pre-trend may be inadequate. Consider a quadratic pre-trend or a log-linear specification.

### Natural Splines

For flexible non-linear pre-trends, natural cubic splines provide a smooth non-parametric fit:


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
from patsy import dmatrix
import pandas as pd

# Add spline basis to dataframe (for pre-period only)
n_knots = 3  # Adjust based on pre-period length
pre_data = df[df["treated"] == 0].copy()

# Create spline basis
spline_basis = dmatrix(
    f"bs(t, df={n_knots + 1}, include_intercept=False)",
    pre_data,
    return_type="dataframe",
)

# Fit spline model to pre-period
# Use to check adequacy of linear trend assumption
```

</div>

---

## Model Selection

### In Frequentist ITS

Use information criteria (AIC, BIC) to compare:
- Full model (level + slope change): 4 parameters
- Level-only model: 3 parameters
- Slope-only model: 3 parameters

Lower AIC = better fit penalized for complexity. BIC penalizes complexity more strongly.

### In Bayesian ITS (CausalPy)

Use **Leave-One-Out Cross-Validation (LOO)** via ArviZ:


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import arviz as az

# Fit both models
full_model_result = cp.InterruptedTimeSeries(
    data=df, treatment_time=t_star,
    formula="y ~ 1 + t + treated + t_post",
    model=cp.pymc_models.LinearRegression(sample_kwargs={"draws": 1000, ...})
)

level_only_result = cp.InterruptedTimeSeries(
    data=df, treatment_time=t_star,
    formula="y ~ 1 + t + treated",
    model=cp.pymc_models.LinearRegression(sample_kwargs={"draws": 1000, ...})
)

# Compare with LOO-CV
loo_full = az.loo(full_model_result.idata, pointwise=True)
loo_level = az.loo(level_only_result.idata, pointwise=True)

comparison = az.compare({"full": full_model_result.idata, "level_only": level_only_result.idata})
print(comparison)
```

</div>

The model with higher LOO ELPD (expected log pointwise predictive density) is preferred.

---

## Handling Multiple Seasonal Periods

Many policy-relevant outcomes have seasonal patterns (hospital admissions, crime rates, economic indicators). Ignoring seasonality inflates the residual variance and can create apparent "effects" at intervention dates that coincide with seasonal peaks or troughs.

### Approach 1: Month-of-Year Dummy Variables

Add indicator variables for each calendar month (or quarter, depending on data frequency):


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# Add month indicators to the dataframe
df["month"] = df["date"].dt.month

# Formula with monthly fixed effects
formula = "y ~ 1 + t + treated + t_post + C(month)"
```

</div>

This controls for monthly seasonal patterns without assuming a functional form.

### Approach 2: Fourier Terms

Use sine/cosine pairs to model smooth seasonal patterns:

$$\text{Seasonal}(t) = \sum_{k=1}^K \left[ a_k \sin\left(\frac{2\pi k t}{P}\right) + b_k \cos\left(\frac{2\pi k t}{P}\right) \right]$$

where $P$ is the seasonal period (12 for monthly data, 52 for weekly data).


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def add_fourier_terms(df: pd.DataFrame, period: int, n_terms: int = 2) -> pd.DataFrame:
    """
    Add Fourier sine/cosine terms for seasonal adjustment.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a column 't' with numeric time index
    period : int
        Seasonal period (12 for monthly, 4 for quarterly, 52 for weekly)
    n_terms : int
        Number of sine-cosine pairs (higher = more flexible seasonality)
    """
    df = df.copy()
    for k in range(1, n_terms + 1):
        df[f"sin_{k}"] = np.sin(2 * np.pi * k * df["t"] / period)
        df[f"cos_{k}"] = np.cos(2 * np.pi * k * df["t"] / period)
    return df

# Usage
df_seasonal = add_fourier_terms(df, period=12, n_terms=2)
formula = "y ~ 1 + t + treated + t_post + sin_1 + cos_1 + sin_2 + cos_2"
```

</div>

---

## Connections

<div class="callout-info">

<strong>How this connects to the rest of the course:</strong>

</div>

- **Builds on:** ITS Introduction (Guide 1)
- **Leads to:** CausalPy ITS API (Guide 3), Bayesian ITS (Module 02)
- **Related to:** Time series regression, structural break tests, HAC standard errors


## Practice Questions

### Question 1: Conceptual Check
**Question:** In your own words, explain the core concept of Segmented Regression for ITS and why it matters for practical applications. What problem does it solve that simpler approaches cannot?

### Question 2: Application
**Question:** Describe a real-world scenario where you would apply the techniques from this guide. What assumptions would you need to verify before proceeding?

## Further Reading

- Chatfield, C. (2003). *The Analysis of Time Series: An Introduction with R* (6th ed.) — comprehensive time series reference
- Newey, W.K. & West, K.D. (1987). "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica*
- Wagner, A.K. et al. (2002). "Segmented regression analysis of interrupted time series studies in medication use research." *Journal of Clinical Pharmacy and Therapeutics*


## Resources

<a class="link-card" href="../notebooks/01_its_smoking_ban.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises for this topic.</div>
</a>
