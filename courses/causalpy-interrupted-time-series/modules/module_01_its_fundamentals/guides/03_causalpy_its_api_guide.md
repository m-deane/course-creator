# CausalPy ITS API Walkthrough

> **Reading time:** ~9 min | **Module:** 1 — Its Fundamentals | **Prerequisites:** Module 0 — Causal Foundations

## In Brief

CausalPy's `InterruptedTimeSeries` class provides a high-level interface to Bayesian ITS models built on PyMC. It handles design matrix construction, model building, sampling, and visualization, while exposing enough flexibility to customize priors, formulas, and sampling parameters.

<div class="callout-key">

<strong>Key Concept:</strong> CausalPy's `InterruptedTimeSeries` class provides a high-level interface to Bayesian ITS models built on PyMC. It handles design matrix construction, model building, sampling, and visualization, while exposing enough flexibility to customize priors, formulas, and sampling parameters.

</div>

## Key Insight

CausalPy is a wrapper around PyMC that adds causal inference semantics: it knows which variables are the treatment, when the intervention occurs, and how to compute the counterfactual. You write the formula in Wilkinson notation and CausalPy handles the rest.

---

## Installation and Import


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Install (if not already done in Module 00)

# pip install causalpy pymc arviz

import causalpy as cp
import pymc as pm
import arviz as az
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

</div>
</div>

### Package Version Check


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
print(f"CausalPy: {cp.__version__}")
print(f"PyMC: {pm.__version__}")
print(f"ArviZ: {az.__version__}")
```

</div>
</div>

---

## Data Requirements

`InterruptedTimeSeries` requires a pandas DataFrame with:

1. **Numeric time variable:** An integer or float column indexing time periods
2. **Outcome variable:** The variable you want to estimate the causal effect on
3. **Pre-built indicator columns:** You must compute `treated` and `time_since_intervention` yourself

The function `treatment_time` accepts an integer index (row number in the DataFrame at which the intervention occurs).

### Canonical Data Preparation


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import numpy as np
import pandas as pd

def prepare_its_dataframe(
    dates: pd.Series,
    outcome: pd.Series,
    intervention_date,
    date_col: str = "date",
    outcome_col: str = "y",
) -> pd.DataFrame:
    """
    Prepare a DataFrame for CausalPy ITS analysis.

    Parameters
    ----------
    dates : pd.Series
        Series of dates (or any ordered time index)
    outcome : pd.Series
        Time series of the outcome variable
    intervention_date : any
        The date at which the intervention occurred (must be comparable to dates values)
    date_col : str
        Name for the date column in the output DataFrame
    outcome_col : str
        Name for the outcome column in the output DataFrame

    Returns
    -------
    pd.DataFrame with columns: date_col, outcome_col, t (numeric), treated, t_post
    And attribute: intervention_index (integer index of the first treated observation)
    """
    df = pd.DataFrame({
        date_col: dates,
        outcome_col: outcome,
    })

    # Numeric time index (0-based)
    df["t"] = np.arange(len(df))

    # Treatment indicator: 1 for all observations at or after intervention_date
    df["treated"] = (df[date_col] >= intervention_date).astype(float)

    # Time since intervention (0 in pre-period, 0 at t*, then 1, 2, 3, ...)
    # Note: the first post-intervention observation gets t_post = 0
    intervention_idx = df.index[df["treated"] == 1][0]
    df["t_post"] = np.maximum(df["t"] - intervention_idx, 0).astype(float)

    # Store intervention index as an attribute for use with CausalPy
    df.attrs["intervention_index"] = int(intervention_idx)

    return df
```

</div>
</div>

---

## The `InterruptedTimeSeries` Class

### Constructor Arguments


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
cp.InterruptedTimeSeries(
    data,              # pd.DataFrame — must contain all variables in formula
    treatment_time,    # int — index of the first treated observation
    formula,           # str — Wilkinson formula string
    model,             # cp.pymc_models.* — the PyMC model object
)
```

</div>
</div>

### Formula Specification

The formula follows the standard Python formula language (Wilkinson notation, via `formulaic`):


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Full ITS model: level change + slope change
formula = "y ~ 1 + t + treated + t_post"

# Level-only model: immediate effect, no slope change
formula = "y ~ 1 + t + treated"

# Slope-only model: gradual effect, no immediate jump
formula = "y ~ 1 + t + t_post"

# With seasonal controls (monthly dummy variables)
formula = "y ~ 1 + t + treated + t_post + C(calendar_month)"

# With Fourier seasonal terms
formula = "y ~ 1 + t + treated + t_post + sin_1 + cos_1 + sin_2 + cos_2"

# With additional covariate
formula = "y ~ 1 + t + treated + t_post + covariate_name"
```

</div>
</div>

**Important:** Every variable named in the formula must exist as a column in `data`.

---

## Model Objects

CausalPy provides several built-in model objects. For ITS, you will primarily use:

### `cp.pymc_models.LinearRegression`

Standard Gaussian linear regression with weakly informative default priors.


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
model = cp.pymc_models.LinearRegression(
    sample_kwargs={
        "draws": 1000,          # Posterior samples per chain
        "tune": 1000,           # Warmup/adaptation iterations
        "chains": 4,            # Number of parallel chains
        "target_accept": 0.9,  # NUTS acceptance rate (higher = more stable)
        "progressbar": True,    # Show sampling progress
        "random_seed": 42,      # For reproducibility
    }
)
```

</div>
</div>

### Sampling Parameter Guidelines

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `draws` | 1000–2000 | 1000 is usually sufficient for good estimates |
| `tune` | 1000 | Match draws for difficult models |
| `chains` | 4 | Minimum for R-hat convergence assessment |
| `target_accept` | 0.8–0.95 | 0.95 for hierarchical or complex models |
| `random_seed` | Any integer | Set for reproducibility |

---

## Fitting the Model


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import causalpy as cp

# Prepare data (see above)
df = prepare_its_dataframe(dates, outcome, intervention_date="2023-01-01")
t_star = df.attrs["intervention_index"]

# Fit the ITS model
result = cp.InterruptedTimeSeries(
    data=df,
    treatment_time=t_star,
    formula="y ~ 1 + t + treated + t_post",
    model=cp.pymc_models.LinearRegression(
        sample_kwargs={
            "draws": 1000,
            "tune": 1000,
            "chains": 4,
            "progressbar": True,
            "random_seed": 42,
        }
    ),
)
```

</div>
</div>

CausalPy will:
1. Build the design matrix from the formula
2. Construct the PyMC model with default priors
3. Run NUTS sampling
4. Store results in `result.idata` (ArviZ `InferenceData`)

---

## Accessing Results

### Summary Table


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Human-readable summary of posterior estimates
print(result.summary())

# Or use ArviZ for more control
summary = az.summary(
    result.idata,
    var_names=["Intercept", "t", "treated", "t_post", "sigma"],
    stat_focus="mean",
    hdi_prob=0.94,
)
print(summary)
```

</div>
</div>

### Raw Posterior Samples


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# All posterior samples (4 chains × 1000 draws × n_variables)
posterior = result.idata.posterior

# Extract specific parameter
beta_treated_samples = posterior["treated"].values.flatten()
print(f"Level change — Posterior mean: {beta_treated_samples.mean():.3f}")
print(f"Level change — 94% HDI: {az.hdi(beta_treated_samples, hdi_prob=0.94)}")
print(f"P(level change > 0): {(beta_treated_samples > 0).mean():.2%}")

# Extract slope change
beta_slope_samples = posterior["t_post"].values.flatten()
```

</div>
</div>

### Counterfactual Prediction


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# The InferenceData contains posterior predictive samples

# Including the counterfactual (mu_cf) computed by CausalPy

# Mean counterfactual trajectory
if "mu_cf" in result.idata.posterior:
    counterfactual_mean = result.idata.posterior["mu_cf"].mean(("chain", "draw"))
```

</div>
</div>

---

## Visualization

### Built-in Plot


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# CausalPy's standard ITS plot (two panels)
fig, axes = result.plot()

# Customize
axes[0].set_title("ITS Analysis: [Your Description]")
axes[0].set_xlabel("Time Period")
axes[0].set_ylabel("Outcome Variable")
axes[1].set_ylabel("Estimated Causal Impact")
plt.tight_layout()
plt.show()
```

</div>
</div>

The two-panel plot shows:
- **Top panel:** Observed data + fitted posterior mean + 94% posterior band + counterfactual
- **Bottom panel:** Estimated causal impact at each post-intervention time point

### Posterior Distribution Plots (ArviZ)


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python

# Posterior distributions of all parameters
az.plot_posterior(
    result.idata,
    var_names=["Intercept", "t", "treated", "t_post"],
    ref_val=0,
    figsize=(14, 4),
)
plt.tight_layout()
plt.show()

# Forest plot: compare multiple parameters
az.plot_forest(
    result.idata,
    var_names=["treated", "t_post"],
    figsize=(8, 3),
    hdi_prob=0.94,
)
plt.show()

# Trace plots for convergence assessment
az.plot_trace(
    result.idata,
    var_names=["treated", "t_post"],
    figsize=(12, 5),
)
plt.tight_layout()
plt.show()
```


---

## Convergence Diagnostics

Always check convergence before interpreting results.


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python

# R-hat (should be < 1.01 for all parameters)
rhat = az.rhat(result.idata)
print("R-hat values (should be < 1.01):")
for var in ["Intercept", "t", "treated", "t_post"]:
    if var in rhat:
        print(f"  {var}: {float(rhat[var].values):.4f}")

# Effective Sample Size (should be > 400)
ess = az.ess(result.idata)
print("\nEffective Sample Size (should be > 400):")
for var in ["Intercept", "t", "treated", "t_post"]:
    if var in ess:
        print(f"  {var}: {float(ess[var].values):.0f}")

# Divergences (should be 0 or very few)
n_divergences = result.idata.sample_stats["diverging"].sum().item()
print(f"\nNumber of divergences: {n_divergences}")
if n_divergences > 10:
    print("WARNING: High divergences. Consider increasing target_accept or reparameterizing.")
```


---

## Computing the Cumulative Causal Impact


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
def compute_cumulative_impact(
    result,
    var_level: str = "treated",
    var_slope: str = "t_post",
    n_post: int = None,
) -> dict:
    """
    Compute cumulative causal impact from ITS posterior samples.

    Parameters
    ----------
    result : CausalPy InterruptedTimeSeries result
    var_level : str
        Name of the level change variable in the posterior
    var_slope : str
        Name of the slope change variable in the posterior
    n_post : int
        Number of post-intervention periods

    Returns
    -------
    dict with 'mean', 'hdi_lower', 'hdi_upper', 'prob_positive'
    """
    posterior = result.idata.posterior
    beta_level = posterior[var_level].values.flatten()
    beta_slope = posterior[var_slope].values.flatten()

    if n_post is None:
        # Count from data
        n_post = int(result.summary()["post_obs"])

    k_values = np.arange(0, n_post)

    # Causal impact at each post-intervention time: beta_2 + beta_3 * k
    impact_matrix = (
        beta_level[:, np.newaxis] + beta_slope[:, np.newaxis] * k_values[np.newaxis, :]
    )
    cumulative_samples = impact_matrix.sum(axis=1)

    hdi = az.hdi(cumulative_samples, hdi_prob=0.94)

    return {
        "mean": float(cumulative_samples.mean()),
        "hdi_lower": float(hdi[0]),
        "hdi_upper": float(hdi[1]),
        "prob_positive": float((cumulative_samples > 0).mean()),
        "samples": cumulative_samples,
    }
```


---

## Custom PyMC Models

For advanced use cases, you can pass custom PyMC models to CausalPy. This allows you to:
- Specify informative priors
- Add AR(1) error structure
- Use non-Gaussian likelihoods (e.g., Poisson for count outcomes)

### Example: Informative Priors


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
import pymc as pm

class InformativeITSModel(cp.pymc_models.LinearRegression):
    """
    ITS model with domain-informed priors.
    Override the default weakly informative priors.
    """

    def build_model(self, X, y, coords):
        with pm.Model(coords=coords) as self.model:
            X_ = pm.Data("X", X, dims=["obs", "coeffs"])
            y_ = pm.Data("y_obs", y, dims=["obs"])

            # Domain-informed priors
            # E.g., we know the intervention probably reduced outcome (negative)
            # and the effect is likely between -20 and 0 units
            beta = pm.Normal(
                "beta",
                mu=[100, 0.5, -10, 0],   # Prior means for [intercept, t, treated, t_post]
                sigma=[20, 2, 8, 1],      # Prior std deviations
                dims=["coeffs"],
            )
            sigma = pm.HalfNormal("sigma", sigma=10)

            mu = pm.Deterministic("mu", pm.math.dot(X_, beta))
            y_hat = pm.Normal("y_hat", mu=mu, sigma=sigma, observed=y_, dims=["obs"])
```


---

## Complete Workflow Example


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
import causalpy as cp
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv("your_policy_data.csv", parse_dates=["date"])

# 2. Prepare ITS columns
df["t"] = np.arange(len(df))
intervention_idx = df.index[df["date"] >= "2023-01-01"][0]
df["treated"] = (df["t"] >= intervention_idx).astype(float)
df["t_post"] = np.maximum(df["t"] - intervention_idx, 0).astype(float)

# 3. Fit the model
result = cp.InterruptedTimeSeries(
    data=df,
    treatment_time=int(intervention_idx),
    formula="outcome ~ 1 + t + treated + t_post",
    model=cp.pymc_models.LinearRegression(
        sample_kwargs={"draws": 1000, "tune": 1000, "chains": 4, "random_seed": 42}
    ),
)

# 4. Check convergence
rhat_max = float(az.rhat(result.idata).to_array().max())
print(f"Max R-hat: {rhat_max:.4f}")
assert rhat_max < 1.02, "Convergence issue detected"

# 5. Visualize
fig, axes = result.plot()
plt.tight_layout()
plt.show()

# 6. Extract key estimates
posterior = result.idata.posterior
level_change = posterior["treated"].values.flatten()
print(f"\nLevel change: {level_change.mean():.2f} (94% HDI: {az.hdi(level_change, hdi_prob=0.94)})")
print(f"P(positive effect): {(level_change > 0).mean():.1%}")
```


---

## Common Errors and Solutions

| Error | Likely Cause | Solution |
|-------|-------------|----------|
| `KeyError: 'treated'` | Column name mismatch | Ensure formula variables match DataFrame column names |
| `Shape mismatch` | Wrong `treatment_time` index | Check that `treatment_time` is the integer row index, not a date |
| `High R-hat (> 1.1)` | Sampler not converged | Increase `tune`, increase `draws`, or increase `target_accept` |
| `Many divergences` | Model misspecification or scale issues | Reparameterize or use stronger priors |
| `Slow sampling` | Large dataset or complex formula | Reduce `chains`, use JAX backend, or simplify formula |
| `SamplingWarning: ESS low` | Too few effective samples | Increase `draws`, check for high autocorrelation in chains |

---


## Practice Questions

### Question 1: Conceptual Check
**Question:** In your own words, explain the core concept of CausalPy ITS API Walkthrough and why it matters for practical applications. What problem does it solve that simpler approaches cannot?

### Question 2: Application
**Question:** Describe a real-world scenario where you would apply the techniques from this guide. What assumptions would you need to verify before proceeding?

## Connections

<div class="callout-info">

<strong>How this connects to the rest of the course:</strong>


- **Builds on:** ITS Introduction (Guide 1), Segmented Regression (Guide 2)
- **Leads to:** Notebooks 1–3 in this module, Bayesian ITS internals (Module 02)
- **Related to:** ArviZ documentation, PyMC documentation

## Reference: CausalPy API Summary

| Attribute/Method | Type | Description |
|-----------------|------|-------------|
| `result.idata` | `az.InferenceData` | Full ArviZ InferenceData with posterior, prior, observed |
| `result.plot()` | figure, axes | Two-panel standard ITS visualization |
| `result.summary()` | DataFrame | Posterior mean, HDI, R-hat for all parameters |
| `result.model` | `pm.Model` | The underlying PyMC model object |
| `result.formula` | str | The formula used to fit the model |


## Resources

<a class="link-card" href="../notebooks/01_its_smoking_ban.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises for this topic.</div>
</a>
