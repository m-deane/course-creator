# CausalPy's DifferenceInDifferences API

> **Reading time:** ~8 min | **Module:** 4 — Difference In Differences | **Prerequisites:** Module 1 — ITS Fundamentals

## Learning Objectives

By the end of this guide, you will be able to:
1. Prepare panel data in the format CausalPy's DiD class expects
2. Specify the DiD formula with interaction terms correctly
3. Interpret the Bayesian DiD output including posterior distributions
4. Use CausalPy's built-in plotting functions for DiD results
5. Run sensitivity analyses by adjusting priors and model specifications

---

## 1. CausalPy's DiD Architecture

CausalPy implements DiD through its `DifferenceInDifferences` class in the `causalpy.experiments` module. Unlike standard TWFE regression, CausalPy's implementation is **Bayesian by default**, providing:

- Posterior distributions over the treatment effect
- Credible intervals (not just confidence intervals)
- Natural uncertainty quantification
- Prior elicitation for domain knowledge incorporation

The class supports both PyMC (Bayesian) and scikit-learn (frequentist) backends via the `model` argument.

---

## 2. Data Format Requirements

CausalPy's `DifferenceInDifferences` expects a `pandas.DataFrame` with:

| Column | Type | Description |
|--------|------|-------------|
| Outcome variable | float | The variable whose change you're measuring |
| Group variable | int/bool | 1 = treated group, 0 = control group |
| Time variable | int/bool | 1 = post-treatment period, 0 = pre-treatment period |
| Any covariates | float/int | Additional control variables |

**Key constraint:** The time variable is binary (pre/post) in the canonical DiD. For panel extensions, you can encode multiple periods as integer counts.

### Preparing Your Data

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import pandas as pd
import numpy as np

# Example: Card & Krueger (1994) style minimum wage data
# New Jersey (treated) vs Pennsylvania (control)
# Before and after NJ minimum wage increase in 1992

nj_data = pd.DataFrame({
    "state": ["NJ"] * 410 + ["PA"] * 357,
    "period": (["pre"] * 205 + ["post"] * 205) + (["pre"] * 178 + ["post"] * 179),
    "fte_employment": np.concatenate([
        np.random.normal(20.4, 8.0, 205),   # NJ pre
        np.random.normal(21.0, 8.5, 205),   # NJ post (with treatment effect)
        np.random.normal(23.3, 7.5, 178),   # PA pre
        np.random.normal(21.2, 7.8, 179),   # PA post (slight decline)
    ]),
    "chain": np.random.choice(["Burger King", "KFC", "Wendys", "Roy Rogers"], 767)
})

# Create binary variables required by CausalPy
nj_data["treated"] = (nj_data["state"] == "NJ").astype(int)
nj_data["post"] = (nj_data["period"] == "post").astype(int)

print(nj_data.head())
print(f"\nShape: {nj_data.shape}")
print(f"\nGroup counts:\n{nj_data.groupby(['state', 'period']).size()}")
```

</div>

---

## 3. Fitting the DiD Model

### Basic Syntax

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import causalpy as cp

result = cp.DifferenceInDifferences(
    data=nj_data,
    formula="fte_employment ~ 1 + post + treated + post:treated",
    time_variable_name="post",
    group_variable_name="treated",
    model=cp.pymc_models.LinearRegression(
        sample_kwargs={"draws": 2000, "tune": 1000, "chains": 4, "target_accept": 0.9}
    )
)
```

</div>

### Formula Structure

The formula must include:
- `1`: intercept (the control group pre-period mean)
- `post`: the time trend shared by both groups
- `treated`: the group-level difference in pre-period means
- `post:treated`: **the interaction — this is the DiD treatment effect**

You can add covariates to control for observed differences:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
# With covariates
result = cp.DifferenceInDifferences(
    data=nj_data,
    formula="fte_employment ~ 1 + post + treated + post:treated + C(chain)",
    time_variable_name="post",
    group_variable_name="treated",
    model=cp.pymc_models.LinearRegression()
)
```

</div>

### Using the Frequentist Backend

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
from sklearn.linear_model import LinearRegression as SKLinReg

result_freq = cp.DifferenceInDifferences(
    data=nj_data,
    formula="fte_employment ~ 1 + post + treated + post:treated",
    time_variable_name="post",
    group_variable_name="treated",
    model=cp.skl_models.LinearRegression()
)
```

</div>

---

## 4. Understanding the Output

### Summary Table

```python
print(result.summary())
```

The summary displays:
- `Intercept`: control group mean in pre-period
- `post`: change over time for the control group (counterfactual time trend)
- `treated`: pre-period difference between treated and control
- `post:treated`: **the DiD treatment effect estimate**

For the Bayesian backend, you get:
- Posterior mean
- Posterior standard deviation
- 94% highest density interval (HDI)
- $\hat{R}$ convergence diagnostic

### Accessing the Posterior

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
# Get the posterior samples for the treatment effect
treatment_effect_samples = result.idata.posterior["post:treated"].values.flatten()

print(f"Posterior mean: {treatment_effect_samples.mean():.3f}")
print(f"Posterior std: {treatment_effect_samples.std():.3f}")
print(f"94% HDI: {np.percentile(treatment_effect_samples, [3, 97])}")
print(f"P(effect > 0): {(treatment_effect_samples > 0).mean():.3f}")
```

</div>

The full posterior lets you compute any summary statistic and answer questions like "What is the probability the effect exceeds 1 FTE employee?"

---

## 5. Built-in Plotting

### Main DiD Plot

```python
fig, ax = result.plot()
```

CausalPy's DiD plot shows:
- Observed group means in pre and post periods
- The treatment effect with credible interval
- The counterfactual — what the treated group would have looked like without treatment

### Customising the Plot

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# DiD plot on left axis
result.plot(ax=axes[0])
axes[0].set_title("DiD Estimate: FTE Employment")
axes[0].set_xlabel("Period")
axes[0].set_ylabel("FTE Employment")

# Posterior distribution of treatment effect on right
treatment_samples = result.idata.posterior["post:treated"].values.flatten()
axes[1].hist(treatment_samples, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
axes[1].axvline(x=0, color="red", linestyle="--", label="Zero effect")
axes[1].axvline(x=treatment_samples.mean(), color="black", linestyle="-", label="Posterior mean")
axes[1].set_xlabel("Treatment Effect (FTE employees)")
axes[1].set_ylabel("Posterior density")
axes[1].set_title("Posterior Distribution of DiD Treatment Effect")
axes[1].legend()

plt.tight_layout()
plt.show()
```

</div>

### Trace Plots for Diagnostics

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import arviz as az

# Check MCMC convergence
az.plot_trace(result.idata, var_names=["post:treated", "post", "treated"])
plt.suptitle("MCMC Trace Plots")
plt.tight_layout()
plt.show()

# Check R-hat convergence statistics
az.summary(result.idata, var_names=["post:treated"])
```

</div>

---

## 6. Prior Specification

CausalPy's Bayesian backend lets you specify informative priors when you have domain knowledge.

### Default Priors

By default, CausalPy uses weakly informative priors:
- Coefficients: $\mathcal{N}(0, 10)$
- Outcome noise: $\text{HalfNormal}(1)$

### Specifying Custom Priors

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
# Suppose we believe the treatment effect is small and positive
# based on economic theory (min wage increases employment slightly or is neutral)
result_informative = cp.DifferenceInDifferences(
    data=nj_data,
    formula="fte_employment ~ 1 + post + treated + post:treated",
    time_variable_name="post",
    group_variable_name="treated",
    model=cp.pymc_models.LinearRegression(
        priors={
            "Intercept": {"dist": "Normal", "kwargs": {"mu": 20, "sigma": 5}},
            "post": {"dist": "Normal", "kwargs": {"mu": 0, "sigma": 3}},
            "treated": {"dist": "Normal", "kwargs": {"mu": -3, "sigma": 3}},
            "post:treated": {"dist": "Normal", "kwargs": {"mu": 0, "sigma": 5}},
            "sigma": {"dist": "HalfNormal", "kwargs": {"sigma": 5}},
        }
    )
)
```

</div>

### Prior Sensitivity Analysis

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
# Compare three prior specifications
prior_configs = [
    {"post:treated": {"dist": "Normal", "kwargs": {"mu": 0, "sigma": 2}}},   # tight
    {"post:treated": {"dist": "Normal", "kwargs": {"mu": 0, "sigma": 10}}},  # diffuse
    {"post:treated": {"dist": "Normal", "kwargs": {"mu": 1, "sigma": 5}}},   # informative
]

estimates = []
for priors in prior_configs:
    r = cp.DifferenceInDifferences(
        data=nj_data,
        formula="fte_employment ~ 1 + post + treated + post:treated",
        time_variable_name="post",
        group_variable_name="treated",
        model=cp.pymc_models.LinearRegression(priors=priors)
    )
    samples = r.idata.posterior["post:treated"].values.flatten()
    estimates.append({"mean": samples.mean(), "hdi_3": np.percentile(samples, 3),
                      "hdi_97": np.percentile(samples, 97)})

# Plot prior sensitivity
fig, ax = plt.subplots(figsize=(8, 4))
labels = ["Tight prior", "Diffuse prior", "Informative prior"]
for i, (est, label) in enumerate(zip(estimates, labels)):
    ax.errorbar(est["mean"], i, xerr=[[est["mean"]-est["hdi_3"]], [est["hdi_97"]-est["mean"]]],
                fmt="o", capsize=5, label=label)
ax.axvline(x=0, color="red", ls="--", alpha=0.5)
ax.set_xlabel("Treatment Effect Estimate")
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
ax.set_title("Prior Sensitivity Analysis")
plt.tight_layout()
plt.show()
```

</div>

---

## 7. Model Checks and Diagnostics

### Posterior Predictive Check

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
# Does the model generate data that looks like the observed data?
with result.model:
    ppc = pm.sample_posterior_predictive(result.idata)

az.plot_ppc(ppc, observed_ys="fte_employment")
plt.title("Posterior Predictive Check")
plt.show()
```

</div>

### Convergence Diagnostics

```python
# R-hat should be < 1.01 for convergence
print(az.summary(result.idata)[["mean", "sd", "hdi_3%", "hdi_97%", "r_hat", "ess_bulk"]])
```

If $\hat{R} > 1.01$:
- Increase `tune` samples
- Increase `target_accept` (towards 0.95)
- Check for model misspecification
- Reparameterise if necessary

---

## 8. Complete Worked Example

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import causalpy as cp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

# Simulate Card & Krueger style data
np.random.seed(2024)
n_nj, n_pa = 410, 357
TRUE_EFFECT = 2.75  # true treatment effect in FTE

data = pd.DataFrame({
    "fte": np.concatenate([
        np.random.normal(20.4, 8.0, n_nj // 2),        # NJ pre
        np.random.normal(20.4 + TRUE_EFFECT, 8.5, n_nj // 2),  # NJ post
        np.random.normal(23.3, 7.5, n_pa // 2),        # PA pre
        np.random.normal(22.8, 7.8, n_pa // 2),        # PA post (slight decline)
    ]),
    "treated": [1] * n_nj + [0] * n_pa,
    "post": ([0] * (n_nj // 2) + [1] * (n_nj // 2) +
             [0] * (n_pa // 2) + [1] * (n_pa // 2)),
    "state": ["NJ"] * n_nj + ["PA"] * n_pa,
})

# Fit Bayesian DiD
result = cp.DifferenceInDifferences(
    data=data,
    formula="fte ~ 1 + post + treated + post:treated",
    time_variable_name="post",
    group_variable_name="treated",
    model=cp.pymc_models.LinearRegression(
        sample_kwargs={"draws": 2000, "tune": 1000, "chains": 4}
    )
)

# Convergence check
summary = az.summary(result.idata)
print("Convergence diagnostics:")
print(summary[["mean", "sd", "r_hat", "ess_bulk"]])
print(f"\nTrue treatment effect: {TRUE_EFFECT:.2f}")

# Extract treatment effect
tau_samples = result.idata.posterior["post:treated"].values.flatten()
print(f"Posterior mean: {tau_samples.mean():.2f}")
print(f"94% HDI: [{np.percentile(tau_samples, 3):.2f}, {np.percentile(tau_samples, 97):.2f}]")
print(f"P(tau > 0): {(tau_samples > 0).mean():.3f}")

# Plot
result.plot()
plt.suptitle(f"DiD: NJ vs PA Fast Food Employment\nTrue effect = {TRUE_EFFECT:.2f} FTE")
plt.tight_layout()
plt.show()
```

</div>

---

## 9. API Reference Summary

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | DataFrame | Panel data in long format |
| `formula` | str | Patsy formula with interaction term |
| `time_variable_name` | str | Name of the binary time column |
| `group_variable_name` | str | Name of the binary group column |
| `model` | Model object | PyMC or sklearn model backend |
| `priors` | dict | Prior distributions for coefficients |

| Method | Returns | Description |
|--------|---------|-------------|
| `.summary()` | str | Coefficient table with HDIs |
| `.plot()` | fig, axes | DiD visualisation |
| `.idata` | InferenceData | Full posterior samples (Bayesian) |

---

## 10. Summary

CausalPy's `DifferenceInDifferences` class brings Bayesian inference to the DiD framework:

- Posterior distributions provide richer uncertainty quantification than p-values
- Prior elicitation lets you incorporate domain knowledge
- Built-in plots show the counterfactual and treatment effect visually
- MCMC diagnostics flag convergence issues before you interpret results

The key to correct usage is:
1. Ensure your data has binary group and time variables
2. Include the `post:treated` interaction in your formula
3. Check convergence ($\hat{R} < 1.01$) before interpreting posteriors
4. Use prior sensitivity analysis to check robustness

---

**Previous:** [02 — Staggered DiD](02_staggered_did_guide.md)
**Next:** [Module 05 — Regression Discontinuity](../../module_05_regression_discontinuity/guides/01_rdd_fundamentals_guide.md)

<div class="callout-key">
<strong>Key Concept:</strong> **Previous:** [02 — Staggered DiD](02_staggered_did_guide.md)
**Next:** [Module 05 — Regression Discontinuity](../../module_05_regression_discontinuity/guides/01_rdd_fundamentals_guide.md)
</div>



## Resources

<a class="link-card" href="../notebooks/01_did_labour_economics.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises for this topic.</div>
</a>
