---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# CausalPy DiD API

## Bayesian Difference-in-Differences in Practice

Module 04.3 | Causal Inference with CausalPy

<!-- Speaker notes: We've built the theoretical foundation. Now we implement. CausalPy's DifferenceInDifferences class wraps the PyMC Bayesian backend with a clean interface. The Bayesian approach gives us more than point estimates — we get full posterior distributions, natural uncertainty quantification, and the ability to incorporate prior knowledge. -->

---

## CausalPy DiD at a Glance

```python
import causalpy as cp

result = cp.DifferenceInDifferences(
    data=df,
    formula="outcome ~ 1 + post + treated + post:treated",
    time_variable_name="post",
    group_variable_name="treated",
    model=cp.pymc_models.LinearRegression()
)

result.plot()
print(result.summary())
```

Three key arguments: `data`, `formula`, `model`

<!-- Speaker notes: This is all it takes to run a Bayesian DiD in CausalPy. Three required arguments — data, formula, model. The formula drives the interpretation. The model argument controls whether you're going Bayesian with PyMC or frequentist with sklearn. Let's break down each component. -->

---

## Data Format

CausalPy expects **long-format panel data** with binary encodings:

| Column | Values | Role |
|--------|--------|------|
| `outcome` | continuous | What you're measuring |
| `treated` | 0 or 1 | Group indicator |
| `post` | 0 or 1 | Time indicator (before/after) |
| covariates | any | Optional controls |

```python
# Ensure binary columns
df["treated"] = (df["group"] == "treatment").astype(int)
df["post"] = (df["period"] == "after").astype(int)
```

<!-- Speaker notes: Data preparation is often where DiD analyses go wrong. CausalPy needs binary 0/1 indicators for the group and time variables. If your group column has strings like "NJ" and "PA", you need to create a binary indicator. Same for periods: convert "pre" and "post" or actual years to a binary 0/1 variable. For multi-period data, you'll need to define what "post" means — usually a binary indicator for whether the period is at or after the treatment date. -->

---

## The Formula: Every Term Matters

$$\text{outcome} \sim 1 + \text{post} + \text{treated} + \text{post:treated}$$

| Term | Coefficient | Meaning |
|------|-------------|---------|
| `1` (Intercept) | $\alpha$ | Control group, pre-period baseline |
| `post` | $\beta$ | Time trend: control group change |
| `treated` | $\gamma$ | Pre-period group difference |
| `post:treated` | **$\tau$** | **DiD treatment effect** |

The `:` in `post:treated` creates the interaction term.

<!-- Speaker notes: Every term in this formula has a specific causal interpretation. The intercept is your control group's pre-period average. The post coefficient captures the control group's change over time — this is your counterfactual time trend. The treated coefficient captures the pre-period level difference between groups. And the interaction is the DiD estimate — the additional change for the treated group beyond what the control group experienced. -->

---

## Adding Covariates

Control for observed differences with additional terms:

```python
# Add restaurant chain fixed effects
formula = "fte ~ 1 + post + treated + post:treated + C(chain)"

# Add continuous covariate
formula = "wages ~ 1 + post + treated + post:treated + log_employment"

# Multiple covariates
formula = ("outcome ~ 1 + post + treated + post:treated "
           "+ age + C(sector) + urban_dummy")
```

Covariates increase precision; they do **not** fix parallel trends violations.

<!-- Speaker notes: Covariates in DiD serve two purposes: precision and partial addressing of confounding. If certain observable characteristics drive parallel trends violations, controlling for them can make the assumption more plausible — this is "conditional parallel trends." But if parallel trends is violated for unobserved reasons, adding covariates doesn't help. The interaction term is always your treatment effect; the covariates just let you condition on observables. -->

---

## Model Backend Options

<div class="columns">

**Bayesian (default):**
```python
model=cp.pymc_models.LinearRegression(
    sample_kwargs={
        "draws": 2000,
        "tune": 1000,
        "chains": 4,
        "target_accept": 0.9
    }
)
```

**Frequentist:**
```python
model=cp.skl_models.LinearRegression()
```

</div>

Bayesian: full posterior distributions
Frequentist: point estimates + confidence intervals

<!-- Speaker notes: CausalPy supports both backends. The Bayesian backend uses PyMC under the hood — full MCMC sampling, posterior distributions, credible intervals. The frequentist backend uses scikit-learn — fast, no MCMC, standard OLS. For exploratory work or large datasets, sklearn is faster. For final analysis and reporting, PyMC gives you richer uncertainty quantification. For production pipelines where you care about full uncertainty, Bayesian is the right choice. -->

---

## Interpreting the Posterior

The key output is the posterior distribution over $\tau$ (`post:treated`):

```python
import arviz as az
import numpy as np

# Extract posterior samples
tau = result.idata.posterior["post:treated"].values.flatten()

# Summary statistics
print(f"Posterior mean:  {tau.mean():.3f}")
print(f"Posterior std:   {tau.std():.3f}")
print(f"94% HDI: [{np.percentile(tau, 3):.3f}, {np.percentile(tau, 97):.3f}]")
print(f"P(τ > 0): {(tau > 0).mean():.3f}")
```

Posterior mean > 0 AND 94% HDI excludes 0 → strong evidence of positive effect

<!-- Speaker notes: The posterior distribution is far more informative than a single p-value. The mean gives you the best single estimate. The HDI tells you where 94% of the posterior probability lies — analogous to a confidence interval but with a direct probabilistic interpretation. And P(tau > 0) is a direct answer to "how sure are we the effect is positive?" A p-value from frequentist regression doesn't have that direct interpretation; the posterior probability does. -->

---

## Built-in Plotting

```python
# Main DiD visualisation
fig, ax = result.plot()
```

The plot shows:
- **Blue dots:** observed group means (pre and post)
- **Dashed line:** counterfactual (treated without treatment)
- **Shaded region:** credible interval on treatment effect
- **Arrow:** treatment effect magnitude

<!-- Speaker notes: CausalPy's DiD plot is designed for communication. The counterfactual line — what would have happened absent treatment — is explicitly shown, which makes the causal reasoning transparent. The shaded region around the treatment effect quantifies uncertainty visually. This plot can go directly into a presentation or paper with minimal modification. -->

<div class="callout-insight">
Insight:  observed group means (pre and post)
- 
</div>

---

## Customising Plots

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: DiD plot
result.plot(ax=axes[0])
axes[0].set_title("Employment: NJ vs PA")

# Right: posterior distribution
tau = result.idata.posterior["post:treated"].values.flatten()
axes[1].hist(tau, bins=60, color="steelblue", alpha=0.8, edgecolor="white")
axes[1].axvline(0, color="red", ls="--", label="No effect")
axes[1].axvline(tau.mean(), color="black", ls="-", label=f"Mean = {tau.mean():.2f}")
axes[1].fill_betweenx(
    [0, axes[1].get_ylim()[1]],
    np.percentile(tau, 3), np.percentile(tau, 97),
    alpha=0.2, color="steelblue", label="94% HDI"
)
axes[1].legend()
axes[1].set_xlabel("Treatment Effect (FTE)")
axes[1].set_title("Posterior Distribution")

plt.tight_layout()
plt.show()
```

<!-- Speaker notes: Combining CausalPy's default plot with a custom posterior histogram gives you a powerful two-panel figure. The left shows the DiD in outcome space — the actual means and counterfactual. The right shows the distribution of the treatment effect estimate, which communicates uncertainty directly. This two-panel format is ideal for papers and presentations. -->

---

## MCMC Convergence Diagnostics

**Always check before interpreting:**

```python
az.summary(result.idata)[["mean", "sd", "r_hat", "ess_bulk"]]
```

| Diagnostic | Good | Warning |
|-----------|------|---------|
| $\hat{R}$ | < 1.01 | > 1.01: chains not mixed |
| ESS bulk | > 400 | < 100: inefficient sampling |
| ESS tail | > 400 | Low: tail estimates unreliable |

Also check trace plots: `az.plot_trace(result.idata)`

<!-- Speaker notes: MCMC diagnostics are non-negotiable before interpreting results. R-hat measures whether multiple chains are exploring the same part of the parameter space — it should be very close to 1.0. ESS (effective sample size) tells you how many independent samples you effectively have after accounting for autocorrelation. Low ESS means your credible intervals might be too narrow. If either diagnostic fails, don't report results — fix the sampling first. -->

<div class="callout-warning">
Warning: Always check before interpreting:
</div>

---

## Prior Specification

```python
model = cp.pymc_models.LinearRegression(
    priors={
        "Intercept": {
            "dist": "Normal",
            "kwargs": {"mu": 20, "sigma": 5}   # domain knowledge: ~20 FTE
        },
        "post:treated": {
            "dist": "Normal",
            "kwargs": {"mu": 0, "sigma": 5}     # neutral prior on effect
        },
        "sigma": {
            "dist": "HalfNormal",
            "kwargs": {"sigma": 5}
        }
    }
)
```

<!-- Speaker notes: Prior specification is where Bayesian DiD adds unique value. If you're running a food policy study and you know from previous research that minimum wage effects on employment are typically between -5% and +5%, you can encode that. This makes your analysis more efficient — you're using all available information, not just the data in front of you. It also makes the assumptions explicit and debatable, which is scientifically healthier than pretending you have no prior knowledge. -->

---

## Prior Sensitivity Analysis

Does the result hold across different prior beliefs?

```python
priors_list = [
    ("Tight", {"post:treated": {"dist": "Normal", "kwargs": {"mu": 0, "sigma": 2}}}),
    ("Diffuse", {"post:treated": {"dist": "Normal", "kwargs": {"mu": 0, "sigma": 20}}}),
    ("Positive lean", {"post:treated": {"dist": "Normal", "kwargs": {"mu": 2, "sigma": 5}}}),
]

results = {}
for name, priors in priors_list:
    r = cp.DifferenceInDifferences(
        data=df, formula="y ~ 1 + post + treated + post:treated",
        time_variable_name="post", group_variable_name="treated",
        model=cp.pymc_models.LinearRegression(priors=priors)
    )
    results[name] = r.idata.posterior["post:treated"].values.flatten()
```

If estimates are stable across priors → result is data-driven

<!-- Speaker notes: Prior sensitivity analysis is a best practice in Bayesian analysis. You're checking: does my conclusion change if I had a different prior belief? If the posterior is similar across tight, diffuse, and directionally different priors, the data is doing the heavy lifting and the result is robust. If the posterior is very sensitive to priors, you probably don't have enough data to be confident, and you should report that uncertainty clearly. -->

---

## Posterior Predictive Check

Does the model generate realistic data?

```python
import pymc as pm

with result.model:
    ppc = pm.sample_posterior_predictive(result.idata)

az.plot_ppc(ppc, observed_ys="outcome")
plt.title("Posterior Predictive Check")
```

If observed data lies within the posterior predictive distribution → model is reasonable

<!-- Speaker notes: The posterior predictive check asks: if we simulate data from our fitted model, does it look like the real data? If the observed distribution is in the tails of the predictive distribution, the model is missing something important — maybe there's heteroscedasticity, or the outcome is not normally distributed. A good predictive check is necessary before reporting results, though a perfect fit is not required. -->

---

## Frequentist vs Bayesian Output

<div class="columns">

**Frequentist (sklearn):**
```
Coef    Std Err   t-stat  p-value
β₀  20.4   0.4      51.0   <0.001
β₁  -1.3   0.6      -2.2   0.027
β₂  2.9    0.6       4.8   <0.001
τ   2.7    0.8       3.4   0.001
```

**Bayesian (PyMC):**
```
Mean   SD    HDI 3%  HDI 97%  R̂
20.4   0.4   19.7    21.1    1.00
-1.3   0.6   -2.4    -0.2    1.00
2.9    0.6    1.8     4.1    1.00
2.7    0.8    1.2     4.2    1.00
```

</div>

Interpretation: 94% HDI = "94% posterior probability the effect is in this range"

<!-- Speaker notes: Both give you similar point estimates and similar uncertainty ranges — in large samples, credible intervals and confidence intervals often coincide numerically. The interpretation differs. The Bayesian HDI has a direct probability statement: "There is 94% posterior probability the treatment effect is between 1.2 and 4.2 FTE." The frequentist confidence interval only says: "If we repeated this experiment many times, 95% of the intervals constructed this way would contain the true parameter." Most practitioners prefer the Bayesian interpretation for communication. -->

---

## Complete Workflow

```python
# 1. Prepare data
df = prepare_panel_data(raw_df)

# 2. Exploratory: plot pre-trends
plot_pretrends(df, group="treated", time="period", outcome="y")

# 3. Fit model
result = cp.DifferenceInDifferences(
    data=df, formula="y ~ 1 + post + treated + post:treated",
    time_variable_name="post", group_variable_name="treated",
    model=cp.pymc_models.LinearRegression()
)

# 4. Diagnostics
az.summary(result.idata)  # R-hat, ESS

# 5. Interpret
result.plot()
tau = result.idata.posterior["post:treated"].values.flatten()
print(f"Treatment effect: {tau.mean():.2f} [{np.percentile(tau,3):.2f}, {np.percentile(tau,97):.2f}]")

# 6. Sensitivity
run_prior_sensitivity(df)
```

<!-- Speaker notes: This is the production workflow. Notice there are six steps and interpretation comes fifth — after diagnostics. This order matters. Too many analysts jump straight to the point estimate without checking convergence or model fit. Following this workflow ensures that what you report is reliable. Also notice that the workflow ends with sensitivity analysis — you should always check if your conclusion holds under different modelling choices. -->

<div class="callout-key">
Key Point: Always plot the pre-treatment trends for treated and control groups before running DiD -- visual inspection catches violations that statistical tests miss.
</div>

---

## Summary

| Component | Key Point |
|-----------|-----------|
| Data format | Long panel, binary group/time columns |
| Formula | Always include `post:treated` interaction |
| Bayesian output | Posterior distributions, HDIs, P(τ>0) |
| Convergence | Check R̂ < 1.01 and ESS > 400 |
| Priors | Use domain knowledge; run sensitivity analysis |
| Plotting | `result.plot()` shows counterfactual and effect |

<!-- Speaker notes: These are the six things to remember for CausalPy DiD. Data format matters — wrong encoding gives wrong results. Formula structure is non-negotiable — the interaction is the treatment effect. Bayesian output is richer but requires convergence checking. Priors are a feature, not a bug — use them. And always plot — the visual makes the causal reasoning transparent. -->

<div class="callout-info">
Info: CausalPy's DiD implementation supports both frequentist and Bayesian estimation. Start with frequentist for speed, switch to Bayesian for richer uncertainty quantification.
</div>

---

<!-- _class: lead -->

## Next: Module 05 — Regression Discontinuity

When treatment assignment depends on crossing a **threshold**

→ [Module 05 — RDD Fundamentals](../../module_05_regression_discontinuity/guides/01_rdd_fundamentals_guide.md)

<!-- Speaker notes: We've now covered DiD thoroughly — both the theory and the CausalPy implementation. Next we move to regression discontinuity designs, where identification comes not from a comparison group but from the discontinuity in treatment probability at a known threshold. RDD is one of the most credibly causal designs in economics and policy research. -->
