# Synthetic Control: Building a Counterfactual from Donor Units

> **Reading time:** ~8 min | **Module:** 3 — Synthetic Control | **Prerequisites:** Module 1 — ITS Fundamentals

## In Brief

Synthetic control (SC) constructs a counterfactual for a treated unit by forming a weighted
combination of untreated "donor" units. The weights are chosen so the weighted donor average
closely matches the treated unit's pre-intervention trajectory. After the intervention, any
divergence between the treated unit and its synthetic counterpart is the estimated causal effect.

<div class="callout-key">

<strong>Key Concept:</strong> Synthetic control (SC) constructs a counterfactual for a treated unit by forming a weighted
combination of untreated "donor" units. The weights are chosen so the weighted donor average
closely matches the treated unit's pre-intervention trajectory.

</div>

## The Core Problem That SC Solves

ITS requires that the pre-intervention trend would have continued unchanged without the
intervention. This assumption is violated when:

- The intervention coincides with a macroeconomic shock (recession, epidemic, policy change)
- Seasonal patterns shift around the intervention date
- The treated unit was selected for intervention because it was trending unusually (regression to the mean)

In all these cases, an untreated comparison group provides a natural control for what would
have happened without the intervention. But what if there is no single good comparison unit?
Synthetic control answers: build a weighted combination of many units that collectively
match the treated unit.

---

## The California Tobacco Study

The canonical synthetic control paper (Abadie, Diamond, and Hainmueller, 2010) studied the
1988 California Proposition 99, which raised the cigarette tax by 25 cents per pack and funded
anti-smoking campaigns. California was the first US state to implement such a sweeping policy.

**Design:**
- Treated unit: California
- Donors: 38 other US states that did not implement similar policies
- Outcome: per-capita cigarette consumption (packs per year)
- Pre-intervention: 1970–1988 (19 years)
- Post-intervention: 1989–2000 (12 years)

The synthetic California is a weighted average of specific donor states (primarily Colorado,
Connecticut, Montana, Nevada, Utah) chosen so that the pre-1988 cigarette consumption trajectory
of the synthetic California exactly matches the real California.

---

## Formal Setup

### Notation

- $J + 1$ units total: unit 1 is treated, units $2, \ldots, J+1$ are donors
- $T$ time periods: pre-intervention $t = 1, \ldots, T_0$; post-intervention $t = T_0 + 1, \ldots, T$
- $Y_{jt}$ is the observed outcome for unit $j$ at time $t$
- $Y_{1t}(0)$ is the potential outcome for the treated unit without intervention (counterfactual)

### The Synthetic Control Estimator

Choose donor weights $\mathbf{w} = (w_2, \ldots, w_{J+1})$ with $w_j \geq 0$ and $\sum_j w_j = 1$
to minimize the pre-intervention discrepancy:

$$\mathbf{w}^* = \arg\min_{\mathbf{w}} \left\| \mathbf{X}_1 - \mathbf{X}_0 \mathbf{w} \right\|_V$$

where:
- $\mathbf{X}_1$ is a vector of pre-intervention characteristics for the treated unit
- $\mathbf{X}_0$ is the matrix of pre-intervention characteristics for donor units
- $\|\cdot\|_V$ is a weighted norm determined by predictor importance matrix $V$

The counterfactual is then:

$$\widehat{Y}_{1t}(0) = \sum_{j=2}^{J+1} w_j^* Y_{jt} \quad \text{for } t > T_0$$

The estimated treatment effect at time $t$ is:

$$\hat{\alpha}_{1t} = Y_{1t} - \widehat{Y}_{1t}(0)$$

---

## Why Weights Instead of Regression?

A natural question: why not just run a regression of the treated unit's pre-period outcome on the
donor units' pre-period outcomes?

Several problems with regression:

1. **Extrapolation**: regression can assign large positive and negative coefficients that cancel out.
   The synthetic control forces $w_j \geq 0$, which prevents this extrapolation and makes the
   counterfactual interpretable as a convex combination of real observed units.

2. **Overfitting**: with many donor units and few pre-periods, regression will perfectly fit the
   pre-period regardless of whether the fit is driven by signal or noise.

3. **Transparency**: each donor's contribution is explicit. You can report "the synthetic California
   is 40% Colorado + 35% Utah + 25% Montana" — a concrete, verifiable statement.

---

## Pre-Intervention Fit Quality

The synthetic control estimate is credible only if the pre-intervention fit is good.

**Assessing pre-intervention fit:**
<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>

```python
import numpy as np
import pandas as pd

def pre_period_fit(y_treated_pre, y_synthetic_pre):
    """
    Compute pre-intervention fit statistics.

    Parameters
    ----------
    y_treated_pre : array-like
        Observed outcome for treated unit in pre-intervention period
    y_synthetic_pre : array-like
        Synthetic control outcome in pre-intervention period

    Returns
    -------
    dict with MSPE, RMSPE, and R-squared
    """
    y_t = np.array(y_treated_pre)
    y_s = np.array(y_synthetic_pre)

    residuals = y_t - y_s
    mspe = np.mean(residuals ** 2)
    rmspe = np.sqrt(mspe)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_t - y_t.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot

    return {"MSPE": mspe, "RMSPE": rmspe, "R_squared": r_squared}
```

</div>

**Rule of thumb:** RMSPE in the pre-period should be below 10–20% of the treated unit's
pre-intervention standard deviation. If pre-period fit is poor, the counterfactual extrapolation
is unreliable.

---

## Choosing Donor Units (The Donor Pool)

Not all available units should be included in the donor pool. Exclude units that:

1. **Were also treated**: including a treated donor contaminates the counterfactual
2. **Experienced structural breaks**: a donor that had its own unrelated intervention violates
   the parallel-trends-like assumption
3. **Are too dissimilar**: donors whose trajectories are fundamentally different from the treated
   unit cannot contribute positively to the pre-period fit and may hurt it

**Practical donor pool construction:**
<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>

```python
# Example: selecting donor states for a California tobacco analysis
all_states = ["AL", "AK", "AZ", ..., "WY"]  # 50 states

# Remove states with similar tobacco policies before 1988
states_with_tobacco_programs = ["AK", "HI", "MD", "MI", "NJ", "NY", "WA"]
donor_pool = [s for s in all_states if s not in ["CA"] + states_with_tobacco_programs]

print(f"Donor pool size: {len(donor_pool)}")
```

</div>

---

## CausalPy Implementation

CausalPy's `SyntheticControl` class uses a Bayesian approach to weight estimation, placing
a Dirichlet prior on the weights and estimating the posterior distribution of the counterfactual.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>

```python
import causalpy as cp
import pandas as pd

# Data format: long form with columns [unit, time, outcome, predictor_columns]
sc_model = cp.SyntheticControl(
    data=panel_df,
    treatment_time=1988,          # First post-intervention period
    formula="cigsale ~ 1",        # Outcome and predictors
    group_variable_name="state",  # Column identifying units
    treated_group="California",   # Which unit is treated
    model=cp.pymc_models.WeightedSumFitter(
        sample_kwargs={"draws": 1000, "tune": 1000, "chains": 4, "random_seed": 42}
    ),
)

# Posterior of donor weights
sc_model.plot_weights()

# Counterfactual trajectory
sc_model.plot()
```

</div>

The Bayesian approach gives a full posterior distribution over the counterfactual trajectory,
enabling principled uncertainty quantification for the causal effect at each post-intervention
period.

---

## The Counterfactual Gap

The causal effect estimate at time $t$ is:

$$\hat{\alpha}_t = Y_{1t} - \sum_j w_j^* Y_{jt}$$

For a pointwise effect estimate with uncertainty bounds:
<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>

```python
import numpy as np
import arviz as az

# Extract posterior samples of donor weights: shape (n_samples, n_donors)
w_samples = sc_model.idata.posterior["weights"].values.reshape(-1, n_donors)

# Counterfactual trajectories: shape (n_samples, n_post_periods)
y_cf_samples = w_samples @ Y_donors_post.T  # (n_samples, n_post)

# Observed treated unit (scalar per period)
y_treated_post = Y_treated_post  # shape (n_post,)

# Gap: shape (n_samples, n_post)
gap_samples = y_treated_post[np.newaxis, :] - y_cf_samples

# Point estimate and HDI
gap_mean = gap_samples.mean(axis=0)
gap_hdi = az.hdi(gap_samples, hdi_prob=0.94)

print(f"Immediate post-intervention gap: {gap_mean[0]:.1f} "
      f"[{gap_hdi[0, 0]:.1f}, {gap_hdi[0, 1]:.1f}]")
```

</div>

---

## Assumptions

### Assumption 1: No Interference Between Units

The intervention in California does not affect cigarette consumption in Colorado. This is
the same as the SUTVA assumption in potential outcomes.

### Assumption 2: No Anticipation

Donors do not change behavior before the intervention in anticipation of the treated unit's
policy. This is often problematic when policies are debated publicly before implementation.

### Assumption 3: Good Pre-Intervention Fit

The convex hull of donor outcomes in the pre-period contains the treated unit's trajectory.
If the treated unit is an outlier (extreme values no donor combination can match), the
synthetic control estimator will have poor pre-period fit and the causal estimate is unreliable.

### Assumption 4: Sparse True Effect

The causal effect should be concentrated in a detectable portion of post-intervention periods.
If the effect is very small relative to post-intervention noise, the placebo test will not
distinguish the treated unit from its donors.

---

## Comparison: ITS vs Synthetic Control

| Dimension | ITS | Synthetic Control |
|-----------|-----|-------------------|
| Data requirements | Single unit, long series | Panel: multiple units |
| Comparison group | Counterfactual from pre-trend | Weighted donor average |
| Concurrent event threat | High | Lower (donors share the event) |
| Inference method | Bayesian posterior | Permutation (placebo) tests |
| Pre-intervention needed | 10–20+ periods | 10–20+ periods |
| Post-intervention needed | 5+ periods | 1+ period |
| Interpretability | Simple regression | Transparent weights |

**When to use synthetic control over ITS:**
- You have panel data (multiple units, one treated)
- You are concerned about concurrent events that affect only the treated unit
- The pre-intervention trend is not clearly extrapolable (structural breaks)
- You need a non-parametric approach that does not impose a functional form on the trend

---

## Limitations

1. **Requires panel data**: if you have only one unit, synthetic control is not possible
2. **Donor pool is consequential**: different donor pools can give different estimates
3. **Perfect pre-fit is impossible**: some residual pre-period discrepancy always exists
4. **Post-period extrapolation**: if the intervention is long, donors may diverge from the
   treated unit for reasons unrelated to the intervention
5. **Multiple treated units**: standard synthetic control is for a single treated unit;
   extensions exist (Doudchenko and Imbens, 2017) but are more complex

---

## Connections

<div class="callout-info">

<strong>How this connects to the rest of the course:</strong>

</div>

- **Builds on:** Module 00 (potential outcomes, SUTVA), Module 01 (ITS fundamentals)
- **Extension of:** Difference-in-differences (SC is a generalization with unit-specific weights)
- **Leads to:** Notebook 01 (synthetic control basics), Notebook 03 (CausalPy SC API)


## Practice Questions

### Question 1: Conceptual Check
**Question:** In your own words, explain the core concept of Synthetic Control: Building a Counterfactual from Donor Units and why it matters for practical applications. What problem does it solve that simpler approaches cannot?

### Question 2: Application
**Question:** Describe a real-world scenario where you would apply the techniques from this guide. What assumptions would you need to verify before proceeding?

## Further Reading

- Abadie, A., Diamond, A., and Hainmueller, J. (2010). "Synthetic Control Methods for Comparative
  Case Studies: Estimating the Effect of California's Tobacco Control Program." *Journal of the
  American Statistical Association*, 105(490), 493–505.
- Abadie, A. (2021). "Using Synthetic Controls: Feasibility, Data Requirements, and Methodological
  Aspects." *Journal of Economic Literature*, 59(2), 391–425.
- Doudchenko, N., and Imbens, G. W. (2017). "Balancing, Regression, Difference-In-Differences
  and Synthetic Control Methods: A Synthesis." NBER Working Paper 22791.


## Resources

<a class="link-card" href="../notebooks/01_synthetic_control_basics.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises for this topic.</div>
</a>
