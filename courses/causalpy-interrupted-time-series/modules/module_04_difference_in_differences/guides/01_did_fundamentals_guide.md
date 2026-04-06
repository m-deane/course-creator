# Difference-in-Differences: Fundamentals

> **Reading time:** ~8 min | **Module:** 4 — Difference In Differences | **Prerequisites:** Module 1 — ITS Fundamentals

## Learning Objectives

By the end of this guide, you will be able to:
1. Explain the parallel trends assumption and why it is the identifying condition for DiD
2. Derive the DiD estimator from the two-period, two-group setup
3. Interpret DiD estimates as average treatment effects on the treated (ATT)
4. Assess the plausibility of parallel trends using pre-treatment data
5. Implement the canonical two-period DiD in CausalPy

---

<div class="callout-key">

<strong>Key Concept:</strong> DiD compares how treated and control groups change over time, rather than comparing their levels. By "differencing out" the time trend, DiD isolates the treatment effect even when groups have different baseline levels.

</div>

## 1. Motivation: The Counterfactual Problem Revisited

Every causal question requires a counterfactual: what would have happened to the treated units had they not been treated? In an ITS design, the pre-treatment trend of the *same* unit provides that counterfactual. In a synthetic control, a weighted combination of donor units does. Difference-in-differences (DiD) takes a different approach: it uses a **comparison group** of untreated units to build the counterfactual.

The core idea is elegant:

> The change over time in the control group estimates the counterfactual change that would have occurred in the treatment group absent treatment.

Subtracting the control group's change from the treatment group's change isolates the treatment effect.

---

## 2. The Two-Period, Two-Group Setup

Consider the simplest DiD setting:

- Two time periods: **pre** (t = 0) and **post** (t = 1)
- Two groups: **treated** (D = 1) and **control** (D = 0)
- Treatment occurs at t = 1 for the treated group only

Define:

| Group | Pre-period mean | Post-period mean | Difference over time |
|-------|----------------|-----------------|---------------------|
| Treated | $\bar{Y}_{1,0}$ | $\bar{Y}_{1,1}$ | $\Delta_1 = \bar{Y}_{1,1} - \bar{Y}_{1,0}$ |
| Control | $\bar{Y}_{0,0}$ | $\bar{Y}_{0,1}$ | $\Delta_0 = \bar{Y}_{0,1} - \bar{Y}_{0,0}$ |

The **DiD estimator** is:

$$\hat{\tau}_{DiD} = \Delta_1 - \Delta_0 = (\bar{Y}_{1,1} - \bar{Y}_{1,0}) - (\bar{Y}_{0,1} - \bar{Y}_{0,0})$$

This is the "difference of differences" — hence the name.

### Equivalent Regression Formulation

The same estimator arises from the regression:

$$Y_{it} = \alpha + \beta \cdot \text{Post}_t + \gamma \cdot \text{Treated}_i + \tau \cdot (\text{Post}_t \times \text{Treated}_i) + \epsilon_{it}$$

where:
- $\alpha$ = control group pre-period level
- $\beta$ = time trend common to both groups
- $\gamma$ = pre-period level difference between groups
- $\tau$ = **the DiD treatment effect**

The interaction term $\text{Post}_t \times \text{Treated}_i$ is 1 only for treated units in the post-period.

---

<div class="callout-danger">

<strong>Danger:</strong> The DiD estimator is biased if parallel trends is violated. No amount of additional data or fancier estimation will fix this -- it is a design assumption, not a statistical assumption.

</div>

## 3. The Parallel Trends Assumption

DiD rests on one core identifying assumption:

**Parallel Trends (PT):** In the absence of treatment, the average outcome for the treated group would have followed the same trend as the average outcome for the control group.

Formally:

$$E[Y_{it}(0) - Y_{it-1}(0) \mid D_i = 1] = E[Y_{it}(0) - Y_{it-1}(0) \mid D_i = 0]$$

This says the *counterfactual* change for the treated equals the *observed* change for the control.

### What Parallel Trends Is and Is Not

**It does NOT require:**
- The levels of outcomes to be equal across groups
- The groups to be identical in any other way
- Outcomes to literally be parallel in pre-periods (though this is suggestive)

**It DOES require:**
- Absent treatment, the trajectory of outcomes would have been the same
- No time-varying confounders that affect treated and control groups differently

### Visualising Parallel Trends

```

Outcome
  |                            * (treated, observed post)
  |               /
  |              * (treated pre)
  |             /               - - - (treated counterfactual = parallel to control)
  |            /               /
  |           * (control pre) * (control post)
  |          /               /
  |         /               /
  |________/_______________/___________> Time
           Pre             Post
                    ^
                    Treatment
```

The DiD estimate = (observed treated post) - (counterfactual treated post)
               = (observed treated post) - [(treated pre) + (control change)]

---

## 4. Identification: When Does DiD Give Causal Estimates?

### The Two-Way Fixed Effects (TWFE) Model

The regression formulation generalises naturally to panel data with many units and time periods:

$$Y_{it} = \alpha_i + \lambda_t + \tau \cdot D_{it} + \epsilon_{it}$$

where:
- $\alpha_i$ = unit fixed effects (absorb time-invariant differences between units)
- $\lambda_t$ = time fixed effects (absorb common shocks affecting all units)
- $D_{it}$ = treatment indicator (1 if unit i is treated at time t)
- $\tau$ = the DiD estimate

This is the **two-way fixed effects (TWFE)** estimator. It extends DiD to many units and periods under the assumption that parallel trends holds conditional on unit and time fixed effects.

### Threats to Identification

| Threat | Description | Solution |
|--------|-------------|----------|
| Violation of parallel trends | Treated and control units were on different trajectories before treatment | Find better control groups; use conditional PT |
| Anticipation effects | Units change behaviour *before* treatment expecting it | Extend pre-period; test for pre-trends |
| Compositional changes | The composition of groups changes over time | Use balanced panel or account for entry/exit |
| Spillovers (SUTVA violation) | Treatment affects control units indirectly | Define treatment carefully; use geographic distance |
| Heterogeneous treatment effects with staggered adoption | Different cohorts have different effects | Use modern staggered DiD estimators (Module 04.2) |

---

## 5. Assessing Parallel Trends: Pre-Trend Tests

You cannot directly test parallel trends since the counterfactual is unobserved. However, you can examine **pre-treatment trends**:

If the two groups had parallel trends before treatment, this is consistent with (but does not prove) the parallel trends assumption holding for the treatment period.

### The Event Study Regression

$$Y_{it} = \alpha_i + \lambda_t + \sum_{k \neq -1} \beta_k \cdot \mathbf{1}[t - T_i^* = k] \cdot D_i + \epsilon_{it}$$

where $T_i^*$ is the treatment date for unit i. The coefficients $\beta_k$ for k < 0 test pre-treatment parallel trends. If pre-treatment $\beta_k \approx 0$, the parallel trends assumption is more credible.

By convention, $\beta_{-1} = 0$ (the period immediately before treatment is the baseline).

### Practical Guidance for Pre-Trend Assessment

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

# Load panel data
df = pd.read_csv("panel_data.csv")

# Compute group means by period
group_means = df.groupby(["period", "treated"])["outcome"].mean().reset_index()

# Plot pre-treatment trends
pre_treatment = group_means[group_means["period"] < 0]  # periods before treatment

fig, ax = plt.subplots(figsize=(10, 5))
for group, label in [(0, "Control"), (1, "Treated")]:
    subset = pre_treatment[pre_treatment["treated"] == group]
    ax.plot(subset["period"], subset["outcome"], marker="o", label=label)

ax.axvline(x=0, color="red", linestyle="--", label="Treatment")
ax.set_xlabel("Periods relative to treatment")
ax.set_ylabel("Mean outcome")
ax.set_title("Pre-Treatment Parallel Trends Check")
ax.legend()
plt.show()
```

</div>

---

## 6. Interpreting the DiD Estimate

The DiD estimator identifies the **Average Treatment Effect on the Treated (ATT)**:

$$\tau_{ATT} = E[Y_{it}(1) - Y_{it}(0) \mid D_i = 1, t = \text{Post}]$$

This answers: "What was the average treatment effect for units that were actually treated, in the period they were treated?"

Note: DiD does **not** identify the Average Treatment Effect (ATE) unless we additionally assume that the treatment effect is the same for treated and untreated units (homogeneous treatment effects).

### Practical Interpretation

If the DiD estimate is $\hat{\tau} = 0.15$ for an outcome measured in log wages:

- Treated units experienced a 15 percentage point increase in wages attributable to treatment
- This is the average effect across all treated units (ATT)
- It does not generalise to untreated units without additional assumptions

---

## 7. Assumptions Checklist

Before running a DiD analysis, verify:

- [ ] **Treatment timing is known** — you know exactly when treatment occurred
- [ ] **Treatment is binary and absorbing** — units do not move in and out of treatment
- [ ] **No spillovers** — control units are not affected by the treatment
- [ ] **No anticipation** — units do not change behaviour before treatment
- [ ] **Parallel trends is plausible** — pre-treatment trends look parallel
- [ ] **Stable composition** — the same units are observed throughout
- [ ] **No simultaneous policy changes** — no other interventions hit treated but not control units at the same time

---

## 8. A Minimal CausalPy DiD Example

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>

```python
import causalpy as cp
import pandas as pd

# Simulate two-period panel data
np.random.seed(42)
n_units = 200
periods = [0, 1]

data = pd.DataFrame({
    "unit": np.repeat(range(n_units), 2),
    "period": np.tile(periods, n_units),
    "treated": np.repeat(np.random.binomial(1, 0.5, n_units), 2),
})

# True treatment effect = 2.0
data["outcome"] = (
    1.0                                    # intercept
    + 0.5 * data["period"]                 # time trend
    + 1.5 * data["treated"]               # group level difference
    + 2.0 * (data["period"] * data["treated"])  # true treatment effect
    + np.random.normal(0, 0.5, len(data))  # noise
)

# Run DiD using CausalPy
result = cp.DifferenceInDifferences(
    data=data,
    formula="outcome ~ 1 + period + treated + period:treated",
    time_variable_name="period",
    group_variable_name="treated",
)

print(result.summary())
```

</div>

The key coefficient is the interaction `period:treated`, which estimates the DiD treatment effect.

---

## 9. Summary

| Concept | Key Point |
|---------|-----------|
| DiD estimator | Difference in before-after changes between treated and control groups |
| Parallel trends | The central identifying assumption — absent treatment, trends would be equal |
| ATT | DiD estimates the average treatment effect on the treated |
| TWFE | Extends DiD to multiple units and periods using fixed effects |
| Pre-trend test | Examine pre-treatment trends to assess parallel trends plausibility |
| Threats | Anticipation, spillovers, compositional change, simultaneous policies |

---


## Practice Questions

### Question 1: Conceptual Check
**Question:** In your own words, explain the core concept of Difference-in-Differences: Fundamentals and why it matters for practical applications. What problem does it solve that simpler approaches cannot?

### Question 2: Application
**Question:** Describe a real-world scenario where you would apply the techniques from this guide. What assumptions would you need to verify before proceeding?

## Further Reading

- Angrist & Pischke (2009), *Mostly Harmless Econometrics*, Chapter 5
- Card & Krueger (1994), "Minimum Wages and Employment" — the canonical DiD study
- Callaway & Sant'Anna (2021), "Difference-in-Differences with Multiple Time Periods"
- Roth et al. (2023), "What's Trending in Difference-in-Differences?"

---

**Next:** [02 — Staggered DiD and Event Studies](02_staggered_did_guide.md)
