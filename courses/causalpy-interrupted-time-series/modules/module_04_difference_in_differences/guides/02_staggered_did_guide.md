# Staggered Difference-in-Differences and Event Studies

> **Reading time:** ~8 min | **Module:** 4 — Difference In Differences | **Prerequisites:** Module 1 — ITS Fundamentals

## Learning Objectives

By the end of this guide, you will be able to:
1. Explain why TWFE DiD produces biased estimates under staggered adoption
2. Implement the Callaway–Sant'Anna and Sun–Abraham estimators
3. Construct event study plots with proper normalisation
4. Run pre-trend tests and interpret their results
5. Choose the appropriate staggered DiD estimator for a given setting

---

## 1. The Problem with Staggered Adoption

In many real-world settings, a policy or treatment does not arrive for all units simultaneously. Different states, firms, or individuals adopt treatment at different times — a pattern called **staggered adoption** or **staggered rollout**.

### Why TWFE Breaks Down

With staggered adoption, the standard TWFE regression:

$$Y_{it} = \alpha_i + \lambda_t + \tau \cdot D_{it} + \epsilon_{it}$$

uses **already-treated units as controls for later-treated units**. This is problematic because:

1. Early-treated units have their own treatment effect "baked in" — they are not clean controls
2. If treatment effects grow over time (dynamic effects), using early-treated units as controls can produce **negative weights** in the TWFE estimand
3. The TWFE coefficient is a weighted average of group-time ATTs, but some weights can be **negative**, making the estimate misleading or even opposite in sign to all individual effects

### Goodman-Bacon Decomposition

Goodman-Bacon (2021) shows that the TWFE estimator decomposes into a weighted average of all possible 2×2 DiD comparisons:

$$\hat{\tau}_{TWFE} = \sum_{k \in \mathcal{K}} \sum_{\ell \in \mathcal{L}} \hat{w}_{k\ell} \hat{\tau}_{k\ell}^{2\times2}$$

where the weights $\hat{w}_{k\ell}$ depend on sample shares and treatment timing variance. Crucially, some weights can be negative.

---

## 2. Cohort-Based Thinking

The key insight from modern staggered DiD literature is to think in **cohorts**: groups of units that are first treated at the same time.

### Defining Cohorts

Let $G_i$ = the period when unit $i$ is first treated (or $\infty$ if never treated).

A **cohort** is the set of units sharing the same $G_i$:
- Cohort 2010: units first treated in period 2010
- Cohort 2014: units first treated in period 2014
- Never-treated: units with $G_i = \infty$

### Group-Time Average Treatment Effects

Callaway & Sant'Anna (2021) define the **group-time ATT**:

$$ATT(g, t) = E[Y_t(g) - Y_t(0) \mid G = g]$$

This is the average treatment effect for cohort $g$ at time period $t$.

These are the fundamental building blocks. Any aggregate estimand (overall ATT, event study, etc.) is a weighted average of group-time ATTs.

---

## 3. Modern Staggered DiD Estimators

### Callaway & Sant'Anna (2021)

**Key idea:** Compare each cohort only to "clean" controls — never-treated units or not-yet-treated units.

**Steps:**
1. Estimate $ATT(g,t)$ for every cohort $g$ and period $t$
2. Aggregate into a single treatment effect or event study plot
3. Standard errors via bootstrap

**Control group options:**
- *Never-treated:* only units that never receive treatment. Cleaner, but may be a small or unrepresentative group.
- *Not-yet-treated:* units that haven't been treated *yet* at time $t$. Larger sample, but assumes no anticipation.


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Using the csdid package (Python port)
from csdid import ATTgt

att_gt = ATTgt(
    data=df,
    yname="outcome",
    gname="cohort",      # first treatment period
    idname="unit_id",
    tname="period",
    control_group="never_treated"
)
att_gt.fit()
att_gt.aggregate("event")  # event study aggregation
att_gt.plot()
```

</div>
</div>

### Sun & Abraham (2021)

**Key idea:** Interact unit-level cohort indicators with period indicators in a regression.

**The saturated regression:**

$$Y_{it} = \alpha_i + \lambda_t + \sum_{g} \sum_{k \neq -1} \delta_{g,k} \cdot \mathbf{1}[G_i = g] \cdot \mathbf{1}[t - g = k] + \epsilon_{it}$$

The coefficients $\delta_{g,k}$ are cohort-specific event study estimates. Aggregate them to get overall event study plots or a single treatment effect.

This approach works within standard regression software by:
1. Creating cohort × relative-time interaction dummies
2. Running OLS
3. Aggregating coefficients with appropriate weights

---

## 4. Event Study Plots

An event study plot displays treatment effects over time relative to treatment adoption.

### Construction

1. Define event time: $k = t - G_i$ (periods since treatment for cohort $G_i$)
2. Estimate $\beta_k$ for each $k$ in the event study regression
3. Normalise: set $\beta_{-1} = 0$ (or average of pre-period as baseline)
4. Plot $\beta_k$ with confidence/credible intervals

### Interpreting the Plot

```

β_k
  |
  |                   ●—●—●   (treatment effect stabilises)
  |               ●
0 |●—●—●—●—●—●—●           (pre-period: flat = parallel trends support)
  |
  |___-5__-4__-3__-2__-1___0___1___2___3___→ k
                         ↑
                     Treatment
```

**Pre-period pattern (k < 0):**
- Flat, near-zero: supports parallel trends
- Sloping trend: parallel trends violated — treatment is correlated with pre-existing trends

**Post-period pattern (k ≥ 0):**
- Jump at k=0: immediate effect
- Gradual rise: treatment builds over time
- Fade-out: effect dissipates
- Negative then positive: possible anticipation then effect

---

## 5. Pre-Trend Testing

### Formal Tests

#### Joint F-test on Pre-Period Coefficients

Test $H_0: \beta_{-5} = \beta_{-4} = \ldots = \beta_{-2} = 0$ (excluding $\beta_{-1}$ which is normalised):


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
from scipy.stats import chi2

# Get pre-period coefficients and covariance matrix
pre_coefs = event_study_coefs[pre_indices]
pre_cov = event_study_cov[np.ix_(pre_indices, pre_indices)]

# Chi-squared statistic
W = pre_coefs @ np.linalg.inv(pre_cov) @ pre_coefs
p_value = 1 - chi2.cdf(W, df=len(pre_indices))
print(f"Pre-trend test: W = {W:.2f}, p = {p_value:.3f}")
```

</div>
</div>

#### Roth (2022) Sensitivity Analysis

A failure to reject the pre-trend test does not mean parallel trends holds — it may be low-powered. Roth (2022) proposes sensitivity analyses that bound the treatment effect under specific violations of parallel trends.

### What Pre-Trend Tests Can and Cannot Do

| Can do | Cannot do |
|--------|-----------|
| Detect clear pre-existing trends | Prove parallel trends holds in the post-period |
| Raise red flags about confounding | Rule out subtle violations |
| Build credibility with reviewers | Guarantee causal identification |
| Guide choice of control group | Test for post-treatment confounding |

---

## 6. Aggregating Group-Time ATTs

Once you have $ATT(g,t)$ estimates, you can aggregate in several ways:

### Overall ATT

Simple average weighted by cohort size and number of post-treatment periods:

$$\overline{ATT} = \sum_{g,t: t \geq g} w_{g,t} \cdot ATT(g,t)$$

### Event Study (Dynamic Effects)

Average $ATT(g,t)$ across cohorts for each event-time $k = t - g$:

$$\theta^{ES}(k) = \sum_{g: g+k \geq 0} w_g \cdot ATT(g, g+k)$$

### Group-Specific Effects

For each cohort, average over post-treatment periods:

$$\theta^{group}(g) = \frac{1}{T - g} \sum_{t \geq g} ATT(g, t)$$

### Calendar-Time Effects

For each calendar period $t$, average over cohorts treated by that time:

$$\theta^{cal}(t) = \sum_{g \leq t} w_g \cdot ATT(g, t)$$

---

## 7. Worked Example: Staggered Adoption in Python


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate staggered adoption data
np.random.seed(42)
n_units = 150
n_periods = 16  # periods 1-16, treatment starts period 5

units = range(n_units)
periods = range(1, n_periods + 1)

# Assign cohorts: 1/3 treated at t=5, 1/3 at t=9, 1/3 never treated
cohort = np.repeat([5, 9, np.inf], n_units // 3)

rows = []
for u, g in enumerate(cohort):
    unit_fe = np.random.normal(0, 1)
    for t in periods:
        treated = 1 if (t >= g and g != np.inf) else 0
        event_time = (t - g) if g != np.inf else np.nan
        # True effect: 2.0 + 0.3 * k (growing effect)
        true_effect = (2.0 + 0.3 * max(0, t - g)) if treated else 0
        y = 1.0 + 0.4 * t + unit_fe + true_effect + np.random.normal(0, 0.5)
        rows.append({
            "unit": u, "period": t, "cohort": g,
            "treated": treated, "outcome": y, "event_time": event_time
        })

df = pd.DataFrame(rows)

# Event study: aggregate ATT(g,t) by event time

# Use never-treated as control
never_treated = df[df["cohort"] == np.inf].copy()
never_treated_means = never_treated.groupby("period")["outcome"].mean()

event_coefs = {}
for k in range(-4, 8):  # event times -4 to 7
    subset_rows = []
    for g in [5, 9]:
        t = g + k
        if t < 1 or t > n_periods:
            continue
        treated_g = df[(df["cohort"] == g) & (df["period"] == t)]["outcome"]
        treated_g_pre = df[(df["cohort"] == g) & (df["period"] == g - 1)]["outcome"]
        control_t = df[(df["cohort"] == np.inf) & (df["period"] == t)]["outcome"]
        control_pre = df[(df["cohort"] == np.inf) & (df["period"] == g - 1)]["outcome"]

        if len(treated_g) > 0 and len(control_t) > 0:
            att_g_t = (treated_g.mean() - treated_g_pre.mean()) - (control_t.mean() - control_pre.mean())
            subset_rows.append(att_g_t)

    if subset_rows:
        event_coefs[k] = np.mean(subset_rows)

# Plot event study
ks = sorted(event_coefs.keys())
betas = [event_coefs[k] for k in ks]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(ks, betas, marker="o", color="steelblue", linewidth=2)
ax.axvline(x=0, color="red", linestyle="--", alpha=0.7, label="Treatment onset")
ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
ax.set_xlabel("Periods relative to treatment (k)")
ax.set_ylabel("Estimated ATT")
ax.set_title("Event Study: Staggered DiD")
ax.legend()
plt.tight_layout()
plt.show()
```


---

## 8. Choosing Between Estimators

| Situation | Recommended Estimator |
|-----------|----------------------|
| Single treatment date, two periods | Classic TWFE DiD |
| Staggered adoption, homogeneous effects | TWFE (unbiased under homogeneity) |
| Staggered adoption, heterogeneous effects | Callaway–Sant'Anna or Sun–Abraham |
| Small number of treated units | Synthetic DiD (Arkhangelsky et al.) |
| Continuous treatment | See Callaway, Goodman-Bacon & Sant'Anna |
| Treatment turns off | See de Chaisemartin & D'Haultfoeuille |

---

## 9. Summary

| Concept | Key Point |
|---------|-----------|
| Staggered adoption | Units treated at different times — common in practice |
| TWFE bias | Uses early-treated as controls, can have negative weights |
| Cohort ATT | $ATT(g,t)$: fundamental building block of staggered DiD |
| Callaway-Sant'Anna | Compares each cohort to clean (never/not-yet-treated) controls |
| Sun-Abraham | Saturated regression approach to staggered DiD |
| Event study | Plots $ATT$ by event time; pre-period tests parallel trends |
| Pre-trend test | Cannot prove PT but can detect violations |

---


## Practice Questions

### Question 1: Conceptual Check
**Question:** In your own words, explain the core concept of Staggered Difference-in-Differences and Event Studies and why it matters for practical applications. What problem does it solve that simpler approaches cannot?

### Question 2: Application
**Question:** Describe a real-world scenario where you would apply the techniques from this guide. What assumptions would you need to verify before proceeding?

## Further Reading

- Callaway & Sant'Anna (2021), "Difference-in-Differences with Multiple Time Periods"
- Sun & Abraham (2021), "Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects"
- Goodman-Bacon (2021), "Difference-in-Differences with Variation in Treatment Timing"
- Roth et al. (2023), "What's Trending in Difference-in-Differences? A Synthesis of the Recent Econometrics Literature"
- de Chaisemartin & D'Haultfoeuille (2020), "Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects"

---

**Previous:** [01 — DiD Fundamentals](01_did_fundamentals_guide.md)
**Next:** [03 — CausalPy DiD API](03_causalpy_did_api_guide.md)

<div class="callout-key">

<strong>Key Concept:</strong> **Previous:** [01 — DiD Fundamentals](01_did_fundamentals_guide.md)
**Next:** [03 — CausalPy DiD API](03_causalpy_did_api_guide.md)



## Resources

<a class="link-card" href="../notebooks/01_did_labour_economics.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises for this topic.</div>
</a>
