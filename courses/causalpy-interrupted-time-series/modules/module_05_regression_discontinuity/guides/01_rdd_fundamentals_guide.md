# Regression Discontinuity Designs: Fundamentals

> **Reading time:** ~9 min | **Module:** 5 — Regression Discontinuity | **Prerequisites:** Module 0 — Causal Foundations

## Learning Objectives

By the end of this guide, you will be able to:
1. Explain the identification strategy in a sharp RDD
2. Distinguish sharp from fuzzy RDD and know when each applies
3. Interpret the RDD estimate as a Local Average Treatment Effect (LATE)
4. State the continuity assumption and assess its plausibility
5. Identify and address common threats to RDD validity

---

## 1. The Core Idea: Exploiting a Threshold

Many policy rules assign treatment based on whether a running variable crosses a threshold:

- Students scoring above 50 on an entrance exam get a scholarship
- Households with income below $30,000 qualify for a housing subsidy
- Countries with GDP per capita below $1,000 receive development aid
- Companies with more than 50 employees must comply with a regulation

In each case, there is a **running variable** (test score, income, GDP, employees) and a **cutoff** that determines treatment assignment.

The regression discontinuity design (RDD) exploits this structure: units just below and just above the cutoff are likely very similar, yet one group is treated and the other is not. The **discontinuity in treatment** at the cutoff, combined with **continuity of everything else**, identifies a causal effect.

---

## 2. Sharp RDD: The Setup

In a **sharp RDD**:

$$D_i = \mathbf{1}[X_i \geq c]$$

where:
- $X_i$ = running variable (also called "forcing variable" or "score")
- $c$ = the cutoff
- $D_i$ = treatment indicator (1 if treated, 0 if not)

Treatment jumps **discontinuously** from 0 to 1 at the cutoff. Below the cutoff, no one is treated. Above it, everyone is treated.

### The Estimand

The RDD estimator identifies the treatment effect **at the cutoff**:

$$\tau_{RDD} = \lim_{x \downarrow c} E[Y_i \mid X_i = x] - \lim_{x \uparrow c} E[Y_i \mid X_i = x]$$

This is the difference in the expected outcome just above the cutoff versus just below it. It's a **Local Average Treatment Effect (LATE)** — local to the cutoff.

### Visual Intuition

```

Outcome
  |                                    /
  |                                   / ← treated units above cutoff
  |                                  /
  |          ← untreated             |← Jump = treatment effect
  |         /                        |
  |        /                        /
  |       /                        /
  |______________________________/___→ Running Variable X
                              ↑
                           Cutoff c
```

The size of the jump at the cutoff estimates the treatment effect.

---

## 3. The Continuity Assumption

The RDD identification assumption is **continuity**:

**Assumption (Continuity):** The conditional expectation of the potential outcomes, $E[Y_i(0) \mid X_i = x]$ and $E[Y_i(1) \mid X_i = x]$, is continuous in $x$ at the cutoff $c$.

This says that, absent treatment, the regression function would not jump at the cutoff. The only reason for a discontinuity in the observed outcome is the discontinuity in treatment.

### Implications

If continuity holds:
- Units just below the cutoff are valid counterfactuals for units just above
- Any jump at the cutoff is attributable to treatment, not to the running variable itself

If continuity is violated:
- Some other variable also jumps at the cutoff (compound discontinuity)
- Units self-select to be just above or below the cutoff (manipulation)
- The cutoff coincides with another policy threshold

---

## 4. Sharp vs Fuzzy RDD

| Feature | Sharp RDD | Fuzzy RDD |
|---------|-----------|-----------|
| Treatment rule | $D = \mathbf{1}[X \geq c]$ (exact) | Treatment probability discontinuously rises at $c$ |
| Treatment compliance | 100% — everyone follows the rule | Partial — some above cutoff not treated, some below treated |
| Identification | Continuity | Continuity + instrument |
| Estimand | $E[Y(1) - Y(0) \mid X = c]$ | LATE for compliers at cutoff |
| Analogy | Ideal experiment at cutoff | Instrumental variables |

### Fuzzy RDD as IV

In a fuzzy RDD, the treatment probability jumps at the cutoff but does not go from 0 to 1. The cutoff serves as an instrument:

$$\tau_{fuzzy} = \frac{\text{Jump in } E[Y \mid X] \text{ at } c}{\text{Jump in } E[D \mid X] \text{ at } c} = \frac{\text{Reduced form}}{\text{First stage}}$$

This is exactly the IV (Wald) estimator. The LATE interpretation: the effect for units who comply with the rule — who would be treated above the cutoff and untreated below.

---

## 5. Estimation

### Parametric (Global Polynomial) Approach

Fit a polynomial in the running variable on each side of the cutoff:

$$Y_i = \alpha + \tau D_i + f(X_i - c) + D_i \cdot g(X_i - c) + \epsilon_i$$

where $f(\cdot)$ and $g(\cdot)$ are polynomials and $D_i = \mathbf{1}[X_i \geq c]$.

The coefficient $\tau$ estimates the jump at the cutoff.

**Problems with global polynomial RDD:**
- High-order polynomials can produce erratic estimates near boundaries
- The estimate depends heavily on polynomial order choice
- Units far from the cutoff influence the estimate but are irrelevant for identification

### Local Linear Regression (Preferred)

Fit separate linear regressions on each side of the cutoff, using only observations within a bandwidth $h$:

$$Y_i = \alpha + \tau D_i + \beta (X_i - c) + \gamma D_i (X_i - c) + \epsilon_i \quad \text{for } |X_i - c| \leq h$$

This is the **local linear regression** estimator. It:
- Uses only nearby observations (within bandwidth)
- Allows different slopes on each side
- Reduces boundary bias compared to global polynomials


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

def sharp_rdd_local_linear(df, outcome, running_var, cutoff, bandwidth):
    """Estimate sharp RDD using local linear regression."""
    # Restrict to bandwidth
    mask = np.abs(df[running_var] - cutoff) <= bandwidth
    local_df = df[mask].copy()

    # Normalise running variable around cutoff
    local_df['x_centered'] = local_df[running_var] - cutoff
    local_df['treated'] = (local_df[running_var] >= cutoff).astype(int)

    # Local linear regression
    formula = f'{outcome} ~ treated + x_centered + treated:x_centered'
    model = smf.ols(formula, data=local_df).fit()

    return model.params['treated'], model.bse['treated'], model
```

</div>
</div>

---

## 6. Bandwidth Selection

The bandwidth $h$ is the most important tuning choice in RDD:

| Bandwidth | Tradeoff |
|-----------|---------|
| Too narrow | Few observations → high variance; insufficient power |
| Too wide | Many observations → low variance, but units far from cutoff may have different counterfactuals |
| Optimal | Minimises mean squared error |

### Imbens-Kalyanaraman (IK) Optimal Bandwidth

The IK bandwidth minimises the asymptotic mean squared error of the local linear estimator. It accounts for:
- The curvature of the regression function near the cutoff
- The density of the running variable at the cutoff
- The outcome variance


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python

# Using rdrobust package
from rdrobust import rdrobust, rdbwselect

# Estimate with data-driven optimal bandwidth
result = rdrobust(y=df['outcome'], x=df['running_var'], c=0)
print(result.summary())

# Or separately compute bandwidth
bw = rdbwselect(y=df['outcome'], x=df['running_var'], c=0)
print(f"Optimal bandwidth: {bw.bws['h']:.3f}")
```

</div>
</div>

---

## 7. Threats to Validity

### Manipulation of the Running Variable

If units can precisely control their running variable value and prefer to be treated (or untreated), they will bunch just above (or below) the cutoff. This violates continuity — the distribution of potential outcomes is not continuous at the cutoff.

**Detection:** Histogram density test (McCrary test)


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
from rdrobust import rddensity

density_test = rddensity(df['running_var'], c=0)
print(density_test.summary())

# p > 0.05: no significant discontinuity in density (good)
```

</div>
</div>

A spike in the running variable's density just above the cutoff is a red flag.

### Compound Discontinuity

If another variable also jumps discontinuously at the cutoff, you cannot separate the treatment effect from the other variable's effect.

**Detection:** Run the RDD on **baseline covariates** — they should show no jump at the cutoff.


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
for covariate in ['age', 'income', 'prior_score']:
    rdd_covariate = rdrobust(y=df[covariate], x=df['running_var'], c=0)
    print(f"{covariate}: τ = {rdd_covariate.coef[0]:.3f}, p = {rdd_covariate.pv[0]:.3f}")

# All should have p >> 0.05 (no jump in covariates)
```

</div>
</div>

### Sorting Near the Cutoff

Even without precise manipulation, units near the cutoff may differ systematically from those further away if the running variable is correlated with unobservables. This is less of a threat than full manipulation but still warrants examination.

---

## 8. Checking RDD Validity

A complete RDD validity check includes:

1. **Density test (McCrary):** No discontinuity in the running variable's density at the cutoff
2. **Covariate balance:** No discontinuity in baseline covariates at the cutoff
3. **Placebo cutoffs:** RDD at artificial cutoffs away from the true cutoff should give null estimates
4. **Donut RDD:** Exclude observations very close to the cutoff and check if estimates are stable (tests for local manipulation)
5. **Bandwidth sensitivity:** Estimates should be qualitatively stable across a range of bandwidths

---

## 9. The Local Nature of RDD

A crucial limitation of RDD is **external validity**. The RDD estimate is **local to the cutoff** — it identifies the treatment effect for units whose running variable is near $c$. This may not generalise to:

- Units far from the cutoff
- Units in different contexts
- The average treatment effect in the population

This local nature is both a strength (high internal validity) and a weakness (limited generalisability). When reporting RDD results, always specify: "This is the effect of [treatment] for units near the cutoff of [running variable] = [c]."

---

## 10. CausalPy RDD: Quick Look


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
import causalpy as cp

result = cp.RegressionDiscontinuity(
    data=df,
    formula='outcome ~ 1 + x_centered',
    running_variable_name='x_centered',
    model=cp.pymc_models.LinearRegression(),
    bandwidth=0.5,
    epsilon=0.01  # small value to define "near the cutoff"
)

result.plot()
print(result.summary())
```

</div>
</div>

The `epsilon` parameter defines the neighbourhood of the cutoff used for the treatment effect estimate. The plot shows both sides of the regression and the estimated jump.

---

## 11. Summary

| Concept | Key Point |
|---------|-----------|
| Running variable | Determines treatment via threshold rule |
| Sharp RDD | Treatment is exactly determined by crossing cutoff |
| Fuzzy RDD | Treatment probability discontinuously rises at cutoff |
| Continuity assumption | No jump in potential outcomes at cutoff (absent treatment) |
| LATE at cutoff | RDD identifies the effect only for units near the cutoff |
| Local linear | Preferred estimator — flexible, low boundary bias |
| Bandwidth | Key tuning parameter: narrow = precise but variable; wide = smoother but potentially biased |
| Manipulation test | Density should be continuous at the cutoff |

---


## Practice Questions

### Question 1: Conceptual Check
**Question:** In your own words, explain the core concept of Regression Discontinuity Designs: Fundamentals and why it matters for practical applications. What problem does it solve that simpler approaches cannot?

### Question 2: Application
**Question:** Describe a real-world scenario where you would apply the techniques from this guide. What assumptions would you need to verify before proceeding?

## Further Reading

- Imbens & Lemieux (2008), "Regression Discontinuity Designs: A Guide to Practice"
- Lee & Lemieux (2010), "Regression Discontinuity Designs in Economics"
- Cattaneo, Idrobo & Titiunik (2019-2020), *A Practical Introduction to Regression Discontinuity Designs* (Cambridge Elements)
- Gelman & Imbens (2019), "Why High-Order Polynomials Should Not Be Used in Regression Discontinuity Designs"
<div class="callout-key">

<strong>Key Concept:</strong> - Imbens & Lemieux (2008), "Regression Discontinuity Designs: A Guide to Practice"
- Lee & Lemieux (2010), "Regression Discontinuity Designs in Economics"
- Cattaneo, Idrobo & Titiunik (2019-2020), *A Practical Introduction to Regression Discontinuity Designs* (Cambridge Elements)
- Gelman & Imbe...

</div>


---

**Next:** [02 — Bandwidth Selection and Sensitivity Analysis](02_bandwidth_selection_guide.md)
