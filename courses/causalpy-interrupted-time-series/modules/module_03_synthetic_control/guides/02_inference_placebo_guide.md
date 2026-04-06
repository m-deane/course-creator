# Inference for Synthetic Control: Placebo Tests and Permutation Methods

> **Reading time:** ~9 min | **Module:** 3 — Synthetic Control | **Prerequisites:** Module 1 — ITS Fundamentals

## In Brief

Classical frequentist inference assumes a large number of independent observations. Synthetic
control has one treated unit and a small number of post-intervention periods — classical standard
errors are not valid. The solution is **permutation inference**: reassign the treatment to each
donor unit in turn, compute the same causal estimate, and compare the treated unit's estimate to
the distribution of placebo estimates. If the treated unit's effect is larger than nearly all
placebo effects, the result is statistically significant by permutation.

<div class="callout-key">

<strong>Key Concept:</strong> Classical frequentist inference assumes a large number of independent observations. Synthetic
control has one treated unit and a small number of post-intervention periods — classical standard
errors are not valid.

</div>

## The Fundamental Inference Problem

With one treated unit, there is no classical sampling distribution. The observed causal effect
could arise from:
1. A genuine treatment effect
2. Poor pre-period fit that creates spurious divergence
3. Chance noise in the post-period

Permutation inference addresses all three by using the distribution of placebo estimates as the
null distribution. The key insight: if the treatment has no effect, the treated unit should look
like a typical donor unit — its gap from the synthetic counterpart should be similar to the gaps
of donors from their own synthetic counterparts.

---

## In-Space Placebo Tests

**Procedure:**
1. For each donor unit $j \in \{2, \ldots, J+1\}$:
   a. Treat unit $j$ as if it were the treated unit
   b. Construct a synthetic counterpart using the remaining donors (excluding unit $j$)
   c. Compute the post-intervention gap for unit $j$
2. Compare the treated unit's gap to the distribution of donor gaps

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def synthetic_control_weights(y_treated_pre, Y_donors_pre, n_pre):
    """
    Solve for SC weights: minimize squared distance between
    treated pre-period outcome and weighted donor sum.

    Parameters
    ----------
    y_treated_pre : array of shape (n_pre,)
    Y_donors_pre : array of shape (n_pre, n_donors)
    n_pre : int

    Returns
    -------
    weights : array of shape (n_donors,)
    """
    n_donors = Y_donors_pre.shape[1]
    x0 = np.ones(n_donors) / n_donors  # Equal initial weights

    def objective(w):
        y_synthetic = Y_donors_pre @ w
        return np.sum((y_treated_pre - y_synthetic) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0.0, 1.0)] * n_donors

    result = minimize(
        objective, x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    return result.x


def run_placebo_tests(Y_all, n_pre, treated_idx=0):
    """
    Run in-space placebo tests for all units.

    Parameters
    ----------
    Y_all : array of shape (n_periods, n_units)
        Outcome matrix with treated unit as column 0
    n_pre : int
        Number of pre-intervention periods
    treated_idx : int
        Column index of the treated unit (default 0)

    Returns
    -------
    gaps : array of shape (n_post, n_units)
        Post-intervention gap for each unit (treated + all donors)
    rmspe_pre : array of shape (n_units,)
        Pre-intervention RMSPE for each unit (fit quality)
    """
    n_periods, n_units = Y_all.shape
    n_post = n_periods - n_pre

    gaps = np.zeros((n_post, n_units))
    rmspe_pre = np.zeros(n_units)

    for j in range(n_units):
        # Unit j plays the role of "treated"
        y_j_pre = Y_all[:n_pre, j]
        y_j_post = Y_all[n_pre:, j]

        # All other units are donors
        donor_cols = [k for k in range(n_units) if k != j]
        Y_donors_pre = Y_all[:n_pre, :][:, donor_cols]
        Y_donors_post = Y_all[n_pre:, :][:, donor_cols]

        # Optimize weights
        w = synthetic_control_weights(y_j_pre, Y_donors_pre, n_pre)

        # Pre-period fit quality
        y_synthetic_pre = Y_donors_pre @ w
        rmspe_pre[j] = np.sqrt(np.mean((y_j_pre - y_synthetic_pre) ** 2))

        # Post-period gap
        y_synthetic_post = Y_donors_post @ w
        gaps[:, j] = y_j_post - y_synthetic_post

    return gaps, rmspe_pre
```

</div>

---

## The Permutation P-Value

The permutation p-value is:

$$p = \frac{|\{j : \text{RMSPE\_post}(j) / \text{RMSPE\_pre}(j) \geq \text{RMSPE\_post}(1) / \text{RMSPE\_pre}(1)\}|}{J + 1}$$

where RMSPE_post is the root mean squared post-period gap and RMSPE_pre is the root mean squared
pre-period residual.

The ratio RMSPE_post / RMSPE_pre is used rather than the raw post-period gap because units with
poor pre-period fit (high RMSPE_pre) will naturally have larger post-period gaps from noise alone.
Normalizing by RMSPE_pre focuses attention on post-period gaps that are large *relative to
pre-period fit quality*.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>

```python
def placebo_p_value(gaps, rmspe_pre, n_pre, treated_idx=0):
    """
    Compute the permutation p-value for the treated unit.

    Parameters
    ----------
    gaps : array of shape (n_post, n_units)
    rmspe_pre : array of shape (n_units,)
    n_pre : int
    treated_idx : int

    Returns
    -------
    p_value : float
    """
    n_post = gaps.shape[0]

    # Post-period RMSPE for each unit
    rmspe_post = np.sqrt(np.mean(gaps ** 2, axis=0))

    # Ratio: post-period gap relative to pre-period fit
    ratio = rmspe_post / rmspe_pre

    # Permutation p-value: fraction of units with ratio >= treated unit
    treated_ratio = ratio[treated_idx]
    p_value = (ratio >= treated_ratio).mean()

    return p_value, ratio


# Common threshold: exclude donors with rmspe_pre > 2 * treated unit's rmspe_pre
def filtered_p_value(gaps, rmspe_pre, treated_idx=0, rmspe_multiplier=2.0):
    """
    Compute p-value excluding poor-fit donors.

    Donors with pre-period RMSPE more than rmspe_multiplier times the treated
    unit's RMSPE are excluded — their post-period gaps are driven by poor fit,
    not the treatment.
    """
    treated_rmspe_pre = rmspe_pre[treated_idx]
    good_fit_mask = rmspe_pre <= rmspe_multiplier * treated_rmspe_pre
    good_fit_mask[treated_idx] = True  # Always include treated unit

    gaps_filtered = gaps[:, good_fit_mask]
    rmspe_pre_filtered = rmspe_pre[good_fit_mask]
    new_treated_idx = np.where(good_fit_mask)[0].tolist().index(treated_idx)

    return placebo_p_value(gaps_filtered, rmspe_pre_filtered, None, new_treated_idx)
```

</div>

---

## Visualizing Placebo Tests: The Spaghetti Plot

The canonical synthetic control inference plot shows all placebo gaps alongside the treated
unit's gap. The treated unit should stand out — its gap should be clearly larger in the post-period.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_placebo_gaps(gaps, rmspe_pre, time_post, treated_idx=0,
                      rmspe_multiplier=2.0, ax=None):
    """
    Plot all placebo gaps vs the treated unit gap.

    Parameters
    ----------
    gaps : array of shape (n_post, n_units)
    rmspe_pre : array of shape (n_units,)
    time_post : array of shape (n_post,) — post-intervention time points
    treated_idx : int
    rmspe_multiplier : float — exclude donors with poor pre-period fit
    ax : matplotlib axis (optional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    treated_rmspe_pre = rmspe_pre[treated_idx]
    good_fit_mask = rmspe_pre <= rmspe_multiplier * treated_rmspe_pre

    # Plot donor placebos (gray)
    for j in range(gaps.shape[1]):
        if j == treated_idx:
            continue
        color = "#aaaaaa" if good_fit_mask[j] else "#dddddd"
        alpha = 0.4 if good_fit_mask[j] else 0.15
        ax.plot(time_post, gaps[:, j], color=color, alpha=alpha, linewidth=1)

    # Plot treated unit (blue, highlighted)
    ax.plot(time_post, gaps[:, treated_idx], color="#2980b9", linewidth=2.5,
            label="Treated unit", zorder=5)

    ax.axhline(0, color="black", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Time (post-intervention)")
    ax.set_ylabel("Gap (treated − synthetic)")
    ax.set_title("Placebo Tests: Treated Unit vs All Donors")
    ax.legend()

    return ax
```

</div>

---

## In-Time Placebo Tests

An alternative to in-space placebos is to run the analysis with a **placebo treatment date**
within the pre-intervention period.

**Logic:** If we pretend the intervention happened 5 years earlier (when we know there was no
treatment), the synthetic control should show no divergence at that placebo date. If the SC shows
a large gap at a placebo date, either the pre-period fit is poor or the donor pool is poorly
chosen.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>

```python
def in_time_placebo(Y_all, true_t0, placebo_t0, treated_idx=0):
    """
    Run in-time placebo: treat placebo_t0 as if it were the intervention.
    Use only pre-placebo data for weight optimization.

    Parameters
    ----------
    Y_all : array of shape (n_periods, n_units)
    true_t0 : int — true intervention period
    placebo_t0 : int — placebo treatment period (must be < true_t0)
    treated_idx : int

    Returns
    -------
    gap : array of placebo-period gap for treated unit
    rmspe_placebo_pre : float — pre-placebo RMSPE
    """
    assert placebo_t0 < true_t0, "Placebo date must be before true intervention"

    y_treated_pre = Y_all[:placebo_t0, treated_idx]
    donor_cols = [k for k in range(Y_all.shape[1]) if k != treated_idx]
    Y_donors_pre = Y_all[:placebo_t0, :][:, donor_cols]
    Y_donors_post = Y_all[placebo_t0:true_t0, :][:, donor_cols]
    y_treated_post = Y_all[placebo_t0:true_t0, treated_idx]

    w = synthetic_control_weights(y_treated_pre, Y_donors_pre, placebo_t0)
    y_synthetic_pre = Y_donors_pre @ w
    y_synthetic_post = Y_donors_post @ w

    rmspe_pre = np.sqrt(np.mean((y_treated_pre - y_synthetic_pre) ** 2))
    gap = y_treated_post - y_synthetic_post

    return gap, rmspe_pre
```

</div>

**Interpretation:**
- A large placebo gap in the pre-period is a red flag — the synthetic control is unreliable
- A small placebo gap (near zero) supports the validity of the approach

---

## When the Donor Pool Is Small

With fewer than 10 donors, placebo tests have low power. The permutation distribution has only
$J$ points (where $J$ is the number of donors), so the minimum achievable p-value is $1/(J+1)$.

With 7 donors, the minimum p-value is $1/8 = 0.125$. Even if the treated unit has the largest
gap, the p-value cannot be below 0.125. This is not a problem with the method — it is an honest
reflection of the limited evidence available with few comparison units.

**Practical guidance for small donor pools:**
1. Report the p-value with the donor pool size explicitly stated
2. Report the rank of the treated unit (e.g., "largest gap of 8 units")
3. Compute the minimum achievable p-value: $1 / (J + 1)$
4. Consider whether additional donors can be added (expand the pool)

---

## Confidence Sets for the Treatment Effect

The permutation distribution can be inverted to construct confidence sets for the treatment effect.

For a null hypothesis $H_0: \alpha = \alpha_0$ (effect is exactly $\alpha_0$), subtract $\alpha_0$
from the treated unit's post-intervention outcomes and rerun the placebo test. The confidence set
at level $1 - \alpha$ is the set of $\alpha_0$ values for which the null is not rejected.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>

```python
def confidence_set_sc(Y_all, n_pre, treated_idx=0, alpha=0.1, alpha0_grid=None):
    """
    Construct a confidence set by inverting placebo tests.

    For each hypothesized effect alpha_0 in alpha0_grid:
    - Subtract alpha_0 from treated unit's post-period outcomes
    - Rerun placebo tests
    - Collect alpha_0 values where p-value > alpha

    This is computationally intensive — run on a coarse grid first.
    """
    if alpha0_grid is None:
        alpha0_grid = np.linspace(-50, 10, 61)

    n_post = Y_all.shape[0] - n_pre
    in_set = []

    for alpha_0 in alpha0_grid:
        Y_shifted = Y_all.copy()
        # Subtract hypothesized effect from treated unit's post-period
        Y_shifted[n_pre:, treated_idx] -= alpha_0

        gaps, rmspe_pre = run_placebo_tests(Y_shifted, n_pre, treated_idx)
        p_val, _ = placebo_p_value(gaps, rmspe_pre, n_pre, treated_idx)

        if p_val > alpha:
            in_set.append(alpha_0)

    return in_set
```

</div>

---

## Synthetic Control vs ITS: Inference Comparison

| Aspect | ITS (Bayesian) | Synthetic Control |
|--------|---------------|-------------------|
| Uncertainty quantification | Posterior distribution | Permutation p-values |
| Confidence intervals | HDI from posterior | Inversion of permutation test |
| Distributional assumptions | Gaussian likelihood | None (non-parametric) |
| Minimum observations needed | 5+ post periods | 1+ post period |
| Multiple testing | Handled by posterior | Requires multiple testing correction |
| Software | CausalPy + PyMC | CausalPy + scipy.optimize |

---

## Reporting Standards

A complete synthetic control report includes:

1. **Pre-intervention fit plot**: treated vs. synthetic trajectory for the full pre-period
2. **RMSPE table**: pre-period RMSPE for treated unit and all donors
3. **Weight table**: non-zero donor weights
4. **Post-intervention gap plot**: with donor placebos overlaid
5. **Permutation p-value**: with donor pool size stated explicitly
6. **Sensitivity**: results with different donor pools or different predictor sets

**Template results table:**

| Year | Treated | Synthetic | Gap | 94% HDI |
|------|---------|-----------|-----|---------|
| 1989 | 102.3 | 114.7 | −12.4 | [−19.1, −5.7] |
| 1990 | 99.1 | 115.2 | −16.1 | [−23.4, −8.8] |
| ... | ... | ... | ... | ... |

---

## Connections

<div class="callout-info">

<strong>How this connects to the rest of the course:</strong>

</div>

- **Builds on:** Guide 01 (synthetic control basics), Module 00 (potential outcomes)
- **Leads to:** Notebook 02 (placebo tests in code), Notebook 03 (CausalPy SC API)
- **Related to:** Difference-in-differences (SC is a generalization), randomization inference


## Practice Questions

### Question 1: Conceptual Check
**Question:** In your own words, explain the core concept of Inference for Synthetic Control: Placebo Tests and Permutation Methods and why it matters for practical applications. What problem does it solve that simpler approaches cannot?

### Question 2: Application
**Question:** Describe a real-world scenario where you would apply the techniques from this guide. What assumptions would you need to verify before proceeding?

## Further Reading

- Abadie, A., Diamond, A., and Hainmueller, J. (2010). "Synthetic Control Methods for Comparative
  Case Studies." *Journal of the American Statistical Association*, 105(490), 493–505.
- Firpo, S., and Possebom, V. (2018). "Synthetic Control Method: Inference, Sensitivity Analysis
  and Confidence Sets." *Journal of Causal Inference*, 6(2).
- Cattaneo, M. D., Feng, Y., and Titiunik, R. (2021). "Prediction Intervals for Synthetic Control
  Methods." *Journal of the American Statistical Association*, 116(536), 1865–1880.


## Resources

<a class="link-card" href="../notebooks/01_synthetic_control_basics.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises for this topic.</div>
</a>
