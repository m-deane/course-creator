# Why OLS Fails with High-Dimensional Controls

> **Reading time:** ~5 min | **Module:** 1 — Ols Limitations | **Prerequisites:** Module 0 — Causal Inference Problem

## In Brief

You will learn why the obvious fix for high-dimensional confounding — using Lasso to select variables, then running OLS — produces biased treatment effects and invalid confidence intervals. The post-selection inference problem is fundamental, not a fixable nuance.

<div class="callout-insight">

<strong>Key Insight:</strong> Lasso selects variables that predict the outcome $Y$ well, not variables that confound the treatment $D$. Dropping a confounder of $D$ that is a weak predictor of $Y$ reintroduces omitted variable bias — and the standard errors do not account for the selection step.

</div>

<div class="callout-key">

<strong>Key Concept:</strong> You will learn why the obvious fix for high-dimensional confounding — using Lasso to select variables, then running OLS — produces biased treatment effects and invalid confidence intervals. The post-selection inference problem is fundamental, not a fixable nuance.

</div>

## Visual Explanation

```

THE POST-SELECTION INFERENCE TRAP

Step 1: Lasso selects         Step 2: OLS on selected
┌─────────────────────┐      ┌─────────────────────┐
│ X1 ✓ (predicts Y)   │      │ Y ~ D + X1 + X3     │
│ X2 ✗ (dropped!)     │ ──►  │                      │
│ X3 ✓ (predicts Y)   │      │ But X2 confounds D!  │
│ ...                  │      │ → OVB reintroduced   │
│ X200 ✗              │      │ → SEs are wrong      │
└─────────────────────┘      └─────────────────────┘

Problem: Lasso optimises prediction of Y, not confounding of D
```

## How Regularisation Bias Appears

When you add an L1 penalty to OLS, the Lasso objective is:

$$\hat{\beta}_{Lasso} = \arg\min_\beta \frac{1}{2n}\|Y - X\beta\|^2 + \lambda\|\beta\|_1$$

The penalty $\lambda\|\beta\|_1$ shrinks coefficients toward zero. Variables with small but nonzero effects get dropped entirely. The treatment coefficient $\theta$ is shrunk along with everything else.

In commodity markets, consider estimating the effect of carbon tax increases on power generation costs. Many controls (weather, fuel prices, demand, renewable capacity) have small but real confounding effects. Lasso may drop some of these, reintroducing bias.


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, Lasso
import statsmodels.api as sm

np.random.seed(42)
n = 1000
p = 200

# Simulate: many weak confounders
X = np.random.randn(n, p)

# Treatment correlated with several controls (confounding)
D = 0.3 * X[:, 0] + 0.2 * X[:, 1] + 0.15 * X[:, 3] + 0.1 * X[:, 7] + np.random.randn(n) * 0.5

# Outcome depends on treatment and confounders (some overlap with D's confounders)
true_effect = 1.5
Y = (true_effect * D + 0.8 * X[:, 0] + 0.5 * X[:, 2]
     + 0.3 * X[:, 3] + 0.2 * X[:, 5] + np.random.randn(n))

# Step 1: Lasso to select variables
lasso = LassoCV(cv=5, random_state=42).fit(X, Y)
selected = np.where(np.abs(lasso.coef_) > 0)[0]
print(f"Lasso selected {len(selected)} of {p} variables: {selected}")
print(f"Key confounders of D (X0, X1, X3, X7):")
print(f"  X0 selected: {0 in selected}")
print(f"  X1 selected: {1 in selected}")
print(f"  X3 selected: {3 in selected}")
print(f"  X7 selected: {7 in selected}")
```

</div>
</div>

Lasso selects based on prediction of $Y$. Variables like $X_1$ and $X_7$ confound $D$ but may be weak predictors of $Y$ — Lasso drops them, reintroducing bias.

## Why Post-Selection OLS Gives Invalid Inference

After Lasso selection, running OLS on the selected variables ignores the uncertainty from the selection step:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Step 2: OLS on Lasso-selected variables
X_selected = X[:, selected]
DX_selected = np.column_stack([D, X_selected])
post_lasso_ols = sm.OLS(Y, sm.add_constant(DX_selected)).fit()

# Compare to true effect
print(f"\nTrue effect:            {true_effect:.2f}")
print(f"Post-Lasso OLS:         {post_lasso_ols.params[1]:.2f}")
print(f"Post-Lasso OLS SE:      {post_lasso_ols.bse[1]:.3f}")
print(f"Post-Lasso 95% CI:      [{post_lasso_ols.conf_int()[1][0]:.2f}, {post_lasso_ols.conf_int()[1][1]:.2f}]")

# Check if true effect falls in CI
in_ci = post_lasso_ols.conf_int()[1][0] <= true_effect <= post_lasso_ols.conf_int()[1][1]
print(f"True effect in CI:      {in_ci}")
```

</div>
</div>

<div class="callout-warning">

<strong>Warning:</strong> The standard errors from post-selection OLS are WRONG. They ignore the model selection step, producing confidence intervals that are too narrow. The actual coverage of "95% CIs" from post-Lasso OLS is often 60-80% — far below the nominal 95%.

</div>

## How to Demonstrate the Coverage Problem

Run a Monte Carlo simulation to check actual coverage:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def coverage_simulation(n_sims=500, n=1000, p=200, true_theta=1.5):
    """Check actual coverage of post-Lasso OLS confidence intervals."""
    covers = 0
    biases = []

    for sim in range(n_sims):
        X_sim = np.random.randn(n, p)
        D_sim = 0.3 * X_sim[:, 0] + 0.2 * X_sim[:, 1] + np.random.randn(n) * 0.5
        Y_sim = true_theta * D_sim + 0.8 * X_sim[:, 0] + 0.5 * X_sim[:, 2] + np.random.randn(n)

        # Lasso selection
        lasso_sim = LassoCV(cv=5, random_state=sim).fit(X_sim, Y_sim)
        sel = np.where(np.abs(lasso_sim.coef_) > 0)[0]

        if len(sel) == 0:
            sel = np.array([0])

        # Post-selection OLS
        DX_sel = np.column_stack([D_sim, X_sim[:, sel]])
        model = sm.OLS(Y_sim, sm.add_constant(DX_sel)).fit()

        ci = model.conf_int()[1]
        if ci[0] <= true_theta <= ci[1]:
            covers += 1
        biases.append(model.params[1] - true_theta)

    coverage = covers / n_sims
    mean_bias = np.mean(biases)
    return coverage, mean_bias

coverage, bias = coverage_simulation(n_sims=200)
print(f"Nominal coverage:  95%")
print(f"Actual coverage:   {coverage:.1%}")
print(f"Mean bias:         {bias:+.3f}")
```

</div>
</div>

The actual coverage is substantially below 95%, confirming that post-Lasso OLS inference is invalid.

## How the Double Selection Fix Works (Preview)

Belloni, Chernozhukov, and Hansen (2014) proposed "double selection" — run Lasso on BOTH equations:

1. Lasso: select variables predicting $Y$
2. Lasso: select variables predicting $D$
3. Union of both sets → controls for OLS

This ensures confounders of $D$ are not dropped, even if they are weak predictors of $Y$.


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Double selection
lasso_y = LassoCV(cv=5, random_state=42).fit(X, Y)
lasso_d = LassoCV(cv=5, random_state=42).fit(X, D)

selected_y = set(np.where(np.abs(lasso_y.coef_) > 0)[0])
selected_d = set(np.where(np.abs(lasso_d.coef_) > 0)[0])
selected_union = sorted(selected_y | selected_d)

print(f"Selected for Y:  {sorted(selected_y)}")
print(f"Selected for D:  {sorted(selected_d)}")
print(f"Union:           {selected_union}")

# OLS on union
X_union = X[:, selected_union]
DX_union = np.column_stack([D, X_union])
double_sel_ols = sm.OLS(Y, sm.add_constant(DX_union)).fit()
print(f"\nDouble selection OLS: {double_sel_ols.params[1]:.2f} (SE: {double_sel_ols.bse[1]:.3f})")
print(f"True effect:         {true_effect:.2f}")
```

</div>

Double selection improves over single Lasso, but DML goes further — it replaces the linear Lasso with any ML model and adds orthogonal scores for robustness.

## Connections

<div class="callout-info">

<strong>How this connects to the rest of the course:</strong>


**Builds on:**
- Module 00: The causal inference problem and FWL theorem
- Lasso regression and regularisation paths

**Leads to:**
- Module 02: The orthogonalisation trick (replacing Lasso with flexible ML)
- Module 03: Why orthogonal scores provide robustness that double selection lacks

**Related concepts:**
- Post-double-selection (Belloni, Chernozhukov, Hansen 2014)
- Uniformly valid inference after model selection
- Selective inference (Berk et al., 2013)

## Practice Problems

### Conceptual

**1. Selection Asymmetry:**
Explain why Lasso selecting variables for $Y$ is not the same as selecting confounders of $D$. Give a commodity market example where a variable is a strong confounder of $D$ but a weak predictor of $Y$.

### Implementation

**2. Coverage Simulation:**
Extend the coverage simulation to compare three methods: (a) post-Lasso OLS, (b) double selection OLS, (c) DML with random forests. Plot coverage rates for each. Which achieves nominal 95% coverage?


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
def compare_coverage(n_sims=200, n=1000, p=200, true_theta=1.5):
    """Compare coverage of three estimation methods."""
    # Your implementation here
    pass
```



## Resources

<a class="link-card" href="../notebooks/01_regularisation_bias_notebook.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises for this topic.</div>
</a>
