# The Orthogonalisation Trick

> **Reading time:** ~5 min | **Module:** 2 — Orthogonalisation | **Prerequisites:** Module 1 — OLS Limitations

## In Brief

You will learn Robinson's partially linear model and implement the residual-on-residual regression that forms the core of DML. The orthogonalisation trick partials out confounders using ML, then estimates the treatment effect from the cleaned-up residuals in about 30 lines of Python.

<div class="callout-insight">
<strong>Key Insight:</strong> Instead of selecting which controls to include, DML residualises BOTH the outcome and the treatment using ML. The treatment effect lives in the correlation between residuals — everything ML explains is confounding, everything left over is the causal signal.
</div>

<div class="callout-key">
<strong>Key Concept:</strong> You will learn Robinson's partially linear model and implement the residual-on-residual regression that forms the core of DML. The orthogonalisation trick partials out confounders using ML, then estimates the treatment effect from the cleaned-up residuals in about 30 lines of Python.
</div>

## The Orthogonalisation Pipeline

<div class="flow">
<div class="flow-step blue">1. ML predicts Y from X</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">2. ML predicts D from X</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step mint">3. Compute residuals</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. OLS on residuals</div>
</div>

```
THE DML PIPELINE (Orthogonalisation)

     Y (outcome)              D (treatment)
         │                        │
    ┌────▼────┐              ┌────▼────┐
    │  ML #1  │              │  ML #2  │
    │ ĝ(X)   │              │ m̂(X)   │
    └────┬────┘              └────┬────┘
         │                        │
    Ỹ = Y - ĝ(X)           D̃ = D - m̂(X)
         │                        │
         └──────────┬─────────────┘
                    │
              ┌─────▼─────┐
              │ OLS: Ỹ~D̃ │
              │  θ̂ = coef │
              └───────────┘
```

<div class="callout-warning">
<strong>Warning:</strong> If you only residualise the outcome (Y) but not the treatment (D), you get the Frisch-Waugh-Lovell result for OLS but NOT a valid DML estimator. Both must be residualised to achieve orthogonality.
</div>

## How Robinson's Partially Linear Model Works

The partially linear model (Robinson, 1988) assumes:

$$Y = \theta D + g_0(X) + \epsilon, \quad E[\epsilon|D, X] = 0$$
$$D = m_0(X) + V, \quad E[V|X] = 0$$

where $g_0(X) = E[Y - \theta D | X]$ captures the (possibly nonlinear) effect of controls on the outcome, and $m_0(X) = E[D|X]$ captures how controls predict treatment assignment.

In commodity terms, consider weather shocks on natural gas basis. The treatment $D$ is the weather shock intensity, the outcome $Y$ is the natural gas basis spread, and $X$ includes storage levels, pipeline capacity, regional demand, and heating degree days. The function $g_0(X)$ captures the complex nonlinear relationship between these factors and basis spreads.

Taking conditional expectations and subtracting:

$$Y - E[Y|X] = \theta(D - E[D|X]) + \epsilon$$
$$\tilde{Y} = \theta \tilde{D} + \epsilon$$

This is the residual-on-residual regression. The treatment effect $\theta$ is the slope of the regression of outcome residuals on treatment residuals.

## How to Implement DML in 30 Lines

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

def manual_dml(Y, D, X, n_folds=5):
    """
    Manual DML implementation in ~30 lines.

    Parameters
    ----------
    Y : array (n,) - outcome
    D : array (n,) - treatment
    X : array (n, p) - controls
    n_folds : int - number of cross-fitting folds

    Returns
    -------
    theta : float - treatment effect estimate
    se : float - standard error
    """
    n = len(Y)
    resid_Y = np.zeros(n)
    resid_D = np.zeros(n)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(X):
        # Fit ML models on training fold
        rf_y = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_d = RandomForestRegressor(n_estimators=100, random_state=42)

        rf_y.fit(X[train_idx], Y[train_idx])
        rf_d.fit(X[train_idx], D[train_idx])

        # Predict on held-out fold
        resid_Y[test_idx] = Y[test_idx] - rf_y.predict(X[test_idx])
        resid_D[test_idx] = D[test_idx] - rf_d.predict(X[test_idx])

    # Treatment effect: OLS of residuals
    theta = np.sum(resid_D * resid_Y) / np.sum(resid_D ** 2)

    # Standard error (heteroskedasticity-robust)
    epsilon = resid_Y - theta * resid_D
    se = np.sqrt(np.mean(resid_D ** 2 * epsilon ** 2) /
                 (np.mean(resid_D ** 2) ** 2) / n)

    return theta, se
```

</div>

This function implements the full DML algorithm with cross-fitting. Each fold trains ML models on training data and predicts on held-out data, ensuring all residuals are out-of-sample.

## How to Apply It to Weather Shocks on Natural Gas

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
np.random.seed(42)
n = 2000
p = 50

# Simulate: weather shocks on natural gas basis
X = np.random.randn(n, p)  # Storage, pipeline, demand, HDD, etc.

# Weather shock (treatment) depends on seasonal patterns in X
D = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + np.random.randn(n) * 0.3

# Basis spread (outcome) depends on treatment AND nonlinear controls
true_theta = 0.8
Y = (true_theta * D + np.exp(0.3 * X[:, 0]) + 0.5 * np.abs(X[:, 2])
     + 0.3 * X[:, 3] * X[:, 4] + np.random.randn(n) * 0.5)

# Run our manual DML
theta_hat, se_hat = manual_dml(Y, D, X)

print(f"True effect:   {true_theta:.2f}")
print(f"DML estimate:  {theta_hat:.2f}")
print(f"Standard error:{se_hat:.3f}")
print(f"95% CI:        [{theta_hat - 1.96*se_hat:.3f}, {theta_hat + 1.96*se_hat:.3f}]")

# Compare to OLS
DX = np.column_stack([D, X])
ols_model = sm.OLS(Y, sm.add_constant(DX)).fit()
print(f"\nOLS estimate:  {ols_model.params[1]:.2f}")
print(f"OLS SE:        {ols_model.bse[1]:.3f}")
```

</div>

DML handles the nonlinear relationships (sin, squared terms, interactions) that OLS misses. The ML first stages capture these patterns, and the residuals isolate the causal signal.

<div class="callout-warning">
<strong>Warning:</strong> The residualisation must be done out-of-sample (cross-fitting). Using in-sample ML predictions creates overfitting bias that contaminates the treatment effect. Module 04 covers why this matters and how cross-fitting fixes it.
</div>

## Why This Works: The Orthogonality Intuition

The key insight is geometric. After residualisation:

$$\tilde{Y} = \theta \tilde{D} + \epsilon$$

The residuals $\tilde{D}$ are **orthogonal** to the space spanned by $X$ (by construction — ML has removed the $X$-predictable component). This means:

1. Any remaining correlation between $\tilde{Y}$ and $\tilde{D}$ is causal (not confounded)
2. The treatment effect $\theta$ is identified by this residual correlation
3. Errors in the ML estimates ($\hat{g}$ and $\hat{m}$) affect both residuals but their impact on $\theta$ cancels out (to first order)

Point 3 is the orthogonality property that gives DML its name. Module 03 formalises this with Neyman orthogonal scores.

## Connections

<div class="callout-info">
<strong>How this connects to the rest of the course:</strong>
</div>

**Builds on:**
- Module 00: FWL theorem and the residual-on-residual idea
- Module 01: Why Lasso-based approaches fail

**Leads to:**
- Module 03: Neyman orthogonal scores (formal robustness guarantee)
- Module 04: Cross-fitting (eliminating overfitting bias)
- Module 05: `doubleml` implementation of PLR

**Related concepts:**
- Robinson (1988) semiparametric efficiency
- Partially linear regression in biostatistics
- Control function approach in econometrics

## Practice Problems

### Implementation

**1. Compare ML Models:**
Modify `manual_dml` to accept any sklearn regressor. Run it with: (a) LinearRegression, (b) RandomForest, (c) GradientBoosting, (d) Lasso. Compare the estimates when the true DGP has nonlinear confounding.

**2. In-Sample vs Out-of-Sample:**
Run DML with in-sample predictions (no cross-fitting) vs out-of-sample predictions. Plot the bias across 100 simulations for each approach.


## Resources

<a class="link-card" href="../notebooks/01_orthogonalisation_trick_notebook.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises for this topic.</div>
</a>
