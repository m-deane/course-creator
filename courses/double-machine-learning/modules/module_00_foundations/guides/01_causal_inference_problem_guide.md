# The Causal Inference Problem

## In Brief

You will learn why naive regression fails for causal inference when controls are high-dimensional, and how partialling out confounders with machine learning while preserving valid statistical inference forms the foundation of Double/Debiased Machine Learning.

> 💡 **Key Insight:** OLS forces you to choose between two bad options — omit confounders and get bias, or include too many and get variance explosion. DML breaks this tradeoff by using ML for prediction and econometrics for inference, keeping the best of both worlds.

## Visual Explanation

```
THE CAUSAL INFERENCE TRADEOFF

Few Controls (OLS)          Many Controls (OLS)         DML Approach
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│ ✗ Omitted var    │      │ ✗ Overfitting    │      │ ✓ ML handles     │
│   bias           │      │ ✗ Variance       │      │   high-dim X     │
│ ✗ Confounders    │      │   explosion      │      │ ✓ Orthogonality  │
│   distort θ      │      │ ✗ Multicollin.   │      │   protects θ     │
│                  │      │                  │      │ ✓ Cross-fitting   │
│ Bias: HIGH       │      │ Variance: HIGH   │      │   removes overfit │
│ Variance: LOW    │      │ Bias: LOW        │      │                  │
└──────────────────┘      └──────────────────┘      └──────────────────┘
```

## How Naive Regression Goes Wrong

Consider estimating the effect of OPEC production cuts on crude oil calendar spreads. The treatment $D$ is the size of the production cut (million barrels/day), the outcome $Y$ is the 1-3 month WTI calendar spread, and the controls $X$ include global demand indicators, inventory levels, shipping rates, refinery utilisation, and dozens more.

Here is what happens when you run OLS with too few controls:

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

np.random.seed(42)
n = 1000

# Simulate confounders (global demand, inventories, etc.)
X = np.random.randn(n, 5)

# Treatment: OPEC production cut (correlated with demand conditions)
D = 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.randn(n) * 0.5

# Outcome: calendar spread (depends on treatment AND confounders)
true_effect = 2.0  # True causal effect of production cuts on spread
Y = true_effect * D + 1.5 * X[:, 0] + 0.8 * X[:, 2] + np.random.randn(n)

# Naive OLS: no controls
naive = sm.OLS(Y, sm.add_constant(D)).fit()
print(f"True effect:     {true_effect:.2f}")
print(f"Naive OLS:       {naive.params[1]:.2f} (biased — confounders omitted)")
```

The naive estimate is biased upward because demand conditions ($X_0$) drive both OPEC decisions and spreads. OLS attributes some of the demand effect to the production cut.

## Why Adding More Controls Breaks Your Estimate

Now add all controls. With 5 controls this works fine, but watch what happens when we scale to realistic dimensions:

```python
# Scale up: 200 controls (realistic for commodity markets)
p = 200
X_large = np.random.randn(n, p)

# Only a few controls actually matter
D_large = 0.5 * X_large[:, 0] + 0.3 * X_large[:, 1] + np.random.randn(n) * 0.5
Y_large = (true_effect * D_large + 1.5 * X_large[:, 0]
           + 0.8 * X_large[:, 2] + np.random.randn(n))

# OLS with all 200 controls
controls_and_D = np.column_stack([D_large, X_large])
ols_full = sm.OLS(Y_large, sm.add_constant(controls_and_D)).fit()
print(f"OLS with {p} controls: {ols_full.params[1]:.2f}")
print(f"Standard error:        {ols_full.bse[1]:.2f} (massive uncertainty)")
print(f"95% CI width:          {2 * 1.96 * ols_full.bse[1]:.2f}")
```

With 200 controls and 1000 observations, OLS is unstable. The standard errors blow up because the model is overfitting to noise in the controls.

> ⚠️ **Warning:** Adding more controls to OLS does NOT always reduce bias. When $p/n$ is large, OLS suffers from regularisation bias (if you penalise) or variance explosion (if you don't). DML solves both problems simultaneously.

## How Frisch-Waugh-Lovell Solves the Problem (in Theory)

The Frisch-Waugh-Lovell (FWL) theorem says you can estimate the treatment effect in two steps:

The FWL theorem decomposes the treatment effect estimation for the model $Y = \theta D + X\beta + \epsilon$ into:

$$\tilde{Y} = Y - \hat{\Pi}_Y X \quad \text{(residualise outcome)}$$
$$\tilde{D} = D - \hat{\Pi}_D X \quad \text{(residualise treatment)}$$
$$\hat{\theta}_{FWL} = \frac{\tilde{D}'\tilde{Y}}{\tilde{D}'\tilde{D}} \quad \text{(regress residuals)}$$

In commodity terms: strip out everything that global demand, inventories, and other controls explain about both the production cut and the spread. Whatever correlation remains between the residuals IS the causal effect.

```python
# FWL with 5 controls (works perfectly)
X5 = X_large[:, :5]

# Step 1: Residualise Y on X
resid_Y = sm.OLS(Y_large, sm.add_constant(X5)).fit().resid

# Step 2: Residualise D on X
resid_D = sm.OLS(D_large, sm.add_constant(X5)).fit().resid

# Step 3: Regress residuals
fwl_result = sm.OLS(resid_Y, sm.add_constant(resid_D)).fit()
print(f"FWL estimate (5 controls): {fwl_result.params[1]:.2f}")
print(f"Standard error:            {fwl_result.bse[1]:.2f}")
```

FWL gives the same result as OLS with controls, but it reveals the key insight: the treatment effect estimation is really about residuals.

## Where OLS Stops and DML Begins

FWL works when you can estimate $E[Y|X]$ and $E[D|X]$ well with OLS. But OLS can only fit linear relationships. In commodity markets, the relationship between controls and outcomes is nonlinear — inventory effects are asymmetric, demand shocks interact with supply constraints, and seasonal patterns are complex.

DML replaces the OLS first stages with ML models:

$$\tilde{Y} = Y - \hat{g}(X) \quad \text{where } \hat{g} \approx E[Y|X] \text{ (random forest, gradient boosting, etc.)}$$
$$\tilde{D} = D - \hat{m}(X) \quad \text{where } \hat{m} \approx E[D|X] \text{ (random forest, gradient boosting, etc.)}$$
$$\hat{\theta}_{DML} = \frac{\tilde{D}'\tilde{Y}}{\tilde{D}'\tilde{D}}$$

The critical addition: cross-fitting and Neyman orthogonal scores ensure that ML estimation errors in $\hat{g}$ and $\hat{m}$ do not contaminate the treatment effect $\hat{\theta}$.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict

# DML preview (simplified — no cross-fitting yet)
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# ML residualisation with all 200 controls
Y_hat = cross_val_predict(rf, X_large, Y_large, cv=5)
D_hat = cross_val_predict(rf, X_large, D_large, cv=5)

resid_Y_ml = Y_large - Y_hat
resid_D_ml = D_large - D_hat

# Treatment effect from residuals
theta_dml = sm.OLS(resid_Y_ml, sm.add_constant(resid_D_ml)).fit()
print(f"DML preview (200 controls): {theta_dml.params[1]:.2f}")
print(f"Standard error:             {theta_dml.bse[1]:.2f}")
print(f"True effect:                {true_effect:.2f}")
```

This preview shows DML recovering the treatment effect even with 200 controls. The full DML procedure (Modules 02-04) adds cross-fitting and orthogonal scores to make this rigorous.

## Connections

**Builds on:**
- OLS regression and the omitted variables bias formula
- Frisch-Waugh-Lovell theorem
- Basic ML models (random forests, gradient boosting)

**Leads to:**
- Module 01: Why OLS with Lasso pre-selection fails for causal inference
- Module 02: The orthogonalisation trick (Robinson's partially linear model)
- Module 03: Neyman orthogonal scores (why DML is robust to first-stage errors)
- Module 04: Cross-fitting (eliminating overfitting bias)

**Related concepts:**
- Propensity score methods (partial out treatment assignment probability)
- Instrumental variables (partial out endogeneity via exclusion restriction)
- Targeted learning / TMLE (another doubly robust approach)

## Practice Problems

### Conceptual

**1. Omitted Variable Bias Direction:**
You estimate the effect of carbon tax increases on power generation fuel mix. You omit natural gas prices from your controls. Natural gas prices are positively correlated with carbon tax levels (both rise during economic expansion) and negatively affect coal generation share (gas substitutes for coal).

**Question:** In which direction is your OLS estimate biased? Write out the OVB formula and determine the sign.

### Implementation

**2. FWL vs OLS Equivalence:**
Implement the FWL decomposition and verify it gives identical point estimates to OLS with controls:

```python
def verify_fwl_equivalence(Y, D, X):
    """
    Show that FWL and OLS with controls produce
    the same treatment effect estimate.

    Parameters:
    -----------
    Y : array (n,) - outcome
    D : array (n,) - treatment
    X : array (n, p) - controls

    Returns:
    --------
    dict with 'ols_theta', 'fwl_theta', 'match' (bool)
    """
    # Your implementation here
    pass
```

**3. Dimension Scaling:**
Run the OLS-with-controls estimator for $p \in \{10, 50, 100, 200, 500\}$ with $n=1000$ fixed. Plot the standard error of $\hat{\theta}$ against $p$. At what ratio $p/n$ does OLS become unreliable?
