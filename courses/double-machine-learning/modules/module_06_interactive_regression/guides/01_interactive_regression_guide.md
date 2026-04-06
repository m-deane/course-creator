# Interactive Regression Models

> **Reading time:** ~5 min | **Module:** 6 — Interactive Regression | **Prerequisites:** Module 5 — PLR

## In Brief

You will learn how to estimate causal effects of binary treatments using the Interactive Regression Model (IRM), which extends DML to handle heterogeneous treatment effects and propensity score nuisance functions. The IRM estimates both ATE and ATTE using `doubleml.DoubleMLIRM`.

<div class="callout-insight">
<strong>Key Insight:</strong> When treatment is binary (sanctions on/off, policy adopted/not), the PLR model is too restrictive because it assumes a constant treatment effect. The IRM allows the treatment effect to vary with covariates by modelling both the outcome function AND the propensity score.
</div>

<div class="callout-key">
<strong>Key Concept:</strong> You will learn how to estimate causal effects of binary treatments using the Interactive Regression Model (IRM), which extends DML to handle heterogeneous treatment effects and propensity score nuisance functions. The IRM estimates both ATE and ATTE using `doubleml.DoubleMLIRM`.
</div>

## Visual Explanation

```
IRM vs PLR

PLR (continuous treatment):        IRM (binary treatment):
Y = θD + g₀(X) + ε               Y = g₀(D,X) + ε
                                   E[D|X] = m₀(X)  (propensity score)

θ is constant for all X            ATE = E[g(1,X) - g(0,X)]
                                   ATTE = E[g(1,X) - g(0,X) | D=1]
```

## How the IRM Works

The IRM uses two nuisance functions:
1. **Outcome model:** $g_0(d, X) = E[Y | D=d, X]$ — predicts outcome for each treatment level
2. **Propensity score:** $m_0(X) = P(D=1 | X)$ — probability of treatment given controls

The doubly robust score for ATE is:

$$\psi_{ATE} = g(1, X) - g(0, X) + \frac{D(Y - g(1,X))}{m(X)} - \frac{(1-D)(Y - g(0,X))}{1 - m(X)} - \theta$$

This score is orthogonal — errors in $\hat{g}$ and $\hat{m}$ contribute only as products.

In the commodity example: sanctions impact on shipping freight rates. Treatment $D$ is whether a shipping route is under sanctions (binary), outcome $Y$ is the freight rate premium, and controls $X$ include route distance, vessel type, cargo type, insurance costs, and port infrastructure.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import numpy as np
import pandas as pd
from doubleml import DoubleMLData, DoubleMLIRM
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

np.random.seed(42)
n = 3000
p = 30

X = np.random.randn(n, p)
col_names = [f'X{i}' for i in range(p)]

# Propensity: probability of sanctions depends on route characteristics
propensity = 1 / (1 + np.exp(-(0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 3])))
D = np.random.binomial(1, propensity)

# Outcome: freight rate premium (heterogeneous treatment effect)
Y_0 = 2.0 + 0.5 * X[:, 0] + 0.3 * X[:, 2] + np.random.randn(n) * 0.5
Y_1 = Y_0 + 1.5 + 0.8 * X[:, 0]  # Treatment effect varies with X[0]
Y = D * Y_1 + (1 - D) * Y_0

true_ate = np.mean(Y_1 - Y_0)
true_atte = np.mean((Y_1 - Y_0)[D == 1])

print(f"True ATE:  {true_ate:.3f}")
print(f"True ATTE: {true_atte:.3f}")
print(f"Treatment prevalence: {D.mean():.1%}")
```

</div>

## How to Estimate ATE and ATTE

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
df = pd.DataFrame(X, columns=col_names)
df['freight_premium'] = Y
df['sanctions'] = D

dml_data = DoubleMLData(df, y_col='freight_premium',
                        d_cols='sanctions', x_cols=col_names)

# ATE estimation
irm_ate = DoubleMLIRM(dml_data,
                       ml_g=RandomForestRegressor(200, random_state=42),
                       ml_m=RandomForestClassifier(200, random_state=42),
                       score='ATE', n_folds=5, trimming_threshold=0.05)
irm_ate.fit()
print("ATE Results:")
print(irm_ate.summary)

# ATTE estimation
irm_atte = DoubleMLIRM(dml_data,
                        ml_g=RandomForestRegressor(200, random_state=42),
                        ml_m=RandomForestClassifier(200, random_state=42),
                        score='ATTE', n_folds=5, trimming_threshold=0.05)
irm_atte.fit()
print("\nATTE Results:")
print(irm_atte.summary)
```

</div>

<div class="callout-warning">
<strong>Warning:</strong> The `trimming_threshold` parameter drops observations with extreme propensity scores (very close to 0 or 1). Without trimming, the AIPW score can be numerically unstable. Set to 0.01-0.05 depending on your treatment prevalence.
</div>

## How to Diagnose Propensity Score Quality

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import matplotlib.pyplot as plt

# Get propensity score predictions
ml_m = RandomForestClassifier(200, random_state=42)
# Propensity scores from the fitted model
propensity_hat = irm_ate.models['ml_m']['sanctions'][0][0].predict_proba(X)[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(propensity_hat[D == 0], bins=30, alpha=0.5, label='Untreated', density=True)
axes[0].hist(propensity_hat[D == 1], bins=30, alpha=0.5, label='Treated', density=True)
axes[0].set_xlabel('Propensity Score')
axes[0].set_title('Propensity Score Distribution')
axes[0].legend()

axes[1].scatter(propensity, propensity_hat, alpha=0.1, s=5)
axes[1].plot([0, 1], [0, 1], 'r--')
axes[1].set_xlabel('True Propensity')
axes[1].set_ylabel('Estimated Propensity')
axes[1].set_title('Propensity Score Calibration')
plt.tight_layout()
plt.show()
```

</div>

Good overlap between treated and untreated propensity distributions is essential for IRM.

## Connections

<div class="callout-info">
<strong>How this connects to the rest of the course:</strong>
</div>

**Builds on:**
- Module 05: PLR with `doubleml`
- Propensity score methods
- Augmented inverse probability weighting (AIPW)

**Leads to:**
- Module 08: Heterogeneous treatment effects (CATE)
- Module 09: Production pipeline with IRM

**Related concepts:**
- Inverse probability weighting (IPW)
- Doubly robust estimation
- Marginal structural models

## Practice Problems

### Implementation

**1. ATE vs ATTE:**
Simulate a scenario where ATE and ATTE differ substantially (treated units have a larger effect). Estimate both with IRM and verify the difference matches the simulation.

**2. Propensity Score Overlap:**
Simulate a scenario with poor overlap (propensity scores near 0 or 1). Show that trimming matters by comparing results with `trimming_threshold=0.01` vs `0.1`.


## Resources

<a class="link-card" href="../notebooks/01_interactive_regression_notebook.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises for this topic.</div>
</a>
