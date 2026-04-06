# Instrumental Variables with DML

> **Reading time:** ~5 min | **Module:** 7 — Instrumental Variables | **Prerequisites:** Modules 0-4, IV concepts

## In Brief

You will learn how to combine instrumental variables with DML using the Partially Linear IV (PLIV) model. When treatment is endogenous (correlated with unobservable confounders), standard DML fails. PLIV uses an instrument $Z$ that affects $Y$ only through $D$, with ML first stages replacing the traditional linear 2SLS.

<div class="callout-insight">

<strong>Key Insight:</strong> Classical 2SLS uses linear first stages that miss nonlinear relationships between the instrument and the endogenous treatment. PLIV replaces these with ML models while preserving the IV identification strategy. The result: valid causal estimates even when the first-stage relationship is complex and controls are high-dimensional.

</div>

<div class="callout-key">

<strong>Key Concept:</strong> You will learn how to combine instrumental variables with DML using the Partially Linear IV (PLIV) model. When treatment is endogenous (correlated with unobservable confounders), standard DML fails.

</div>

## Visual Explanation

```

PLIV DAG

    Z (instrument)
    │
    ▼
    D (treatment)  ←──── U (unobserved confounder)
    │                        │
    ▼                        ▼
    Y (outcome)  ◄───────────┘

Conditions:
1. Relevance: Z → D (instrument affects treatment)
2. Exclusion: Z ⊥ Y | D, X (instrument affects Y only through D)
3. Exogeneity: Z ⊥ U | X (instrument is uncorrelated with unobservables)
```

## How PLIV Extends Standard DML

The PLIV model is:

$$Y = \theta D + g_0(X) + \epsilon, \quad E[\epsilon | Z, X] = 0$$
$$D = r_0(X) + h_0(Z, X) + V$$

where $Z$ is the instrument, $g_0(X)$ captures control effects on $Y$, and $r_0(X) + h_0(Z, X)$ captures how controls and the instrument predict $D$.


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np
import pandas as pd
from doubleml import DoubleMLData, DoubleMLPLIV
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)
n = 3000
p = 20

X = np.random.randn(n, p)
col_names = [f'X{i}' for i in range(p)]

# Instrument: weather anomaly (affects shipping but not commodity price directly)
Z = 0.5 * X[:, 0] + np.random.randn(n)

# Unobserved confounder: market sentiment
U = np.random.randn(n)

# Treatment: shipping volume (endogenous — affected by both Z and U)
D = 0.6 * Z + 0.3 * X[:, 1] + 0.4 * U + np.random.randn(n) * 0.3

# Outcome: commodity price spread (affected by D and U, NOT directly by Z)
true_theta = 0.5
Y = (true_theta * D + 0.5 * X[:, 0] + 0.3 * X[:, 2]
     + 0.5 * U + np.random.randn(n) * 0.3)  # U confounds!

print(f"True effect: {true_theta}")
print(f"Corr(D, U): {np.corrcoef(D, U)[0,1]:.3f} (endogenous!)")
print(f"Corr(Z, U): {np.corrcoef(Z, U)[0,1]:.3f} (instrument exogenous)")
```

</div>

## How to Estimate PLIV with doubleml


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
df = pd.DataFrame(X, columns=col_names)
df['price_spread'] = Y
df['shipping_volume'] = D
df['weather_anomaly'] = Z

dml_data = DoubleMLData(df,
                        y_col='price_spread',
                        d_cols='shipping_volume',
                        x_cols=col_names,
                        z_cols='weather_anomaly')

pliv = DoubleMLPLIV(dml_data,
                     ml_l=RandomForestRegressor(200, random_state=42),
                     ml_m=RandomForestRegressor(200, random_state=42),
                     ml_r=RandomForestRegressor(200, random_state=42),
                     n_folds=5)
pliv.fit()

print(pliv.summary)
print(f"\nTrue effect: {true_theta}")
```

</div>

<div class="callout-warning">

<strong>Warning:</strong> Weak instruments (low correlation between $Z$ and $D$) produce unreliable estimates with wide confidence intervals. Check the first-stage F-statistic (or its ML analogue) before trusting PLIV results.

</div>

## How to Check Instrument Strength


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict

# First-stage prediction quality
rf = RandomForestRegressor(200, random_state=42)
D_hat_with_Z = cross_val_predict(rf, np.column_stack([X, Z.reshape(-1, 1)]), D, cv=5)
D_hat_no_Z = cross_val_predict(rf, X, D, cv=5)

r2_with_z = r2_score(D, D_hat_with_Z)
r2_no_z = r2_score(D, D_hat_no_Z)
partial_r2 = (r2_with_z - r2_no_z) / (1 - r2_no_z)

print(f"R² (with Z):     {r2_with_z:.3f}")
print(f"R² (without Z):  {r2_no_z:.3f}")
print(f"Partial R² of Z: {partial_r2:.3f}")
print(f"Instrument is {'STRONG' if partial_r2 > 0.05 else 'WEAK'}")
```

</div>

The partial $R^2$ measures how much additional predictive power the instrument adds beyond the controls. Values below 0.05 suggest a weak instrument.

## Connections

<div class="callout-info">

<strong>How this connects to the rest of the course:</strong>

</div>

**Builds on:**
- Module 05: PLR with `doubleml`
- Classical 2SLS and IV estimation
- Instrument validity conditions

**Leads to:**
- Module 09: Production pipeline with IV diagnostics

**Related concepts:**
- Two-stage least squares (2SLS)
- Local average treatment effect (LATE)
- Weak instrument robust inference

## Practice Problems

### Implementation

**1. Weak vs Strong Instrument:**
Simulate data with varying instrument strength (coefficient on Z from 0.1 to 1.0). Plot the PLIV standard error and CI width against instrument strength.

**2. OLS vs PLIV:**
Show that OLS is biased (endogeneity) while PLIV recovers the true effect when the instrument is valid.


## Resources

<a class="link-card" href="../notebooks/01_iv_with_dml_notebook.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises for this topic.</div>
</a>
