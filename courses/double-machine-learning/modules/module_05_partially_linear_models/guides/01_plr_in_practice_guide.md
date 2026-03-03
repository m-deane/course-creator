# Partially Linear Models in Practice

## In Brief

You will learn how to estimate causal effects using `doubleml.DoubleMLPLR` end-to-end, including data preparation, nuisance model selection, hyperparameter tuning, and inference. This module takes you from manual DML (Module 02) to production-ready estimation with a battle-tested library.

> 💡 **Key Insight:** The `doubleml` package handles cross-fitting, orthogonal scores, and standard errors automatically. Your job is to choose good nuisance ML models and validate the results — the library handles the econometric machinery.

## Visual Explanation

```
doubleml PLR PIPELINE

Data(Y, D, X) → DoubleMLData → DoubleMLPLR(ml_l, ml_m) → .fit() → .summary
                                     │                        │
                                     │ Nuisance models:       │ Results:
                                     │ ml_l: Y ~ X            │ θ̂, SE, CI
                                     │ ml_m: D ~ X            │ p-value
                                     └────────────────────────-┘
```

## How to Set Up a PLR Pipeline with doubleml

The commodity example: estimating the causal effect of carbon price policy changes on power generation fuel mix. Treatment $D$ is the carbon price change, outcome $Y$ is the coal share of generation, and controls $X$ include gas prices, renewable capacity, demand, weather, and 45 other market variables.

```python
import numpy as np
import pandas as pd
from doubleml import DoubleMLData, DoubleMLPLR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV

np.random.seed(42)
n = 2000
p = 50

# Simulate carbon price policy scenario
X = np.random.randn(n, p)
col_names = [f'X{i}' for i in range(p)]

# Carbon price change (treatment) - driven by political and economic factors
D = (0.5 * np.sin(X[:, 0]) + 0.3 * X[:, 1] ** 2
     + 0.2 * X[:, 3] + np.random.randn(n) * 0.5)

# Coal generation share (outcome) - depends on carbon price AND market conditions
true_theta = -0.5  # Higher carbon price reduces coal share
Y = (true_theta * D + 0.8 * np.exp(0.2 * X[:, 0])
     + 0.4 * X[:, 2] + 0.3 * X[:, 4] * X[:, 5]
     + np.random.randn(n) * 0.3)

# Create DoubleML data object
df = pd.DataFrame(X, columns=col_names)
df['coal_share'] = Y
df['carbon_price_change'] = D

dml_data = DoubleMLData(df,
                        y_col='coal_share',
                        d_cols='carbon_price_change',
                        x_cols=col_names)
print(dml_data)
```

The `DoubleMLData` object organises your data into the Y, D, X structure that DML requires.

## How to Choose Nuisance Models

The two nuisance models predict $E[Y|X]$ and $E[D|X]$. These are pure prediction tasks — use whatever ML model predicts best.

```python
# Option 1: Random Forest (good default)
ml_l_rf = RandomForestRegressor(n_estimators=200, max_depth=10,
                                 min_samples_leaf=5, random_state=42)
ml_m_rf = RandomForestRegressor(n_estimators=200, max_depth=10,
                                 min_samples_leaf=5, random_state=42)

# Option 2: Gradient Boosting (often more accurate)
ml_l_gb = GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                     learning_rate=0.1, random_state=42)
ml_m_gb = GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                     learning_rate=0.1, random_state=42)

# Option 3: Lasso (if you suspect sparsity)
ml_l_lasso = LassoCV(cv=5, random_state=42)
ml_m_lasso = LassoCV(cv=5, random_state=42)
```

> ⚠️ **Warning:** The nuisance models should be tuned for PREDICTION accuracy, not for interpretability. A poorly tuned nuisance model produces noisy residuals, which widens confidence intervals (though orthogonality protects against bias). Use cross-validated prediction scores to select the best model.

## How to Fit and Interpret Results

```python
# Fit PLR with gradient boosting nuisance models
dml_plr = DoubleMLPLR(dml_data,
                       ml_l=ml_l_gb,
                       ml_m=ml_m_gb,
                       n_folds=5,
                       score='partialling out')
dml_plr.fit()

print(dml_plr.summary)
print(f"\nTrue effect: {true_theta:.2f}")
print(f"DML estimate: {dml_plr.coef[0]:.4f}")
print(f"Standard error: {dml_plr.se[0]:.4f}")
print(f"95% CI: [{dml_plr.confint().iloc[0, 0]:.4f}, {dml_plr.confint().iloc[0, 1]:.4f}]")
print(f"p-value: {dml_plr.pval[0]:.4f}")
```

The `summary` output includes the coefficient, standard error, t-statistic, and p-value. The confidence intervals use the normal approximation justified by DML's asymptotic theory.

## How to Compare Nuisance Model Choices

```python
results = {}
for name, ml_l, ml_m in [
    ('Lasso', ml_l_lasso, ml_m_lasso),
    ('Random Forest', ml_l_rf, ml_m_rf),
    ('Gradient Boosting', ml_l_gb, ml_m_gb),
]:
    dml = DoubleMLPLR(dml_data, ml_l=ml_l, ml_m=ml_m, n_folds=5)
    dml.fit()
    results[name] = {
        'coef': dml.coef[0],
        'se': dml.se[0],
        'ci_low': dml.confint().iloc[0, 0],
        'ci_high': dml.confint().iloc[0, 1]
    }
    print(f"{name:<20} theta={dml.coef[0]:.4f}  SE={dml.se[0]:.4f}")
```

If all nuisance models give similar treatment effects, the result is robust. If they differ substantially, the nonlinearity of confounding matters and you should prefer the most flexible model.

## How to Visualise Confidence Intervals

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))
names = list(results.keys())
coefs = [results[n]['coef'] for n in names]
ci_lows = [results[n]['ci_low'] for n in names]
ci_highs = [results[n]['ci_high'] for n in names]

y_pos = range(len(names))
ax.errorbar(coefs, y_pos, xerr=[[c - l for c, l in zip(coefs, ci_lows)],
                                  [h - c for c, h in zip(coefs, ci_highs)]],
            fmt='o', capsize=5, markersize=8, linewidth=2)
ax.axvline(x=true_theta, color='red', linestyle='--', label=f'True = {true_theta}')
ax.set_yticks(y_pos)
ax.set_yticklabels(names)
ax.set_xlabel('Treatment Effect')
ax.set_title('PLR Estimates by Nuisance Model Choice')
ax.legend()
plt.tight_layout()
plt.show()
```

## Connections

**Builds on:**
- Modules 02-04: Orthogonalisation, scores, cross-fitting
- `doubleml` package API

**Leads to:**
- Module 06: Interactive regression models (binary treatment)
- Module 07: Instrumental variables with DML
- Module 09: Production pipeline

## Practice Problems

### Implementation

**1. Hyperparameter Sensitivity:**
Run PLR with random forests using `n_estimators` in {50, 100, 200, 500}. Plot the treatment effect and CI for each. How sensitive is the result to the number of trees?

**2. Repeated Cross-Fitting:**
Run PLR 10 times with different `random_state` values for the cross-fitting. Plot the distribution of estimates. How much variability comes from the fold assignment?
