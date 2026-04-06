# Heterogeneous Treatment Effects

> **Reading time:** ~5 min | **Module:** 8 — Heterogeneous Effects | **Prerequisites:** Modules 0-5

## In Brief

You will learn how to estimate Conditional Average Treatment Effects (CATE) — how treatment effects vary across individuals — using `econml.dml.DML`. Beyond the average effect, CATE reveals WHO benefits most from a treatment. BLP and GATES analyses provide formal tests for heterogeneity.

<div class="callout-insight">
<strong>Key Insight:</strong> The ATE tells you the treatment works on average. CATE tells you FOR WHOM it works best. In commodity markets, this means identifying which sectors, regions, or time periods are most affected by inventory surprises, policy changes, or supply shocks.
</div>

<div class="callout-key">
<strong>Key Concept:</strong> You will learn how to estimate Conditional Average Treatment Effects (CATE) — how treatment effects vary across individuals — using `econml.dml.DML`. Beyond the average effect, CATE reveals WHO benefits most from a treatment.
</div>

## Visual Explanation

```
FROM ATE TO CATE

ATE (one number):                 CATE (function of X):
                                  τ(X) = E[Y(1) - Y(0) | X]
┌─────────────────┐
│  θ = 0.5        │              ┌──────────────────────┐
│  (average)      │              │ τ(energy) = 1.2      │
└─────────────────┘              │ τ(metals) = 0.3      │
                                 │ τ(agriculture) = -0.1│
                                 └──────────────────────┘
```

## How CATE Estimation Works with econml

The `econml.dml.DML` estimator uses the DML framework to estimate CATE as a function of observable characteristics:

$$\tau(X) = E[Y(1) - Y(0) | X]$$

The approach:
1. Residualise $Y$ and $D$ using ML (same as standard DML)
2. Model the residual-on-residual relationship as varying with $X$
3. Use a flexible final-stage model (forest, linear) to estimate $\tau(X)$

Commodity example: heterogeneous effects of inventory surprises on commodity futures across sectors. Some sectors (energy) are highly sensitive to inventory news, while others (precious metals) are barely affected.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import numpy as np
import pandas as pd
from econml.dml import DML, LinearDML, CausalForestDML
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

np.random.seed(42)
n = 5000
p = 20

X = np.random.randn(n, p)

# Sector indicator (0=energy, 1=metals, 2=agriculture)
sector = np.random.choice([0, 1, 2], n, p=[0.4, 0.3, 0.3])
W = np.column_stack([X, (sector == 0).astype(float),
                     (sector == 1).astype(float)])

# Treatment: inventory surprise (continuous)
D = 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.randn(n) * 0.5

# Heterogeneous treatment effect
tau = np.where(sector == 0, 1.2,  # Energy: strong effect
       np.where(sector == 1, 0.3,  # Metals: moderate effect
                -0.1))              # Agriculture: near zero

Y = tau * D + 0.5 * X[:, 0] + 0.3 * X[:, 2] + np.random.randn(n) * 0.5

print(f"True CATE by sector:")
print(f"  Energy:      {np.mean(tau[sector == 0]):.2f}")
print(f"  Metals:      {np.mean(tau[sector == 1]):.2f}")
print(f"  Agriculture: {np.mean(tau[sector == 2]):.2f}")
print(f"  Overall ATE: {np.mean(tau):.2f}")
```

</div>

## How to Estimate CATE with CausalForestDML

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
# CausalForestDML: nonparametric CATE estimation
cf_dml = CausalForestDML(
    model_y=GradientBoostingRegressor(200, max_depth=5, random_state=42),
    model_t=GradientBoostingRegressor(200, max_depth=5, random_state=42),
    n_estimators=200,
    random_state=42
)
cf_dml.fit(Y, D, X=X, W=W)

# Predict CATE for each observation
cate_hat = cf_dml.effect(X)

# Compare to true CATE by sector
for s, name in enumerate(['Energy', 'Metals', 'Agriculture']):
    mask = sector == s
    print(f"{name:<15} True: {np.mean(tau[mask]):.2f}  "
          f"Est: {np.mean(cate_hat[mask]):.2f}")
```

</div>

## How to Run BLP Analysis

Best Linear Projection (BLP) tests whether CATE varies with observable characteristics:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
# BLP: project CATE onto covariates
from econml.dml import LinearDML

ldml = LinearDML(
    model_y=GradientBoostingRegressor(200, max_depth=5, random_state=42),
    model_t=GradientBoostingRegressor(200, max_depth=5, random_state=42),
    random_state=42
)
ldml.fit(Y, D, X=X[:, :5], W=W)

# Summary with p-values for heterogeneity
print(ldml.summary())
```

</div>

The BLP summary shows which covariates significantly predict treatment effect heterogeneity.

## How to Run GATES Analysis

Group Average Treatment Effects (GATES) sorts observations by estimated CATE and computes group averages:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
# GATES: sort by estimated CATE, compute group means
cate_sorted_idx = np.argsort(cate_hat)
n_groups = 5
group_size = n // n_groups

gates = []
for g in range(n_groups):
    start = g * group_size
    end = (g + 1) * group_size if g < n_groups - 1 else n
    group_idx = cate_sorted_idx[start:end]
    gates.append({
        'group': g + 1,
        'mean_cate': np.mean(cate_hat[group_idx]),
        'true_cate': np.mean(tau[group_idx]),
        'n': len(group_idx)
    })

gates_df = pd.DataFrame(gates)
print(gates_df.to_string(index=False))
```

</div>

GATES provides a simple visual test: if the group averages are flat, there is no heterogeneity. If they increase monotonically, the CATE model captures real variation.

<div class="callout-warning">
<strong>Warning:</strong> CATE estimation requires large samples. With small samples, the CATE function overfits and the BLP/GATES tests are underpowered. A sample of 5,000+ is a reasonable starting point for reliable CATE estimation.
</div>

## How to Visualise Treatment Effect Heterogeneity

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# GATES bar chart
axes[0].bar(gates_df['group'], gates_df['mean_cate'],
            color='steelblue', alpha=0.7, label='Estimated')
axes[0].bar(gates_df['group'], gates_df['true_cate'],
            color='crimson', alpha=0.3, label='True')
axes[0].set_xlabel('CATE Quintile')
axes[0].set_ylabel('Average Treatment Effect')
axes[0].set_title('GATES: Sorted Group Effects')
axes[0].legend()

# CATE by sector
sector_cates = [np.mean(cate_hat[sector == s]) for s in range(3)]
sector_true = [np.mean(tau[sector == s]) for s in range(3)]
x = np.arange(3)
axes[1].bar(x - 0.15, sector_true, 0.3, label='True', color='crimson', alpha=0.7)
axes[1].bar(x + 0.15, sector_cates, 0.3, label='Estimated', color='steelblue', alpha=0.7)
axes[1].set_xticks(x)
axes[1].set_xticklabels(['Energy', 'Metals', 'Agriculture'])
axes[1].set_title('CATE by Commodity Sector')
axes[1].legend()

plt.tight_layout()
plt.show()
```

</div>

## Connections

<div class="callout-info">
<strong>How this connects to the rest of the course:</strong>
</div>

**Builds on:**
- Module 05: PLR for average effects
- Module 06: IRM for binary treatment heterogeneity

**Leads to:**
- Module 09: Production pipeline with CATE reporting

**Related concepts:**
- Causal forests (Athey and Wager, 2019)
- Meta-learners (S-learner, T-learner, X-learner)
- Sorted effects (Chernozhukov et al., 2018)

## Practice Problems

### Implementation

**1. Meta-Learner Comparison:**
Compare `CausalForestDML` to `LinearDML` for CATE estimation. When does the linear model miss important heterogeneity?

**2. Sample Size Sensitivity:**
Run CATE estimation with n in {500, 1000, 2000, 5000, 10000}. Plot the RMSE of CATE estimates against sample size.


## Resources

<a class="link-card" href="../notebooks/01_heterogeneous_effects_notebook.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises for this topic.</div>
</a>
