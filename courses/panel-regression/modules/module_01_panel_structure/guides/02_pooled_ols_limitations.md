# Limitations of Pooled OLS: When Simple Models Fail

> **Reading time:** ~20 min | **Module:** 01 — Panel Structure | **Prerequisites:** Module 0 Foundations


## Overview


<div class="callout-key">

**Key Concept Summary:** Pooled OLS treats panel data as a single cross-section, ignoring the panel structure. This guide explores when and why this approach fails.

</div>

Pooled OLS treats panel data as a single cross-section, ignoring the panel structure. This guide explores when and why this approach fails.

## The Pooled OLS Assumption

Pooled OLS assumes:

$$y_{it} = \alpha + X_{it}\beta + \epsilon_{it}$$

<div class="callout-insight">

**Insight:** Panel data lets you control for unobservable differences between entities that are constant over time. This is the single most important reason to prefer panel data over repeated cross-sections.

</div>


where $\epsilon_{it}$ is i.i.d. across all observations.

This ignores:
- **Entity heterogeneity**: Unobserved differences between units
- **Time dependence**: Correlation within entities over time

## Demonstrating Bias

<div class="callout-warning">

**Warning:** Reporting results without appropriate standard errors is a common mistake. In panel data, conventional OLS standard errors are almost always wrong -- use clustered or heteroskedasticity-robust standard errors.

</div>


### Simulation: Omitted Entity Effects

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>


```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
import matplotlib.pyplot as plt

np.random.seed(42)

# Simulation parameters
n_entities = 100
n_periods = 20
true_beta = 1.5

# Generate entity-specific effects (unobserved)
entity_effects = np.random.normal(0, 2, n_entities)

# Generate data
data = []
for i in range(n_entities):
    for t in range(n_periods):
        # X is correlated with entity effect (endogeneity!)
        x = 3 + 0.5 * entity_effects[i] + np.random.normal(0, 1)

        # True DGP includes entity effect
        y = 2 + true_beta * x + entity_effects[i] + np.random.normal(0, 0.5)

        data.append({
            'entity': i, 'time': t,
            'x': x, 'y': y,
            'true_effect': entity_effects[i]
        })

df = pd.DataFrame(data)

# Pooled OLS (ignores entity structure)
pooled = smf.ols('y ~ x', data=df).fit()

# Fixed Effects (accounts for entity heterogeneity)
df_panel = df.set_index(['entity', 'time'])
fe = PanelOLS(df_panel['y'], df_panel[['x']], entity_effects=True).fit()

print("Comparison of Estimates:")
print("=" * 50)
print(f"True β:        {true_beta:.4f}")
print(f"Pooled OLS β:  {pooled.params['x']:.4f}  (Biased!)")
print(f"Fixed Effects: {fe.params['x']:.4f}")
print(f"\nPooled OLS Bias: {pooled.params['x'] - true_beta:.4f}")
```


</div>

### Why Does Bias Occur?

The bias arises from the correlation between X and the error term:

$$E[\hat{\beta}_{OLS}] = \beta + \frac{Cov(X_{it}, \alpha_i)}{Var(X_{it})}$$

When $Cov(X_{it}, \alpha_i) \neq 0$, the estimate is biased.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>


```python
# Visualize the endogeneity
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Entity effects vs X
entity_x_means = df.groupby('entity')['x'].mean()
ax1 = axes[0]
ax1.scatter(entity_effects, entity_x_means, alpha=0.6)
ax1.set_xlabel('Entity Effect (αᵢ)')
ax1.set_ylabel('Mean X for Entity')
ax1.set_title(f'Correlation: {np.corrcoef(entity_effects, entity_x_means)[0,1]:.3f}')

# Right: Pooled vs FE fits
ax2 = axes[1]
x_range = np.linspace(df['x'].min(), df['x'].max(), 100)

# Pooled OLS line
y_pooled = pooled.params['Intercept'] + pooled.params['x'] * x_range
ax2.plot(x_range, y_pooled, 'r-', linewidth=2, label=f'Pooled (β={pooled.params["x"]:.2f})')

# True relationship (average entity effect = 0)
y_true = 2 + true_beta * x_range
ax2.plot(x_range, y_true, 'g--', linewidth=2, label=f'True (β={true_beta})')

ax2.scatter(df['x'], df['y'], alpha=0.1, s=10)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Pooled OLS vs True Relationship')
ax2.legend()

plt.tight_layout()
plt.show()
```


</div>

## Serial Correlation Problem

Even without endogeneity, pooled OLS ignores within-entity correlation:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>


```python
def demonstrate_serial_correlation(df, entity_col, time_col, y_col, x_cols):
    """
    Show that pooled OLS residuals are serially correlated.
    """
    # Fit pooled OLS
    formula = f"{y_col} ~ {' + '.join(x_cols)}"
    pooled = smf.ols(formula, data=df).fit()

    # Get residuals
    df['residual'] = pooled.resid

    # Calculate within-entity autocorrelation
    autocorrs = []
    for entity in df[entity_col].unique():
        entity_resid = df[df[entity_col] == entity]['residual'].values
        if len(entity_resid) > 1:
            autocorr = np.corrcoef(entity_resid[:-1], entity_resid[1:])[0, 1]
            if not np.isnan(autocorr):
                autocorrs.append(autocorr)

    mean_autocorr = np.mean(autocorrs)

    print("Serial Correlation Analysis:")
    print(f"  Mean within-entity autocorrelation: {mean_autocorr:.4f}")
    print(f"  Number of entities: {len(autocorrs)}")

    # Visual
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(autocorrs, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(mean_autocorr, color='red', linestyle='--', label=f'Mean: {mean_autocorr:.3f}')
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Within-Entity Autocorrelation')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Residual Autocorrelation by Entity')
    ax.legend()
    plt.show()

    return mean_autocorr

demonstrate_serial_correlation(df, 'entity', 'time', 'y', ['x'])
```


</div>

## Consequences for Inference

### Incorrect Standard Errors

```python
def compare_standard_errors(df):
    """
    Compare standard errors under different assumptions.
    """
    df_panel = df.set_index(['entity', 'time'])

    # Pooled OLS with default SE
    pooled = smf.ols('y ~ x', data=df).fit()

    # Pooled with clustered SE (correct for serial correlation)
    pooled_cluster = smf.ols('y ~ x', data=df).fit(
        cov_type='cluster', cov_kwds={'groups': df['entity']}
    )

    # Fixed effects with clustered SE
    fe_cluster = PanelOLS(df_panel['y'], df_panel[['x']],
                          entity_effects=True).fit(cov_type='clustered', cluster_entity=True)

    print("Standard Error Comparison:")
    print("=" * 60)
    print(f"{'Method':<35} {'SE(β)':<10} {'t-stat':<10}")
    print("-" * 60)
    print(f"{'Pooled OLS (default SE)':<35} {pooled.bse['x']:<10.4f} {pooled.tvalues['x']:<10.2f}")
    print(f"{'Pooled OLS (clustered SE)':<35} {pooled_cluster.bse['x']:<10.4f} {pooled_cluster.tvalues['x']:<10.2f}")
    print(f"{'Fixed Effects (clustered SE)':<35} {fe_cluster.std_errors['x']:<10.4f} {fe_cluster.tstats['x']:<10.2f}")
    print("=" * 60)
    print("\nNote: Default SE underestimates uncertainty due to serial correlation")

compare_standard_errors(df)
```

### Type I Error Inflation

```python
def simulate_type1_error(n_simulations=500):
    """
    Show that pooled OLS has inflated Type I errors.
    """
    np.random.seed(42)

    results = {'pooled': [], 'pooled_cluster': [], 'fe': []}

    for _ in range(n_simulations):
        # Generate data with entity effects but NO X effect (true β = 0)
        data = []
        for i in range(50):
            entity_effect = np.random.normal(0, 2)
            for t in range(15):
                x = 3 + 0.5 * entity_effect + np.random.normal(0, 1)
                y = 2 + 0 * x + entity_effect + np.random.normal(0, 0.5)  # β = 0
                data.append({'entity': i, 'time': t, 'x': x, 'y': y})

        df_sim = pd.DataFrame(data)
        df_sim_panel = df_sim.set_index(['entity', 'time'])

        # Pooled OLS
        pooled = smf.ols('y ~ x', data=df_sim).fit()
        results['pooled'].append(pooled.pvalues['x'] < 0.05)

        # Pooled with clustering
        pooled_cluster = smf.ols('y ~ x', data=df_sim).fit(
            cov_type='cluster', cov_kwds={'groups': df_sim['entity']}
        )
        results['pooled_cluster'].append(pooled_cluster.pvalues['x'] < 0.05)

        # Fixed effects
        fe = PanelOLS(df_sim_panel['y'], df_sim_panel[['x']],
                      entity_effects=True).fit(cov_type='clustered', cluster_entity=True)
        results['fe'].append(fe.pvalues['x'] < 0.05)

    print("Type I Error Rates (Nominal α = 0.05):")
    print("=" * 50)
    print(f"Pooled OLS (default SE):    {np.mean(results['pooled']):.3f}")
    print(f"Pooled OLS (clustered SE):  {np.mean(results['pooled_cluster']):.3f}")
    print(f"Fixed Effects (clustered):  {np.mean(results['fe']):.3f}")

simulate_type1_error()
```

## When Pooled OLS Is Acceptable

Pooled OLS may be appropriate when:

### 1. No Entity Heterogeneity

```python
# Test for entity effects
from scipy import stats

def test_for_entity_effects(df, y_col, x_cols, entity_col):
    """
    F-test for presence of entity fixed effects.
    """
    # Restricted: Pooled OLS
    formula = f"{y_col} ~ {' + '.join(x_cols)}"
    restricted = smf.ols(formula, data=df).fit()

    # Unrestricted: LSDV
    formula_fe = f"{y_col} ~ {' + '.join(x_cols)} + C({entity_col})"
    unrestricted = smf.ols(formula_fe, data=df).fit()

    # F-test
    n_entities = df[entity_col].nunique()
    n_obs = len(df)
    k = len(x_cols) + 1

    f_stat = ((restricted.ssr - unrestricted.ssr) / (n_entities - 1)) / \
             (unrestricted.ssr / (n_obs - n_entities - k))

    p_value = 1 - stats.f.cdf(f_stat, n_entities - 1, n_obs - n_entities - k)

    print("F-Test for Entity Fixed Effects:")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Conclusion: {'Reject pooled OLS' if p_value < 0.05 else 'Pooled OLS may be acceptable'}")

    return f_stat, p_value

test_for_entity_effects(df, 'y', ['x'], 'entity')
```

### 2. X Is Uncorrelated with Entity Effects

```python
def test_endogeneity_hausman(df, y_col, x_cols, entity_col, time_col):
    """
    Informal endogeneity test via RE-FE comparison.
    """
    from linearmodels.panel import RandomEffects

    df_panel = df.set_index([entity_col, time_col])

    # Fixed Effects
    fe = PanelOLS(df_panel[y_col], df_panel[x_cols], entity_effects=True).fit()

    # Random Effects
    re = RandomEffects(df_panel[y_col], df_panel[x_cols]).fit()

    # Compare coefficients
    diff = fe.params['x'] - re.params['x']

    print("Endogeneity Check (FE vs RE comparison):")
    print(f"  FE estimate: {fe.params['x']:.4f}")
    print(f"  RE estimate: {re.params['x']:.4f}")
    print(f"  Difference: {diff:.4f}")
    print(f"  Suggestion: {'Possible endogeneity - use FE' if abs(diff) > 0.1 else 'RE may be appropriate'}")

    return diff

test_endogeneity_hausman(df, 'y', ['x'], 'entity', 'time')
```

## Diagnostic Checklist

<div class="callout-danger">

**Danger:** Never include a lagged dependent variable in a fixed effects model without using an appropriate estimator (e.g., Arellano-Bond GMM). The within-transformation creates mechanical correlation between the transformed lagged variable and the transformed error, biasing all coefficients.

</div>


Before using pooled OLS on panel data:

| Check | Method | Action if Failed |
|-------|--------|------------------|
| Entity heterogeneity | F-test | Use Fixed Effects |
| Serial correlation | DW test, autocorrelation | Cluster SE or FE |
| Heteroskedasticity | BP test | Robust SE |
| Endogeneity | Hausman test | Use Fixed Effects |

```python
def pooled_ols_diagnostic(df, y_col, x_cols, entity_col, time_col):
    """
    Complete diagnostic for pooled OLS appropriateness.
    """
    print("=" * 60)
    print("POOLED OLS DIAGNOSTIC REPORT")
    print("=" * 60)

    # 1. Entity effects test
    print("\n1. Testing for Entity Fixed Effects:")
    f_stat, p_fe = test_for_entity_effects(df, y_col, x_cols, entity_col)

    # 2. Endogeneity check
    print("\n2. Checking for Endogeneity:")
    diff = test_endogeneity_hausman(df, y_col, x_cols, entity_col, time_col)

    # 3. Serial correlation
    print("\n3. Checking for Serial Correlation:")
    autocorr = demonstrate_serial_correlation(df, entity_col, time_col, y_col, x_cols)

    # Summary
    print("\n" + "=" * 60)
    print("RECOMMENDATION:")
    if p_fe < 0.05:
        print("  → Use Fixed Effects (significant entity heterogeneity)")
    elif abs(diff) > 0.1:
        print("  → Use Fixed Effects (possible endogeneity)")
    elif abs(autocorr) > 0.2:
        print("  → Use clustered standard errors at minimum")
    else:
        print("  → Pooled OLS may be acceptable")
    print("=" * 60)

# Run diagnostic
pooled_ols_diagnostic(df, 'y', ['x'], 'entity', 'time')
```

## Key Takeaways

1. **Pooled OLS ignores panel structure** and can lead to biased estimates when X correlates with entity effects

2. **Serial correlation** within entities violates i.i.d. assumption, leading to incorrect standard errors

3. **Type I error inflation** occurs when using default SE with panel data

4. **Always test** for entity effects and endogeneity before accepting pooled OLS

5. **At minimum**, use clustered standard errors when working with panel data


---

## Conceptual Practice Questions

**Practice Question 1:** Why does pooled OLS produce biased estimates when there is unobserved heterogeneity correlated with the regressors?

**Practice Question 2:** Under what conditions is pooled OLS actually the correct estimator for panel data?


---

## Cross-References

<a class="link-card" href="./01_pooled_ols.md">
  <div class="link-card-title">01 Pooled Ols</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_pooled_ols.md">
  <div class="link-card-title">01 Pooled Ols — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./02_data_formats.md">
  <div class="link-card-title">02 Data Formats</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_data_formats.md">
  <div class="link-card-title">02 Data Formats — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./03_between_within_decomposition.md">
  <div class="link-card-title">03 Between Within Decomposition</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_between_within_decomposition.md">
  <div class="link-card-title">03 Between Within Decomposition — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

