# Clustered Standard Errors in Panel Data

> **Reading time:** ~20 min | **Module:** 05 — Advanced Topics | **Prerequisites:** Module 4


## Why Cluster Standard Errors?


<div class="callout-key">

**Key Concept Summary:** Panel data violates the i.i.d. assumption in two ways:

</div>

Panel data violates the i.i.d. assumption in two ways:
1. **Within-entity correlation**: Observations from the same entity are correlated over time
2. **Heteroskedasticity**: Error variance may differ across entities

Clustered standard errors account for these violations.

## The Clustering Intuition

Standard OLS assumes:
$$Var(\hat{\beta}) = \sigma^2 (X'X)^{-1}$$

With clustering, we use the "sandwich" estimator:
$$Var(\hat{\beta}) = (X'X)^{-1} \left(\sum_{g} X_g' \hat{u}_g \hat{u}_g' X_g\right) (X'X)^{-1}$$

where $g$ indexes clusters (entities).


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
import matplotlib.pyplot as plt
from scipy import stats

def demonstrate_clustering_importance():
    """
    Show why clustering matters with simulated data.
    """
    np.random.seed(42)

    # Generate data with within-entity correlation
    N = 50
    T = 20

    data = []
    for i in range(N):
        # Entity-specific error component
        entity_shock = np.random.normal(0, 2)

        # AR(1) errors within entity
        error_prev = 0
        rho_error = 0.7  # Serial correlation

<div class="callout-insight">

**Insight:** Panel data lets you control for unobservable differences between entities that are constant over time. This is the single most important reason to prefer panel data over repeated cross-sections.

</div>


        for t in range(T):
            x = np.random.normal(5, 1)
            # Correlated errors
            epsilon = rho_error * error_prev + np.random.normal(0, 0.5)
            error_prev = epsilon

            y = 2 + 1.5 * x + entity_shock + epsilon
            data.append({'entity': i, 'time': t, 'x': x, 'y': y})

    df = pd.DataFrame(data)

    # Estimate with different SE
    # 1. Standard (incorrect)
    ols_standard = smf.ols('y ~ x', data=df).fit()

    # 2. Heteroskedasticity-robust (still incorrect)
    ols_robust = smf.ols('y ~ x', data=df).fit(cov_type='HC1')

    # 3. Entity-clustered (correct)
    ols_clustered = smf.ols('y ~ x', data=df).fit(
        cov_type='cluster', cov_kwds={'groups': df['entity']}
    )

    print("COMPARISON OF STANDARD ERRORS:")
    print("=" * 60)
    print(f"{'Method':<25} {'SE(β)':<12} {'t-stat':<12} {'95% CI Width':<15}")
    print("-" * 60)

    for name, model in [('Standard OLS', ols_standard),
                        ('Robust (HC1)', ols_robust),
                        ('Entity-Clustered', ols_clustered)]:
        se = model.bse['x']
        t = model.tvalues['x']
        ci_width = model.conf_int().loc['x', 1] - model.conf_int().loc['x', 0]
        print(f"{name:<25} {se:<12.4f} {t:<12.2f} {ci_width:<15.4f}")

    print("\nNote: Standard SE is too small → inflated t-stat → false confidence")

    return df

df = demonstrate_clustering_importance()
```


</div>
</div>

## Types of Clustering

<div class="callout-warning">

**Warning:** Clustering at too fine a level understates standard errors; clustering at too coarse a level reduces power. The correct level is the level at which the treatment varies or at which observations are correlated.

</div>


### 1. Entity (One-Way) Clustering

The most common approach for panel data.


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def entity_clustering(df, y_col, x_cols, entity_col, time_col):
    """
    Entity-level clustering.
    """
    df_panel = df.set_index([entity_col, time_col])

    # Fixed effects with entity clustering
    fe = PanelOLS(df_panel[y_col], df_panel[x_cols],
                  entity_effects=True).fit(cov_type='clustered',
                                           cluster_entity=True)

    print("Fixed Effects with Entity-Clustered SE:")
    print(fe.summary.tables[1])

    return fe

fe_model = entity_clustering(df, 'y', ['x'], 'entity', 'time')
```


</div>
</div>

### 2. Time (One-Way) Clustering

For cross-sectional correlation across entities.


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def time_clustering(df, y_col, x_cols, entity_col, time_col):
    """
    Time-level clustering (for cross-sectional correlation).
    """
    df_panel = df.set_index([entity_col, time_col])

    # Fixed effects with time clustering
    fe = PanelOLS(df_panel[y_col], df_panel[x_cols],
                  entity_effects=True).fit(cov_type='clustered',
                                           cluster_time=True)

    print("Fixed Effects with Time-Clustered SE:")
    print(fe.summary.tables[1])

    return fe
```


</div>
</div>

### 3. Two-Way Clustering

For both within-entity and cross-sectional correlation.

```python
def two_way_clustering(df, y_col, x_cols, entity_col, time_col):
    """
    Two-way (entity and time) clustering.
    """
    df_panel = df.set_index([entity_col, time_col])

    # Two-way clustering
    fe = PanelOLS(df_panel[y_col], df_panel[x_cols],
                  entity_effects=True).fit(cov_type='clustered',
                                           cluster_entity=True,
                                           cluster_time=True)

    print("Fixed Effects with Two-Way Clustered SE:")
    print(fe.summary.tables[1])

    return fe

# Compare all three
print("\n" + "=" * 70)
print("COMPARISON OF CLUSTERING APPROACHES")
print("=" * 70)

print("\n1. Entity Clustering:")
entity_clustering(df, 'y', ['x'], 'entity', 'time')

print("\n2. Time Clustering:")
time_clustering(df, 'y', ['x'], 'entity', 'time')

print("\n3. Two-Way Clustering:")
two_way_clustering(df, 'y', ['x'], 'entity', 'time')
```

## When to Use Each Type

```python
def clustering_decision_guide():
    """
    Guide for choosing clustering level.
    """
    guide = """
CLUSTERING DECISION GUIDE:
================================================================================

1. ENTITY CLUSTERING (Most Common)
   Use when:
   - Errors are correlated within entities over time
   - Treatment varies at entity level
   - Entity is the unit of inference

   Example: Firm-level shocks persist over time

2. TIME CLUSTERING
   Use when:
   - Common shocks affect all entities in a period
   - Cross-sectional dependence is the concern

   Example: Market-wide events affect all firms

3. TWO-WAY CLUSTERING
   Use when:
   - Both within-entity AND cross-sectional correlation
   - Most conservative approach

   Example: Firms in same industry affected by both
            firm-specific persistence AND industry shocks

RULES OF THUMB:
--------------------------------------------------------------------------------
- Panel with entity FE: Cluster by entity (at minimum)
- If time FE included: Still cluster by entity (time FE don't fix serial corr)
- If worried about cross-sectional corr: Use two-way clustering
- When in doubt: Two-way is conservative
    """
    print(guide)

clustering_decision_guide()
```

## Finite Sample Corrections

Clustering can be imprecise with few clusters.

```python
def assess_cluster_quality(df, entity_col, time_col):
    """
    Assess whether clustering is reliable.
    """
    N = df[entity_col].nunique()
    T = df[time_col].nunique()
    n_obs = len(df)

    print("CLUSTER QUALITY ASSESSMENT:")
    print("=" * 60)
    print(f"Number of entities (clusters): {N}")
    print(f"Number of time periods: {T}")
    print(f"Total observations: {n_obs}")
    print(f"Average obs per cluster: {n_obs / N:.1f}")
    print()

    # Rules of thumb
    warnings = []

    if N < 30:
        warnings.append("⚠ Few clusters (<30): Consider finite-sample corrections")

    if N < 10:
        warnings.append("⚠ Very few clusters (<10): Clustering may be unreliable")

    if T > N:
        warnings.append("⚠ T > N: Consider time clustering or two-way")

    cluster_sizes = df.groupby(entity_col).size()
    if cluster_sizes.std() / cluster_sizes.mean() > 0.5:
        warnings.append("⚠ Unbalanced clusters: May affect inference")

    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  {w}")
    else:
        print("✓ Cluster structure appears adequate")

    return {
        'N': N, 'T': T, 'n_obs': n_obs,
        'avg_cluster_size': n_obs / N,
        'cluster_cv': cluster_sizes.std() / cluster_sizes.mean()
    }

assess_cluster_quality(df, 'entity', 'time')
```

## Simulation: Coverage Properties

```python
def simulate_coverage(n_simulations=500):
    """
    Simulate coverage rates for different SE approaches.
    """
    np.random.seed(42)

    true_beta = 1.5
    N = 50
    T = 15

    results = {
        'standard': {'estimates': [], 'covers': []},
        'robust': {'estimates': [], 'covers': []},
        'clustered': {'estimates': [], 'covers': []}
    }

    for _ in range(n_simulations):
        # Generate data with within-entity correlation
        data = []
        for i in range(N):
            entity_shock = np.random.normal(0, 2)
            error_prev = 0

            for t in range(T):
                x = np.random.normal(5, 1)
                epsilon = 0.6 * error_prev + np.random.normal(0, 0.5)
                error_prev = epsilon
                y = 2 + true_beta * x + entity_shock + epsilon
                data.append({'entity': i, 'time': t, 'x': x, 'y': y})

        df_sim = pd.DataFrame(data)

        # Standard
        ols_std = smf.ols('y ~ x', data=df_sim).fit()
        results['standard']['estimates'].append(ols_std.params['x'])
        ci = ols_std.conf_int().loc['x']
        results['standard']['covers'].append(ci[0] <= true_beta <= ci[1])

        # Robust
        ols_rob = smf.ols('y ~ x', data=df_sim).fit(cov_type='HC1')
        results['robust']['estimates'].append(ols_rob.params['x'])
        ci = ols_rob.conf_int().loc['x']
        results['robust']['covers'].append(ci[0] <= true_beta <= ci[1])

        # Clustered
        ols_cl = smf.ols('y ~ x', data=df_sim).fit(
            cov_type='cluster', cov_kwds={'groups': df_sim['entity']})
        results['clustered']['estimates'].append(ols_cl.params['x'])
        ci = ols_cl.conf_int().loc['x']
        results['clustered']['covers'].append(ci[0] <= true_beta <= ci[1])

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Distribution of estimates
    ax1 = axes[0]
    for method in ['standard', 'robust', 'clustered']:
        ax1.hist(results[method]['estimates'], bins=30, alpha=0.5, label=method)
    ax1.axvline(true_beta, color='red', linestyle='--', linewidth=2, label='True β')
    ax1.set_xlabel('Estimated β')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Estimates')
    ax1.legend()

    # Right: Coverage rates
    ax2 = axes[1]
    methods = ['Standard', 'Robust', 'Clustered']
    coverages = [np.mean(results['standard']['covers']),
                 np.mean(results['robust']['covers']),
                 np.mean(results['clustered']['covers'])]

    bars = ax2.bar(methods, coverages, color=['red', 'orange', 'green'])
    ax2.axhline(0.95, color='black', linestyle='--', linewidth=2, label='Nominal (95%)')
    ax2.set_ylabel('Coverage Rate')
    ax2.set_title('95% CI Coverage Rates')
    ax2.set_ylim(0, 1)

    # Add value labels
    for bar, cov in zip(bars, coverages):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{cov:.1%}', ha='center', fontsize=12)

    ax2.legend()

    plt.tight_layout()
    plt.show()

    print("\nCOVERAGE RATES (should be 95%):")
    print("=" * 50)
    print(f"Standard SE:  {np.mean(results['standard']['covers']):.1%}")
    print(f"Robust SE:    {np.mean(results['robust']['covers']):.1%}")
    print(f"Clustered SE: {np.mean(results['clustered']['covers']):.1%}")

simulate_coverage()
```

## Alternative: Bootstrap Clustering

For few clusters or unbalanced panels.

```python
def cluster_bootstrap(df, y_col, x_cols, entity_col, n_bootstrap=500):
    """
    Cluster bootstrap for standard errors.
    """
    entities = df[entity_col].unique()
    N = len(entities)

    bootstrap_estimates = []

    for _ in range(n_bootstrap):
        # Sample clusters (entities) with replacement
        sampled_entities = np.random.choice(entities, size=N, replace=True)

        # Build bootstrap sample
        bootstrap_data = []
        for idx, entity in enumerate(sampled_entities):
            entity_data = df[df[entity_col] == entity].copy()
            entity_data[entity_col] = idx  # Renumber entities
            bootstrap_data.append(entity_data)

        df_boot = pd.concat(bootstrap_data, ignore_index=True)

        # Estimate
        formula = f'{y_col} ~ {" + ".join(x_cols)}'
        result = smf.ols(formula, data=df_boot).fit()
        bootstrap_estimates.append(result.params[x_cols[0]])

    # Calculate bootstrap SE
    boot_se = np.std(bootstrap_estimates)
    original = smf.ols(f'{y_col} ~ {" + ".join(x_cols)}', data=df).fit()

    print("CLUSTER BOOTSTRAP RESULTS:")
    print("=" * 50)
    print(f"Point estimate: {original.params[x_cols[0]]:.4f}")
    print(f"Bootstrap SE:   {boot_se:.4f}")
    print(f"Analytic SE:    {original.bse[x_cols[0]]:.4f}")
    print(f"\n95% Bootstrap CI: [{np.percentile(bootstrap_estimates, 2.5):.4f}, "
          f"{np.percentile(bootstrap_estimates, 97.5):.4f}]")

    return boot_se, bootstrap_estimates

boot_se, boot_ests = cluster_bootstrap(df, 'y', ['x'], 'entity')
```

## Practical Recommendations

<div class="callout-danger">

**Danger:** Never include a lagged dependent variable in a fixed effects model without using an appropriate estimator (e.g., Arellano-Bond GMM). The within-transformation creates mechanical correlation between the transformed lagged variable and the transformed error, biasing all coefficients.

</div>


```python
def clustering_recommendations():
    """
    Practical recommendations for clustering.
    """
    recs = """
PRACTICAL RECOMMENDATIONS FOR CLUSTERING:
================================================================================

1. DEFAULT: Always cluster by entity in panel data
   - Serial correlation is almost always present
   - Clustering never hurts if done correctly

2. NUMBER OF CLUSTERS
   - Ideally: N ≥ 50 clusters
   - Acceptable: N ≥ 30 clusters
   - Problematic: N < 20 clusters → consider wild cluster bootstrap

3. IMPLEMENTATION
   - Use linearmodels: cluster_entity=True
   - Use statsmodels: cov_type='cluster', cov_kwds={'groups': entity}

4. REPORTING
   - Always report the clustering level
   - Note the number of clusters
   - Consider showing results with alternative clustering

5. TWO-WAY CLUSTERING
   - More conservative than one-way
   - Appropriate when both entity and time correlation present
   - May be too conservative → larger SE, less power

6. SMALL SAMPLE ADJUSTMENTS
   - Use finite-sample corrections when available
   - Consider wild cluster bootstrap
   - Be cautious interpreting marginal significance

7. COMMON MISTAKES TO AVOID
   - Don't forget to cluster (most common error!)
   - Don't cluster at wrong level (too fine or too coarse)
   - Don't compare p-values across different clustering specs
    """
    print(recs)

clustering_recommendations()
```

## Key Takeaways

1. **Always cluster** standard errors in panel data

2. **Entity clustering** handles within-entity serial correlation

3. **Two-way clustering** is more conservative, handles cross-sectional correlation

4. **Minimum clusters**: Need ~30+ for reliable inference

5. **Bootstrap** helps with few clusters or unbalanced panels

6. **Report clustering** - it affects inference significantly


---

## Conceptual Practice Questions

**Practice Question 1:** Why do conventional standard errors understate uncertainty in panel data with within-entity correlation?

**Practice Question 2:** At what level should you cluster standard errors -- entity level, time level, or both?


---

## Cross-References

<a class="link-card" href="./01_dynamic_panels.md">
  <div class="link-card-title">01 Dynamic Panels</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_dynamic_panels.md">
  <div class="link-card-title">01 Dynamic Panels — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./02_nickell_bias.md">
  <div class="link-card-title">02 Nickell Bias</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_nickell_bias.md">
  <div class="link-card-title">02 Nickell Bias — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

