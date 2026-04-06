# Exploratory Panel Data Analysis

> **Reading time:** ~20 min | **Module:** 00 — Foundations | **Prerequisites:** None (entry point)


## Introduction


<div class="callout-key">

**Key Concept Summary:** Before running panel regressions, thorough exploratory analysis helps you understand your data's structure, identify potential issues, and choose appropriate methods. This guide covers essential di...

</div>

Before running panel regressions, thorough exploratory analysis helps you understand your data's structure, identify potential issues, and choose appropriate methods. This guide covers essential diagnostic techniques.

## Visualizing Panel Data

### Individual Time Series Plots

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create sample panel data
np.random.seed(42)
n_entities = 10
n_periods = 20

data = []
for i in range(n_entities):
    entity_effect = np.random.normal(0, 2)
    for t in range(n_periods):
        y = 10 + entity_effect + 0.5*t + np.random.normal(0, 1)
        x = 5 + 0.3*t + np.random.normal(0, 0.5)
        data.append({'entity': i, 'time': t, 'y': y, 'x': x})

df = pd.DataFrame(data)

# Plot individual entity trajectories
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Y over time by entity
ax1 = axes[0, 0]
for entity in df['entity'].unique():
    entity_data = df[df['entity'] == entity]
    ax1.plot(entity_data['time'], entity_data['y'], alpha=0.7, label=f'Entity {entity}')
ax1.set_xlabel('Time')
ax1.set_ylabel('Y')
ax1.set_title('Y Trajectories by Entity')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# X over time by entity
ax2 = axes[0, 1]
for entity in df['entity'].unique():
    entity_data = df[df['entity'] == entity]
    ax2.plot(entity_data['time'], entity_data['x'], alpha=0.7)
ax2.set_xlabel('Time')
ax2.set_ylabel('X')
ax2.set_title('X Trajectories by Entity')

# Scatter plot: pooled
ax3 = axes[1, 0]
ax3.scatter(df['x'], df['y'], alpha=0.5)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_title('Pooled Scatter Plot (X vs Y)')

# Scatter with entity coloring
ax4 = axes[1, 1]
scatter = ax4.scatter(df['x'], df['y'], c=df['entity'], cmap='tab10', alpha=0.6)
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_title('Scatter by Entity')
plt.colorbar(scatter, ax=ax4, label='Entity')

plt.tight_layout()
plt.show()
```


</div>

### Within vs Between Variation

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>


```python
def plot_within_between(df, entity_col, x_col, y_col):
    """
    Visualize within and between entity relationships.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

<div class="callout-insight">

**Insight:** Panel data lets you control for unobservable differences between entities that are constant over time. This is the single most important reason to prefer panel data over repeated cross-sections.

</div>


    # 1. Raw scatter (total variation)
    ax1 = axes[0]
    ax1.scatter(df[x_col], df[y_col], alpha=0.5)

    # Add pooled regression line
    z = np.polyfit(df[x_col], df[y_col], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df[x_col].min(), df[x_col].max(), 100)
    ax1.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Pooled slope: {z[0]:.3f}')
    ax1.set_xlabel(x_col)
    ax1.set_ylabel(y_col)
    ax1.set_title('Total Variation (Pooled OLS)')
    ax1.legend()

    # 2. Between variation (entity means)
    ax2 = axes[1]
    entity_means = df.groupby(entity_col)[[x_col, y_col]].mean()
    ax2.scatter(entity_means[x_col], entity_means[y_col], s=100, alpha=0.7)

    # Between regression line
    z_between = np.polyfit(entity_means[x_col], entity_means[y_col], 1)
    p_between = np.poly1d(z_between)
    ax2.plot(x_line, p_between(x_line), 'b-', linewidth=2,
             label=f'Between slope: {z_between[0]:.3f}')
    ax2.set_xlabel(f'{x_col} (entity mean)')
    ax2.set_ylabel(f'{y_col} (entity mean)')
    ax2.set_title('Between Variation')
    ax2.legend()

    # 3. Within variation (deviations from entity means)
    ax3 = axes[2]
    df_within = df.copy()
    df_within[f'{x_col}_demean'] = df[x_col] - df.groupby(entity_col)[x_col].transform('mean')
    df_within[f'{y_col}_demean'] = df[y_col] - df.groupby(entity_col)[y_col].transform('mean')

    ax3.scatter(df_within[f'{x_col}_demean'], df_within[f'{y_col}_demean'], alpha=0.5)

    # Within regression line
    z_within = np.polyfit(df_within[f'{x_col}_demean'], df_within[f'{y_col}_demean'], 1)
    p_within = np.poly1d(z_within)
    x_line_within = np.linspace(df_within[f'{x_col}_demean'].min(),
                                 df_within[f'{x_col}_demean'].max(), 100)
    ax3.plot(x_line_within, p_within(x_line_within), 'g-', linewidth=2,
             label=f'Within slope: {z_within[0]:.3f}')
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel(f'{x_col} (demeaned)')
    ax3.set_ylabel(f'{y_col} (demeaned)')
    ax3.set_title('Within Variation (Fixed Effects)')
    ax3.legend()

    plt.tight_layout()
    plt.show()

    return {
        'pooled_slope': z[0],
        'between_slope': z_between[0],
        'within_slope': z_within[0]
    }

# Visualize
slopes = plot_within_between(df, 'entity', 'x', 'y')
print("\nSlope comparison:")
for name, value in slopes.items():
    print(f"  {name}: {value:.4f}")
```


</div>

## Testing for Entity Effects

<div class="callout-warning">

**Warning:** Reporting results without appropriate standard errors is a common mistake. In panel data, conventional OLS standard errors are almost always wrong -- use clustered or heteroskedasticity-robust standard errors.

</div>


### Visual Test: Entity Mean Differences

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>


```python
def plot_entity_effects(df, entity_col, y_col):
    """
    Visualize entity-specific means with confidence intervals.
    """
    # Calculate entity means and standard errors
    entity_stats = df.groupby(entity_col)[y_col].agg(['mean', 'std', 'count'])
    entity_stats['se'] = entity_stats['std'] / np.sqrt(entity_stats['count'])
    entity_stats['ci_lower'] = entity_stats['mean'] - 1.96 * entity_stats['se']
    entity_stats['ci_upper'] = entity_stats['mean'] + 1.96 * entity_stats['se']
    entity_stats = entity_stats.sort_values('mean')

    # Grand mean
    grand_mean = df[y_col].mean()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x_pos = range(len(entity_stats))
    ax.errorbar(x_pos, entity_stats['mean'],
                yerr=[entity_stats['mean'] - entity_stats['ci_lower'],
                      entity_stats['ci_upper'] - entity_stats['mean']],
                fmt='o', capsize=3, capthick=1, markersize=8)

    ax.axhline(grand_mean, color='red', linestyle='--',
               label=f'Grand mean: {grand_mean:.2f}')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(entity_stats.index)
    ax.set_xlabel('Entity')
    ax.set_ylabel(f'{y_col} Mean')
    ax.set_title('Entity-Specific Means with 95% CI')
    ax.legend()

    plt.tight_layout()
    plt.show()

    # F-test for entity effects
    from scipy import stats

    # Between-group variance
    entity_means = df.groupby(entity_col)[y_col].transform('mean')
    ss_between = ((entity_means - grand_mean) ** 2).sum()
    df_between = df[entity_col].nunique() - 1

    # Within-group variance
    ss_within = ((df[y_col] - entity_means) ** 2).sum()
    df_within = len(df) - df[entity_col].nunique()

    # F-statistic
    f_stat = (ss_between / df_between) / (ss_within / df_within)
    p_value = 1 - stats.f.cdf(f_stat, df_between, df_within)

    print(f"\nF-test for entity effects:")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Conclusion: {'Significant entity effects' if p_value < 0.05 else 'No significant entity effects'}")

    return f_stat, p_value

# Run test
f_stat, p_value = plot_entity_effects(df, 'entity', 'y')
```


</div>

### Testing for Time Effects

```python
def test_time_effects(df, time_col, y_col):
    """
    Test for significant time effects.
    """
    # Time period means
    time_means = df.groupby(time_col)[y_col].mean()
    grand_mean = df[y_col].mean()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time_means.index, time_means.values, 'o-', linewidth=2, markersize=8)
    ax.axhline(grand_mean, color='red', linestyle='--', label=f'Grand mean: {grand_mean:.2f}')
    ax.set_xlabel('Time')
    ax.set_ylabel(f'{y_col} Mean')
    ax.set_title('Time-Specific Means')
    ax.legend()
    plt.tight_layout()
    plt.show()

    # F-test for time effects
    from scipy import stats

    time_means_all = df.groupby(time_col)[y_col].transform('mean')

    ss_time = ((time_means_all - grand_mean) ** 2).sum()
    df_time = df[time_col].nunique() - 1

    ss_residual = ((df[y_col] - time_means_all) ** 2).sum()
    df_residual = len(df) - df[time_col].nunique()

    f_stat = (ss_time / df_time) / (ss_residual / df_residual)
    p_value = 1 - stats.f.cdf(f_stat, df_time, df_residual)

    print(f"\nF-test for time effects:")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")

    return f_stat, p_value

test_time_effects(df, 'time', 'y')
```

## Correlation Analysis

### Within-Entity Correlations

```python
def within_correlation(df, entity_col, var1, var2):
    """
    Calculate within-entity correlation (correlation of demeaned variables).
    """
    # Demean both variables
    df_copy = df.copy()
    df_copy[f'{var1}_demean'] = df[var1] - df.groupby(entity_col)[var1].transform('mean')
    df_copy[f'{var2}_demean'] = df[var2] - df.groupby(entity_col)[var2].transform('mean')

    # Within correlation
    within_corr = df_copy[f'{var1}_demean'].corr(df_copy[f'{var2}_demean'])

    # Between correlation (entity means)
    entity_means = df.groupby(entity_col)[[var1, var2]].mean()
    between_corr = entity_means[var1].corr(entity_means[var2])

    # Total correlation
    total_corr = df[var1].corr(df[var2])

    return {
        'total': total_corr,
        'between': between_corr,
        'within': within_corr
    }

correlations = within_correlation(df, 'entity', 'x', 'y')
print("Correlation decomposition:")
for name, value in correlations.items():
    print(f"  {name}: {value:.4f}")
```

### Entity-Level Correlation Matrix

```python
def entity_correlation_matrix(df, entity_col, variables):
    """
    Compute correlation matrix at entity level (between-entity correlations).
    """
    entity_means = df.groupby(entity_col)[variables].mean()
    return entity_means.corr()

# If we had more variables
df['z'] = df['x'] * 0.5 + np.random.normal(0, 0.5, len(df))
corr_matrix = entity_correlation_matrix(df, 'entity', ['x', 'y', 'z'])

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
ax.set_title('Between-Entity Correlation Matrix')
plt.show()
```

## Detecting Common Panel Data Issues

### Serial Correlation in Residuals

```python
def test_serial_correlation(df, entity_col, time_col, residual_col):
    """
    Test for first-order serial correlation within entities.
    """
    from scipy import stats

    # Sort data
    df_sorted = df.sort_values([entity_col, time_col])

    # Create lagged residual
    df_sorted['resid_lag'] = df_sorted.groupby(entity_col)[residual_col].shift(1)

    # Remove NAs
    df_complete = df_sorted.dropna(subset=['resid_lag'])

    # Correlation
    rho = df_complete[residual_col].corr(df_complete['resid_lag'])

    # Approximate test (Durbin-Watson style)
    n = len(df_complete)
    se_rho = 1 / np.sqrt(n)
    t_stat = rho / se_rho
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))

    print(f"Serial correlation test:")
    print(f"  Estimated rho: {rho:.4f}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")

    return rho, p_value

# First, create simple residuals (y - mean)
df['resid'] = df['y'] - df['y'].mean()
test_serial_correlation(df, 'entity', 'time', 'resid')
```

### Heteroskedasticity Across Entities

```python
def test_heteroskedasticity_across_entities(df, entity_col, residual_col):
    """
    Test if residual variance differs across entities.
    """
    from scipy import stats

    # Entity-specific variances
    entity_vars = df.groupby(entity_col)[residual_col].var()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Box plot of residuals by entity
    df.boxplot(column=residual_col, by=entity_col, ax=axes[0])
    axes[0].set_title('Residual Distribution by Entity')
    axes[0].set_xlabel('Entity')

    # Bar plot of variances
    entity_vars.plot(kind='bar', ax=axes[1])
    axes[1].axhline(entity_vars.mean(), color='red', linestyle='--',
                    label=f'Mean var: {entity_vars.mean():.4f}')
    axes[1].set_title('Residual Variance by Entity')
    axes[1].set_xlabel('Entity')
    axes[1].set_ylabel('Variance')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    # Bartlett's test for equal variances
    groups = [group[residual_col].values for _, group in df.groupby(entity_col)]
    stat, p_value = stats.bartlett(*groups)

    print(f"Bartlett's test for equal variances:")
    print(f"  Test statistic: {stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Conclusion: {'Heteroskedasticity detected' if p_value < 0.05 else 'Homoskedasticity'}")

    return stat, p_value

test_heteroskedasticity_across_entities(df, 'entity', 'resid')
```

### Missing Data Patterns

```python
def analyze_missing_patterns(df, entity_col, time_col, value_cols):
    """
    Analyze patterns of missing data in panel.
    """
    # Create complete index
    entities = df[entity_col].unique()
    times = df[time_col].unique()

    # Check for missing entity-time combinations
    complete_index = pd.MultiIndex.from_product([entities, times],
                                                  names=[entity_col, time_col])

    df_indexed = df.set_index([entity_col, time_col])
    missing_combos = set(complete_index) - set(df_indexed.index)

    print(f"Missing entity-time combinations: {len(missing_combos)}")

    if len(missing_combos) > 0:
        print("Missing observations:")
        for combo in list(missing_combos)[:10]:
            print(f"  Entity {combo[0]}, Time {combo[1]}")
        if len(missing_combos) > 10:
            print(f"  ... and {len(missing_combos) - 10} more")

    # Check for missing values within observations
    print(f"\nMissing values in variables:")
    for col in value_cols:
        n_missing = df[col].isna().sum()
        pct_missing = n_missing / len(df) * 100
        print(f"  {col}: {n_missing} ({pct_missing:.2f}%)")

    # Visualize missing pattern
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create pivot table for visualization
    pivot = df.pivot_table(
        index=entity_col,
        columns=time_col,
        values=value_cols[0],
        aggfunc='count'
    ).fillna(0)

    sns.heatmap(pivot, cmap='YlGn', cbar_kws={'label': 'Observations'}, ax=ax)
    ax.set_title('Data Availability Matrix')
    plt.tight_layout()
    plt.show()

analyze_missing_patterns(df, 'entity', 'time', ['x', 'y'])
```

## Summary Statistics Report

<div class="callout-danger">

**Danger:** Never include a lagged dependent variable in a fixed effects model without using an appropriate estimator (e.g., Arellano-Bond GMM). The within-transformation creates mechanical correlation between the transformed lagged variable and the transformed error, biasing all coefficients.

</div>


```python
def panel_eda_report(df, entity_col, time_col, outcome_var, predictors):
    """
    Generate comprehensive EDA report for panel data.
    """
    print("=" * 60)
    print("PANEL DATA EXPLORATORY ANALYSIS REPORT")
    print("=" * 60)

    # Basic structure
    print("\n1. DATA STRUCTURE")
    print("-" * 40)
    print(f"   Entities (N): {df[entity_col].nunique()}")
    print(f"   Time periods (T): {df[time_col].nunique()}")
    print(f"   Total observations: {len(df)}")
    print(f"   Balanced panel: {len(df) == df[entity_col].nunique() * df[time_col].nunique()}")

    # Variable summary
    print("\n2. VARIABLE SUMMARY")
    print("-" * 40)
    all_vars = [outcome_var] + predictors
    print(df[all_vars].describe().T.to_string())

    # Variation decomposition
    print("\n3. VARIATION DECOMPOSITION")
    print("-" * 40)
    for var in all_vars:
        decomp = decompose_variation(df, entity_col, var)
        print(f"\n   {var}:")
        print(f"      Between share: {decomp['between_share']:.1%}")
        print(f"      Within share: {decomp['within_share']:.1%}")

    # Correlations
    print("\n4. CORRELATIONS WITH OUTCOME")
    print("-" * 40)
    for pred in predictors:
        corrs = within_correlation(df, entity_col, pred, outcome_var)
        print(f"\n   {pred} vs {outcome_var}:")
        print(f"      Total: {corrs['total']:.4f}")
        print(f"      Between: {corrs['between']:.4f}")
        print(f"      Within: {corrs['within']:.4f}")

    print("\n" + "=" * 60)

# Generate report
panel_eda_report(df, 'entity', 'time', 'y', ['x', 'z'])
```

## Key Takeaways

1. **Always visualize trajectories** - Plot individual entity paths to understand heterogeneity

2. **Decompose variation** - Understanding within vs. between variation guides model choice

3. **Test for effects** - Formal tests help justify fixed/random effects specification

4. **Check for violations** - Serial correlation and heteroskedasticity require robust standard errors

5. **Document missing patterns** - Missingness can bias estimates if not random


---

## Conceptual Practice Questions

**Practice Question 1:** What problem does this approach solve that simpler methods cannot?

**Practice Question 2:** What are the key assumptions, and how would you test them in practice?


---

## Cross-References

<a class="link-card" href="./01_ols_review.md">
  <div class="link-card-title">01 Ols Review</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_ols_review.md">
  <div class="link-card-title">01 Ols Review — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./01_panel_data_concepts.md">
  <div class="link-card-title">01 Panel Data Concepts</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_panel_data_concepts.md">
  <div class="link-card-title">01 Panel Data Concepts — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./02_data_structures.md">
  <div class="link-card-title">02 Data Structures</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_data_structures.md">
  <div class="link-card-title">02 Data Structures — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

