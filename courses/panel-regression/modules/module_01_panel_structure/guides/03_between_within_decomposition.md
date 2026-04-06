# Between and Within Variation: The Heart of Panel Data

> **Reading time:** ~18 min | **Module:** 01 — Panel Structure | **Prerequisites:** Module 0 Foundations


## Introduction


<div class="callout-key">

**Key Concept Summary:** Understanding this decomposition is fundamental to choosing the right estimator.

</div>

Panel data contains two sources of variation:
- **Between variation**: Differences across entities (cross-sectional)
- **Within variation**: Changes over time for each entity (longitudinal)

Understanding this decomposition is fundamental to choosing the right estimator.

## Mathematical Decomposition

For any variable $X_{it}$:

$$X_{it} = \bar{X} + (\bar{X}_i - \bar{X}) + (X_{it} - \bar{X}_i)$$

Where:
- $\bar{X}$ = grand mean
- $(\bar{X}_i - \bar{X})$ = between component
- $(X_{it} - \bar{X}_i)$ = within component

### Variance Decomposition

$$Var(X) = Var_{between}(\bar{X}_i) + Var_{within}(X_{it} - \bar{X}_i)$$


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def variance_decomposition(df, entity_col, variable):
    """
    Decompose total variance into between and within components.
    """
    # Grand mean
    grand_mean = df[variable].mean()

    # Entity means
    entity_means = df.groupby(entity_col)[variable].transform('mean')

    # Between component: entity mean - grand mean
    between = entity_means - grand_mean

    # Within component: observation - entity mean
    within = df[variable] - entity_means

    # Variances
    var_total = df[variable].var()
    var_between = between.var()
    var_within = within.var()

```

<div class="callout-insight">

**Insight:** Panel data lets you control for unobservable differences between entities that are constant over time. This is the single most important reason to prefer panel data over repeated cross-sections.

</div>

```python
    return {
        'total': var_total,
        'between': var_between,
        'within': var_within,
        'between_pct': var_between / var_total * 100,
        'within_pct': var_within / var_total * 100
    }

# Example with simulated data
np.random.seed(42)
n_firms = 50
n_years = 10

data = []
for i in range(n_firms):
    # Firm-specific level (creates between variation)
    firm_size = np.random.lognormal(3, 0.5)
    firm_profitability = np.random.uniform(0.05, 0.20)

    for t in range(n_years):
        # Time variation within firm (creates within variation)
        size_shock = np.random.normal(0, 0.1) * firm_size
        profit_shock = np.random.normal(0, 0.03)

        data.append({
            'firm': i,
            'year': 2010 + t,
            'size': firm_size + size_shock + t * 0.05 * firm_size,  # Growth trend
            'profitability': firm_profitability + profit_shock
        })

df = pd.DataFrame(data)

# Decompose both variables
for var in ['size', 'profitability']:
    decomp = variance_decomposition(df, 'firm', var)
    print(f"\nVariance Decomposition for {var}:")
    print(f"  Total Variance:   {decomp['total']:.4f}")
    print(f"  Between Variance: {decomp['between']:.4f} ({decomp['between_pct']:.1f}%)")
    print(f"  Within Variance:  {decomp['within']:.4f} ({decomp['within_pct']:.1f}%)")
```


</div>
</div>

## Visualizing the Decomposition

<div class="callout-warning">

**Warning:** Reporting results without appropriate standard errors is a common mistake. In panel data, conventional OLS standard errors are almost always wrong -- use clustered or heteroskedasticity-robust standard errors.

</div>


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
def plot_variance_decomposition(df, entity_col, variable, n_sample=10):
    """
    Visualize between and within variation.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Sample entities for clarity
    sample_entities = df[entity_col].unique()[:n_sample]
    df_sample = df[df[entity_col].isin(sample_entities)]

    # 1. Spaghetti plot (shows both variations)
    ax1 = axes[0, 0]
    for entity in sample_entities:
        entity_data = df_sample[df_sample[entity_col] == entity]
        ax1.plot(entity_data['year'], entity_data[variable], 'o-', alpha=0.7)

    ax1.axhline(df[variable].mean(), color='red', linestyle='--',
                linewidth=2, label='Grand Mean')
    ax1.set_xlabel('Year')
    ax1.set_ylabel(variable)
    ax1.set_title('Raw Data: Between + Within Variation')
    ax1.legend()

    # 2. Entity means (between variation only)
    ax2 = axes[0, 1]
    entity_means = df.groupby(entity_col)[variable].mean().sort_values()
    ax2.barh(range(len(entity_means)), entity_means.values, alpha=0.7)
    ax2.axvline(df[variable].mean(), color='red', linestyle='--',
                linewidth=2, label='Grand Mean')
    ax2.set_xlabel(f'Mean {variable}')
    ax2.set_ylabel('Entity (sorted)')
    ax2.set_title('Between Variation: Entity Means')
    ax2.legend()

    # 3. Demeaned data (within variation only)
    ax3 = axes[1, 0]
    df_sample['demeaned'] = df_sample[variable] - \
                            df_sample.groupby(entity_col)[variable].transform('mean')

    for entity in sample_entities:
        entity_data = df_sample[df_sample[entity_col] == entity]
        ax3.plot(entity_data['year'], entity_data['demeaned'], 'o-', alpha=0.7)

    ax3.axhline(0, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Year')
    ax3.set_ylabel(f'Demeaned {variable}')
    ax3.set_title('Within Variation Only (Entity Mean Removed)')

    # 4. Variance pie chart
    ax4 = axes[1, 1]
    decomp = variance_decomposition(df, entity_col, variable)
    sizes = [decomp['between_pct'], decomp['within_pct']]
    labels = [f"Between\n({decomp['between_pct']:.1f}%)",
              f"Within\n({decomp['within_pct']:.1f}%)"]
    colors = ['steelblue', 'coral']
    ax4.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90)
    ax4.set_title('Variance Decomposition')

    plt.tight_layout()
    plt.show()

plot_variance_decomposition(df, 'firm', 'size')
plot_variance_decomposition(df, 'firm', 'profitability')
```


</div>
</div>

## Implications for Estimation

### Fixed Effects Uses Only Within Variation

$$\tilde{y}_{it} = y_{it} - \bar{y}_i$$

FE uses only within-entity changes over time.


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
from linearmodels.panel import PanelOLS

def illustrate_fe_within(df, y_col, x_col, entity_col, time_col):
    """
    Show that FE uses only within variation.
    """
    # Set up panel
    df_panel = df.set_index([entity_col, time_col])

    # Fixed effects
    fe = PanelOLS(df_panel[y_col], df_panel[[x_col]], entity_effects=True).fit()

    # Manual within estimation
    df['y_demean'] = df[y_col] - df.groupby(entity_col)[y_col].transform('mean')
    df['x_demean'] = df[x_col] - df.groupby(entity_col)[x_col].transform('mean')

    import statsmodels.formula.api as smf
    within_manual = smf.ols('y_demean ~ x_demean - 1', data=df).fit()

    # Between estimation (entity means)
    entity_means = df.groupby(entity_col)[[y_col, x_col]].mean().reset_index()
    between = smf.ols(f'{y_col} ~ {x_col}', data=entity_means).fit()

    print("Estimator Comparison:")
    print("=" * 50)
    print(f"Fixed Effects (linearmodels):  {fe.params[x_col]:.4f}")
    print(f"Within (manual demeaning):     {within_manual.params['x_demean']:.4f}")
    print(f"Between (entity means only):   {between.params[x_col]:.4f}")

    return fe.params[x_col], between.params[x_col]

# Add a relationship between size and profitability
df['profit_adj'] = 0.02 + 0.001 * df['size'] + np.random.normal(0, 0.02, len(df))

fe_coef, between_coef = illustrate_fe_within(df, 'profit_adj', 'size', 'firm', 'year')
```


</div>
</div>

### When Between and Within Effects Differ

If the FE and Between estimates differ substantially, there may be:
1. Unobserved confounders correlated with X
2. Measurement error (attenuated within estimates)
3. Different underlying relationships

```python
def compare_between_within(df, y_col, x_col, entity_col, time_col):
    """
    Compare between and within estimators with visualization.
    """
    # Entity means
    entity_means = df.groupby(entity_col)[[y_col, x_col]].mean()

    # Within-transformed data
    df['y_within'] = df[y_col] - df.groupby(entity_col)[y_col].transform('mean')
    df['x_within'] = df[x_col] - df.groupby(entity_col)[x_col].transform('mean')

    # Estimate
    import statsmodels.formula.api as smf

    between = smf.ols(f'{y_col} ~ {x_col}', data=entity_means).fit()
    within = smf.ols('y_within ~ x_within - 1', data=df).fit()

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Between relationship
    ax1 = axes[0]
    ax1.scatter(entity_means[x_col], entity_means[y_col], alpha=0.6)
    x_range = np.linspace(entity_means[x_col].min(), entity_means[x_col].max(), 100)
    ax1.plot(x_range, between.params['Intercept'] + between.params[x_col] * x_range,
             'r-', linewidth=2, label=f'β = {between.params[x_col]:.4f}')
    ax1.set_xlabel(f'Mean {x_col}')
    ax1.set_ylabel(f'Mean {y_col}')
    ax1.set_title('Between Relationship (Entity Means)')
    ax1.legend()

    # Within relationship
    ax2 = axes[1]
    ax2.scatter(df['x_within'], df['y_within'], alpha=0.1, s=10)
    x_range_w = np.linspace(df['x_within'].min(), df['x_within'].max(), 100)
    ax2.plot(x_range_w, within.params['x_within'] * x_range_w,
             'r-', linewidth=2, label=f'β = {within.params["x_within"]:.4f}')
    ax2.set_xlabel(f'Demeaned {x_col}')
    ax2.set_ylabel(f'Demeaned {y_col}')
    ax2.set_title('Within Relationship (Deviations from Mean)')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    print(f"\nBetween estimate: {between.params[x_col]:.4f}")
    print(f"Within estimate:  {within.params['x_within']:.4f}")
    print(f"Difference:       {between.params[x_col] - within.params['x_within']:.4f}")

compare_between_within(df, 'profit_adj', 'size', 'firm', 'year')
```

## Practical Applications

### Choosing Estimator Based on Variation

```python
def recommend_estimator(df, entity_col, y_col, x_cols):
    """
    Recommend estimator based on variance structure.
    """
    print("=" * 60)
    print("ESTIMATOR RECOMMENDATION BASED ON VARIANCE STRUCTURE")
    print("=" * 60)

    recommendations = []

    for var in [y_col] + x_cols:
        decomp = variance_decomposition(df, entity_col, var)
        print(f"\n{var}:")
        print(f"  Between: {decomp['between_pct']:.1f}%")
        print(f"  Within:  {decomp['within_pct']:.1f}%")

        if decomp['within_pct'] < 10:
            print(f"  ⚠️ WARNING: Very low within variation!")
            recommendations.append('low_within')
        elif decomp['between_pct'] < 10:
            print(f"  Note: Mostly within variation")
            recommendations.append('high_within')
        else:
            recommendations.append('balanced')

    print("\n" + "-" * 60)
    print("RECOMMENDATION:")

    if 'low_within' in recommendations:
        print("  → Fixed Effects may be problematic (insufficient within variation)")
        print("  → Consider Random Effects or Between estimator")
        print("  → Check if time-invariant variables are important")
    else:
        print("  → Fixed Effects is appropriate")
        print("  → Sufficient within variation for identification")

    print("=" * 60)

recommend_estimator(df, 'firm', 'profit_adj', ['size'])
```

### Measurement Error Implications

Within variation is more susceptible to measurement error attenuation:

```python
def simulate_measurement_error(n_sim=100):
    """
    Show how measurement error affects between vs within estimates.
    """
    true_beta = 0.5
    results = {'between': [], 'within': [], 'noise_ratio': []}

    for noise_level in np.linspace(0, 0.5, 10):
        for _ in range(n_sim):
            # Generate data
            data = []
            for i in range(50):
                x_true_i = np.random.normal(10, 3)  # Entity-level true X
                for t in range(10):
                    x_true_it = x_true_i + np.random.normal(0, 0.5)  # Within variation
                    x_measured = x_true_it + np.random.normal(0, noise_level * 3)  # Measurement error
                    y = 2 + true_beta * x_true_it + np.random.normal(0, 0.5)
                    data.append({'entity': i, 'time': t, 'x': x_measured, 'y': y})

            df_sim = pd.DataFrame(data)

            # Between estimate
            means = df_sim.groupby('entity')[['x', 'y']].mean()
            import statsmodels.formula.api as smf
            between = smf.ols('y ~ x', data=means).fit()

            # Within estimate
            df_sim['y_w'] = df_sim['y'] - df_sim.groupby('entity')['y'].transform('mean')
            df_sim['x_w'] = df_sim['x'] - df_sim.groupby('entity')['x'].transform('mean')
            within = smf.ols('y_w ~ x_w - 1', data=df_sim).fit()

            results['between'].append(between.params['x'])
            results['within'].append(within.params['x_w'])
            results['noise_ratio'].append(noise_level)

    # Plot
    results_df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(10, 6))

    for noise in results_df['noise_ratio'].unique():
        subset = results_df[results_df['noise_ratio'] == noise]
        ax.scatter([noise - 0.01] * len(subset), subset['between'],
                   alpha=0.3, c='blue', s=10)
        ax.scatter([noise + 0.01] * len(subset), subset['within'],
                   alpha=0.3, c='red', s=10)

    # Means
    mean_between = results_df.groupby('noise_ratio')['between'].mean()
    mean_within = results_df.groupby('noise_ratio')['within'].mean()

    ax.plot(mean_between.index, mean_between.values, 'b-', linewidth=2, label='Between')
    ax.plot(mean_within.index, mean_within.values, 'r-', linewidth=2, label='Within')
    ax.axhline(true_beta, color='green', linestyle='--', linewidth=2, label=f'True β = {true_beta}')

    ax.set_xlabel('Noise Level (Measurement Error)')
    ax.set_ylabel('Estimated β')
    ax.set_title('Measurement Error: Between vs Within Estimates')
    ax.legend()

    plt.tight_layout()
    plt.show()

simulate_measurement_error()
```

## Summary Table

<div class="callout-danger">

**Danger:** Never include a lagged dependent variable in a fixed effects model without using an appropriate estimator (e.g., Arellano-Bond GMM). The within-transformation creates mechanical correlation between the transformed lagged variable and the transformed error, biasing all coefficients.

</div>


| Aspect | Between Variation | Within Variation |
|--------|------------------|------------------|
| Source | Cross-sectional differences | Longitudinal changes |
| Captures | Permanent entity differences | Time-varying changes |
| FE uses | ✗ Eliminated | ✓ All |
| RE uses | ✓ Weighted | ✓ Weighted |
| Sensitivity to measurement error | Lower | Higher |
| Time-invariant effects | Identifiable | Not identifiable |

## Key Takeaways

1. **Panel data = Between + Within** variation - understand both before modeling

2. **Fixed Effects uses only within variation** - requires sufficient time variation for identification

3. **Between and within estimates can differ** due to confounders, measurement error, or heterogeneous effects

4. **Low within variation** is a red flag for FE - may need alternative approaches

5. **Visualize the decomposition** to understand your data structure before estimation


---

## Conceptual Practice Questions

**Practice Question 1:** What problem does this approach solve that simpler methods cannot?

**Practice Question 2:** What are the key assumptions, and how would you test them in practice?


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

<a class="link-card" href="./02_pooled_ols_limitations.md">
  <div class="link-card-title">02 Pooled Ols Limitations</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_pooled_ols_limitations.md">
  <div class="link-card-title">02 Pooled Ols Limitations — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

