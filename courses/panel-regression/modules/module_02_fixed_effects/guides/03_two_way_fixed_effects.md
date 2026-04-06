# Two-Way Fixed Effects: Entity and Time

> **Reading time:** ~20 min | **Module:** 02 — Fixed Effects | **Prerequisites:** Module 1


## Introduction

<div class="flow">
<div class="flow-step mint">1. Compute Entity Means</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">2. Demean Variables</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">3. Run OLS on Demeaned</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. Adjust Standard Errors</div>
</div>


<div class="callout-key">

**Key Concept Summary:** Without time FE, these confound the X-Y relationship.

</div>

Two-way fixed effects (TWFE) controls for both:
- **Entity effects** ($\alpha_i$): Time-invariant entity characteristics
- **Time effects** ($\lambda_t$): Entity-invariant time shocks

$$y_{it} = \alpha_i + \lambda_t + X_{it}\beta + \epsilon_{it}$$

## Why Add Time Fixed Effects?

### Common Shocks

Many factors affect all entities simultaneously:
- Macroeconomic conditions
- Regulatory changes
- Market-wide trends
- Seasonal patterns

Without time FE, these confound the X-Y relationship.

### Example: Investment and Growth


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import matplotlib.pyplot as plt

# Simulate data with time shocks
np.random.seed(42)
n_firms = 100
n_years = 15

# Macroeconomic shocks (affect all firms)
time_shocks = np.random.normal(0, 2, n_years)  # e.g., recessions, booms

```

<div class="callout-insight">

**Insight:** Fixed effects are not a method -- they are a way of thinking about unobserved heterogeneity. The within-transformation eliminates time-invariant confounders, which is the single most important advantage of panel data.

</div>

```python
data = []
for i in range(n_firms):
    firm_effect = np.random.normal(0, 3)
    for t in range(n_years):
        # Investment responds to macro conditions
        investment = 10 + 0.5 * time_shocks[t] + np.random.normal(0, 1)

        # Growth depends on investment AND macro conditions
        growth = 2 + 0.8 * investment + 1.5 * time_shocks[t] + firm_effect + np.random.normal(0, 1)

        data.append({
            'firm': i, 'year': t + 2010,
            'investment': investment, 'growth': growth,
            'macro_shock': time_shocks[t]
        })

df = pd.DataFrame(data)
df_panel = df.set_index(['firm', 'year'])

# Compare specifications

# 1. Entity FE only
fe_entity = PanelOLS(df_panel['growth'], df_panel[['investment']],
                     entity_effects=True).fit()

# 2. Two-way FE
fe_twoway = PanelOLS(df_panel['growth'], df_panel[['investment']],
                      entity_effects=True, time_effects=True).fit()

print("Impact of time fixed effects:")
print(f"\nEntity FE only:")
print(f"  Investment coefficient: {fe_entity.params['investment']:.4f}")
print(f"  (Biased by common macro shocks)")

print(f"\nTwo-way FE:")
print(f"  Investment coefficient: {fe_twoway.params['investment']:.4f}")
print(f"  (True effect ≈ 0.80)")
```


</div>
</div>

## The Two-Way Transformation

<div class="callout-warning">

**Warning:** Fixed effects estimates identify only from within-entity variation. If your variable of interest has little within-entity variation (e.g., industry sector), fixed effects will produce large standard errors or fail entirely.

</div>


### Mathematical Form

TWFE uses double-demeaning:

$$\tilde{y}_{it} = y_{it} - \bar{y}_i - \bar{y}_t + \bar{\bar{y}}$$

where:
- $\bar{y}_i$ = entity mean
- $\bar{y}_t$ = time mean
- $\bar{\bar{y}}$ = grand mean

### Manual Implementation


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
def double_demean(df, entity_col, time_col, variables):
    """
    Apply two-way fixed effects transformation.
    """
    df_out = df.copy()

    for var in variables:
        # Entity means
        entity_mean = df.groupby(entity_col)[var].transform('mean')

        # Time means
        time_mean = df.groupby(time_col)[var].transform('mean')

        # Grand mean
        grand_mean = df[var].mean()

        # Double demean
        df_out[f'{var}_dd'] = df[var] - entity_mean - time_mean + grand_mean

    return df_out

# Apply transformation
df_transformed = double_demean(df, 'firm', 'year', ['growth', 'investment'])

# Estimate on transformed data
import statsmodels.formula.api as smf

manual_twfe = smf.ols('growth_dd ~ investment_dd - 1', data=df_transformed).fit()
print(f"\nManual two-way transformation:")
print(f"  Investment coefficient: {manual_twfe.params['investment_dd']:.4f}")
```


</div>
</div>

## Visualizing Time Effects


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
def plot_time_effects(model_results, year_range):
    """
    Plot estimated time effects.
    """
    # Extract time effects
    time_effects = model_results.estimated_effects

    # If using linearmodels, time effects need extraction
    # Here we'll compute them from residuals

    fig, ax = plt.subplots(figsize=(12, 5))

    # Time means of dependent variable (proxy for time effects)
    time_means = df.groupby('year')['growth'].mean()
    ax.plot(time_means.index, time_means.values, 'o-', linewidth=2, markersize=8)
    ax.axhline(time_means.mean(), color='red', linestyle='--', alpha=0.7)

    ax.set_xlabel('Year')
    ax.set_ylabel('Average Growth')
    ax.set_title('Time Effects: Average Growth by Year')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Also plot the macro shocks for comparison
    fig, ax = plt.subplots(figsize=(12, 5))
    years = range(2010, 2010 + n_years)
    ax.bar(years, time_shocks, alpha=0.7)
    ax.set_xlabel('Year')
    ax.set_ylabel('Macro Shock')
    ax.set_title('True Macro Shocks (Simulated)')
    ax.axhline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.show()

plot_time_effects(fe_twoway, range(2010, 2025))
```


</div>
</div>

## When to Use Two-Way FE

### Use TWFE When:

1. **Common time trends exist**
   - All entities affected by business cycles
   - Industry-wide shocks
   - Policy changes affecting all units

2. **X varies with aggregate conditions**
   - Investment correlates with economic climate
   - Hiring correlates with labor market conditions

3. **Time effects are nuisance parameters**
   - Interest is in X effect, not time pattern

### Don't Use TWFE When:

1. **Time variation is the interest**
   - Studying effect of a policy that varies only by time
   - Time dummies would absorb the effect

2. **Limited time periods**
   - Few periods = imprecise time effect estimates
   - May overfit

3. **Treatment varies only by time**
   - TWFE cannot identify effects of variables that only vary by time

## Testing for Time Effects

### F-Test for Time Effects

```python
from scipy import stats

def test_time_effects(df, entity_col, time_col, y_col, x_cols):
    """
    F-test for significance of time effects.
    """
    df_panel = df.set_index([entity_col, time_col])

    # Restricted model: entity FE only
    restricted = PanelOLS(df_panel[y_col], df_panel[x_cols],
                          entity_effects=True).fit()

    # Unrestricted model: two-way FE
    unrestricted = PanelOLS(df_panel[y_col], df_panel[x_cols],
                            entity_effects=True, time_effects=True).fit()

    # F-statistic
    n_time_effects = df[time_col].nunique() - 1
    n_obs = len(df)
    n_params_unrest = unrestricted.df_model

    ssr_r = restricted.resid_ss
    ssr_u = unrestricted.resid_ss

    f_stat = ((ssr_r - ssr_u) / n_time_effects) / (ssr_u / (n_obs - n_params_unrest))
    p_value = 1 - stats.f.cdf(f_stat, n_time_effects, n_obs - n_params_unrest)

    print("F-test for time effects:")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  Degrees of freedom: ({n_time_effects}, {n_obs - n_params_unrest})")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Conclusion: {'Significant time effects' if p_value < 0.05 else 'No significant time effects'}")

    return f_stat, p_value

test_time_effects(df, 'firm', 'year', 'growth', ['investment'])
```

## Alternative: Time Trends

Sometimes time fixed effects are too flexible. Consider:

### Linear Time Trend

$$y_{it} = \alpha_i + \gamma t + X_{it}\beta + \epsilon_{it}$$

```python

# Add linear time trend
df['trend'] = df['year'] - df['year'].min()
df_panel_trend = df.set_index(['firm', 'year'])

fe_trend = PanelOLS(
    df_panel_trend['growth'],
    df_panel_trend[['investment', 'trend']],
    entity_effects=True
).fit()

print("\nEntity FE + Linear Trend:")
print(f"  Investment: {fe_trend.params['investment']:.4f}")
print(f"  Trend: {fe_trend.params['trend']:.4f}")
```

### Entity-Specific Trends

$$y_{it} = \alpha_i + \gamma_i t + X_{it}\beta + \epsilon_{it}$$

```python

# Entity-specific trends (more demanding)
df['entity_trend'] = df['firm'].astype(str) + '_' + df['trend'].astype(str)

# Using LSDV for entity-specific trends
import statsmodels.formula.api as smf

fe_entity_trend = smf.ols(
    'growth ~ investment + C(firm) + C(firm):trend',
    data=df
).fit()

print(f"\nEntity FE + Entity-specific trends:")
print(f"  Investment: {fe_entity_trend.params['investment']:.4f}")
```

## TWFE with Staggered Treatment

**Warning**: TWFE has known problems with staggered treatment adoption (different units treated at different times). Recent econometrics literature shows TWFE can produce biased estimates when:

1. Treatment effects are heterogeneous
2. Treatment is staggered across units

```python
def illustrate_twfe_problem():
    """
    Demonstrate potential TWFE bias with staggered treatment.
    """
    print("\n" + "="*60)
    print("CAUTION: TWFE with Staggered Treatment")
    print("="*60)
    print("""
    When treatment occurs at different times for different units,
    standard TWFE can produce biased estimates if:

    1. Treatment effects vary across units
    2. Treatment effects change over time since treatment
    3. Never-treated units serve as controls for early-treated

    Recent solutions include:
    - Callaway & Sant'Anna (2021)
    - Sun & Abraham (2021)
    - Goodman-Bacon decomposition

    For details, see the advanced topics module.
    """)

illustrate_twfe_problem()
```

## Practical Implementation

<div class="callout-danger">

**Danger:** Never include a lagged dependent variable in a fixed effects model without using an appropriate estimator (e.g., Arellano-Bond GMM). The within-transformation creates mechanical correlation between the transformed lagged variable and the transformed error, biasing all coefficients.

</div>


### Recommended Specification

```python

# Full two-way fixed effects with clustered SEs
final_twfe = PanelOLS(
    df_panel['growth'],
    df_panel[['investment']],
    entity_effects=True,
    time_effects=True
).fit(cov_type='clustered', cluster_entity=True)

print("\nTwo-Way Fixed Effects (Recommended):")
print(final_twfe.summary.tables[1])
```

### Reporting Results

```python
from linearmodels.panel import compare

# Compare specifications
models = {
    'Pooled OLS': PanelOLS(df_panel['growth'], df_panel[['investment']]).fit(),
    'Entity FE': PanelOLS(df_panel['growth'], df_panel[['investment']],
                          entity_effects=True).fit(),
    'Two-Way FE': PanelOLS(df_panel['growth'], df_panel[['investment']],
                           entity_effects=True, time_effects=True).fit()
}

comparison = compare(models)
print("\nModel Comparison:")
print(comparison)
```

## Key Takeaways

1. **TWFE controls for common shocks** that affect all entities in a time period

2. **Use when X correlates with aggregate conditions** (business cycles, policy changes)

3. **Test whether time effects are significant** before including

4. **Consider simpler alternatives** (linear trend) when time FE overfits

5. **Be cautious with staggered treatment** - TWFE may be biased

6. **Always cluster standard errors** by entity (or two-way cluster)


---

## Conceptual Practice Questions

**Practice Question 1:** Why can fixed effects not estimate the impact of time-invariant variables like gender or geographic region?

**Practice Question 2:** When would entity fixed effects alone be insufficient, requiring two-way (entity + time) fixed effects?


---

## Cross-References

<a class="link-card" href="./01_fixed_effects_intuition.md">
  <div class="link-card-title">01 Fixed Effects Intuition</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_fixed_effects_intuition.md">
  <div class="link-card-title">01 Fixed Effects Intuition — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./02_lsdv_vs_within.md">
  <div class="link-card-title">02 Lsdv Vs Within</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_lsdv_vs_within.md">
  <div class="link-card-title">02 Lsdv Vs Within — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

