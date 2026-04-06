# Panel Data Structures in Python

> **Reading time:** ~20 min | **Module:** 00 — Foundations | **Prerequisites:** None (entry point)


## Overview


<div class="callout-key">

**Key Concept Summary:** This guide covers how to structure, manipulate, and prepare panel data in Python using pandas and specialized libraries. Proper data structure is essential before any panel regression analysis.

</div>

This guide covers how to structure, manipulate, and prepare panel data in Python using pandas and specialized libraries. Proper data structure is essential before any panel regression analysis.

## Understanding Panel Data Formats

### Long Format (Stacked)

The most common format for panel data analysis. Each row represents one observation of one entity at one time period.


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
import pandas as pd
import numpy as np

# Example: Long format panel
long_data = pd.DataFrame({
    'firm_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    'year': [2020, 2021, 2022, 2020, 2021, 2022, 2020, 2021, 2022],
    'revenue': [100, 110, 115, 200, 220, 210, 150, 160, 180],
    'employees': [10, 12, 11, 25, 28, 27, 15, 16, 18],
    'rd_spending': [5, 6, 7, 12, 15, 14, 8, 9, 10]
})

print("Long format:")
print(long_data)
```


</div>
</div>

Output:
```
   firm_id  year  revenue  employees  rd_spending
0        1  2020      100         10            5
1        1  2021      110         12            6
2        1  2022      115         11            7
3        2  2020      200         25           12
4        2  2021      220         28           15
5        2  2022      210         27           14
6        3  2020      150         15            8
7        3  2021      160         16            9
8        3  2022      180         18           10
```

### Wide Format (Unstacked)

Variables spread across columns by time period. Common in downloaded datasets but needs conversion for analysis.

<div class="callout-insight">

**Insight:** Panel data lets you control for unobservable differences between entities that are constant over time. This is the single most important reason to prefer panel data over repeated cross-sections.

</div>


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python

# Example: Wide format
wide_data = pd.DataFrame({
    'firm_id': [1, 2, 3],
    'revenue_2020': [100, 200, 150],
    'revenue_2021': [110, 220, 160],
    'revenue_2022': [115, 210, 180],
    'employees_2020': [10, 25, 15],
    'employees_2021': [12, 28, 16],
    'employees_2022': [11, 27, 18]
})

print("Wide format:")
print(wide_data)
```


</div>
</div>

### Converting Between Formats

```python
def wide_to_long(df, id_vars, value_vars_pattern, var_name='variable', value_name='value'):
    """
    Convert wide format to long format.

    Parameters:
    -----------
    df : DataFrame
        Wide format data
    id_vars : list
        Columns to keep as identifiers
    value_vars_pattern : str
        Pattern for variable columns (e.g., 'revenue_')
    """
    # Identify value columns
    value_cols = [c for c in df.columns if value_vars_pattern in c]

    # Melt the dataframe
    melted = pd.melt(
        df,
        id_vars=id_vars,
        value_vars=value_cols,
        var_name='temp_var',
        value_name=value_name
    )

    # Extract year and variable name
    melted['year'] = melted['temp_var'].str.extract(r'(\d{4})').astype(int)
    melted['variable'] = melted['temp_var'].str.replace(r'_\d{4}', '', regex=True)

    # Pivot to get one column per variable
    result = melted.pivot_table(
        index=id_vars + ['year'],
        columns='variable',
        values=value_name
    ).reset_index()

    result.columns.name = None
    return result

# Convert our wide data
long_converted = pd.wide_to_long(
    wide_data,
    stubnames=['revenue', 'employees'],
    i='firm_id',
    j='year',
    sep='_'
).reset_index()

print("Converted to long format:")
print(long_converted)
```

## Setting Up Panel Structure with MultiIndex

<div class="callout-warning">

**Warning:** Reporting results without appropriate standard errors is a common mistake. In panel data, conventional OLS standard errors are almost always wrong -- use clustered or heteroskedasticity-robust standard errors.

</div>


### Creating a Proper Panel Index


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python

# Set MultiIndex for panel structure
panel_data = long_data.copy()
panel_data = panel_data.set_index(['firm_id', 'year'])

print("Panel with MultiIndex:")
print(panel_data)
print(f"\nIndex names: {panel_data.index.names}")
print(f"Index levels: {panel_data.index.nlevels}")
```


</div>
</div>

### Accessing Data in Panel Structure

```python

# Access specific entity
print("Firm 1 data:")
print(panel_data.loc[1])

# Access specific time period across all entities
print("\nYear 2021 data:")
print(panel_data.xs(2021, level='year'))

# Access specific observation
print("\nFirm 2, Year 2021:")
print(panel_data.loc[(2, 2021)])
```

### Iterating Over Entities

```python

# Group by entity
for firm_id, group in panel_data.groupby(level='firm_id'):
    print(f"\nFirm {firm_id}:")
    print(f"  Revenue range: {group['revenue'].min()} - {group['revenue'].max()}")
    print(f"  Average employees: {group['employees'].mean():.1f}")
```

## Checking Panel Balance

### Balanced vs Unbalanced Panels

```python
def check_panel_balance(df, entity_col, time_col):
    """
    Check if panel is balanced and provide diagnostics.
    """
    # Count observations per entity
    obs_per_entity = df.groupby(entity_col).size()

    # Count observations per time period
    obs_per_time = df.groupby(time_col).size()

    # Expected counts for balanced panel
    n_entities = df[entity_col].nunique()
    n_periods = df[time_col].nunique()
    expected_obs = n_entities * n_periods
    actual_obs = len(df)

    is_balanced = actual_obs == expected_obs

    result = {
        'is_balanced': is_balanced,
        'n_entities': n_entities,
        'n_periods': n_periods,
        'expected_observations': expected_obs,
        'actual_observations': actual_obs,
        'missing_observations': expected_obs - actual_obs,
        'obs_per_entity': obs_per_entity.describe(),
        'obs_per_time': obs_per_time.describe()
    }

    if not is_balanced:
        # Find entities with incomplete data
        incomplete_entities = obs_per_entity[obs_per_entity < n_periods].index.tolist()
        result['incomplete_entities'] = incomplete_entities

        # Find periods with incomplete data
        incomplete_periods = obs_per_time[obs_per_time < n_entities].index.tolist()
        result['incomplete_periods'] = incomplete_periods

    return result

# Check our data
balance_check = check_panel_balance(long_data, 'firm_id', 'year')
print(f"Panel is balanced: {balance_check['is_balanced']}")
print(f"Entities: {balance_check['n_entities']}")
print(f"Periods: {balance_check['n_periods']}")
```

### Creating Unbalanced Panel for Testing

```python

# Create unbalanced panel (remove some observations)
unbalanced_data = long_data.drop([2, 7])  # Remove firm 1's 2022 and firm 3's 2021

print("Unbalanced panel:")
print(unbalanced_data)

balance_check = check_panel_balance(unbalanced_data, 'firm_id', 'year')
print(f"\nPanel is balanced: {balance_check['is_balanced']}")
print(f"Missing observations: {balance_check['missing_observations']}")
print(f"Incomplete entities: {balance_check.get('incomplete_entities', 'None')}")
```

### Balancing an Unbalanced Panel

```python
def balance_panel(df, entity_col, time_col, method='dropna'):
    """
    Balance a panel dataset.

    Parameters:
    -----------
    method : str
        'dropna': Keep only entities with complete data
        'fillna': Fill missing with NaN (creates rectangular data)
        'interpolate': Interpolate missing values
    """
    if method == 'dropna':
        # Find entities with complete data
        n_periods = df[time_col].nunique()
        complete_entities = df.groupby(entity_col).size()
        complete_entities = complete_entities[complete_entities == n_periods].index
        return df[df[entity_col].isin(complete_entities)]

    elif method == 'fillna':
        # Create complete index
        entities = df[entity_col].unique()
        periods = df[time_col].unique()
        complete_index = pd.MultiIndex.from_product(
            [entities, periods],
            names=[entity_col, time_col]
        )

        # Reindex to create rectangular data
        df_indexed = df.set_index([entity_col, time_col])
        return df_indexed.reindex(complete_index).reset_index()

    elif method == 'interpolate':
        # First create rectangular data, then interpolate
        df_rect = balance_panel(df, entity_col, time_col, method='fillna')
        df_rect = df_rect.set_index([entity_col, time_col])

        # Interpolate within each entity
        numeric_cols = df_rect.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_rect[col] = df_rect.groupby(level=entity_col)[col].transform(
                lambda x: x.interpolate(method='linear')
            )

        return df_rect.reset_index()

# Balance the unbalanced data
balanced = balance_panel(unbalanced_data, 'firm_id', 'year', method='dropna')
print("Balanced (dropna method):")
print(balanced)
```

## Computing Panel Statistics

### Within and Between Variation

```python
def decompose_variation(df, entity_col, variable):
    """
    Decompose variation into within and between components.

    Returns:
    --------
    dict with total, between, within variation and percentages
    """
    # Total variation
    total_var = df[variable].var()

    # Between variation (variance of entity means)
    entity_means = df.groupby(entity_col)[variable].transform('mean')
    between_var = entity_means.var()

    # Within variation (variance around entity means)
    within_var = (df[variable] - entity_means).var()

    return {
        'total_variance': total_var,
        'between_variance': between_var,
        'within_variance': within_var,
        'between_share': between_var / total_var,
        'within_share': within_var / total_var,
        # Note: between + within ≈ total (not exact due to degrees of freedom)
    }

# Decompose variation for revenue
variation = decompose_variation(long_data, 'firm_id', 'revenue')
print("Revenue variation decomposition:")
for key, value in variation.items():
    if 'share' in key:
        print(f"  {key}: {value:.1%}")
    else:
        print(f"  {key}: {value:.2f}")
```

### Summary Statistics by Entity and Time

```python
def panel_summary(df, entity_col, time_col, value_cols):
    """
    Generate comprehensive panel summary statistics.
    """
    summary = {}

    for col in value_cols:
        col_stats = {
            'overall': {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            },
            'between': {
                'mean': df.groupby(entity_col)[col].mean().mean(),
                'std': df.groupby(entity_col)[col].mean().std(),
            },
            'within': {
                'std': (df[col] - df.groupby(entity_col)[col].transform('mean')).std()
            }
        }
        summary[col] = col_stats

    return summary

# Get summary
summary = panel_summary(long_data, 'firm_id', 'year', ['revenue', 'employees', 'rd_spending'])

for var, stats in summary.items():
    print(f"\n{var.upper()}:")
    print(f"  Overall mean: {stats['overall']['mean']:.2f}")
    print(f"  Between-entity std: {stats['between']['std']:.2f}")
    print(f"  Within-entity std: {stats['within']['std']:.2f}")
```

## Using linearmodels PanelData

The `linearmodels` library provides specialized panel data handling:

```python
from linearmodels.panel import PanelData

# Create PanelData object
panel = PanelData(panel_data)

print("PanelData summary:")
print(f"  Entities: {panel.nentity}")
print(f"  Time periods: {panel.nobs / panel.nentity:.0f}")
print(f"  Total observations: {panel.nobs}")
print(f"  Variables: {list(panel.vars)}")
print(f"  Is balanced: {panel.balanced}")
```

### Accessing Panel Properties

```python

# Entity and time information
print("Entities:", panel.entities[:5])  # First 5 entities
print("Time periods:", panel.time[:3])  # First 3 time periods

# Get data in different formats
print("\nAs numpy array shape:", panel.values2d.shape)
print("As DataFrame:\n", panel.dataframe.head())
```

## Practical Example: Financial Panel Data

<div class="callout-danger">

**Danger:** Never include a lagged dependent variable in a fixed effects model without using an appropriate estimator (e.g., Arellano-Bond GMM). The within-transformation creates mechanical correlation between the transformed lagged variable and the transformed error, biasing all coefficients.

</div>


```python

# Simulate realistic financial panel data
np.random.seed(42)

n_firms = 50
n_years = 10
years = range(2014, 2024)

# Generate firm characteristics (time-invariant)
firm_effects = np.random.normal(0, 1, n_firms)

# Generate panel data
data = []
for i in range(n_firms):
    for t, year in enumerate(years):
        # Revenue grows with firm effect and time trend
        revenue = 100 + 50*firm_effects[i] + 5*t + np.random.normal(0, 10)

        # Investment depends on lagged revenue (simplified)
        investment = 10 + 0.1*revenue + np.random.normal(0, 5)

        # ROA depends on firm effect and investment
        roa = 5 + 2*firm_effects[i] + 0.05*investment + np.random.normal(0, 2)

        data.append({
            'firm_id': i + 1,
            'year': year,
            'revenue': max(revenue, 10),  # Floor at 10
            'investment': max(investment, 0),
            'roa': roa,
            'industry': np.random.choice(['Tech', 'Finance', 'Manufacturing'])
        })

financial_panel = pd.DataFrame(data)

# Set up proper panel structure
financial_panel = financial_panel.set_index(['firm_id', 'year'])

print("Financial panel data:")
print(financial_panel.head(10))

# Summary statistics
print("\nDescriptive statistics:")
print(financial_panel.describe())
```

## Key Takeaways

1. **Long format is standard** for panel regression - one row per entity-time observation

2. **MultiIndex structure** in pandas enables efficient panel operations

3. **Check balance** before analysis - unbalanced panels need special handling

4. **Variation decomposition** (within vs. between) is fundamental to understanding what panel methods estimate

5. **linearmodels.PanelData** provides specialized functionality for panel analysis


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

