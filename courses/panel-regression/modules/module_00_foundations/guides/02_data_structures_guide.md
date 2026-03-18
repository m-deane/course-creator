# Panel Data Structures in Python

## In Brief

Panel data can be stored in two primary formats — long and wide — but regression libraries require long format with a proper MultiIndex. Mastering format conversion, the `pandas` MultiIndex, and `linearmodels` `PanelData` wrapper is the prerequisite for all panel estimation. Getting the structure wrong produces silent errors that corrupt your results.

## Key Insight

Long format stores one row per entity-time observation. The entity and time identifiers together form the unique key, analogous to a primary key in a relational database. This structure is what `linearmodels`, `statsmodels`, and `plm` all expect. Wide format stores one row per entity with time periods as columns — it is useful for visualization and entity-level operations, but must be converted before regression.

**The rule:** store in long format, convert to wide temporarily when needed.

## Formal Definition

A balanced panel with $N$ entities and $T$ periods has $N \times T$ observations. In long format:

$$\text{Long format shape} = (N \times T) \times (2 + K)$$

where $2$ columns identify the entity and time and $K$ columns contain variables.

In wide format:

$$\text{Wide format shape} = N \times (1 + K \times T)$$

where $1$ column identifies the entity and $K \times T$ columns contain the variable-period combinations.

**MultiIndex structure required by `linearmodels`:**

```
DataFrame with MultiIndex
  Level 0: entity identifier (e.g., firm_id)
  Level 1: time identifier (e.g., year)
  Columns: dependent variable + regressors
```

The `linearmodels` package validates this structure when a `PanelOLS` or `RandomEffects` model is instantiated.

## Intuitive Explanation

Think of panel data as a cube with three axes: entities, time periods, and variables. Long format "stacks" the time slices — each time period for each entity gets its own row. Wide format "unfolds" the time axis into columns.

**Why long format for regression?** Regression algorithms process observations row by row. Each row needs to be a complete observation with an outcome and predictor values. In long format, each row is exactly that: the outcome and predictors for entity $i$ at time $t$. Wide format cannot represent this cleanly because a single entity's row would need to hold all time periods simultaneously.

**The `pandas` MultiIndex** is a two-level row index where the first level is the entity and the second is the time period. Think of it as a two-column primary key. Setting `df.set_index(['entity', 'time'])` tells `pandas` and `linearmodels` that rows should be interpreted as entity-time pairs, not independent observations.

## Code Implementation

```python
import pandas as pd
import numpy as np
from linearmodels.panel import PanelData

# ---- Construct a balanced panel in long format ----
np.random.seed(42)
firms = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
years = list(range(2015, 2023))  # 8 years

long_data = pd.DataFrame({
    'firm_id': np.repeat(firms, len(years)),
    'year':    np.tile(years, len(firms)),
    'revenue': np.random.lognormal(mean=5, sigma=0.3, size=len(firms) * len(years)),
    'employees': np.random.randint(10000, 100000, size=len(firms) * len(years)),
    'rd_spend': np.random.lognormal(mean=3, sigma=0.4, size=len(firms) * len(years))
})

print(f"Long format shape: {long_data.shape}")
print(f"Expected: ({len(firms) * len(years)}, {long_data.shape[1]})")
print(long_data.head())
```

**Wide to long conversion:**

```python
# Simulate receiving data in wide format (common from Bloomberg, World Bank, etc.)
wide_data = long_data.pivot(index='firm_id', columns='year', values='revenue')
wide_data.columns = [f'rev_{yr}' for yr in wide_data.columns]
wide_data = wide_data.reset_index()

print(f"Wide format shape: {wide_data.shape}")

# Convert back to long using pd.wide_to_long
long_converted = pd.wide_to_long(
    wide_data,
    stubnames=['rev'],
    i='firm_id',
    j='year',
    sep='_'
).reset_index()

long_converted = long_converted.rename(columns={'rev': 'revenue'})
print(f"Converted back to long: {long_converted.shape}")
```

**MultiIndex setup — the critical step:**

```python
# Sort first to ensure lag operations work correctly
long_data = long_data.sort_values(['firm_id', 'year']).reset_index(drop=True)

# Set the panel MultiIndex: entity first, time second
panel = long_data.set_index(['firm_id', 'year'])

print(f"Index names: {panel.index.names}")
print(f"Index levels: {panel.index.nlevels}")

# Access patterns
print("\n--- Access patterns ---")
print("Entity AAPL (all years):")
print(panel.loc['AAPL'])

print("\nYear 2020 (all firms):")
print(panel.xs(2020, level='year'))

print("\nSpecific observation (MSFT, 2019):")
print(panel.loc[('MSFT', 2019)])
```

**PanelData wrapper and diagnostics:**

```python
# linearmodels requires PanelData or a MultiIndex DataFrame
pdata = PanelData(panel)

print(f"Entities: {pdata.nentity}")
print(f"Time periods: {pdata.nobs // pdata.nentity}")
print(f"Total observations: {pdata.nobs}")
print(f"Balanced: {pdata.balanced}")
```

**Balance check with the full diagnostic:**

```python
def check_panel_balance(df, entity_col, time_col):
    """
    Comprehensive panel balance diagnostic.
    Returns dict with balance status and observation counts.
    """
    n_entities = df[entity_col].nunique()
    n_periods = df[time_col].nunique()
    expected_obs = n_entities * n_periods
    actual_obs = len(df)
    obs_per_entity = df.groupby(entity_col).size()
    obs_per_time = df.groupby(time_col).size()

    # Identify missing entity-time combinations
    complete_idx = pd.MultiIndex.from_product(
        [df[entity_col].unique(), df[time_col].unique()],
        names=[entity_col, time_col]
    )
    actual_idx = df.set_index([entity_col, time_col]).index
    missing_combos = complete_idx.difference(actual_idx)

    return {
        'is_balanced': actual_obs == expected_obs,
        'n_entities': n_entities,
        'n_periods': n_periods,
        'completeness': actual_obs / expected_obs,
        'min_obs_per_entity': obs_per_entity.min(),
        'max_obs_per_entity': obs_per_entity.max(),
        'n_missing_combinations': len(missing_combos),
        'missing_combinations': missing_combos
    }

result = check_panel_balance(long_data, 'firm_id', 'year')
print(f"Balanced: {result['is_balanced']}")
print(f"Completeness: {result['completeness']:.1%}")
```

**Creating lags and leads safely:**

```python
def add_panel_lags(df, entity_col, time_col, var_col, lags=1):
    """
    Add lagged values to a panel DataFrame, correctly handling
    entity boundaries so lag(t=1) for entity A does not
    accidentally pick up the last observation of entity B.
    """
    df = df.sort_values([entity_col, time_col]).copy()
    for lag in range(1, lags + 1):
        df[f'{var_col}_lag{lag}'] = df.groupby(entity_col)[var_col].shift(lag)
    return df

long_data = add_panel_lags(long_data, 'firm_id', 'year', 'revenue', lags=2)
print(long_data[['firm_id', 'year', 'revenue', 'revenue_lag1', 'revenue_lag2']].head(10))
```

## Common Pitfalls

**Running regressions on wide format data.** `linearmodels` raises `ValueError: MultiIndex required` if you pass wide-format data. Always call `.set_index([entity, time])` before passing to any panel estimator.

**Duplicate entity-time pairs.** If your data has duplicate (entity, time) combinations — from a bad merge or duplicate rows — `set_index` will not raise an error, but your regression results will be wrong. Always check:

```python
n_dupes = df.duplicated(subset=['firm_id', 'year']).sum()
assert n_dupes == 0, f"Found {n_dupes} duplicate entity-time pairs"
```

**Unsorted data before lags.** `groupby().shift()` shifts positions within each group but does not sort. If rows are not in time order within each entity, lags will be computed incorrectly with no warning.

**Integer column names after pivot.** When converting from long to wide with `pivot()`, year columns become integers. You cannot access them with dot notation (`df.2020` is a syntax error). Use `df[2020]` or rename columns to strings.

**Implicit panel assumption violated.** Regression assumes the same entity-time grid for all variables. If you have missing values in some variables but not others, the effective sample size differs by specification. Use `.dropna()` after constructing all variables and before setting the index.

**Memory inefficiency with sparse unbalanced panels.** Wide format stores `NaN` for every missing observation, wasting memory. A panel of 10,000 firms with 50% missingness in wide format uses double the memory of long format.

## Connections

**Builds on:**
- `pandas` DataFrame operations: `groupby`, `pivot`, `melt`, `merge`
- Relational database concepts: primary keys, joins, tidy data principles
- Time series indexing in `pandas`

**Leads to:**
- Entity demeaning (the within transformation) using `groupby().transform('mean')`
- First-differencing (requires sorted lags within entities)
- `linearmodels` `PanelOLS`, `RandomEffects`, `PooledOLS` — all require the MultiIndex

**Related to:**
- `xarray` for multi-dimensional panel data with more than two dimensions
- `pymc` panel models where the same MultiIndex structure is needed
- R's `plm` package which uses a `pdata.frame` with the same entity-time structure

## Practice Problems

1. **Reshape and verify.** Download the World Bank GDP per capita data for 50 countries over 20 years (wide format, one column per year). Convert it to long format, set the MultiIndex, and run the balance check. Identify which countries have missing years. Then confirm that `PanelData(panel).balanced` returns `False`.

2. **Lag safety check.** Create a panel where firm `B`'s first observation in year 2015 is immediately after firm `A`'s last observation in year 2022 (unsorted). Compute `df.groupby('firm')['y'].shift(1)` on the unsorted data and on sorted data. Show that unsorted produces wrong lags at entity boundaries.

3. **Memory comparison.** Generate a panel of 5,000 entities and 20 periods with 40% of observations missing at random. Measure memory usage (in MB) for both long and wide format representations. Report the ratio and explain when this matters in practice.

## Further Reading

- `pandas` documentation: "Reshaping and Pivot Tables" — the definitive reference for `melt`, `pivot`, `wide_to_long`, and MultiIndex operations.
- `linearmodels` documentation: "Panel Data" — covers `PanelData`, `PanelOLS`, and the exact index requirements.
- Wickham, H. (2014). "Tidy Data." *Journal of Statistical Software*, 59(10). The philosophical foundation for long format: each observation is a row, each variable is a column, each observational unit is a table.
- McKinney, W. (2010). "Data Structures for Statistical Computing in Python." *Proceedings of SciPy*, 51–56. The original `pandas` paper explaining the design choices behind MultiIndex.
