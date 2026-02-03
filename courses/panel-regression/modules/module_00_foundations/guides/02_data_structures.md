# Panel Data Structures

## In Brief

Panel data combines cross-sectional and time-series dimensions, tracking multiple entities over time. Proper organization—whether "long" or "wide" format—is critical for analysis and determines which estimation methods are accessible.

## Key Insight

The same panel dataset can be represented in multiple formats. Long format (one row per entity-time observation) is preferred for most panel regression methods, while wide format (one row per entity, multiple time columns) is useful for visualization and certain calculations. Understanding how to transform between them is essential.

## Formal Definition

**Panel Data:** A dataset with observations indexed by both entity $i \in \{1, ..., N\}$ and time $t \in \{1, ..., T\}$:

$$\{(y_{it}, X_{it}) : i = 1, ..., N; \; t = 1, ..., T\}$$

**Long Format (Stacked):**
- Total rows: $n = N \times T$ (if balanced)
- Each row contains: `[entity_id, time_id, y, X1, X2, ...]`
- Standard format for regression analysis

**Wide Format (Unstacked):**
- Total rows: $N$ (one per entity)
- Each row contains: `[entity_id, y_t1, y_t2, ..., y_tT, X1_t1, X1_t2, ..., X1_tT, ...]`
- Useful for entity-level summaries and time-series operations

**Balanced Panel:** Every entity observed in every time period ($T_i = T$ for all $i$)

**Unbalanced Panel:** Entities observed for different numbers of periods ($T_i$ varies)

## Intuitive Explanation

Think of panel data as a "cube" with three dimensions:
- **Rows (N):** Different entities (firms, countries, individuals)
- **Columns (K):** Variables measured (GDP, revenue, age)
- **Depth (T):** Time periods

Long format "slices" this cube horizontally (by time) and stacks the slices on top of each other. Wide format keeps one slice per entity but spreads time periods across multiple columns.

**Why Format Matters:**
- **Long format:** Required for fixed effects, random effects, pooled OLS
- **Wide format:** Easier to compute entity-level statistics (means, growth rates)
- **Transformation:** Frequently needed to prepare data for specific analyses

**Real-World Analogy:**
- Long format is like a student attendance sheet where each row is "Student-Date"
- Wide format is like a gradebook where each row is a student and each column is an assignment date

## Code Implementation

```python
import pandas as pd
import numpy as np

# ============================================================================
# Creating Panel Data from Scratch
# ============================================================================

def create_panel_data(n_entities=100, n_periods=10, seed=42):
    """
    Create synthetic balanced panel dataset.

    Parameters
    ----------
    n_entities : int
        Number of cross-sectional units
    n_periods : int
        Number of time periods
    seed : int
        Random seed for reproducibility

    Returns
    -------
    df : pd.DataFrame
        Panel data in long format
    """
    np.random.seed(seed)

    # Create entity and time indices
    entities = np.repeat(np.arange(1, n_entities + 1), n_periods)
    time = np.tile(np.arange(1, n_periods + 1), n_entities)

    # Create entity-specific effects (time-invariant)
    alpha_i = np.random.randn(n_entities) * 2
    alpha = np.repeat(alpha_i, n_periods)

    # Create time-varying variables
    X = np.random.randn(n_entities * n_periods) * 5 + 10

    # Create outcome with entity effects
    epsilon = np.random.randn(n_entities * n_periods)
    y = 5 + alpha + 0.8 * X + epsilon

    # Construct DataFrame
    df = pd.DataFrame({
        'entity_id': entities,
        'time': time,
        'y': y,
        'X': X
    })

    return df


# ============================================================================
# Converting Between Long and Wide Format
# ============================================================================

def long_to_wide(df, entity_col='entity_id', time_col='time', value_cols=None):
    """
    Convert long format panel to wide format.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data in long format
    entity_col : str
        Name of entity identifier column
    time_col : str
        Name of time identifier column
    value_cols : list of str, optional
        Columns to pivot. If None, pivots all non-index columns.

    Returns
    -------
    df_wide : pd.DataFrame
        Panel data in wide format
    """
    if value_cols is None:
        value_cols = [col for col in df.columns
                     if col not in [entity_col, time_col]]

    df_wide = df.pivot(index=entity_col,
                       columns=time_col,
                       values=value_cols)

    # Flatten column names
    if isinstance(df_wide.columns, pd.MultiIndex):
        df_wide.columns = [f'{var}_t{time}' for var, time in df_wide.columns]

    df_wide.reset_index(inplace=True)

    return df_wide


def wide_to_long(df_wide, entity_col='entity_id', stub_names=None,
                 time_sep='_t', time_name='time'):
    """
    Convert wide format panel to long format.

    Parameters
    ----------
    df_wide : pd.DataFrame
        Panel data in wide format
    entity_col : str
        Name of entity identifier column
    stub_names : list of str
        Variable name prefixes (e.g., ['y', 'X'] for y_t1, X_t1, etc.)
    time_sep : str
        Separator between variable name and time indicator
    time_name : str
        Name for time column in long format

    Returns
    -------
    df_long : pd.DataFrame
        Panel data in long format
    """
    # Extract time periods from column names
    time_cols = [col for col in df_wide.columns if time_sep in col]
    if not time_cols:
        raise ValueError("No time-varying columns found with separator: " + time_sep)

    # Determine stub names if not provided
    if stub_names is None:
        stub_names = list(set([col.split(time_sep)[0] for col in time_cols]))

    # Use pd.wide_to_long
    df_long = pd.wide_to_long(
        df_wide,
        stubnames=stub_names,
        i=entity_col,
        j=time_name,
        sep=time_sep,
        suffix=r'\d+'
    ).reset_index()

    return df_long


# ============================================================================
# Panel Data Indexing
# ============================================================================

def set_panel_index(df, entity_col='entity_id', time_col='time'):
    """
    Set MultiIndex for panel data (entity, time).

    This enables convenient selection and groupby operations.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data in long format
    entity_col : str
        Name of entity identifier column
    time_col : str
        Name of time identifier column

    Returns
    -------
    df_indexed : pd.DataFrame
        DataFrame with MultiIndex
    """
    df_indexed = df.set_index([entity_col, time_col]).sort_index()
    return df_indexed


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Create sample panel data
    print("Creating sample panel data...")
    df_long = create_panel_data(n_entities=5, n_periods=4)
    print("\nLong Format (first 12 rows):")
    print(df_long.head(12))

    print(f"\nShape: {df_long.shape}")
    print(f"Entities: {df_long['entity_id'].nunique()}")
    print(f"Time periods: {df_long['time'].nunique()}")

    # Convert to wide format
    print("\n" + "="*70)
    print("Converting to wide format...")
    df_wide = long_to_wide(df_long, value_cols=['y', 'X'])
    print("\nWide Format:")
    print(df_wide.head())
    print(f"\nShape: {df_wide.shape}")

    # Convert back to long format
    print("\n" + "="*70)
    print("Converting back to long format...")
    df_long_reconstructed = wide_to_long(df_wide, stub_names=['y', 'X'])
    print("\nReconstructed Long Format (first 12 rows):")
    print(df_long_reconstructed.head(12))

    # Verify they match
    df_long_sorted = df_long.sort_values(['entity_id', 'time']).reset_index(drop=True)
    df_recon_sorted = df_long_reconstructed.sort_values(['entity_id', 'time']).reset_index(drop=True)
    match = df_long_sorted[['entity_id', 'time', 'y', 'X']].equals(
        df_recon_sorted[['entity_id', 'time', 'y', 'X']]
    )
    print(f"\nFormats match after round-trip: {match}")

    # Set panel index
    print("\n" + "="*70)
    print("Setting panel MultiIndex...")
    df_indexed = set_panel_index(df_long)
    print("\nIndexed DataFrame (first 12 rows):")
    print(df_indexed.head(12))

    # Example: Select all observations for entity 1
    print("\nAll observations for entity_id=1:")
    print(df_indexed.loc[1])

    # Example: Select specific entity-time observation
    print("\nObservation for entity_id=2, time=3:")
    print(df_indexed.loc[(2, 3)])
```

## Common Pitfalls

**1. Confusing Long and Wide Format**
- **Issue:** Attempting panel regression on wide-format data.
- **Symptom:** Error messages about missing entity/time indices or incorrect degrees of freedom.
- **Solution:** Always convert to long format before running panel regressions. Use `pd.wide_to_long()` or the custom function above.

**2. Duplicate Entity-Time Pairs**
- **Issue:** Multiple rows with same entity and time identifiers.
- **Symptom:** Errors when setting MultiIndex or unexpected regression results.
- **Detection:**
  ```python
  df.duplicated(subset=['entity_id', 'time']).sum()  # Should be 0
  ```
- **Solution:** Investigate why duplicates exist (data error? need to aggregate?), then remove or aggregate appropriately.

**3. Unsorted Panel Data**
- **Issue:** Data not sorted by entity and time.
- **Consequence:** Lag/lead operations produce incorrect results; visualizations are misleading.
- **Prevention:** Always sort after loading:
  ```python
  df = df.sort_values(['entity_id', 'time']).reset_index(drop=True)
  ```

**4. Mixed Time Formats**
- **Issue:** Time stored as strings ('2020-Q1'), dates, or integers inconsistently.
- **Consequence:** Sorting fails; time operations produce errors.
- **Solution:** Standardize time format before analysis:
  ```python
  # For dates
  df['time'] = pd.to_datetime(df['time'])
  # For integers
  df['time'] = df['time'].astype(int)
  ```

**5. Missing Entity or Time Identifiers**
- **Issue:** Forgetting to include entity_id or time columns in dataset.
- **Consequence:** Cannot distinguish panel structure; reverts to cross-sectional analysis.
- **Prevention:** Always verify required columns exist:
  ```python
  required_cols = ['entity_id', 'time']
  assert all(col in df.columns for col in required_cols), \
      f"Missing required columns: {required_cols}"
  ```

**6. Implicit vs Explicit Panel Structure**
- **Issue:** Assuming data is panel when it's actually repeated cross-sections (different entities each period).
- **Detection:** Check if same entities appear in multiple periods:
  ```python
  entities_per_period = df.groupby('time')['entity_id'].nunique()
  total_unique_entities = df['entity_id'].nunique()
  is_panel = (entities_per_period > 1).all() and \
             (df.groupby('entity_id')['time'].nunique() > 1).any()
  ```

## Connections

**Builds on:**
- **Data manipulation:** pandas DataFrame operations, indexing
- **Relational databases:** Keys, joins, normalization (long format is "normalized")
- **Matrix algebra:** Panel data as stacked matrices

**Leads to:**
- **Within transformation:** Requires calculating entity-specific means (easy in long format)
- **First-differencing:** Requires time-ordering within entities
- **Balanced vs unbalanced panels:** Detection and handling strategies

**Related to:**
- **Tidy data principles:** Long format follows "tidy" structure (one observation per row)
- **Time series:** Each entity is a separate time series
- **Cross-sectional data:** Each time period is a cross-section

## Practice Problems

### 1. Conceptual: Balanced vs Unbalanced Panels

**Question:** You have data on 100 firms over 10 years. Some firms enter the sample after year 1, and some exit before year 10. Is this a balanced or unbalanced panel? What are the implications for estimation?

<details>
<summary>Solution</summary>

This is an **unbalanced panel** because not all firms are observed in all time periods.

**Implications:**
1. **Sample size:** $n < N \times T$ (fewer observations than in balanced case)
2. **Estimation:** Most panel estimators (FE, RE, pooled OLS) handle unbalanced panels without modification
3. **Bias concerns:**
   - If firms exit due to failure (attrition), this could create **selection bias**
   - If entry/exit is related to the outcome variable, estimates may be biased
4. **Degrees of freedom:** Number of entity dummies in FE equals number of unique entities, not total possible entities
5. **Standard errors:** Unbalancedness can affect variance estimation; use robust standard errors

**Best practices:**
- Investigate reasons for unbalancedness
- Test for systematic differences between entities that enter/exit vs those that remain
- Consider selection models if attrition is non-random
</details>

### 2. Implementation: Detecting Panel Structure

**Question:** Write a function that takes a DataFrame and entity/time column names, and returns a dictionary with panel characteristics: number of entities, number of time periods, whether balanced, min/max observations per entity.

<details>
<summary>Solution</summary>

```python
def describe_panel(df, entity_col='entity_id', time_col='time'):
    """
    Describe panel structure of a dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze
    entity_col : str
        Name of entity identifier column
    time_col : str
        Name of time identifier column

    Returns
    -------
    info : dict
        Dictionary with panel characteristics
    """
    n_entities = df[entity_col].nunique()
    n_periods = df[time_col].nunique()
    total_obs = len(df)

    # Observations per entity
    obs_per_entity = df.groupby(entity_col).size()
    min_obs = obs_per_entity.min()
    max_obs = obs_per_entity.max()
    mean_obs = obs_per_entity.mean()

    # Check if balanced
    is_balanced = (min_obs == max_obs == n_periods)

    # Check for duplicates
    duplicates = df.duplicated(subset=[entity_col, time_col]).sum()

    # Check for gaps in time periods
    time_values = sorted(df[time_col].unique())
    has_gaps = False
    if len(time_values) > 1:
        if isinstance(time_values[0], (int, float)):
            expected_range = set(range(int(time_values[0]),
                                      int(time_values[-1]) + 1))
            actual_range = set(time_values)
            has_gaps = (expected_range != actual_range)

    return {
        'n_entities': n_entities,
        'n_periods': n_periods,
        'total_observations': total_obs,
        'expected_obs_if_balanced': n_entities * n_periods,
        'is_balanced': is_balanced,
        'min_obs_per_entity': min_obs,
        'max_obs_per_entity': max_obs,
        'mean_obs_per_entity': mean_obs,
        'has_duplicates': duplicates > 0,
        'n_duplicates': duplicates,
        'has_time_gaps': has_gaps,
        'completeness': total_obs / (n_entities * n_periods) if n_periods > 0 else 0
    }

# Example usage
df = create_panel_data(n_entities=100, n_periods=10)
info = describe_panel(df)
for key, value in info.items():
    print(f"{key}: {value}")
```
</details>

### 3. Extension: Creating Lagged Variables

**Question:** Write a function that adds lagged variables to a panel dataset. The function should handle unbalanced panels correctly (not carrying lags across different entities).

<details>
<summary>Solution</summary>

```python
def create_lags(df, var_name, lags=1, entity_col='entity_id', time_col='time'):
    """
    Create lagged variables for panel data.

    Handles unbalanced panels correctly by not carrying lags across entities.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data in long format
    var_name : str
        Name of variable to lag
    lags : int or list of int
        Number of lags to create. If int, creates lags 1 through lags.
        If list, creates specified lag periods.
    entity_col : str
        Name of entity identifier column
    time_col : str
        Name of time identifier column

    Returns
    -------
    df_with_lags : pd.DataFrame
        DataFrame with lagged variables added
    """
    df_out = df.copy()

    # Ensure data is sorted
    df_out = df_out.sort_values([entity_col, time_col])

    # Convert lags to list
    if isinstance(lags, int):
        lag_list = list(range(1, lags + 1))
    else:
        lag_list = lags

    # Create lagged variables
    for lag in lag_list:
        lag_col_name = f'{var_name}_lag{lag}'
        df_out[lag_col_name] = (
            df_out.groupby(entity_col)[var_name]
            .shift(lag)
        )

    return df_out

# Example usage
df = create_panel_data(n_entities=3, n_periods=5)
df_with_lags = create_lags(df, 'y', lags=[1, 2])
print(df_with_lags[['entity_id', 'time', 'y', 'y_lag1', 'y_lag2']].head(15))
```

**Key insight:** Using `.groupby(entity_col).shift()` ensures lags are computed within each entity, preventing "leakage" across entities.
</details>

## Further Reading

**Essential:**
- **Wickham, H.** (2014). "Tidy Data." *Journal of Statistical Software*, 59(10), 1-23. *Principles of data organization that align with long format panel data.*

- **McKinney, W.** (2017). *Python for Data Analysis* (2nd ed.), Chapter 8: Data Wrangling: Join, Combine, and Reshape. *Comprehensive guide to pandas reshaping operations.*

**Panel Data Specific:**
- **Hsiao, C.** (2014). *Analysis of Panel Data* (3rd ed.), Chapter 1: Introduction. *Covers panel data structures and notation.*

- **Wooldridge, J. M.** (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.), Chapter 10: Basic Linear Unobserved Effects Panel Data Models. *Mathematical treatment of panel structure.*

**Practical Guides:**
- **VanderPlas, J.** (2016). *Python Data Science Handbook*, Chapter 3: Data Manipulation with Pandas. *Excellent practical guide to pandas operations for data reshaping.*

- **Stata Documentation:** "reshape — Convert data from wide to long form and vice versa." *Clear examples and edge cases, applicable beyond Stata.*
