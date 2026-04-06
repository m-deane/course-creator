# Panel Data Formats: Long vs Wide and Multi-Index

> **Reading time:** ~20 min | **Module:** 01 — Panel Structure | **Prerequisites:** Module 0 Foundations


## In Brief


<div class="callout-key">

**Key Concept Summary:** Panel data can be stored in two formats: **long format** (stacked, one row per observation) or **wide format** (matrix, one row per entity). Long format is preferred for regression analysis and modern

</div>

Panel data can be stored in two formats: **long format** (stacked, one row per observation) or **wide format** (matrix, one row per entity). Long format is preferred for regression analysis and modern panel libraries, while wide format suits time series operations and visualization. Efficient panel analysis requires mastery of format conversion, multi-index DataFrames, and reshaping operations.

> 💡 **Key Insight:** Format choice affects memory usage, computation speed, and code clarity. Long format enables straightforward filtering and grouping but creates sparse storage for unbalanced panels. Wide format enables vectorized time series operations but struggles with varying entity attributes. The solution: store in long format with multi-index (entity, time), convert to wide for specific operations, leverage pandas' hierarchical indexing.

## Formal Definition

### Long Format (Stacked)

Each observation is a separate row:

$$\text{Long: } (i, t, y_{it}, x_{it}) \text{ for } i=1,...,N; \, t=1,...,T$$

**Structure:**
```
entity | time | y    | x1   | x2
-------|------|------|------|-----
1      | 2020 | 5.2  | 3.1  | 7.4
1      | 2021 | 5.8  | 3.3  | 7.6
2      | 2020 | 4.1  | 2.9  | 6.8
2      | 2021 | 4.5  | 3.0  | 7.1
```

**Dimensions:** $(N \times T)$ rows × $(2 + K)$ columns

### Wide Format (Matrix)

Each entity is a row, time periods are columns:

$$\text{Wide: } Y \in \mathbb{R}^{N \times T}, \quad X_k \in \mathbb{R}^{N \times T}$$

<div class="callout-insight">

**Insight:** Panel data lets you control for unobservable differences between entities that are constant over time. This is the single most important reason to prefer panel data over repeated cross-sections.

</div>


**Structure for variable $y$:**
```
entity | 2020 | 2021 | 2022
-------|------|------|-----
1      | 5.2  | 5.8  | 6.1
2      | 4.1  | 4.5  | 4.7
3      | 6.3  | 6.5  | 6.8
```

**Dimensions:** $N$ rows × $T$ columns (per variable)

### Multi-Index DataFrame

Hierarchical indexing for efficient panel operations:

**Structure:**
```python
df.index = pd.MultiIndex.from_arrays([entity_ids, time_ids],
                                      names=['entity', 'time'])
```

**Access patterns:**
- Single entity: `df.loc[entity_id]`
- Single time: `df.xs(time, level='time')`
- Cross-section: `df.loc[(slice(None), time), :]`

### Reshaping Operations

**Long to Wide:**
$$\text{pivot}(\text{df}_{\text{long}}, \text{index}=i, \text{columns}=t, \text{values}=y)$$

**Wide to Long:**
$$\text{melt}(\text{df}_{\text{wide}}, \text{id\_vars}=i, \text{var\_name}=t, \text{value\_name}=y)$$

## Intuitive Explanation

<div class="callout-warning">

**Warning:** Reporting results without appropriate standard errors is a common mistake. In panel data, conventional OLS standard errors are almost always wrong -- use clustered or heteroskedasticity-robust standard errors.

</div>


### When to Use Each Format

**Long Format Best For:**
- Regression analysis (statsmodels, linearmodels)
- Groupby operations (entity-specific stats)
- Unbalanced panels (missing observations)
- Adding time-varying covariates
- Filtering by conditions

**Wide Format Best For:**
- Time series plots (one line per entity)
- Correlation matrices across entities
- Vectorized time operations (lags, differences)
- Fixed effects estimation (demeaning)
- Exporting to Excel (human-readable)

### Real-World Example: Wage Data

**Long Format (Analysis-Ready):**

<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
person_id | year | wage | education | experience
----------|------|------|-----------|------------
101       | 2019 | 45000| 16        | 5
101       | 2020 | 48000| 16        | 6
101       | 2021 | 51000| 16        | 7
102       | 2019 | 52000| 18        | 8
```


</div>

**Wide Format (Time Series View):**

<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
person_id | wage_2019 | wage_2020 | wage_2021
----------|-----------|-----------|----------
101       | 45000     | 48000     | 51000
102       | 52000     | 55000     | 58000
```


</div>

### Memory Considerations

**Balanced Panel:**
- Long: $(N \times T) \times (2 + K)$ cells
- Wide: $N \times (T \times K)$ cells
- Memory difference: Minimal (same data)

**Unbalanced Panel (50% missing):**
- Long: $(0.5 \times N \times T) \times (2 + K)$ cells (sparse)
- Wide: $N \times (T \times K)$ cells with NaNs (dense)
- Memory savings: Long format uses 50% less memory

## Code Implementation

### Format Conversion


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np
import pandas as pd
from linearmodels.panel import PanelData

# Generate sample panel data
np.random.seed(42)
N, T = 100, 10  # 100 entities, 10 time periods

entities = np.repeat(range(N), T)
times = np.tile(range(2015, 2015 + T), N)

# Create long format
data_long = pd.DataFrame({
    'entity_id': entities,
    'year': times,
    'wage': np.random.randn(N * T) * 5000 + 50000,
    'education': np.repeat(np.random.randint(12, 20, N), T),
    'experience': np.repeat(range(N), T) % 30 + np.tile(range(T), N)
})

print("=" * 70)
print("LONG FORMAT")
print("=" * 70)
print(data_long.head(10))
print(f"\nShape: {data_long.shape}")
print(f"Memory usage: {data_long.memory_usage(deep=True).sum() / 1024:.1f} KB")

# Convert to wide format
data_wide = data_long.pivot(
    index='entity_id',
    columns='year',
    values='wage'
)

print("\n" + "=" * 70)
print("WIDE FORMAT (wage only)")
print("=" * 70)
print(data_wide.head())
print(f"\nShape: {data_wide.shape}")
print(f"Memory usage: {data_wide.memory_usage(deep=True).sum() / 1024:.1f} KB")

# Convert back to long
data_long_restored = data_wide.reset_index().melt(
    id_vars='entity_id',
    var_name='year',
    value_name='wage'
)

print("\n" + "=" * 70)
print("RESTORED TO LONG")
print("=" * 70)
print(data_long_restored.head(10))

# Verify equivalence
print(f"\nData preserved: {data_long_restored.shape == (N*T, 3)}")
```


</div>

### Multi-Index DataFrames

```python
# Create multi-index DataFrame (long format with hierarchical index)
data_mi = data_long.set_index(['entity_id', 'year'])

print("=" * 70)
print("MULTI-INDEX DATAFRAME")
print("=" * 70)
print(data_mi.head(10))
print(f"\nIndex levels: {data_mi.index.names}")

# Access single entity
entity_50 = data_mi.loc[50]
print("\n" + "-" * 70)
print("Entity 50 (all years):")
print("-" * 70)
print(entity_50)

# Access single year (cross-section)
year_2020 = data_mi.xs(2020, level='year')
print("\n" + "-" * 70)
print(f"Year 2020 cross-section (n={len(year_2020)}):")
print("-" * 70)
print(year_2020.head())

# Access specific observation
obs = data_mi.loc[(50, 2020)]
print("\n" + "-" * 70)
print("Entity 50, Year 2020:")
print("-" * 70)
print(obs)

# Group operations
entity_means = data_mi.groupby(level='entity_id').mean()
print("\n" + "-" * 70)
print("Entity means (first 5):")
print("-" * 70)
print(entity_means.head())

time_means = data_mi.groupby(level='year').mean()
print("\n" + "-" * 70)
print("Time means:")
print("-" * 70)
print(time_means)
```

### Handling Unbalanced Panels

```python
# Create unbalanced panel (some entities missing some years)
np.random.seed(42)
missing_mask = np.random.random(N * T) > 0.8  # 20% missing
data_unbalanced = data_long[~missing_mask].copy()

print("=" * 70)
print("UNBALANCED PANEL")
print("=" * 70)
print(f"Balanced: {N * T} observations")
print(f"Unbalanced: {len(data_unbalanced)} observations")
print(f"Missing: {N * T - len(data_unbalanced)} observations ({(1 - len(data_unbalanced)/(N*T))*100:.1f}%)")

# Check balance by entity
obs_per_entity = data_unbalanced.groupby('entity_id').size()
print(f"\nObservations per entity:")
print(f"  Mean: {obs_per_entity.mean():.1f}")
print(f"  Std: {obs_per_entity.std():.1f}")
print(f"  Min: {obs_per_entity.min()}")
print(f"  Max: {obs_per_entity.max()}")

# Pivot unbalanced (creates NaNs)
data_unbalanced_wide = data_unbalanced.pivot(
    index='entity_id',
    columns='year',
    values='wage'
)

print("\n" + "-" * 70)
print("Unbalanced as wide format (with NaNs):")
print("-" * 70)
print(data_unbalanced_wide.head())
print(f"\nNaN count: {data_unbalanced_wide.isna().sum().sum()}")
print(f"Memory usage (wide): {data_unbalanced_wide.memory_usage(deep=True).sum() / 1024:.1f} KB")
print(f"Memory usage (long): {data_unbalanced.memory_usage(deep=True).sum() / 1024:.1f} KB")

# Identify entities with complete data
complete_entities = obs_per_entity[obs_per_entity == T].index
print(f"\nEntities with complete data: {len(complete_entities)}")

# Create balanced subset
data_balanced_subset = data_unbalanced[
    data_unbalanced['entity_id'].isin(complete_entities)
]
print(f"Balanced subset: {len(data_balanced_subset)} observations")
```

### Advanced Reshaping

```python
# Multiple variables to wide
data_multi = data_long.copy()

# Pivot multiple columns
data_wide_multi = data_multi.pivot(
    index='entity_id',
    columns='year',
    values=['wage', 'experience']
)

print("=" * 70)
print("WIDE FORMAT (MULTIPLE VARIABLES)")
print("=" * 70)
print(data_wide_multi.head())
print(f"\nColumn levels: {data_wide_multi.columns.names}")

# Flatten column names
data_wide_multi.columns = [f"{col[0]}_{col[1]}" for col in data_wide_multi.columns]
print("\n" + "-" * 70)
print("Flattened column names:")
print("-" * 70)
print(data_wide_multi.head())

# Convert back to long (multiple variables)
data_long_multi = data_wide_multi.reset_index().melt(
    id_vars='entity_id',
    var_name='variable_year',
    value_name='value'
)

# Parse variable and year
data_long_multi[['variable', 'year']] = data_long_multi['variable_year'].str.split('_', expand=True)
data_long_multi['year'] = data_long_multi['year'].astype(int)

# Pivot variables back to columns
data_long_restored_multi = data_long_multi.pivot_table(
    index=['entity_id', 'year'],
    columns='variable',
    values='value'
).reset_index()

print("\n" + "-" * 70)
print("Restored to long (multiple variables):")
print("-" * 70)
print(data_long_restored_multi.head(10))
```

### PanelData Wrapper

```python
from linearmodels.panel import PanelData

# Use PanelData for automatic format handling
panel_data = PanelData(data_mi[['wage', 'experience']])

print("=" * 70)
print("PANELDATA WRAPPER")
print("=" * 70)
print(panel_data)

# Access as 3D array (entities × time × variables)
print(f"\nDimensions: {panel_data.shape}")
print(f"Entities: {panel_data.nentity}")
print(f"Time periods: {panel_data.nobs}")
print(f"Variables: {panel_data.nvar}")

# Extract specific components
print("\n" + "-" * 70)
print("First entity:")
print("-" * 70)
print(panel_data.dataframe.loc[0])
```

### Performance Comparison

```python
import time

# Large panel
N_large, T_large = 10000, 50
entities_large = np.repeat(range(N_large), T_large)
times_large = np.tile(range(T_large), N_large)
data_large_long = pd.DataFrame({
    'entity': entities_large,
    'time': times_large,
    'value': np.random.randn(N_large * T_large)
})

print("=" * 70)
print("PERFORMANCE COMPARISON (N=10000, T=50)")
print("=" * 70)

# Benchmark: Filter operation (long format)
start = time.time()
filtered_long = data_large_long[data_large_long['entity'] < 100]
time_long = time.time() - start
print(f"\nFilter in long format: {time_long*1000:.2f}ms")

# Convert to wide
data_large_wide = data_large_long.pivot(
    index='entity',
    columns='time',
    values='value'
)

# Benchmark: Filter operation (wide format)
start = time.time()
filtered_wide = data_large_wide.loc[data_large_wide.index < 100]
time_wide = time.time() - start
print(f"Filter in wide format: {time_wide*1000:.2f}ms")

# Benchmark: Time series operation (lag)
start = time.time()
data_large_wide_lagged = data_large_wide.shift(1, axis=1)
time_wide_lag = time.time() - start
print(f"\nLag operation in wide format: {time_wide_lag*1000:.2f}ms")

# Same operation in long format (requires groupby)
data_large_long_mi = data_large_long.set_index(['entity', 'time'])
start = time.time()
data_large_long_lagged = data_large_long_mi.groupby(level='entity')['value'].shift(1)
time_long_lag = time.time() - start
print(f"Lag operation in long format: {time_long_lag*1000:.2f}ms")

print(f"\nSpeedup (wide vs long for lags): {time_long_lag/time_wide_lag:.1f}x")
```

## Common Pitfalls

**1. Losing Time Order in Long Format**
- Problem: Pivoting unsorted data creates wrong column order
- Symptom: Wide format has years like [2021, 2019, 2020]
- Solution: Sort by entity and time before pivoting

**2. Duplicate Index Entries**
- Problem: Multiple observations per (entity, time)
- Symptom: `ValueError: Index contains duplicate entries`
- Solution: Check uniqueness with `df.groupby(['entity', 'time']).size().max()`

**3. Mixed Data Types in Wide Format**
- Problem: Entity ID becomes column name (integer)
- Symptom: Cannot access columns with `df.2020`
- Solution: Use `.loc[:, 2020]` or convert to string column names

**4. Memory Explosion with Sparse Panels**
- Problem: Unbalanced panel converted to wide creates massive NaN matrix
- Symptom: Out of memory errors
- Solution: Keep unbalanced panels in long format

**5. Incorrect Pivot with Multiple Variables**
- Problem: Trying to pivot with `values=[col1, col2]` creates wrong structure
- Symptom: Nested column MultiIndex
- Solution: Pivot each variable separately or use `pivot_table`

## Connections

**Builds on:**
- Pandas DataFrame fundamentals
- Hierarchical indexing
- Data wrangling techniques

**Leads to:**
- Module 1.3: Data quality (handling missing values in both formats)
- Module 2: Fixed effects estimation (within transformation uses format conversions)
- Module 4: Model selection (different libraries prefer different formats)

**Related concepts:**
- Tidy data principles (long format aligns with tidy data)
- Database normalization (long format is normalized form)
- Matrix operations (wide format enables vectorization)

## Practice Problems

<div class="callout-danger">

**Danger:** Never include a lagged dependent variable in a fixed effects model without using an appropriate estimator (e.g., Arellano-Bond GMM). The within-transformation creates mechanical correlation between the transformed lagged variable and the transformed error, biasing all coefficients.

</div>


1. **Format Detection**
   You receive a dataset with shape (500, 25).
   Could this be long or wide format?
   What additional information do you need?

2. **Reshaping Challenge**
   Panel: 50 countries, 20 years, 3 variables (GDP, population, inflation).
   Long format shape? Wide format shape (all variables)?
   Which format for: (a) regression, (b) correlation matrix, (c) time series plot?

3. **Memory Optimization**
   Unbalanced panel: 1000 entities, 100 time periods, 60% missing observations.
   Calculate memory usage for long vs wide format (assume 8 bytes per float).
   Which format is more efficient?

4. **Access Pattern**
   Multi-index DataFrame: `df.loc[(entity, time), variable]`
   How to extract:
   - All observations for entity 5?
   - All entities in year 2020?
   - Entity 5, years 2018-2020?
   - Average of variable 'x' across all entities in 2020?

5. **Pivot Aggregation**
   Dataset has duplicate (entity, time) pairs (e.g., quarterly data with same year).
   What happens with `.pivot()`? How to fix with `.pivot_table()`?
   Which aggregation function: mean, sum, first, last?

## Further Reading

**Pandas Documentation:**
1. **"Reshaping and Pivot Tables"** - Official pandas guide
2. **"MultiIndex / Advanced Indexing"** - Hierarchical indexing
3. **"Working with Sparse Data"** - Memory-efficient unbalanced panels

**Panel Data Formats:**
4. **"Tidy Data" by Hadley Wickham** - Data structuring principles
5. **"Python for Data Analysis" by Wes McKinney** - Pandas creator's guide
6. **"Effective Pandas" by Matt Harrison** - Advanced pandas patterns

**Performance:**
7. **"High Performance Pandas"** - Memory and speed optimization
8. **"Pandas Performance Tuning"** - Benchmarking and profiling

**Panel-Specific:**
9. **"Econometrics in Python: Part II"** - Panel data structures
10. **"Panel Data Econometrics" by Baltagi** - Mathematical foundations

---

*"Store long, reshape temporarily. Long format is the source of truth."*


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

<a class="link-card" href="./02_pooled_ols_limitations.md">
  <div class="link-card-title">02 Pooled Ols Limitations</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_pooled_ols_limitations.md">
  <div class="link-card-title">02 Pooled Ols Limitations — Companion Slides</div>
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

