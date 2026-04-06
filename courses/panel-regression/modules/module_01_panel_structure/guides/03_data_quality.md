# Panel Data Quality: Missing Data, Unbalanced Panels, and Outliers

> **Reading time:** ~20 min | **Module:** 01 — Panel Structure | **Prerequisites:** Module 0 Foundations


## In Brief


<div class="callout-key">

**Key Concept Summary:** Panel data quality issues—missing observations, unbalanced panels, and outliers—directly affect estimation consistency and inference validity. Missing data creates unbalanced panels where entities ...

</div>

Panel data quality issues—missing observations, unbalanced panels, and outliers—directly affect estimation consistency and inference validity. Missing data creates unbalanced panels where entities have different numbers of observations, requiring careful handling to avoid selection bias. Outliers distort fixed effects estimates more severely than cross-sectional data because they affect multiple time periods for the same entity.

> 💡 **Key Insight:** Panel data quality problems interact with estimation methods. Fixed effects estimation is robust to some missing data patterns (missing completely at random) but biased under others (missing related to unobservables). Outliers in panel data have "leverage across time"—one bad observation for an entity affects all estimated fixed effects. The solution: detect patterns of missingness, test for selection bias, and use robust estimation or outlier-resistant transformations.

## Formal Definition

### Types of Missing Data

**1. Missing Completely at Random (MCAR)**
$$P(\text{Missing}_{it} | y_{it}, x_{it}, \alpha_i) = P(\text{Missing}_{it})$$

Missingness unrelated to observed or unobserved factors.
**Example:** Data lost due to random server crash.

**2. Missing at Random (MAR)**
$$P(\text{Missing}_{it} | y_{it}, x_{it}, \alpha_i) = P(\text{Missing}_{it} | x_{it})$$

Missingness depends only on observables.
**Example:** High-income entities less likely to report, but income is observed.

**3. Missing Not at Random (MNAR)**
$$P(\text{Missing}_{it} | y_{it}, x_{it}, \alpha_i) = P(\text{Missing}_{it} | y_{it}, \alpha_i)$$

Missingness depends on unobservables.
**Example:** Firms hide data when performance is poor.

### Panel Balance

**Balanced Panel:**
$$\{(y_{it}, x_{it}) : i=1,...,N; \, t=1,...,T\}$$
Every entity observed in every period. Dimensions: $N \times T$

**Unbalanced Panel:**
$$\{(y_{it}, x_{it}) : i=1,...,N; \, t \in T_i\}$$
where $T_i \subseteq \{1,...,T\}$ varies by entity.

<div class="callout-insight">

**Insight:** Panel data lets you control for unobservable differences between entities that are constant over time. This is the single most important reason to prefer panel data over repeated cross-sections.

</div>


**Strongly Balanced:** All entities have same time periods
**Weakly Unbalanced:** Some missing, but entities overlap in time

### Outlier Detection

**Leverage in Panel Data:**

For observation $(i,t)$, leverage combines:
$$h_{it} = h_{it}^{\text{within}} + h_{it}^{\text{between}}$$

where:
- $h_{it}^{\text{within}}$: Leverage within entity's time series
- $h_{it}^{\text{between}}$: Leverage across entities at time $t$

**Studentized Residuals:**
$$r_{it}^* = \frac{\hat{\epsilon}_{it}}{\hat{\sigma} \sqrt{1 - h_{it}}}$$

Outlier if $|r_{it}^*| > 3$

### Robust Estimation

**Huber-White Robust Standard Errors:**
$$\hat{V}(\hat{\beta}) = (X'X)^{-1} \left(\sum_{i=1}^N X_i' \hat{\epsilon}_i \hat{\epsilon}_i' X_i \right) (X'X)^{-1}$$

Robust to heteroskedasticity and some outliers.

**Winsorization:**
Replace extreme values with percentile thresholds:
$$y_{it}^{\text{wins}} = \begin{cases}
P_{\alpha} & \text{if } y_{it} < P_{\alpha} \\
y_{it} & \text{if } P_{\alpha} \leq y_{it} \leq P_{1-\alpha} \\
P_{1-\alpha} & \text{if } y_{it} > P_{1-\alpha}
\end{cases}$$

Common: $\alpha = 0.01$ (winsorize at 1st and 99th percentiles)

## Intuitive Explanation

<div class="callout-warning">

**Warning:** Reporting results without appropriate standard errors is a common mistake. In panel data, conventional OLS standard errors are almost always wrong -- use clustered or heteroskedasticity-robust standard errors.

</div>


### Missing Data Patterns

**MCAR (No Bias):**
```
Entity 1: [obs, obs, MISS, obs, obs] → Missing by chance
Entity 2: [obs, MISS, obs, obs, obs] → No pattern
Entity 3: [obs, obs, obs, MISS, obs]
```
FE estimation consistent (loses efficiency, not bias).

**MAR (Conditional on Observables):**
```
Entity 1 (High income): [obs, obs, obs, obs, obs] → Always reports
Entity 2 (Low income):  [obs, MISS, MISS, obs, obs] → Less reporting
```
If income is controlled for, no bias.

**MNAR (Selection Bias):**
```
Entity 1 (Good performance): [obs, obs, obs, obs, obs]
Entity 2 (Bad performance):  [obs, obs, MISS, MISS, MISS] → Hides bad years
```
Bias: Sample only includes good outcomes → overestimate effects.

### Outliers in Panel Data

**Cross-Sectional Outlier (One-Time):**
```
Entity 5, Year 2020: Wage = $500,000 (CEO, rest are $50k)
Impact: Affects only 2020 cross-section, minor influence
```

**Panel Outlier (Persistent):**
```
Entity 5, All Years: Consistently 10x higher than others
Impact: Entity fixed effect absorbs this, but high leverage
```

**Transient Shock (Volatile Entity):**
```
Entity 5: [50k, 52k, 250k, 48k, 51k] → One extreme year
Impact: Distorts within-entity variance, affects FE estimate
```

### Why Panel Outliers Are Worse

Fixed effects estimation demeaning:
$$\tilde{y}_{it} = y_{it} - \bar{y}_i$$

If Entity 5 has one outlier:
$$\bar{y}_5 = \frac{50 + 52 + 250 + 48 + 51}{5} = 90.2$$

All observations for Entity 5 are distorted:
- $\tilde{y}_{51} = 50 - 90.2 = -40.2$ (looks abnormally low!)
- $\tilde{y}_{53} = 250 - 90.2 = 159.8$ (extreme)

One bad year affects all years for that entity.

## Code Implementation

### Detecting Missing Patterns


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Generate panel data with missing patterns
np.random.seed(42)
N, T = 100, 10
entities = np.repeat(range(N), T)
times = np.tile(range(2015, 2015 + T), N)

# Base data
data = pd.DataFrame({
    'entity': entities,
    'time': times,
    'x': np.random.randn(N * T),
    'alpha': np.repeat(np.random.randn(N), T),  # Entity fixed effect
})

# Generate outcome with fixed effects
data['y'] = 2 + 1.5 * data['x'] + data['alpha'] + np.random.randn(N * T)

print("=" * 70)
print("MISSING DATA ANALYSIS")
print("=" * 70)

# Introduce MCAR missingness (10% random)
mcar_mask = np.random.random(N * T) < 0.10
data_mcar = data.copy()
data_mcar.loc[mcar_mask, 'y'] = np.nan

print(f"\nMCAR: {mcar_mask.sum()} missing observations ({mcar_mask.mean()*100:.1f}%)")

# Introduce MAR missingness (depends on x)

# Low x values more likely to be missing
mar_prob = 1 / (1 + np.exp(data['x']))  # Logistic function
mar_mask = np.random.random(N * T) < mar_prob * 0.3  # Scale to ~10% missing
data_mar = data.copy()
data_mar.loc[mar_mask, 'y'] = np.nan

print(f"MAR: {mar_mask.sum()} missing observations ({mar_mask.mean()*100:.1f}%)")

# Introduce MNAR missingness (depends on y itself)

# High y values more likely to be missing
mnar_prob = 1 / (1 + np.exp(-data['y'] + data['y'].mean()))
mnar_mask = np.random.random(N * T) < mnar_prob * 0.15
data_mnar = data.copy()
data_mnar.loc[mnar_mask, 'y'] = np.nan

print(f"MNAR: {mnar_mask.sum()} missing observations ({mnar_mask.mean()*100:.1f}%)")

# Visualize missing patterns
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (data_miss, title) in zip(axes, [
    (data_mcar, 'MCAR'),
    (data_mar, 'MAR'),
    (data_mnar, 'MNAR')
]):
    # Create missingness matrix (entities × time)
    miss_matrix = data_miss.pivot_table(
        index='entity',
        columns='time',
        values='y',
        aggfunc=lambda x: x.isna().any()
    ).astype(int)

    ax.imshow(miss_matrix.iloc[:50], aspect='auto', cmap='RdYlGn_r')
    ax.set_title(f'{title} Pattern (First 50 Entities)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Entity')

plt.tight_layout()
plt.savefig('missing_patterns.png', dpi=150, bbox_inches='tight')
print("\nMissing patterns visualization saved to missing_patterns.png")
```


</div>
</div>

### Testing for Missing Data Bias


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
from linearmodels.panel import PanelOLS

def test_missing_bias(data_full, data_missing, entity_col='entity', time_col='time'):
    """
    Test if missingness introduces bias by comparing full vs missing data estimates.
    """
    # Prepare full data
    data_full_mi = data_full.set_index([entity_col, time_col])

    # Fit on full data
    model_full = PanelOLS(
        data_full_mi['y'],
        data_full_mi[['x']],
        entity_effects=True
    ).fit()

    # Prepare missing data
    data_missing_clean = data_missing.dropna(subset=['y'])
    data_missing_mi = data_missing_clean.set_index([entity_col, time_col])

    # Fit on missing data
    model_missing = PanelOLS(
        data_missing_mi['y'],
        data_missing_mi[['x']],
        entity_effects=True
    ).fit()

    return {
        'coef_full': model_full.params['x'],
        'coef_missing': model_missing.params['x'],
        'bias': model_missing.params['x'] - model_full.params['x'],
        'bias_pct': (model_missing.params['x'] / model_full.params['x'] - 1) * 100,
        'se_full': model_full.std_errors['x'],
        'se_missing': model_missing.std_errors['x']
    }

print("\n" + "=" * 70)
print("MISSING DATA BIAS TEST")
print("=" * 70)

for data_miss, name in [(data_mcar, 'MCAR'), (data_mar, 'MAR'), (data_mnar, 'MNAR')]:
    result = test_missing_bias(data, data_miss)
    print(f"\n{name}:")
    print(f"  True coefficient: 1.500")
    print(f"  Full data estimate: {result['coef_full']:.4f}")
    print(f"  Missing data estimate: {result['coef_missing']:.4f}")
    print(f"  Bias: {result['bias']:.4f} ({result['bias_pct']:.2f}%)")
    print(f"  SE (full): {result['se_full']:.4f}")
    print(f"  SE (missing): {result['se_missing']:.4f}")
```


</div>
</div>

### Handling Unbalanced Panels


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Create strongly unbalanced panel
obs_per_entity = np.random.randint(3, T+1, N)  # 3 to T observations per entity
data_unbalanced = []

for i in range(N):
    n_obs = obs_per_entity[i]
    time_periods = np.random.choice(range(2015, 2015 + T), n_obs, replace=False)
    time_periods.sort()

    entity_data = pd.DataFrame({
        'entity': [i] * n_obs,
        'time': time_periods,
        'x': np.random.randn(n_obs),
        'alpha': [data['alpha'].iloc[i*T]] * n_obs
    })
    entity_data['y'] = 2 + 1.5 * entity_data['x'] + entity_data['alpha'] + np.random.randn(n_obs)

    data_unbalanced.append(entity_data)

data_unbalanced = pd.concat(data_unbalanced, ignore_index=True)

print("\n" + "=" * 70)
print("UNBALANCED PANEL ANALYSIS")
print("=" * 70)

obs_per_entity_actual = data_unbalanced.groupby('entity').size()
print(f"\nTotal observations: {len(data_unbalanced)}")
print(f"Observations per entity:")
print(f"  Mean: {obs_per_entity_actual.mean():.1f}")
print(f"  Std: {obs_per_entity_actual.std():.1f}")
print(f"  Min: {obs_per_entity_actual.min()}")
print(f"  Max: {obs_per_entity_actual.max()}")

# Check time overlap
time_coverage = data_unbalanced.groupby('time').size()
print(f"\nTime period coverage:")
print(time_coverage)

# Estimate with unbalanced panel
data_unbalanced_mi = data_unbalanced.set_index(['entity', 'time'])
model_unbalanced = PanelOLS(
    data_unbalanced_mi['y'],
    data_unbalanced_mi[['x']],
    entity_effects=True
).fit()

print(f"\nUnbalanced panel FE estimate:")
print(f"  Coefficient: {model_unbalanced.params['x']:.4f}")
print(f"  SE: {model_unbalanced.std_errors['x']:.4f}")
print(f"  True value: 1.500")
```



### Outlier Detection and Treatment

```python
from scipy.stats import zscore

def detect_panel_outliers(data, value_col, entity_col='entity', time_col='time', threshold=3):
    """
    Detect outliers in panel data using multiple methods.
    """
    data = data.copy()

    # 1. Global Z-score
    data['zscore_global'] = zscore(data[value_col].dropna())

    # 2. Within-entity Z-score
    data['zscore_within'] = data.groupby(entity_col)[value_col].transform(
        lambda x: zscore(x.dropna()) if len(x.dropna()) > 2 else 0
    )

    # 3. Temporal Z-score (across entities at each time)
    data['zscore_temporal'] = data.groupby(time_col)[value_col].transform(
        lambda x: zscore(x.dropna()) if len(x.dropna()) > 2 else 0
    )

    # Identify outliers
    data['outlier_global'] = np.abs(data['zscore_global']) > threshold
    data['outlier_within'] = np.abs(data['zscore_within']) > threshold
    data['outlier_temporal'] = np.abs(data['zscore_temporal']) > threshold

    # Combined outlier (any method flags it)
    data['outlier_any'] = (data['outlier_global'] |
                           data['outlier_within'] |
                           data['outlier_temporal'])

    return data

# Introduce outliers
data_with_outliers = data.copy()
outlier_indices = np.random.choice(len(data), size=20, replace=False)
data_with_outliers.loc[outlier_indices, 'y'] = data_with_outliers.loc[outlier_indices, 'y'] + np.random.choice([-1, 1], 20) * np.random.uniform(10, 20, 20)

print("\n" + "=" * 70)
print("OUTLIER DETECTION")
print("=" * 70)

# Detect outliers
data_with_outliers = detect_panel_outliers(data_with_outliers, 'y')

print(f"\nOutliers detected:")
print(f"  Global method: {data_with_outliers['outlier_global'].sum()}")
print(f"  Within-entity method: {data_with_outliers['outlier_within'].sum()}")
print(f"  Temporal method: {data_with_outliers['outlier_temporal'].sum()}")
print(f"  Any method: {data_with_outliers['outlier_any'].sum()}")

# Compare estimates with/without outliers
data_clean = data_with_outliers[~data_with_outliers['outlier_any']].copy()

for dataset, name in [(data, 'True (no outliers)'),
                      (data_with_outliers, 'With outliers'),
                      (data_clean, 'Outliers removed')]:
    data_mi = dataset.set_index(['entity', 'time'])
    model = PanelOLS(data_mi['y'], data_mi[['x']], entity_effects=True).fit()
    print(f"\n{name}:")
    print(f"  Coefficient: {model.params['x']:.4f}")
    print(f"  SE: {model.std_errors['x']:.4f}")


# Winsorization
def winsorize_panel(data, value_col, lower=0.01, upper=0.99):
    """Winsorize values at specified percentiles."""
    p_lower = data[value_col].quantile(lower)
    p_upper = data[value_col].quantile(upper)

    data_wins = data.copy()
    data_wins[value_col] = data_wins[value_col].clip(lower=p_lower, upper=p_upper)

    return data_wins

data_winsorized = winsorize_panel(data_with_outliers, 'y')

print("\n" + "=" * 70)
print("WINSORIZATION")
print("=" * 70)

print(f"\nOriginal y statistics:")
print(data_with_outliers['y'].describe())

print(f"\nWinsorized y statistics (1%, 99%):")
print(data_winsorized['y'].describe())

# Estimate with winsorized data
data_wins_mi = data_winsorized.set_index(['entity', 'time'])
model_wins = PanelOLS(data_wins_mi['y'], data_wins_mi[['x']], entity_effects=True).fit()

print(f"\nWinsorized estimate:")
print(f"  Coefficient: {model_wins.params['x']:.4f}")
print(f"  SE: {model_wins.std_errors['x']:.4f}")
```

## Common Pitfalls

**1. Assuming All Missing Data is MCAR**
- Problem: Treating MNAR as MCAR by simply dropping missing observations
- Symptom: Biased estimates, results don't replicate
- Solution: Test missingness patterns, use selection models if MNAR

**2. Using Listwise Deletion with Unbalanced Panels**
- Problem: Dropping all entities with any missing data
- Symptom: Massive sample size reduction, loss of efficiency
- Solution: Use all available observations (FE handles unbalanced panels)

**3. Ignoring Within-Entity Outliers**
- Problem: Only checking global outliers, missing entity-specific shocks
- Symptom: Some entities have huge residuals
- Solution: Check within-entity deviations, not just global

**4. Over-Aggressive Outlier Removal**
- Problem: Removing observations because they don't fit model
- Symptom: Artificially high R-squared, biased estimates
- Solution: Use objective criteria (e.g., 3σ rule), document removals

**5. Not Checking Balance After Cleaning**
- Problem: Data cleaning creates unbalanced panel without noticing
- Symptom: Confusion about sample size, entity counts
- Solution: Always report balance statistics after each cleaning step

## Connections

**Builds on:**
- Module 1.2: Data formats (handling missing data in long vs wide)
- Statistical foundations (missing data theory, outlier detection)
- Data wrangling (pandas operations for cleaning)

**Leads to:**
- Module 2: Fixed effects (how FE handles unbalanced panels)
- Module 3: Random effects (RE requires stricter missing data assumptions)
- Module 5: Dynamic panels (lagged variables amplify missing data issues)

**Related concepts:**
- Multiple imputation (advanced missing data handling)
- Selection models (Heckman correction for MNAR)
- Robust regression (M-estimation, LAD regression)

## Practice Problems

<div class="callout-danger">

**Danger:** Never include a lagged dependent variable in a fixed effects model without using an appropriate estimator (e.g., Arellano-Bond GMM). The within-transformation creates mechanical correlation between the transformed lagged variable and the transformed error, biasing all coefficients.



1. **Missing Pattern Diagnosis**
   Panel: 50 firms, 10 years. Missing observations cluster in recession years (2008-2009).
   Is this MCAR, MAR, or MNAR? How to test? Implications for FE estimation?

2. **Balance Calculation**
   100 entities, 20 time periods.
   Balanced panel: How many observations?
   After dropping entities with <10 observations: 85 entities remain.
   New sample size? Is this still unbalanced?

3. **Outlier Impact**
   Entity 5 has 10 observations. Year 2020 value is 10× the entity mean.
   After demeaning for FE, what happens to:
   - Entity 5's mean
   - All other years for entity 5
   - The FE coefficient estimate

4. **Winsorization Trade-off**
   Winsorize at 1% vs 5%.
   Which is more aggressive?
   Which preserves more information?
   When to use each?

5. **Sample Selection**
   Dataset: All firms 2010-2020.
   Some firms exit before 2020 (bankruptcy).
   If you study firm growth and drop firms that exit, what bias is introduced?
   How to correct?



## Further Reading

**Missing Data Theory:**
1. **"Statistical Analysis with Missing Data" by Little & Rubin** - Definitive reference
2. **"Multiple Imputation for Nonresponse in Surveys" by Rubin** - MI methodology
3. **"Missing Data in Longitudinal Studies" by Molenberghs & Kenward** - Panel-specific

**Panel Data Quality:**
4. **"Panel Data Econometrics" by Baltagi** - Chapter on data issues
5. **"Microeconometrics" by Cameron & Trivedi** - Sample selection, attrition
6. **"Econometric Analysis of Panel Data" by Hsiao** - Unbalanced panels

**Outlier Detection:**
7. **"Robust Regression and Outlier Detection" by Rousseeuw & Leroy** - Robust methods
8. **"Outliers in Statistical Data" by Barnett & Lewis** - Detection theory
9. **"Regression Diagnostics" by Belsley et al.** - Influential observations

**Applied:**
10. **"Panel Data Cleaning Best Practices"** - Practical guide (various sources)
11. **"Data Quality in Empirical Research"** - Economics-focused
12. **"Pandas for Data Cleaning"** - Implementation guide

---

*"Clean data is not the data you have, it's the data you prepare."*


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

