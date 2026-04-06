# LSDV vs Within Transformation: Two Approaches to Fixed Effects

> **Reading time:** ~20 min | **Module:** 02 — Fixed Effects | **Prerequisites:** Module 1


## Overview

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

**Key Concept Summary:** Both give identical coefficient estimates, but differ in computation and what they reveal.

</div>

Fixed effects can be estimated two ways:
1. **LSDV (Least Squares Dummy Variables)**: Include entity dummies explicitly
2. **Within Transformation**: Demean data to remove entity effects

Both give identical coefficient estimates, but differ in computation and what they reveal.

## The LSDV Approach

<div class="compare">
  <div class="compare-card">
    <div class="header before">LSDV (Dummy Variable)</div>
    <div class="body">
      Include N-1 entity dummies explicitly. Estimates entity intercepts directly. Fails with large N (many parameters).
    </div>
  </div>
  <div class="compare-card">
    <div class="header after">Within Estimator</div>
    <div class="body">
      Demean by entity. Numerically identical slopes to LSDV. Efficient for large N. Does not estimate entity intercepts.
    </div>
  </div>
</div>

### Concept

Include dummy variables for each entity (minus one reference category):

$$y_{it} = \alpha + \sum_{j=2}^{N} \delta_j D_{ij} + X_{it}\beta + \epsilon_{it}$$

where $D_{ij} = 1$ if $i = j$, else 0.

### Implementation


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS

# Create sample data
np.random.seed(42)
n_entities = 50
n_periods = 10

# Generate panel data with entity effects
data = []
entity_effects = np.random.normal(0, 3, n_entities)

for i in range(n_entities):
    for t in range(n_periods):
        x = np.random.normal(5, 2) + 0.3 * entity_effects[i]  # x correlated with entity effect
        y = 2 + 1.5 * x + entity_effects[i] + np.random.normal(0, 1)
        data.append({'entity': i, 'time': t, 'x': x, 'y': y})

df = pd.DataFrame(data)

<div class="callout-insight">

**Insight:** Fixed effects are not a method -- they are a way of thinking about unobserved heterogeneity. The within-transformation eliminates time-invariant confounders, which is the single most important advantage of panel data.

</div>


# LSDV: Include entity dummies explicitly
lsdv_model = smf.ols('y ~ x + C(entity)', data=df).fit()

print("LSDV Results:")
print(f"  x coefficient: {lsdv_model.params['x']:.4f}")
print(f"  x std error: {lsdv_model.bse['x']:.4f}")
print(f"  R-squared: {lsdv_model.rsquared:.4f}")
print(f"  Number of parameters: {len(lsdv_model.params)}")
```


</div>

### Extracting Entity Effects


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# Extract entity fixed effects from LSDV
entity_dummies = [col for col in lsdv_model.params.index if 'entity' in col]
entity_effects_estimated = lsdv_model.params[entity_dummies]

# Reference category effect (absorbed in intercept)
# Entity 0 is reference, its effect = intercept - grand mean of other effects
reference_effect = lsdv_model.params['Intercept']

print(f"\nEstimated entity effects (first 5):")
for i, effect in enumerate(entity_effects_estimated[:5]):
    print(f"  Entity {i+1}: {effect:.4f}")
```


</div>

## The Within Transformation

<div class="callout-warning">

**Warning:** Fixed effects estimates identify only from within-entity variation. If your variable of interest has little within-entity variation (e.g., industry sector), fixed effects will produce large standard errors or fail entirely.

</div>


### Concept

Subtract entity means from all variables:

$$\tilde{y}_{it} = y_{it} - \bar{y}_i$$
$$\tilde{X}_{it} = X_{it} - \bar{X}_i$$

Then estimate: $\tilde{y}_{it} = \tilde{X}_{it}\beta + \tilde{\epsilon}_{it}$

### Implementation


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# Manual within transformation
df['y_mean'] = df.groupby('entity')['y'].transform('mean')
df['x_mean'] = df.groupby('entity')['x'].transform('mean')

df['y_demean'] = df['y'] - df['y_mean']
df['x_demean'] = df['x'] - df['x_mean']

# Within estimator: OLS on demeaned data
within_manual = smf.ols('y_demean ~ x_demean - 1', data=df).fit()  # No intercept!

print("\nWithin Transformation (Manual):")
print(f"  x coefficient: {within_manual.params['x_demean']:.4f}")
print(f"  x std error (unadjusted): {within_manual.bse['x_demean']:.4f}")
```


</div>

### Using linearmodels

```python
# Using linearmodels PanelOLS
df_panel = df.set_index(['entity', 'time'])

fe_model = PanelOLS(df_panel['y'], df_panel[['x']], entity_effects=True)
fe_results = fe_model.fit()

print("\nWithin Transformation (linearmodels):")
print(f"  x coefficient: {fe_results.params['x']:.4f}")
print(f"  x std error (clustered): {fe_results.std_errors['x']:.4f}")
```

## Comparing the Two Approaches

### Coefficient Equivalence

```python
print("\n" + "="*50)
print("COEFFICIENT COMPARISON")
print("="*50)
print(f"LSDV coefficient:   {lsdv_model.params['x']:.6f}")
print(f"Within coefficient: {within_manual.params['x_demean']:.6f}")
print(f"Difference: {abs(lsdv_model.params['x'] - within_manual.params['x_demean']):.10f}")
```

### Why Coefficients Are Identical

Mathematically, both methods solve the same normal equations. The within transformation algebraically eliminates the entity dummies.

Proof outline:
1. LSDV partitioned regression: regress y on X after partialling out dummies
2. Partialling out dummies = demeaning within entities
3. Therefore: $\hat{\beta}_{LSDV} = \hat{\beta}_{Within}$

### Standard Error Differences

```python
# Standard errors differ!
print("\n" + "="*50)
print("STANDARD ERROR COMPARISON")
print("="*50)
print(f"LSDV SE:                    {lsdv_model.bse['x']:.6f}")
print(f"Within SE (unadjusted):     {within_manual.bse['x_demean']:.6f}")
print(f"Within SE (DF adjusted):    {fe_results.std_errors['x']:.6f}")
```

The within transformation loses degrees of freedom ($N-1$ for entity effects), which must be corrected:

$$SE_{adjusted} = SE_{unadjusted} \times \sqrt{\frac{NT - K}{NT - N - K}}$$

```python
# Manual degrees of freedom correction
N = df['entity'].nunique()
T = df['time'].nunique()
K = 1  # One regressor

df_correction = np.sqrt((N*T - K) / (N*T - N - K))
within_se_corrected = within_manual.bse['x_demean'] * df_correction

print(f"\nManually corrected SE: {within_se_corrected:.6f}")
```

## Advantages and Disadvantages

### LSDV Advantages

| Advantage | Explanation |
|-----------|-------------|
| Entity effects visible | Can examine, test, plot individual effects |
| Standard output | Regular OLS output, easy interpretation |
| Flexible | Can add interactions with entity dummies |

### LSDV Disadvantages

| Disadvantage | Explanation |
|--------------|-------------|
| Computational | Many parameters for large N |
| Memory intensive | Stores full dummy matrix |
| Incidental parameters | Many effects = potential bias |

### Within Transformation Advantages

| Advantage | Explanation |
|-----------|-------------|
| Computationally efficient | Only K parameters |
| Scales well | Works with millions of entities |
| Clean output | Focus on coefficients of interest |

### Within Transformation Disadvantages

| Disadvantage | Explanation |
|--------------|-------------|
| Entity effects not estimated | Must compute separately if needed |
| DF adjustment needed | SEs need correction |
| Less flexible | Harder to add entity-specific slopes |

## When to Use Each

### Use LSDV When:

```python
# 1. You need entity effects
# 2. N is small (< 100 entities)
# 3. You want interactions with entity

# Example: Entity-specific slopes
lsdv_slopes = smf.ols('y ~ x * C(entity)', data=df.head(100)).fit()  # Only first 100 obs
print("Entity-specific slopes model estimated")
```

### Use Within Transformation When:

```python
# 1. Large N (hundreds or thousands of entities)
# 2. Only care about coefficients, not effects
# 3. Computational efficiency matters

# Example: Large panel
# Within transformation handles this easily
large_df = pd.concat([df.assign(entity=df['entity'] + i*n_entities) for i in range(100)])
large_df = large_df.set_index(['entity', 'time'])

fe_large = PanelOLS(large_df['y'], large_df[['x']], entity_effects=True).fit()
print(f"Estimated with {large_df.index.get_level_values('entity').nunique()} entities")
```

## Recovering Entity Effects After Within Estimation

```python
def recover_entity_effects(df, entity_col, y_col, x_cols, beta):
    """
    Recover entity fixed effects after within estimation.

    alpha_i = y_bar_i - X_bar_i @ beta
    """
    effects = {}

    for entity in df[entity_col].unique():
        entity_data = df[df[entity_col] == entity]
        y_mean = entity_data[y_col].mean()
        x_means = entity_data[x_cols].mean()

        effect = y_mean - (x_means @ beta)
        effects[entity] = effect

    return pd.Series(effects)

# Recover effects
beta = np.array([fe_results.params['x']])
entity_effects_recovered = recover_entity_effects(df, 'entity', 'y', ['x'], beta)

print("\nRecovered entity effects (first 5):")
print(entity_effects_recovered.head())
```

## Practical Recommendations

<div class="callout-danger">

**Danger:** Never include a lagged dependent variable in a fixed effects model without using an appropriate estimator (e.g., Arellano-Bond GMM). The within-transformation creates mechanical correlation between the transformed lagged variable and the transformed error, biasing all coefficients.

</div>


1. **For most panel analysis**: Use `linearmodels.PanelOLS` with `entity_effects=True`
   - Handles DF correction automatically
   - Efficient computation
   - Easy clustering

2. **When you need entity effects**: Either LSDV or recover after within

3. **For very large panels**: Always use within transformation

4. **For standard errors**: Always cluster by entity

```python
# Recommended approach
final_model = PanelOLS(
    df_panel['y'],
    df_panel[['x']],
    entity_effects=True
).fit(cov_type='clustered', cluster_entity=True)

print("\nRecommended specification:")
print(final_model.summary.tables[1])
```

## Key Takeaways

1. **LSDV and within give identical coefficients** - they solve the same problem

2. **Standard errors differ** due to degrees of freedom correction

3. **Use within for large N** - LSDV becomes impractical

4. **Entity effects can be recovered** after within estimation if needed

5. **Always cluster standard errors** regardless of method


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

<a class="link-card" href="./03_two_way_fixed_effects.md">
  <div class="link-card-title">03 Two Way Fixed Effects</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_two_way_fixed_effects.md">
  <div class="link-card-title">03 Two Way Fixed Effects — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

