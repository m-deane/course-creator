# Pooled OLS and Its Limitations

> **Reading time:** ~12 min | **Module:** 01 — Panel Structure | **Prerequisites:** Module 0 Foundations


## The Pooled OLS Approach


<div class="callout-key">

**Key Concept Summary:** Pooled OLS ignores the panel structure entirely, treating all observations as independent:

</div>

Pooled OLS ignores the panel structure entirely, treating all observations as independent:

$$y_{it} = \beta_0 + x_{it}'\beta + \epsilon_{it}$$

### Implementation


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PooledOLS

# Generate panel data
np.random.seed(42)
N, T = 100, 10
entity = np.repeat(range(N), T)
time = np.tile(range(T), N)

# True model: y = 2 + 1.5*x + entity_effect + error
alpha_i = np.repeat(np.random.randn(N), T)  # Entity fixed effects
x = np.random.randn(N * T) + 0.5 * alpha_i  # x correlated with entity effect!
epsilon = np.random.randn(N * T)
y = 2 + 1.5 * x + alpha_i + epsilon

# Create DataFrame
data = pd.DataFrame({
    'entity': entity,
    'time': time,
    'y': y,
    'x': x
}).set_index(['entity', 'time'])

# Method 1: statsmodels (ignores panel structure)
X_pooled = sm.add_constant(data['x'])
pooled_sm = sm.OLS(data['y'], X_pooled).fit()
print("Pooled OLS (statsmodels):")
print(f"  x coefficient: {pooled_sm.params['x']:.4f}")
print(f"  True value: 1.5")

# Method 2: linearmodels PooledOLS
pooled_lm = PooledOLS(data['y'], sm.add_constant(data['x'])).fit()
print(f"\nPooled OLS (linearmodels):")
print(pooled_lm.summary.tables[1])
```


</div>
</div>

### What Pooled OLS Assumes

For consistency, pooled OLS requires:

$$E[\epsilon_{it} | x_{i1}, x_{i2}, ..., x_{iT}] = 0$$

This is the **strict exogeneity** assumption—all regressors uncorrelated with all errors.

## The Omitted Variable Problem

### The Composite Error

When entity effects exist:
$$y_{it} = \beta_0 + x_{it}'\beta + \underbrace{\alpha_i + \epsilon_{it}}_{u_{it}}$$

The error $u_{it}$ contains:
- $\alpha_i$: Entity-specific, time-invariant component
- $\epsilon_{it}$: Idiosyncratic, random component

### Correlation Creates Bias

If $\text{Cov}(x_{it}, \alpha_i) \neq 0$:

<div class="callout-insight">

**Insight:** Panel data lets you control for unobservable differences between entities that are constant over time. This is the single most important reason to prefer panel data over repeated cross-sections.

</div>


$$\hat{\beta}_{pooled} \xrightarrow{p} \beta + \underbrace{\frac{\text{Cov}(x_{it}, \alpha_i)}{\text{Var}(x_{it})}}_{\text{Omitted Variable Bias}}$$

**Example: Returns to Education**

| Variable | Interpretation |
|----------|----------------|
| $y_{it}$ | Log wages |
| $x_{it}$ | Years of education |
| $\alpha_i$ | Innate ability (unobserved) |

Ability correlates with both education and wages → Pooled OLS overestimates returns to education.

## Serial Correlation in Errors

<div class="callout-warning">

**Warning:** Reporting results without appropriate standard errors is a common mistake. In panel data, conventional OLS standard errors are almost always wrong -- use clustered or heteroskedasticity-robust standard errors.

</div>


Even if $\text{Cov}(x_{it}, \alpha_i) = 0$, the composite error has structure:

### Error Covariance

For observations within the same entity:
$$\text{Cov}(u_{it}, u_{is}) = \text{Cov}(\alpha_i + \epsilon_{it}, \alpha_i + \epsilon_{is}) = \sigma_\alpha^2$$

For observations across entities:
$$\text{Cov}(u_{it}, u_{js}) = 0 \text{ for } i \neq j$$

### Intraclass Correlation

$$\rho = \frac{\sigma_\alpha^2}{\sigma_\alpha^2 + \sigma_\epsilon^2}$$

This measures the proportion of variance due to entity effects.


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def estimate_intraclass_correlation(data, y_col, entity_col):
    """Estimate intraclass correlation from panel data."""

    # Between variance
    entity_means = data.groupby(entity_col)[y_col].mean()
    between_var = entity_means.var()

    # Within variance
    within_var = data.groupby(entity_col)[y_col].var().mean()

    # ICC
    rho = between_var / (between_var + within_var)
    return rho

data_flat = data.reset_index()
rho = estimate_intraclass_correlation(data_flat, 'y', 'entity')
print(f"Intraclass correlation: {rho:.3f}")
```


</div>
</div>

### Impact on Standard Errors

Pooled OLS standard errors assume independent observations. With serial correlation:

$$\text{True } SE(\hat{\beta}) > \text{Reported } SE(\hat{\beta})$$

This leads to:
- **Overstated t-statistics**
- **False rejections** of null hypotheses
- **Invalid confidence intervals**

## Clustered Standard Errors

### The Solution for Inference

Cluster standard errors at the entity level to account for within-entity correlation:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Pooled OLS with clustered standard errors
from linearmodels.panel import PooledOLS

# Cluster at entity level
pooled_clustered = PooledOLS(
    data['y'],
    sm.add_constant(data['x'])
).fit(cov_type='clustered', cluster_entity=True)

print("Pooled OLS with Clustered SE:")
print(pooled_clustered.summary.tables[1])

# Compare standard errors
print(f"\nHomoskedastic SE: {pooled_lm.std_errors['x']:.4f}")
print(f"Clustered SE: {pooled_clustered.std_errors['x']:.4f}")
```



### The Sandwich Estimator

Clustered standard errors use the sandwich formula:

$$\hat{V}(\hat{\beta}) = (X'X)^{-1} \left( \sum_{i=1}^N X_i' \hat{u}_i \hat{u}_i' X_i \right) (X'X)^{-1}$$

where $X_i$ and $\hat{u}_i$ are stacked for each entity.

## When is Pooled OLS Appropriate?

### Valid Use Cases

1. **No entity effects**: $\sigma_\alpha^2 = 0$
2. **Uncorrelated effects**: $\text{Cov}(x_{it}, \alpha_i) = 0$
3. **Baseline comparison**: To show bias relative to FE/RE
4. **Random assignment**: In experimental settings

### Diagnostic Tests

```python
def test_pooled_validity(data, y_col, x_cols, entity_col):
    """
    Test whether pooled OLS is appropriate.
    """
    import statsmodels.formula.api as smf

    # Breusch-Pagan LM test for random effects
    # H0: Var(alpha_i) = 0 (pooled OLS is fine)

    # Fit pooled model
    formula = f"{y_col} ~ {' + '.join(x_cols)}"
    pooled = smf.ols(formula, data=data.reset_index()).fit()

    # Get residuals
    resid = pooled.resid.values

    # Reshape to panel
    n_entities = data.reset_index()[entity_col].nunique()
    n_time = len(resid) // n_entities
    resid_panel = resid.reshape(n_entities, n_time)

    # LM statistic
    sum_T_resid = resid_panel.sum(axis=1)
    LM = (n_entities * n_time / (2 * (n_time - 1))) * \
         (sum_T_resid @ sum_T_resid / (resid @ resid) - 1)**2

    from scipy import stats
    p_value = 1 - stats.chi2.cdf(LM, 1)

    return {'LM_statistic': LM, 'p_value': p_value}

# Run test
result = test_pooled_validity(data, 'y', ['x'], 'entity')
print(f"LM statistic: {result['LM_statistic']:.2f}")
print(f"P-value: {result['p_value']:.4f}")
if result['p_value'] < 0.05:
    print("Reject H0: Entity effects exist. Pooled OLS inappropriate.")
```

## Summary: Pooled OLS Limitations

<div class="callout-danger">

**Danger:** Never include a lagged dependent variable in a fixed effects model without using an appropriate estimator (e.g., Arellano-Bond GMM). The within-transformation creates mechanical correlation between the transformed lagged variable and the transformed error, biasing all coefficients.



| Issue | Consequence | Solution |
|-------|-------------|----------|
| Omitted entity effects | Biased coefficients | Fixed Effects |
| Correlated entity effects | Biased coefficients | Fixed Effects |
| Serial correlation | Wrong standard errors | Clustered SE |
| Heteroskedasticity | Wrong standard errors | Robust SE |

## Key Takeaways

1. **Pooled OLS ignores panel structure**, treating all observations as independent draws

2. **Entity effects create bias** when correlated with regressors—the classic omitted variable bias

3. **Serial correlation** within entities invalidates standard errors even without bias

4. **Clustered standard errors** fix inference but not bias

5. **Testing is essential**: Use LM tests to detect entity effects before choosing a model


---

## Conceptual Practice Questions

**Practice Question 1:** Why does pooled OLS produce biased estimates when there is unobserved heterogeneity correlated with the regressors?

**Practice Question 2:** Under what conditions is pooled OLS actually the correct estimator for panel data?


---

## Cross-References

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

<a class="link-card" href="./03_between_within_decomposition.md">
  <div class="link-card-title">03 Between Within Decomposition</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_between_within_decomposition.md">
  <div class="link-card-title">03 Between Within Decomposition — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

