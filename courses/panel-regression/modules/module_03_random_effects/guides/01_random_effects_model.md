# Random Effects Model

> **Reading time:** ~12 min | **Module:** 03 — Random Effects | **Prerequisites:** Module 2


## The Random Effects Framework

<div class="flow">
<div class="flow-step mint">1. Estimate Variance Components</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">2. Compute GLS Weights</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">3. Quasi-Demean</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. Run GLS</div>
</div>


<div class="callout-key">

**Key Concept Summary:** The Random Effects (RE) model treats entity effects as random draws from a distribution:

</div>

The Random Effects (RE) model treats entity effects as random draws from a distribution:

$$y_{it} = \beta_0 + x_{it}'\beta + \alpha_i + \epsilon_{it}$$

where:
- $\alpha_i \sim (0, \sigma_\alpha^2)$ — random entity effects
- $\epsilon_{it} \sim (0, \sigma_\epsilon^2)$ — idiosyncratic errors
- $\alpha_i \perp \epsilon_{it}$ — effects independent of errors

### The Critical Assumption

**Random effects requires:**
$$E[\alpha_i | x_{i1}, x_{i2}, ..., x_{iT}] = 0$$

The entity effects must be **uncorrelated with all regressors**.

## Comparing FE and RE

<div class="callout-insight">

**Insight:** Random effects assumes the unobserved entity effect is uncorrelated with regressors. When this holds, RE is more efficient than FE because it uses both within- and between-entity variation.

</div>


| Aspect | Fixed Effects | Random Effects |
|--------|--------------|----------------|
| Entity effects | Parameters to estimate | Random variables |
| Time-invariant vars | Cannot estimate | Can estimate |
| Efficiency | Less efficient | More efficient |
| Key assumption | None on $\alpha_i, x_{it}$ correlation | $\alpha_i \perp x_{it}$ |
| Degrees of freedom | Loses $N-1$ df | Doesn't lose df |

## The GLS Transformation

<div class="callout-warning">

**Warning:** Reporting results without appropriate standard errors is a common mistake. In panel data, conventional OLS standard errors are almost always wrong -- use clustered or heteroskedasticity-robust standard errors.

</div>


### Error Structure

The composite error $u_{it} = \alpha_i + \epsilon_{it}$ has covariance:

$$\Omega_i = E[u_i u_i'] = \sigma_\epsilon^2 I_T + \sigma_\alpha^2 \mathbf{1}_T \mathbf{1}_T'$$

where $\mathbf{1}_T$ is a $T \times 1$ vector of ones.

### The Quasi-Demeaning Transformation

RE uses a partial demeaning:

$$y_{it} - \theta\bar{y}_i = \beta_0(1-\theta) + (x_{it} - \theta\bar{x}_i)'\beta + (u_{it} - \theta\bar{u}_i)$$

where:
$$\theta = 1 - \sqrt{\frac{\sigma_\epsilon^2}{\sigma_\epsilon^2 + T\sigma_\alpha^2}}$$

### Interpretation of $\theta$

- $\theta = 0$: No demeaning → Pooled OLS
- $\theta = 1$: Full demeaning → Fixed Effects
- $0 < \theta < 1$: Partial demeaning → Random Effects

As $\sigma_\alpha^2 \to \infty$ or $T \to \infty$, $\theta \to 1$ (RE approaches FE).

## Implementation in Python


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import RandomEffects, PanelOLS

# Generate data with uncorrelated effects (RE appropriate)
np.random.seed(42)
N, T = 100, 10

alpha_i = np.repeat(np.random.randn(N), T)  # Entity effects
x = np.random.randn(N * T)  # x UNCORRELATED with alpha (important!)
epsilon = np.random.randn(N * T)
y = 2 + 1.5 * x + alpha_i + epsilon

data = pd.DataFrame({
    'entity': np.repeat(range(N), T),
    'time': np.tile(range(T), N),
    'y': y,
    'x': x
}).set_index(['entity', 'time'])

# Fit Random Effects model
re_model = RandomEffects(data['y'], sm.add_constant(data['x']))
re_results = re_model.fit()

print("Random Effects Results:")
print(re_results.summary.tables[1])

# Compare with Fixed Effects
fe_model = PanelOLS(data['y'], sm.add_constant(data['x']), entity_effects=True)
fe_results = fe_model.fit()

print("\nFixed Effects Results:")
print(fe_results.summary.tables[1])

print(f"\nTheta (quasi-demeaning parameter): {re_results.theta.iloc[0]:.4f}")
```


</div>
</div>

## Estimating Variance Components

### Method of Moments


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def estimate_variance_components(data, y_col, x_cols, entity_col, time_col):
    """
    Estimate sigma_alpha^2 and sigma_epsilon^2 using method of moments.
    """
    from linearmodels.panel import PanelOLS
    import statsmodels.api as sm

    # Fit FE model to get residuals
    panel = data.set_index([entity_col, time_col])
    y = panel[y_col]
    X = sm.add_constant(panel[x_cols])

    fe_model = PanelOLS(y, X, entity_effects=True).fit()

    # Within residuals
    resid_within = fe_model.resids.values

    # Degrees of freedom
    N = data[entity_col].nunique()
    T = data[time_col].nunique()
    K = len(x_cols)

    # Sigma_epsilon^2 from within residuals
    sigma2_epsilon = (resid_within @ resid_within) / (N * T - N - K)

    # Between residuals
    entity_means = data.groupby(entity_col)[[y_col] + x_cols].mean()
    y_between = entity_means[y_col]
    X_between = sm.add_constant(entity_means[x_cols])

    between_model = sm.OLS(y_between, X_between).fit()
    resid_between = between_model.resid.values
    sigma2_between = (resid_between @ resid_between) / (N - K - 1)

    # Sigma_alpha^2
    sigma2_alpha = max(0, sigma2_between - sigma2_epsilon / T)

    # Theta
    theta = 1 - np.sqrt(sigma2_epsilon / (sigma2_epsilon + T * sigma2_alpha))

    return {
        'sigma2_epsilon': sigma2_epsilon,
        'sigma2_alpha': sigma2_alpha,
        'theta': theta,
        'icc': sigma2_alpha / (sigma2_alpha + sigma2_epsilon)
    }

# Estimate
data_flat = data.reset_index()
var_components = estimate_variance_components(
    data_flat, 'y', ['x'], 'entity', 'time'
)
print("Variance Components:")
for k, v in var_components.items():
    print(f"  {k}: {v:.4f}")
```


</div>
</div>

## Time-Invariant Variables

A key advantage of RE: estimating effects of time-invariant variables.


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Add time-invariant variable
data_flat = data.reset_index()
data_flat['gender'] = np.repeat(np.random.choice([0, 1], N), T)

# FE cannot estimate gender effect

# RE can!
data_panel = data_flat.set_index(['entity', 'time'])

re_with_invariant = RandomEffects(
    data_panel['y'],
    sm.add_constant(data_panel[['x', 'gender']])
).fit()

print("RE with Time-Invariant Variable:")
print(re_with_invariant.summary.tables[1])
```



## The Mundlak Approach

Add entity means to control for correlation:

$$y_{it} = \beta_0 + x_{it}'\beta + \bar{x}_i'\gamma + \alpha_i + \epsilon_{it}$$

This makes RE robust to correlation between $\alpha_i$ and $x_{it}$.

```python

# Mundlak correction
data_flat = data.reset_index()
data_flat['x_mean'] = data_flat.groupby('entity')['x'].transform('mean')

data_mundlak = data_flat.set_index(['entity', 'time'])

re_mundlak = RandomEffects(
    data_mundlak['y'],
    sm.add_constant(data_mundlak[['x', 'x_mean']])
).fit()

print("Random Effects with Mundlak Correction:")
print(re_mundlak.summary.tables[1])

# Beta on x should match FE if correlation exists
print(f"\nRE beta (x): {re_mundlak.params['x']:.4f}")
print(f"FE beta (x): {fe_results.params['x']:.4f}")
```

## When to Use Random Effects

### RE is Appropriate When:

1. **Entities are random sample** from larger population
2. **No correlation** between entity effects and regressors
3. **Interest in time-invariant** variable effects
4. **Efficiency matters** (small samples)

### RE is Inappropriate When:

1. **Selection on unobservables** (e.g., ability bias)
2. **Entities are not exchangeable** (specific countries, not random sample)
3. **Policy variables correlated** with unobserved heterogeneity

## Advantages of Random Effects

<div class="callout-danger">

**Danger:** Never include a lagged dependent variable in a fixed effects model without using an appropriate estimator (e.g., Arellano-Bond GMM). The within-transformation creates mechanical correlation between the transformed lagged variable and the transformed error, biasing all coefficients.



| Advantage | Explanation |
|-----------|-------------|
| Efficiency | Uses both within and between variation |
| Time-invariant variables | Can estimate coefficients |
| Fewer parameters | Doesn't estimate $N-1$ dummies |
| Extrapolation | Can predict for new entities |

## Summary

The random effects model provides efficient estimation when entity effects are uncorrelated with regressors:

$$y_{it} - \theta\bar{y}_i = (1-\theta)\beta_0 + (x_{it} - \theta\bar{x}_i)'\beta + \text{error}$$

Key points:
- RE is a **weighted average** of between and within estimators
- The weight $\theta$ depends on **variance components**
- RE requires **strict exogeneity** plus **orthogonality** of effects
- Use **Mundlak correction** if correlation is suspected
- **Hausman test** compares FE and RE to detect violations


---

## Conceptual Practice Questions

**Practice Question 1:** What is the key assumption that distinguishes random effects from fixed effects, and when is it likely to be violated?

**Practice Question 2:** Why does random effects estimation produce more efficient estimates than fixed effects when its assumptions hold?


---

## Cross-References

<a class="link-card" href="./02_random_effects_assumptions.md">
  <div class="link-card-title">02 Random Effects Assumptions</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_random_effects_assumptions.md">
  <div class="link-card-title">02 Random Effects Assumptions — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./03_correlated_random_effects.md">
  <div class="link-card-title">03 Correlated Random Effects</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_correlated_random_effects.md">
  <div class="link-card-title">03 Correlated Random Effects — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

