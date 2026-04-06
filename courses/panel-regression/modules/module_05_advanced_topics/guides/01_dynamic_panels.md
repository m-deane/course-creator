# Dynamic Panel Models

> **Reading time:** ~14 min | **Module:** 05 — Advanced Topics | **Prerequisites:** Module 4


## In Brief


<div class="callout-key">

**Key Concept Summary:** Dynamic panels add lagged dependent variables to the standard panel model, capturing state dependence — the idea that today's outcome depends on yesterday's. This creates a critical problem: fixed ...

</div>

Dynamic panels add lagged dependent variables to the standard panel model, capturing state dependence — the idea that today's outcome depends on yesterday's. This creates a critical problem: fixed effects estimation produces severely biased coefficients (Nickell bias) that grows worse as the time dimension shrinks. The solution is instrumental variable estimation via the Arellano-Bond or Blundell-Bond GMM estimators.

## Key Insight

<div class="callout-insight">

**Insight:** The Nickell bias is small when T is large relative to N, which is the opposite of the typical panel setting. In short panels (T < 10), the bias can be severe enough to reverse the sign of the lagged dependent variable coefficient.

</div>


> 💡 **Never apply standard fixed effects to a dynamic panel.** The bias from Nickell (1981) is not a rounding error — it can be as large as the true coefficient itself when $T$ is small. Use GMM estimators designed for this setting.

## The Dynamic Panel Framework

<div class="callout-warning">

**Warning:** Reporting results without appropriate standard errors is a common mistake. In panel data, conventional OLS standard errors are almost always wrong -- use clustered or heteroskedasticity-robust standard errors.

</div>


Dynamic panels include lagged dependent variables:

$$y_{it} = \rho y_{i,t-1} + x_{it}'\beta + \alpha_i + \epsilon_{it}$$

where $|\rho| < 1$ ensures stationarity.

### Why Dynamic Models?

Many economic relationships exhibit **state dependence**:
- Current wages depend on past wages (human capital)
- Investment depends on previous investment (adjustment costs)
- Consumption shows habit persistence
- Trade relationships are sticky

## The Nickell Bias Problem

### FE with Lagged Dependent Variable

Applying FE to dynamic panels creates bias:

$$y_{it} - \bar{y}_i = \rho(y_{i,t-1} - \bar{y}_i) + (x_{it} - \bar{x}_i)'\beta + (\epsilon_{it} - \bar{\epsilon}_i)$$

**Problem**: $(y_{i,t-1} - \bar{y}_i)$ is correlated with $(\epsilon_{it} - \bar{\epsilon}_i)$ because $\bar{y}_i$ contains $y_{i,t-1}$!

### Bias Formula

The FE estimator of $\rho$ has bias:

$$\text{plim}(\hat{\rho}_{FE} - \rho) = -\frac{1 + \rho}{T - 1} + O(T^{-2})$$

For $\rho = 0.5$ and $T = 5$:
$$\text{Bias} \approx -\frac{1.5}{4} = -0.375$$

> ⚠️ **Warning:** This is severe negative bias — applying FE to a dynamic panel with $T=5$ can bias $\hat{\rho}$ by as much as -0.375, nearly as large as the true coefficient itself. Never apply standard FE to a dynamic panel without checking for Nickell bias first.


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
import statsmodels.api as sm

def simulate_nickell_bias(N, T, rho_true, n_sims=500):
    """Demonstrate Nickell bias in dynamic panels."""
    fe_estimates = []

    for _ in range(n_sims):
        # Generate AR(1) panel
        alpha_i = np.random.randn(N)
        y = np.zeros((N, T))

        # Initial values
        y[:, 0] = alpha_i / (1 - rho_true) + np.random.randn(N)

        # Generate series
        for t in range(1, T):
            y[:, t] = rho_true * y[:, t-1] + alpha_i + np.random.randn(N)

        # Reshape to panel
        entity = np.repeat(range(N), T)
        time = np.tile(range(T), N)
        y_flat = y.flatten()
        y_lag = np.concatenate([np.zeros(N), y[:, :-1].flatten()])

        # Remove first period (no lag)
        mask = time > 0

        df = pd.DataFrame({
            'entity': entity[mask],
            'time': time[mask],
            'y': y_flat[mask],
            'y_lag': y_lag[mask]
        }).set_index(['entity', 'time'])

        # FE estimation
        try:
            fe = PanelOLS(
                df['y'],
                sm.add_constant(df['y_lag']),
                entity_effects=True
            ).fit()
            fe_estimates.append(fe.params['y_lag'])
        except:
            continue

    return {
        'mean_estimate': np.mean(fe_estimates),
        'true_value': rho_true,
        'bias': np.mean(fe_estimates) - rho_true,
        'theoretical_bias': -(1 + rho_true) / (T - 1)
    }

# Demonstrate bias
result = simulate_nickell_bias(100, 5, 0.5, n_sims=300)
print("Nickell Bias Demonstration:")
print(f"  True rho: {result['true_value']:.3f}")
print(f"  FE estimate: {result['mean_estimate']:.3f}")
print(f"  Actual bias: {result['bias']:.3f}")
print(f"  Theoretical bias: {result['theoretical_bias']:.3f}")
```


</div>
</div>

## The Anderson-Hsiao Estimator

### First Differencing

Take first differences to remove $\alpha_i$:

$$\Delta y_{it} = \rho \Delta y_{i,t-1} + \Delta x_{it}'\beta + \Delta\epsilon_{it}$$

**Problem**: $\Delta y_{i,t-1} = y_{i,t-1} - y_{i,t-2}$ still correlates with $\Delta\epsilon_{it} = \epsilon_{it} - \epsilon_{i,t-1}$ through $\epsilon_{i,t-1}$.

### Instrumental Variables Solution

Use $y_{i,t-2}$ as an instrument for $\Delta y_{i,t-1}$:
- **Relevant**: $y_{i,t-2}$ correlates with $\Delta y_{i,t-1}$
- **Valid**: $y_{i,t-2}$ is uncorrelated with $\Delta\epsilon_{it}$


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
from linearmodels.iv import IV2SLS

def anderson_hsiao(data, y_col, x_cols=None, entity_col='entity', time_col='time'):
    """
    Anderson-Hsiao IV estimator for dynamic panels.
    """
    df = data.copy()

    # Sort
    df = df.sort_values([entity_col, time_col])

    # Create differences and lags
    df['y_diff'] = df.groupby(entity_col)[y_col].diff()
    df['y_lag'] = df.groupby(entity_col)[y_col].shift(1)
    df['y_lag_diff'] = df.groupby(entity_col)['y_lag'].diff()
    df['y_lag2'] = df.groupby(entity_col)[y_col].shift(2)

    # Drop missing
    df = df.dropna()

    # Set up IV regression
    # Endogenous: Delta y_{t-1}
    # Instrument: y_{t-2}

    formula = "y_diff ~ 1 + [y_lag_diff ~ y_lag2]"

    result = IV2SLS.from_formula(formula, df).fit()

    return result

# Example usage
np.random.seed(42)
N, T = 100, 10
rho_true = 0.6

# Generate data
alpha_i = np.random.randn(N)
y = np.zeros((N, T))
y[:, 0] = alpha_i / (1 - rho_true) + np.random.randn(N)

for t in range(1, T):
    y[:, t] = rho_true * y[:, t-1] + alpha_i + np.random.randn(N)

df = pd.DataFrame({
    'entity': np.repeat(range(N), T),
    'time': np.tile(range(T), N),
    'y': y.flatten()
})

ah_result = anderson_hsiao(df, 'y')
print("Anderson-Hsiao Results:")
print(f"  rho estimate: {ah_result.params['y_lag_diff']:.4f}")
print(f"  True rho: {rho_true}")
```


</div>
</div>

## The Arellano-Bond Estimator (GMM)

### More Instruments = More Efficiency

Arellano-Bond uses all valid lags as instruments:

For period $t$: instruments are $\{y_{i,1}, y_{i,2}, ..., y_{i,t-2}\}$

Moment conditions:
$$E[y_{i,s} \cdot \Delta\epsilon_{it}] = 0 \quad \text{for } s \leq t-2$$

### Implementation


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
from linearmodels.panel import PanelOLS
from linearmodels.iv import IVGMM

def arellano_bond_simple(data, y_col, entity_col='entity', time_col='time', max_lags=None):
    """
    Simplified Arellano-Bond GMM estimator.
    """
    df = data.copy().sort_values([entity_col, time_col])

    # Create differences
    df['y_diff'] = df.groupby(entity_col)[y_col].diff()
    df['y_lag_diff'] = df.groupby(entity_col)['y_diff'].shift(1)

    # Create instrument matrix (lags of y in levels)
    T = df[time_col].nunique()
    max_lags = max_lags or T - 2

    for lag in range(2, min(max_lags + 1, T)):
        df[f'y_lag{lag}'] = df.groupby(entity_col)[y_col].shift(lag)

    # Drop missing
    df = df.dropna()

    # Instrument columns
    inst_cols = [c for c in df.columns if c.startswith('y_lag') and c != 'y_lag_diff']

    if len(inst_cols) == 0:
        raise ValueError("Not enough time periods for instruments")

    # Run 2SLS with multiple instruments
    instruments = ' + '.join(inst_cols)
    formula = f"y_diff ~ 1 + [y_lag_diff ~ {instruments}]"

    result = IV2SLS.from_formula(formula, df).fit()

    return result

# Run Arellano-Bond
ab_result = arellano_bond_simple(df, 'y')
print("\nArellano-Bond GMM Results:")
print(f"  rho estimate: {ab_result.params['y_lag_diff']:.4f}")
print(f"  True rho: {rho_true}")
```



## System GMM (Blundell-Bond)

### Adding Level Equations

Arellano-Bond uses only differenced equations. System GMM adds level equations:

**Difference equation**: $\Delta y_{it} = \rho\Delta y_{i,t-1} + \Delta\epsilon_{it}$
- Instruments: $y_{i,t-2}, y_{i,t-3}, ...$

**Level equation**: $y_{it} = \rho y_{i,t-1} + \alpha_i + \epsilon_{it}$
- Instruments: $\Delta y_{i,t-1}$

### When System GMM Helps

- Persistent series ($\rho$ close to 1): Difference GMM has weak instruments
- Short panels: More moment conditions improve efficiency
- Random effects assumption plausible: $E[\alpha_i \Delta y_{i,t-1}] = 0$

## Specification Tests

### Arellano-Bond AR Tests

Test for serial correlation in differenced residuals:

- **AR(1)**: Expected (by construction)
- **AR(2)**: Should NOT be present if model is correct

```python
def arellano_bond_ar_test(residuals, data, entity_col, time_col, order=2):
    """
    Test for serial correlation in first-differenced residuals.
    """
    from scipy import stats

    df = data.copy()
    df['resid'] = residuals
    df = df.sort_values([entity_col, time_col])

    # Lag residuals
    df['resid_lag'] = df.groupby(entity_col)['resid'].shift(order)
    df = df.dropna()

    # Regression of resid on lagged resid
    correlation = df['resid'].corr(df['resid_lag'])

    # Approximate test statistic
    n = len(df)
    z = correlation * np.sqrt(n)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        'order': order,
        'correlation': correlation,
        'z_statistic': z,
        'p_value': p_value
    }
```

### Hansen/Sargan Test

Tests overidentifying restrictions (validity of instruments):

$$J = N \cdot \bar{g}' W \bar{g} \sim \chi^2(\text{# instruments} - \text{# parameters})$$

Rejection suggests some instruments are invalid.

## Practical Guidelines

<div class="callout-danger">

**Danger:** Never include a lagged dependent variable in a fixed effects model without using an appropriate estimator (e.g., Arellano-Bond GMM). The within-transformation creates mechanical correlation between the transformed lagged variable and the transformed error, biasing all coefficients.



### Choosing the Estimator

| Situation | Recommended Estimator |
|-----------|----------------------|
| Large T, small N | FE (bias diminishes) |
| Small T, large N | GMM (Arellano-Bond) |
| Persistent series | System GMM |
| Many instruments | Limit lags, collapse instruments |

### Common Issues

1. **Too many instruments**: Leads to overfitting, weak Hansen test
2. **Weak instruments**: Large T helps; use System GMM
3. **AR(2) rejection**: Model misspecification; add more lags



## Key Takeaways

1. **Dynamic panels** with lagged dependent variables create endogeneity

2. **FE is biased** with fixed T (Nickell bias): $-\frac{1+\rho}{T-1}$

3. **Anderson-Hsiao** uses $y_{t-2}$ as IV for $\Delta y_{t-1}$

4. **Arellano-Bond** GMM uses all valid lags as instruments

5. **System GMM** adds level equations for persistent series

6. **Specification tests**: AR(2) and Hansen tests are essential


---

## Conceptual Practice Questions

**Practice Question 1:** Why does including a lagged dependent variable in a fixed effects model create bias, and in which direction?

**Practice Question 2:** How does the Arellano-Bond GMM estimator address the Nickell bias problem?


---

## Cross-References

<a class="link-card" href="./02_nickell_bias.md">
  <div class="link-card-title">02 Nickell Bias</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_nickell_bias.md">
  <div class="link-card-title">02 Nickell Bias — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./03_clustered_standard_errors.md">
  <div class="link-card-title">03 Clustered Standard Errors</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_clustered_standard_errors.md">
  <div class="link-card-title">03 Clustered Standard Errors — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

