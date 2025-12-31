# Panel Data Concepts and Structure

## What is Panel Data?

Panel data (also called longitudinal or cross-sectional time series data) combines cross-sectional and time series dimensions:

$$y_{it} \text{ where } i = 1, ..., N \text{ (entities) and } t = 1, ..., T \text{ (time periods)}$$

### The Three Dimensions of Variation

```
                    Time (t)
                    ──────────────────────►
                    t=1    t=2    t=3    t=T
                   ┌────┬────┬────┬───┬────┐
            i=1    │y₁₁ │y₁₂ │y₁₃ │...│y₁ₜ │  Between
Entity      i=2    │y₂₁ │y₂₂ │y₂₃ │...│y₂ₜ │  Variation
(i)         i=3    │y₃₁ │y₃₂ │y₃₃ │...│y₃ₜ │     ▲
  │         ...    │... │... │... │...│... │     │
  ▼         i=N    │yₙ₁ │yₙ₂ │yₙ₃ │...│yₙₜ │     ▼
                   └────┴────┴────┴───┴────┘
                        ◄────────────────►
                         Within Variation
```

**Total Variation** = **Between Variation** + **Within Variation**

## Panel Data vs. Other Data Structures

| Structure | Cross-Sectional | Time Series | Example |
|-----------|-----------------|-------------|---------|
| Cross-sectional | Yes | No | Survey of firms at one point |
| Time series | No | Yes | GDP over 50 years |
| Pooled cross-section | Yes | Yes (different units) | CPS each year |
| **Panel data** | Yes | Yes (same units) | Same firms over 10 years |

## Why Panel Data Matters

### 1. Control for Unobserved Heterogeneity

The fundamental advantage: remove time-invariant confounders.

**Cross-sectional regression problem:**
$$y_i = \beta_0 + \beta_1 x_i + \underbrace{\alpha_i}_{\text{unobserved ability}} + \epsilon_i$$

If $\text{Corr}(\alpha_i, x_i) \neq 0$, OLS is biased.

**Panel solution (fixed effects):**
$$y_{it} - \bar{y}_i = \beta_1(x_{it} - \bar{x}_i) + (\epsilon_{it} - \bar{\epsilon}_i)$$

The entity-specific $\alpha_i$ is differenced out!

### 2. More Data, Better Precision

With $N$ entities and $T$ periods:
- Total observations: $N \times T$
- More variation to exploit
- Tighter confidence intervals

### 3. Study Dynamics

Panel data enables analysis of:
- State dependence: Does $y_{t-1}$ affect $y_t$?
- Duration effects: How long do shocks persist?
- Adjustment processes: How fast do entities converge?

## Balanced vs. Unbalanced Panels

### Balanced Panel
Every entity observed in every period:

```python
import pandas as pd

# Balanced panel
balanced = pd.DataFrame({
    'entity': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    'year': [2020, 2021, 2022, 2020, 2021, 2022, 2020, 2021, 2022],
    'y': [10, 12, 14, 20, 22, 21, 15, 17, 19]
})

# Check balance
print(balanced.groupby('entity').size())
# A    3
# B    3
# C    3
```

### Unbalanced Panel
Some entity-period combinations missing:

```python
# Unbalanced panel (firm C missing 2021)
unbalanced = pd.DataFrame({
    'entity': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C'],
    'year': [2020, 2021, 2022, 2020, 2021, 2022, 2020, 2022],
    'y': [10, 12, 14, 20, 22, 21, 15, 19]
})

# Missing value implications
# - Is missingness random (MCAR)?
# - Related to observables (MAR)?
# - Related to unobservables (MNAR)? ← Selection bias!
```

## Panel Data Notation

### The General Model

$$y_{it} = \alpha + x_{it}'\beta + u_{it}$$

where:
- $y_{it}$: outcome for entity $i$ at time $t$
- $x_{it}$: $K \times 1$ vector of regressors
- $\beta$: $K \times 1$ parameter vector
- $u_{it}$: composite error term

### Error Component Decomposition

$$u_{it} = \alpha_i + \lambda_t + \epsilon_{it}$$

| Component | Name | Interpretation |
|-----------|------|----------------|
| $\alpha_i$ | Entity effect | Time-invariant entity characteristics |
| $\lambda_t$ | Time effect | Entity-invariant time shocks |
| $\epsilon_{it}$ | Idiosyncratic error | Random variation |

## Setting Up Panel Data in Python

### Using pandas MultiIndex

```python
import pandas as pd
import numpy as np

# Create panel data
np.random.seed(42)
N, T = 100, 10  # 100 entities, 10 periods

data = pd.DataFrame({
    'entity': np.repeat(range(N), T),
    'time': np.tile(range(T), N),
    'x': np.random.randn(N * T),
    'y': np.random.randn(N * T)
})

# Set MultiIndex
panel = data.set_index(['entity', 'time'])
print(panel.head(10))

# Access specific entity
print(panel.loc[0])  # All periods for entity 0

# Access specific time
print(panel.xs(5, level='time').head())  # Period 5 for all entities
```

### Using linearmodels PanelData

```python
from linearmodels.panel import PanelData

# Create PanelData object
panel_data = PanelData(panel)

# Properties
print(f"Entities: {panel_data.nentity}")
print(f"Time periods: {panel_data.nobs / panel_data.nentity}")
print(f"Balanced: {panel_data.balanced}")
```

## Computing Panel Variation

```python
def panel_variation(panel_df, var, entity_col='entity', time_col='time'):
    """Decompose variation into between and within components."""

    # Total variation
    total_var = panel_df[var].var()

    # Between variation (variation of entity means)
    entity_means = panel_df.groupby(entity_col)[var].transform('mean')
    between_var = entity_means.var()

    # Within variation (variation around entity means)
    within_var = (panel_df[var] - entity_means).var()

    return {
        'total': total_var,
        'between': between_var,
        'within': within_var,
        'between_share': between_var / total_var,
        'within_share': within_var / total_var
    }

# Example
variation = panel_variation(data, 'y')
print(f"Between share: {variation['between_share']:.1%}")
print(f"Within share: {variation['within_share']:.1%}")
```

## Key Takeaways

1. **Panel data** combines cross-sectional and time series dimensions with the same entities observed repeatedly

2. **Variation decomposition** into between and within components is fundamental to understanding what panel methods can identify

3. **Unobserved heterogeneity** ($\alpha_i$) can be controlled with panel methods, addressing a major source of omitted variable bias

4. **Balance matters**: Unbalanced panels require attention to selection and missing data mechanisms

5. **Proper data structure** (MultiIndex in pandas, PanelData in linearmodels) is essential for analysis
