# Panel Data Concepts and Structure

## In Brief

Panel data (also called longitudinal or cross-sectional time series data) tracks the same entities across multiple time periods. It provides two dimensions of variation simultaneously, enabling identification strategies that pure cross-sectional or time-series data cannot support.

## Key Insight

The defining feature of panel data is the double subscript $y_{it}$: the same entity $i$ appears at multiple time periods $t$. This repeated observation creates two distinct sources of variation — **between** (differences across entities) and **within** (changes over time for each entity). Choosing which source of variation to exploit drives every estimation decision in panel econometrics.

## Formal Definition

A panel dataset consists of observations:

$$\{(y_{it}, x_{it}) : i = 1, \ldots, N; \; t = 1, \ldots, T_i\}$$

where:
- $i$ indexes entities (firms, individuals, countries, etc.)
- $t$ indexes time periods
- $T_i$ is the number of observed periods for entity $i$
- When $T_i = T$ for all $i$, the panel is **balanced**; otherwise it is **unbalanced**

The general one-way error component model is:

$$y_{it} = \alpha + x_{it}'\beta + u_{it}$$

where the composite error decomposes as:

$$u_{it} = \alpha_i + \lambda_t + \epsilon_{it}$$

| Component | Name | Interpretation |
|-----------|------|----------------|
| $\alpha_i$ | Entity effect | Time-invariant, entity-specific characteristics |
| $\lambda_t$ | Time effect | Entity-invariant shocks in period $t$ |
| $\epsilon_{it}$ | Idiosyncratic error | Random variation unexplained by either |

The **total variation** in any variable decomposes exactly into:

$$\text{Var}(y_{it}) = \underbrace{\text{Var}(\bar{y}_i)}_{\text{between}} + \underbrace{\text{Var}(y_{it} - \bar{y}_i)}_{\text{within}}$$

where $\bar{y}_i = T^{-1}\sum_t y_{it}$ is the entity mean.

## Intuitive Explanation

Think of panel data as a spreadsheet with three dimensions: entities (rows), time periods (columns), and variables (sheets). You can ask two fundamentally different questions:

1. **Between question:** Do entities with more of $X$ tend to have more of $Y$? (compare different rows)
2. **Within question:** When an entity gets more of $X$, does its $Y$ go up? (compare a row to itself over time)

The within question is more credible for causal inference because comparing an entity to itself over time eliminates all stable differences between entities — whether you measured them or not. A firm's culture, a person's innate ability, a country's geography: all are constant and get absorbed into $\alpha_i$.

**Analogy:** Suppose you want to know whether exercise improves health. A cross-section compares exercisers to non-exercisers today — but they may differ in income, diet, and genetics. Panel data lets you follow the same people over time, comparing each person's health when they exercise more versus less. This controls for all their stable characteristics.

## Code Implementation

```python
import pandas as pd
import numpy as np
from linearmodels.panel import PanelData

np.random.seed(42)

# --- Simulate a balanced panel: 100 firms, 10 years ---
N, T = 100, 10
entity_ids = np.repeat(np.arange(N), T)
time_ids = np.tile(np.arange(T), N)

# True model: y_it = alpha_i + 1.5*x_it + epsilon_it
# alpha_i is firm fixed effect correlated with x_it
alpha_i = np.random.randn(N)  # entity effects
alpha_broadcast = np.repeat(alpha_i, T)

x_it = 0.5 * alpha_broadcast + np.random.randn(N * T)  # x correlated with alpha
y_it = alpha_broadcast + 1.5 * x_it + np.random.randn(N * T) * 0.5

df = pd.DataFrame({
    'firm': entity_ids,
    'year': time_ids,
    'x': x_it,
    'y': y_it
})

# --- Variation decomposition ---
def decompose_variation(df, entity_col, variable):
    """Decompose total variance into between and within components."""
    total_var = df[variable].var()

    # Between variation: variance of entity means
    entity_means = df.groupby(entity_col)[variable].transform('mean')
    between_var = entity_means.var()

    # Within variation: variance of demeaned values
    within_var = (df[variable] - entity_means).var()

    return {
        'total': total_var,
        'between': between_var,
        'within': within_var,
        'between_share': between_var / total_var,
        'within_share': within_var / total_var
    }

stats = decompose_variation(df, 'firm', 'y')
print(f"Total variance:   {stats['total']:.3f}")
print(f"Between variance: {stats['between']:.3f} ({stats['between_share']:.1%})")
print(f"Within variance:  {stats['within']:.3f} ({stats['within_share']:.1%})")

# --- Set proper panel index for linearmodels ---
panel = df.set_index(['firm', 'year'])
pdata = PanelData(panel)

print(f"\nEntities: {pdata.nentity}")
print(f"Time periods: {pdata.nobs // pdata.nentity}")
print(f"Balanced: {pdata.balanced}")
```

**Expected output:** You will see that both between and within variation are substantial, and the panel is balanced.

```python
# --- Balanced vs unbalanced panel check ---
def check_panel_balance(df, entity_col, time_col):
    """Diagnose panel balance and report missingness."""
    n_entities = df[entity_col].nunique()
    n_periods = df[time_col].nunique()
    expected = n_entities * n_periods
    actual = len(df)

    obs_per_entity = df.groupby(entity_col).size()

    return {
        'is_balanced': actual == expected,
        'n_entities': n_entities,
        'n_periods': n_periods,
        'completeness': actual / expected,
        'min_obs_entity': obs_per_entity.min(),
        'max_obs_entity': obs_per_entity.max()
    }

balance = check_panel_balance(df, 'firm', 'year')
print(balance)
```

## Common Pitfalls

**Confusing panel data with pooled cross-sections.** A pooled cross-section draws different units each period (e.g., a new random sample each year). Panel data tracks the *same* units. They look similar structurally but are fundamentally different: pooled cross-sections cannot identify within-entity dynamics.

**Ignoring unbalancedness.** Unbalanced panels are not just a nuisance — if missingness is related to the outcome (Missing Not At Random), your estimators are biased. Always investigate *why* observations are missing.

**Treating entity ID as a numeric variable.** If your entity IDs are integers, pandas may try to use them in arithmetic. Always confirm the ID column is treated as a categorical identifier, not a regressor.

**Forgetting to sort before computing lags.** Lag operations using `groupby().shift()` assume data is sorted by entity then time. Unsorted data produces wrong lags silently — no error is raised.

```python
# Always sort before creating lags
df = df.sort_values(['firm', 'year']).reset_index(drop=True)
df['y_lag'] = df.groupby('firm')['y'].shift(1)
```

**Assuming balanced when data is unbalanced.** Some estimators make the balanced panel assumption internally. Using `PanelData(df).balanced` confirms the structure before running models.

## Connections

**Builds on:**
- OLS regression in matrix form (the foundation for all panel estimators)
- Time series concepts (stationarity, autocorrelation)
- Omitted variable bias in cross-sectional regression

**Leads to:**
- Pooled OLS (ignores the panel structure — the naive baseline)
- Fixed effects estimation (exploits within variation to control for $\alpha_i$)
- Random effects estimation (uses both between and within variation efficiently)
- Between-within decomposition (quantifying how much variation each estimator uses)

**Related to:**
- Difference-in-differences (a specific FE application with a treated/control structure)
- Repeated measures ANOVA (the statistical analog in experimental settings)
- Mixed effects models (the hierarchical modeling perspective on the same structure)

## Practice Problems

1. **Variation audit.** Using the `linearmodels` wage panel dataset (`from linearmodels.datasets import wage_panel`), decompose the within and between variation in log wages (`lwage`) and in union status (`union`). Which variable has more within variation? What does this imply for the statistical power of a fixed effects estimator?

2. **Missing data mechanism.** You have a panel of 200 firms over 5 years. After loading, you find 15% of observations are missing. Write code to determine whether missingness is concentrated in certain firms (some firms drop out) or distributed randomly across all firms and years. Why does this distinction matter?

3. **Error component simulation.** Simulate a panel dataset where the true model has $\alpha_i \sim N(0, 2)$, $\lambda_t \sim N(0, 0.5)$, $\epsilon_{it} \sim N(0, 1)$, and $y_{it} = \alpha_i + \lambda_t + 0.8 x_{it} + \epsilon_{it}$ with $\text{Cov}(x_{it}, \alpha_i) = 1$. Compute the intraclass correlation $\rho = \sigma_\alpha^2 / (\sigma_\alpha^2 + \sigma_\epsilon^2)$. Then run pooled OLS and compare the coefficient on $x$ to the true value of 0.8. Explain the discrepancy using the omitted variable bias formula.

## Further Reading

- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data* (6th ed.), Wiley. Chapter 1 provides the most complete treatment of panel data structure and notation.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.), MIT Press. Chapter 10 covers the error component model with careful attention to identification.
- Hsiao, C. (2014). *Analysis of Panel Data* (3rd ed.), Cambridge University Press. Chapter 1 discusses the advantages of panel data and the historical development of panel methods.
- Mundlak, Y. (1978). "On the Pooling of Time Series and Cross Section Data." *Econometrica*, 46(1), 69–85. The foundational paper establishing the between/within decomposition and its implications for estimation.
