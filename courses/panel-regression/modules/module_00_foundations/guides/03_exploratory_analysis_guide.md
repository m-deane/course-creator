# Exploratory Panel Data Analysis

## In Brief

Exploratory analysis of panel data goes beyond standard EDA by decomposing variation into between and within components, testing for entity and time effects, and diagnosing serial correlation and heteroskedasticity. These diagnostics directly determine which estimator is appropriate and what standard errors are required.

## Key Insight

In standard cross-sectional EDA, a scatter plot of $y$ against $x$ tells you the relationship. In panel EDA, the same scatter plot hides three distinct relationships: the pooled relationship (mixing everything), the between relationship (comparing entities to each other), and the within relationship (tracking each entity over time). When these slopes differ substantially, using pooled OLS will produce biased estimates. Fixed effects uses only the within slope.

## Formal Definition

**Variation decomposition:**

$$\text{Var}(y_{it}) = \underbrace{\text{Var}(\bar{y}_i)}_{\text{between}} + \underbrace{\text{Var}(y_{it} - \bar{y}_i)}_{\text{within}}$$

**Three correlation types for $(x, y)$:**

| Type | Formula | Uses |
|------|---------|------|
| Total | $\text{Corr}(x_{it}, y_{it})$ | All variation |
| Between | $\text{Corr}(\bar{x}_i, \bar{y}_i)$ | Cross-entity differences |
| Within | $\text{Corr}(x_{it} - \bar{x}_i, y_{it} - \bar{y}_i)$ | Within-entity changes |

**F-test for entity effects:**

$$F = \frac{SS_{\text{between}} / (N-1)}{SS_{\text{within}} / (NT - N)} \sim F(N-1, NT-N)$$

**Intraclass Correlation Coefficient (ICC):**

$$\rho = \frac{\sigma_\alpha^2}{\sigma_\alpha^2 + \sigma_\epsilon^2}$$

where $\rho$ close to 1 indicates most variation is between-entity (strong entity effects).

## Intuitive Explanation

Imagine tracking 50 countries' GDP growth and trade openness over 20 years. A pooled scatter plot shows a positive relationship — richer, more open economies tend to grow. But is this because openness *causes* growth (within-country effect), or because developed economies happen to be both open and fast-growing (between-country confounding)?

The between scatter plots entity means: one dot per country, at its average openness and average growth. The within scatter plots each observation as deviations from the country's own average: does a country grow faster in years when it is more open than its own average? Only the within variation speaks to causation, because it controls for everything stable about each country.

When the between slope and within slope diverge dramatically, this is a signal that unobserved entity characteristics are correlated with the regressor — exactly the case where pooled OLS is biased and fixed effects is necessary.

## Code Implementation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from linearmodels.datasets import wage_panel

# Load the Vella & Verbeek (1998) wage panel
data = wage_panel.load()
data = data.sort_values(['nr', 'year']).reset_index(drop=True)

N = data['nr'].nunique()
T = data['year'].nunique()
print(f"Panel: N={N} individuals, T={T} years, {N*T} observations")
```

**Entity trajectory visualization:**

```python
def plot_entity_trajectories(df, entity_col, time_col, y_col,
                             n_entities=20, figsize=(14, 6)):
    """
    Plot time series for a random sample of entities.
    Reveals whether trajectories are parallel (FE appropriate)
    or diverge in ways suggesting time-varying entity effects.
    """
    sample_entities = np.random.choice(
        df[entity_col].unique(), size=n_entities, replace=False
    )
    sample_df = df[df[entity_col].isin(sample_entities)]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Raw trajectories
    for entity, grp in sample_df.groupby(entity_col):
        axes[0].plot(grp[time_col], grp[y_col], alpha=0.5, linewidth=0.8)
    axes[0].set_title(f'Entity Trajectories: {y_col}')
    axes[0].set_xlabel(time_col)
    axes[0].set_ylabel(y_col)

    # Demeaned trajectories (within variation only)
    sample_df = sample_df.copy()
    entity_means = sample_df.groupby(entity_col)[y_col].transform('mean')
    sample_df['y_demeaned'] = sample_df[y_col] - entity_means

    for entity, grp in sample_df.groupby(entity_col):
        axes[1].plot(grp[time_col], grp['y_demeaned'], alpha=0.5, linewidth=0.8)
    axes[1].axhline(0, color='black', linewidth=1)
    axes[1].set_title(f'Demeaned Trajectories (Within Variation)')
    axes[1].set_xlabel(time_col)

    plt.tight_layout()
    plt.show()

plot_entity_trajectories(data, 'nr', 'year', 'lwage', n_entities=30)
```

**Between vs within variation scatter:**

```python
def plot_between_within(df, entity_col, x_col, y_col, figsize=(15, 5)):
    """
    Three-panel plot showing pooled, between, and within slopes.
    The key diagnostic: if between and within slopes differ,
    entity effects are correlated with X -> fixed effects required.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. Pooled (all variation mixed)
    axes[0].scatter(df[x_col], df[y_col], alpha=0.2, s=8)
    m, b = np.polyfit(df[x_col], df[y_col], 1)
    xl = np.linspace(df[x_col].min(), df[x_col].max(), 100)
    axes[0].plot(xl, m * xl + b, 'r-', linewidth=2)
    axes[0].set_title(f'Pooled: slope = {m:.3f}')
    axes[0].set_xlabel(x_col)
    axes[0].set_ylabel(y_col)

    # 2. Between (entity means)
    means = df.groupby(entity_col)[[x_col, y_col]].mean()
    axes[1].scatter(means[x_col], means[y_col], s=40, alpha=0.7)
    m_b, b_b = np.polyfit(means[x_col], means[y_col], 1)
    xl_b = np.linspace(means[x_col].min(), means[x_col].max(), 100)
    axes[1].plot(xl_b, m_b * xl_b + b_b, 'r-', linewidth=2)
    axes[1].set_title(f'Between: slope = {m_b:.3f}')
    axes[1].set_xlabel(f'Mean {x_col}')
    axes[1].set_ylabel(f'Mean {y_col}')

    # 3. Within (demeaned)
    x_dm = df[x_col] - df.groupby(entity_col)[x_col].transform('mean')
    y_dm = df[y_col] - df.groupby(entity_col)[y_col].transform('mean')
    axes[2].scatter(x_dm, y_dm, alpha=0.2, s=8)
    m_w, b_w = np.polyfit(x_dm, y_dm, 1)
    xl_w = np.linspace(x_dm.min(), x_dm.max(), 100)
    axes[2].plot(xl_w, m_w * xl_w + b_w, 'r-', linewidth=2)
    axes[2].set_title(f'Within: slope = {m_w:.3f}')
    axes[2].set_xlabel(f'Demeaned {x_col}')
    axes[2].set_ylabel(f'Demeaned {y_col}')

    plt.suptitle('Variation Decomposition: Pooled vs Between vs Within')
    plt.tight_layout()
    plt.show()

    return {'pooled': m, 'between': m_b, 'within': m_w}

slopes = plot_between_within(data, 'nr', 'exper', 'lwage')
print(f"Slope comparison: {slopes}")
```

**F-test for entity and time effects:**

```python
def f_test_entity_effects(df, entity_col, time_col, y_col):
    """
    F-test for the null hypothesis that all entity effects are zero.
    Rejection -> entity effects exist -> consider FE or RE.
    """
    N = df[entity_col].nunique()
    T = df[time_col].nunique()
    n_obs = len(df)

    # Between sum of squares
    grand_mean = df[y_col].mean()
    entity_means = df.groupby(entity_col)[y_col].transform('mean')
    ss_between = ((entity_means - grand_mean) ** 2).sum()
    df_between = N - 1

    # Within sum of squares
    ss_within = ((df[y_col] - entity_means) ** 2).sum()
    df_within = n_obs - N

    f_stat = (ss_between / df_between) / (ss_within / df_within)
    p_value = 1 - stats.f.cdf(f_stat, df_between, df_within)

    print(f"F-test for entity effects:")
    print(f"  F({df_between}, {df_within}) = {f_stat:.3f}")
    print(f"  p-value = {p_value:.4f}")
    print(f"  {'Reject H0: entity effects are significant' if p_value < 0.05 else 'Fail to reject H0: no strong entity effects'}")
    return f_stat, p_value

f_stat, p_val = f_test_entity_effects(data, 'nr', 'year', 'lwage')
```

**Variation decomposition and ICC:**

```python
def decompose_variation(df, entity_col, variable):
    """Full variation decomposition with ICC computation."""
    total_var = df[variable].var()
    entity_means = df.groupby(entity_col)[variable].transform('mean')
    between_var = entity_means.var()
    within_var = (df[variable] - entity_means).var()

    # ICC: proportion of variance due to entity effects
    icc = between_var / total_var

    return {
        'total': total_var,
        'between': between_var,
        'within': within_var,
        'between_share': between_var / total_var,
        'within_share': within_var / total_var,
        'icc': icc
    }

for var in ['lwage', 'exper', 'union', 'married']:
    d = decompose_variation(data, 'nr', var)
    print(f"{var:10s}: between={d['between_share']:.1%}, within={d['within_share']:.1%}, ICC={d['icc']:.3f}")
```

**Serial correlation detection:**

```python
def test_serial_correlation(df, entity_col, time_col, y_col):
    """
    Test for first-order serial correlation in the outcome variable.
    If present in residuals after estimation, clustered SE are needed.
    """
    df_sorted = df.sort_values([entity_col, time_col]).copy()
    df_sorted['y_lag'] = df_sorted.groupby(entity_col)[y_col].shift(1)
    df_clean = df_sorted.dropna(subset=['y_lag'])

    rho = df_clean[y_col].corr(df_clean['y_lag'])
    n = len(df_clean)

    # Approximate SE for correlation: 1/sqrt(n)
    t_stat = rho / (1 / np.sqrt(n))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))

    print(f"Serial correlation (rho): {rho:.4f}")
    print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")
    print(f"{'Serial correlation detected: use clustered SE' if p_value < 0.05 else 'No significant serial correlation'}")
    return rho, p_value

test_serial_correlation(data, 'nr', 'year', 'lwage')
```

**Complete EDA report:**

```python
def panel_eda_report(df, entity_col, time_col, outcome, predictors):
    """
    Generate a structured EDA report for panel data.
    Call this before running any regression.
    """
    print("=" * 60)
    print("PANEL EDA REPORT")
    print("=" * 60)

    print(f"\n1. DATA STRUCTURE")
    N = df[entity_col].nunique()
    T = df[time_col].nunique()
    balance = len(df) / (N * T)
    print(f"   N={N} entities, T={T} periods, {len(df)} observations")
    print(f"   Balance: {balance:.1%}")

    print(f"\n2. VARIATION DECOMPOSITION")
    for var in [outcome] + predictors:
        d = decompose_variation(df, entity_col, var)
        print(f"   {var:12s}: between={d['between_share']:.0%}, within={d['within_share']:.0%}")

    print(f"\n3. CORRELATION WITH OUTCOME ({outcome})")
    for pred in predictors:
        total_c = df[pred].corr(df[outcome])
        means = df.groupby(entity_col)[[pred, outcome]].mean()
        between_c = means[pred].corr(means[outcome])
        pred_dm = df[pred] - df.groupby(entity_col)[pred].transform('mean')
        out_dm = df[outcome] - df.groupby(entity_col)[outcome].transform('mean')
        within_c = pred_dm.corr(out_dm)
        print(f"   {pred:12s}: total={total_c:.3f}, between={between_c:.3f}, within={within_c:.3f}")

    print(f"\n4. ENTITY EFFECTS TEST")
    f_test_entity_effects(df, entity_col, time_col, outcome)

panel_eda_report(data, 'nr', 'year', 'lwage', ['exper', 'expersq', 'union', 'married'])
```

## Common Pitfalls

**Stopping at the pooled scatter plot.** The pooled scatter mixes between and within variation. Always create all three panels — pooled, between, and within — before concluding anything about the X-Y relationship.

**Ignoring time effects when testing entity effects.** The F-test above tests entity effects marginally. If strong time effects exist (e.g., a global wage trend), they inflate the between variation and can affect the F-statistic. Test for time effects separately.

**Interpreting ICC without context.** An ICC of 0.6 means 60% of variance is between-entity. This says nothing about whether the entity effects are correlated with your regressors — that is the Hausman test's job. High ICC merely implies that entity effects are large and important.

**Using raw residuals for serial correlation tests.** Testing serial correlation in the raw outcome $y_{it}$ mixes entity effects with idiosyncratic error. Test serial correlation in the within-transformed residuals (after fitting a model) to assess the idiosyncratic error structure.

**Treating outlier entities as erroneous.** An entity that looks like an outlier in raw data may be substantively important. Before dropping, investigate whether it influences the within-slope versus only the between-slope.

## Connections

**Builds on:**
- Variation decomposition (between vs within)
- F-test mechanics and chi-squared distribution
- Scatter plot interpretation

**Leads to:**
- Model selection: the EDA results directly inform whether to run pooled OLS, FE, or RE
- Standard error choice: serial correlation and heteroskedasticity detected here determine whether to cluster
- Two-way fixed effects: if time effects are significant in the F-test, add time dummies

**Related to:**
- Difference-in-differences: the parallel trends assumption is testable with pre-period trajectory plots
- Hierarchical modeling: ICC is also used in multilevel model diagnostics

## Practice Problems

1. **Grunfeld investment data.** Using `plm`'s Grunfeld dataset (available in Python via `linearmodels`), run the complete `panel_eda_report` for `inv` as outcome and `value`, `capital` as predictors. Report the within-share of variation for each variable. Do the between and within slopes for `value` differ substantially? What does this imply for fixed effects estimation?

2. **Simpson's paradox in panel data.** Construct a simulated panel where the pooled slope is *negative* but the within slope is *positive*. This requires a specific pattern of entity effects correlated with $x$. Plot all three scatter panels to visualize the paradox and explain in words why it occurs.

3. **Time effects test.** Extend the `panel_eda_report` function to include an F-test for time effects (the test symmetric to the entity effects F-test). Apply it to the wage panel. If time effects are significant, what does this suggest about model specification?

## Further Reading

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*, Chapter 10.2. Covers variation decomposition and its link to estimator choice.
- Angrist, J. D. & Pischke, J. S. (2009). *Mostly Harmless Econometrics*, Princeton. Chapter 5.1 provides intuition for within vs between variation through real examples.
- Vella, F. & Verbeek, M. (1998). "Whose Wages Do Unions Raise? A Dynamic Model of Unionism and Wage Rate Determination for Young Men." *Journal of Applied Econometrics*, 13(2), 163–183. The source paper for the wage panel dataset used in this course.
