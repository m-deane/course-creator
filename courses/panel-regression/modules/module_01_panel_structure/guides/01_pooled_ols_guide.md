# Pooled OLS and Its Limitations

## In Brief

Pooled OLS applies standard OLS to stacked panel observations, treating each entity-time pair as an independent observation. It is the simplest panel estimator and the correct baseline for comparison. Its central limitation: when unobserved entity effects exist and correlate with regressors, Pooled OLS coefficients are biased. Even when unbiased, panel data's within-entity correlation invalidates its standard errors.

## Key Insight

Pooled OLS ignores the panel structure entirely. This creates two distinct problems that compound each other. First, if stable entity characteristics ($\alpha_i$) affect both the regressor and the outcome, the OLS estimate conflates the causal effect of $X$ with the effect of the entity characteristic — this is omitted variable bias. Second, even if no bias exists, observations within the same entity are correlated (they share the same $\alpha_i$), so the i.i.d. assumption fails and standard errors are too small, producing spurious significance.

## Formal Definition

The Pooled OLS model:

$$y_{it} = \beta_0 + x_{it}'\beta + u_{it}, \quad u_{it} = \alpha_i + \epsilon_{it}$$

where:
- $\alpha_i$: unobserved entity-specific effect (time-invariant)
- $\epsilon_{it}$: idiosyncratic error

**OLS estimator applied to pooled data:**

$$\hat{\beta}_{POLS} = \left(\sum_{i,t} x_{it} x_{it}'\right)^{-1} \left(\sum_{i,t} x_{it} y_{it}\right)$$

**Consistency condition:** $E[u_{it} | x_{i1}, \ldots, x_{iT}] = 0$, which requires $\text{Cov}(x_{it}, \alpha_i) = 0$.

**Bias when condition fails:** Applying probability limits:

$$\hat{\beta}_{POLS} \xrightarrow{p} \beta + \underbrace{\frac{\text{Cov}(x_{it}, \alpha_i)}{\text{Var}(x_{it})}}_{\text{omitted variable bias}}$$

**Error covariance structure:** Within entity $i$, errors across periods are correlated:

$$\text{Cov}(u_{it}, u_{is}) = \sigma_\alpha^2 \quad \text{for } t \neq s$$

The **intraclass correlation** (ICC) is:

$$\rho = \frac{\sigma_\alpha^2}{\sigma_\alpha^2 + \sigma_\epsilon^2}$$

High $\rho$ means entity effects dominate and standard errors are severely underestimated by pooled OLS.

**Breusch-Pagan LM test** for $H_0: \sigma_\alpha^2 = 0$ (pooled OLS valid):

$$LM = \frac{NT}{2(T-1)} \left[\frac{\sum_i (\sum_t \hat{u}_{it})^2}{\sum_{it} \hat{u}_{it}^2} - 1\right]^2 \sim \chi^2(1)$$

## Intuitive Explanation

Suppose you want to estimate the effect of employee training expenditure on firm productivity using 10 years of data on 200 firms. Pooled OLS compares all 2,000 observations as if they were independent.

The problem: well-managed firms both invest more in training *and* achieve higher productivity due to their management quality — an unobserved entity characteristic $\alpha_i$. Because management quality ($\alpha_i$) is correlated with training spend ($x_{it}$), OLS attributes some of management's effect to training. The estimated effect of training is inflated.

The serial correlation problem is separate: even if management quality were uncorrelated with training (no bias), observations from the same firm are correlated through the shared $\alpha_i$. The true variance of $\hat{\beta}$ is larger than OLS reports because the 10 observations from Firm A are not 10 independent pieces of information — they all share the same $\alpha_A$.

**The fix for bias:** Fixed effects (removes $\alpha_i$ by demeaning).
**The fix for standard errors even without bias:** Clustered standard errors (correct for within-entity correlation without eliminating entity effects).

## Code Implementation

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PooledOLS
from linearmodels.datasets import wage_panel
from scipy import stats

# Load wage panel data
data = wage_panel.load()
data = data.sort_values(['nr', 'year']).reset_index(drop=True)
panel = data.set_index(['nr', 'year'])

print(f"Panel: {panel.index.get_level_values(0).nunique()} individuals, "
      f"{panel.index.get_level_values(1).nunique()} years")
```

**Fitting Pooled OLS two ways:**

```python
# Method 1: statsmodels OLS (ignores panel structure)
# Treating 4360 observations as if they were i.i.d.
X_pooled = sm.add_constant(data[['exper', 'expersq', 'union', 'married']])
pooled_sm = sm.OLS(data['lwage'], X_pooled).fit()

# Method 2: linearmodels PooledOLS (same estimates, panel-aware SE options)
exog = sm.add_constant(panel[['exper', 'expersq', 'union', 'married']])
pooled_lm = PooledOLS(panel['lwage'], exog).fit()
pooled_clustered = PooledOLS(panel['lwage'], exog).fit(
    cov_type='clustered', cluster_entity=True
)

print("Coefficient estimates (all methods identical):")
print(f"  exper coefficient: {pooled_lm.params['exper']:.4f}")

print("\nStandard error comparison for 'union':")
print(f"  OLS (i.i.d.): {pooled_sm.bse['union']:.4f}")
print(f"  Clustered:    {pooled_clustered.std_errors['union']:.4f}")
print(f"  Ratio:        {pooled_clustered.std_errors['union'] / pooled_sm.bse['union']:.2f}x larger")
```

**Demonstrating omitted variable bias:**

```python
def demonstrate_ols_bias(n_firms=200, n_years=10, true_beta=1.5, seed=42):
    """
    Show that Pooled OLS is biased when entity effects correlate
    with the regressor, while Fixed Effects recovers the true parameter.
    """
    np.random.seed(seed)
    N, T = n_firms, n_years

    # Generate correlated entity effects
    alpha = np.random.randn(N) * 2           # Entity fixed effects
    x = np.repeat(alpha * 0.7, T) + np.random.randn(N * T)  # x correlated with alpha

    y = (np.repeat(alpha, T)           # Entity effect
         + true_beta * x               # True causal effect
         + np.random.randn(N * T))     # Idiosyncratic noise

    firm_id = np.repeat(np.arange(N), T)
    time_id = np.tile(np.arange(T), N)

    df = pd.DataFrame({'firm': firm_id, 'time': time_id, 'x': x, 'y': y})
    panel = df.set_index(['firm', 'time'])

    # Pooled OLS
    exog_ols = sm.add_constant(panel[['x']])
    pols = PooledOLS(panel['y'], exog_ols).fit()

    # Fixed Effects (for comparison)
    from linearmodels.panel import PanelOLS
    fe = PanelOLS.from_formula('y ~ x + EntityEffects', data=panel).fit()

    print(f"True beta: {true_beta}")
    print(f"Pooled OLS: {pols.params['x']:.4f} (biased upward)")
    print(f"Fixed Effects: {fe.params['x']:.4f} (approximately unbiased)")
    print(f"OVB: {pols.params['x'] - true_beta:.4f}")

demonstrate_ols_bias()
```

**Breusch-Pagan LM test for entity effects:**

```python
def breusch_pagan_lm_test(df, entity_col, outcome_col, exog_cols):
    """
    Test H0: sigma_alpha^2 = 0 (no entity effects, pooled OLS valid).
    Rejection -> entity effects exist -> consider FE or RE.
    """
    X = sm.add_constant(df[exog_cols])
    resids = sm.OLS(df[outcome_col], X).fit().resid

    N = df[entity_col].nunique()
    T_bar = len(df) / N  # Average T (works for unbalanced)
    n_obs = len(df)

    df_r = df.copy()
    df_r['resid'] = resids

    # Sum of residuals per entity
    entity_resid_sums = df_r.groupby(entity_col)['resid'].sum()
    sum_sq_entity = (entity_resid_sums ** 2).sum()
    sum_sq_total = (resids ** 2).sum()

    LM = (N * T_bar / (2 * (T_bar - 1))) * (sum_sq_entity / sum_sq_total - 1) ** 2
    p_value = 1 - stats.chi2.cdf(LM, df=1)

    print(f"Breusch-Pagan LM Test for Entity Effects:")
    print(f"  LM statistic: {LM:.4f}")
    print(f"  p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("  Conclusion: Reject H0. Entity effects are significant -> use FE or RE")
    else:
        print("  Conclusion: Fail to reject H0. Pooled OLS may be acceptable")
    return LM, p_value

breusch_pagan_lm_test(
    data, 'nr', 'lwage', ['exper', 'expersq', 'union', 'married']
)
```

**ICC estimation:**

```python
def estimate_icc(df, entity_col, y_col):
    """
    Estimate the intraclass correlation coefficient.
    High ICC means entity effects dominate and clustered SE are critical.
    """
    entity_means = df.groupby(entity_col)[y_col].transform('mean')
    between_var = entity_means.var()
    within_var = (df[y_col] - entity_means).var()
    icc = between_var / (between_var + within_var)

    T_avg = len(df) / df[entity_col].nunique()
    # Design effect: how much clustering inflates variance of mean
    deff = 1 + (T_avg - 1) * icc

    print(f"ICC (rho): {icc:.4f}")
    print(f"Average T: {T_avg:.1f}")
    print(f"Design effect: {deff:.2f}x")
    print(f"Effective sample size: {len(df) / deff:.0f} (vs {len(df)} nominal)")
    return icc

estimate_icc(data, 'nr', 'lwage')
```

## Common Pitfalls

**Using pooled OLS as a final model without testing.** The Breusch-Pagan LM test checks whether entity effects are significant. Run it before accepting pooled OLS. Significant effects require FE or RE.

**Using i.i.d. standard errors with panel data.** Even when pooled OLS is consistent (no correlated entity effects), the i.i.d. errors assumption fails because observations from the same entity share $\alpha_i$. Always use `cov_type='clustered', cluster_entity=True` for any panel regression, including pooled OLS.

**Interpreting the pooled $R^2$ as a measure of model fit.** Pooled $R^2$ includes variation explained by entity effects even when the model has no entity dummies. Comparing it with the FE within-$R^2$ is misleading — they measure different things.

**Confusing "no entity effects" with "no correlation between effects and regressors."** Even if entity effects are large ($\sigma_\alpha^2$ is large), if they are uncorrelated with $X$ ($\text{Cov}(\alpha_i, x_{it}) = 0$), pooled OLS is consistent (though inefficient compared to RE). The bias test is the Hausman test, not the Breusch-Pagan test.

**Claiming pooled OLS is valid because the LM test fails to reject.** Failure to reject the null of no entity effects ($\sigma_\alpha^2 = 0$) in small panels (small $N$ or $T$) may reflect low power, not genuine absence of effects.

## Connections

**Builds on:**
- OLS in matrix form
- Omitted variable bias formula: $\text{plim}(\hat{\beta}) = \beta + (X'X)^{-1}X'\alpha$
- Sandwich/robust standard errors

**Leads to:**
- Fixed effects: addresses both the bias (by eliminating $\alpha_i$) and the serial correlation
- Random effects: addresses the efficiency loss from ignoring the error structure
- Breusch-Pagan test: the formal decision rule for whether to move beyond pooled OLS
- Clustered standard errors: the minimum correction for any panel regression

**Related to:**
- Repeated measures ANOVA: pooled OLS ignoring group structure is the same problem
- Multilevel models: the $\alpha_i$ is a group-level random effect

## Practice Problems

1. **Bias quantification.** In the `demonstrate_ols_bias` function, vary the correlation between $x_{it}$ and $\alpha_i$ from 0 to 1 in steps of 0.1. Plot the pooled OLS bias as a function of this correlation. At what correlation does the bias become economically meaningful (say, exceeding 10% of the true parameter value)?

2. **Standard error inflation.** Using the wage panel, compare the t-statistic on `union` from pooled OLS with i.i.d. errors versus clustered errors. How many times larger is the clustered standard error? What conclusion changes (significant vs insignificant) when switching to clustered errors?

3. **LM test power.** Simulate a panel with $N=50$, $T=3$, and a true $\sigma_\alpha^2 = 0.5$, $\sigma_\epsilon^2 = 1$. Run the Breusch-Pagan LM test 1,000 times (with new data each time). What fraction of tests reject at the 5% level? Repeat with $N=500$. Comment on the power properties.

## Further Reading

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*. Chapter 10.4 covers pooled OLS, its consistency conditions, and the LM test.
- Breusch, T. S. & Pagan, A. R. (1980). "The Lagrange Multiplier Test and its Applications to Model Specification in Econometrics." *Review of Economic Studies*, 47(1), 239–253. The original paper deriving the LM test for random effects.
- Cameron, A. C. & Miller, D. L. (2015). "A Practitioner's Guide to Cluster-Robust Inference." *Journal of Human Resources*, 50(2), 317–372. Definitive guide on when and how to cluster standard errors.
