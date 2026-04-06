# Specification Tests for Panel Models

> **Reading time:** ~20 min | **Module:** 04 — Model Selection | **Prerequisites:** Module 3


## Overview


<div class="callout-key">

**Key Concept Summary:** Beyond the Hausman test, several specification tests help validate panel model assumptions:

</div>

Beyond the Hausman test, several specification tests help validate panel model assumptions:

1. **F-test for fixed effects**
2. **Breusch-Pagan LM test for random effects**
3. **Robust Hausman test**
4. **Serial correlation tests**
5. **Heteroskedasticity tests**

## F-Test for Fixed Effects

Tests whether entity fixed effects are jointly significant.

$$H_0: \alpha_1 = \alpha_2 = ... = \alpha_N$$


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS, PooledOLS
from scipy import stats

def f_test_fixed_effects(df, y_col, x_cols, entity_col, time_col):
    """
    F-test for presence of entity fixed effects.
    """
    df_panel = df.set_index([entity_col, time_col])

    # Restricted model: Pooled OLS
    pooled = PooledOLS(df_panel[y_col], df_panel[x_cols]).fit()

    # Unrestricted model: Fixed Effects
    fe = PanelOLS(df_panel[y_col], df_panel[x_cols], entity_effects=True).fit()

    # Calculate F-statistic
    N = df[entity_col].nunique()  # Number of entities
    n = len(df)  # Total observations
    K = len(x_cols)  # Number of regressors

    # RSS from each model
    rss_restricted = pooled.resid_ss
    rss_unrestricted = fe.resid_ss

    # Degrees of freedom
    df_num = N - 1  # Number of restrictions
    df_denom = n - N - K

<div class="callout-insight">

**Insight:** Panel data lets you control for unobservable differences between entities that are constant over time. This is the single most important reason to prefer panel data over repeated cross-sections.

</div>


    # F-statistic
    f_stat = ((rss_restricted - rss_unrestricted) / df_num) / (rss_unrestricted / df_denom)
    p_value = 1 - stats.f.cdf(f_stat, df_num, df_denom)

    print("F-Test for Entity Fixed Effects:")
    print("=" * 60)
    print(f"H0: All entity effects are equal (pooled OLS is appropriate)")
    print(f"H1: Entity effects differ (fixed effects needed)")
    print()
    print(f"F-statistic: {f_stat:.4f}")
    print(f"Degrees of freedom: ({df_num}, {df_denom})")
    print(f"p-value: {p_value:.6f}")
    print()

    if p_value < 0.05:
        print("Result: REJECT H0 - Entity fixed effects are significant")
        print("Recommendation: Use Fixed Effects over Pooled OLS")
    else:
        print("Result: Cannot reject H0 - Pooled OLS may be appropriate")

    return f_stat, p_value

# Example data
np.random.seed(42)
n_entities = 100
n_periods = 15

data = []
entity_effects = np.random.normal(0, 2, n_entities)

for i in range(n_entities):
    for t in range(n_periods):
        x1 = np.random.normal(5, 1) + 0.3 * entity_effects[i]
        x2 = np.random.normal(3, 0.5)
        y = 2 + 1.5 * x1 - 0.8 * x2 + entity_effects[i] + np.random.normal(0, 0.5)
        data.append({'entity': i, 'time': t, 'x1': x1, 'x2': x2, 'y': y})

df = pd.DataFrame(data)

f_test_fixed_effects(df, 'y', ['x1', 'x2'], 'entity', 'time')
```


</div>

## Breusch-Pagan LM Test

<div class="callout-warning">

**Warning:** Reporting results without appropriate standard errors is a common mistake. In panel data, conventional OLS standard errors are almost always wrong -- use clustered or heteroskedasticity-robust standard errors.

</div>


Tests for random effects vs pooled OLS.

$$H_0: \sigma_u^2 = 0$$ (No random effects)


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def breusch_pagan_lm(df, y_col, x_cols, entity_col, time_col):
    """
    Breusch-Pagan Lagrange Multiplier test for random effects.
    """
    # Pooled OLS
    formula = f'{y_col} ~ {" + ".join(x_cols)}'
    pooled = smf.ols(formula, data=df).fit()

    # Get residuals
    residuals = pooled.resid
    df_temp = df.copy()
    df_temp['resid'] = residuals

    # Group by entity
    N = df[entity_col].nunique()
    T_avg = len(df) / N  # Average T

    # Sum of squared group sums
    group_sums = df_temp.groupby(entity_col)['resid'].sum()
    sum_squared_group_sums = (group_sums ** 2).sum()

    # Total sum of squares of residuals
    total_ss = (residuals ** 2).sum()

    # LM statistic
    # LM = (nT / 2(T-1)) * [sum_i (sum_t e_it)^2 / sum_i sum_t e_it^2 - 1]^2
    n = len(df)
    ratio = sum_squared_group_sums / total_ss

    # Balanced formula
    T = df.groupby(entity_col).size().values[0]  # Assume balanced
    lm_stat = (n * T / (2 * (T - 1))) * ((T * ratio - 1) ** 2)

    # Under H0, LM ~ chi-squared(1)
    p_value = 1 - stats.chi2.cdf(lm_stat, 1)

    print("Breusch-Pagan LM Test for Random Effects:")
    print("=" * 60)
    print(f"H0: σ²_u = 0 (No entity-specific variance)")
    print(f"H1: σ²_u > 0 (Random effects present)")
    print()
    print(f"LM statistic: {lm_stat:.4f}")
    print(f"Degrees of freedom: 1")
    print(f"p-value: {p_value:.6f}")
    print()

    if p_value < 0.05:
        print("Result: REJECT H0 - Random effects are significant")
        print("Recommendation: Use RE or FE over Pooled OLS")
    else:
        print("Result: Cannot reject H0 - Pooled OLS may be appropriate")

    return lm_stat, p_value

breusch_pagan_lm(df, 'y', ['x1', 'x2'], 'entity', 'time')
```


</div>

## Robust Hausman Test

The standard Hausman test can have size distortions with heteroskedasticity. A robust version:


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
from linearmodels.panel import RandomEffects

def robust_hausman_test(df, y_col, x_cols, entity_col, time_col):
    """
    Robust Hausman test using auxiliary regression.
    """
    df_test = df.copy()
    df_panel = df.set_index([entity_col, time_col])

    # Calculate entity means (Mundlak terms)
    for x in x_cols:
        df_test[f'{x}_bar'] = df_test.groupby(entity_col)[x].transform('mean')

    # Auxiliary regression: RE model augmented with entity means
    x_bar_cols = [f'{x}_bar' for x in x_cols]

    # Using mixed effects
    formula = f'{y_col} ~ {" + ".join(x_cols)} + {" + ".join(x_bar_cols)}'
    aug_model = smf.mixedlm(formula, data=df_test, groups=entity_col).fit()

    # Test joint significance of x_bar terms
    print("Robust Hausman Test (Mundlak Approach):")
    print("=" * 60)
    print(f"H0: E[α_i | X_it] = 0 (RE is consistent)")
    print(f"H1: E[α_i | X_it] ≠ 0 (FE required)")
    print()
    print("Coefficients on entity means (x̄ᵢ):")

    significant = False
    for col in x_bar_cols:
        coef = aug_model.params[col]
        se = aug_model.bse[col]
        t_stat = coef / se
        p_val = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        print(f"  {col}: {coef:.4f} (t={t_stat:.2f}, p={p_val:.4f})")
        if p_val < 0.05:
            significant = True

    print()
    if significant:
        print("Result: At least one x̄ coefficient is significant")
        print("Recommendation: Use Fixed Effects")
    else:
        print("Result: No significant correlation detected")
        print("Recommendation: Random Effects may be appropriate")

    return aug_model

robust_hausman_test(df, 'y', ['x1', 'x2'], 'entity', 'time')
```


</div>

## Serial Correlation Tests

### Wooldridge Test for AR(1) Serial Correlation

```python
def wooldridge_serial_correlation_test(df, y_col, x_cols, entity_col, time_col):
    """
    Wooldridge test for serial correlation in panel data.
    """
    df_test = df.copy()
    df_panel = df.set_index([entity_col, time_col])

    # First-difference the data
    df_test = df_test.sort_values([entity_col, time_col])

    for col in [y_col] + x_cols:
        df_test[f'd_{col}'] = df_test.groupby(entity_col)[col].diff()

    # Drop first period
    df_fd = df_test.dropna()

    # Regress differenced y on differenced x
    fd_formula = f'd_{y_col} ~ {" + ".join(["d_" + x for x in x_cols])}'
    fd_model = smf.ols(fd_formula, data=df_fd).fit()

    # Get residuals and lag them
    df_fd['resid'] = fd_model.resid
    df_fd['resid_lag'] = df_fd.groupby(entity_col)['resid'].shift(1)

    # Drop missing
    df_fd = df_fd.dropna()

    # Test: regress residual on lagged residual
    test_model = smf.ols('resid ~ resid_lag - 1', data=df_fd).fit()

    # Under H0 of no serial correlation, coefficient should be -0.5
    # Test if coefficient differs from -0.5
    coef = test_model.params['resid_lag']
    se = test_model.bse['resid_lag']
    t_stat = (coef - (-0.5)) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

    print("Wooldridge Test for Serial Correlation:")
    print("=" * 60)
    print(f"H0: No first-order serial correlation")
    print(f"H1: AR(1) serial correlation present")
    print()
    print(f"Coefficient on lagged residual: {coef:.4f}")
    print(f"Expected under H0: -0.5")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.6f}")
    print()

    if p_value < 0.05:
        print("Result: REJECT H0 - Serial correlation detected")
        print("Recommendation: Use clustered standard errors or AR(1) correction")
    else:
        print("Result: Cannot reject H0 - No evidence of serial correlation")

    return t_stat, p_value

wooldridge_serial_correlation_test(df, 'y', ['x1', 'x2'], 'entity', 'time')
```

## Heteroskedasticity Tests

### Modified Wald Test

```python
def modified_wald_test(df, y_col, x_cols, entity_col, time_col):
    """
    Modified Wald test for groupwise heteroskedasticity in FE model.
    """
    df_panel = df.set_index([entity_col, time_col])

    # Fit FE model
    fe = PanelOLS(df_panel[y_col], df_panel[x_cols], entity_effects=True).fit()

    # Get residuals
    residuals = fe.resids

    # Calculate variance by entity
    df_temp = df.copy()
    df_temp['resid_sq'] = residuals.values ** 2

    entity_vars = df_temp.groupby(entity_col)['resid_sq'].mean()

    # Overall mean squared residual
    sigma_sq = df_temp['resid_sq'].mean()

    # Wald statistic
    N = df[entity_col].nunique()
    T = df[time_col].nunique()

    # Under H0: all entity variances equal
    wald_stat = N * sum((entity_vars - sigma_sq) ** 2) / (2 * sigma_sq ** 2)

    p_value = 1 - stats.chi2.cdf(wald_stat, N - 1)

    print("Modified Wald Test for Groupwise Heteroskedasticity:")
    print("=" * 60)
    print(f"H0: σ²_i = σ² for all i (Homoskedasticity)")
    print(f"H1: σ²_i differs across entities")
    print()
    print(f"Wald statistic: {wald_stat:.4f}")
    print(f"Degrees of freedom: {N - 1}")
    print(f"p-value: {p_value:.6f}")
    print()

    if p_value < 0.05:
        print("Result: REJECT H0 - Heteroskedasticity detected")
        print("Recommendation: Use robust or clustered standard errors")
    else:
        print("Result: Cannot reject H0 - Homoskedasticity assumption OK")

    return wald_stat, p_value

modified_wald_test(df, 'y', ['x1', 'x2'], 'entity', 'time')
```

## Comprehensive Specification Testing

<div class="callout-danger">

**Danger:** Never include a lagged dependent variable in a fixed effects model without using an appropriate estimator (e.g., Arellano-Bond GMM). The within-transformation creates mechanical correlation between the transformed lagged variable and the transformed error, biasing all coefficients.

</div>


```python
def run_all_specification_tests(df, y_col, x_cols, entity_col, time_col):
    """
    Run comprehensive battery of specification tests.
    """
    print("=" * 70)
    print("COMPREHENSIVE PANEL DATA SPECIFICATION TESTS")
    print("=" * 70)

    results = {}

    # 1. F-test for FE
    print("\n" + "-" * 70)
    print("TEST 1: F-Test for Fixed Effects")
    print("-" * 70)
    f_stat, f_pval = f_test_fixed_effects(df, y_col, x_cols, entity_col, time_col)
    results['f_test'] = {'stat': f_stat, 'p': f_pval, 'fe_significant': f_pval < 0.05}

    # 2. BP-LM test for RE
    print("\n" + "-" * 70)
    print("TEST 2: Breusch-Pagan LM Test")
    print("-" * 70)
    lm_stat, lm_pval = breusch_pagan_lm(df, y_col, x_cols, entity_col, time_col)
    results['bp_lm'] = {'stat': lm_stat, 'p': lm_pval, 're_significant': lm_pval < 0.05}

    # 3. Robust Hausman
    print("\n" + "-" * 70)
    print("TEST 3: Robust Hausman Test")
    print("-" * 70)
    hausman_model = robust_hausman_test(df, y_col, x_cols, entity_col, time_col)
    results['hausman'] = hausman_model

    # 4. Serial correlation
    print("\n" + "-" * 70)
    print("TEST 4: Wooldridge Serial Correlation Test")
    print("-" * 70)
    sc_stat, sc_pval = wooldridge_serial_correlation_test(df, y_col, x_cols, entity_col, time_col)
    results['serial_corr'] = {'stat': sc_stat, 'p': sc_pval, 'present': sc_pval < 0.05}

    # 5. Heteroskedasticity
    print("\n" + "-" * 70)
    print("TEST 5: Modified Wald Heteroskedasticity Test")
    print("-" * 70)
    het_stat, het_pval = modified_wald_test(df, y_col, x_cols, entity_col, time_col)
    results['heterosked'] = {'stat': het_stat, 'p': het_pval, 'present': het_pval < 0.05}

    # Summary recommendations
    print("\n" + "=" * 70)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 70)

    # Model choice
    if results['f_test']['fe_significant'] or results['bp_lm']['re_significant']:
        print("\n1. MODEL STRUCTURE:")
        print("   Entity effects are present (reject pooled OLS)")

        # Check Hausman result (simplified)
        x_bar_cols = [f'{x}_bar' for x in x_cols]
        hausman_significant = any(hausman_model.pvalues[col] < 0.05 for col in x_bar_cols
                                  if col in hausman_model.pvalues)

        if hausman_significant:
            print("   → Use FIXED EFFECTS (correlation with regressors detected)")
        else:
            print("   → RANDOM EFFECTS may be appropriate (no correlation detected)")
    else:
        print("\n1. MODEL STRUCTURE:")
        print("   → POOLED OLS may be appropriate (no significant entity effects)")

    # Standard errors
    print("\n2. STANDARD ERRORS:")
    if results['serial_corr']['present'] or results['heterosked']['present']:
        if results['serial_corr']['present'] and results['heterosked']['present']:
            print("   → Use two-way clustered standard errors (serial corr + heterosked)")
        elif results['serial_corr']['present']:
            print("   → Use clustered standard errors by entity (serial correlation)")
        else:
            print("   → Use robust/clustered standard errors (heteroskedasticity)")
    else:
        print("   → Standard errors appear valid, but clustering is still recommended")

    return results

# Run all tests
all_results = run_all_specification_tests(df, 'y', ['x1', 'x2'], 'entity', 'time')
```

## Key Takeaways

1. **F-test and BP-LM** help choose between pooled and panel models

2. **Hausman test** (robust version) guides FE vs RE choice

3. **Serial correlation** requires clustered standard errors

4. **Heteroskedasticity** requires robust standard errors

5. **Run multiple tests** for comprehensive model validation


---

## Conceptual Practice Questions

**Practice Question 1:** What problem does this approach solve that simpler methods cannot?

**Practice Question 2:** What are the key assumptions, and how would you test them in practice?


---

## Cross-References

<a class="link-card" href="./01_hausman_test.md">
  <div class="link-card-title">01 Hausman Test</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_hausman_test.md">
  <div class="link-card-title">01 Hausman Test — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./03_practical_model_choice.md">
  <div class="link-card-title">03 Practical Model Choice</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_practical_model_choice.md">
  <div class="link-card-title">03 Practical Model Choice — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

