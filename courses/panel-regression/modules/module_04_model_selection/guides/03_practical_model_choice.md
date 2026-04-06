# Practical Guide to Panel Model Selection

> **Reading time:** ~19 min | **Module:** 04 — Model Selection | **Prerequisites:** Module 3


## Overview


<div class="callout-key">

**Key Concept Summary:** Choosing the right panel model involves balancing statistical criteria with practical considerations. This guide provides a decision framework.

</div>

Choosing the right panel model involves balancing statistical criteria with practical considerations. This guide provides a decision framework.

## The Decision Tree

<div class="callout-insight">

**Insight:** Panel data lets you control for unobservable differences between entities that are constant over time. This is the single most important reason to prefer panel data over repeated cross-sections.

</div>


```
                    Do entity effects exist?
                           |
                    F-test / BP-LM
                           |
              ┌────────────┴────────────┐
              │                         │
           Yes (p<0.05)              No (p≥0.05)
              │                         │
              │                    Pooled OLS
              │
    Are effects correlated with X?
              |
        Hausman Test
              |
    ┌─────────┴─────────┐
    │                   │
 Yes (p<0.05)       No (p≥0.05)
    │                   │
Fixed Effects      Random Effects
```

## Step-by-Step Selection Process

<div class="callout-warning">

**Warning:** Reporting results without appropriate standard errors is a common mistake. In panel data, conventional OLS standard errors are almost always wrong -- use clustered or heteroskedasticity-robust standard errors.

</div>


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS
from scipy import stats

class PanelModelSelector:
    """
    Comprehensive panel model selection framework.
    """

    def __init__(self, df, y_col, x_cols, entity_col, time_col):
        self.df = df.copy()
        self.y_col = y_col
        self.x_cols = x_cols
        self.entity_col = entity_col
        self.time_col = time_col

        # Set up panel structure
        self.df_panel = df.set_index([entity_col, time_col])

        # Store test results
        self.tests = {}
        self.recommendation = None

    def run_selection(self, verbose=True):
        """
        Run complete model selection procedure.
        """
        if verbose:
            print("=" * 70)
            print("PANEL MODEL SELECTION PROCEDURE")
            print("=" * 70)

        # Step 1: Test for entity effects
        self._test_entity_effects(verbose)

        if not self.tests['entity_effects_significant']:
            self.recommendation = 'pooled_ols'
            if verbose:
                print("\n→ RECOMMENDATION: Pooled OLS")
            return self.recommendation

        # Step 2: Hausman test (FE vs RE)
        self._hausman_test(verbose)

        if self.tests['hausman_reject']:
            self.recommendation = 'fixed_effects'
            if verbose:
                print("\n→ RECOMMENDATION: Fixed Effects")
        else:
            self.recommendation = 'random_effects'
            if verbose:
                print("\n→ RECOMMENDATION: Random Effects")

        # Step 3: Check for additional issues
        self._additional_diagnostics(verbose)

        return self.recommendation

    def _test_entity_effects(self, verbose=True):
        """Test for presence of entity effects."""
        # F-test
        pooled = PooledOLS(self.df_panel[self.y_col],
                           self.df_panel[self.x_cols]).fit()
        fe = PanelOLS(self.df_panel[self.y_col],
                      self.df_panel[self.x_cols],
                      entity_effects=True).fit()

        N = self.df[self.entity_col].nunique()
        n = len(self.df)
        K = len(self.x_cols)

        rss_r = pooled.resid_ss
        rss_u = fe.resid_ss

        f_stat = ((rss_r - rss_u) / (N - 1)) / (rss_u / (n - N - K))
        f_pval = 1 - stats.f.cdf(f_stat, N - 1, n - N - K)

        self.tests['f_stat'] = f_stat
        self.tests['f_pval'] = f_pval
        self.tests['entity_effects_significant'] = f_pval < 0.05

        if verbose:
            print("\n1. TEST FOR ENTITY EFFECTS (F-test)")
            print(f"   F-statistic: {f_stat:.4f}")
            print(f"   p-value: {f_pval:.6f}")
            print(f"   Result: {'Significant' if f_pval < 0.05 else 'Not significant'}")

    def _hausman_test(self, verbose=True):
        """Hausman test for FE vs RE."""
        fe = PanelOLS(self.df_panel[self.y_col],
                      self.df_panel[self.x_cols],
                      entity_effects=True).fit()
        re = RandomEffects(self.df_panel[self.y_col],
                           self.df_panel[self.x_cols]).fit()

        # Coefficient difference
        b_fe = fe.params.values
        b_re = re.params.values

        # Variance of difference
        V_fe = fe.cov.values
        V_re = re.cov.values
        V_diff = V_fe - V_re

        # Hausman statistic
        diff = b_fe - b_re
        try:
            H = diff.T @ np.linalg.inv(V_diff) @ diff
            h_pval = 1 - stats.chi2.cdf(H, len(self.x_cols))
        except:
            # If matrix is singular, use robust alternative
            H = np.sum(diff ** 2 / np.diag(V_diff))
            h_pval = 1 - stats.chi2.cdf(H, len(self.x_cols))

        self.tests['hausman_stat'] = H
        self.tests['hausman_pval'] = h_pval
        self.tests['hausman_reject'] = h_pval < 0.05

        if verbose:
            print("\n2. HAUSMAN TEST (FE vs RE)")
            print(f"   H-statistic: {H:.4f}")
            print(f"   p-value: {h_pval:.6f}")
            print(f"   Result: {'Reject RE (use FE)' if h_pval < 0.05 else 'Cannot reject RE'}")

    def _additional_diagnostics(self, verbose=True):
        """Check for serial correlation and heteroskedasticity."""
        if verbose:
            print("\n3. ADDITIONAL DIAGNOSTICS")

        # Fit recommended model
        if self.recommendation == 'fixed_effects':
            model = PanelOLS(self.df_panel[self.y_col],
                             self.df_panel[self.x_cols],
                             entity_effects=True).fit()
        elif self.recommendation == 'random_effects':
            model = RandomEffects(self.df_panel[self.y_col],
                                  self.df_panel[self.x_cols]).fit()
        else:
            model = PooledOLS(self.df_panel[self.y_col],
                              self.df_panel[self.x_cols]).fit()

        residuals = model.resids

        # Check serial correlation (simplified)
        df_temp = self.df.copy()
        df_temp['resid'] = residuals.values
        df_temp = df_temp.sort_values([self.entity_col, self.time_col])
        df_temp['resid_lag'] = df_temp.groupby(self.entity_col)['resid'].shift(1)
        df_temp = df_temp.dropna()

        autocorr = df_temp['resid'].corr(df_temp['resid_lag'])

        if verbose:
            print(f"   Residual autocorrelation: {autocorr:.4f}")
            if abs(autocorr) > 0.2:
                print("   ⚠ Serial correlation detected - cluster standard errors")

        # Check heteroskedasticity
        entity_vars = df_temp.groupby(self.entity_col)['resid'].var()
        cv_var = entity_vars.std() / entity_vars.mean()

        if verbose:
            print(f"   CV of entity variances: {cv_var:.4f}")
            if cv_var > 0.5:
                print("   ⚠ Heteroskedasticity detected - use robust SE")

        self.tests['autocorr'] = autocorr
        self.tests['cv_variance'] = cv_var

    def fit_recommended(self, cluster=True):
        """
        Fit the recommended model with appropriate standard errors.
        """
        if self.recommendation is None:
            self.run_selection(verbose=False)

        cov_type = 'clustered' if cluster else 'unadjusted'

        if self.recommendation == 'fixed_effects':
            model = PanelOLS(self.df_panel[self.y_col],
                             self.df_panel[self.x_cols],
                             entity_effects=True)
            if cluster:
                return model.fit(cov_type='clustered', cluster_entity=True)
            return model.fit()

        elif self.recommendation == 'random_effects':
            model = RandomEffects(self.df_panel[self.y_col],
                                  self.df_panel[self.x_cols])
            return model.fit()

        else:
            model = PooledOLS(self.df_panel[self.y_col],
                              self.df_panel[self.x_cols])
            if cluster:
                return model.fit(cov_type='clustered',
                                 cluster_entity=True)
            return model.fit()

# Example usage
np.random.seed(42)
n_entities = 100
n_periods = 12

# Generate data with entity effects
data = []
for i in range(n_entities):
    alpha_i = np.random.normal(0, 2)
    for t in range(n_periods):
        x1 = np.random.normal(5, 1) + 0.4 * alpha_i
        x2 = np.random.normal(3, 0.5)
        y = 2 + 1.5 * x1 - 0.8 * x2 + alpha_i + np.random.normal(0, 0.5)
        data.append({'entity': i, 'time': t, 'x1': x1, 'x2': x2, 'y': y})

df = pd.DataFrame(data)

# Run selection
selector = PanelModelSelector(df, 'y', ['x1', 'x2'], 'entity', 'time')
recommendation = selector.run_selection()

# Fit recommended model
results = selector.fit_recommended()
print("\n" + "=" * 70)
print("FINAL MODEL RESULTS")
print("=" * 70)
print(results.summary.tables[1])
```


</div>
</div>

## Practical Considerations Beyond Tests

### 1. Research Question Matters


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def consider_research_question():
    """
    Model choice depends on what you want to learn.
    """
    scenarios = {
        'within_entity_effect': {
            'question': 'How does X affect Y for the same entity over time?',
            'model': 'Fixed Effects',
            'reason': 'Eliminates all between-entity variation'
        },
        'population_average_effect': {
            'question': 'What is the average effect of X on Y across entities?',
            'model': 'Random Effects (if valid) or Population-Averaged GEE',
            'reason': 'Uses both within and between variation'
        },
        'time_invariant_effects': {
            'question': 'How do time-constant factors (gender, region) affect Y?',
            'model': 'Random Effects or CRE',
            'reason': 'FE eliminates time-invariant variables'
        },
        'prediction': {
            'question': 'Predict Y for new entities?',
            'model': 'Random Effects',
            'reason': 'Provides entity-level predictions'
        }
    }

    print("MODEL CHOICE BY RESEARCH QUESTION:")
    print("=" * 70)
    for key, info in scenarios.items():
        print(f"\nScenario: {key}")
        print(f"  Question: {info['question']}")
        print(f"  Model: {info['model']}")
        print(f"  Reason: {info['reason']}")

consider_research_question()
```


</div>
</div>

### 2. Data Characteristics


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def assess_data_characteristics(df, entity_col, time_col, x_cols):
    """
    Assess data structure for model choice implications.
    """
    print("DATA CHARACTERISTICS ASSESSMENT:")
    print("=" * 70)

    N = df[entity_col].nunique()
    T_values = df.groupby(entity_col).size()
    T_mean = T_values.mean()
    T_min = T_values.min()
    T_max = T_values.max()

    print(f"\nPanel Dimensions:")
    print(f"  N (entities): {N}")
    print(f"  T (periods): {T_mean:.1f} (range: {T_min}-{T_max})")

    # Large N, small T
    if N > 100 and T_mean < 10:
        print("\n  → Large N, small T: FE is typical, watch for Nickell bias if dynamic")

    # Small N, large T
    if N < 50 and T_mean > 20:
        print("\n  → Small N, large T: Time-series methods may be relevant")

    # Balanced vs unbalanced
    if T_min != T_max:
        balance_ratio = T_min / T_max
        print(f"\n  Unbalanced panel (min/max = {balance_ratio:.2f})")
        if balance_ratio < 0.5:
            print("  → Strongly unbalanced: check for attrition bias")

    # Within variation
    print("\nWithin-Entity Variation:")
    for x in x_cols:
        total_var = df[x].var()
        within_var = (df[x] - df.groupby(entity_col)[x].transform('mean')).var()
        within_pct = within_var / total_var * 100

        print(f"  {x}: {within_pct:.1f}% within-entity")
        if within_pct < 10:
            print(f"    ⚠ Low within variation - FE estimates may be imprecise")

assess_data_characteristics(df, 'entity', 'time', ['x1', 'x2'])
```



### 3. Sample Selection Considerations

```python
def discuss_sample_issues():
    """
    Sample selection affects model choice.
    """
    issues = """
SAMPLE SELECTION CONSIDERATIONS:
================================================================================

1. ENTITY SELECTION
   - Fixed sample (all firms in an industry): FE appropriate
   - Random sample from population: RE potentially appropriate
   - Selected sample (e.g., surviving firms): Selection bias concerns

2. ATTRITION
   - Non-random dropout creates bias
   - FE doesn't solve selection on unobservables that change over time
   - Consider selection models (Heckman) if attrition is outcome-dependent

3. ENTRY
   - New entities entering the panel
   - May have different characteristics
   - Test for structural breaks

4. TIME-VARYING SELECTION
   - Observations missing non-randomly
   - Create indicators for missingness patterns
   - Test if results robust to excluding entities with gaps

PRACTICAL ADVICE:
- Document your sample construction
- Test sensitivity to sample restrictions
- Report results for balanced and unbalanced samples
    """
    print(issues)

discuss_sample_issues()
```

## Model Comparison Table

<div class="callout-danger">

**Danger:** Never include a lagged dependent variable in a fixed effects model without using an appropriate estimator (e.g., Arellano-Bond GMM). The within-transformation creates mechanical correlation between the transformed lagged variable and the transformed error, biasing all coefficients.



```python
def create_comparison_table(df, y_col, x_cols, entity_col, time_col):
    """
    Create a comprehensive model comparison table.
    """
    df_panel = df.set_index([entity_col, time_col])

    results = {}

    # Pooled OLS
    pooled = PooledOLS(df_panel[y_col], df_panel[x_cols]).fit()
    results['Pooled OLS'] = {
        'coefs': pooled.params.to_dict(),
        'se': pooled.std_errors.to_dict(),
        'r2': pooled.rsquared
    }

    # Pooled with clustered SE
    pooled_cl = PooledOLS(df_panel[y_col], df_panel[x_cols]).fit(
        cov_type='clustered', cluster_entity=True)
    results['Pooled (Clustered)'] = {
        'coefs': pooled_cl.params.to_dict(),
        'se': pooled_cl.std_errors.to_dict(),
        'r2': pooled_cl.rsquared
    }

    # Fixed Effects
    fe = PanelOLS(df_panel[y_col], df_panel[x_cols],
                  entity_effects=True).fit(cov_type='clustered', cluster_entity=True)
    results['Fixed Effects'] = {
        'coefs': fe.params.to_dict(),
        'se': fe.std_errors.to_dict(),
        'r2': fe.rsquared_within
    }

    # Random Effects
    re = RandomEffects(df_panel[y_col], df_panel[x_cols]).fit()
    results['Random Effects'] = {
        'coefs': re.params.to_dict(),
        'se': re.std_errors.to_dict(),
        'r2': re.rsquared
    }

    # Print comparison
    print("\nMODEL COMPARISON TABLE:")
    print("=" * 90)
    print(f"{'Variable':<15}", end='')
    for model in results.keys():
        print(f"{model:<18}", end='')
    print()
    print("-" * 90)

    for x in x_cols:
        print(f"{x:<15}", end='')
        for model, res in results.items():
            coef = res['coefs'][x]
            se = res['se'][x]
            print(f"{coef:.4f} ({se:.4f})", end='  ')
        print()

    print("-" * 90)
    print(f"{'R²':<15}", end='')
    for model, res in results.items():
        print(f"{res['r2']:.4f}{'':>13}", end='')
    print()
    print("=" * 90)

    return results

comparison = create_comparison_table(df, 'y', ['x1', 'x2'], 'entity', 'time')
```



## Key Takeaways

1. **Follow the decision tree** but don't be mechanical about it

2. **Consider your research question** - different questions need different models

3. **Understand your data** - N, T, balance, within variation all matter

4. **Report multiple specifications** - robustness builds credibility

5. **Always cluster standard errors** by entity at minimum

6. **Document your choices** - transparent reasoning is essential


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

<a class="link-card" href="./02_specification_tests.md">
  <div class="link-card-title">02 Specification Tests</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_specification_tests.md">
  <div class="link-card-title">02 Specification Tests — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

