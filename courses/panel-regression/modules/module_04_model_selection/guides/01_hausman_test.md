# The Hausman Test: FE vs RE Selection

> **Reading time:** ~14 min | **Module:** 04 — Model Selection | **Prerequisites:** Module 3


## The Fundamental Question

<div class="flow">
<div class="flow-step mint">1. Estimate FE Model</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">2. Estimate RE Model</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">3. Compute Test Statistic</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. Interpret Result</div>
</div>


<div class="callout-key">

**Key Concept Summary:** The answer depends on whether entity effects correlate with regressors:

</div>

**Should we use Fixed Effects or Random Effects?**

The answer depends on whether entity effects correlate with regressors:
- If $\text{Cov}(\alpha_i, x_{it}) = 0$: RE is efficient and consistent
- If $\text{Cov}(\alpha_i, x_{it}) \neq 0$: RE is inconsistent, FE is required

## The Hausman Test Logic

<div class="compare">
  <div class="compare-card">
    <div class="header before">Fixed Effects</div>
    <div class="body">
      Consistent even if entity effects correlate with regressors. Less efficient. Cannot estimate time-invariant effects.
    </div>
  </div>
  <div class="compare-card">
    <div class="header after">Random Effects</div>
    <div class="body">
      More efficient when assumptions hold. Can estimate time-invariant effects. Biased if entity effects correlate with regressors.
    </div>
  </div>
</div>

### Key Insight

Under H0 (RE assumptions hold):
- **Both FE and RE are consistent**
- **RE is more efficient**

Under H1 (RE assumptions violated):
- **Only FE is consistent**
- **RE is biased**

### Test Construction

If H0 is true, FE and RE should give similar estimates:

<div class="callout-insight">

**Insight:** The Hausman test is necessary but not sufficient. A non-rejection does not prove RE is correct -- it could mean you lack power. Always complement with economic reasoning about whether unobserved effects are likely correlated with regressors.

</div>


$$H = (\hat{\beta}_{FE} - \hat{\beta}_{RE})'[\text{Var}(\hat{\beta}_{FE}) - \text{Var}(\hat{\beta}_{RE})]^{-1}(\hat{\beta}_{FE} - \hat{\beta}_{RE})$$

Under H0: $H \sim \chi^2(K)$ where $K$ is the number of time-varying regressors.

### Interpretation

| Result | Conclusion | Action |
|--------|------------|--------|
| Fail to reject H0 | No evidence of correlation | RE preferred (efficiency) |
| Reject H0 | Entity effects correlated | FE required (consistency) |

## Implementation

<div class="callout-warning">

**Warning:** Reporting results without appropriate standard errors is a common mistake. In panel data, conventional OLS standard errors are almost always wrong -- use clustered or heteroskedasticity-robust standard errors.

</div>


### Using linearmodels


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects

# Generate data with correlated effects (FE should be chosen)
np.random.seed(42)
N, T = 100, 10

# Entity effects
alpha_i = np.repeat(np.random.randn(N) * 2, T)

# x correlated with alpha_i
x = np.random.randn(N * T) + 0.7 * alpha_i  # CORRELATION!
epsilon = np.random.randn(N * T)
y = 2 + 1.5 * x + alpha_i + epsilon

data = pd.DataFrame({
    'entity': np.repeat(range(N), T),
    'time': np.tile(range(T), N),
    'y': y,
    'x': x
}).set_index(['entity', 'time'])

# Fit both models
exog = sm.add_constant(data['x'])

fe_model = PanelOLS(data['y'], exog, entity_effects=True).fit()
re_model = RandomEffects(data['y'], exog).fit()

# Hausman test (built-in comparison)
from linearmodels.panel import compare

comparison = compare({'FE': fe_model, 'RE': re_model})
print(comparison)

# Manual Hausman test
def hausman_test(fe_results, re_results):
    """
    Perform Hausman test for FE vs RE.
    """
    from scipy import stats

    # Get coefficients (exclude constant for FE)
    beta_fe = fe_results.params.drop('const', errors='ignore')
    beta_re = re_results.params.drop('const', errors='ignore')

    # Ensure same variables
    common_vars = beta_fe.index.intersection(beta_re.index)
    beta_fe = beta_fe[common_vars]
    beta_re = beta_re[common_vars]

    # Variance difference
    var_fe = fe_results.cov[common_vars].loc[common_vars]
    var_re = re_results.cov[common_vars].loc[common_vars]
    var_diff = var_fe - var_re

    # Hausman statistic
    beta_diff = beta_fe - beta_re

    try:
        H = beta_diff @ np.linalg.inv(var_diff) @ beta_diff
    except np.linalg.LinAlgError:
        # Use pseudo-inverse if singular
        H = beta_diff @ np.linalg.pinv(var_diff) @ beta_diff

    # Degrees of freedom
    df = len(common_vars)

    # P-value
    p_value = 1 - stats.chi2.cdf(H, df)

    return {
        'statistic': H,
        'df': df,
        'p_value': p_value,
        'beta_fe': beta_fe,
        'beta_re': beta_re,
        'conclusion': 'Use FE' if p_value < 0.05 else 'RE acceptable'
    }

result = hausman_test(fe_model, re_model)
print(f"\nHausman Test Results:")
print(f"  Statistic: {result['statistic']:.4f}")
print(f"  DF: {result['df']}")
print(f"  P-value: {result['p_value']:.4f}")
print(f"  Conclusion: {result['conclusion']}")
```


</div>
</div>

## Limitations of the Hausman Test

### 1. Power Issues

Low power when:
- Small samples
- Little within variation
- True difference is small


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def hausman_power_simulation(N, T, correlation, n_sims=500):
    """Simulate power of Hausman test."""
    rejections = 0

    for _ in range(n_sims):
        # Generate data
        alpha_i = np.repeat(np.random.randn(N), T)
        x = np.random.randn(N * T) + correlation * alpha_i
        epsilon = np.random.randn(N * T)
        y = 1 + x + alpha_i + epsilon

        df = pd.DataFrame({
            'entity': np.repeat(range(N), T),
            'time': np.tile(range(T), N),
            'y': y, 'x': x
        }).set_index(['entity', 'time'])

        exog = sm.add_constant(df['x'])

        try:
            fe = PanelOLS(df['y'], exog, entity_effects=True).fit()
            re = RandomEffects(df['y'], exog).fit()

            result = hausman_test(fe, re)
            if result['p_value'] < 0.05:
                rejections += 1
        except:
            continue

    return rejections / n_sims

# Test power at different correlations
print("Hausman Test Power:")
for corr in [0.0, 0.3, 0.5, 0.7]:
    power = hausman_power_simulation(50, 5, corr, n_sims=200)
    print(f"  Correlation = {corr}: Power = {power:.1%}")
```


</div>
</div>

### 2. Variance Matrix Issues

The difference $\text{Var}(\hat{\beta}_{FE}) - \text{Var}(\hat{\beta}_{RE})$ may not be positive definite.

Solutions:
- Use robust variance estimators
- Use generalized inverse
- Use auxiliary regression approach

### 3. Only Tests Time-Varying Variables

Cannot test time-invariant variables (they're not in FE).

## The Mundlak Alternative

Instead of choosing between FE and RE, use Mundlak's approach:

$$y_{it} = x_{it}'\beta + \bar{x}_i'\gamma + \alpha_i + \epsilon_{it}$$

Test: $H_0: \gamma = 0$


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# Mundlak test
data_flat = data.reset_index()
data_flat['x_mean'] = data_flat.groupby('entity')['x'].transform('mean')
data_mundlak = data_flat.set_index(['entity', 'time'])

re_mundlak = RandomEffects(
    data_mundlak['y'],
    sm.add_constant(data_mundlak[['x', 'x_mean']])
).fit()

print("Mundlak Test:")
print(f"  Coefficient on x_mean: {re_mundlak.params['x_mean']:.4f}")
print(f"  T-stat: {re_mundlak.tstats['x_mean']:.4f}")
print(f"  P-value: {re_mundlak.pvalues['x_mean']:.4f}")

if re_mundlak.pvalues['x_mean'] < 0.05:
    print("  Conclusion: Entity effects correlated with x. Use FE.")
else:
    print("  Conclusion: No evidence of correlation. RE acceptable.")
```


</div>
</div>

## Practical Decision Framework

```
                        Start
                          │
                          ▼
              ┌───────────────────────┐
              │ Is N "the population" │
              │ or a random sample?   │
              └───────────────────────┘
                    │           │
              Population    Sample
                    │           │
                    ▼           ▼
              Use FE      Run Hausman Test
                                │
                    ┌───────────┴───────────┐
                    │                       │
              Reject H0               Fail to reject
                    │                       │
                    ▼                       ▼
              Use FE                  Use RE
              (consistency)           (efficiency)
```

## Beyond the Hausman Test

<div class="callout-danger">

**Danger:** Never include a lagged dependent variable in a fixed effects model without using an appropriate estimator (e.g., Arellano-Bond GMM). The within-transformation creates mechanical correlation between the transformed lagged variable and the transformed error, biasing all coefficients.

</div>


### Information Criteria

Compare models using AIC/BIC:

```python
def panel_aic_bic(results, n_obs, k_params, n_effects=0):
    """Calculate AIC and BIC for panel models."""
    ll = -0.5 * n_obs * (1 + np.log(2 * np.pi) + np.log(results.resid_ss / n_obs))

    # Total parameters
    k = k_params + n_effects

    aic = -2 * ll + 2 * k
    bic = -2 * ll + k * np.log(n_obs)

    return {'AIC': aic, 'BIC': bic}
```

### Correlated Random Effects (CRE)

A unified approach that nests FE and RE:
1. Run RE with Mundlak terms
2. Get FE coefficients on time-varying variables
3. Get RE efficiency on time-invariant variables

## Key Takeaways

1. **Hausman test** compares FE and RE to detect correlation between effects and regressors

2. **Rejection** → Entity effects correlated → Use FE for consistency

3. **Non-rejection** → RE acceptable → Use RE for efficiency

4. **Limitations**: Low power, variance matrix issues, only tests time-varying variables

5. **Mundlak approach** provides a robust alternative that combines benefits of both

6. **Practical rule**: When in doubt, FE is safer—it's always consistent


---

## Conceptual Practice Questions

**Practice Question 1:** The Hausman test compares FE and RE estimates. If they are very similar, what does that tell you about the correlation between unobserved effects and regressors?

**Practice Question 2:** What are the limitations of relying solely on the Hausman test for model selection?


---

## Cross-References

<a class="link-card" href="./02_specification_tests.md">
  <div class="link-card-title">02 Specification Tests</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_specification_tests.md">
  <div class="link-card-title">02 Specification Tests — Companion Slides</div>
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

