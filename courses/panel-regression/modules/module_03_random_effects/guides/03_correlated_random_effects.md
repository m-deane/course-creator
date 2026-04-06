# Correlated Random Effects: Bridging FE and RE

> **Reading time:** ~19 min | **Module:** 03 — Random Effects | **Prerequisites:** Module 2


## The CRE Approach

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

**Key Concept Summary:** Correlated Random Effects (CRE), also known as the Mundlak approach, relaxes the RE assumption while retaining its advantages.

</div>

Correlated Random Effects (CRE), also known as the Mundlak approach, relaxes the RE assumption while retaining its advantages.

The key insight: Model the correlation between $u_i$ and $X_{it}$ explicitly:

$$u_i = \gamma \bar{X}_i + \omega_i$$

Where $\omega_i$ is uncorrelated with $X_{it}$.

## The Mundlak Specification

Substituting into the RE model:

$$y_{it} = \alpha + X_{it}\beta + \gamma \bar{X}_i + \omega_i + \epsilon_{it}$$

This:
- Controls for entity-level confounding via $\bar{X}_i$
- Recovers the FE estimate of $\beta$
- Allows estimation of time-invariant effects
- Provides a natural Hausman test


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS, RandomEffects
import matplotlib.pyplot as plt

def demonstrate_cre():
    """
    Demonstrate the Correlated Random Effects approach.
    """
    np.random.seed(42)

    n_entities = 100
    n_periods = 15
    true_beta = 1.5

    # Generate data with endogeneity
    data = []
    for i in range(n_entities):
        # Entity effect
        u_i = np.random.normal(0, 2)

        # Time-invariant characteristic (observable)
        z_i = np.random.normal(0, 1)

        for t in range(n_periods):
            # X correlated with u_i (endogeneity)
            x = 5 + 0.6 * u_i + np.random.normal(0, 1)

            # Y depends on X, u_i, and z_i
            y = 3 + true_beta * x + 0.8 * z_i + u_i + np.random.normal(0, 0.5)

<div class="callout-insight">

**Insight:** Random effects assumes the unobserved entity effect is uncorrelated with regressors. When this holds, RE is more efficient than FE because it uses both within- and between-entity variation.

</div>


            data.append({
                'entity': i, 'time': t,
                'x': x, 'y': y, 'z': z_i
            })

    df = pd.DataFrame(data)
    df_panel = df.set_index(['entity', 'time'])

    # Add entity means of X (Mundlak approach)
    df['x_bar'] = df.groupby('entity')['x'].transform('mean')

    print("=" * 70)
    print("CORRELATED RANDOM EFFECTS DEMONSTRATION")
    print("=" * 70)
    print(f"True β = {true_beta}")
    print()

    # 1. Pooled OLS (biased due to endogeneity)
    pooled = smf.ols('y ~ x', data=df).fit()
    print(f"1. Pooled OLS:         β = {pooled.params['x']:.4f} (biased)")

    # 2. Random Effects (biased)
    re = RandomEffects(df_panel['y'], df_panel[['x']]).fit()
    print(f"2. Random Effects:     β = {re.params['x']:.4f} (biased)")

    # 3. Fixed Effects (consistent)
    fe = PanelOLS(df_panel['y'], df_panel[['x']], entity_effects=True).fit()
    print(f"3. Fixed Effects:      β = {fe.params['x']:.4f} (consistent)")

    # 4. CRE / Mundlak (consistent, and can include z)
    cre = smf.mixedlm('y ~ x + x_bar + z', data=df, groups='entity').fit()
    print(f"4. CRE/Mundlak:        β = {cre.params['x']:.4f} (consistent)")
    print(f"   Effect of z:        γ = {cre.params['z']:.4f}")

    print()
    print("Note: CRE recovers the FE estimate while also estimating time-invariant effects")

    return df, cre

df, cre_model = demonstrate_cre()
```


</div>

## Why CRE Works

<div class="callout-warning">

**Warning:** Reporting results without appropriate standard errors is a common mistake. In panel data, conventional OLS standard errors are almost always wrong -- use clustered or heteroskedasticity-robust standard errors.

</div>


### Mathematical Intuition

Including $\bar{X}_i$ in the model:

1. **Absorbs the correlation** between $X_{it}$ and $u_i$
2. **The within-variation** in $X_{it}$ identifies $\beta$ (same as FE)
3. **Time-invariant variables** remain estimable (unlike FE)


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def visualize_cre_mechanics(df):
    """
    Visualize how CRE partitions variation.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Calculate components
    df['x_within'] = df['x'] - df.groupby('entity')['x'].transform('mean')
    df['x_bar'] = df.groupby('entity')['x'].transform('mean')

    # Sample entities for visualization
    sample_entities = df['entity'].unique()[:8]
    df_sample = df[df['entity'].isin(sample_entities)]

    # 1. Raw X variation
    ax1 = axes[0]
    for entity in sample_entities:
        entity_data = df_sample[df_sample['entity'] == entity]
        ax1.scatter(entity_data['x'], entity_data['y'], alpha=0.5, s=20)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Total Variation in X')

    # 2. Between variation (entity means)
    ax2 = axes[1]
    entity_means = df.groupby('entity')[['x', 'y']].mean()
    ax2.scatter(entity_means['x'], entity_means['y'], alpha=0.6, s=30)

    # Fit between regression
    between_fit = smf.ols('y ~ x', data=entity_means).fit()
    x_range = np.linspace(entity_means['x'].min(), entity_means['x'].max(), 100)
    ax2.plot(x_range, between_fit.params['Intercept'] + between_fit.params['x'] * x_range,
             'r-', linewidth=2, label=f'Between: β={between_fit.params["x"]:.2f}')
    ax2.set_xlabel('Entity Mean X')
    ax2.set_ylabel('Entity Mean Y')
    ax2.set_title('Between Variation (x̄ᵢ controls for this)')
    ax2.legend()

    # 3. Within variation (demeaned)
    ax3 = axes[2]
    y_within = df_sample['y'] - df_sample.groupby('entity')['y'].transform('mean')
    for entity in sample_entities:
        entity_data = df_sample[df_sample['entity'] == entity]
        y_w = entity_data['y'] - entity_data['y'].mean()
        ax3.scatter(entity_data['x_within'], y_w, alpha=0.5, s=20)

    # Fit within regression
    df['y_within'] = df['y'] - df.groupby('entity')['y'].transform('mean')
    within_fit = smf.ols('y_within ~ x_within - 1', data=df).fit()
    x_w_range = np.linspace(df['x_within'].min(), df['x_within'].max(), 100)
    ax3.plot(x_w_range, within_fit.params['x_within'] * x_w_range,
             'r-', linewidth=2, label=f'Within: β={within_fit.params["x_within"]:.2f}')
    ax3.set_xlabel('X - X̄ᵢ')
    ax3.set_ylabel('Y - Ȳᵢ')
    ax3.set_title('Within Variation (identifies β)')
    ax3.legend()

    plt.tight_layout()
    plt.show()

    print("\nCRE Decomposition:")
    print(f"  Between coefficient (biased): {between_fit.params['x']:.4f}")
    print(f"  Within coefficient (consistent): {within_fit.params['x_within']:.4f}")
    print("  CRE controls for between variation via x̄ᵢ")

visualize_cre_mechanics(df)
```


</div>

## Implementation Options

### 1. OLS with Group Means


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def cre_ols(df, y_col, x_cols, z_cols, entity_col):
    """
    CRE via OLS with entity means.
    """
    df_cre = df.copy()

    # Add entity means for time-varying X
    for x in x_cols:
        df_cre[f'{x}_bar'] = df_cre.groupby(entity_col)[x].transform('mean')

    # Build formula
    x_terms = ' + '.join(x_cols)
    x_bar_terms = ' + '.join([f'{x}_bar' for x in x_cols])
    z_terms = ' + '.join(z_cols) if z_cols else ''

    formula = f'{y_col} ~ {x_terms} + {x_bar_terms}'
    if z_terms:
        formula += f' + {z_terms}'

    model = smf.ols(formula, data=df_cre).fit(
        cov_type='cluster', cov_kwds={'groups': df_cre[entity_col]}
    )

    return model

# Example
cre_ols_model = cre_ols(df, 'y', ['x'], ['z'], 'entity')
print("\nCRE via OLS:")
print(cre_ols_model.summary().tables[1])
```


</div>

### 2. Mixed Effects (Random Intercepts)

```python
def cre_mixed(df, y_col, x_cols, z_cols, entity_col):
    """
    CRE via mixed effects model.
    """
    df_cre = df.copy()

    # Add entity means
    for x in x_cols:
        df_cre[f'{x}_bar'] = df_cre.groupby(entity_col)[x].transform('mean')

    # Build formula
    x_terms = ' + '.join(x_cols)
    x_bar_terms = ' + '.join([f'{x}_bar' for x in x_cols])
    z_terms = ' + '.join(z_cols) if z_cols else ''

    formula = f'{y_col} ~ {x_terms} + {x_bar_terms}'
    if z_terms:
        formula += f' + {z_terms}'

    model = smf.mixedlm(formula, data=df_cre, groups=entity_col).fit()

    return model

cre_mixed_model = cre_mixed(df, 'y', ['x'], ['z'], 'entity')
print("\nCRE via Mixed Effects:")
print(f"  β (x):     {cre_mixed_model.params['x']:.4f}")
print(f"  γ (x_bar): {cre_mixed_model.params['x_bar']:.4f}")
print(f"  δ (z):     {cre_mixed_model.params['z']:.4f}")
```

## The Mundlak Test

A key advantage of CRE: The coefficient on $\bar{X}_i$ provides a test for endogeneity.

If $\gamma = 0$, there's no correlation between $X$ and $u_i$, and RE is appropriate.

```python
def mundlak_test(df, y_col, x_cols, entity_col):
    """
    Mundlak test for endogeneity (FE vs RE).
    """
    df_test = df.copy()

    # Add entity means
    for x in x_cols:
        df_test[f'{x}_bar'] = df_test.groupby(entity_col)[x].transform('mean')

    # Fit CRE model
    x_terms = ' + '.join(x_cols)
    x_bar_terms = ' + '.join([f'{x}_bar' for x in x_cols])
    formula = f'{y_col} ~ {x_terms} + {x_bar_terms}'

    cre = smf.ols(formula, data=df_test).fit()

    # Test H0: all gamma = 0
    x_bar_cols = [f'{x}_bar' for x in x_cols]

    # F-test for joint significance of x_bar terms
    hypotheses = ' = '.join([f'{col} = 0' for col in x_bar_cols]).replace(' = 0 = ', ' = ')
    if len(x_bar_cols) == 1:
        hypotheses = f'{x_bar_cols[0]} = 0'

    f_test = cre.f_test(' = '.join([f'{col}' for col in x_bar_cols]) + ' = 0' if len(x_bar_cols) == 1
                        else f"({', '.join(x_bar_cols)} = 0)")

    print("Mundlak Test (equivalent to Hausman):")
    print("=" * 50)
    print("H0: E[u_i | X_it] = 0 (RE is appropriate)")
    print("H1: E[u_i | X_it] ≠ 0 (Use FE)")
    print()

    for col in x_bar_cols:
        print(f"  {col}: {cre.params[col]:.4f} (t={cre.tvalues[col]:.2f}, p={cre.pvalues[col]:.4f})")

    print()
    if any(cre.pvalues[col] < 0.05 for col in x_bar_cols):
        print("Conclusion: Reject H0 - significant correlation, use FE")
    else:
        print("Conclusion: Cannot reject H0 - RE may be appropriate")

    return cre

mundlak_result = mundlak_test(df, 'y', ['x'], 'entity')
```

## CRE with Multiple Time-Varying Variables

```python
def full_cre_example():
    """
    CRE with multiple time-varying and time-invariant variables.
    """
    np.random.seed(42)

    n_entities = 150
    n_periods = 12

    data = []
    for i in range(n_entities):
        u_i = np.random.normal(0, 2)

        # Time-invariant characteristics
        z1 = np.random.normal(0, 1)  # e.g., industry
        z2 = np.random.choice([0, 1])  # e.g., region

        for t in range(n_periods):
            # Time-varying X (multiple)
            x1 = 5 + 0.5 * u_i + 0.2 * z1 + np.random.normal(0, 1)
            x2 = 3 + 0.3 * u_i + np.random.normal(0, 1)

            # True model
            y = 2 + 1.5 * x1 - 0.8 * x2 + 0.6 * z1 + 1.2 * z2 + u_i + np.random.normal(0, 0.5)

            data.append({
                'entity': i, 'time': t,
                'x1': x1, 'x2': x2,
                'z1': z1, 'z2': z2,
                'y': y
            })

    df_full = pd.DataFrame(data)
    df_full_panel = df_full.set_index(['entity', 'time'])

    # Add means
    for x in ['x1', 'x2']:
        df_full[f'{x}_bar'] = df_full.groupby('entity')[x].transform('mean')

    print("=" * 70)
    print("FULL CRE EXAMPLE")
    print("=" * 70)
    print("True parameters: β1=1.5, β2=-0.8, δ1=0.6, δ2=1.2")
    print()

    # 1. FE (can't estimate z effects)
    fe = PanelOLS(df_full_panel['y'], df_full_panel[['x1', 'x2']], entity_effects=True).fit()
    print("Fixed Effects (cannot estimate z1, z2):")
    print(f"  β1: {fe.params['x1']:.4f}")
    print(f"  β2: {fe.params['x2']:.4f}")

    # 2. RE (biased for x1, x2)
    re = RandomEffects(df_full_panel['y'], df_full_panel[['x1', 'x2', 'z1', 'z2']]).fit()
    print("\nRandom Effects (biased):")
    print(f"  β1: {re.params['x1']:.4f}")
    print(f"  β2: {re.params['x2']:.4f}")
    print(f"  δ1: {re.params['z1']:.4f}")
    print(f"  δ2: {re.params['z2']:.4f}")

    # 3. CRE (consistent and estimates z)
    cre = smf.mixedlm('y ~ x1 + x2 + x1_bar + x2_bar + z1 + z2',
                       data=df_full, groups='entity').fit()
    print("\nCorrelated Random Effects:")
    print(f"  β1: {cre.params['x1']:.4f} (consistent)")
    print(f"  β2: {cre.params['x2']:.4f} (consistent)")
    print(f"  δ1: {cre.params['z1']:.4f} (estimable!)")
    print(f"  δ2: {cre.params['z2']:.4f} (estimable!)")

    return df_full, cre

df_full, cre_full = full_cre_example()
```

## Comparison: FE vs RE vs CRE

<div class="callout-danger">

**Danger:** Never include a lagged dependent variable in a fixed effects model without using an appropriate estimator (e.g., Arellano-Bond GMM). The within-transformation creates mechanical correlation between the transformed lagged variable and the transformed error, biasing all coefficients.

</div>


| Feature | FE | RE | CRE |
|---------|----|----|-----|
| Consistent with endogeneity | ✓ | ✗ | ✓ |
| Estimates time-invariant effects | ✗ | ✓ | ✓ |
| Efficient | ✗ | ✓ | Moderate |
| Built-in Hausman test | ✗ | ✗ | ✓ |
| Flexibility | Low | Low | High |

## Key Takeaways

1. **CRE bridges FE and RE** by explicitly modeling the correlation between entity effects and regressors

2. **Include entity means** of time-varying X to control for endogeneity

3. **Time-invariant effects remain estimable** unlike in pure FE

4. **Coefficient on $\bar{X}_i$** provides a natural Hausman-type test

5. **CRE is more flexible** - can be extended to nonlinear models, multilevel structures


---

## Conceptual Practice Questions

**Practice Question 1:** What is the key assumption that distinguishes random effects from fixed effects, and when is it likely to be violated?

**Practice Question 2:** Why does random effects estimation produce more efficient estimates than fixed effects when its assumptions hold?


---

## Cross-References

<a class="link-card" href="./01_random_effects_model.md">
  <div class="link-card-title">01 Random Effects Model</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_random_effects_model.md">
  <div class="link-card-title">01 Random Effects Model — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./02_random_effects_assumptions.md">
  <div class="link-card-title">02 Random Effects Assumptions</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_random_effects_assumptions.md">
  <div class="link-card-title">02 Random Effects Assumptions — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

