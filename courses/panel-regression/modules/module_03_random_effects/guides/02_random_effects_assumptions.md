# Random Effects Assumptions and GLS Estimation

> **Reading time:** ~19 min | **Module:** 03 — Random Effects | **Prerequisites:** Module 2


## The Random Effects Model

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

**Key Concept Summary:** Random Effects (RE) treats entity-specific effects as random draws from a distribution:

</div>

Random Effects (RE) treats entity-specific effects as random draws from a distribution:

$$y_{it} = \alpha + X_{it}\beta + u_i + \epsilon_{it}$$

Where:
- $u_i \sim N(0, \sigma_u^2)$ is the random entity effect
- $\epsilon_{it} \sim N(0, \sigma_\epsilon^2)$ is the idiosyncratic error
- $u_i$ and $\epsilon_{it}$ are independent

## Key Assumptions

### 1. Zero Correlation with Regressors

The critical assumption:

$$E[u_i | X_{it}] = 0 \quad \forall t$$

This means the entity effect is uncorrelated with all regressors across all time periods.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS, RandomEffects
import matplotlib.pyplot as plt

def demonstrate_re_assumption_violation():
    """
    Show what happens when RE assumption is violated.
    """
    np.random.seed(42)
    n_entities = 100
    n_periods = 15

    # Generate data
    data = []

    for i in range(n_entities):
        # Entity effect
        u_i = np.random.normal(0, 2)

        for t in range(n_periods):
            # Case 1: X correlated with u_i (violates RE assumption)
            x_correlated = 5 + 0.7 * u_i + np.random.normal(0, 1)

            # Case 2: X uncorrelated with u_i (RE assumption holds)
            x_uncorrelated = 5 + np.random.normal(0, 1)

            # Y depends on X and u_i
            y = 3 + 1.5 * x_correlated + u_i + np.random.normal(0, 0.5)

            data.append({
                'entity': i, 'time': t,
                'x_bad': x_correlated,  # Correlated with u_i
                'x_good': x_uncorrelated,  # Uncorrelated with u_i
                'y': y,
                'u_i': u_i
            })

    df = pd.DataFrame(data)
    df_panel = df.set_index(['entity', 'time'])

    print("=" * 60)
    print("RANDOM EFFECTS ASSUMPTION DEMONSTRATION")
    print("=" * 60)

    # Case 1: RE assumption violated
    print("\n--- Case 1: X correlated with entity effect (VIOLATION) ---")
    print(f"Correlation(X, u_i): {df.groupby('entity')[['x_bad', 'u_i']].mean().corr().iloc[0,1]:.3f}")

    fe_bad = PanelOLS(df_panel['y'], df_panel[['x_bad']], entity_effects=True).fit()
    re_bad = RandomEffects(df_panel['y'], df_panel[['x_bad']]).fit()

    print(f"True β = 1.5")
    print(f"FE estimate: {fe_bad.params['x_bad']:.4f}")
    print(f"RE estimate: {re_bad.params['x_bad']:.4f} (BIASED!)")

    # Now with good X (should work for both)
    df['y_good'] = 3 + 1.5 * df['x_good'] + df['u_i'] + np.random.normal(0, 0.5, len(df))
    df_panel = df.set_index(['entity', 'time'])

<div class="callout-insight">

**Insight:** Random effects assumes the unobserved entity effect is uncorrelated with regressors. When this holds, RE is more efficient than FE because it uses both within- and between-entity variation.

</div>


    print("\n--- Case 2: X uncorrelated with entity effect (VALID) ---")
    print(f"Correlation(X, u_i): {df.groupby('entity')[['x_good', 'u_i']].mean().corr().iloc[0,1]:.3f}")

    fe_good = PanelOLS(df_panel['y_good'], df_panel[['x_good']], entity_effects=True).fit()
    re_good = RandomEffects(df_panel['y_good'], df_panel[['x_good']]).fit()

    print(f"True β = 1.5")
    print(f"FE estimate: {fe_good.params['x_good']:.4f}")
    print(f"RE estimate: {re_good.params['x_good']:.4f}")

    return df

df = demonstrate_re_assumption_violation()
```

</div>

### 2. Composite Error Structure

The total error $v_{it} = u_i + \epsilon_{it}$ has a specific covariance structure:

$$Var(v_{it}) = \sigma_u^2 + \sigma_\epsilon^2$$

$$Cov(v_{it}, v_{is}) = \sigma_u^2 \quad (t \neq s)$$

$$Cov(v_{it}, v_{jt}) = 0 \quad (i \neq j)$$

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def visualize_error_structure():
    """
    Visualize the composite error structure.
    """
    np.random.seed(42)

    sigma_u = 2.0
    sigma_eps = 1.0

    n_entities = 5
    n_periods = 20

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Generate errors
    errors = {}
    for i in range(n_entities):
        u_i = np.random.normal(0, sigma_u)
        eps = np.random.normal(0, sigma_eps, n_periods)
        errors[i] = u_i + eps

    # Left: Error paths by entity
    ax1 = axes[0]
    for i, err in errors.items():
        ax1.plot(range(n_periods), err, 'o-', label=f'Entity {i}', alpha=0.7)
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Composite Error (uᵢ + εᵢₜ)')
    ax1.set_title('Error Paths by Entity')
    ax1.legend()

    # Right: Covariance matrix for one entity
    ax2 = axes[1]

    # Theoretical covariance matrix
    cov_matrix = np.full((n_periods, n_periods), sigma_u**2)
    np.fill_diagonal(cov_matrix, sigma_u**2 + sigma_eps**2)

    im = ax2.imshow(cov_matrix, cmap='Blues')
    ax2.set_xlabel('Time s')
    ax2.set_ylabel('Time t')
    ax2.set_title('Covariance Structure: Cov(vᵢₜ, vᵢₛ)')
    plt.colorbar(im, ax=ax2)

    plt.tight_layout()
    plt.show()

    print("Variance Components:")
    print(f"  σ²_u (entity effect):     {sigma_u**2}")
    print(f"  σ²_ε (idiosyncratic):     {sigma_eps**2}")
    print(f"  Total variance:           {sigma_u**2 + sigma_eps**2}")
    print(f"  Within-entity correlation: {sigma_u**2 / (sigma_u**2 + sigma_eps**2):.3f}")

visualize_error_structure()
```

</div>

## GLS Estimation

<div class="callout-warning">

**Warning:** Reporting results without appropriate standard errors is a common mistake. In panel data, conventional OLS standard errors are almost always wrong -- use clustered or heteroskedasticity-robust standard errors.

</div>


### Why OLS Is Inefficient

OLS ignores the error correlation structure, leading to inefficient (though still consistent, if RE assumption holds) estimates.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def compare_ols_gls_efficiency():
    """
    Compare OLS vs GLS efficiency under RE model.
    """
    np.random.seed(42)
    n_sim = 500

    ols_estimates = []
    gls_estimates = []

    true_beta = 2.0

    for _ in range(n_sim):
        data = []
        for i in range(50):
            u_i = np.random.normal(0, 2)
            for t in range(10):
                x = np.random.normal(5, 1)  # X independent of u_i
                eps = np.random.normal(0, 1)
                y = 1 + true_beta * x + u_i + eps
                data.append({'entity': i, 'time': t, 'x': x, 'y': y})

        df_sim = pd.DataFrame(data)
        df_sim_panel = df_sim.set_index(['entity', 'time'])

        # OLS (ignores error structure)
        ols = smf.ols('y ~ x', data=df_sim).fit()
        ols_estimates.append(ols.params['x'])

        # GLS/Random Effects (accounts for error structure)
        re = RandomEffects(df_sim_panel['y'], df_sim_panel[['x']]).fit()
        gls_estimates.append(re.params['x'])

    # Compare distributions
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(ols_estimates, bins=30, alpha=0.5, label=f'OLS (SE={np.std(ols_estimates):.4f})')
    ax.hist(gls_estimates, bins=30, alpha=0.5, label=f'GLS/RE (SE={np.std(gls_estimates):.4f})')
    ax.axvline(true_beta, color='red', linestyle='--', linewidth=2, label=f'True β = {true_beta}')

    ax.set_xlabel('Estimated β')
    ax.set_ylabel('Frequency')
    ax.set_title('OLS vs GLS Efficiency')
    ax.legend()

    plt.tight_layout()
    plt.show()

    print("Efficiency Comparison:")
    print(f"  OLS standard error:  {np.std(ols_estimates):.4f}")
    print(f"  GLS standard error:  {np.std(gls_estimates):.4f}")
    print(f"  Efficiency gain:     {(np.std(ols_estimates) / np.std(gls_estimates) - 1) * 100:.1f}%")

compare_ols_gls_efficiency()
```

</div>

### The GLS Transformation

RE estimation uses a quasi-differencing transformation:

$$y_{it} - \theta \bar{y}_i = (1-\theta)\alpha + (X_{it} - \theta \bar{X}_i)\beta + (v_{it} - \theta \bar{v}_i)$$

Where $\theta = 1 - \sqrt{\frac{\sigma_\epsilon^2}{\sigma_\epsilon^2 + T\sigma_u^2}}$

```python
def demonstrate_gls_transformation():
    """
    Show the GLS quasi-differencing transformation.
    """
    np.random.seed(42)

    # Generate data
    n_entities = 50
    T = 10
    sigma_u = 2.0
    sigma_eps = 1.0

    data = []
    for i in range(n_entities):
        u_i = np.random.normal(0, sigma_u)
        for t in range(T):
            x = np.random.normal(5, 1)
            y = 3 + 1.5 * x + u_i + np.random.normal(0, sigma_eps)
            data.append({'entity': i, 'time': t, 'x': x, 'y': y})

    df = pd.DataFrame(data)

    # Calculate theta
    theta = 1 - np.sqrt(sigma_eps**2 / (sigma_eps**2 + T * sigma_u**2))

    print(f"GLS Transformation Parameter:")
    print(f"  θ = {theta:.4f}")
    print(f"  (θ=0: Pooled OLS, θ=1: Fixed Effects)")

    # Apply transformation
    y_bar = df.groupby('entity')['y'].transform('mean')
    x_bar = df.groupby('entity')['x'].transform('mean')

    df['y_gls'] = df['y'] - theta * y_bar
    df['x_gls'] = df['x'] - theta * x_bar

    # Compare transformations
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original data
    ax1 = axes[0]
    ax1.scatter(df['x'], df['y'], alpha=0.3, s=10)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Original Data')

    # Within transformation (θ=1)
    ax2 = axes[1]
    y_within = df['y'] - y_bar
    x_within = df['x'] - x_bar
    ax2.scatter(x_within, y_within, alpha=0.3, s=10)
    ax2.set_xlabel('X - X̄ᵢ')
    ax2.set_ylabel('Y - Ȳᵢ')
    ax2.set_title('Within Transformation (θ=1, FE)')

    # GLS transformation
    ax3 = axes[2]
    ax3.scatter(df['x_gls'], df['y_gls'], alpha=0.3, s=10)
    ax3.set_xlabel(f'X - {theta:.2f}X̄ᵢ')
    ax3.set_ylabel(f'Y - {theta:.2f}Ȳᵢ')
    ax3.set_title(f'GLS Transformation (θ={theta:.2f}, RE)')

    plt.tight_layout()
    plt.show()

    # Estimate on transformed data
    from scipy import stats

    # GLS estimate
    gls_result = smf.ols('y_gls ~ x_gls', data=df).fit()
    print(f"\nGLS estimate: {gls_result.params['x_gls']:.4f}")

    # Compare to linearmodels
    df_panel = df.set_index(['entity', 'time'])
    re = RandomEffects(df_panel['y'], df_panel[['x']]).fit()
    print(f"Random Effects: {re.params['x']:.4f}")

demonstrate_gls_transformation()
```

## Variance Components Estimation

```python
def estimate_variance_components(df, entity_col, y_col, x_cols):
    """
    Estimate variance components for RE model.
    """
    df_panel = df.set_index([entity_col, 'time'])

    # Get within residuals (from FE)
    fe = PanelOLS(df_panel[y_col], df_panel[x_cols], entity_effects=True).fit()
    within_resid = fe.resids

    # σ²_ε from within residuals
    N = df[entity_col].nunique()
    T = df['time'].nunique()
    K = len(x_cols)

    sigma_eps_sq = (within_resid ** 2).sum() / (N * T - N - K)

    # Get between residuals
    entity_means = df.groupby(entity_col)[[y_col] + x_cols].mean()
    between_model = smf.ols(f"{y_col} ~ {' + '.join(x_cols)}", data=entity_means).fit()
    between_resid = between_model.resid

    sigma_between_sq = (between_resid ** 2).sum() / (N - K - 1)

    # σ²_u = σ²_between - σ²_ε/T
    sigma_u_sq = max(0, sigma_between_sq - sigma_eps_sq / T)

    # Calculate theta
    theta = 1 - np.sqrt(sigma_eps_sq / (sigma_eps_sq + T * sigma_u_sq)) if sigma_u_sq > 0 else 0

    # Intraclass correlation
    rho = sigma_u_sq / (sigma_u_sq + sigma_eps_sq) if (sigma_u_sq + sigma_eps_sq) > 0 else 0

    print("Variance Components Estimation:")
    print("=" * 50)
    print(f"  σ²_ε (idiosyncratic): {sigma_eps_sq:.4f}")
    print(f"  σ²_u (entity effect): {sigma_u_sq:.4f}")
    print(f"  θ (quasi-difference): {theta:.4f}")
    print(f"  ρ (intraclass corr):  {rho:.4f}")

    return {
        'sigma_eps_sq': sigma_eps_sq,
        'sigma_u_sq': sigma_u_sq,
        'theta': theta,
        'rho': rho
    }

# Example
np.random.seed(42)
data = []
for i in range(100):
    u_i = np.random.normal(0, 2)
    for t in range(15):
        x = np.random.normal(5, 1)
        y = 3 + 1.5 * x + u_i + np.random.normal(0, 1)
        data.append({'entity': i, 'time': t, 'x': x, 'y': y})

df_example = pd.DataFrame(data)
components = estimate_variance_components(df_example, 'entity', 'y', ['x'])
```

## Testing RE Assumptions

### Breusch-Pagan LM Test

Tests for the presence of random effects (H0: $\sigma_u^2 = 0$):

```python
from scipy import stats

def breusch_pagan_lm_test(df, entity_col, y_col, x_cols):
    """
    Breusch-Pagan Lagrange Multiplier test for random effects.
    """
    # Pooled OLS residuals
    formula = f"{y_col} ~ {' + '.join(x_cols)}"
    pooled = smf.ols(formula, data=df).fit()
    resid = pooled.resid

    # Sum of residuals by entity
    df['resid'] = resid
    entity_sums = df.groupby(entity_col)['resid'].sum()

    N = df[entity_col].nunique()
    T = df.groupby(entity_col).size().mean()  # Average T
    n_obs = len(df)

    # LM statistic
    numerator = (entity_sums ** 2).sum()
    denominator = (resid ** 2).sum()

    lm_stat = (n_obs / (2 * (T - 1))) * ((T**2 * numerator / denominator) - 1) ** 2

    # Under H0, LM ~ chi-squared(1)
    p_value = 1 - stats.chi2.cdf(lm_stat, 1)

    print("Breusch-Pagan LM Test for Random Effects:")
    print("=" * 50)
    print(f"  H0: σ²_u = 0 (no random effects)")
    print(f"  LM statistic: {lm_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Conclusion: {'Reject H0 - use RE or FE' if p_value < 0.05 else 'Cannot reject H0 - pooled OK'}")

    return lm_stat, p_value

breusch_pagan_lm_test(df_example, 'entity', 'y', ['x'])
```

## Advantages of Random Effects

1. **Efficiency**: Uses both between and within variation
2. **Time-invariant variables**: Can estimate effects of constant covariates
3. **Interpretable**: Variance components have clear meaning
4. **Smaller samples**: Better performance with limited T

## When to Use Random Effects

<div class="callout-danger">

**Danger:** Never include a lagged dependent variable in a fixed effects model without using an appropriate estimator (e.g., Arellano-Bond GMM). The within-transformation creates mechanical correlation between the transformed lagged variable and the transformed error, biasing all coefficients.

</div>


| Condition | Use RE? |
|-----------|---------|
| Entity effect uncorrelated with X | ✓ Yes |
| Time-invariant variables important | ✓ Yes |
| Small T, need efficiency | ✓ Yes |
| Entities are random sample | ✓ Yes |
| Entity effect correlated with X | ✗ No (use FE) |
| Hausman test rejects RE | ✗ No (use FE) |

## Key Takeaways

1. **RE assumes $E[u_i|X_{it}] = 0$** - entity effects uncorrelated with regressors

2. **GLS estimation** exploits error structure for efficiency gains

3. **Theta parameter** determines how much quasi-differencing (0=OLS, 1=FE)

4. **Variance components** decompose total variance into entity and idiosyncratic parts

5. **Test assumptions** before using RE - Hausman test is crucial


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

<a class="link-card" href="./03_correlated_random_effects.md">
  <div class="link-card-title">03 Correlated Random Effects</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_correlated_random_effects.md">
  <div class="link-card-title">03 Correlated Random Effects — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

