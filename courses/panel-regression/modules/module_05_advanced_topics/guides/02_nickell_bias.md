# Nickell Bias in Dynamic Panels

> **Reading time:** ~19 min | **Module:** 05 — Advanced Topics | **Prerequisites:** Module 4


## The Problem


<div class="callout-key">

**Key Concept Summary:** When you include a lagged dependent variable in a fixed effects model:

</div>

When you include a lagged dependent variable in a fixed effects model:

$$y_{it} = \alpha_i + \rho y_{i,t-1} + X_{it}\beta + \epsilon_{it}$$

The fixed effects estimator is **inconsistent** as N → ∞ with fixed T.

This is called **Nickell bias** (or dynamic panel bias).

## Why Does Bias Occur?

The within transformation creates correlation between the transformed lagged dependent variable and the transformed error.

### Mathematical Intuition

After within-transformation:

$$\tilde{y}_{it} = \rho \tilde{y}_{i,t-1} + \tilde{X}_{it}\beta + \tilde{\epsilon}_{it}$$

where $\tilde{y}_{it} = y_{it} - \bar{y}_i$

The problem: $\tilde{y}_{i,t-1}$ is correlated with $\tilde{\epsilon}_{it}$ because:
- $\bar{y}_i$ contains $y_{it}$
- $y_{it}$ depends on $\epsilon_{it}$
- Therefore $\tilde{y}_{i,t-1}$ depends on $\epsilon_{it}$


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
import matplotlib.pyplot as plt

def demonstrate_nickell_bias(true_rho=0.5, n_simulations=100):
    """
    Demonstrate Nickell bias with simulations.
    """
    np.random.seed(42)

    T_values = [5, 10, 20, 50, 100]
    N = 100

    results = {T: [] for T in T_values}

    for T in T_values:
        for _ in range(n_simulations):
            # Generate dynamic panel data
            data = []
            for i in range(N):
                alpha_i = np.random.normal(0, 1)
                y_prev = np.random.normal(0, 1)  # Initial value

                for t in range(T):
                    epsilon = np.random.normal(0, 0.5)
                    y = alpha_i + true_rho * y_prev + epsilon
                    data.append({
                        'entity': i, 'time': t,
                        'y': y, 'y_lag': y_prev
                    })
                    y_prev = y

```

<div class="callout-insight">

**Insight:** The Nickell bias is small when T is large relative to N, which is the opposite of the typical panel setting. In short panels (T < 10), the bias can be severe enough to reverse the sign of the lagged dependent variable coefficient.

</div>

```python
            df = pd.DataFrame(data)
            df_panel = df.set_index(['entity', 'time'])

            # FE estimation
            try:
                fe = PanelOLS(df_panel['y'], df_panel[['y_lag']],
                              entity_effects=True).fit()
                results[T].append(fe.params['y_lag'])
            except:
                pass

    # Theoretical bias (Nickell formula)
    # Bias ≈ -(1 + rho) / (T - 1) for small T
    theoretical_bias = {T: -(1 + true_rho) / (T - 1) for T in T_values}

    # Plot results
    fig, ax = plt.subplots(figsize=(12, 6))

    positions = range(len(T_values))
    bp = ax.boxplot([results[T] for T in T_values], positions=positions)

    # Add true value line
    ax.axhline(true_rho, color='green', linestyle='--',
               linewidth=2, label=f'True ρ = {true_rho}')

    # Add theoretical bias
    expected_biased = [true_rho + theoretical_bias[T] for T in T_values]
    ax.scatter(positions, expected_biased, color='red', s=100, marker='x',
               label='Theoretical (with Nickell bias)', zorder=5)

    ax.set_xticks(positions)
    ax.set_xticklabels([f'T={T}' for T in T_values])
    ax.set_xlabel('Number of Time Periods')
    ax.set_ylabel('Estimated ρ')
    ax.set_title('Nickell Bias: FE Estimator in Dynamic Panels')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary
    print("NICKELL BIAS DEMONSTRATION:")
    print("=" * 60)
    print(f"True ρ = {true_rho}")
    print()
    print(f"{'T':<10} {'Mean Est.':<15} {'Bias':<15} {'Theoretical':<15}")
    print("-" * 60)
    for T in T_values:
        mean_est = np.mean(results[T])
        bias = mean_est - true_rho
        theo_bias = theoretical_bias[T]
        print(f"{T:<10} {mean_est:<15.4f} {bias:<15.4f} {theo_bias:<15.4f}")

demonstrate_nickell_bias(true_rho=0.5)
```


</div>
</div>

## Magnitude of Bias

<div class="callout-warning">

**Warning:** Reporting results without appropriate standard errors is a common mistake. In panel data, conventional OLS standard errors are almost always wrong -- use clustered or heteroskedasticity-robust standard errors.

</div>


The Nickell bias formula (first-order approximation):

$$E[\hat{\rho}_{FE}] - \rho \approx -\frac{1+\rho}{T-1}$$

Key insights:
- Bias is **always negative** (downward)
- Bias decreases as T increases
- Bias is larger when true ρ is larger


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
def plot_bias_magnitude():
    """
    Plot Nickell bias as a function of T and rho.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Bias vs T for different rho
    ax1 = axes[0]
    T_range = np.arange(3, 51)

    for rho in [0.3, 0.5, 0.7, 0.9]:
        bias = -(1 + rho) / (T_range - 1)
        ax1.plot(T_range, bias, label=f'ρ = {rho}')

    ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('T (Number of Periods)')
    ax1.set_ylabel('Nickell Bias')
    ax1.set_title('Nickell Bias vs Panel Length')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Bias as % of true rho
    ax2 = axes[1]

    for rho in [0.3, 0.5, 0.7, 0.9]:
        bias = -(1 + rho) / (T_range - 1)
        bias_pct = bias / rho * 100
        ax2.plot(T_range, bias_pct, label=f'ρ = {rho}')

    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('T (Number of Periods)')
    ax2.set_ylabel('Bias as % of True ρ')
    ax2.set_title('Relative Nickell Bias')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Rule of thumb table
    print("\nRULE OF THUMB: When is Nickell bias 'acceptable'?")
    print("=" * 50)
    print("(< 10% bias generally considered manageable)")
    print()
    print(f"{'ρ':<10} {'Min T for <10% bias':<25}")
    print("-" * 35)
    for rho in [0.3, 0.5, 0.7, 0.9]:
        # Solve: -(1+rho)/(T-1) / rho = -0.10
        # T = 1 + (1+rho)/(0.10*rho)
        min_T = int(np.ceil(1 + (1 + rho) / (0.10 * rho)))
        print(f"{rho:<10} {min_T:<25}")

plot_bias_magnitude()
```


</div>
</div>

## Solutions to Nickell Bias

### 1. Anderson-Hsiao Estimator

Use deeper lags as instruments for the differenced equation.


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
from linearmodels.iv import IV2SLS

def anderson_hsiao(df, entity_col, time_col, y_col, x_cols=None):
    """
    Anderson-Hsiao IV estimator for dynamic panels.
    """
    df_ah = df.copy().sort_values([entity_col, time_col])

    # Create differences and lags
    df_ah['y_diff'] = df_ah.groupby(entity_col)[y_col].diff()
    df_ah['y_lag'] = df_ah.groupby(entity_col)[y_col].shift(1)
    df_ah['y_lag_diff'] = df_ah.groupby(entity_col)['y_lag'].diff()
    df_ah['y_lag2'] = df_ah.groupby(entity_col)[y_col].shift(2)  # Instrument

    # Difference X if provided
    if x_cols:
        for x in x_cols:
            df_ah[f'{x}_diff'] = df_ah.groupby(entity_col)[x].diff()

    # Drop missing
    df_ah = df_ah.dropna()

    # IV regression: Δy_t on Δy_{t-1}, instrumented by y_{t-2}
    formula = 'y_diff ~ 1 + [y_lag_diff ~ y_lag2]'

    # Using statsmodels IV
    from statsmodels.regression.linear_model import OLS
    from statsmodels.sandbox.regression.gmm import IV2SLS as smIV

    # First stage
    first_stage = smf.ols('y_lag_diff ~ y_lag2', data=df_ah).fit()

    # Second stage
    df_ah['y_lag_diff_hat'] = first_stage.fittedvalues
    second_stage = smf.ols('y_diff ~ y_lag_diff_hat - 1', data=df_ah).fit()

    print("Anderson-Hsiao IV Estimator:")
    print("=" * 50)
    print(f"First stage F-stat: {first_stage.fvalue:.2f}")
    print(f"Estimated ρ: {second_stage.params['y_lag_diff_hat']:.4f}")
    print(f"Standard error: {second_stage.bse['y_lag_diff_hat']:.4f}")

    return second_stage.params['y_lag_diff_hat'], second_stage

# Example
np.random.seed(42)
N, T = 100, 10
true_rho = 0.6

data = []
for i in range(N):
    alpha_i = np.random.normal(0, 1)
    y_prev = np.random.normal(0, 1)
    for t in range(T):
        epsilon = np.random.normal(0, 0.5)
        y = alpha_i + true_rho * y_prev + epsilon
        data.append({'entity': i, 'time': t, 'y': y})
        y_prev = y

df_dyn = pd.DataFrame(data)

print(f"True ρ = {true_rho}\n")

# Compare FE and Anderson-Hsiao
df_dyn['y_lag'] = df_dyn.groupby('entity')['y'].shift(1)
df_dyn_clean = df_dyn.dropna()
df_panel = df_dyn_clean.set_index(['entity', 'time'])

fe = PanelOLS(df_panel['y'], df_panel[['y_lag']], entity_effects=True).fit()
print(f"FE estimate: {fe.params['y_lag']:.4f} (biased)")

ah_est, ah_model = anderson_hsiao(df_dyn, 'entity', 'time', 'y')
```


</div>
</div>

### 2. Arellano-Bond GMM Estimator

Uses all available lags as instruments (more efficient).

```python
def arellano_bond_basic(df, entity_col, time_col, y_col):
    """
    Simplified Arellano-Bond style GMM.

    Note: For production use, use specialized packages like
    pydynpd or linearmodels' GMM capabilities.
    """
    df_ab = df.copy().sort_values([entity_col, time_col])

    # Create differences
    df_ab['y_diff'] = df_ab.groupby(entity_col)[y_col].diff()
    df_ab['y_lag_diff'] = df_ab.groupby(entity_col)['y_diff'].shift(1)

    # Create instruments: y_{t-2}, y_{t-3}, ... for each period
    for lag in range(2, 5):
        df_ab[f'y_lag{lag}'] = df_ab.groupby(entity_col)[y_col].shift(lag)

    df_ab = df_ab.dropna()

    # GMM with multiple instruments
    instruments = [f'y_lag{i}' for i in range(2, 5)]

    # Two-stage regression (simplified - true AB uses more sophisticated GMM)
    first_stage = smf.ols(
        f'y_lag_diff ~ {" + ".join(instruments)}',
        data=df_ab
    ).fit()

    df_ab['y_lag_diff_hat'] = first_stage.fittedvalues

    second_stage = smf.ols('y_diff ~ y_lag_diff_hat - 1', data=df_ab).fit()

    print("\nArellano-Bond Style GMM (Simplified):")
    print("=" * 50)
    print(f"Instruments: {instruments}")
    print(f"First stage F-stat: {first_stage.fvalue:.2f}")
    print(f"Estimated ρ: {second_stage.params['y_lag_diff_hat']:.4f}")

    return second_stage.params['y_lag_diff_hat']

ab_est = arellano_bond_basic(df_dyn, 'entity', 'time', 'y')
```

### 3. Bias-Corrected FE

For moderately large T, bias correction can work well.

```python
def bias_corrected_fe(df, entity_col, time_col, y_col, x_cols=None):
    """
    Bias-corrected FE estimator for dynamic panels.
    """
    df_bc = df.copy().sort_values([entity_col, time_col])
    df_bc['y_lag'] = df_bc.groupby(entity_col)[y_col].shift(1)
    df_bc = df_bc.dropna()

    df_panel = df_bc.set_index([entity_col, time_col])

    # Standard FE estimate
    if x_cols:
        regressors = ['y_lag'] + x_cols
    else:
        regressors = ['y_lag']

    fe = PanelOLS(df_panel[y_col], df_panel[regressors], entity_effects=True).fit()
    rho_fe = fe.params['y_lag']

    # Bias correction (first-order)
    T = df_bc.groupby(entity_col).size().mean()
    bias = -(1 + rho_fe) / (T - 1)

    rho_corrected = rho_fe - bias

    print("Bias-Corrected FE Estimator:")
    print("=" * 50)
    print(f"FE estimate:        {rho_fe:.4f}")
    print(f"Estimated bias:     {bias:.4f}")
    print(f"Corrected estimate: {rho_corrected:.4f}")

    return rho_corrected, rho_fe, bias

bc_rho, fe_rho, bias = bias_corrected_fe(df_dyn, 'entity', 'time', 'y')
```

## Comparison of Methods

```python
def compare_dynamic_panel_methods(true_rho=0.6, N=100, T=10, n_simulations=200):
    """
    Compare different methods for dynamic panel estimation.
    """
    np.random.seed(42)

    results = {
        'FE': [],
        'FE_corrected': [],
        'Anderson_Hsiao': []
    }

    for _ in range(n_simulations):
        # Generate data
        data = []
        for i in range(N):
            alpha_i = np.random.normal(0, 1)
            y_prev = np.random.normal(0, 1)
            for t in range(T):
                epsilon = np.random.normal(0, 0.5)
                y = alpha_i + true_rho * y_prev + epsilon
                data.append({'entity': i, 'time': t, 'y': y})
                y_prev = y

        df_sim = pd.DataFrame(data)
        df_sim['y_lag'] = df_sim.groupby('entity')['y'].shift(1)
        df_sim = df_sim.dropna()

        # FE
        df_panel = df_sim.set_index(['entity', 'time'])
        try:
            fe = PanelOLS(df_panel['y'], df_panel[['y_lag']], entity_effects=True).fit()
            results['FE'].append(fe.params['y_lag'])

            # Bias corrected
            bias = -(1 + fe.params['y_lag']) / (T - 1)
            results['FE_corrected'].append(fe.params['y_lag'] - bias)
        except:
            pass

        # Anderson-Hsiao (simplified inline)
        df_ah = df_sim.copy()
        df_ah['y_diff'] = df_ah.groupby('entity')['y'].diff()
        df_ah['y_lag_diff'] = df_ah.groupby('entity')['y_lag'].diff()
        df_ah['y_lag2'] = df_ah.groupby('entity')['y'].shift(2)
        df_ah = df_ah.dropna()

        try:
            first = smf.ols('y_lag_diff ~ y_lag2', data=df_ah).fit()
            df_ah['fitted'] = first.fittedvalues
            second = smf.ols('y_diff ~ fitted - 1', data=df_ah).fit()
            results['Anderson_Hsiao'].append(second.params['fitted'])
        except:
            pass

    # Visualize
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = ['FE\n(Biased)', 'FE\n(Corrected)', 'Anderson-\nHsiao']
    data_plot = [results['FE'], results['FE_corrected'], results['Anderson_Hsiao']]

    bp = ax.boxplot(data_plot, labels=labels)
    ax.axhline(true_rho, color='red', linestyle='--', linewidth=2,
               label=f'True ρ = {true_rho}')
    ax.set_ylabel('Estimated ρ')
    ax.set_title(f'Comparison of Dynamic Panel Estimators (N={N}, T={T})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Summary table
    print("\nCOMPARISON OF ESTIMATORS:")
    print("=" * 60)
    print(f"True ρ = {true_rho}, N = {N}, T = {T}")
    print()
    print(f"{'Method':<20} {'Mean':<12} {'Bias':<12} {'RMSE':<12}")
    print("-" * 56)

    for method, ests in results.items():
        mean = np.mean(ests)
        bias = mean - true_rho
        rmse = np.sqrt(np.mean((np.array(ests) - true_rho) ** 2))
        print(f"{method:<20} {mean:<12.4f} {bias:<12.4f} {rmse:<12.4f}")

compare_dynamic_panel_methods(true_rho=0.6, N=100, T=10)
```

## When to Worry About Nickell Bias

<div class="callout-danger">

**Danger:** Never include a lagged dependent variable in a fixed effects model without using an appropriate estimator (e.g., Arellano-Bond GMM). The within-transformation creates mechanical correlation between the transformed lagged variable and the transformed error, biasing all coefficients.

</div>


| Scenario | Concern Level | Action |
|----------|--------------|--------|
| T < 10 | High | Use IV/GMM methods |
| 10 ≤ T < 20 | Moderate | Consider bias correction |
| 20 ≤ T < 50 | Low | FE often acceptable |
| T ≥ 50 | Minimal | Standard FE is fine |

## Key Takeaways

1. **Nickell bias is always negative** - FE underestimates persistence

2. **Bias severity depends on T** - larger T means smaller bias

3. **Anderson-Hsiao** uses lagged levels as instruments for differenced equation

4. **Arellano-Bond GMM** is more efficient but complex

5. **Bias correction** works for moderate T

6. **Rule of thumb**: If T > 20-30, Nickell bias is usually acceptable


---

## Conceptual Practice Questions

**Practice Question 1:** Why does including a lagged dependent variable in a fixed effects model create bias, and in which direction?

**Practice Question 2:** How does the Arellano-Bond GMM estimator address the Nickell bias problem?


---

## Cross-References

<a class="link-card" href="./01_dynamic_panels.md">
  <div class="link-card-title">01 Dynamic Panels</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_dynamic_panels.md">
  <div class="link-card-title">01 Dynamic Panels — Companion Slides</div>
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

