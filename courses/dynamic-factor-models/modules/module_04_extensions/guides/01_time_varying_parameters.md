# Time-Varying Parameters in Dynamic Factor Models

## In Brief

Time-varying parameter (TVP) DFMs allow factor loadings, dynamics, or error variances to evolve over time, adapting to structural breaks like recessions, policy regime changes, or pandemics. By treating parameters as stochastic processes, TVP-DFMs maintain forecast accuracy during periods when constant-parameter models fail catastrophically.

## Key Insight

Economic relationships aren't constant. Pre-2008, housing starts strongly predicted GDP; post-crisis, the relationship weakened. Pre-COVID, supply chains were stable; post-COVID, they're volatile. Constant-parameter DFMs assume Λ and Φ never change—an assumption violated by every major economic shock. Time-varying parameters provide insurance against model misspecification at the cost of increased estimation uncertainty.

---

## Visual Explanation

```
Constant vs Time-Varying Parameters:
═══════════════════════════════════════════════════════════

CONSTANT PARAMETERS (Standard DFM):
   λ_{IP} = 0.8  (always)

   2000  2008  2015  2020  2024
   ──────────────────────────────→
    0.8   0.8   0.8   0.8   0.8    ← Never changes!


TIME-VARYING PARAMETERS:
   λ_{IP,t} evolves over time

   2000  2008  2015  2020  2024
   ──────────────────────────────→
    0.8   0.6   0.7   0.3   0.5    ← Adapts to structural changes!
           ↓            ↓
    Financial     COVID-19
     Crisis      Shock


Evolution Mechanisms:
═══════════════════════════════════════════════════════════

1. RANDOM WALK (Most common):
   λ_t = λ_{t-1} + ω_t,   ω_t ~ N(0, σ²_ω)

   ┌─────────────────────────────────┐
   │  λ_t                            │
   │   │    ╱╲    ╱╲                 │
   │   │   ╱  ╲  ╱  ╲    ╱           │
   │   │  ╱    ╲╱    ╲  ╱            │
   │   │ ╱           ╲╱              │
   │   └─────────────────────────→ t │
   │     Smooth drift                │
   └─────────────────────────────────┘

2. ROLLING WINDOW (Discrete jumps):
   λ_t = estimate on [t-W, t]

   ┌─────────────────────────────────┐
   │  λ_t                            │
   │   ├──────┤                      │
   │   │      ├──────┤               │
   │   │      │      ├──────┤        │
   │   │      │      │      ├──────  │
   │   └─────────────────────────→ t │
   │     Window size W               │
   └─────────────────────────────────┘

3. STRUCTURAL BREAKS (Jump at known dates):
   λ_t = λ_1  if t < T_break
         λ_2  if t ≥ T_break

   ┌─────────────────────────────────┐
   │  λ_t                            │
   │   ────────────┐                 │
   │               │                 │
   │               └────────────────  │
   │               ↑                 │
   │         Break point             │
   │         (e.g., COVID)           │
   └─────────────────────────────────┘
```

---

## Formal Definition

### State-Space TVP-DFM

**Measurement equation** (with time-varying loadings):
$$X_t = \Lambda_t F_t + e_t, \quad e_t \sim N(0, \Sigma_{e,t})$$

**Transition equation** (factor dynamics):
$$F_t = \Phi_t F_{t-1} + \eta_t, \quad \eta_t \sim N(0, Q_t)$$

**Parameter evolution equations:**
$$\Lambda_t = \Lambda_{t-1} + \omega^\Lambda_t, \quad \omega^\Lambda_t \sim N(0, \Sigma_\Lambda)$$
$$\Phi_t = \Phi_{t-1} + \omega^\Phi_t, \quad \omega^\Phi_t \sim N(0, \Sigma_\Phi)$$
$$\log(\sigma_{e,i,t}^2) = \log(\sigma_{e,i,t-1}^2) + \omega^{\sigma}_t, \quad \omega^{\sigma}_t \sim N(0, \sigma_\omega^2)$$

**Key insight:** Parameters become states in an augmented state-space system.

### Augmented State-Space Formulation

Define augmented state:
$$\alpha_t = \begin{bmatrix} F_t \\ \text{vec}(\Lambda_t) \\ \text{vec}(\Phi_t) \end{bmatrix}$$

Then:
$$X_t = Z_t(\alpha_t) + e_t$$
$$\alpha_t = T_t \alpha_{t-1} + R_t \epsilon_t$$

**Challenge:** System is **nonlinear** in states (Λ_t multiplies F_t). Requires extended/unscented Kalman filter or particle filter.

### Simplified Linear Approximation

**Assumption:** Loadings vary slowly, can linearize around current estimate.

**Restricted evolution:**
- Allow only **loadings** to vary: Λ_t ~ random walk
- Keep **dynamics** constant: Φ_t = Φ (fixed)
- Keep **error variances** constant: Σ_e,t = Σ_e

**Estimator:** Two-step procedure
1. Estimate constant-parameter DFM on initial subsample → get Φ̂, Σ̂_e
2. Run Kalman filter with time-varying Λ_t, holding Φ, Σ_e fixed

---

## Intuitive Explanation

### Why Parameters Change

**Scenario:** GDP factor loading on employment

- **2000s (Pre-crisis):** λ_{employment} ≈ 0.7
  - Manufacturing strong, employment closely tracks GDP

- **2010s (Post-crisis):** λ_{employment} ≈ 0.5
  - Service sector dominates, employment less cyclical

- **2020s (Post-COVID):** λ_{employment} ≈ 0.3
  - Remote work, gig economy, structural change

**Constant-parameter model:** Uses average λ ≈ 0.5 → misses current relationship
**Time-varying model:** Adapts λ → maintains forecast accuracy

### Random Walk vs Structural Breaks

**Random walk parameters:**
- **Pro:** Smooth adaptation, no need to specify break dates
- **Con:** Slow to react to abrupt changes
- **Use when:** Gradual structural evolution (demographics, technology)

**Structural break specification:**
- **Pro:** Sharp adjustment at known dates (COVID: March 2020)
- **Con:** Must pre-specify break timing
- **Use when:** Observable regime changes (policy shifts, crises)

**Hybrid:** Random walk with jump at known breaks

### Estimation Challenges

**Curse of dimensionality:** With N=10, r=2:
- Constant parameters: 20 loadings to estimate
- Time-varying (T=100): 2000 loading paths!

**Regularization strategies:**
1. **Strong priors:** Bayesian shrinkage (loadings don't change much)
2. **Factor structure:** Only first r factors vary, rest constant
3. **Rolling windows:** Estimate on [t-W, t], not full history
4. **Forgetting factors:** Exponentially downweight old data

---

## Code Implementation

### Rolling-Window Estimation

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

def rolling_window_dfm(data, window_size=60, step=1, k_factors=3):
    """
    Estimate time-varying DFM via rolling windows.

    Parameters
    ----------
    data : pd.DataFrame (T x N)
        Time series data
    window_size : int
        Estimation window (e.g., 60 months = 5 years)
    step : int
        Re-estimation frequency (1 = every period)
    k_factors : int
        Number of factors

    Returns
    -------
    results : dict
        'loadings': List of loading matrices over time
        'factors': Filtered factors at each time
        'dates': Dates for each estimation
    """
    T = len(data)
    loadings_history = []
    factors_history = []
    dates_history = []

    for t in range(window_size, T, step):
        # Estimation window: [t - window_size, t]
        window_data = data.iloc[t-window_size:t]

        # Estimate DFM on window
        dfm = DynamicFactor(
            window_data,
            k_factors=k_factors,
            factor_order=1,
            error_cov_type='diagonal'
        )

        try:
            dfm_res = dfm.fit(maxiter=500, disp=False)

            # Extract loadings (N x r)
            loadings = dfm_res.params['loading'].reshape(data.shape[1], k_factors)

            # Extract factors (filtered)
            factors = dfm_res.factors.filtered

            loadings_history.append(loadings)
            factors_history.append(factors.iloc[-1].values)  # Last period factor
            dates_history.append(data.index[t-1])

        except:
            # Estimation failed (e.g., non-convergence)
            continue

    return {
        'loadings': loadings_history,
        'factors': np.array(factors_history),
        'dates': dates_history
    }

# Example usage
np.random.seed(42)
dates = pd.date_range('2000-01-01', '2024-01-01', freq='M')
T = len(dates)

# Simulate data with structural break
factor = np.cumsum(np.random.randn(T)) * 0.1
data = pd.DataFrame({
    f'Var{i}': (0.5 + 0.3 * (dates > '2015-01-01')) * factor + 0.5 * np.random.randn(T)
    for i in range(5)
}, index=dates)

results = rolling_window_dfm(data, window_size=60, step=12, k_factors=2)

print(f"Estimated {len(results['loadings'])} rolling windows")
print(f"Loading evolution (Var0 on Factor0):")
for i, date in enumerate(results['dates']):
    print(f"  {date.strftime('%Y-%m')}: {results['loadings'][i][0, 0]:.3f}")
```

### Forgetting Factor (Exponentially Weighted MLE)

```python
def ewma_dfm(data, decay=0.99, k_factors=3):
    """
    Time-varying DFM with exponential forgetting.

    Parameters
    ----------
    decay : float in (0, 1)
        Forgetting factor (0.99 = slow adaptation, 0.9 = fast)

    Returns
    -------
    Weighted DFM giving more weight to recent observations.
    """
    T, N = data.shape

    # Exponential weights: recent obs get weight 1, old obs discounted
    weights = decay ** np.arange(T-1, -1, -1)
    weights /= weights.sum()  # Normalize

    # Weighted DFM (requires custom likelihood)
    # Simplified: use WLS for bridge equation (not full DFM)

    # Placeholder: full implementation requires weighted Kalman filter
    print(f"EWMA weights: recent={weights[-1]:.4f}, oldest={weights[0]:.6f}")

    return weights

# Example
weights = ewma_dfm(data, decay=0.99)
```

### Structural Break at Known Date

```python
def dfm_with_break(data, break_date, k_factors=3):
    """
    Estimate DFM separately before/after known structural break.

    Parameters
    ----------
    break_date : str or pd.Timestamp
        Known break point (e.g., '2020-03-01' for COVID)
    """
    break_date = pd.Timestamp(break_date)

    # Split data
    data_pre = data.loc[:break_date]
    data_post = data.loc[break_date:]

    # Estimate separate DFMs
    dfm_pre = DynamicFactor(data_pre, k_factors=k_factors, factor_order=1)
    dfm_pre_res = dfm_pre.fit(maxiter=500, disp=False)

    dfm_post = DynamicFactor(data_post, k_factors=k_factors, factor_order=1)
    dfm_post_res = dfm_post.fit(maxiter=500, disp=False)

    # Compare loadings
    loadings_pre = dfm_pre_res.params['loading'].reshape(data.shape[1], k_factors)
    loadings_post = dfm_post_res.params['loading'].reshape(data.shape[1], k_factors)

    loading_change = np.abs(loadings_post - loadings_pre).max()

    print(f"Structural break analysis at {break_date.strftime('%Y-%m')}")
    print(f"Max loading change: {loading_change:.3f}")

    if loading_change > 0.3:
        print("→ SIGNIFICANT structural break detected")
    else:
        print("→ Loadings relatively stable")

    return {
        'pre': dfm_pre_res,
        'post': dfm_post_res,
        'loadings_pre': loadings_pre,
        'loadings_post': loadings_post
    }

# Example: COVID break
break_results = dfm_with_break(data, break_date='2020-03-01')
```

---

## Common Pitfalls

### 1. Over-Parameterization

**Problem:** Allowing all parameters to vary → millions of parameters

**Solution:** Vary only subset (e.g., loadings only, keep Φ and Σ_e constant)

### 2. Insufficient Data for TVP Estimation

**Problem:** Need long time series to estimate parameter paths

**Rule of thumb:** Window size ≥ 10 × (number of parameters)

### 3. Ignoring Parameter Uncertainty

**Problem:** Treating time-varying estimates as if known with certainty

**Solution:** Bayesian TVP-DFM with posterior credible intervals

### 4. Misspecifying Evolution Variance

**Problem:** σ²_ω too small → parameters don't adapt; too large → random noise

**Solution:** Cross-validation or marginal likelihood optimization

### 5. Confusing Measurement and Parameter Variation

**Problem:** Is forecast error due to time-varying parameters or just noise?

**Diagnostic:** Compare constant-parameter DFM forecast errors over time. If systematic pattern (e.g., worse post-2015), suggests TVP needed.

---

## Connections

### Builds On
- **Kalman Filter** (Module 2): Extended to time-varying system matrices
- **State-Space Models**: Parameters as augmented states

### Leads To
- **Structural VAR with TVP**: Identify time-varying impulse responses
- **Bayesian DFM**: Shrinkage priors on parameter evolution
- **Machine Learning Integration**: Neural networks for parameter paths

### Related To
- **Markov-Switching Models**: Discrete regime changes
- **Threshold Models**: Parameters depend on state variables
- **Stochastic Volatility**: Time-varying error variances

---

## Practice Problems

### Conceptual

1. **When to use TVP-DFM?**
   - You have T=500, estimate DFM on full sample with RMSE=1.0. Rolling 60-month windows: RMSE=0.8. Is TVP worth the complexity?

2. **Forgetting factor calibration**
   - With λ=0.95 forgetting factor, how many periods until an observation has <5% weight?

3. **Structural break timing**
   - How would you test for unknown break dates? (Hint: Bai-Perron sequential testing)

### Implementation

4. **Rolling window backtesting**
   ```python
   # Implement rolling-window nowcast evaluation:
   # - 5-year estimation windows
   # - Re-estimate every 6 months
   # - Compare to constant-parameter DFM
   # - Report improvement in RMSE
   ```

5. **Loading evolution visualization**
   ```python
   # Plot how factor loadings evolve over time
   # - Use rolling windows
   # - Show 95% confidence bands
   # - Mark known structural break dates
   ```

6. **Chow test for parameter stability**
   ```python
   # Test H0: parameters constant across subsamples
   # - Split at candidate break date
   # - Estimate separate DFMs
   # - Compute likelihood ratio statistic
   ```

### Extension

7. **Bayesian TVP-DFM**
   - Implement simple Gibbs sampler for TVP loadings
   - Prior: λ_t | λ_{t-1} ~ N(λ_{t-1}, 0.01²)
   - Compare posterior loading paths to rolling window estimates

---

## Further Reading

### Essential

- **Primiceri, G. E. (2005).** "Time Varying Structural Vector Autoregressions and Monetary Policy." *Review of Economic Studies, 72*(3), 821-852.
  - *Foundational TVP-VAR paper, extends to DFM*

- **Del Negro, M., & Otrok, C. (2008).** "Dynamic Factor Models with Time-Varying Parameters: Measuring Changes in International Business Cycles." *Federal Reserve Bank of New York Staff Report 326.*
  - *TVP-DFM for international data*

### Advanced

- **Koop, G., & Korobilis, D. (2013).** "Large Time-Varying Parameter VARs." *Journal of Econometrics, 177*(2), 185-198.
  - *Computational methods for high-dimensional TVP models*

---

**Next Guide:** Mixed-Frequency Models - combining monthly and quarterly data
