# Forecast Evaluation and Comparison

## In Brief

Forecast evaluation goes beyond computing mean squared error—it requires proper scoring rules, real-time data vintages, statistical tests for forecast comparison, and decomposition of forecast errors by source. Rigorous evaluation distinguishes genuinely skillful forecasts from lucky guesses and identifies specific model weaknesses for improvement.

## Key Insight

A forecast that's "usually right" but catastrophically wrong during recessions is worse than one with moderate error in all states. Proper evaluation must penalize distributional mis-specification (overconfident intervals), directional errors (sign mistakes), and systematic biases—not just point forecast accuracy. The gold standard: out-of-sample performance on real-time data vintages.

---

## Visual Explanation

```
Forecast Evaluation Framework:
═══════════════════════════════════════════════════════════

1. Out-of-Sample Framework (Pseudo Real-Time)
   ┌──────────────────────────────────────────────────────┐
   │  Training Window    │  Forecast  │  Realized  │      │
   │  (Expanding/Rolling)│  Period    │  Outcome   │      │
   ├──────────────────────────────────────────────────────┤
   │  2000Q1 ... 2010Q4  │ → 2011Q1   │  2.3%      │      │
   │  2000Q1 ... 2011Q1  │ → 2011Q2   │  1.8%      │      │
   │  2000Q1 ... 2011Q2  │ → 2011Q3   │  0.5%      │      │
   │         ...         │    ...     │   ...      │      │
   │  2000Q1 ... 2023Q4  │ → 2024Q1   │  ???       │      │
   └──────────────────────────────────────────────────────┘
                          ↑                ↑
                    Nowcast at t    Compare to first-release
                                    (not final revised!)


2. Scoring Rules Comparison:
   ┌─────────────────────────────────────────────────────┐
   │ Metric         │ Penalizes        │ Use When        │
   ├─────────────────────────────────────────────────────┤
   │ MSE            │ Large errors²    │ Symmetric loss  │
   │ MAE            │ Abs errors       │ Outlier-robust  │
   │ MAPE           │ % errors         │ Scale-free      │
   │ CRPS           │ Full distribution│ Interval fcst   │
   │ Log Score      │ Probability      │ Density fcst    │
   │ Dir. Accuracy  │ Sign errors      │ Trading signals │
   └─────────────────────────────────────────────────────┘


3. Diebold-Mariano Test Flow:
   ════════════════════════════════════════════════════

   Model A errors: e₁ᴬ, e₂ᴬ, ..., eₜᴬ
   Model B errors: e₁ᴮ, e₂ᴮ, ..., eₜᴮ
                    │
                    ▼
   Loss differential: dₜ = L(e₁ᴬ) - L(eₜᴮ)
                    │
                    ▼
   Test: H₀: E[dₜ] = 0  (equal predictive accuracy)
         Hₐ: E[dₜ] ≠ 0  (one model superior)
                    │
                    ▼
   DM statistic: t = d̄ / (σ̂ₐ/√T)  ~  N(0,1)
                    │
                    ▼
   Decision: |t| > 1.96 → Reject H₀ at 5% level


4. Forecast Combination:
   ═══════════════════════════════════════════════════

   Individual Forecasts:
   ŷₜᴬ (DFM) ──┐
   ŷₜᴮ (AR)  ──┼──→  Combination:  ŷₜᶜ = Σ wᵢ ŷₜⁱ
   ŷₜᶜ (VAR) ──┘
                       ↑
                  Weights: wᵢ ≥ 0, Σwᵢ = 1

   Optimal weights minimize MSE:
   w* = Σₑ⁻¹ · 1 / (1' Σₑ⁻¹ 1)
   where Σₑ = Cov(forecast errors)
```

---

## Formal Definition

### Scoring Rules

A **scoring rule** $S(F, y)$ evaluates a forecast distribution $F$ against realized outcome $y$.

**Proper scoring rule:** Truth-telling is optimal. If true distribution is $G$, then:
$$\mathbb{E}_G[S(G, y)] \leq \mathbb{E}_G[S(F, y)] \quad \forall F \neq G$$

#### Common Scoring Rules

**1. Mean Squared Error (MSE)**
$$\text{MSE} = \frac{1}{T} \sum_{t=1}^T (y_t - \hat{y}_t)^2$$

- Quadratic loss: $L(e) = e^2$
- Optimal point forecast: conditional mean $\mathbb{E}[y_t | \mathcal{I}_t]$
- **Limitation:** Doesn't evaluate uncertainty

**2. Mean Absolute Error (MAE)**
$$\text{MAE} = \frac{1}{T} \sum_{t=1}^T |y_t - \hat{y}_t|$$

- Linear loss: $L(e) = |e|$
- Optimal point forecast: conditional median
- **Advantage:** Robust to outliers

**3. Continuous Ranked Probability Score (CRPS)**

For forecast CDF $F$ and realized $y$:
$$\text{CRPS}(F, y) = \int_{-\infty}^{\infty} [F(z) - \mathbf{1}\{y \leq z\}]^2 dz$$

- Evaluates *entire forecast distribution*
- Reduces to MAE if point forecast
- **Proper for interval and density forecasts**

**4. Logarithmic Score (Log Predictive Density)**
$$\text{LS} = -\frac{1}{T} \sum_{t=1}^T \log p(y_t | \hat{F}_t)$$

- Evaluates forecast density at realized value
- **Heavily penalizes overconfidence**
- Used in Bayesian model comparison (marginal likelihood)

**5. Directional Accuracy**
$$\text{DA} = \frac{1}{T} \sum_{t=1}^T \mathbf{1}\{\text{sign}(\hat{y}_t) = \text{sign}(y_t)\}$$

- Binary: did forecast get direction right?
- Critical for trading signals (long/short decisions)

### Diebold-Mariano Test for Forecast Comparison

**Setup:** Two competing forecasts $\{\hat{y}_t^{(1)}\}$ and $\{\hat{y}_t^{(2)}\}$ for same target $\{y_t\}$.

**Null hypothesis:** Equal predictive accuracy
$$H_0: \mathbb{E}[L(e_t^{(1)}) - L(e_t^{(2)})] = 0$$

**Test statistic:**
$$\text{DM} = \frac{\bar{d}}{\sqrt{\hat{V}(d) / T}} \xrightarrow{d} N(0, 1)$$

where:
- $d_t = L(e_t^{(1)}) - L(e_t^{(2)})$ is loss differential
- $\bar{d} = \frac{1}{T}\sum_{t=1}^T d_t$
- $\hat{V}(d)$ is HAC-robust variance estimator accounting for autocorrelation

**Small-sample correction (Harvey et al., 1997):**
$$\text{DM}^* = \text{DM} \cdot \sqrt{\frac{T + 1 - 2h + T^{-1} h(h-1)}{T}}$$

where $h$ is forecast horizon.

### Forecast Encompassing Test

**Question:** Does forecast A contain *all* information in forecast B (and more)?

**Regression test:**
$$y_t = \alpha + \beta_1 \hat{y}_t^A + \beta_2 \hat{y}_t^B + \epsilon_t$$

- If $\beta_1 = 1, \beta_2 = 0$: A encompasses B (B adds nothing)
- If $\beta_1, \beta_2 \in (0, 1)$: Both contain unique information → combine

---

## Intuitive Explanation

### Why MSE Isn't Enough

**Scenario:** Forecasting GDP growth (mean ≈ 2%, std ≈ 2%)

**Model A:** Always predicts 2.0% (unconditional mean)
- MSE = 4.0
- Never captures turning points
- Useless for policymaking

**Model B:** Predicts 1.5% in recessions, 2.5% in expansions
- MSE = 3.8 (slightly better)
- Captures directional changes
- **Much more valuable** despite small MSE improvement

**Lesson:** Evaluate conditional forecasts, not just unconditional errors.

### The Real-Time Data Problem

**Backtesting pitfall:**

```python
# WRONG: Using final revised GDP
y_true = gdp_final_revised_2024  # Incorporates 2025-2026 revisions!
forecast_error = y_true - y_forecast_2020  # Unfair comparison
```

**Correct approach:**

```python
# RIGHT: Using first-release (advance estimate) GDP
y_advance = gdp_advance_estimate  # As published 30 days after quarter
forecast_error = y_advance - y_forecast  # Fair comparison
```

**Why?** Your forecast in 2020 couldn't incorporate information from 2025 revisions. Evaluate against data *as it was known* at forecast time.

### Forecast Combination: Why Averaging Works

**Diversification of errors:**
- Model A overreacts to employment data
- Model B overreacts to surveys
- Average of A and B: errors partially cancel

**Mathematical intuition:**

If models are unbiased but have uncorrelated errors:
$$\text{Var}(\text{average forecast}) = \frac{1}{n^2} \sum_i \text{Var}(e_i) = \frac{\sigma^2}{n}$$

Even simple averaging reduces variance by factor of $n$!

---

## Code Implementation

### Out-of-Sample Backtesting Framework

```python
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def expanding_window_backtest(data, target_col, predictor_cols,
                               model_fn, start_date, end_date):
    """
    Pseudo-real-time out-of-sample backtesting.

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset including target and predictors
    target_col : str
        Column name of target variable
    predictor_cols : list of str
        Column names of predictors
    model_fn : callable
        Function that takes (X_train, y_train) and returns fitted model
    start_date : str
        First forecast date
    end_date : str
        Last forecast date

    Returns
    -------
    results : pd.DataFrame
        Columns: date, y_true, y_pred, error
    """
    results = []
    dates = pd.date_range(start_date, end_date, freq='Q')

    for date in dates:
        # Training data: all data before forecast date
        train = data.loc[:date].iloc[:-1]  # Exclude current period

        # Fit model on training data
        X_train = train[predictor_cols].values
        y_train = train[target_col].values
        model = model_fn(X_train, y_train)

        # Forecast current period
        X_test = data.loc[date, predictor_cols].values.reshape(1, -1)
        y_pred = model.predict(X_test)[0]

        # Actual realization (use first-release, not final!)
        y_true = data.loc[date, target_col]

        results.append({
            'date': date,
            'y_true': y_true,
            'y_pred': y_pred,
            'error': y_true - y_pred
        })

    return pd.DataFrame(results)

# Example: Simple OLS model
from sklearn.linear_model import LinearRegression

def ols_model(X, y):
    return LinearRegression().fit(X, y)

# Run backtest
results = expanding_window_backtest(
    data=gdp_data,
    target_col='gdp_growth',
    predictor_cols=['factor1', 'factor2', 'factor3'],
    model_fn=ols_model,
    start_date='2010-01-01',
    end_date='2023-12-31'
)
```

### Computing Scoring Rules

```python
def compute_scoring_rules(y_true, y_pred, y_pred_dist=None):
    """
    Compute multiple forecast evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        Realized values
    y_pred : array-like
        Point forecasts
    y_pred_dist : list of scipy.stats distributions (optional)
        Forecast distributions for CRPS and log score

    Returns
    -------
    scores : dict
    """
    errors = y_true - y_pred

    scores = {
        'MSE': np.mean(errors ** 2),
        'RMSE': np.sqrt(np.mean(errors ** 2)),
        'MAE': np.mean(np.abs(errors)),
        'MAPE': np.mean(np.abs(errors / y_true)) * 100,
        'Bias': np.mean(errors),
        'Directional_Accuracy': np.mean(np.sign(y_pred) == np.sign(y_true))
    }

    # CRPS (if distributional forecasts available)
    if y_pred_dist is not None:
        from properscoring import crps_ensemble
        crps_values = [crps_ensemble(y, [dist.rvs(1000)])
                       for y, dist in zip(y_true, y_pred_dist)]
        scores['CRPS'] = np.mean(crps_values)

    return scores

# Usage
scores_dfm = compute_scoring_rules(results['y_true'], results['y_pred'])
print(f"DFM Nowcast Performance:")
for metric, value in scores_dfm.items():
    print(f"  {metric}: {value:.3f}")
```

### Diebold-Mariano Test Implementation

```python
from scipy import stats
from statsmodels.stats.sandwich_covariance import cov_hac

def diebold_mariano_test(errors_a, errors_b, loss_fn=lambda e: e**2,
                         h=1, harvey_adj=True):
    """
    Diebold-Mariano test for equal predictive accuracy.

    Parameters
    ----------
    errors_a, errors_b : array-like
        Forecast errors from models A and B
    loss_fn : callable
        Loss function (default: squared error)
    h : int
        Forecast horizon (for small-sample adjustment)
    harvey_adj : bool
        Apply Harvey et al. (1997) small-sample correction

    Returns
    -------
    dm_stat : float
        DM test statistic (N(0,1) under H0)
    p_value : float
        Two-sided p-value
    """
    # Compute loss differential
    d = loss_fn(errors_a) - loss_fn(errors_b)
    T = len(d)

    # Mean differential
    d_bar = np.mean(d)

    # HAC-robust variance (allowing for autocorrelation)
    d_centered = d - d_bar
    var_d = cov_hac(d_centered.reshape(-1, 1))[0, 0]

    # DM statistic
    dm_stat = d_bar / np.sqrt(var_d / T)

    # Harvey small-sample adjustment
    if harvey_adj:
        dm_stat *= np.sqrt((T + 1 - 2*h + T**-1 * h * (h-1)) / T)

    # P-value (two-sided)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

    return dm_stat, p_value

# Example: Compare DFM vs AR benchmark
dm_stat, p_val = diebold_mariano_test(
    errors_a=results_dfm['error'].values,
    errors_b=results_ar['error'].values,
    h=1
)

print(f"Diebold-Mariano Test: DFM vs AR")
print(f"  DM statistic: {dm_stat:.3f}")
print(f"  p-value: {p_val:.4f}")
if p_val < 0.05:
    winner = "DFM" if dm_stat < 0 else "AR"
    print(f"  → {winner} significantly outperforms at 5% level")
```

### Optimal Forecast Combination

```python
def optimal_forecast_combination(forecasts, y_true):
    """
    Compute MSE-optimal forecast combination weights.

    Parameters
    ----------
    forecasts : pd.DataFrame (T x K)
        K forecasts for T periods
    y_true : array-like (T,)
        Realized values

    Returns
    -------
    weights : array (K,)
        Optimal combination weights (sum to 1)
    combined_forecast : array (T,)
        Combined forecast
    """
    # Compute forecast errors
    errors = forecasts.values - y_true.values.reshape(-1, 1)

    # Error covariance matrix
    Sigma = np.cov(errors, rowvar=False)

    # Optimal weights: w = Σ⁻¹·1 / (1'Σ⁻¹·1)
    Sigma_inv = np.linalg.inv(Sigma)
    ones = np.ones(Sigma_inv.shape[0])
    weights = Sigma_inv @ ones / (ones @ Sigma_inv @ ones)

    # Ensure non-negativity (constrained optimization)
    # For unconstrained, use above; for w >= 0, use quadratic programming

    # Combined forecast
    combined = forecasts.values @ weights

    return weights, combined

# Example: Combine DFM, AR, VAR forecasts
forecasts = pd.DataFrame({
    'DFM': results_dfm['y_pred'],
    'AR': results_ar['y_pred'],
    'VAR': results_var['y_pred']
})

weights, combined = optimal_forecast_combination(forecasts, y_true)
print(f"Optimal Combination Weights:")
for name, w in zip(forecasts.columns, weights):
    print(f"  {name}: {w:.3f}")
```

---

## Common Pitfalls

### 1. In-Sample Overfitting

**Problem:** Testing on same data used for training.

**Example:**
```python
# WRONG
model.fit(X, y)
predictions = model.predict(X)
mse = mean_squared_error(y, predictions)  # Artificially low!
```

**Solution:** Always use out-of-sample data or cross-validation.

### 2. Look-Ahead Bias in Backtesting

**Problem:** Using information not available at forecast time.

**Sources:**
- Final revised data instead of first release
- Predictors published after target (e.g., using March retail sales to forecast March GDP when retail sales is released in April)
- Parameters estimated on full sample

**Solution:** Simulate real-time information set strictly.

### 3. Ignoring Forecast Horizon Effects

**Problem:** Comparing 1-quarter ahead vs 4-quarter ahead forecasts using same metric.

**Issue:** Longer horizons have higher unconditional variance → larger errors expected.

**Solution:** Normalize by horizon or compute relative RMSE:
$$\text{Relative RMSE} = \frac{\text{RMSE}_{\text{model}}}{\text{RMSE}_{\text{benchmark}}}$$

### 4. Multiple Testing Without Correction

**Problem:** Testing 20 models, finding 1 "significant" at 5% level.

**Issue:** Expected to find 1 false positive by chance (20 × 0.05 = 1).

**Solution:**
- Bonferroni correction: use $\alpha / K$ significance level
- False Discovery Rate control
- Pre-specify primary comparison

### 5. Cherry-Picking Evaluation Period

**Problem:** Reporting performance only on favorable periods.

**Example:** "Our model achieved 0.3 RMSE during 2010-2019" (excludes 2008-2009 crisis).

**Solution:** Report full-sample and sub-period results, including recessions.

---

## Connections

### Builds On
- **Nowcasting** (Previous Guide): Forecasts to evaluate
- **Kalman Filter**: Produces forecast distributions, not just points
- **Statistical Inference**: Hypothesis testing framework

### Leads To
- **Model Selection**: Choose best model based on evaluation
- **Forecast Combination**: Ensemble methods exploit weaknesses
- **Real-Time Monitoring**: Automated evaluation dashboards

### Related To
- **Cross-Validation**: Out-of-sample evaluation for parameter tuning
- **Bayesian Model Averaging**: Posterior probability weights for combination
- **Online Learning**: Update model as new data arrives

---

## Practice Problems

### Conceptual

1. **Scoring Rule Selection**
   - Why might MAE be preferred over MSE for evaluating inflation forecasts?
   - When is CRPS more informative than RMSE?

2. **Real-Time Data**
   - GDP is revised substantially 2-3 years after initial release. Should we evaluate nowcasts against initial or final GDP? Justify.

3. **Forecast Encompassing**
   - Model A: RMSE = 1.0, Model B: RMSE = 1.2. Does this prove A encompasses B? Why or why not?

### Implementation

4. **Backtest Your Nowcaster**
   ```python
   # From previous notebook:
   # - Implement expanding window backtest for 2010-2023
   # - Compute MSE, MAE, directional accuracy
   # - Compare to AR(1) benchmark
   # - Conduct Diebold-Mariano test
   ```

5. **Real vs Pseudo Real-Time**
   ```python
   # Compare two backtests:
   # A) Using real-time vintages from ALFRED
   # B) Using final revised data throughout
   # How much does data revision affect evaluation?
   ```

6. **Optimal Combination**
   ```python
   # You have forecasts from:
   # - DFM (your model)
   # - AR(2) benchmark
   # - SPF consensus (Survey of Professional Forecasters)
   # Find optimal combination weights and evaluate combined forecast
   ```

### Extension

7. **State-Dependent Evaluation**
   - Evaluate forecasts separately during expansions vs recessions
   - Do DFMs perform better in one regime? Why?
   - Construct regime-switching combination weights

8. **Density Forecast Calibration**
   - Construct 68% and 95% prediction intervals from your DFM
   - Check calibration: do 68% of realizations fall in 68% interval?
   - If miscalibrated, how would you adjust uncertainty estimates?

---

## Further Reading

### Essential

- **Diebold, F. X., & Mariano, R. S. (1995).** "Comparing Predictive Accuracy." *Journal of Business & Economic Statistics, 13*(3), 253-263.
  - *Original DM test paper - essential for forecast comparison*

- **Gneiting, T., & Raftery, A. E. (2007).** "Strictly Proper Scoring Rules, Prediction, and Estimation." *Journal of the American Statistical Association, 102*(477), 359-378.
  - *Comprehensive treatment of proper scoring rules*

- **Clark, T. E., & McCracken, M. W. (2013).** "Advances in Forecast Evaluation." *Handbook of Economic Forecasting, Vol 2*, 1107-1201.
  - *Authoritative review of modern forecast evaluation methods*

### Recommended

- **Harvey, D., Leybourne, S., & Newbold, P. (1997).** "Testing the Equality of Prediction Mean Squared Errors." *International Journal of Forecasting, 13*(2), 281-291.
  - *Small-sample corrections for DM test*

- **Timmermann, A. (2006).** "Forecast Combinations." *Handbook of Economic Forecasting, Vol 1*, 135-196.
  - *Theory and practice of forecast combination methods*

### Advanced

- **Giacomini, R., & White, H. (2006).** "Tests of Conditional Predictive Ability." *Econometrica, 74*(6), 1545-1578.
  - *Evaluate forecasts conditional on economic states*

- **Rossi, B., & Sekhposyan, T. (2010).** "Have Economic Models' Forecasting Performance for US Output Growth and Inflation Changed Over Time, and When?" *International Journal of Forecasting, 26*(4), 808-835.
  - *Time-varying forecast evaluation and structural breaks*

---

**Next Guide:** Missing Data Handling - Kalman filter techniques for ragged edges and irregular patterns
