# Mixed-Frequency Dynamic Factor Models

## In Brief

Mixed-frequency DFMs combine variables observed at different temporal frequencies (daily, monthly, quarterly) into a unified framework. By modeling high-frequency factors and using skip-sampling or temporal aggregation, mixed-frequency models exploit all available data without artificial aggregation, improving nowcasting accuracy and handling ragged edges naturally.

## Key Insight

GDP is quarterly, but employment is monthly. Naive solution: aggregate employment to quarterly → loses 2/3 of data! Smart solution: model monthly employment factor, map to quarterly GDP via averaging within quarter. Result: use all 3 months of employment data to predict current-quarter GDP. Mixed-frequency models formalize this idea using state-space methods or MIDAS regression.

---

## Visual Explanation

```
Problem: How to use monthly data for quarterly target?
═══════════════════════════════════════════════════════════

Q1 GDP (quarterly):
      ├──────────────────┤
   Jan    Feb    Mar    Apr

Employment (monthly):
      ├──┤├──┤├──┤├──┤
   Jan Feb Mar Apr


NAIVE APPROACH (aggregate to quarterly):
Employment_Q1 = mean(Jan, Feb, Mar)

Result: Single quarterly observation, loses timing info!


MIXED-FREQUENCY APPROACH:
1. Model monthly factor from employment
   F_Jan, F_Feb, F_Mar  (three separate estimates)

2. Link to quarterly GDP via aggregation:
   GDP_Q1 = β₀ + β₁ · (F_Jan + F_Feb + F_Mar)/3 + ε

Result: All three months inform GDP nowcast!


State-Space Mixed-Frequency DFM:
═══════════════════════════════════════════════════════════

HIGH FREQUENCY (monthly):
X^(m)_t = Λ^(m) F_t + e^(m)_t     (e.g., IP, employment)

LOW FREQUENCY (quarterly):
y^(q)_τ = β' · (F_{3τ-2} + F_{3τ-1} + F_{3τ})/3 + e^(q)_τ

FACTOR DYNAMICS (monthly):
F_t = Φ F_{t-1} + η_t


Skip-Sampling Observation Pattern:
═══════════════════════════════════════════════════════════

Time:  Jan  Feb  Mar │ Apr  May  Jun │ Jul  Aug  Sep
       ─────────────────────────────────────────────
IP:     ✓    ✓    ✓  │  ✓    ✓    ✓  │  ✓    ✓    ✓   (monthly)
GDP:    ?    ?    ✓  │  ?    ?    ✓  │  ?    ?    ✓   (end of quarter only)

Kalman filter:
- Months 1-2: update with IP only (GDP missing)
- Month 3: update with IP AND GDP (both observed)
```

---

## Formal Definition

### State-Space Mixed-Frequency DFM

**High-frequency measurement equation** (monthly):
$$X^{(m)}_t = \Lambda^{(m)} F_t + e^{(m)}_t, \quad t = 1, 2, 3, ...$$

**Low-frequency measurement equation** (quarterly):
$$y^{(q)}_\tau = \Lambda^{(q)} \bar{F}_\tau + e^{(q)}_\tau, \quad \tau = 1, 2, 3, ...$$

where $\bar{F}_\tau = \frac{1}{3}(F_{3\tau-2} + F_{3\tau-1} + F_{3\tau})$ is quarterly average.

**Factor dynamics** (monthly):
$$F_t = \Phi F_{t-1} + \eta_t$$

**Skip-sampling formulation:** $y^{(q)}_\tau$ only observed at $t = 3, 6, 9, ...$ (end of quarters).

### Temporal Aggregation Operator

Define aggregation matrix $C_\tau$ that averages monthly factors:
$$\bar{F}_\tau = C_\tau \begin{bmatrix} F_{3\tau-2} \\ F_{3\tau-1} \\ F_{3\tau} \end{bmatrix} = \frac{1}{3} [1, 1, 1] \begin{bmatrix} F_{3\tau-2} \\ F_{3\tau-1} \\ F_{3\tau} \end{bmatrix}$$

**Alternative:** Stock variable (end-of-quarter): $C = [0, 0, 1]$ (use only $F_{3\tau}$)

### MIDAS Regression Approach

**Distributed lag specification:**
$$y^{(q)}_\tau = \alpha + \sum_{k=0}^{K} \beta_k(\theta) X^{(m)}_{3\tau-k} + \epsilon_\tau$$

where $\beta_k(\theta)$ are lag weights parameterized by $\theta$ (e.g., exponential Almon lag).

**Common weight functions:**
- Exponential Almon: $\beta_k = \exp(\theta_1 k + \theta_2 k^2)$
- Beta lag: $\beta_k = B(k/K; \theta_1, \theta_2)$ (Beta density)

**Estimation:** Nonlinear least squares for $\theta$, conditional on weight function.

---

## Intuitive Explanation

### Why Mixed-Frequency Matters

**Scenario:** Nowcasting Q1 2024 GDP on March 15.

**Available data:**
- January employment: ✓ (released Feb 5)
- February employment: ✓ (released Mar 5)
- March employment: ✗ (released Apr 5)
- Q1 GDP: ✗ (released Apr 30)

**Quarterly aggregation approach:**
- Wait until April 5 to get all 3 months of employment
- Aggregate to quarterly: Emp_Q1 = mean(Jan, Feb, Mar)
- Nowcast GDP using Emp_Q1

**Problem:** Can't nowcast until March data arrives (delayed by 3 weeks)!

**Mixed-frequency approach:**
- Model monthly factor using Jan+Feb employment (available now!)
- Forecast March factor using VAR dynamics: $\hat{F}_{\text{Mar}} = \Phi^2 F_{\text{Jan}}$
- Nowcast GDP: $\hat{y}_{Q1} = \beta \cdot (\hat{F}_{\text{Jan}} + \hat{F}_{\text{Feb}} + \hat{F}_{\text{Mar}})/3$

**Advantage:** Produce nowcast 3 weeks earlier with uncertainty band for missing March.

### MIDAS vs State-Space

**MIDAS (Distributed lag):**
- **Pro:** Simple to estimate (NLS), interpretable lag structure
- **Con:** Doesn't handle ragged edges well, no factor structure
- **Use:** Single predictor → single target (e.g., monthly sales → quarterly GDP)

**State-Space mixed-frequency DFM:**
- **Pro:** Many predictors → factor → target, handles ragged edges optimally
- **Con:** More complex (Kalman filter), requires more data
- **Use:** Multi-variable nowcasting (IP, employment, surveys → GDP)

---

## Code Implementation

### State-Space Mixed-Frequency DFM

```python
import numpy as np
import pandas as pd
from scipy.linalg import block_diag

def mixed_frequency_dfm_kalman(X_monthly, y_quarterly, k_factors=2):
    """
    Estimate mixed-frequency DFM: monthly X → quarterly y.

    Parameters
    ----------
    X_monthly : pd.DataFrame (T_month x N)
        Monthly indicators
    y_quarterly : pd.Series (T_quarter)
        Quarterly target

    Returns
    -------
    nowcast : float
        Current quarter GDP nowcast
    factors_monthly : pd.DataFrame
        Estimated monthly factors
    """
    from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

    # Step 1: Estimate monthly DFM
    dfm = DynamicFactor(
        X_monthly,
        k_factors=k_factors,
        factor_order=1,
        error_cov_type='diagonal'
    )
    dfm_res = dfm.fit(maxiter=500, disp=False)

    # Step 2: Extract monthly factors
    factors_monthly = dfm_res.factors.filtered

    # Step 3: Aggregate to quarterly (average within quarter)
    factors_quarterly = factors_monthly.resample('Q').mean()

    # Step 4: Bridge equation (quarterly factors → quarterly GDP)
    from sklearn.linear_model import LinearRegression

    # Align quarterly factors with GDP
    common_idx = factors_quarterly.index.intersection(y_quarterly.index)
    X_bridge = factors_quarterly.loc[common_idx]
    y_bridge = y_quarterly.loc[common_idx]

    bridge = LinearRegression()
    bridge.fit(X_bridge, y_bridge)

    # Step 5: Nowcast current quarter
    # Use all available monthly factors in current quarter
    current_quarter_start = X_monthly.index[-1].to_period('Q').to_timestamp()
    current_quarter_factors = factors_monthly.loc[current_quarter_start:].mean().values.reshape(1, -1)

    nowcast = bridge.predict(current_quarter_factors)[0]

    return nowcast, factors_monthly

# Example usage
np.random.seed(42)
dates_m = pd.date_range('2000-01-01', '2024-03-01', freq='M')
T_m = len(dates_m)

# Monthly data
factor = np.cumsum(np.random.randn(T_m)) * 0.1
X_monthly = pd.DataFrame({
    'IP': 0.8 * factor + 0.5 * np.random.randn(T_m),
    'Employment': 0.7 * factor + 0.6 * np.random.randn(T_m)
}, index=dates_m)

# Quarterly GDP (from quarterly-averaged factor)
factor_q = pd.Series(factor, index=dates_m).resample('Q').mean()
y_quarterly = 2.0 + 1.5 * factor_q + 0.3 * np.random.randn(len(factor_q))

nowcast, factors = mixed_frequency_dfm_kalman(X_monthly, y_quarterly)
print(f"Q1 2024 GDP Nowcast: {nowcast:.2f}%")
```

### MIDAS Regression

```python
from scipy.optimize import minimize

def midas_regression(X_high, y_low, lag_order=12):
    """
    MIDAS regression: high-frequency X → low-frequency y.

    Parameters
    ----------
    X_high : pd.Series (monthly)
    y_low : pd.Series (quarterly)
    lag_order : int
        Number of high-freq lags to include

    Returns
    -------
    fitted_model : dict
        Contains weights, forecasts, parameters
    """
    # Align data: for each quarter, get previous K months
    X_lagged = []
    y_aligned = []

    for q_date in y_low.index:
        # Get monthly data for this quarter and previous lags
        month_end = q_date
        month_start = month_end - pd.DateOffset(months=lag_order)

        X_window = X_high.loc[month_start:month_end]

        if len(X_window) >= lag_order:
            X_lagged.append(X_window.values[-lag_order:][::-1])  # Reverse for lag order
            y_aligned.append(y_low.loc[q_date])

    X_lagged = np.array(X_lagged)
    y_aligned = np.array(y_aligned)

    # Exponential Almon weights: β_k = exp(θ₁·k + θ₂·k²)
    def almon_weights(theta):
        k = np.arange(lag_order)
        w = np.exp(theta[0] * k + theta[1] * k**2)
        return w / w.sum()  # Normalize

    # Objective: minimize SSR
    def objective(params):
        alpha, theta = params[0], params[1:3]
        weights = almon_weights(theta)

        # Weighted sum of lags
        X_weighted = (X_lagged * weights).sum(axis=1)

        # Residuals
        resid = y_aligned - (alpha + X_weighted)

        return np.sum(resid**2)

    # Optimize
    result = minimize(objective, x0=[y_aligned.mean(), 0, 0], method='BFGS')

    alpha_hat = result.x[0]
    theta_hat = result.x[1:3]
    weights_hat = almon_weights(theta_hat)

    # Fitted values
    X_weighted = (X_lagged * weights_hat).sum(axis=1)
    fitted = alpha_hat + X_weighted

    return {
        'alpha': alpha_hat,
        'theta': theta_hat,
        'weights': weights_hat,
        'fitted': fitted,
        'residuals': y_aligned - fitted
    }

# Example
midas_model = midas_regression(
    X_high=X_monthly['IP'],
    y_low=y_quarterly,
    lag_order=9
)

print(f"MIDAS weights (highest = most recent):")
print(midas_model['weights'][::-1])
```

---

## Common Pitfalls

### 1. Mixing Stock and Flow Variables

**Problem:** GDP is flow (sum within quarter), while asset prices are stocks (end-of-quarter)

**Solution:** Use different aggregation: flow → average, stock → last observation

### 2. Alignment Errors

**Problem:** February monthly data actually contains information through mid-March (publication lag)

**Solution:** Account for publication timing, not calendar dates

### 3. Overfitting MIDAS Lag Structure

**Problem:** Including 24 monthly lags for quarterly prediction → overfit

**Solution:** Use BIC to select lag order, typically 6-12 lags sufficient

### 4. Ignoring Factor Dynamics in Bridge Equation

**Problem:** Bridge equation assumes factors are exogenous, but they're serially correlated

**Solution:** Include lagged factors in bridge or use full state-space system

### 5. Not Accounting for Revisions

**Problem:** Monthly data gets revised, quarterly data gets revised → misalignment

**Solution:** Use real-time vintages for both frequencies

---

## Connections

### Builds On
- **Kalman Filter** (Module 2): Skip-sampling is special case of missing data
- **Temporal Aggregation**: Flow vs stock distinction
- **Nowcasting** (Module 3): Mixed-frequency critical for ragged edges

### Leads To
- **Partial Aggregation**: Weekly + monthly + quarterly models
- **High-Frequency Econometrics**: Daily data for monthly targets
- **Real-Time Data Flow**: Continuous updating as data arrives

### Related To
- **Temporal Disaggregation**: Go opposite direction (quarterly → monthly)
- **MIDAS Literature**: Distributed lag models
- **State-Space Irregular Sampling**: General framework

---

## Practice Problems

### Conceptual

1. **Stock vs Flow:** Which aggregation for: GDP (flow), unemployment rate (stock), S&P 500 (stock)?

2. **Information Content:** You have 90 days of stock returns to predict quarterly earnings. How many effective observations?

### Implementation

3. **Build Mixed-Frequency Nowcaster:**
   ```python
   # Use monthly IP + quarterly GDP
   # Implement both state-space and MIDAS
   # Compare nowcast accuracy
   ```

4. **Optimal Lag Selection:**
   ```python
   # For MIDAS with monthly → quarterly
   # Try lag orders 3, 6, 9, 12
   # Select using BIC
   # Does more lags always help?
   ```

---

## Further Reading

### Essential

- **Ghysels, E., Santa-Clara, P., & Valkanov, R. (2004).** "The MIDAS Touch: Mixed Data Sampling Regression Models." *UCLA/UNC Working Paper.*
  - *Original MIDAS paper*

- **Mariano, R. S., & Murasawa, Y. (2003).** "A New Coincident Index of Business Cycles Based on Monthly and Quarterly Series." *Journal of Applied Econometrics, 18*(4), 427-443.
  - *State-space mixed-frequency DFM*

---

**Next Guide:** Large Datasets - handling hundreds of predictors efficiently
