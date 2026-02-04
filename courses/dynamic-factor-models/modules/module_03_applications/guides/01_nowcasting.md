# Nowcasting with Dynamic Factor Models

## In Brief

Nowcasting uses high-frequency indicators (monthly data) and dynamic factor models to estimate current-quarter economic activity (GDP) before official statistics are released. By combining timely but noisy signals through a DFM, we produce real-time estimates that inform policy and trading decisions weeks before official data publication.

## Key Insight

GDP is released 4-6 weeks after quarter-end, but investors and policymakers need to know *now* whether the economy is growing or contracting. The solution: extract a latent "economic activity" factor from dozens of timely monthly indicators (employment, production, sales), then use the factor to predict GDP. As new data arrives, the Kalman filter optimally updates the nowcast.

---

## Visual Explanation

```
Timeline of Information Flow:
════════════════════════════════════════════════════════════════

Q1 Ends              GDP Advance      GDP Final
(March 31)           Estimate         Estimate
    │                    │                │
    │<---4-6 weeks------>│<--2 months---->│
    │                    │                │
    ▼                    ▼                ▼
┌───────────────────────────────────────────────────────────┐
│  Jan   Feb   Mar  │ Apr   May   Jun  │ Jul   Aug   Sep   │
│   ▲     ▲     ▲    │                  │                   │
│   │     │     │    │                  │                   │
│  IP₁   IP₂   IP₃  │ We wait here --> │ Finally get GDP!  │
│  EMP₁  EMP₂  EMP₃ │ with no data     │                   │
│  PMI₁  PMI₂  PMI₃ │                  │                   │
└───────────────────────────────────────────────────────────┘
         │
         └──> DFM Nowcast: "Q1 GDP ≈ 2.3% ± 0.8%"
              Available: April 15 (2 weeks before official!)


DFM Nowcasting Flow:
═══════════════════════════════════════════════════════════

┌─────────────────┐
│ Monthly Data    │  IP, Employment, Sales, PMI, Housing
│ (N=50-100)      │  Released throughout the quarter
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Factor          │  Extract r=3-5 latent factors
│ Extraction      │  (Economic activity, financial, sentiment)
│ (DFM)           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Bridge          │  GDP_t = β₀ + β₁·Factor_t + ε_t
│ Equation        │  (Links quarterly GDP to monthly factors)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Kalman Filter   │  Optimally combines:
│ Update          │  - New data arrivals (news)
│                 │  - Factor dynamics (persistence)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ GDP Nowcast     │  Point estimate + confidence interval
│ & Uncertainty   │  Updates daily as data arrives
└─────────────────┘
```

---

## Formal Definition

### State-Space Nowcasting Model

**Measurement Equation** (Monthly indicators):
$$X_t = \Lambda F_t + e_t, \quad e_t \sim N(0, \Sigma_e)$$

where:
- $X_t$: $N \times 1$ vector of monthly indicators at month $t$
- $F_t$: $r \times 1$ vector of latent monthly factors
- $\Lambda$: $N \times r$ factor loadings
- $e_t$: Idiosyncratic errors

**Transition Equation** (Factor dynamics):
$$F_t = \Phi F_{t-1} + \eta_t, \quad \eta_t \sim N(0, Q)$$

**Bridge Equation** (Quarterly GDP):
$$y_\tau = \beta_0 + \beta_1 \bar{F}_\tau + \epsilon_\tau, \quad \epsilon_\tau \sim N(0, \sigma^2)$$

where:
- $y_\tau$: GDP growth in quarter $\tau$
- $\bar{F}_\tau = \frac{1}{3}(F_{3\tau-2} + F_{3\tau-1} + F_{3\tau})$: Quarterly average of monthly factors
- $\beta_1$: Sensitivity of GDP to factors

### Nowcast Computation

At time $t$ within quarter $\tau$, the nowcast is:

$$\hat{y}_{\tau|t} = \beta_0 + \beta_1 \mathbb{E}\left[\bar{F}_\tau \mid X_{1:t}\right]$$

**Components:**
- $\mathbb{E}[F_s \mid X_{1:t}]$ for $s \leq t$: Filtered factor estimates (Kalman filter)
- $\mathbb{E}[F_s \mid X_{1:t}]$ for $s > t$: Forecasted factors (VAR projection)

**Uncertainty:**
$$\text{Var}(\hat{y}_{\tau|t}) = \beta_1^2 \cdot \text{Var}(\bar{F}_\tau \mid X_{1:t}) + \sigma^2$$

---

## Intuitive Explanation

### The Core Problem

Imagine you're the Fed Chair in April 2024. You need to decide on interest rates, but Q1 GDP won't be released until May. However, you already have:
- March employment report (326k jobs added)
- March industrial production (+0.4%)
- March PMI surveys (indicating expansion)

**Question:** Can you infer Q1 GDP from these timely indicators?

**Answer:** Yes, via nowcasting!

### The DFM Solution

1. **Historical relationship:** GDP historically co-moves with IP, employment, etc.
2. **Extract the signal:** Use DFM to find the common "economic activity" factor
3. **Bridge to GDP:** Estimate $\text{GDP} \approx f(\text{activity factor})$
4. **Real-time update:** As March data arrives, update the activity factor → update GDP nowcast

### Ragged Edge Problem

Not all data arrives simultaneously:

```
End of March data availability:

Industrial Production:  Available April 15
Employment:             Available April 5
Retail Sales:           Available April 12
GDP (Q1):               Available April 30

┌─────────────────────────────────┐
│ IP:   ███████████░░░░ (70% obs) │  ← Missing March
│ EMP:  ████████████████ (100%)   │
│ Sales:████████████░░░  (75%)    │
│ GDP:  ░░░░░░░░░░░░░░░░ (0%!)    │
└─────────────────────────────────┘
     Ragged edge →
```

**DFM handles this naturally:** Kalman filter treats missing observations as "unknown" and integrates out the uncertainty.

### News vs Uncertainty Decomposition

When your nowcast changes from 2.1% to 2.5%, is it because:
1. **News:** New data came in stronger than expected?
2. **Reduced uncertainty:** No new data, but more confident about existing estimates?

The Kalman filter separates these:
- **News:** $\mathbb{E}[F_t \mid X_{1:t}] - \mathbb{E}[F_t \mid X_{1:t-1}]$
- **Uncertainty reduction:** $\text{Var}(F_t \mid X_{1:t}) - \text{Var}(F_t \mid X_{1:t-1})$

---

## Code Implementation

### Complete Nowcasting System

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from pandas_datareader import data as pdr

def build_nowcast_model(monthly_data, gdp_data, n_factors=3):
    """
    Build DFM-based nowcasting model.

    Parameters
    ----------
    monthly_data : pd.DataFrame (T_month x N)
        Monthly indicators (standardized)
    gdp_data : pd.Series (T_quarter)
        Quarterly GDP growth
    n_factors : int
        Number of latent factors

    Returns
    -------
    model : fitted DynamicFactor object
    bridge_params : dict with bridge equation coefficients
    """
    # Step 1: Estimate DFM on monthly data
    dfm = DynamicFactor(
        monthly_data,
        k_factors=n_factors,
        factor_order=2,  # VAR(2) for factors
        error_cov_type='diagonal'
    )
    dfm_res = dfm.fit(maxiter=1000, disp=False)

    # Step 2: Extract monthly factors
    factors_monthly = dfm_res.factors.filtered

    # Step 3: Aggregate to quarterly (average within quarter)
    factors_quarterly = factors_monthly.resample('Q').mean()

    # Step 4: Align with GDP data
    common_index = factors_quarterly.index.intersection(gdp_data.index)
    X_bridge = factors_quarterly.loc[common_index]
    y_bridge = gdp_data.loc[common_index]

    # Step 5: Estimate bridge equation via OLS
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_bridge, y_bridge)

    bridge_params = {
        'intercept': lr.intercept_,
        'factor_coefs': lr.coef_,
        'residual_var': np.var(y_bridge - lr.predict(X_bridge))
    }

    return dfm_res, bridge_params

def nowcast_gdp(dfm_model, bridge_params, current_month):
    """
    Produce GDP nowcast for current quarter.

    Parameters
    ----------
    dfm_model : fitted DynamicFactor
    bridge_params : dict from build_nowcast_model
    current_month : int (1-12)
        Current month (for quarter determination)

    Returns
    -------
    nowcast : float
        GDP growth nowcast
    std_error : float
        Nowcast standard error
    """
    # Get filtered factors up to current period
    factors = dfm_model.factors.filtered

    # Determine quarter and months to average
    quarter = (current_month - 1) // 3
    quarter_months = [quarter*3 + 1, quarter*3 + 2, quarter*3 + 3]

    # Extract current quarter factors (some may be forecasts)
    current_quarter_factors = factors.iloc[-3:].mean(axis=0).values

    # Bridge equation prediction
    intercept = bridge_params['intercept']
    coefs = bridge_params['factor_coefs']

    nowcast = intercept + coefs @ current_quarter_factors

    # Uncertainty (simplified - ignores factor uncertainty)
    std_error = np.sqrt(bridge_params['residual_var'])

    return nowcast, std_error
```

### Handling Ragged Edge with Missing Data

```python
def nowcast_with_ragged_edge(monthly_data_ragged, gdp_history, n_factors=3):
    """
    Nowcast GDP allowing for missing recent observations (ragged edge).

    Parameters
    ----------
    monthly_data_ragged : pd.DataFrame with NaNs
        Recent months have missing values (ragged edge)
    """
    # DynamicFactor handles missing data automatically via Kalman filter
    dfm = DynamicFactor(
        monthly_data_ragged,
        k_factors=n_factors,
        factor_order=1,
        error_cov_type='diagonal'
    )

    # Fit model (EM algorithm handles missing data)
    dfm_res = dfm.fit(maxiter=500, disp=False)

    # Filtered factors account for missing observations
    factors = dfm_res.factors.filtered  # Uncertainty higher for ragged periods
    factors_smooth = dfm_res.factors.smoothed  # Uses all data (backcasting too)

    return dfm_res, factors, factors_smooth
```

---

## Common Pitfalls

### 1. Ignoring Publication Lags

**Problem:** Using revised (final) data for backtesting instead of real-time vintages.

**Impact:** Artificially inflates nowcast accuracy. Real-time data is noisier and subject to revisions.

**Solution:**
- Use ALFRED database for real-time vintages
- Backtest with data *as it was available* at each point in time
- Account for revision patterns in uncertainty bands

### 2. Overfitting the Bridge Equation

**Problem:** Including too many factors or lags in bridge equation.

**Impact:** In-sample fit improves, but out-of-sample nowcasts degrade (overfitting).

**Solution:**
- Keep bridge equation simple (3-5 factors, no lags)
- Use cross-validation for factor number selection
- Regularize with LASSO if many potential predictors

### 3. Treating All Indicators Equally

**Problem:** Giving equal weight to timely-but-noisy vs lagged-but-accurate indicators.

**Impact:** Nowcast becomes too volatile, reacting to noise.

**Solution:**
- Explicitly model measurement error variance (higher for flash estimates)
- Use exponential moving average for smoothing
- Downweight indicators with frequent revisions

### 4. Not Updating as Data Arrives

**Problem:** Producing one nowcast per month instead of updating daily.

**Impact:** Miss important signals from early data releases.

**Solution:**
- Set up automated pipeline to refresh when FRED updates
- Use Kalman filter's sequential update property
- Maintain nowcast history to track revisions

### 5. Ignoring Structural Breaks

**Problem:** Using pre-COVID parameters for post-COVID nowcasting.

**Impact:** Factor loadings and bridge coefficients shift during regime changes.

**Solution:**
- Rolling window estimation (e.g., last 10 years only)
- Bayesian model averaging over structural break points
- Expand uncertainty bands during high-volatility periods

---

## Connections

### Builds On
- **Kalman Filter** (Module 2): Optimal filtering with missing data
- **State-Space Models** (Module 2): DFM in state-space form
- **Bridge Equations**: Link high-frequency factors to low-frequency targets

### Leads To
- **Mixed-Frequency Models** (Module 4): MIDAS as alternative to bridge equations
- **Real-Time Data Flow** (Module 5): Automated nowcasting pipelines
- **Forecast Combination** (Module 6): Ensemble nowcasts

### Related To
- **Kalman Gain Interpretation**: News vs uncertainty decomposition
- **Missing Data Imputation**: DFM as optimal interpolator
- **Temporal Aggregation**: Monthly → quarterly averaging

---

## Practice Problems

### Conceptual

1. **Timeliness vs Accuracy Trade-off**
   - Would you prefer a nowcast with MSE=0.5 available 4 weeks early, or MSE=0.3 available 1 week early? Under what circumstances?
   - How does the value of timeliness change around FOMC meetings vs mid-quarter?

2. **News Decomposition**
   - Your nowcast revised from 2.0% to 2.4%. March employment came in at +300k vs +250k expected. Decompose the 0.4pp revision into "employment news" vs "other factors."
   - What does it mean if nowcast changes but no new data arrived?

3. **Ragged Edge Severity**
   - Which matters more for Q1 GDP nowcasting: having March IP data or having March retail sales data? How would you test this empirically?

### Implementation

4. **Build a Simple Nowcaster**
   ```python
   # Using FRED data:
   # - Download IP, employment, retail sales (2000-present)
   # - Download quarterly GDP
   # - Estimate 2-factor DFM
   # - Build bridge equation
   # - Produce nowcast for most recent quarter
   # - Compare to NY Fed Nowcast (if available)
   ```

5. **Backtest Framework**
   ```python
   # Implement pseudo-out-of-sample backtesting:
   # - For each quarter from 2010-2023:
   #   - Use only data available at end-of-quarter
   #   - Produce nowcast
   #   - Compare to first-release GDP (advance estimate)
   # - Report RMSE, MAE, directional accuracy
   ```

6. **Sensitivity Analysis**
   ```python
   # How sensitive is your nowcast to:
   # - Number of factors (1 vs 5)?
   # - Factor dynamics (VAR(1) vs VAR(2))?
   # - Missing the last month of data?
   # Create visualization showing nowcast ± sensitivity range
   ```

### Extension

7. **Real-Time Vintage Comparison**
   - Download ALFRED vintages for GDP (advance, 2nd, 3rd, annual revisions)
   - Compare your nowcast to each vintage
   - Does your nowcast predict revisions? (i.e., is it closer to final than to advance?)

8. **Multi-Horizon Nowcasting**
   - Extend to nowcast current quarter AND next quarter simultaneously
   - How does uncertainty increase for next-quarter forecast?
   - When (in the calendar) does next-quarter forecast become more accurate than current-quarter nowcast?

---

## Further Reading

### Essential

- **Giannone, D., Reichlin, L., & Small, D. (2008).** "Nowcasting: The Real-Time Informational Content of Macroeconomic Data." *Journal of Monetary Economics, 55*(4), 665-676.
  - *Foundational paper on DFM-based nowcasting with ragged edge*

- **Bańbura, M., & Modugno, M. (2014).** "Maximum Likelihood Estimation of Factor Models on Datasets with Arbitrary Pattern of Missing Data." *Journal of Applied Econometrics, 29*(1), 133-160.
  - *Kalman filter approach to ragged-edge nowcasting*

- **Federal Reserve Bank of New York.** "Nowcasting Report."
  - *Operational DFM nowcast, updated weekly with code on GitHub*
  - URL: https://www.newyorkfed.org/research/policy/nowcast

### Recommended

- **Bok, B., Caratelli, D., Giannone, D., Sbordone, A. M., & Tambalotti, A. (2018).** "Macroeconomic Nowcasting and Forecasting with Big Data." *Annual Review of Economics, 10*, 615-643.
  - *Comprehensive review of nowcasting methods and applications*

- **Chernis, T., & Sekkel, R. (2017).** "A Dynamic Factor Model for Nowcasting Canadian GDP Growth." *Bank of Canada Staff Working Paper 2017-2.*
  - *Practical implementation guide with code examples*

### Advanced

- **Bańbura, M., Giannone, D., Modugno, M., & Reichlin, L. (2013).** "Now-Casting and the Real-Time Data Flow." *Handbook of Economic Forecasting, Vol 2*, 195-237.
  - *Comprehensive treatment of real-time data issues and solutions*

- **Schorfheide, F., & Song, D. (2015).** "Real-Time Forecasting with a Mixed-Frequency VAR." *Journal of Business & Economic Statistics, 33*(3), 366-380.
  - *Alternative mixed-frequency approach to nowcasting*

---

**Next Guide:** Forecast Evaluation - rigorous assessment of nowcast accuracy and comparison methods
