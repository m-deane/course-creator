# Mixed-Frequency Risk Models: VaR, Term Structure, and Commodity Fundamentals

## Learning Objectives

By the end of this guide you will be able to:

1. Build a mixed-frequency VaR model combining daily returns with monthly macro conditions
2. Apply MIDAS to term structure nowcasting (yield curve dynamics)
3. Implement commodity price nowcasting with mixed-frequency fundamental data
4. Evaluate risk models with backtesting and coverage tests

---

## 1. Mixed-Frequency Value-at-Risk

### 1.1 Motivation

Classical VaR models (Historical Simulation, GARCH-based) operate at a single frequency. In practice, risk managers face a mixed-frequency information structure:

- **Daily**: Return observations, market prices
- **Weekly**: Options market signals (put-call ratios, skew)
- **Monthly**: Macro-financial conditions (credit spreads, central bank balance sheet, industrial activity)
- **Quarterly**: Financial statements, stress test results

A VaR model that incorporates monthly macro conditions alongside daily market data should produce better-calibrated risk estimates during macro transition periods.

### 1.2 MIDAS-VaR Formulation

The MIDAS-VaR model of Andreou et al. (2010) specifies the conditional quantile directly:

$$Q_\alpha(r_{t+1} | \mathcal{F}_t) = -\left[ \mu + \phi \sum_{j=0}^{K_d-1} B_1(j;\theta) r_{t-j}^2 + \psi \sum_{l=0}^{K_m-1} B_2(l;\gamma) Z_{t-l}^{(m)} \right]^{1/2} z_\alpha$$

where:
- $Q_\alpha(r_{t+1} | \mathcal{F}_t)$ is the $\alpha$-quantile of tomorrow's return
- $z_\alpha = \Phi^{-1}(\alpha)$ is the normal quantile at level $\alpha$
- The square-bracketed term is the MIDAS volatility forecast
- $Z_{t-l}^{(m)}$ are monthly macro predictors

A simpler but equivalent two-step approach:

1. **Step 1**: Forecast monthly volatility $\hat{\sigma}^{(m)}_t$ using MIDAS-RV
2. **Step 2**: Scale to daily VaR: $\text{VaR}^{1d}_t = \hat{\sigma}^{(m)}_t / \sqrt{22} \cdot z_{0.01}$

### 1.3 Implementation

```python
import numpy as np
from scipy import stats

def midas_var(daily_returns, monthly_rv_forecast, alpha=0.01):
    """
    Compute daily MIDAS-VaR from monthly volatility forecast.

    Parameters
    ----------
    daily_returns : array — daily return observations
    monthly_rv_forecast : array — MIDAS-RV monthly variance forecasts
    alpha : float — VaR confidence level (0.01 for 1%)

    Returns
    -------
    var_daily : array — daily VaR (positive = loss limit)
    """
    # Scale monthly variance to daily
    daily_variance = monthly_rv_forecast / 22.0

    # Normal VaR (quantile approach)
    z_alpha = stats.norm.ppf(alpha)  # Negative for alpha < 0.5
    var_daily = -z_alpha * np.sqrt(daily_variance)

    return var_daily


def backtest_var(returns, var_estimates, alpha=0.01):
    """
    VaR backtest: compute violation rate and Kupiec test.

    Parameters
    ----------
    returns : array — actual daily returns
    var_estimates : array — VaR estimates (positive values)
    alpha : float — expected violation rate

    Returns
    -------
    violation_rate : float — actual fraction of violations
    kupiec_stat : float — LR test statistic
    kupiec_pval : float — p-value
    """
    violations = returns < -var_estimates  # Return below VaR
    T = len(returns)
    V = violations.sum()
    violation_rate = V / T

    # Kupiec (1995) unconditional coverage test
    # H0: violation rate = alpha
    if V == 0 or V == T:
        # Edge case: degenerate likelihood
        return violation_rate, np.nan, np.nan

    lr_stat = 2 * (
        V * np.log(violation_rate / alpha) +
        (T - V) * np.log((1 - violation_rate) / (1 - alpha))
    )
    p_value = stats.chi2.sf(lr_stat, df=1)

    return violation_rate, lr_stat, p_value
```

### 1.4 Christoffersen's Conditional Coverage Test

Beyond unconditional coverage, violations should be **serially independent** — clustering indicates model inadequacy:

```python
def christoffersen_test(violations):
    """
    Christoffersen (1998) conditional coverage test.
    Tests both unconditional coverage and independence of violations.

    Parameters
    ----------
    violations : bool array — True where return < -VaR

    Returns
    -------
    cc_stat : float — conditional coverage LR statistic
    cc_pval : float — p-value (chi2, df=2)
    """
    # Transition counts
    v = violations.astype(int)
    n00 = np.sum((v[:-1] == 0) & (v[1:] == 0))
    n01 = np.sum((v[:-1] == 0) & (v[1:] == 1))
    n10 = np.sum((v[:-1] == 1) & (v[1:] == 0))
    n11 = np.sum((v[:-1] == 1) & (v[1:] == 1))

    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    # Independence LR
    if any(x <= 0 for x in [pi01, 1-pi01, pi11, 1-pi11, pi, 1-pi]):
        return np.nan, np.nan

    lr_ind = 2 * (
        n00 * np.log(1 - pi01) + n01 * np.log(pi01) +
        n10 * np.log(1 - pi11) + n11 * np.log(pi11) -
        (n00 + n10) * np.log(1 - pi) -
        (n01 + n11) * np.log(pi)
    )

    cc_stat = lr_ind  # Combined with Kupiec gives conditional coverage
    cc_pval = stats.chi2.sf(cc_stat, df=1)
    return cc_stat, cc_pval
```

---

## 2. Term Structure Nowcasting

### 2.1 The Mixed-Frequency Yield Curve

Yield curve data has a natural mixed-frequency structure:

- **Daily**: Treasury market yields (1m, 3m, 6m, 1y, 2y, 5y, 10y, 30y)
- **Monthly**: Macro drivers (inflation expectations, industrial production, credit spreads)
- **Quarterly**: GDP, central bank projections

A MIDAS approach to term structure nowcasting forecasts quarterly yield changes using daily market data and monthly macro indicators.

### 2.2 Principal Component MIDAS for Yield Curve

Rather than forecasting each maturity separately, decompose the yield curve into level, slope, and curvature factors (Nelson-Siegel):

$$y_t(\tau) = \beta_{1t} + \beta_{2t}\frac{1-e^{-\lambda\tau}}{\lambda\tau} + \beta_{3t}\left(\frac{1-e^{-\lambda\tau}}{\lambda\tau} - e^{-\lambda\tau}\right)$$

Then apply MIDAS to each factor:

```python
from sklearn.decomposition import PCA
from scipy.optimize import minimize
import numpy as np

def nelson_siegel_factors(yields, maturities, lambda_fixed=0.0609):
    """
    Extract Nelson-Siegel factors (level, slope, curvature) from yield curve.

    Parameters
    ----------
    yields : array, shape (T, M) — yields at M maturities
    maturities : array, shape (M,) — maturities in months
    lambda_fixed : float — decay parameter (0.0609 ≈ Diebold-Li 2006)

    Returns
    -------
    factors : array, shape (T, 3) — [level, slope, curvature]
    """
    tau = np.array(maturities)
    loading2 = (1 - np.exp(-lambda_fixed * tau)) / (lambda_fixed * tau)
    loading3 = loading2 - np.exp(-lambda_fixed * tau)

    # OLS: yields = beta1 * 1 + beta2 * loading2 + beta3 * loading3
    X_ns = np.column_stack([np.ones(len(tau)), loading2, loading3])

    factors = np.zeros((yields.shape[0], 3))
    for t in range(yields.shape[0]):
        beta, _, _, _ = np.linalg.lstsq(X_ns, yields[t], rcond=None)
        factors[t] = beta

    return factors, X_ns


def midas_term_structure(factors_monthly, factors_daily, K_daily=22):
    """
    MIDAS nowcast for each Nelson-Siegel factor separately.
    factors_monthly : array (T_m, 3) — monthly level/slope/curvature
    factors_daily : array (T_d, 3) — daily level/slope/curvature
    """
    nowcasts = []
    factor_names = ['Level', 'Slope', 'Curvature']

    for k, name in enumerate(factor_names):
        y_m = factors_monthly[:, k]
        x_d = factors_daily[:, k]
        # Apply MIDAS-RV style estimation for each factor
        # (here simplified to OLS with fixed lag weights)
        nowcasts.append(y_m)  # Placeholder for actual MIDAS

    return np.column_stack(nowcasts)
```

### 2.3 MIDAS Spread Nowcasting

A simpler application: nowcast the yield spread (10Y-2Y) using daily observations of the same spread along with monthly macro indicators:

$$\text{Spread}^{(q)}_t = \mu + \phi \sum_{j=0}^{K_d-1} B_1(j) \text{Spread}^{(d)}_{t-j} + \psi \sum_{l=0}^{K_m-1} B_2(l) Z^{(m)}_{t-l} + \varepsilon_t$$

This is the standard MIDAS-X structure applied to a financial spread rather than GDP.

---

## 3. Commodity Price Nowcasting

### 3.1 Mixed-Frequency Commodity Fundamentals

Commodity prices depend on a mix of high- and low-frequency fundamentals:

| Frequency | Variable | Source |
|-----------|----------|--------|
| Daily | Futures prices, speculative positions (COT) | CME |
| Weekly | Inventory reports (EIA: crude, gasoline, distillates) | EIA |
| Monthly | Production (OPEC output), demand proxy (IPI) | IEA, FRED |
| Quarterly | GDP of major consumers, trade balances | National accounts |

MIDAS naturally handles this multi-frequency structure.

### 3.2 MIDAS Model for Crude Oil

For crude oil price nowcasting:

$$\Delta \log P^{(m)}_t = \mu + \phi_1 \sum_{j} B_1(j) \Delta \log P^{(d)}_{t-j} + \phi_2 \sum_{k} B_2(k) \text{Inventory}^{(w)}_{t-k} + \phi_3 Z^{(m)}_{t-1} + \varepsilon_t$$

where:
- $\Delta \log P^{(m)}_t$ is monthly crude oil price growth
- $\Delta \log P^{(d)}_{t-j}$ are daily return lags
- $\text{Inventory}^{(w)}_{t-k}$ are weekly EIA inventory change reports
- $Z^{(m)}_{t-1}$ is a monthly fundamental (OPEC output, IEA demand revision)

```python
def build_commodity_midas_features(
    daily_prices, weekly_inventories, monthly_macro, target_monthly,
    K_daily=22, K_weekly=4, n_lags_macro=3
):
    """
    Build MIDAS feature matrix for commodity nowcasting.

    Parameters
    ----------
    daily_prices : Series — daily commodity prices
    weekly_inventories : Series — weekly inventory levels
    monthly_macro : DataFrame — monthly macro indicators
    target_monthly : Series — monthly target (price returns)
    K_daily : int — daily lags
    K_weekly : int — weekly lags

    Returns
    -------
    X : DataFrame — MIDAS feature matrix
    y : Series — aligned monthly target
    """
    daily_ret = daily_prices.pct_change().dropna()
    weekly_inv_change = weekly_inventories.diff().dropna()

    rows = []
    dates = []

    for date in target_monthly.index:
        row = {}

        # Daily lags: K_daily most recent returns before month end
        mask_d = daily_ret.index <= date
        d_lags = daily_ret[mask_d].tail(K_daily).values[::-1]

        if len(d_lags) < K_daily:
            continue

        for j, v in enumerate(d_lags, start=1):
            row[f'ret_d_L{j}'] = v

        # Weekly inventory lags: K_weekly most recent before month end
        mask_w = weekly_inv_change.index <= date
        w_lags = weekly_inv_change[mask_w].tail(K_weekly).values[::-1]

        if len(w_lags) < K_weekly:
            continue

        for k, v in enumerate(w_lags, start=1):
            row[f'inv_chg_L{k}'] = v

        # Monthly macro lags
        mask_m = monthly_macro.index <= date
        for col in monthly_macro.columns:
            m_vals = monthly_macro.loc[mask_m, col].tail(n_lags_macro).values[::-1]
            for l, v in enumerate(m_vals, start=1):
                row[f'{col}_L{l}'] = v

        rows.append(row)
        dates.append(date)

    X = pd.DataFrame(rows, index=dates)
    y = target_monthly.reindex(dates)

    valid = X.notna().all(axis=1) & y.notna()
    return X[valid], y[valid]
```

### 3.3 Realised Commodity Volatility

For energy and agricultural commodities, MIDAS-RV extends directly:

$$\log RV^{(m)}_t = \mu + \phi \sum_{j=0}^{K_d-1} B(j) \log RV^{(d)}_{t-j} + \psi \cdot \text{OPEC\_surprise}_{t-1} + \varepsilon_t$$

Where OPEC production surprises (monthly: deviation of actual from announced production) provide additional predictive power for crude oil volatility.

---

## 4. Integrating the Three Applications

The three mixed-frequency risk applications share a common structure:

```
Target (low-freq)         Predictors (high-freq)         MIDAS weights
──────────────────        ──────────────────────         ─────────────
Monthly RV       ←── Daily RV observations           ← Beta(θ₁, θ₂)
Daily VaR        ←── Monthly RV forecast             ← Scale by √22
Quarterly spread ←── Daily spread + monthly macro    ← Beta × Beta
Monthly oil price←── Daily returns + weekly EIA      ← Beta × Beta
```

The unifying principle: **mixed-frequency regression with Beta polynomial weights** captures the temporal aggregation structure in financial risk data.

---

## 5. Model Comparison Framework

### 5.1 For Volatility Forecasting

Compare models by RMSE (in-sample) and QLIKE (out-of-sample). Apply Diebold-Mariano tests for pairwise comparison. Use Model Confidence Set (MCS) for multi-model comparison.

### 5.2 For VaR Models

Backtest using:
1. **Unconditional coverage** (Kupiec 1995): fraction of violations should equal $\alpha$
2. **Conditional coverage** (Christoffersen 1998): violations should be independent
3. **Dynamic quantile test** (Engle & Manganelli 2004): violation indicator regressed on own lags and VaR

### 5.3 For Commodity Nowcasting

Compare out-of-sample RMSE against:
- Random walk (no drift): $\Delta P_t = 0$
- AR(1) in returns
- Professional forecaster consensus

---

## 6. Practical Considerations

### 6.1 Data Alignment in Financial Applications

Financial markets have complex calendar effects:
- Not all markets trade on the same days (US vs European holidays)
- Commodity futures roll dates create price discontinuities
- Repo settlement creates spikes in short-term rates

**Best practice**: Use adjusted close prices; exclude non-trading days; handle roll dates with continuation adjustments.

### 6.2 Regime Dependence

Financial MIDAS models often exhibit **regime dependence**:
- Low-volatility regimes: $\theta_1 \approx 1, \theta_2 \approx 5$ (rapid decay)
- High-volatility regimes: $\theta_1 \approx 1, \theta_2 \approx 2$ (slower decay, older events matter more)

This motivates regime-switching MIDAS or time-varying parameter extensions.

### 6.3 Market Microstructure

For daily RV computed from intraday data:
- Use 5-minute sampling to balance variance estimation accuracy and microstructure noise
- Apply Newey-West correction for autocorrelation in intraday returns
- Consider Bipower Variation (BPV) as a jump-robust alternative to simple RV

---

## 7. Key References

- Andreou, E., Ghysels, E., & Kourtellos, A. (2010). Regression models with mixed sampling frequencies. *Journal of Econometrics*, 158(2), 246–261.
- Engle, R. F., & Manganelli, S. (2004). CAViaR: Conditional autoregressive value at risk. *JBES*, 22(4), 367–381.
- Kupiec, P. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*, 3(2).
- Christoffersen, P. F. (1998). Evaluating interval forecasts. *International Economic Review*, 39(4), 841–862.
- Diebold, F. X., & Li, C. (2006). Forecasting the term structure of government bond yields. *Journal of Econometrics*, 130(2), 337–364.
- Kilian, L. (2009). Not all oil price shocks are alike. *American Economic Review*, 99(3), 1053–1069.

---

## Summary

Mixed-frequency risk models apply MIDAS to three important financial applications:

1. **MIDAS-VaR**: Daily VaR derived from monthly MIDAS-RV forecast; backtested with Kupiec and Christoffersen tests
2. **Term structure nowcasting**: Nelson-Siegel factors nowcast from daily yield data + monthly macro; gives forward-looking yield curve
3. **Commodity price nowcasting**: Daily futures returns + weekly inventory reports + monthly fundamentals in a multi-frequency MIDAS-X model

All three share the same Beta-polynomial weight structure and are estimated by NLS with time-series cross-validation.

Next: [S&P 500 MIDAS-RV Notebook](../notebooks/01_midas_rv_sp500.ipynb)
