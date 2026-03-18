# MIDAS-RV: Realised Volatility Forecasting

## Learning Objectives

By the end of this guide you will be able to:

1. Derive the MIDAS-RV model of Ghysels, Santa-Clara, and Valkanov (2005)
2. Implement the MIDAS-RV estimator with Beta polynomial weights in Python
3. Compare MIDAS-RV against GARCH and HAR-RV benchmarks
4. Apply mixed-frequency extensions: MIDAS-RV-X with macro predictors
5. Evaluate volatility forecast accuracy with QLIKE and MSE loss functions

---

## 1. The Volatility Forecasting Problem

Volatility is the central input to option pricing, risk management, and portfolio construction. It is:

- **Unobservable**: We cannot directly observe the "true" volatility at any point
- **Persistent**: High volatility today predicts high volatility tomorrow (ARCH effects)
- **Asymmetric**: Volatility tends to be higher after negative returns (leverage effect)
- **Multi-scale**: Different components (short-run noise, medium-run cycles, long-run trends) operate at different frequencies

The advent of high-frequency transaction data since the 1990s enabled computation of **realised volatility (RV)** as a proxy for true latent volatility. MIDAS-RV exploits the natural mixed-frequency structure: we forecast monthly or quarterly equity return volatility using daily or intraday RV.

---

## 2. Realised Volatility

### 2.1 Definition

For day $d$ with $n$ intraday return observations $r_{d,i}$:

$$RV_d = \sum_{i=1}^{n} r_{d,i}^2$$

Under the Andersen-Bollerslev theory, as $n \to \infty$ and the sampling interval shrinks, $RV_d$ converges to the integrated variance:

$$RV_d \xrightarrow{p} \int_{d-1}^{d} \sigma^2(t) \, dt$$

### 2.2 Realised Variation Measures

Beyond simple RV, several refinements exist:

| Measure | Formula | Property |
|---------|---------|----------|
| RV | $\sum r_i^2$ | Baseline |
| BPV | $\sum \|r_i\| \cdot \|r_{i-1}\|$ | Robust to jumps |
| RQ | $\sum r_i^4$ | Related to variance of variance |
| MinRV, MedRV | Truncated versions | Jump-robust |

In the MIDAS-RV context, the daily RV series is the high-frequency input. Monthly or quarterly RV is the target:

$$RV^{(m)}_t = \sum_{d \in \text{quarter}_t} RV_d$$

---

## 3. The MIDAS-RV Model

### 3.1 Original Formulation (Ghysels, Santa-Clara, Valkanov 2005)

$$r^2_{t+1} = \mu + \phi \sum_{j=0}^{K-1} B(j; \theta_1, \theta_2) \cdot RV_{t-j}^{(d)} + \varepsilon_{t+1}$$

where:
- $r^2_{t+1}$ is the squared monthly or quarterly return (proxy for low-frequency volatility)
- $RV_{t-j}^{(d)}$ is realised volatility at lag $j$ daily periods before time $t$
- $B(j; \theta_1, \theta_2)$ is the two-parameter Beta weighting function
- $K$ is the number of daily lags (typically 22 for one month, 66 for one quarter)

The Beta weights are normalised to sum to 1:

$$B(j; \theta_1, \theta_2) = \frac{(j/K)^{\theta_1-1}(1-j/K)^{\theta_2-1}}{\sum_{k=0}^{K-1}(k/K)^{\theta_1-1}(1-k/K)^{\theta_2-1}}$$

### 3.2 Equivalent Formulation in Logs

Because RV is right-skewed and always positive, working in logs improves normality:

$$\log RV^{(m)}_{t+1} = \mu + \phi \sum_{j=0}^{K-1} B(j; \theta_1, \theta_2) \cdot \log RV^{(d)}_{t-j} + \varepsilon_{t+1}$$

Empirically, the log specification produces better-calibrated forecast intervals.

### 3.3 Key Economic Insight

The MIDAS-RV model captures that **not all lags of daily RV are equally informative**. The Beta weights allow the most informative lag pattern to be learned from the data:

- If $\theta_1 < 1$: Highest weights at recent lags (most recent day dominates)
- If $\theta_1 = \theta_2 = 1$: Uniform weights (all lags equal)
- If $\theta_1 > 1, \theta_2 > 1$: Interior mode (medium lags most important)
- If $\theta_1 > 1, \theta_2 < 1$: Hump shape (most important lag in the middle)

---

## 4. Implementation

### 4.1 Beta Weight Function

```python
import numpy as np
from scipy.optimize import minimize
from scipy.special import beta as beta_function

def beta_weights(K, theta1, theta2):
    """
    Compute normalised Beta polynomial weights for MIDAS-RV.

    Parameters
    ----------
    K : int — number of high-frequency lags
    theta1, theta2 : float — Beta distribution parameters (both > 0)

    Returns
    -------
    weights : array, shape (K,) — normalised weights summing to 1
    """
    # Evaluation points: uniform grid on [0, 1]
    x = np.linspace(0.001, 0.999, K)
    # Un-normalised beta density
    unnorm = x**(theta1 - 1) * (1 - x)**(theta2 - 1)
    # Normalise
    weights = unnorm / unnorm.sum()
    return weights


def plot_beta_weights(ax, K=22):
    """Illustrate different Beta weight shapes."""
    configs = [
        (1.0, 5.0, 'θ₁=1, θ₂=5 (geometric decay)'),
        (1.0, 1.0, 'θ₁=1, θ₂=1 (uniform)'),
        (2.0, 5.0, 'θ₁=2, θ₂=5 (hump-shaped)'),
        (5.0, 1.0, 'θ₁=5, θ₂=1 (reverse decay)'),
    ]
    lags = np.arange(1, K + 1)
    for theta1, theta2, label in configs:
        weights = beta_weights(K, theta1, theta2)
        ax.plot(lags, weights, linewidth=2, label=label, marker='o', markersize=3)
    ax.set_xlabel('Daily lag j')
    ax.set_ylabel('Weight B(j; θ₁, θ₂)')
    ax.set_title('Beta Weighting Functions for MIDAS-RV')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
```

### 4.2 MIDAS-RV Estimator

```python
from scipy.optimize import minimize

def midas_rv_objective(params, rv_daily, rv_monthly, K):
    """
    NLS objective for MIDAS-RV.

    Parameters
    ----------
    params : array [mu, phi, theta1, theta2]
    rv_daily : array — daily RV observations
    rv_monthly : array — monthly RV target
    K : int — daily lags per month

    Returns
    -------
    sse : float — sum of squared errors
    """
    mu, phi, theta1, theta2 = params

    # Enforce positivity constraints on beta parameters
    if theta1 <= 0 or theta2 <= 0:
        return 1e10

    weights = beta_weights(K, theta1, theta2)

    T = len(rv_monthly)
    errors = np.zeros(T)

    for t in range(T):
        # Daily RV lags ending at t*K (current month)
        end_idx = (t + 1) * K
        start_idx = end_idx - K
        if start_idx < 0 or end_idx > len(rv_daily):
            continue

        daily_lags = rv_daily[start_idx:end_idx][::-1]  # Most recent first
        fitted = mu + phi * np.dot(weights, daily_lags)
        errors[t] = rv_monthly[t] - fitted

    return np.sum(errors**2)


def fit_midas_rv(rv_daily, rv_monthly, K=22, log_transform=True):
    """
    Fit MIDAS-RV model by NLS.

    Parameters
    ----------
    rv_daily : array — daily RV observations
    rv_monthly : array — monthly RV target
    K : int — daily lags per month
    log_transform : bool — fit in logs

    Returns
    -------
    params : dict — estimated parameters
    fitted : array — in-sample fitted values
    """
    if log_transform:
        rv_d = np.log(rv_daily + 1e-10)
        rv_m = np.log(rv_monthly + 1e-10)
    else:
        rv_d = rv_daily.copy()
        rv_m = rv_monthly.copy()

    # Initial values: mu=mean(rv_m), phi=0.5, theta1=1, theta2=5
    x0 = [np.mean(rv_m), 0.5, 1.0, 5.0]
    bounds = [(None, None), (0, 2), (0.01, 20), (0.01, 20)]

    result = minimize(
        midas_rv_objective,
        x0=x0,
        args=(rv_d, rv_m, K),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 2000}
    )

    if not result.success:
        print(f"Warning: optimiser did not converge. Message: {result.message}")

    mu, phi, theta1, theta2 = result.x
    weights = beta_weights(K, theta1, theta2)

    T = len(rv_m)
    fitted = np.zeros(T)
    for t in range(T):
        end_idx = (t + 1) * K
        start_idx = end_idx - K
        if start_idx < 0 or end_idx > len(rv_d):
            continue
        daily_lags = rv_d[start_idx:end_idx][::-1]
        fitted[t] = mu + phi * np.dot(weights, daily_lags)

    return {
        'mu': mu, 'phi': phi,
        'theta1': theta1, 'theta2': theta2,
        'weights': weights,
        'sse': result.fun,
        'converged': result.success
    }, fitted
```

---

## 5. HAR-RV Benchmark

The **Heterogeneous Autoregressive model** (Corsi 2009) is the standard benchmark for RV forecasting. It exploits the multi-scale structure of volatility:

$$RV^{(d)}_{t+1} = c + \beta^{(d)} RV^{(d)}_t + \beta^{(w)} RV^{(w)}_t + \beta^{(m)} RV^{(m)}_t + \varepsilon_{t+1}$$

where $RV^{(w)}_t = \frac{1}{5}\sum_{j=0}^{4} RV^{(d)}_{t-j}$ (weekly average) and $RV^{(m)}_t = \frac{1}{22}\sum_{j=0}^{21} RV^{(d)}_{t-j}$ (monthly average).

```python
def fit_har_rv(rv_daily):
    """
    Fit HAR-RV: daily target, daily/weekly/monthly components as predictors.
    """
    T = len(rv_daily)
    max_lag = 22  # Monthly component

    rv_d = rv_daily[max_lag:]    # Daily target
    rv_d_lag = rv_daily[max_lag-1:-1]  # Daily predictor
    rv_w_lag = np.array([rv_daily[i-5:i].mean() for i in range(max_lag, T+1)])[:-1]
    rv_m_lag = np.array([rv_daily[i-22:i].mean() for i in range(max_lag, T+1)])[:-1]

    X = np.column_stack([np.ones(len(rv_d)), rv_d_lag, rv_w_lag, rv_m_lag])
    # OLS
    beta = np.linalg.lstsq(X, rv_d, rcond=None)[0]
    fitted = X @ beta
    residuals = rv_d - fitted
    rmse = np.sqrt(np.mean(residuals**2))

    return beta, fitted, residuals, rmse
```

---

## 6. Volatility Forecast Evaluation

### 6.1 Loss Functions

Volatility forecasting uses specialised loss functions because squared returns are the noisy proxy for latent variance:

**MSE** (symmetric, simple):
$$MSE = \frac{1}{T}\sum_t (\sigma^2_t - \hat{\sigma}^2_t)^2$$

**QLIKE** (asymmetric, based on quasi-likelihood):
$$QLIKE = \frac{1}{T}\sum_t \left(\frac{\sigma^2_t}{\hat{\sigma}^2_t} - \log\frac{\sigma^2_t}{\hat{\sigma}^2_t} - 1\right)$$

QLIKE penalises under-prediction of variance more than MSE (relevant for risk management where under-prediction is costly).

```python
def qlike_loss(actual, forecast):
    """
    QLIKE loss function for volatility forecasting.
    Actual and forecast should be variance (not standard deviation).
    """
    u = actual / (forecast + 1e-10)
    return np.mean(u - np.log(u) - 1)


def mse_loss(actual, forecast):
    return np.mean((actual - forecast)**2)


def evaluate_vol_forecasts(actuals, forecasts_dict, transform='level'):
    """
    Compare volatility forecasts with MSE and QLIKE.

    Parameters
    ----------
    actuals : array — realised variance (RV)
    forecasts_dict : dict — {model_name: forecast_array}
    transform : str — 'level' (variance) or 'log' (log-variance)
    """
    if transform == 'log':
        actuals = np.exp(actuals)
        forecasts_dict = {k: np.exp(v) for k, v in forecasts_dict.items()}

    print(f"{'Model':<20} {'MSE':<15} {'QLIKE':<15}")
    print("-" * 50)
    for name, fcasts in forecasts_dict.items():
        mse = mse_loss(actuals, fcasts)
        ql = qlike_loss(actuals, fcasts)
        print(f"{name:<20} {mse:<15.6f} {ql:<15.6f}")
```

### 6.2 Mincer-Zarnowitz Regression

Test forecast efficiency: regress realised variance on the forecast:

$$RV_{t+1} = a + b \cdot \hat{RV}_{t+1} + \varepsilon_{t+1}$$

An efficient forecast satisfies $H_0: a=0, b=1$.

```python
from scipy import stats as scipy_stats

def mincer_zarnowitz(actuals, forecasts, name='Model'):
    """
    Mincer-Zarnowitz regression test of forecast efficiency.
    H0: intercept = 0, slope = 1 (efficient forecast).
    """
    X = np.column_stack([np.ones(len(forecasts)), forecasts])
    beta = np.linalg.lstsq(X, actuals, rcond=None)[0]
    a, b = beta

    fitted = X @ beta
    resid = actuals - fitted
    s2 = np.mean(resid**2)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(s2 * np.diag(XtX_inv))

    t_a = a / se[0]  # H0: a = 0
    t_b = (b - 1) / se[1]  # H0: b = 1

    print(f"\n{name} — Mincer-Zarnowitz Test:")
    print(f"  Intercept: {a:.6f} (SE={se[0]:.6f}, t={t_a:.3f})")
    print(f"  Slope:     {b:.6f} (SE={se[1]:.6f}, t={t_b:.3f})")
    print(f"  R²:        {1 - np.var(resid)/np.var(actuals):.4f}")
    if abs(t_a) < 1.96 and abs(t_b) < 1.96:
        print("  Forecast is EFFICIENT (fail to reject H0: a=0, b=1)")
    else:
        print("  Forecast is BIASED (reject H0: a=0, b=1)")
    return a, b
```

---

## 7. MIDAS-RV-X: Adding Macro Predictors

The MIDAS-RV-X model extends MIDAS-RV with additional mixed-frequency macro predictors:

$$\log RV^{(m)}_{t+1} = \mu + \phi \sum_{j=0}^{K_1-1} B_1(j) \log RV^{(d)}_{t-j} + \psi \sum_{l=0}^{K_2-1} B_2(l) Z^{(m)}_{t-l} + \varepsilon_{t+1}$$

where $Z^{(m)}_{t-l}$ is a monthly macro predictor (e.g., default spread, TED spread, VIX index).

This specification jointly models short-run volatility dynamics (daily RV component) and longer-run macro-financial conditions (monthly predictor component).

```python
def midas_rv_x_objective(params, rv_daily, rv_monthly, macro_monthly, K_rv=22, K_macro=6):
    """
    NLS objective for MIDAS-RV-X (with macro predictor).
    params: [mu, phi, theta1_rv, theta2_rv, psi, theta1_macro, theta2_macro]
    """
    mu, phi, t1_rv, t2_rv, psi, t1_m, t2_m = params

    if any(p <= 0 for p in [t1_rv, t2_rv, t1_m, t2_m]):
        return 1e10

    w_rv = beta_weights(K_rv, t1_rv, t2_rv)
    w_m = beta_weights(K_macro, t1_m, t2_m)

    rv_d_log = np.log(rv_daily + 1e-10)
    rv_m_log = np.log(rv_monthly + 1e-10)

    T = len(rv_monthly)
    sse = 0.0
    for t in range(max(K_rv // 22, K_macro), T):
        end_d = t * K_rv
        start_d = end_d - K_rv
        if start_d < 0 or end_d > len(rv_d_log):
            continue

        rv_component = phi * np.dot(w_rv, rv_d_log[start_d:end_d][::-1])
        macro_component = psi * np.dot(w_m, macro_monthly[max(0, t-K_macro):t][::-1])
        fitted = mu + rv_component + macro_component
        sse += (rv_m_log[t] - fitted)**2

    return sse
```

---

## 8. Connection to Risk Management

MIDAS-RV feeds directly into Value-at-Risk (VaR) computation:

$$\text{VaR}^{(\alpha)}_t = -z_\alpha \cdot \hat{\sigma}_t \cdot \sqrt{h}$$

where $\hat{\sigma}_t = \sqrt{\hat{RV}^{(m)}_t}$ is the MIDAS-RV forecast and $z_\alpha$ is the quantile from the return distribution.

Alternatively, for a historical simulation approach, MIDAS-RV provides the volatility-scaling factor to adjust historical returns to current conditions.

---

## 9. Key References

- Ghysels, E., Santa-Clara, P., & Valkanov, R. (2005). There is a risk-return trade-off after all. *Journal of Financial Economics*, 76(3), 509–548.
- Ghysels, E., Santa-Clara, P., & Valkanov, R. (2006). Predicting volatility: getting the most out of return data sampled at different frequencies. *Journal of Econometrics*, 131(1-2), 59–95.
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *JFEC*, 7(2), 174–196.
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*, 160(1), 246–256.

---

## Summary

MIDAS-RV is the canonical model for forecasting multi-period volatility using high-frequency daily RV data:

- **Model**: Low-frequency RV (monthly/quarterly) as a function of Beta-weighted daily RV lags
- **Beta weights**: Flexible parameterisation; exponential decay ($\theta_1=1$) is most common empirically
- **Log specification**: Better finite-sample properties; always preferred in practice
- **Benchmarks**: Compare against GARCH-M and HAR-RV using QLIKE loss
- **Extension**: MIDAS-RV-X adds monthly macro predictors (default spreads, VIX) to capture macro-financial regime effects

Next: [Mixed-Frequency Risk Models](02_mixed_freq_risk_guide.md) — daily VaR, term structure nowcasting, commodity fundamentals.
