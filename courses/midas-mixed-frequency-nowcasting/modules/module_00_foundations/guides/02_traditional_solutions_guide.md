# Traditional Solutions to the Mixed-Frequency Problem

> **Reading time:** ~14 min | **Module:** 00 — Foundations | **Prerequisites:** None (entry point)


## In Brief

<div class="flow">
<div class="flow-step mint">1. Collect Data</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">2. Identify Frequencies</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">3. Align Time Indices</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. Build MIDAS Regressors</div>
</div>


<div class="callout-key">

**Key Concept Summary:** Before MIDAS, practitioners handled mixed-frequency data through temporal aggregation, interpolation, and bridge equations. Each approach trades off information loss against tractability. Understan...

</div>

Before MIDAS, practitioners handled mixed-frequency data through temporal aggregation, interpolation, and bridge equations. Each approach trades off information loss against tractability. Understanding their limitations motivates the MIDAS framework precisely.

## Key Insight

<div class="callout-insight">

**Insight:** The mixed-frequency approach preserves within-period dynamics that aggregation destroys. This is especially valuable when the timing of high-frequency movements carries economic information.

</div>


Every traditional solution pre-commits to a data transformation before estimation begins. MIDAS defers that commitment to estimation, letting the data determine the aggregation weights. The gap in forecast accuracy between the two approaches is the empirical measure of how much pre-aggregation costs.

---

## Strategy 1: Temporal Aggregation

<div class="callout-warning">

**Warning:** Be cautious about extrapolating MIDAS performance from stable periods to crisis periods. The relationship between high-frequency indicators and the low-frequency target can shift dramatically during regime changes.

</div>


### What It Is

Transform the high-frequency series into a low-frequency series by applying a fixed aggregation function, then run a standard regression.

**For flow variables** (GDP, production, payrolls — quantities that accumulate over time):

$$\tilde{x}_t = \sum_{j=1}^{m} x_{mt-m+j}^H$$

Sum the high-frequency values within each low-frequency period.

**For stock variables** (prices, rates, levels — quantities measured at a point):

$$\tilde{x}_t = x_{mt}^H \quad \text{(end-of-period sampling)}$$

or

$$\tilde{x}_t = \frac{1}{m} \sum_{j=1}^{m} x_{mt-m+j}^H \quad \text{(period average)}$$

where $m$ is the frequency ratio (e.g., $m=3$ for monthly-to-quarterly, $m=65$ for daily-to-quarterly).

### Worked Example

Monthly industrial production growth $x_\tau^M$ aggregated to quarterly:


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import pandas as pd
import numpy as np

# Monthly IP growth observations within a quarter
# Quarter 2020Q1: Jan, Feb, Mar
monthly_ip = pd.Series(
    [0.003, -0.012, -0.065],  # Jan, Feb, Mar 2020
    index=pd.period_range("2020-01", periods=3, freq="M")
)

# Strategy 1: Last-period sampling
last_period = monthly_ip.iloc[-1]
print(f"Last period: {last_period:.4f}")   # -0.0650 (March only)

# Strategy 2: Equal-weight average
equal_avg = monthly_ip.mean()
print(f"Equal average: {equal_avg:.4f}")  # -0.0247 (smooths the March shock)

# Strategy 3: Sum (appropriate for a flow variable like production index change)
total = monthly_ip.sum()
print(f"Sum: {total:.4f}")  # -0.0740
```

</div>
</div>

Notice that equal averaging effectively weights March 2020 (the large COVID shock) at only 1/3. Last-period sampling captures the shock, but misses the pre-shock values that provide context.

### Information Lost

The equal-weight aggregation from $m$ to 1 period collapses $m$ parameters into 1. The Shannon information reduction equals $\log_2(m)$ bits under a naive model. For monthly-to-quarterly ($m=3$), we lose $\log_2(3) \approx 1.58$ bits per observation. For daily-to-quarterly ($m=65$), we lose $\log_2(65) \approx 6$ bits.

---

## Strategy 2: Interpolation

### What It Is

Rather than aggregating high-frequency data down to low frequency, interpolation infers synthetic high-frequency values for a low-frequency series, then analyzes at the high frequency.

### Chow-Lin Interpolation

The classic method (Chow and Lin, 1971) specifies:

$$y_t^H = c \cdot z_t + \varepsilon_t, \quad \varepsilon_t = \rho \varepsilon_{t-1} + u_t$$

where $z_t$ is a related high-frequency indicator and $c$ is estimated to ensure that the interpolated series aggregates back to the observed low-frequency values.

Formally, let $C$ be the temporal aggregation matrix mapping $T_H$ high-frequency periods to $T_L$ low-frequency periods. We require:

$$C \hat{y}^H = y^L$$

The GLS solution provides minimum-variance interpolated values consistent with the aggregation constraint.


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# Conceptual illustration of Chow-Lin interpolation
# In practice, use statsmodels or the tempdisagg R package (via rpy2)

import numpy as np
from scipy.linalg import block_diag

def chow_lin_interpolate(y_low, z_high, m=3, rho=0.5):
    """
    Interpolate a low-frequency series using a high-frequency indicator.

    Parameters
    ----------
    y_low : np.ndarray, shape (T_L,)
        Observed low-frequency series.
    z_high : np.ndarray, shape (T_H,)
        High-frequency indicator, where T_H = m * T_L.
    m : int
        Frequency ratio (e.g., 3 for monthly from quarterly).
    rho : float
        AR(1) coefficient for the error term.

    Returns
    -------
    y_high : np.ndarray, shape (T_H,)
        Interpolated high-frequency values.
    """
    T_L = len(y_low)
    T_H = len(z_high)
    assert T_H == m * T_L

    # Aggregation matrix C: T_L x T_H with 1/m in the correct blocks
    C = np.zeros((T_L, T_H))
    for t in range(T_L):
        C[t, t*m:(t+1)*m] = 1.0 / m

    # AR(1) covariance matrix for high-frequency residuals
    def ar1_cov(n, rho):
        idx = np.arange(n)
        return rho ** np.abs(idx[:, None] - idx[None, :]) / (1 - rho**2)

    Sigma_H = ar1_cov(T_H, rho)

    # GLS step: estimate beta (coefficient on z_high)
    Sigma_L = C @ Sigma_H @ C.T
    beta_gls = np.linalg.solve(
        z_high.T @ np.linalg.solve(Sigma_H, z_high),
        z_high.T @ np.linalg.solve(Sigma_H, y_low.repeat(m) / m)
    )

    # Prediction and residual-based correction
    y_hat_low = C @ (z_high * beta_gls)
    residuals_low = y_low - y_hat_low
    correction = Sigma_H @ C.T @ np.linalg.solve(Sigma_L, residuals_low)

    y_high = z_high * beta_gls + correction
    return y_high
```

</div>
</div>

### Limitations of Interpolation

1. **Synthetic data problem:** The interpolated high-frequency series is a model output, not a measurement. Subsequent analysis on interpolated data treats model errors as data.

2. **Over-smoothing:** The constraint $Cy^H = y^L$ forces the interpolated series to aggregate correctly, but the resulting path is determined by the indicator $z$. If $z$ is a poor indicator, the interpolated series is misleading.

3. **Asymmetric information use:** Interpolation adds information by construction — you know the quarterly sum must equal the observed value. This creates a form of look-ahead bias if used in real-time forecasting.

4. **Variance understatement:** Standard errors from models using interpolated data are biased downward because they treat synthetic observations as precise measurements.

---

## Strategy 3: Bridge Equations

### What It Is

A bridge equation connects a high-frequency indicator to a low-frequency target via an intermediate aggregation step, then forecasts the missing within-period observations needed for nowcasting.

### The Two-Step Procedure

**Step 1: Quarterly model**

$$y_t^Q = \alpha + \sum_{k=0}^{K} \beta_k \tilde{x}_{t-k}^Q + \varepsilon_t$$

where $\tilde{x}_t^Q$ is the temporally aggregated monthly indicator.

**Step 2: Monthly forecasting model (for nowcasting)**

When the current quarter is incomplete, forecast the missing monthly observations:

$$x_\tau^M = \mu + \sum_{j=1}^{J} \phi_j x_{\tau-j}^M + \eta_\tau$$

Then aggregate the forecasted monthly values to form $\hat{\tilde{x}}_t^Q$ and plug into Step 1.


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

def bridge_equation_nowcast(y_quarterly, x_monthly, current_quarter_obs):
    """
    Nowcast current-quarter GDP using a bridge equation.

    Parameters
    ----------
    y_quarterly : pd.Series
        Historical quarterly GDP growth (in-sample).
    x_monthly : pd.Series
        Historical monthly IP growth.
    current_quarter_obs : list
        Monthly IP observations available for current quarter.
        Length 0, 1, 2, or 3.

    Returns
    -------
    nowcast : float
        Estimated current-quarter GDP growth.
    """
    # Step 1: Aggregate monthly to quarterly
    x_quarterly = x_monthly.resample("QE").mean()

    # Align indices
    common_idx = y_quarterly.index.intersection(x_quarterly.index)
    y = y_quarterly.loc[common_idx].values
    X = x_quarterly.loc[common_idx].values.reshape(-1, 1)

    # Estimate bridge equation
    bridge = LinearRegression().fit(X, y)

    # Step 2: Complete the current quarter using AR(1) forecast for missing months
    n_available = len(current_quarter_obs)
    n_needed = 3 - n_available

    if n_needed > 0:
        # Forecast missing months using AR(1) on historical monthly data
        ar_model = ARIMA(x_monthly, order=(1, 0, 0)).fit()
        forecast = ar_model.forecast(steps=n_needed)
        full_quarter = list(current_quarter_obs) + list(forecast)
    else:
        full_quarter = list(current_quarter_obs)

    x_current = np.mean(full_quarter)

    # Apply bridge equation
    nowcast = bridge.predict([[x_current]])[0]
    return nowcast, bridge.coef_[0], bridge.intercept_
```

</div>
</div>

### Error Compounding in Bridge Equations

Let $\varepsilon_1$ be the error from Step 1 (quarterly regression) and $\varepsilon_2$ be the error from Step 2 (monthly forecasts). The total nowcast error includes:

$$e_{\text{bridge}} = \varepsilon_1 + \hat{\beta}_1 \cdot \delta_x$$

where $\delta_x$ is the error in the aggregated monthly forecast due to $\varepsilon_2$. Under standard assumptions:

$$\text{Var}(e_{\text{bridge}}) = \sigma_{\varepsilon_1}^2 + \hat{\beta}_1^2 \cdot \frac{\sigma_{\varepsilon_2}^2}{m}$$

The variance grows with the squared coefficient on the monthly indicator and the variance of the monthly forecast error. MIDAS avoids this compounding because estimation is done in a single step.

---

## Comparison: When Traditional Methods Hold Up

Despite their limitations, traditional methods remain useful in specific contexts:

| Situation | Recommended Approach |
|-----------|---------------------|
| Long span of data, stable relationship | Temporal aggregation (simple, interpretable) |
| Need high-frequency version of a low-freq series | Chow-Lin interpolation |
| Small number of indicators, real-time focus | Bridge equation |
| Many indicators, mixed frequencies | MIDAS or DFM-MIDAS |
| Very high frequency ratio ($m > 20$) | MIDAS with Beta polynomial |
| Real-time nowcasting with ragged edge | DFM or Factor-MIDAS |

The empirical dominance of MIDAS is strongest when:
1. The frequency ratio $m$ is large (many high-frequency obs per low-frequency period)
2. The timing of within-period activity matters (recent observations more informative)
3. Sample size is moderate-to-large (NLS estimation requires sufficient data)

---

## The Econometric Cost of Pre-Aggregation

Foroni and Marcellino (2013) show formally that temporal aggregation introduces a "generated regressors" problem. When the true DGP is:

$$y_t^Q = \alpha + \beta(L^{1/m}) x_\tau^M + \varepsilon_t$$

but we estimate:

$$y_t^Q = \alpha + \tilde{\beta} \tilde{x}_t^Q + \tilde{\varepsilon}_t$$

the OLS estimator $\hat{\tilde{\beta}}$ is consistent for $\sum_j w_j \beta_j$ (a weighted average of the true lag coefficients) rather than for each $\beta_j$ individually. The standard errors are also biased because $\tilde{\varepsilon}_t$ is a moving average of the true residuals.

The bias is zero only if the true weight function happens to equal the imposed aggregation weights — an assumption that is never tested and rarely true.

---

## Common Pitfalls

**Pitfall 1: Using interpolated data as if it were observed.** When you interpolate monthly GDP from quarterly observations, the resulting series has autocorrelation and variance that reflect the interpolation model, not the underlying economic process. Running any subsequent model on it gives biased standard errors.

**Pitfall 2: Forgetting that bridge equations are two-stage.** When reporting forecast accuracy, bridge equation errors include the monthly forecasting error from Stage 2. Always evaluate end-to-end accuracy, not just the Stage 1 fit.

**Pitfall 3: Mixing vintages.** Published data is revised. A bridge equation estimated on current-vintage data will look better in sample than it would have looked in real time. Use real-time data vintages for honest evaluation.

---

## Connections

- **Builds on:** Guide 01 (mixed-frequency problem statement)
- **Leads to:** Module 01, Guide 01 (the MIDAS equation as a solution)
- **Related to:** Two-step estimation problems in econometrics, generated regressors

---

## Practice Problems

<div class="callout-danger">

**Danger:** Never use future information when constructing the high-frequency regressor matrix. In a real-time nowcasting context, you only have data up to the current date -- using the full quarter of monthly data when nowcasting mid-quarter is a look-ahead bias that invalidates your results.

</div>


1. Write out the aggregation matrix $C$ for the case $m=3$ (monthly to quarterly), $T_L = 4$ quarters. What are its dimensions? What does each row sum to?

2. A forecaster uses equal-weight aggregation of 3 monthly IP observations to predict quarterly GDP. She finds $\hat{\beta} = 0.4$, $R^2 = 0.35$. A colleague uses last-period (Month 3 only) aggregation and finds $\hat{\beta} = 0.55$, $R^2 = 0.40$. What does the higher R-squared of the last-period model suggest about the timing of information in IP data?

3. Under what conditions does the Chow-Lin interpolation give the same result as simple linear interpolation between quarterly observations? (Hint: think about the role of the indicator $z$.)

---

## Further Reading

- Chow, G., & Lin, A. (1971). "Best linear unbiased interpolation, distribution, and extrapolation of time series by related series." *Review of Economics and Statistics*, 53(4), 372–375.
- Foroni, C., & Marcellino, M. (2013). "A survey of econometric methods for mixed-frequency data." *Norges Bank Working Paper.*
- Ghysels, E., Hill, J., & Motegi, K. (2016). "Testing for Granger causality with mixed frequency data." *Journal of Econometrics*, 192(1), 207–230.


---

## Cross-References

<a class="link-card" href="./01_mixed_frequency_problem_guide.md">
  <div class="link-card-title">01 Mixed Frequency Problem</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_mixed_frequency_problem_slides.md">
  <div class="link-card-title">01 Mixed Frequency Problem — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./03_course_datasets_guide.md">
  <div class="link-card-title">03 Course Datasets</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_course_datasets_slides.md">
  <div class="link-card-title">03 Course Datasets — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

