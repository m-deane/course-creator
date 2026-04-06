# Model Monitoring and Reporting for Nowcasting Systems

> **Reading time:** ~20 min | **Module:** 08 — Production Systems | **Prerequisites:** Module 7


## Learning Objectives

<div class="flow">
<div class="flow-step mint">1. Load Ragged Data</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">2. Estimate MIDAS</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">3. Generate Nowcast</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. Update as Data Arrives</div>
</div>


<div class="callout-key">

**Key Concept Summary:** A model that was well-calibrated in 2019 may be systematically biased by 2022. Supply-chain disruptions, policy regime changes, and measurement revisions alter the statistical relationships MIDAS e...

</div>

After reading this guide you will be able to:

1. Implement rolling forecast accuracy monitoring with RMSE, MAE, and bias
2. Detect structural breaks using Chow and CUSUM tests
3. Define and apply re-estimation triggers based on statistical criteria
4. Build a nowcast evolution chart and news decomposition waterfall
5. Design a daily model health report

---

## 1. Why Monitoring Is Non-Negotiable

<div class="callout-insight">

**Insight:** Real-time nowcasting is fundamentally different from pseudo out-of-sample backtesting. The ragged-edge data structure means your model sees different information at different points within a quarter.

</div>


A model that was well-calibrated in 2019 may be systematically biased by 2022. Supply-chain disruptions, policy regime changes, and measurement revisions alter the statistical relationships MIDAS exploits. Without monitoring, you will continue publishing stale forecasts with no indication that the model has drifted.

Monitoring has three goals:

1. **Accuracy tracking**: Is the current RMSE consistent with the backtest period?
2. **Bias detection**: Is the model systematically above or below actuals?
3. **Structural stability**: Have the regression coefficients shifted?

Each goal requires a different diagnostic tool.

---

## 2. Rolling Forecast Accuracy

<div class="callout-warning">

**Warning:** Pseudo out-of-sample exercises that do not properly account for the real-time data vintage will overstate nowcast accuracy. Always use the ragged-edge structure that would have been available at each historical nowcast date.

</div>


### Computing Evaluation Metrics

After each GDP release, compare the terminal nowcast (the last estimate before the advance release) to the advance GDP figure. Accumulate these pairs into a rolling error database.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import numpy as np
import pandas as pd
from typing import Dict, List

def compute_rolling_metrics(
    forecasts: pd.DataFrame,
    actuals: pd.Series,
    window: int = 8,
) -> pd.DataFrame:
    """
    Compute rolling RMSE, MAE, and bias over the last `window` quarters.

    Parameters
    ----------
    forecasts : pd.DataFrame
        Must have columns ['target_period', 'point_forecast'].
        One row per forecast horizon per target period.
    actuals : pd.Series
        Index = target_period strings, values = realised GDP growth.
    window : int
        Rolling window in quarters (8 = 2 years).

    Returns
    -------
    pd.DataFrame
        Columns: target_period, rmse, mae, bias, n_obs
    """
    merged = forecasts.copy()
    merged["actual"] = merged["target_period"].map(actuals)
    merged = merged.dropna(subset=["actual"])
    merged["error"] = merged["point_forecast"] - merged["actual"]

    records = []
    periods = sorted(merged["target_period"].unique())

    for i, period in enumerate(periods):
        start_idx = max(0, i - window + 1)
        window_periods = periods[start_idx : i + 1]
        subset = merged[merged["target_period"].isin(window_periods)]

        rmse = float(np.sqrt((subset["error"] ** 2).mean()))
        mae = float(subset["error"].abs().mean())
        bias = float(subset["error"].mean())
        records.append(
            {
                "target_period": period,
                "rmse": rmse,
                "mae": mae,
                "bias": bias,
                "n_obs": len(subset),
            }
        )

    return pd.DataFrame(records)
```

</div>

### Interpreting Metrics

| Metric | Good signal | Warning signal |
|--------|-------------|----------------|
| RMSE | Stable or declining | Rising trend over 4+ quarters |
| Bias | Near zero | Consistent sign over 6+ quarters |
| MAE | Stable | Spike (single bad forecast vs. drift) |

A single outlier quarter (e.g. COVID-19 2020-Q2) inflates RMSE temporarily but is not evidence of model deterioration. Look for sustained multi-quarter trends, not individual spikes.

---

## 3. Bias Testing

Systematic bias is detected with a simple t-test on the mean forecast error.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
from scipy import stats

def test_forecast_bias(
    errors: np.ndarray,
    alpha: float = 0.05,
) -> Dict:
    """
    Test H0: E[error] = 0 (no bias) using a two-sided t-test.

    Parameters
    ----------
    errors : np.ndarray
        Array of (forecast - actual) values.
    alpha : float
        Significance level.

    Returns
    -------
    dict with keys: mean_error, t_stat, p_value, reject_null
    """
    n = len(errors)
    mean_err = float(np.mean(errors))
    se = float(np.std(errors, ddof=1) / np.sqrt(n))
    t_stat = mean_err / se if se > 0 else 0.0
    p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 1))

    return {
        "mean_error": mean_err,
        "std_error": se,
        "t_stat": t_stat,
        "p_value": p_value,
        "reject_null": p_value < alpha,
        "interpretation": (
            f"Mean forecast error = {mean_err:.3f}. "
            + (
                "BIAS DETECTED: reject unbiasedness."
                if p_value < alpha
                else "No significant bias detected."
            )
        ),
    }
```

</div>

For a production system, run this test monthly. Log the result; alert if `reject_null` is `True` for two consecutive months.

---

## 4. Structural Break Detection

### Chow Test

The Chow test asks whether the regression coefficients are the same in two subsamples. It requires specifying a candidate break date.

$$F = \frac{(RSS_R - RSS_{U1} - RSS_{U2}) / k}{(RSS_{U1} + RSS_{U2}) / (n - 2k)} \sim F(k, n-2k)$$

where $k$ is the number of parameters, $n$ is total observations, and $R$, $U1$, $U2$ denote restricted (pooled) and unrestricted (split) regressions.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def chow_test(
    X: np.ndarray,
    y: np.ndarray,
    break_index: int,
) -> Dict:
    """
    Chow test for structural break at break_index.

    H0: coefficients are the same before and after break_index.

    Parameters
    ----------
    X : np.ndarray, shape (n, k)
        Feature matrix (already includes intercept column if needed).
    y : np.ndarray, shape (n,)
        Target vector.
    break_index : int
        Index of first observation in the second subsample.

    Returns
    -------
    dict with keys: f_stat, p_value, reject_null
    """
    from numpy.linalg import lstsq

    n, k = X.shape

    def ols_rss(Xb, yb):
        beta, _, _, _ = lstsq(Xb, yb, rcond=None)
        resid = yb - Xb @ beta
        return float(resid @ resid)

    rss_r = ols_rss(X, y)
    rss_1 = ols_rss(X[:break_index], y[:break_index])
    rss_2 = ols_rss(X[break_index:], y[break_index:])

    rss_u = rss_1 + rss_2
    f_num = (rss_r - rss_u) / k
    f_den = rss_u / (n - 2 * k)
    f_stat = f_num / f_den if f_den > 0 else float("nan")
    p_value = float(stats.f.sf(f_stat, dfn=k, dfd=n - 2 * k))

    return {
        "break_index": break_index,
        "f_stat": f_stat,
        "p_value": p_value,
        "reject_null": p_value < 0.05,
    }
```

</div>

### CUSUM Test

The CUSUM (Cumulative Sum of Recursive Residuals) test detects parameter instability without requiring a pre-specified break date. It is the preferred test when you do not know *when* a structural break might have occurred.

```python
def cusum_test(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
) -> Dict:
    """
    CUSUM test for parameter stability (Brown-Durbin-Evans, 1975).

    Computes the cumulative sum of recursive residuals and compares to
    the 5% significance band ± c * sqrt(n - k).

    Returns
    -------
    dict with keys: cusum, boundaries, any_crossing, break_index
    """
    n, k = X.shape
    cusum = np.zeros(n - k)
    sigma_sq = []

    # Critical value for 5% level
    c_alpha = {0.01: 1.143, 0.05: 0.948, 0.10: 0.850}.get(alpha, 0.948)

    for t in range(k, n):
        X_t = X[:t]
        y_t = y[:t]
        beta_t = np.linalg.lstsq(X_t, y_t, rcond=None)[0]
        x_new = X[t]
        y_hat = float(x_new @ beta_t)
        resid = y[t] - y_hat
        h_t = float(x_new @ np.linalg.pinv(X_t.T @ X_t) @ x_new)
        sigma_sq.append(resid ** 2 / (1 + h_t))

    sigma_hat = float(np.sqrt(np.mean(sigma_sq)))
    recursive_resids = np.zeros(n - k)

    for t in range(k, n):
        X_t = X[:t]
        y_t = y[:t]
        beta_t = np.linalg.lstsq(X_t, y_t, rcond=None)[0]
        x_new = X[t]
        h_t = float(x_new @ np.linalg.pinv(X_t.T @ X_t) @ x_new)
        recursive_resids[t - k] = (y[t] - float(x_new @ beta_t)) / (
            sigma_hat * np.sqrt(1 + h_t)
        )

    cusum = np.cumsum(recursive_resids)
    m = len(cusum)
    upper = c_alpha * np.sqrt(m) * (1 + 2 * np.arange(m) / m)
    lower = -upper

    crossings = np.where(
        (cusum > upper) | (cusum < lower)
    )[0]
    any_crossing = len(crossings) > 0
    break_index = int(crossings[0]) + k if any_crossing else None

    return {
        "cusum": cusum,
        "upper_boundary": upper,
        "lower_boundary": lower,
        "any_crossing": any_crossing,
        "break_index": break_index,
        "interpretation": (
            "STRUCTURAL BREAK detected." if any_crossing
            else "No structural break detected."
        ),
    }
```

---

## 5. Re-Estimation Triggers

A re-estimation trigger is a rule that says: "when condition X is met, re-fit the model on an updated training window." Over-triggering wastes compute and risks fitting noise; under-triggering means you carry a stale model.

### Three Trigger Types

**Calendar trigger**: Re-estimate once per quarter, regardless of performance. Simple and predictable.

**Performance trigger**: Re-estimate when rolling RMSE exceeds `backtest_rmse * (1 + threshold)` for two consecutive evaluation periods.

**Structural break trigger**: Re-estimate immediately when the CUSUM test crosses its boundary.

```python
class ReEstimationTrigger:
    """
    Monitors forecast performance and emits re-estimation signals.

    Parameters
    ----------
    backtest_rmse : float
        RMSE from the initial backtest period (the baseline).
    rmse_threshold : float
        Relative RMSE increase that triggers re-estimation (e.g. 0.20 = 20%).
    calendar_quarters : int
        Re-estimate at least every N quarters regardless of performance.
    """

    def __init__(
        self,
        backtest_rmse: float,
        rmse_threshold: float = 0.20,
        calendar_quarters: int = 4,
    ):
        self.backtest_rmse = backtest_rmse
        self.rmse_threshold = rmse_threshold
        self.calendar_quarters = calendar_quarters
        self._quarters_since_last = 0
        self._consecutive_exceedances = 0

    def check(
        self,
        current_rmse: float,
        cusum_break: bool,
    ) -> Dict:
        """
        Check whether re-estimation is warranted.

        Returns dict with keys: should_reestimate, reason
        """
        self._quarters_since_last += 1
        reason = []

        # Calendar trigger
        if self._quarters_since_last >= self.calendar_quarters:
            reason.append(
                f"calendar ({self._quarters_since_last} quarters elapsed)"
            )

        # Performance trigger
        threshold = self.backtest_rmse * (1 + self.rmse_threshold)
        if current_rmse > threshold:
            self._consecutive_exceedances += 1
            if self._consecutive_exceedances >= 2:
                reason.append(
                    f"performance (RMSE {current_rmse:.3f} > "
                    f"threshold {threshold:.3f} for 2 periods)"
                )
        else:
            self._consecutive_exceedances = 0

        # Structural break trigger
        if cusum_break:
            reason.append("structural break (CUSUM test)")

        should_reestimate = len(reason) > 0
        if should_reestimate:
            self._quarters_since_last = 0

        return {
            "should_reestimate": should_reestimate,
            "reason": "; ".join(reason) if reason else "no trigger",
        }
```

---

## 6. Nowcast Evolution Chart

The nowcast evolution chart — sometimes called the "horse race" chart — is the standard output of any institutional nowcasting system. It shows how the estimate for a single target quarter evolves as new data arrive.

```python
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_nowcast_evolution(
    forecast_df: pd.DataFrame,
    actual: float,
    target_period: str,
    release_events: List[Dict] = None,
) -> plt.Figure:
    """
    Plot the nowcast evolution for a single target quarter.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Columns: forecast_date (datetime), point_forecast, lower_95, upper_95
    actual : float
        The advance GDP release (final comparison point).
    target_period : str
        Label for the plot title.
    release_events : list of dict
        Optional list of {'date': ..., 'label': ...} for annotation.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    forecast_df = forecast_df.copy()
    forecast_df["forecast_date"] = pd.to_datetime(forecast_df["forecast_date"])
    forecast_df = forecast_df.sort_values("forecast_date")

    # Prediction band
    ax.fill_between(
        forecast_df["forecast_date"],
        forecast_df["lower_95"],
        forecast_df["upper_95"],
        alpha=0.15,
        color="steelblue",
        label="95% prediction interval",
    )

    # Point forecast path
    ax.plot(
        forecast_df["forecast_date"],
        forecast_df["point_forecast"],
        color="steelblue",
        linewidth=2,
        marker="o",
        markersize=4,
        label="Nowcast",
    )

    # Actual (horizontal line)
    ax.axhline(actual, color="firebrick", linewidth=1.5, linestyle="--",
               label=f"Advance GDP: {actual:.2f}%")

    # Release event annotations
    if release_events:
        for event in release_events:
            ax.axvline(
                pd.Timestamp(event["date"]),
                color="gray",
                linewidth=0.8,
                linestyle=":",
                alpha=0.7,
            )
            ax.text(
                pd.Timestamp(event["date"]),
                ax.get_ylim()[1] * 0.95,
                event["label"],
                rotation=90,
                fontsize=7,
                color="gray",
                ha="right",
            )

    ax.set_title(
        f"Nowcast Evolution — {target_period} GDP Growth",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Forecast date")
    ax.set_ylabel("Annualised GDP growth (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=30, ha="right")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig
```

---

## 7. News Decomposition Waterfall

The news decomposition explains *why* the nowcast changed between two consecutive runs. Each data release contributes a positive or negative bar proportional to its coefficient times its surprise.

$$\Delta \hat{y}_{t \to t+1} = \sum_{i=1}^{N} \hat{\beta}_i \cdot (x_i^{\text{new}} - \mathbb{E}[x_i^{\text{new}}])$$

```python
def plot_news_decomposition(
    news: Dict[str, float],
    previous_nowcast: float,
    current_nowcast: float,
    title: str = "News Decomposition",
) -> plt.Figure:
    """
    Waterfall chart showing the contribution of each release to the
    nowcast revision.

    Parameters
    ----------
    news : dict
        Keys = series names, values = contribution to revision (signed).
    previous_nowcast : float
    current_nowcast : float

    Returns
    -------
    plt.Figure
    """
    items = sorted(news.items(), key=lambda x: abs(x[1]), reverse=True)
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    colors = ["steelblue" if v >= 0 else "firebrick" for v in values]

    fig, ax = plt.subplots(figsize=(10, 5))

    running = previous_nowcast
    bar_starts = []
    for v in values:
        bar_starts.append(running if v >= 0 else running + v)
        running += v

    ax.barh(
        range(len(labels)),
        values,
        left=bar_starts,
        color=colors,
        edgecolor="white",
        height=0.6,
    )
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)

    # Start and end markers
    ax.axvline(previous_nowcast, color="black", linewidth=1.5, linestyle="--",
               label=f"Previous: {previous_nowcast:.2f}%")
    ax.axvline(current_nowcast, color="darkgreen", linewidth=1.5, linestyle="-",
               label=f"Current: {current_nowcast:.2f}%")

    ax.set_xlabel("Contribution to nowcast revision (%)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig
```

---

## 8. Daily Health Report

The pipeline generates a one-page health report after every run. The report contains:

1. **Run metadata**: timestamp, data vintage, series used
2. **Current nowcast**: point estimate with 80%/95% interval
3. **Rolling RMSE**: last 8 quarters vs backtest benchmark
4. **Bias test result**: p-value and interpretation
5. **CUSUM status**: pass/fail, last break date if any
6. **Re-estimation trigger**: reason and whether re-estimation ran

```python
def generate_health_report(
    run_metadata: Dict,
    forecast_record,
    rolling_metrics: pd.DataFrame,
    bias_test: Dict,
    cusum_result: Dict,
    reestimation_result: Dict,
) -> str:
    """
    Generate a plain-text health report for logging and email dispatch.
    """
    lines = [
        "=" * 60,
        "NOWCASTING PIPELINE HEALTH REPORT",
        f"Run timestamp : {run_metadata['run_timestamp']}",
        f"Forecast date : {forecast_record.forecast_date}",
        f"Target period : {forecast_record.target_period}",
        "=" * 60,
        "",
        "CURRENT NOWCAST",
        f"  Point estimate : {forecast_record.point_forecast:+.3f}%",
        f"  80% interval   : [{forecast_record.lower_80:+.3f}%, "
        f"{forecast_record.upper_80:+.3f}%]",
        f"  95% interval   : [{forecast_record.lower_95:+.3f}%, "
        f"{forecast_record.upper_95:+.3f}%]",
        f"  Training obs   : {forecast_record.n_train}",
        "",
        "ROLLING ACCURACY (last 8 quarters)",
        f"  RMSE : {rolling_metrics['rmse'].iloc[-1]:.4f}",
        f"  MAE  : {rolling_metrics['mae'].iloc[-1]:.4f}",
        f"  Bias : {rolling_metrics['bias'].iloc[-1]:+.4f}",
        "",
        "BIAS TEST",
        f"  {bias_test['interpretation']}",
        f"  p-value = {bias_test['p_value']:.4f}",
        "",
        "CUSUM STABILITY",
        f"  {cusum_result['interpretation']}",
        (
            f"  Break index: {cusum_result['break_index']}"
            if cusum_result["break_index"] is not None
            else ""
        ),
        "",
        "RE-ESTIMATION",
        f"  Trigger : {reestimation_result['reason']}",
        f"  Action  : {'RE-ESTIMATED' if reestimation_result['should_reestimate'] else 'No action'}",
        "",
        "=" * 60,
    ]
    return "\n".join(line for line in lines if line is not None)
```

---

## 9. Model Comparison Dashboard

When running multiple model variants in parallel (e.g. ElasticNet vs Ridge vs ensemble), the comparison dashboard ranks them by OOS RMSE and flags whether the leading model is statistically superior.

```python
def model_comparison_table(
    model_errors: Dict[str, np.ndarray],
    baseline: str = "elasticnet",
) -> pd.DataFrame:
    """
    Build a model comparison table with DM test p-values against the baseline.

    Parameters
    ----------
    model_errors : dict
        Keys = model names, values = arrays of forecast errors (T,).
    baseline : str
        Name of the baseline model for DM comparisons.

    Returns
    -------
    pd.DataFrame with columns: model, rmse, mae, bias, dm_pvalue, better_than_baseline
    """
    from scipy import stats as sp_stats

    def dm_test(e1: np.ndarray, e2: np.ndarray) -> float:
        """Diebold-Mariano p-value: H0 equal MSE, two-sided."""
        d = e1 ** 2 - e2 ** 2
        n = len(d)
        d_bar = d.mean()
        # Newey-West variance with bandwidth h=1
        gamma0 = np.var(d, ddof=1)
        gamma1 = np.cov(d[:-1], d[1:])[0, 1] if n > 1 else 0.0
        nw_var = gamma0 + 2 * gamma1
        se = np.sqrt(nw_var / n)
        t_stat = d_bar / se if se > 0 else 0.0
        return float(2 * sp_stats.t.sf(abs(t_stat), df=n - 1))

    baseline_errors = model_errors[baseline]
    rows = []
    for name, errors in model_errors.items():
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(np.abs(errors)))
        bias = float(np.mean(errors))
        p_val = dm_test(baseline_errors, errors) if name != baseline else float("nan")
        rows.append(
            {
                "model": name,
                "rmse": rmse,
                "mae": mae,
                "bias": bias,
                "dm_pvalue_vs_baseline": p_val,
                "significantly_better": (
                    (rmse < float(np.sqrt(np.mean(baseline_errors ** 2))))
                    and (p_val < 0.10)
                    if name != baseline
                    else False
                ),
            }
        )
    df = pd.DataFrame(rows).sort_values("rmse")
    return df
```

---

## 10. Alert Rules

Alerts should be actionable — each one maps to a specific investigation or remediation step.

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| Pipeline failure | Exception in any layer | Critical | Check logs, re-run manually |
| Stale forecast | No new forecast in >48h on trading day | High | Check scheduler and data source |
| RMSE spike | Current RMSE > 2× backtest RMSE | High | Inspect recent releases for outliers |
| Sustained bias | Bias t-test rejects H0 two months running | Medium | Check for systematic revision pattern |
| CUSUM break | Test crosses boundary | Medium | Run Chow test at suspected break date |
| Missing indicator | Required series unavailable | Medium | Use carry-forward, flag in report |

Implement alerts as a simple function that writes to a structured log and (in production) sends email or Slack notifications.

```python
def emit_alert(
    level: str,
    message: str,
    metadata: Dict = None,
) -> None:
    """
    Emit a structured alert. In production, route to email/Slack/PagerDuty.
    """
    import json
    import datetime

    record = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "level": level,
        "message": message,
        "metadata": metadata or {},
    }
    # In production: send to notification service
    logger.warning(json.dumps(record))
```

---

## 11. Summary

<div class="callout-danger">

**Danger:** Never use future information when constructing the high-frequency regressor matrix. In a real-time nowcasting context, you only have data up to the current date -- using the full quarter of monthly data when nowcasting mid-quarter is a look-ahead bias that invalidates your results.

</div>


Monitoring a nowcasting system requires three parallel tracks:

**Accuracy track**: Rolling RMSE/MAE/bias computed after each GDP release. Triggers re-estimation if RMSE exceeds the backtest benchmark by more than 20% for two consecutive periods.

**Stability track**: CUSUM test run quarterly. Chow test run at any suspected break date. Triggers immediate re-estimation if structural break detected.

**Operations track**: Daily health report. Alert rules with defined severity levels and remediation actions.

The key outputs — nowcast evolution chart, news decomposition waterfall, rolling RMSE chart — communicate model performance to both technical and non-technical stakeholders. In Notebook 02 you will build an interactive monitoring dashboard using these functions.

---

## References

- Brown, R. L., Durbin, J., & Evans, J. M. (1975). Techniques for testing the constancy of regression relationships over time. *Journal of the Royal Statistical Society, Series B*, 37(2), 149–163.
- Chow, G. C. (1960). Tests of equality between sets of coefficients in two linear regressions. *Econometrica*, 28(3), 591–605.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Banbura, M., & Modugno, M. (2014). Maximum likelihood estimation of factor models on datasets with arbitrary pattern of missing data. *Journal of Applied Econometrics*, 29(1), 133–160.
- Giannone, D., Reichlin, L., & Small, D. (2008). Nowcasting: The real-time informational content of macroeconomic data. *Journal of Monetary Economics*, 55(4), 665–676.


---

## Conceptual Practice Questions

**Practice Question 1:** How does the ragged-edge problem affect the reliability of real-time nowcasts compared to pseudo out-of-sample exercises?

**Practice Question 2:** What is the key difference between direct and iterated multi-step forecasts in a MIDAS context?


---

## Cross-References

<a class="link-card" href="./01_pipeline_architecture_guide.md">
  <div class="link-card-title">01 Pipeline Architecture</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_pipeline_architecture_slides.md">
  <div class="link-card-title">01 Pipeline Architecture — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./03_decision_flowchart_guide.md">
  <div class="link-card-title">03 Decision Flowchart</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_decision_flowchart_slides.md">
  <div class="link-card-title">03 Decision Flowchart — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

