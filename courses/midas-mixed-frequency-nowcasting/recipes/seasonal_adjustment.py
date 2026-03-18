"""
Recipe: Seasonal Adjustment for Mixed-Frequency Data

Many macroeconomic series exhibit strong seasonal patterns that can inflate
MIDAS lag coefficients. This recipe covers:

  1. Detecting seasonality using a Kruskal-Wallis test by month
  2. Simple seasonal adjustment: subtract monthly means
  3. X-11 style moving-average decomposition (STL via statsmodels)
  4. Adjusting high-frequency data before building MIDAS features
  5. Propagating seasonal adjustment into real-time (pseudo-real-time safe)

Note: Seasonal adjustment should always be done on the raw series BEFORE
building the MIDAS feature matrix. Do not seasonally adjust the lag matrix
after construction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# 1. Seasonality detection
# ──────────────────────────────────────────────────────────────────────────────

def kruskal_wallis_seasonality_test(series: pd.Series) -> Dict:
    """
    Test for seasonality using the Kruskal-Wallis test.

    H0: median is the same across all months (no seasonality).
    Groups observations by calendar month and tests for differences.

    Parameters
    ----------
    series : pd.Series (monthly DatetimeIndex)

    Returns
    -------
    dict with: stat, p_value, seasonal, dominant_months
    """
    series = series.dropna()
    series.index = pd.to_datetime(series.index)

    groups = [series[series.index.month == m].values for m in range(1, 13)]
    groups = [g for g in groups if len(g) >= 2]

    if len(groups) < 3:
        return {"stat": np.nan, "p_value": np.nan, "seasonal": False, "dominant_months": []}

    stat, p_value = stats.kruskal(*groups)

    # Identify months that deviate most from the annual mean
    monthly_means = series.groupby(series.index.month).mean()
    overall_mean = float(series.mean())
    deviations = (monthly_means - overall_mean).abs()
    dominant_months = deviations.nlargest(3).index.tolist()

    return {
        "stat": float(stat),
        "p_value": float(p_value),
        "seasonal": p_value < 0.05,
        "dominant_months": dominant_months,
        "monthly_means": monthly_means.to_dict(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. Simple seasonal adjustment: subtract monthly means
# ──────────────────────────────────────────────────────────────────────────────

class MonthlyMeanAdjuster:
    """
    Seasonal adjustment by subtracting estimated monthly means.

    The monthly means are estimated on the training window only.
    When called in production (pseudo-real-time), pass only data
    available as of the forecast date.

    Usage
    -----
        adj = MonthlyMeanAdjuster()
        adj.fit(series_train)
        adjusted_train = adj.transform(series_train)
        adjusted_forecast = adj.transform(series_forecast)
        recovered = adj.inverse_transform(adjusted_forecast)
    """

    def __init__(self):
        self._monthly_means: Optional[pd.Series] = None

    def fit(self, series: pd.Series) -> "MonthlyMeanAdjuster":
        """Estimate monthly seasonal factors from the training series."""
        series = series.dropna()
        series.index = pd.to_datetime(series.index)
        self._monthly_means = series.groupby(series.index.month).mean()
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        """Subtract monthly means. Returns seasonally adjusted series."""
        if self._monthly_means is None:
            raise RuntimeError("Call fit() before transform()")
        series = series.copy()
        series.index = pd.to_datetime(series.index)
        adjusted = series.copy()
        for month, mean in self._monthly_means.items():
            mask = series.index.month == month
            adjusted[mask] = series[mask] - mean
        adjusted.name = f"{series.name}_SA" if series.name else "SA"
        return adjusted

    def inverse_transform(self, adjusted: pd.Series) -> pd.Series:
        """Add back monthly means to recover the original scale."""
        if self._monthly_means is None:
            raise RuntimeError("Call fit() before inverse_transform()")
        adjusted = adjusted.copy()
        adjusted.index = pd.to_datetime(adjusted.index)
        original = adjusted.copy()
        for month, mean in self._monthly_means.items():
            mask = adjusted.index.month == month
            original[mask] = adjusted[mask] + mean
        return original

    def get_seasonal_factors(self) -> pd.Series:
        """Return the estimated monthly seasonal factors (mean by month)."""
        if self._monthly_means is None:
            raise RuntimeError("Model not fitted")
        return self._monthly_means.copy()


# ──────────────────────────────────────────────────────────────────────────────
# 3. STL decomposition (requires statsmodels)
# ──────────────────────────────────────────────────────────────────────────────

def stl_seasonal_adjustment(
    series: pd.Series,
    period: int = 12,
    seasonal: int = 13,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    STL (Seasonal-Trend decomposition using Loess) seasonal adjustment.

    Parameters
    ----------
    series : pd.Series (monthly)
    period : int  (12 for monthly seasonality)
    seasonal : int  (STL seasonal smoother window, must be odd)

    Returns
    -------
    adjusted : pd.Series  (series - seasonal component)
    seasonal : pd.Series  (seasonal component)
    trend    : pd.Series  (trend component)

    Requires statsmodels >= 0.12.
    """
    try:
        from statsmodels.tsa.seasonal import STL
    except ImportError:
        raise ImportError("statsmodels is required for STL. pip install statsmodels")

    series = series.dropna()
    stl = STL(series, period=period, seasonal=seasonal, robust=True)
    result = stl.fit()

    adjusted = series - result.seasonal
    adjusted.name = f"{series.name}_SA" if series.name else "SA"
    seasonal_component = pd.Series(result.seasonal, index=series.index, name="seasonal")
    trend_component = pd.Series(result.trend, index=series.index, name="trend")

    return adjusted, seasonal_component, trend_component


# ──────────────────────────────────────────────────────────────────────────────
# 4. Pseudo-real-time safe seasonal adjustment
# ──────────────────────────────────────────────────────────────────────────────

def pseudo_realtime_seasonal_adjustment(
    series: pd.Series,
    as_of_date: pd.Timestamp,
    train_end: pd.Timestamp,
    method: str = "monthly_mean",
) -> pd.Series:
    """
    Seasonally adjust a series in a pseudo-real-time manner.

    The seasonal factors are estimated using only data available
    through `train_end` (no look-ahead bias).

    Parameters
    ----------
    series : pd.Series (full series including future periods)
    as_of_date : pd.Timestamp (data cut-off for production use)
    train_end : pd.Timestamp (end of training period for factor estimation)
    method : "monthly_mean" | "stl"

    Returns
    -------
    pd.Series: seasonally adjusted series up to as_of_date
    """
    available = series[series.index <= as_of_date].dropna()
    train_sample = available[available.index <= train_end]

    if method == "monthly_mean":
        adjuster = MonthlyMeanAdjuster()
        adjuster.fit(train_sample)
        return adjuster.transform(available)
    elif method == "stl":
        adjusted_train, _, _ = stl_seasonal_adjustment(train_sample)
        # Extend factors by fitting monthly means on the STL-adjusted training residuals
        # Then apply them to the rest of the available sample
        factors = MonthlyMeanAdjuster()
        factors.fit(train_sample)
        return factors.transform(available)
    else:
        raise ValueError(f"Unknown method: '{method}'")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_seasonal_decomposition(
    original: pd.Series,
    adjusted: pd.Series,
    seasonal_factors: pd.Series,
    title: str = "Seasonal Adjustment",
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(original.index, original.values, color="steelblue", linewidth=1.5,
                 label="Original")
    axes[0].plot(adjusted.index, adjusted.values, color="darkorange", linewidth=1.5,
                 linestyle="--", label="Seasonally adjusted")
    axes[0].set_ylabel("Value")
    axes[0].set_title(title, fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(seasonal_factors.index, seasonal_factors.values,
                color="gray", alpha=0.7, width=20)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Seasonal factor")
    axes[1].set_title("Seasonal component", fontsize=10)
    axes[1].grid(axis="y", alpha=0.3)

    residual = adjusted - adjusted.rolling(12, center=True).mean()
    axes[2].plot(residual.index, residual.values, color="steelblue",
                 linewidth=1, alpha=0.8)
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_ylabel("Residual")
    axes[2].set_title("Residual (adjusted - 12-month MA)", fontsize=10)
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Complete recipe
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Synthetic monthly series with strong seasonal pattern
    dates = pd.date_range("2010-01-01", "2023-12-01", freq="MS")
    n = len(dates)
    t = np.arange(n)

    # Seasonal factors (by month)
    seasonal = np.array([
        -0.8, -0.6, 0.1, 0.5, 0.7, 0.8,
         0.6,  0.4, 0.2, 0.0,-0.4,-0.5
    ])
    s_component = np.array([seasonal[d.month - 1] for d in dates])

    # Trend + cycle + noise
    trend = 0.02 * t
    cycle = 1.5 * np.sin(2 * np.pi * t / 48)
    noise = np.random.randn(n) * 0.3

    series = pd.Series(trend + cycle + s_component + noise, index=dates, name="SERIES")

    # Test for seasonality
    result = kruskal_wallis_seasonality_test(series)
    print(f"Kruskal-Wallis test for seasonality:")
    print(f"  Statistic: {result['stat']:.2f}")
    print(f"  p-value:   {result['p_value']:.6f}")
    print(f"  Seasonal:  {result['seasonal']}")
    print(f"  Dominant months: {result['dominant_months']}")
    print()

    # Apply monthly mean seasonal adjustment
    train_end = pd.Timestamp("2020-12-01")
    adjuster = MonthlyMeanAdjuster()
    adjuster.fit(series[series.index <= train_end])

    adjusted = adjuster.transform(series)
    factors = adjuster.get_seasonal_factors()

    # Evaluate: did we remove the seasonal pattern?
    result_after = kruskal_wallis_seasonality_test(adjusted)
    print(f"After seasonal adjustment:")
    print(f"  p-value:  {result_after['p_value']:.6f}")
    print(f"  Seasonal: {result_after['seasonal']}")
    print()

    # Build seasonal factor series for plotting
    seasonal_factor_series = pd.Series(
        [factors[d.month] for d in dates],
        index=dates,
        name="seasonal_factor",
    )

    plot_seasonal_decomposition(series, adjusted, seasonal_factor_series)

    # Pseudo-real-time example
    as_of = pd.Timestamp("2023-10-15")
    sa_prt = pseudo_realtime_seasonal_adjustment(series, as_of, train_end)
    print(f"Pseudo-real-time SA: {len(sa_prt)} observations through {sa_prt.index[-1].date()}")
