"""
Recipe: Handle Ragged-Edge Data in Production

The ragged edge arises at any forecast date because different indicators
have different publication lags. This recipe demonstrates:

  1. Simulating a realistic ragged-edge data environment
  2. Three fill strategies: carry-forward, zero, AR1
  3. Per-indicator lag shift (avoid putting a filled value in lag 1)
  4. Measuring how fill strategy affects the feature matrix quality
  5. Choosing the right strategy based on series properties

Run as a script to see the comparison output.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Publication lag simulation
# ──────────────────────────────────────────────────────────────────────────────

PUB_LAGS_DAYS: Dict[str, int] = {
    "ISM_PMI":  1,
    "PAYEMS":   4,
    "CLAIMS":   5,
    "RETAILSL": 14,
    "INDPRO":   16,
    "CPI":      16,
}


def get_available_observations(
    series: pd.Series,
    as_of_date: pd.Timestamp,
    pub_lag_days: int,
) -> pd.Series:
    """
    Return only the observations available as of `as_of_date`,
    given that each monthly observation is published `pub_lag_days`
    after the end of its reference month.

    Example: PAYEMS for September is published on October 4 (4 days after Sep 30).
    If as_of_date = 2023-10-03, September PAYEMS is not yet available.
    """
    available = []
    for obs_date, val in series.items():
        month_end = obs_date + pd.offsets.MonthEnd(0)
        pub_date = month_end + pd.DateOffset(days=pub_lag_days)
        if pub_date <= as_of_date:
            available.append((obs_date, val))
    if not available:
        return pd.Series(dtype=float, name=series.name)
    dates, vals = zip(*available)
    return pd.Series(vals, index=pd.DatetimeIndex(dates), name=series.name)


def show_ragged_edge(
    series_dict: Dict[str, pd.Series],
    pub_lags: Dict[str, int],
    as_of_date: pd.Timestamp,
    n_months: int = 6,
) -> pd.DataFrame:
    """
    Display the ragged-edge table: which months are available for each series.
    Shows last n_months.
    """
    rows = {}
    target_months = pd.date_range(
        as_of_date - pd.DateOffset(months=n_months - 1),
        as_of_date,
        freq="MS",
    )

    for name, series in series_dict.items():
        lag = pub_lags.get(name, 0)
        avail = get_available_observations(series, as_of_date, lag)
        row = {}
        for m in target_months:
            if m in avail.index:
                row[m.strftime("%b %Y")] = f"{avail[m]:.2f}"
            else:
                row[m.strftime("%b %Y")] = "? (missing)"
        rows[name] = row

    return pd.DataFrame(rows).T


# ──────────────────────────────────────────────────────────────────────────────
# Fill strategies
# ──────────────────────────────────────────────────────────────────────────────

def fill_carry_forward(series: pd.Series, target_end: pd.Timestamp) -> pd.Series:
    """Extend to target_end with carry-forward fill."""
    full_idx = pd.date_range(series.index[0], target_end, freq="MS")
    return series.reindex(full_idx).ffill()


def fill_zero(series: pd.Series, target_end: pd.Timestamp) -> pd.Series:
    """Extend to target_end with zero fill."""
    full_idx = pd.date_range(series.index[0], target_end, freq="MS")
    return series.reindex(full_idx).fillna(0.0)


def fill_ar1(series: pd.Series, target_end: pd.Timestamp, n_obs: int = 24) -> pd.Series:
    """
    Extend to target_end by OLS AR(1) extrapolation.

    Uses the last `n_obs` observations to fit y_t = a + b*y_{t-1}.
    """
    full_idx = pd.date_range(series.index[0], target_end, freq="MS")
    extended = series.reindex(full_idx)
    missing = extended.isna()

    if not missing.any():
        return extended

    observed = extended.dropna()
    y = observed.values[-n_obs:]

    if len(y) < 4:
        return extended.ffill()

    X_ar = np.column_stack([np.ones(len(y) - 1), y[:-1]])
    b = np.linalg.lstsq(X_ar, y[1:], rcond=None)[0]

    last_val = float(y[-1])
    n_missing = int(missing.sum())
    projections = []
    for _ in range(n_missing):
        next_val = float(b[0] + b[1] * last_val)
        projections.append(next_val)
        last_val = next_val

    extended[missing] = projections
    return extended


def select_fill_method(series: pd.Series) -> str:
    """
    Heuristic: choose fill method based on series persistence.

    AR1 coefficient > 0.85: carry_forward
    AR1 coefficient 0.5 to 0.85: ar1
    AR1 coefficient < 0.5 (low persistence growth rate): zero
    """
    if len(series) < 4:
        return "carry_forward"
    y = series.dropna().values
    if len(y) < 4:
        return "carry_forward"
    X_ar = np.column_stack([np.ones(len(y) - 1), y[:-1]])
    b = np.linalg.lstsq(X_ar, y[1:], rcond=None)[0]
    rho = float(b[1])
    if rho > 0.85:
        return "carry_forward"
    elif rho > 0.50:
        return "ar1"
    else:
        return "zero"


# ──────────────────────────────────────────────────────────────────────────────
# Lag-shift option: avoid filled value in lag 1
# ──────────────────────────────────────────────────────────────────────────────

def build_lag_block_with_shift(
    series: pd.Series,
    low_freq_dates: pd.DatetimeIndex,
    n_lags: int,
    shift_if_filled: bool = True,
    fill_flag: Optional[pd.Series] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build lag block. If shift_if_filled is True and fill_flag indicates that
    the most recent observation is filled (not observed), shift lags by 1
    (use lags 2..n_lags+1 instead of 1..n_lags).

    Parameters
    ----------
    series : pd.Series (monthly)
    low_freq_dates : pd.DatetimeIndex
    n_lags : int
    shift_if_filled : bool
    fill_flag : pd.Series (bool), optional
        True where the series value was filled (not observed).
        If None, no shifting is applied.

    Returns
    -------
    block : np.ndarray, shape (T, n_lags)
    lag_names : List[str]
    """
    T = len(low_freq_dates)
    block = np.full((T, n_lags), np.nan)
    name = series.name or "X"
    lag_names = []

    for t, lf_date in enumerate(low_freq_dates):
        available = series[series.index <= lf_date]

        # Determine whether to shift
        shift = 0
        if shift_if_filled and fill_flag is not None:
            most_recent_idx = available.index[-1] if len(available) > 0 else None
            if most_recent_idx is not None and fill_flag.get(most_recent_idx, False):
                shift = 1

        for k in range(n_lags):
            actual_lag = k + shift
            if len(available) > actual_lag:
                block[t, k] = float(available.iloc[-(actual_lag + 1)])

    for k in range(n_lags):
        lag_names.append(f"{name}_lag{k+1}")

    return block, lag_names


# ──────────────────────────────────────────────────────────────────────────────
# Comparison: fill methods vs oracle (known true values)
# ──────────────────────────────────────────────────────────────────────────────

def compare_fill_methods(
    true_series: pd.Series,
    as_of_date: pd.Timestamp,
    pub_lag_days: int,
) -> pd.DataFrame:
    """
    Compare the three fill methods against the oracle (full series).

    Shows: last observed, carry-forward fill, zero fill, AR1 projection,
    and oracle values for each missing period.
    """
    available = get_available_observations(true_series, as_of_date, pub_lag_days)
    target_end = as_of_date.to_period("M").to_timestamp()

    cf = fill_carry_forward(available, target_end)
    zf = fill_zero(available, target_end)
    ar = fill_ar1(available, target_end)

    # Identify filled periods (missing in available but present in oracle)
    full_idx = pd.date_range(available.index[0], target_end, freq="MS")
    is_filled = ~full_idx.isin(available.index)

    df = pd.DataFrame({
        "oracle": true_series.reindex(full_idx),
        "carry_forward": cf,
        "zero_fill": zf,
        "ar1_projection": ar,
        "is_filled": is_filled,
    })

    # Show only the most recent 3 months
    return df.iloc[-3:]


# ──────────────────────────────────────────────────────────────────────────────
# Complete recipe
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Generate synthetic monthly indicator data (2020-2023)
    dates = pd.date_range("2020-01-01", "2023-12-01", freq="MS")
    n = len(dates)

    # AR1 process with rho=0.75 (moderate persistence)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.75 * y[t - 1] + np.random.randn() * 0.5

    true_series = pd.Series(y, index=dates, name="INDPRO")

    # Forecast date: October 15, 2023
    as_of_date = pd.Timestamp("2023-10-15")
    pub_lag_days = 16  # INDPRO: released 16 days after month-end

    print("=== Ragged-Edge Handling Demo ===")
    print(f"As-of date: {as_of_date.date()}")
    print(f"Series: INDPRO (pub lag = {pub_lag_days} days)")
    print()

    # Show available data
    available = get_available_observations(true_series, as_of_date, pub_lag_days)
    print(f"Last available observation: {available.index[-1].date()} "
          f"(value = {available.iloc[-1]:.3f})")
    print()

    # Fill method comparison
    print("Fill method comparison (last 3 months):")
    comparison = compare_fill_methods(true_series, as_of_date, pub_lag_days)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(comparison.to_string())
    print()

    # Heuristic recommendation
    rec = select_fill_method(available)
    print(f"Recommended fill method (heuristic): {rec}")
    print()

    # Show ragged edge table for multiple series
    series_dict = {}
    for name in PUB_LAGS_DAYS:
        vals = np.cumsum(np.random.randn(n) * 0.3)
        series_dict[name] = pd.Series(vals, index=dates, name=name)

    print("Ragged-edge table (as of 2023-10-15):")
    ragged_table = show_ragged_edge(series_dict, PUB_LAGS_DAYS, as_of_date, n_months=4)
    print(ragged_table.to_string())
    print()
    print("Series with '? (missing)' need to be filled before entering the MIDAS matrix.")
