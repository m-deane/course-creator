"""
Recipe: Add a New Monthly Indicator to an Existing MIDAS Model

Copy-paste this recipe when you want to expand your indicator set
without refactoring the entire pipeline.

Steps covered:
  1. Fetch and cache the new series
  2. Fill the ragged edge to the current period
  3. Build the new lag block and append to the existing feature matrix
  4. Re-estimate with ElasticNetCV (uses CV to select regularisation)
  5. Run DM test to confirm the new indicator improves OOS accuracy

Assumptions:
  - X_old : existing (T, K*N_old) feature matrix
  - y     : target series, shape (T,)
  - new_series : pd.Series with DatetimeIndex (monthly)
  - low_freq_dates : pd.DatetimeIndex of target observation dates
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from scipy import stats


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Load the new indicator
# ──────────────────────────────────────────────────────────────────────────────

def load_new_indicator(
    csv_path: str,
    date_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """Load a monthly indicator from CSV."""
    df = pd.read_csv(csv_path, parse_dates=[date_col])
    series = df.set_index(date_col)[value_col]
    series.index = pd.to_datetime(series.index)
    return series.sort_index().dropna()


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Fill the ragged edge
# ──────────────────────────────────────────────────────────────────────────────

def fill_carry_forward(series: pd.Series, target_end: pd.Timestamp) -> pd.Series:
    """Extend series to target_end using carry-forward fill."""
    full_idx = pd.date_range(series.index[0], target_end, freq="MS")
    return series.reindex(full_idx).ffill()


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Build a new lag block for the indicator
# ──────────────────────────────────────────────────────────────────────────────

def build_lag_block(
    series: pd.Series,
    low_freq_dates: pd.DatetimeIndex,
    n_lags: int,
) -> np.ndarray:
    """
    Build an (T, n_lags) lag block aligned to low_freq_dates.
    lag[:,0] = most recent monthly value, lag[:,1] = one period prior, etc.
    """
    T = len(low_freq_dates)
    block = np.full((T, n_lags), np.nan)
    for t, lf_date in enumerate(low_freq_dates):
        available = series[series.index <= lf_date]
        for k in range(n_lags):
            if len(available) > k:
                block[t, k] = float(available.iloc[-(k + 1)])
    return block


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Append new block and re-estimate
# ──────────────────────────────────────────────────────────────────────────────

def expand_and_refit(
    X_old: np.ndarray,
    X_new_block: np.ndarray,
    y: np.ndarray,
    cv_splits: int = 5,
) -> tuple:
    """
    Concatenate old and new feature blocks, fit ElasticNetCV, return model + scaler.

    Returns
    -------
    model : fitted ElasticNetCV
    scaler : fitted StandardScaler (apply to test data before predicting)
    X_expanded : np.ndarray — the full expanded feature matrix
    """
    X_expanded = np.hstack([X_old, X_new_block])

    # Drop rows with NaN (any series missing)
    mask = ~np.isnan(X_expanded).any(axis=1) & ~np.isnan(y)
    X_clean = X_expanded[mask]
    y_clean = y[mask]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    cv = TimeSeriesSplit(n_splits=cv_splits)
    model = ElasticNetCV(l1_ratio=[0.5, 0.7, 1.0], cv=cv, max_iter=10000, random_state=0)
    model.fit(X_scaled, y_clean)

    print(f"Re-fitted on {X_clean.shape[0]} obs, {X_expanded.shape[1]} features")
    print(f"Lambda: {model.alpha_:.6f}, l1_ratio: {model.l1_ratio_:.2f}")
    n_nonzero = int(np.sum(np.abs(model.coef_) > 1e-6))
    print(f"Non-zero coefficients: {n_nonzero} / {X_expanded.shape[1]}")

    return model, scaler, X_expanded


# ──────────────────────────────────────────────────────────────────────────────
# Step 5: DM test — does the new indicator improve OOS accuracy?
# ──────────────────────────────────────────────────────────────────────────────

def expanding_window_rmse(
    X: np.ndarray,
    y: np.ndarray,
    min_train: int = 20,
    cv_splits: int = 5,
) -> np.ndarray:
    """Expanding-window squared errors for ElasticNetCV on feature matrix X."""
    errors = []
    T = len(y)
    for t in range(min_train, T):
        X_tr, y_tr = X[:t], y[:t]
        mask = ~np.isnan(X_tr).any(axis=1) & ~np.isnan(y_tr)
        X_tr, y_tr = X_tr[mask], y_tr[mask]
        if len(y_tr) < 8 or np.isnan(X[t]).any() or np.isnan(y[t]):
            continue
        sc = StandardScaler()
        X_s = sc.fit_transform(X_tr)
        cv = TimeSeriesSplit(n_splits=min(cv_splits, len(y_tr) // 4))
        m = ElasticNetCV(l1_ratio=[0.5, 1.0], cv=cv, max_iter=5000, random_state=0)
        m.fit(X_s, y_tr)
        pred = float(m.predict(sc.transform(X[t:t+1]))[0])
        errors.append(pred - float(y[t]))
    return np.array(errors)


def dm_test(e_old: np.ndarray, e_new: np.ndarray) -> dict:
    """Diebold-Mariano test: H0 equal MSE. Positive dm_stat = new model is better."""
    d = e_old ** 2 - e_new ** 2
    n = len(d)
    d_bar = np.mean(d)
    gamma0 = np.var(d, ddof=1)
    gamma1 = float(np.cov(d[:-1], d[1:])[0, 1]) if n > 2 else 0.0
    nw_var = max(gamma0 + 2 * gamma1, 1e-12)
    dm_stat = d_bar / np.sqrt(nw_var / n)
    p_value = float(2 * stats.t.sf(abs(dm_stat), df=n - 1))
    return {
        "dm_stat": float(dm_stat),
        "p_value": p_value,
        "new_model_better": dm_stat > 0,
        "significant": p_value < 0.10,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Complete recipe (copy-paste and adapt)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Simulate existing model: T=60 quarters, 12 features (2 indicators × 6 lags)
    T = 60
    X_old = np.random.randn(T, 12)
    y = X_old[:, 0] + 0.3 * X_old[:, 2] + np.random.randn(T) * 0.5

    # Simulate new indicator: monthly, 3 lags, correlated with y
    n_months = T * 3
    dates_monthly = pd.date_range("2000-01-01", periods=n_months, freq="MS")
    dates_quarterly = pd.date_range("2000-01-01", periods=T, freq="QS")
    new_monthly = pd.Series(
        np.repeat(y, 3) + np.random.randn(n_months) * 0.2,
        index=dates_monthly,
        name="NEW_INDICATOR",
    )

    # Fill ragged edge
    current_month = dates_monthly[-1]
    filled = fill_carry_forward(new_monthly, current_month)

    # Build lag block (3 lags, aligned to quarterly dates)
    new_block = build_lag_block(filled, dates_quarterly, n_lags=3)

    # Expand and re-fit
    model, scaler, X_expanded = expand_and_refit(X_old, new_block, y)

    # DM test
    print("\nRunning DM test (expanding window)...")
    e_old = expanding_window_rmse(X_old, y, min_train=20)
    e_new = expanding_window_rmse(X_expanded, y, min_train=20)
    n_common = min(len(e_old), len(e_new))
    result = dm_test(e_old[-n_common:], e_new[-n_common:])
    print(f"DM stat: {result['dm_stat']:+.3f}")
    print(f"p-value: {result['p_value']:.4f}")
    print(f"New model better: {result['new_model_better']}")
    print(f"Statistically significant (10%): {result['significant']}")

    if result["new_model_better"] and result["significant"]:
        print("\nConclusion: ADD the new indicator — significant improvement confirmed.")
    elif result["new_model_better"]:
        print("\nConclusion: Marginal improvement, not significant. Consider adding cautiously.")
    else:
        print("\nConclusion: New indicator does NOT improve OOS accuracy. Do not add.")
