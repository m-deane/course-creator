"""
Recipe: Expanding-Window OOS Backtest

Copy-paste this recipe for a complete expanding-window out-of-sample evaluation.

Outputs:
  - DataFrame of (t, forecast, actual, error, squared_error)
  - RMSE, MAE, bias, directional accuracy
  - Plot of forecast path vs actuals
  - Optional: AR(1) benchmark comparison

Adapt the MODEL block to use your chosen estimator.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from scipy import stats
from typing import Dict, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Core expanding-window function
# ──────────────────────────────────────────────────────────────────────────────

def expanding_window_backtest(
    X: np.ndarray,
    y: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    min_train: int = 20,
    cv_splits: int = 5,
    model_name: str = "elasticnet",
) -> pd.DataFrame:
    """
    Expanding-window backtest.

    At each t >= min_train:
      - Fit on X[:t], y[:t]  (expanding training set)
      - Predict X[t]
      - Record forecast, actual, error

    Parameters
    ----------
    X : np.ndarray, shape (T, p)
        Feature matrix (pre-built MIDAS lag matrix or any sklearn features).
    y : np.ndarray, shape (T,)
        Target variable.
    dates : pd.DatetimeIndex, optional
        Observation dates for the output DataFrame.
    min_train : int
        Minimum training window size.
    cv_splits : int
        Number of TimeSeriesSplit CV folds for lambda selection.
    model_name : str
        "elasticnet" | "ridge" | "ar1"

    Returns
    -------
    pd.DataFrame with columns: date, t, forecast, actual, error, sq_error, abs_error
    """
    T = len(y)
    results = []

    for t in range(min_train, T):
        X_train = X[:t]
        y_train = y[:t]
        X_test = X[t:t+1]
        y_test = float(y[t])

        if np.isnan(y_test):
            continue

        # Remove NaN rows from training set
        mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
        X_tr = X_train[mask]
        y_tr = y_train[mask]

        if len(y_tr) < 8:
            continue
        if np.isnan(X_test).any():
            continue

        try:
            if model_name == "ar1":
                # AR(1): regress y_t on y_{t-1}
                X_ar = np.column_stack([np.ones(len(y_tr) - 1), y_tr[:-1]])
                b = np.linalg.lstsq(X_ar, y_tr[1:], rcond=None)[0]
                forecast = float(b[0] + b[1] * y_tr[-1])

            else:
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_te_s = scaler.transform(X_test)

                n_cv = min(cv_splits, len(y_tr) // 4)
                cv = TimeSeriesSplit(n_splits=max(2, n_cv))

                if model_name == "elasticnet":
                    from sklearn.linear_model import ElasticNetCV
                    m = ElasticNetCV(
                        l1_ratio=[0.5, 0.7, 1.0], cv=cv,
                        max_iter=5000, random_state=0
                    )
                elif model_name == "ridge":
                    from sklearn.linear_model import RidgeCV
                    m = RidgeCV(cv=cv)
                elif model_name == "lasso":
                    from sklearn.linear_model import LassoCV
                    m = LassoCV(cv=cv, max_iter=5000, random_state=0)
                else:
                    raise ValueError(f"Unknown model_name: '{model_name}'")

                m.fit(X_tr_s, y_tr)
                forecast = float(m.predict(X_te_s)[0])

            error = forecast - y_test
            row = {
                "t": t,
                "forecast": forecast,
                "actual": y_test,
                "error": error,
                "sq_error": error ** 2,
                "abs_error": abs(error),
            }
            if dates is not None and t < len(dates):
                row["date"] = dates[t]
            results.append(row)

        except Exception:
            continue

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame) -> Dict:
    """
    Compute standard forecast evaluation metrics from a backtest DataFrame.
    """
    errors = df["error"].values
    forecasts = df["forecast"].values
    actuals = df["actual"].values
    n = len(df)

    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    bias = float(np.mean(errors))

    # Bias t-test
    se = float(np.std(errors, ddof=1) / np.sqrt(n))
    t_stat = bias / se if se > 0 else 0.0
    bias_pvalue = float(2 * stats.t.sf(abs(t_stat), df=n - 1))

    # Directional accuracy
    if n > 1:
        dir_acc = float(np.mean(
            np.sign(np.diff(forecasts)) == np.sign(np.diff(actuals))
        ))
    else:
        dir_acc = float("nan")

    # Theil's U2 (relative to random walk)
    rw_errors = actuals[1:] - actuals[:-1]
    theil_u2 = rmse / float(np.sqrt(np.mean(rw_errors ** 2))) if len(rw_errors) > 0 else float("nan")

    return {
        "n_forecasts": n,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "bias_pvalue": bias_pvalue,
        "bias_significant": bias_pvalue < 0.05,
        "directional_accuracy": dir_acc,
        "theil_u2": theil_u2,
    }


def print_metrics(metrics: Dict, label: str = "Model") -> None:
    print(f"\n{'='*45}")
    print(f"  {label} — Forecast Evaluation")
    print(f"{'='*45}")
    print(f"  N forecasts      : {metrics['n_forecasts']}")
    print(f"  RMSE             : {metrics['rmse']:.4f}")
    print(f"  MAE              : {metrics['mae']:.4f}")
    print(f"  Bias             : {metrics['bias']:+.4f}")
    print(f"  Bias p-value     : {metrics['bias_pvalue']:.4f} {'*' if metrics['bias_significant'] else ''}")
    print(f"  Dir. accuracy    : {metrics['directional_accuracy']:.1%}")
    print(f"  Theil U2         : {metrics['theil_u2']:.3f}  {'(<1 = better than RW)' if metrics['theil_u2'] < 1 else '(>1 = worse than RW)'}")
    print(f"{'='*45}")


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_backtest(
    df: pd.DataFrame,
    title: str = "Expanding-Window Backtest",
    date_col: str = "date",
) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    x = df[date_col] if date_col in df.columns else df["t"]

    # Panel 1: Forecasts vs actuals
    ax1.plot(x, df["actual"], color="black", linewidth=2, label="Actual")
    ax1.plot(x, df["forecast"], color="steelblue", linewidth=1.5,
             linestyle="--", marker="o", markersize=3, label="Forecast")
    ax1.set_ylabel("Value")
    ax1.set_title(title, fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Panel 2: Forecast errors
    ax2.bar(x, df["error"], color=["firebrick" if e < 0 else "steelblue" for e in df["error"]],
            alpha=0.7, width=60 if date_col in df.columns else 0.8)
    ax2.axhline(df["error"].mean(), color="darkorange", linewidth=1.5,
                linestyle="--", label=f"Mean error: {df['error'].mean():+.3f}")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Forecast error")
    ax2.set_xlabel("Period")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Complete recipe
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Synthetic quarterly data: T=80 quarters
    T = 80
    K = 6
    dates = pd.date_range("2005-01-01", periods=T, freq="QS")

    X = np.random.randn(T, K)
    y = 0.6 * X[:, 0] + 0.3 * X[:, 1] + np.random.randn(T) * 0.5

    # ── ElasticNet backtest ──
    print("Running ElasticNet expanding-window backtest...")
    df_en = expanding_window_backtest(X, y, dates=dates, min_train=20, model_name="elasticnet")
    metrics_en = compute_metrics(df_en)
    print_metrics(metrics_en, "ElasticNet")

    # ── AR(1) benchmark ──
    print("\nRunning AR(1) benchmark backtest...")
    df_ar = expanding_window_backtest(X, y, dates=dates, min_train=20, model_name="ar1")
    metrics_ar = compute_metrics(df_ar)
    print_metrics(metrics_ar, "AR(1) Benchmark")

    # ── Plot ──
    plot_backtest(df_en, title="ElasticNet MIDAS — Expanding-Window Backtest")

    # ── Summary ──
    print(f"\nRMSE ratio (EN / AR1): {metrics_en['rmse'] / metrics_ar['rmse']:.3f}")
    if metrics_en["rmse"] < metrics_ar["rmse"]:
        print("ElasticNet outperforms AR(1) benchmark.")
    else:
        print("AR(1) benchmark outperforms ElasticNet.")
