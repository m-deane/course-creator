"""
Recipe: Combine Multiple MIDAS Forecasts

Four combination methods covered:
  1. Equal weights (simplest baseline)
  2. Bates-Granger optimal weights (inverse variance weighting)
  3. OLS regression weights (regress actual on forecasts)
  4. Trimmed mean (drop highest and lowest, average the rest)

The recipe includes:
  - A function to evaluate combination vs individual models
  - DM test comparing each combination method to the best individual model
  - Rolling combination weights over time (time-varying performance)

Research finding: simple equal-weight combination outperforms most
sophisticated weighting schemes in short samples because it avoids
overfitting the weight estimation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Combination methods
# ──────────────────────────────────────────────────────────────────────────────

def equal_weight_combination(forecasts: np.ndarray) -> np.ndarray:
    """
    Simple average of all model forecasts.

    Parameters
    ----------
    forecasts : np.ndarray, shape (T, N_models)

    Returns
    -------
    np.ndarray, shape (T,)
    """
    return np.nanmean(forecasts, axis=1)


def bates_granger_weights(
    errors: np.ndarray,
    window: Optional[int] = None,
) -> np.ndarray:
    """
    Bates-Granger (1969) inverse-variance weights.

    w_i = (1 / MSE_i) / sum_j (1 / MSE_j)

    Parameters
    ----------
    errors : np.ndarray, shape (T, N_models)
        Historical forecast errors (forecast - actual).
    window : int, optional
        If provided, compute MSE over the last `window` periods.
        If None, use the full available history.

    Returns
    -------
    np.ndarray, shape (N_models,)  — sums to 1.
    """
    if window is not None:
        errors = errors[-window:]
    mse = np.nanmean(errors ** 2, axis=0)
    mse = np.maximum(mse, 1e-10)  # avoid division by zero
    inv_mse = 1.0 / mse
    return inv_mse / inv_mse.sum()


def ols_combination_weights(
    actuals: np.ndarray,
    forecasts: np.ndarray,
    constrain_sum_to_one: bool = False,
) -> np.ndarray:
    """
    OLS regression weights: regress actual on forecasts without intercept.

    w = argmin ||y - F w||^2

    Parameters
    ----------
    actuals : np.ndarray, shape (T,)
    forecasts : np.ndarray, shape (T, N_models)
    constrain_sum_to_one : bool
        If True, use constrained optimisation so weights sum to 1.

    Returns
    -------
    np.ndarray, shape (N_models,)
    """
    if constrain_sum_to_one:
        # Constrained OLS: w ≥ 0, sum(w) = 1
        N = forecasts.shape[1]

        def objective(w):
            return float(np.sum((actuals - forecasts @ w) ** 2))

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0)] * N
        w0 = np.ones(N) / N

        res = optimize.minimize(
            objective, w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12}
        )
        return res.x if res.success else w0
    else:
        # Unconstrained OLS
        w, _, _, _ = np.linalg.lstsq(forecasts, actuals, rcond=None)
        return w


def trimmed_mean_combination(
    forecasts: np.ndarray,
    trim_fraction: float = 0.20,
) -> np.ndarray:
    """
    Trimmed mean: drop the top and bottom `trim_fraction` of forecasts at each t.

    Parameters
    ----------
    forecasts : np.ndarray, shape (T, N_models)
    trim_fraction : float
        Fraction to trim from each tail (0.20 = trim lowest 20% and highest 20%).

    Returns
    -------
    np.ndarray, shape (T,)
    """
    T, N = forecasts.shape
    k = max(1, int(np.floor(N * trim_fraction)))
    result = np.zeros(T)
    for t in range(T):
        row = forecasts[t][~np.isnan(forecasts[t])]
        if len(row) > 2 * k:
            row = np.sort(row)[k:-k]
        result[t] = float(np.mean(row))
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Time-varying combination (rolling Bates-Granger)
# ──────────────────────────────────────────────────────────────────────────────

def rolling_bg_combination(
    forecasts: np.ndarray,
    actuals: np.ndarray,
    window: int = 8,
    min_history: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rolling Bates-Granger combination with time-varying weights.

    At each t, weights are computed using errors from the last `window` periods.

    Returns
    -------
    combined : np.ndarray, shape (T,)
    weights_history : np.ndarray, shape (T, N_models)
    """
    T, N = forecasts.shape
    combined = np.full(T, np.nan)
    weights_history = np.full((T, N), np.nan)

    for t in range(T):
        if t < min_history:
            # Not enough history: use equal weights
            w = np.ones(N) / N
        else:
            start = max(0, t - window)
            past_errors = forecasts[start:t] - actuals[start:t, np.newaxis]
            w = bates_granger_weights(past_errors)

        weights_history[t] = w
        combined[t] = float(np.nansum(forecasts[t] * w))

    return combined, weights_history


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation and DM test
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_all(
    actuals: np.ndarray,
    forecasts_dict: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Compute RMSE, MAE, bias for all models and combinations.

    Parameters
    ----------
    actuals : np.ndarray, shape (T,)
    forecasts_dict : dict mapping model_name -> forecasts (shape T,)

    Returns pd.DataFrame sorted by RMSE.
    """
    rows = []
    for name, f in forecasts_dict.items():
        e = f - actuals
        mask = ~np.isnan(e)
        e = e[mask]
        rows.append({
            "model": name,
            "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mae": float(np.mean(np.abs(e))),
            "bias": float(np.mean(e)),
            "n": int(mask.sum()),
        })
    return pd.DataFrame(rows).sort_values("rmse")


def dm_test_combinations(
    actuals: np.ndarray,
    forecasts_dict: Dict[str, np.ndarray],
    baseline: str,
) -> pd.DataFrame:
    """
    DM test: all models vs a chosen baseline.

    Returns pd.DataFrame with dm_stat, p_value, better_than_baseline.
    """
    base_errors = forecasts_dict[baseline] - actuals
    rows = []
    for name, f in forecasts_dict.items():
        errors = f - actuals
        mask = ~np.isnan(errors) & ~np.isnan(base_errors)
        e_b = base_errors[mask]
        e_c = errors[mask]
        n = len(e_b)

        if name == baseline or n < 4:
            rows.append({"model": name, "dm_stat": np.nan, "p_value": np.nan,
                         "better": False})
            continue

        d = e_b ** 2 - e_c ** 2
        d_bar = np.mean(d)
        gamma0 = np.var(d, ddof=1)
        gamma1 = float(np.cov(d[:-1], d[1:])[0, 1]) if n > 2 else 0.0
        nw_var = max(gamma0 + 2 * gamma1, 1e-12)
        dm = d_bar / np.sqrt(nw_var / n)
        p = float(2 * stats.t.sf(abs(dm), df=n - 1))
        rows.append({
            "model": name,
            "dm_stat": float(dm),
            "p_value": p,
            "better": bool(dm > 0 and p < 0.10),
        })
    return pd.DataFrame(rows).sort_values("dm_stat", ascending=False, na_position="last")


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_combinations(
    actuals: np.ndarray,
    combinations: Dict[str, np.ndarray],
    title: str = "Forecast Combination Comparison",
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    t = np.arange(len(actuals))

    colors = ["steelblue", "darkorange", "green", "purple", "brown"]
    axes[0].plot(t, actuals, "k-", linewidth=2, label="Actual")
    for (name, f), color in zip(combinations.items(), colors):
        axes[0].plot(t, f, linestyle="--", linewidth=1.5,
                     color=color, label=name, alpha=0.8)
    axes[0].set_ylabel("Value")
    axes[0].set_title(title, fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(axis="y", alpha=0.3)

    # RMSE comparison bar chart
    names = list(combinations.keys())
    rmses = [float(np.sqrt(np.nanmean((combinations[n] - actuals) ** 2))) for n in names]
    bar_colors = colors[:len(names)]
    axes[1].barh(names, rmses, color=bar_colors, alpha=0.8, edgecolor="white")
    for i, v in enumerate(rmses):
        axes[1].text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)
    axes[1].set_xlabel("RMSE")
    axes[1].set_title("RMSE by combination method", fontsize=10, fontweight="bold")
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Complete recipe
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Simulate 5 individual model forecasts
    T = 60
    actuals = np.sin(np.linspace(0, 4 * np.pi, T)) + np.random.randn(T) * 0.3

    np.random.seed(1)
    model_forecasts = np.column_stack([
        actuals + np.random.randn(T) * 0.5 + 0.1,   # slightly biased
        actuals + np.random.randn(T) * 0.6 - 0.1,   # opposite bias
        actuals + np.random.randn(T) * 0.4,          # most accurate
        actuals + np.random.randn(T) * 0.7 + 0.2,
        actuals + np.random.randn(T) * 0.8 - 0.15,
    ])
    model_names = ["Model A", "Model B", "Model C", "Model D", "Model E"]

    # Split: use first 30 obs to estimate weights, evaluate on last 30
    T_eval = 30
    F_train = model_forecasts[:T_eval]
    y_train = actuals[:T_eval]
    F_eval = model_forecasts[T_eval:]
    y_eval = actuals[T_eval:]

    # 1. Equal weights
    eq = equal_weight_combination(F_eval)

    # 2. Bates-Granger weights (from training errors)
    train_errors = F_train - y_train[:, np.newaxis]
    bg_w = bates_granger_weights(train_errors, window=None)
    bg = F_eval @ bg_w

    # 3. OLS weights (constrained)
    ols_w = ols_combination_weights(y_train, F_train, constrain_sum_to_one=True)
    ols_comb = F_eval @ ols_w

    # 4. Trimmed mean (20%)
    trimmed = trimmed_mean_combination(F_eval, trim_fraction=0.20)

    # 5. Best individual model
    best_idx = int(np.argmin(np.sqrt(np.mean((F_train - y_train[:, np.newaxis]) ** 2, axis=0))))
    best_individual = F_eval[:, best_idx]

    all_forecasts = {
        "Equal weight": eq,
        "Bates-Granger": bg,
        "OLS (constrained)": ols_comb,
        "Trimmed mean": trimmed,
        f"Best individual ({model_names[best_idx]})": best_individual,
    }

    # Evaluation
    summary = evaluate_all(y_eval, all_forecasts)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("Evaluation Summary (OOS period)")
    print(summary.to_string(index=False))

    print("\nDM Tests vs Equal Weight:")
    dm_results = dm_test_combinations(y_eval, all_forecasts, baseline="Equal weight")
    print(dm_results.to_string(index=False))

    # Bates-Granger weights breakdown
    print(f"\nBates-Granger weights: {dict(zip(model_names, bg_w.round(3)))}")
    print(f"OLS weights:           {dict(zip(model_names, ols_w.round(3)))}")

    plot_combinations(y_eval, all_forecasts)
