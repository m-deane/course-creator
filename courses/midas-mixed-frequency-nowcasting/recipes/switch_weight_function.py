"""
Recipe: Switch Between Beta and Almon Lag Weight Functions

Use this recipe when you want to compare Beta polynomial versus Almon
polynomial weighting for a single-indicator MIDAS model.

The Beta polynomial (2 parameters) is more flexible for hump-shaped
and monotone-decreasing profiles. Almon polynomial (degree d gives d+1
parameters) can represent polynomial decay patterns.

This recipe:
  1. Fits Beta weights by NLS
  2. Fits Almon weights by NLS
  3. Fits unrestricted OLS (no restriction on lag profile)
  4. Compares in-sample SSE and OOS RMSE
  5. Plots the three lag weight profiles
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, stats
from typing import Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Weight functions
# ──────────────────────────────────────────────────────────────────────────────

def beta_weights(n_lags: int, theta1: float, theta2: float) -> np.ndarray:
    """Normalised Beta polynomial weights, shape (n_lags,)."""
    j = np.linspace(0.01, 0.99, n_lags)
    w = (j ** (theta1 - 1)) * ((1 - j) ** (theta2 - 1))
    return w / (w.sum() + 1e-12)


def almon_weights(n_lags: int, gamma: np.ndarray) -> np.ndarray:
    """
    Almon polynomial weights.
    gamma : polynomial coefficients of degree len(gamma)-1.
    w(j) = exp(sum_p gamma_p * j^p) / normalisation
    """
    j = np.arange(1, n_lags + 1, dtype=float)
    exponent = sum(float(gamma[p]) * (j ** p) for p in range(len(gamma)))
    exponent -= exponent.max()  # numerical stability
    w = np.exp(exponent)
    return w / (w.sum() + 1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# NLS objective factory
# ──────────────────────────────────────────────────────────────────────────────

def make_sse_objective(
    X: np.ndarray,  # (T, K)
    y: np.ndarray,  # (T,)
    weight_fn,
) -> callable:
    """
    Return a function params -> SSE where:
      params[0] = alpha (intercept)
      params[1] = beta (slope)
      params[2:] = weight function parameters
    """
    def sse(params: np.ndarray) -> float:
        alpha = params[0]
        beta = params[1]
        theta = params[2:]
        try:
            w = weight_fn(theta)
            if np.any(np.isnan(w)) or w.sum() < 1e-8:
                return 1e12
            y_hat = alpha + beta * (X @ w)
            return float(np.sum((y - y_hat) ** 2))
        except Exception:
            return 1e12
    return sse


# ──────────────────────────────────────────────────────────────────────────────
# Fit functions
# ──────────────────────────────────────────────────────────────────────────────

def fit_beta_midas(
    X: np.ndarray,
    y: np.ndarray,
    n_lags: int,
    starts: list = None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Fit Beta-weighted MIDAS by NLS.

    Returns (params_best, sse_best, weights_best)
    """
    if starts is None:
        starts = [
            [float(y.mean()), 0.5, 1.0, 5.0],
            [float(y.mean()), 0.5, 2.0, 2.0],
            [float(y.mean()), 0.5, 5.0, 1.0],
            [float(y.mean()), -0.5, 1.0, 3.0],
        ]

    weight_fn = lambda theta: beta_weights(n_lags, theta[0], theta[1])
    obj = make_sse_objective(X, y, weight_fn)
    bounds = [(None, None), (None, None), (0.1, 20.0), (0.1, 20.0)]

    best_sse = np.inf
    best_params = starts[0]
    for p0 in starts:
        try:
            res = optimize.minimize(
                obj, p0, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 1000, "ftol": 1e-10}
            )
            if res.success and res.fun < best_sse:
                best_sse = res.fun
                best_params = res.x
        except Exception:
            continue

    weights = beta_weights(n_lags, best_params[2], best_params[3])
    return np.array(best_params), float(best_sse), weights


def fit_almon_midas(
    X: np.ndarray,
    y: np.ndarray,
    n_lags: int,
    degree: int = 2,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Fit Almon-weighted MIDAS by NLS with polynomial of given degree.

    Returns (params_best, sse_best, weights_best)
    """
    weight_fn = lambda theta: almon_weights(n_lags, theta)

    # Initial params: intercept, slope, gamma_0, gamma_1, ..., gamma_degree
    p0 = [float(y.mean()), 0.5] + [0.0] * (degree + 1)
    # Set gamma_1 = -0.1 to get a decaying profile
    p0[3] = -0.1

    obj = make_sse_objective(X, y, weight_fn)
    bounds = (
        [(None, None), (None, None)]
        + [(None, None)] * (degree + 1)
    )

    best_sse = np.inf
    best_params = p0

    starts = [p0.copy() for _ in range(4)]
    starts[1][3] = -0.5
    starts[2][3] = -0.05
    starts[3][2] = 0.5

    for s in starts:
        try:
            res = optimize.minimize(
                obj, s, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 1000, "ftol": 1e-10}
            )
            if res.success and res.fun < best_sse:
                best_sse = res.fun
                best_params = res.x
        except Exception:
            continue

    weights = almon_weights(n_lags, best_params[2:])
    return np.array(best_params), float(best_sse), weights


def fit_unrestricted_midas(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Fit unrestricted MIDAS by OLS (no restriction on lag profile).
    """
    X_ols = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(X_ols, y, rcond=None)[0]
    y_hat = X_ols @ b
    sse = float(np.sum((y - y_hat) ** 2))
    # Normalised coefficient profile (signed)
    coefs = b[1:]
    norm_coefs = coefs / (np.abs(coefs).sum() + 1e-12)
    return b, sse, norm_coefs


# ──────────────────────────────────────────────────────────────────────────────
# OOS RMSE comparison
# ──────────────────────────────────────────────────────────────────────────────

def oosrmse_expanding(
    X: np.ndarray,
    y: np.ndarray,
    weight_fn_name: str,
    n_lags: int,
    min_train: int = 20,
) -> float:
    """Expanding-window RMSE for a given weight function."""
    T = len(y)
    sq_errors = []

    for t in range(min_train, T):
        X_tr, y_tr = X[:t], y[:t]
        mask = ~np.isnan(X_tr).any(axis=1) & ~np.isnan(y_tr)
        X_tr, y_tr = X_tr[mask], y_tr[mask]
        if len(y_tr) < 10 or np.isnan(X[t]).any():
            continue

        try:
            if weight_fn_name == "beta":
                params, _, w = fit_beta_midas(X_tr, y_tr, n_lags)
                pred = params[0] + params[1] * float(X[t] @ w)
            elif weight_fn_name == "almon":
                params, _, w = fit_almon_midas(X_tr, y_tr, n_lags)
                pred = params[0] + params[1] * float(X[t] @ w)
            else:  # unrestricted
                b, _, _ = fit_unrestricted_midas(X_tr, y_tr)
                pred = float(b[0] + X[t] @ b[1:])

            sq_errors.append((pred - float(y[t])) ** 2)
        except Exception:
            continue

    return float(np.sqrt(np.mean(sq_errors))) if sq_errors else float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_weight_comparison(
    weights_beta: np.ndarray,
    weights_almon: np.ndarray,
    weights_unres: np.ndarray,
    n_lags: int,
    title: str = "Lag Weight Profiles",
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    lags = np.arange(1, n_lags + 1)

    ax.plot(lags, weights_beta, "o-", color="steelblue", linewidth=2,
            markersize=6, label="Beta polynomial")
    ax.plot(lags, weights_almon, "s--", color="darkorange", linewidth=2,
            markersize=6, label="Almon polynomial (degree 2)")
    ax.bar(lags, weights_unres, alpha=0.4, color="green", label="Unrestricted OLS")

    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Lag (months, 1 = most recent)")
    ax.set_ylabel("Weight")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Complete recipe
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Synthetic data: y driven by first 5 lags with beta(1,5) weights
    T = 60
    K = 12
    true_weights = beta_weights(K, 1.0, 5.0)

    x = np.random.randn(T + K)
    y = np.zeros(T)
    for t in range(T):
        y[t] = 0.8 * float(x[t : t + K] @ true_weights) + np.random.randn() * 0.3

    X = np.column_stack([x[t : t + K][::-1] for t in range(T)])

    print("Fitting three MIDAS weight specifications...")
    params_beta, sse_beta, w_beta = fit_beta_midas(X, y, K)
    params_almon, sse_almon, w_almon = fit_almon_midas(X, y, K, degree=2)
    b_unres, sse_unres, w_unres = fit_unrestricted_midas(X, y)

    print(f"\nIn-sample SSE:")
    print(f"  Beta (theta={params_beta[2]:.2f}, {params_beta[3]:.2f}): {sse_beta:.4f}")
    print(f"  Almon (degree 2):                                   {sse_almon:.4f}")
    print(f"  Unrestricted OLS:                                   {sse_unres:.4f}")

    print("\nOOS RMSE (expanding window, min_train=20):")
    for name in ["beta", "almon", "unrestricted"]:
        rmse = oosrmse_expanding(X, y, name, K, min_train=20)
        print(f"  {name:<15}: {rmse:.4f}")

    plot_weight_comparison(w_beta, w_almon, w_unres, K)
