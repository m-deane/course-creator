"""
midas_regression_template.py — Production-Ready MIDAS Regression Template

Covers:
  - Beta polynomial weight construction
  - NLS estimation (scipy.optimize.minimize)
  - Unrestricted OLS MIDAS
  - Regularised MIDAS (Elastic Net, Ridge, Lasso) with TimeSeriesSplit CV
  - Expanding-window OOS evaluation
  - Diebold-Mariano test

Usage
-----
    from midas_regression_template import (
        beta_weights, MIDASModel, RegularisedMIDAS, expanding_window_eval
    )

    # Build a 3-indicator MIDAS model with 12 daily lags
    model = MIDASModel(n_lags=12, weight_type='beta')
    model.fit(X_train, y_train)
    forecast = model.predict(X_forecast)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.linear_model import ElasticNetCV, RidgeCV, LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────────────────────────────────────
# Beta polynomial weights
# ──────────────────────────────────────────────────────────────────────────────

def beta_weights(n_lags: int, theta1: float, theta2: float) -> np.ndarray:
    """
    Compute normalised Beta polynomial lag weights.

    w(j) = (j/K)^(theta1-1) * (1 - j/K)^(theta2-1)

    Parameters
    ----------
    n_lags : int
        Number of lag positions K.
    theta1 : float > 0
        First shape parameter.
    theta2 : float > 0
        Second shape parameter.

    Returns
    -------
    np.ndarray, shape (n_lags,), sums to 1.
    """
    j = np.linspace(0.01, 0.99, n_lags)  # avoid endpoints for numerical stability
    w = (j ** (theta1 - 1)) * ((1 - j) ** (theta2 - 1))
    total = w.sum()
    if total < 1e-12:
        return np.ones(n_lags) / n_lags
    return w / total


def almon_weights(n_lags: int, gamma: np.ndarray) -> np.ndarray:
    """
    Compute Almon polynomial lag weights.

    w(j) = exp(sum_p gamma_p * j^p) / normalisation

    Parameters
    ----------
    n_lags : int
    gamma : np.ndarray
        Polynomial coefficients (degree = len(gamma) - 1).

    Returns
    -------
    np.ndarray, shape (n_lags,), sums to 1.
    """
    j = np.arange(1, n_lags + 1, dtype=float)
    exponent = sum(gamma[p] * (j ** p) for p in range(len(gamma)))
    w = np.exp(exponent - exponent.max())  # numerical stability
    return w / w.sum()


# ──────────────────────────────────────────────────────────────────────────────
# Design matrix construction
# ──────────────────────────────────────────────────────────────────────────────

def build_midas_design_matrix(
    high_freq_series: List[pd.Series],
    low_freq_dates: pd.DatetimeIndex,
    n_lags: int,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build the MIDAS design matrix with optional pre-weighting.

    If weights is provided (shape n_lags,), the lag vectors are aggregated
    to a single value per series per period (weighted sum). This is the
    restricted MIDAS design matrix.

    If weights is None, all lags are included as separate features.
    This is the unrestricted MIDAS design matrix.

    Parameters
    ----------
    high_freq_series : list of pd.Series (same length assumed)
    low_freq_dates : pd.DatetimeIndex (target observation dates)
    n_lags : int
    weights : np.ndarray of shape (n_lags,), optional

    Returns
    -------
    X : np.ndarray, shape (T, N) if weights else (T, K*N)
    feature_names : list of str
    """
    T = len(low_freq_dates)
    N = len(high_freq_series)
    restricted = weights is not None

    if restricted:
        X = np.full((T, N), np.nan)
        feature_names = [s.name or f"series_{i}" for i, s in enumerate(high_freq_series)]
    else:
        X = np.full((T, n_lags * N), np.nan)
        feature_names = []

    for j, series in enumerate(high_freq_series):
        name = series.name or f"series_{j}"
        lag_matrix = np.full((T, n_lags), np.nan)
        for t, lf_date in enumerate(low_freq_dates):
            available = series[series.index <= lf_date]
            n_avail = len(available)
            for k in range(n_lags):
                if n_avail > k:
                    lag_matrix[t, k] = float(available.iloc[-(k + 1)])

        if restricted:
            X[:, j] = lag_matrix @ weights
        else:
            col_start = j * n_lags
            X[:, col_start : col_start + n_lags] = lag_matrix
            feature_names.extend([f"{name}_lag{k+1}" for k in range(n_lags)])

    return X, feature_names


# ──────────────────────────────────────────────────────────────────────────────
# Basic MIDAS model with NLS Beta weights
# ──────────────────────────────────────────────────────────────────────────────

class MIDASModel:
    """
    Single-indicator MIDAS regression with Beta polynomial weights.

    Estimates: y_t = alpha + beta * (w(theta)' * x_t) + epsilon_t

    Fits by NLS: joint optimisation over (alpha, beta, theta1, theta2).
    """

    def __init__(self, n_lags: int = 12, weight_type: str = "beta"):
        self.n_lags = n_lags
        self.weight_type = weight_type
        self.alpha_: Optional[float] = None
        self.beta_: Optional[float] = None
        self.theta_: Optional[np.ndarray] = None
        self.weights_: Optional[np.ndarray] = None

    def _compute_weights(self, theta: np.ndarray) -> np.ndarray:
        if self.weight_type == "beta":
            return beta_weights(self.n_lags, theta[0], theta[1])
        elif self.weight_type == "almon":
            return almon_weights(self.n_lags, theta)
        else:
            raise ValueError(f"Unknown weight_type: '{self.weight_type}'")

    def _sse(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        alpha = params[0]
        beta = params[1]
        theta = params[2:]
        try:
            w = self._compute_weights(theta)
            y_hat = alpha + beta * (X @ w)
            return float(np.sum((y - y_hat) ** 2))
        except Exception:
            return 1e12

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MIDASModel":
        """
        Fit NLS with multiple starting points for robustness.

        Parameters
        ----------
        X : np.ndarray, shape (T, K)  — lag matrix for one indicator
        y : np.ndarray, shape (T,)    — target
        """
        if X.shape[1] != self.n_lags:
            raise ValueError(
                f"X has {X.shape[1]} lags but n_lags={self.n_lags}"
            )

        best_sse = np.inf
        best_params = None

        starts = [
            [float(y.mean()), 0.5, 1.0, 5.0],
            [float(y.mean()), 0.5, 2.0, 2.0],
            [float(y.mean()), 0.5, 5.0, 1.0],
            [float(y.mean()), -0.5, 1.0, 3.0],
            [float(y.mean()), 0.5, 1.0, 1.0],
        ]
        bounds = [
            (None, None),   # alpha
            (None, None),   # beta
            (0.1, 20.0),    # theta1
            (0.1, 20.0),    # theta2
        ]

        for p0 in starts:
            try:
                res = optimize.minimize(
                    self._sse,
                    p0,
                    args=(X, y),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 1000, "ftol": 1e-10},
                )
                if res.success and res.fun < best_sse:
                    best_sse = res.fun
                    best_params = res.x
            except Exception:
                continue

        if best_params is None:
            # Fallback: use fixed Beta(1,5) weights and OLS for alpha, beta
            w = beta_weights(self.n_lags, 1.0, 5.0)
            Xw = X @ w
            X_ols = np.column_stack([np.ones(len(y)), Xw])
            b = np.linalg.lstsq(X_ols, y, rcond=None)[0]
            self.alpha_ = float(b[0])
            self.beta_ = float(b[1])
            self.theta_ = np.array([1.0, 5.0])
        else:
            self.alpha_ = float(best_params[0])
            self.beta_ = float(best_params[1])
            self.theta_ = best_params[2:]

        self.weights_ = self._compute_weights(self.theta_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.alpha_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.alpha_ + self.beta_ * (X @ self.weights_)

    def get_weights(self) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model not fitted.")
        return self.weights_.copy()


# ──────────────────────────────────────────────────────────────────────────────
# Regularised MIDAS (multi-indicator)
# ──────────────────────────────────────────────────────────────────────────────

REGULARISED_REGISTRY = {
    "elasticnet": lambda cv: ElasticNetCV(
        l1_ratio=[0.5, 0.7, 1.0], cv=cv, max_iter=10000, random_state=0
    ),
    "ridge": lambda cv: RidgeCV(cv=cv),
    "lasso": lambda cv: LassoCV(cv=cv, max_iter=10000, random_state=0),
}


class RegularisedMIDAS:
    """
    Multi-indicator MIDAS with sklearn-compatible regularised estimator.

    Features are built as an unrestricted lag matrix (no Beta restriction).
    Scaling is applied inside the class (fitted on train only).
    """

    def __init__(
        self,
        n_lags: int = 12,
        model: str = "elasticnet",
        cv_splits: int = 5,
    ):
        self.n_lags = n_lags
        self.model_name = model
        self.cv_splits = cv_splits
        self.scaler = StandardScaler()
        self._model = None
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegularisedMIDAS":
        """
        Fit on the training feature matrix. Scales internally.

        Parameters
        ----------
        X : np.ndarray, shape (T, K*N)  — unrestricted MIDAS lag matrix
        y : np.ndarray, shape (T,)      — target
        """
        cv = TimeSeriesSplit(n_splits=self.cv_splits)
        builder = REGULARISED_REGISTRY.get(self.model_name)
        if builder is None:
            raise ValueError(f"Unknown model '{self.model_name}'")
        self._model = builder(cv)
        X_scaled = self.scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        return self._model.predict(self.scaler.transform(X))

    @property
    def coef_(self) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        return self._model.coef_

    def selected_features(
        self, feature_names: List[str], tol: float = 1e-6
    ) -> List[str]:
        """Return names of non-zero features (useful after Lasso/ElasticNet)."""
        return [
            name for name, c in zip(feature_names, self.coef_)
            if abs(c) > tol
        ]


# ──────────────────────────────────────────────────────────────────────────────
# Expanding-window evaluation
# ──────────────────────────────────────────────────────────────────────────────

def expanding_window_eval(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "elasticnet",
    min_train: int = 20,
    n_lags: Optional[int] = None,
) -> pd.DataFrame:
    """
    Expanding-window out-of-sample evaluation.

    At each t >= min_train:
      - Train on X[:t], y[:t]
      - Predict y[t]
      - Record forecast, actual, error

    Parameters
    ----------
    X : np.ndarray, shape (T, p)
    y : np.ndarray, shape (T,)
    model_type : str  ("elasticnet" | "ridge" | "lasso" | "ar1")
    min_train : int
    n_lags : int, optional
        If model_type=="ar1", uses the last n_lags values of y as AR features.

    Returns
    -------
    pd.DataFrame with columns: t, forecast, actual, error, sq_error, abs_error
    """
    T = len(y)
    results = []

    for t in range(min_train, T):
        X_train = X[:t]
        y_train = y[:t]
        X_test = X[t : t + 1]
        y_test = y[t]

        # Remove NaN rows
        mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
        X_tr = X_train[mask]
        y_tr = y_train[mask]

        if len(y_tr) < 8 or np.isnan(X_test).any() or np.isnan(y_test):
            continue

        try:
            if model_type == "ar1":
                k = n_lags or 1
                X_ar = np.column_stack(
                    [np.ones(len(y_tr))]
                    + [y[:t][mask][k_:len(y_tr)+k_-len(y_tr)] for k_ in range(1, k+1)]
                )
                b = np.linalg.lstsq(X_ar, y_tr, rcond=None)[0]
                x_ar_new = np.array([1.0] + [y[t - k_] for k_ in range(1, k + 1)])
                forecast = float(x_ar_new @ b)
            else:
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_te_s = scaler.transform(X_test)
                cv = TimeSeriesSplit(n_splits=min(5, len(y_tr) // 4))
                builder = REGULARISED_REGISTRY[model_type]
                m = builder(cv)
                m.fit(X_tr_s, y_tr)
                forecast = float(m.predict(X_te_s)[0])

            error = forecast - y_test
            results.append({
                "t": t,
                "forecast": forecast,
                "actual": y_test,
                "error": error,
                "sq_error": error ** 2,
                "abs_error": abs(error),
            })
        except Exception:
            continue

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────────────────────
# Diebold-Mariano test
# ──────────────────────────────────────────────────────────────────────────────

def diebold_mariano_test(
    errors_baseline: np.ndarray,
    errors_challenger: np.ndarray,
    loss: str = "squared",
) -> Dict:
    """
    Diebold-Mariano (1995) test for equal predictive accuracy.

    H0: E[L(e_baseline) - L(e_challenger)] = 0
    Two-sided test. Newey-West variance with bandwidth h=1.

    Parameters
    ----------
    errors_baseline : np.ndarray
    errors_challenger : np.ndarray
    loss : "squared" | "absolute"

    Returns
    -------
    dict with keys: dm_stat, p_value, better_model
    """
    if loss == "squared":
        d = errors_baseline ** 2 - errors_challenger ** 2
    elif loss == "absolute":
        d = np.abs(errors_baseline) - np.abs(errors_challenger)
    else:
        raise ValueError(f"Unknown loss: '{loss}'")

    n = len(d)
    d_bar = float(np.mean(d))
    gamma0 = float(np.var(d, ddof=1))
    gamma1 = float(np.cov(d[:-1], d[1:])[0, 1]) if n > 2 else 0.0
    nw_var = max(gamma0 + 2 * gamma1, 1e-12)
    dm_stat = d_bar / np.sqrt(nw_var / n)
    p_value = float(2 * stats.t.sf(abs(dm_stat), df=n - 1))
    better = "challenger" if dm_stat > 0 else "baseline"

    return {
        "dm_stat": float(dm_stat),
        "p_value": p_value,
        "significant_at_10pct": p_value < 0.10,
        "better_model": better,
        "interpretation": (
            f"{'Challenger' if dm_stat > 0 else 'Baseline'} is more accurate "
            f"({'significant' if p_value < 0.10 else 'not significant'} at 10%, p={p_value:.4f})"
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("midas_regression_template.py loaded successfully.")
    print()

    # Quick demonstration with synthetic data
    np.random.seed(42)
    T = 60
    K = 12

    # Synthetic: y driven by the first 3 lags of x1
    x1 = np.random.randn(T + K)
    y = np.zeros(T)
    w_true = beta_weights(K, 1.0, 5.0)
    for t in range(T):
        y[t] = 0.5 * np.dot(x1[t : t + K][::-1], w_true) + np.random.randn() * 0.2

    X = np.column_stack([x1[t : t + K][::-1] for t in range(T)])

    # Fit MIDASModel
    model = MIDASModel(n_lags=K, weight_type="beta")
    model.fit(X, y)
    preds = model.predict(X)
    rmse = float(np.sqrt(np.mean((y - preds) ** 2)))
    print(f"MIDASModel (NLS): in-sample RMSE = {rmse:.4f}")
    print(f"Estimated theta: {model.theta_}")
    print()

    # Fit RegularisedMIDAS
    rm = RegularisedMIDAS(n_lags=K, model="elasticnet")
    rm.fit(X, y)
    preds_rm = rm.predict(X)
    rmse_rm = float(np.sqrt(np.mean((y - preds_rm) ** 2)))
    print(f"RegularisedMIDAS (ElasticNet): in-sample RMSE = {rmse_rm:.4f}")
    n_nonzero = int(np.sum(np.abs(rm.coef_) > 1e-6))
    print(f"Non-zero coefficients: {n_nonzero} / {K}")
