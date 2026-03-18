"""
forecast_evaluation_template.py — Expanding-Window Forecast Evaluation Framework

Provides:
  - ExpandingWindowEvaluator: runs OOS eval for multiple models simultaneously
  - EvaluationSummary: computes RMSE, MAE, bias, directional accuracy
  - DieboldMarianoTable: pairwise DM tests for all model pairs
  - RollingMetrics: 8-quarter rolling RMSE/MAE/bias
  - CoverageTest: empirical coverage rate for prediction intervals

Usage
-----
    from forecast_evaluation_template import (
        ExpandingWindowEvaluator, EvaluationSummary, DieboldMarianoTable
    )

    evaluator = ExpandingWindowEvaluator(min_train=20)
    evaluator.add_model("elasticnet", en_model)
    evaluator.add_model("ridge", ridge_model)
    evaluator.run(X, y)

    summary = EvaluationSummary(evaluator.results)
    summary.print_table()

    dm_table = DieboldMarianoTable(evaluator.results, baseline="elasticnet")
    dm_table.print_table()
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler


# ──────────────────────────────────────────────────────────────────────────────
# Model wrapper protocol
# ──────────────────────────────────────────────────────────────────────────────

class ModelWrapper:
    """
    Thin wrapper around any sklearn-compatible estimator.
    Handles scaling internally so the evaluator can treat all models uniformly.
    """

    def __init__(self, estimator: Any, scale: bool = True):
        self.estimator = estimator
        self.scale = scale
        self._scaler = StandardScaler() if scale else None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ModelWrapper":
        if self.scale:
            X = self._scaler.fit_transform(X)
        self.estimator.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.scale and self._scaler is not None:
            X = self._scaler.transform(X)
        return self.estimator.predict(X)

    def clone(self) -> "ModelWrapper":
        """Return a new unfitted wrapper with the same configuration."""
        import copy
        return ModelWrapper(copy.deepcopy(self.estimator), self.scale)


class AR1Benchmark:
    """Naive AR(1) benchmark — no feature matrix needed."""

    def __init__(self):
        self._b: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AR1Benchmark":
        if len(y) < 2:
            self._b = np.array([float(y.mean()), 0.0])
            return self
        X_ar = np.column_stack([np.ones(len(y) - 1), y[:-1]])
        self._b = np.linalg.lstsq(X_ar, y[1:], rcond=None)[0]
        return self

    def predict(self, X: np.ndarray, y_last: float) -> float:
        if self._b is None:
            raise RuntimeError("Fit AR1Benchmark before predicting.")
        return float(self._b[0] + self._b[1] * y_last)


# ──────────────────────────────────────────────────────────────────────────────
# Expanding-window evaluator
# ──────────────────────────────────────────────────────────────────────────────

class ExpandingWindowEvaluator:
    """
    Run expanding-window OOS evaluation for multiple models simultaneously.

    At each t >= min_train:
      - Train each model on X[:t], y[:t]
      - Predict X[t]
      - Record (forecast, actual) for each model

    Parameters
    ----------
    min_train : int
        Minimum number of training observations before generating a forecast.
    gap : int
        Number of periods to skip between end of training and forecast period
        (0 = forecast t+1 directly after training on t).
    """

    def __init__(self, min_train: int = 20, gap: int = 0):
        self.min_train = min_train
        self.gap = gap
        self._models: Dict[str, Any] = {}
        self.results: Dict[str, pd.DataFrame] = {}

    def add_model(self, name: str, model: Any) -> "ExpandingWindowEvaluator":
        """Register a model for evaluation."""
        self._models[name] = model
        return self

    def run(self, X: np.ndarray, y: np.ndarray) -> "ExpandingWindowEvaluator":
        """
        Execute expanding-window evaluation for all registered models.

        Results stored in self.results[model_name] as DataFrame with columns:
            t, forecast, actual, error, sq_error, abs_error
        """
        T = len(y)
        all_results: Dict[str, List[Dict]] = {name: [] for name in self._models}

        for t in range(self.min_train, T - self.gap):
            forecast_t = t + self.gap
            if forecast_t >= T:
                break

            X_train = X[:t]
            y_train = y[:t]
            X_test = X[forecast_t : forecast_t + 1]
            y_test = float(y[forecast_t])

            # Skip if target is NaN
            if np.isnan(y_test):
                continue

            # Remove NaN training rows
            mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
            X_tr = X_train[mask]
            y_tr = y_train[mask]

            if len(y_tr) < 8:
                continue

            if np.isnan(X_test).any():
                continue

            for name, model in self._models.items():
                try:
                    clone = model.clone() if hasattr(model, "clone") else model
                    clone.fit(X_tr, y_tr)
                    forecast = float(clone.predict(X_test)[0])
                    error = forecast - y_test
                    all_results[name].append({
                        "t": forecast_t,
                        "forecast": forecast,
                        "actual": y_test,
                        "error": error,
                        "sq_error": error ** 2,
                        "abs_error": abs(error),
                    })
                except Exception as e:
                    pass  # Skip failed runs silently

        self.results = {
            name: pd.DataFrame(rows)
            for name, rows in all_results.items()
        }
        return self


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation summary
# ──────────────────────────────────────────────────────────────────────────────

class EvaluationSummary:
    """
    Compute and display evaluation metrics for all models.
    """

    def __init__(self, results: Dict[str, pd.DataFrame]):
        self.results = results
        self._summary: Optional[pd.DataFrame] = None

    def compute(self) -> pd.DataFrame:
        rows = []
        for name, df in self.results.items():
            if df.empty:
                continue
            errors = df["error"].values
            actuals = df["actual"].values
            forecasts = df["forecast"].values

            rmse = float(np.sqrt(np.mean(errors ** 2)))
            mae = float(np.mean(np.abs(errors)))
            bias = float(np.mean(errors))
            n = len(errors)

            # Directional accuracy: did forecast and actual change in same direction?
            if n > 1:
                dir_acc = float(np.mean(
                    np.sign(np.diff(forecasts)) == np.sign(np.diff(actuals))
                ))
            else:
                dir_acc = float("nan")

            # Mincer-Zarnowitz R²: regress actual on forecast
            X_mz = np.column_stack([np.ones(n), forecasts])
            b_mz = np.linalg.lstsq(X_mz, actuals, rcond=None)[0]
            mz_r2 = float(1 - np.sum((actuals - X_mz @ b_mz) ** 2) / np.var(actuals) / n)

            rows.append({
                "model": name,
                "n_forecasts": n,
                "rmse": rmse,
                "mae": mae,
                "bias": bias,
                "dir_acc": dir_acc,
                "mz_r2": mz_r2,
            })

        self._summary = pd.DataFrame(rows).sort_values("rmse")
        return self._summary

    def print_table(self) -> None:
        if self._summary is None:
            self.compute()
        pd.set_option("display.float_format", "{:.4f}".format)
        print("\nEvaluation Summary")
        print("=" * 70)
        print(self._summary.to_string(index=False))
        print("=" * 70)


# ──────────────────────────────────────────────────────────────────────────────
# Diebold-Mariano table
# ──────────────────────────────────────────────────────────────────────────────

class DieboldMarianoTable:
    """
    Pairwise Diebold-Mariano tests: all challengers vs a chosen baseline.
    """

    def __init__(
        self,
        results: Dict[str, pd.DataFrame],
        baseline: str,
        loss: str = "squared",
    ):
        self.results = results
        self.baseline = baseline
        self.loss = loss
        self._table: Optional[pd.DataFrame] = None

    def _dm_test(
        self, e_base: np.ndarray, e_chal: np.ndarray
    ) -> Tuple[float, float]:
        """Returns (dm_stat, p_value)."""
        if self.loss == "squared":
            d = e_base ** 2 - e_chal ** 2
        else:
            d = np.abs(e_base) - np.abs(e_chal)

        n = len(d)
        if n < 3:
            return float("nan"), float("nan")

        d_bar = float(np.mean(d))
        gamma0 = float(np.var(d, ddof=1))
        gamma1 = float(np.cov(d[:-1], d[1:])[0, 1])
        nw_var = max(gamma0 + 2 * gamma1, 1e-12)
        dm = d_bar / np.sqrt(nw_var / n)
        p = float(2 * stats.t.sf(abs(dm), df=n - 1))
        return float(dm), p

    def compute(self) -> pd.DataFrame:
        if self.baseline not in self.results:
            raise ValueError(f"Baseline '{self.baseline}' not in results")

        base_errors = self.results[self.baseline]["error"].values
        base_rmse = float(np.sqrt(np.mean(base_errors ** 2)))

        rows = []
        for name, df in self.results.items():
            if df.empty:
                continue
            errors = df["error"].values
            rmse = float(np.sqrt(np.mean(errors ** 2)))

            if name == self.baseline:
                dm_stat = float("nan")
                p_value = float("nan")
                better = False
            else:
                n_common = min(len(base_errors), len(errors))
                dm_stat, p_value = self._dm_test(
                    base_errors[-n_common:], errors[-n_common:]
                )
                better = bool(dm_stat > 0 and p_value < 0.10)

            rows.append({
                "model": name,
                "rmse": rmse,
                "rmse_vs_baseline": rmse / base_rmse,
                "dm_stat": dm_stat,
                "p_value": p_value,
                "better_than_baseline": better,
            })

        self._table = pd.DataFrame(rows).sort_values("rmse")
        return self._table

    def print_table(self) -> None:
        if self._table is None:
            self.compute()
        pd.set_option("display.float_format", "{:.4f}".format)
        print(f"\nDiebold-Mariano Table (baseline = {self.baseline})")
        print("=" * 70)
        print(self._table.to_string(index=False))
        print("=" * 70)
        print("better_than_baseline: True if DM stat > 0 and p < 0.10")


# ──────────────────────────────────────────────────────────────────────────────
# Rolling metrics
# ──────────────────────────────────────────────────────────────────────────────

class RollingMetrics:
    """
    Compute rolling RMSE, MAE, and bias over a fixed window.
    """

    def __init__(self, window: int = 8):
        self.window = window

    def compute(self, errors: np.ndarray) -> pd.DataFrame:
        n = len(errors)
        rows = []
        for i in range(n):
            start = max(0, i - self.window + 1)
            w = errors[start : i + 1]
            rows.append({
                "t": i,
                "rmse": float(np.sqrt(np.mean(w ** 2))),
                "mae": float(np.mean(np.abs(w))),
                "bias": float(np.mean(w)),
                "n": len(w),
            })
        return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Prediction interval coverage test
# ──────────────────────────────────────────────────────────────────────────────

class CoverageTest:
    """
    Test whether prediction intervals have correct empirical coverage.

    Compares nominal coverage (e.g. 95%) to empirical coverage.
    Uses a binomial test: H0 coverage = nominal.
    """

    def __init__(self, nominal_coverage: float = 0.95):
        self.nominal = nominal_coverage

    def compute(
        self,
        actuals: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> Dict:
        """
        Parameters
        ----------
        actuals : np.ndarray
        lower : np.ndarray (lower bound of prediction interval)
        upper : np.ndarray (upper bound of prediction interval)

        Returns
        -------
        dict with: empirical_coverage, n_violations, binom_pvalue
        """
        covered = (actuals >= lower) & (actuals <= upper)
        n = len(covered)
        k = int(covered.sum())
        emp_coverage = float(k / n)

        # Binomial test: H0 coverage = self.nominal
        p_value = float(
            stats.binom_test(k, n, self.nominal, alternative="two-sided")
        )

        return {
            "n": n,
            "n_covered": k,
            "n_violations": n - k,
            "empirical_coverage": emp_coverage,
            "nominal_coverage": self.nominal,
            "excess_violations": n - k - int(n * (1 - self.nominal)),
            "binom_pvalue": p_value,
            "correct_coverage": p_value > 0.05,
            "interpretation": (
                f"Empirical coverage = {emp_coverage:.1%} "
                f"(nominal = {self.nominal:.0%}). "
                + (
                    "Coverage is correct (binom test p > 0.05)."
                    if p_value > 0.05
                    else f"Coverage is INCORRECT (p = {p_value:.4f})."
                )
            ),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings
    from sklearn.linear_model import ElasticNetCV, RidgeCV
    from sklearn.model_selection import TimeSeriesSplit

    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Synthetic data
    T, K = 80, 6
    X = np.random.randn(T, K)
    y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(T) * 0.5

    # Define models
    cv = TimeSeriesSplit(n_splits=5)
    en = ModelWrapper(ElasticNetCV(cv=cv, max_iter=5000))
    ridge = ModelWrapper(RidgeCV(cv=cv))

    # Evaluate
    evaluator = ExpandingWindowEvaluator(min_train=20)
    evaluator.add_model("elasticnet", en)
    evaluator.add_model("ridge", ridge)
    evaluator.run(X, y)

    # Summary
    summary = EvaluationSummary(evaluator.results)
    summary.print_table()

    # DM table
    dm_table = DieboldMarianoTable(evaluator.results, baseline="elasticnet")
    dm_table.print_table()

    # Rolling metrics
    rolling = RollingMetrics(window=8)
    en_rolling = rolling.compute(evaluator.results["elasticnet"]["error"].values)
    print(f"\nFinal rolling RMSE (EN): {en_rolling['rmse'].iloc[-1]:.4f}")
