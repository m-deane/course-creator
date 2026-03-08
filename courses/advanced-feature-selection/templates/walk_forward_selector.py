"""
walk_forward_selector.py
------------------------
Time-series-aware feature selection using walk-forward validation.

Key ideas
---------
- Walk-forward windows: expanding or sliding.
- Purged cross-validation (de Prado style) with configurable embargo.
- Feature stability tracking across windows.
- Simple regime detection to flag structural breaks.
- Adaptive re-selection when out-of-sample performance degrades.

Usage
-----
    from walk_forward_selector import WalkForwardConfig, WalkForwardSelector

    cfg = WalkForwardConfig(
        window_type="expanding",
        initial_train_size=252,
        test_size=63,
        embargo_periods=5,
        n_features_to_select=20,
        reselect_threshold=0.05,
    )
    wf = WalkForwardSelector(cfg)
    report = wf.fit(X, y, timestamps)

    # Access per-window selected features
    print(report.window_features)

    # Stability table
    print(report.stability_df)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REGIME_VOLATILITY_WINDOW = 21          # rolling std lookback for regime detection
REGIME_Z_THRESHOLD = 2.0              # z-score threshold to flag a regime change
MIN_TRAIN_SAMPLES = 30                 # guard: skip window if train set too small


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward feature selection.

    Parameters
    ----------
    window_type : str
        ``"expanding"`` (train grows) or ``"sliding"`` (fixed-size train).
    initial_train_size : int
        Number of samples in the first training window.
    test_size : int
        Number of samples per test fold.
    embargo_periods : int
        Samples dropped between train end and test start (purging leakage).
    n_features_to_select : int
        Target number of features to select per window.
    selection_method : str
        One of ``{"mutual_info", "lasso", "random_forest"}``.
    task : str
        ``"classification"`` or ``"regression"``.
    reselect_threshold : float
        Performance drop (absolute) that triggers forced re-selection.
    regime_detection : bool
        Whether to flag regime changes and force re-selection.
    random_state : int
        Reproducibility seed.
    n_jobs : int
        Joblib parallelism.
    """

    window_type: str = "expanding"
    initial_train_size: int = 252
    test_size: int = 63
    embargo_periods: int = 5
    n_features_to_select: int = 20
    selection_method: str = "mutual_info"
    task: str = "classification"
    reselect_threshold: float = 0.05
    regime_detection: bool = True
    random_state: int = 42
    n_jobs: int = -1

    def __post_init__(self) -> None:
        valid_windows = {"expanding", "sliding"}
        valid_methods = {"mutual_info", "lasso", "random_forest"}
        if self.window_type not in valid_windows:
            raise ValueError(f"window_type must be one of {valid_windows}")
        if self.selection_method not in valid_methods:
            raise ValueError(f"selection_method must be one of {valid_methods}")
        if self.task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
        if self.embargo_periods < 0:
            raise ValueError("embargo_periods must be >= 0")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class WalkForwardReport:
    """Results from a completed walk-forward selection run.

    Attributes
    ----------
    window_features : list[list[int]]
        Per-window selected feature indices.
    window_scores : list[float]
        Out-of-sample performance score per window.
    stability_df : pd.DataFrame
        Selection frequency and stability per feature.
    regime_flags : list[bool]
        Whether a regime change was detected at window start.
    reselection_flags : list[bool]
        Whether re-selection was triggered by performance degradation.
    n_windows : int
        Number of completed walk-forward windows.
    """

    window_features: List[List[int]]
    window_scores: List[float]
    stability_df: pd.DataFrame
    regime_flags: List[bool]
    reselection_flags: List[bool]
    n_windows: int

    def consensus_features(self, min_frequency: float = 0.5) -> List[int]:
        """Return features selected in at least ``min_frequency`` fraction of windows.

        Parameters
        ----------
        min_frequency : float
            Minimum selection rate in ``(0, 1]``.

        Returns
        -------
        list[int]
            Feature indices sorted by selection frequency (descending).
        """
        stable = self.stability_df[self.stability_df["selection_rate"] >= min_frequency]
        return stable.sort_values("selection_rate", ascending=False).index.tolist()


# ---------------------------------------------------------------------------
# Feature selection helpers
# ---------------------------------------------------------------------------
def _select_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
    method: str,
    task: str,
    random_state: int,
    n_jobs: int,
) -> List[int]:
    """Run a single-window feature selection; return list of selected indices."""
    n_feat = X_train.shape[1]
    k = min(k, n_feat)

    if method == "mutual_info":
        scorer = mutual_info_classif if task == "classification" else mutual_info_regression
        scores = scorer(X_train, y_train, random_state=random_state)
        return list(np.argsort(scores)[::-1][:k])

    if method == "lasso":
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_train)
        model = LassoCV(cv=3, random_state=random_state, n_jobs=n_jobs)
        model.fit(Xs, y_train)
        coefs = np.abs(model.coef_)
        nonzero = np.where(coefs > 0)[0].tolist()
        if len(nonzero) < k:
            # Fall back to top-k by coef magnitude
            nonzero = list(np.argsort(coefs)[::-1][:k])
        return nonzero[:k]

    if method == "random_forest":
        rf = RandomForestClassifier(
            n_estimators=100, random_state=random_state, n_jobs=n_jobs
        ) if task == "classification" else __import__(
            "sklearn.ensemble", fromlist=["RandomForestRegressor"]
        ).RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=n_jobs)
        rf.fit(X_train, y_train)
        imps = rf.feature_importances_
        return list(np.argsort(imps)[::-1][:k])

    raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------
def _detect_regime_change(
    y_series: np.ndarray,
    lookback: int = REGIME_VOLATILITY_WINDOW,
    z_threshold: float = REGIME_Z_THRESHOLD,
) -> bool:
    """Return True if the most recent period's volatility is anomalous.

    Parameters
    ----------
    y_series : np.ndarray
        Target values in the current training window.
    lookback : int
        Rolling window length for volatility estimation.
    z_threshold : float
        Number of historical std deviations to flag as a regime change.
    """
    if len(y_series) < lookback * 2:
        return False
    returns = np.diff(y_series.astype(float))
    rolling_vols = np.array([
        returns[i - lookback:i].std()
        for i in range(lookback, len(returns) + 1)
    ])
    if rolling_vols.std() == 0:
        return False
    recent_z = (rolling_vols[-1] - rolling_vols[:-1].mean()) / rolling_vols[:-1].std()
    return float(abs(recent_z)) > z_threshold


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def _oos_score(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    selected: List[int],
    task: str,
    random_state: int,
    n_jobs: int,
) -> float:
    """Train a quick RF on selected features; return OOS score."""
    if not selected:
        return 0.0

    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    X_tr = X_train[:, selected]
    X_te = X_test[:, selected]

    if task == "classification":
        model = RandomForestClassifier(
            n_estimators=50, random_state=random_state, n_jobs=n_jobs
        )
        model.fit(X_tr, y_train)
        if len(np.unique(y_test)) < 2:
            return float(np.mean(model.predict(X_te) == y_test))
        return roc_auc_score(y_test, model.predict_proba(X_te)[:, 1])
    else:
        model = RandomForestRegressor(
            n_estimators=50, random_state=random_state, n_jobs=n_jobs
        )
        model.fit(X_tr, y_train)
        return r2_score(y_test, model.predict(X_te))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class WalkForwardSelector:
    """Feature selector with temporal walk-forward validation.

    Walks through time, selecting features on each training window and
    evaluating on the held-out test window. Supports regime detection and
    adaptive re-selection.

    Parameters
    ----------
    config : WalkForwardConfig
    """

    def __init__(self, config: WalkForwardConfig) -> None:
        self.config = config
        self._report: Optional[WalkForwardReport] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> WalkForwardReport:
        """Run the full walk-forward selection loop.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)
        timestamps : np.ndarray, optional
            Temporal index for each sample. Used for logging only.
        feature_names : list[str], optional
            Human-readable feature names for the stability report.

        Returns
        -------
        WalkForwardReport
        """
        cfg = self.config
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        n_samples, n_features = X.shape

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        windows = self._build_windows(n_samples)
        logger.info(
            "Walk-forward: %d windows | window_type=%s | embargo=%d",
            len(windows), cfg.window_type, cfg.embargo_periods,
        )

        window_features: List[List[int]] = []
        window_scores: List[float] = []
        regime_flags: List[bool] = []
        reselection_flags: List[bool] = []

        prev_score: Optional[float] = None
        prev_features: Optional[List[int]] = None

        for w_idx, (train_idx, test_idx) in enumerate(windows):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            if len(X_train) < MIN_TRAIN_SAMPLES:
                logger.warning("Window %d: train too small (%d). Skipping.", w_idx, len(X_train))
                continue

            # Regime detection
            regime_change = False
            if cfg.regime_detection:
                regime_change = _detect_regime_change(y_train)
                if regime_change:
                    logger.info("Window %d: regime change detected.", w_idx)

            # Adaptive re-selection trigger
            force_reselect = regime_change
            if prev_score is not None and prev_features is not None:
                current_score = _oos_score(
                    X_train, y_train, X_test, y_test,
                    prev_features, cfg.task, cfg.random_state, cfg.n_jobs,
                )
                if (prev_score - current_score) > cfg.reselect_threshold:
                    logger.info(
                        "Window %d: performance drop %.4f -> %.4f. Re-selecting.",
                        w_idx, prev_score, current_score,
                    )
                    force_reselect = True

            if w_idx == 0 or force_reselect:
                selected = _select_features(
                    X_train, y_train,
                    cfg.n_features_to_select,
                    cfg.selection_method,
                    cfg.task,
                    cfg.random_state,
                    cfg.n_jobs,
                )
            else:
                selected = prev_features  # type: ignore[assignment]

            score = _oos_score(
                X_train, y_train, X_test, y_test,
                selected, cfg.task, cfg.random_state, cfg.n_jobs,
            )

            window_features.append(selected)
            window_scores.append(score)
            regime_flags.append(regime_change)
            reselection_flags.append(force_reselect)

            logger.debug(
                "Window %d | train=%d test=%d selected=%d score=%.4f",
                w_idx, len(train_idx), len(test_idx), len(selected), score,
            )

            prev_score = score
            prev_features = selected

        stability_df = self._build_stability_report(
            window_features, n_features, feature_names
        )

        self._report = WalkForwardReport(
            window_features=window_features,
            window_scores=window_scores,
            stability_df=stability_df,
            regime_flags=regime_flags,
            reselection_flags=reselection_flags,
            n_windows=len(window_features),
        )
        logger.info(
            "Walk-forward complete. %d windows. Mean OOS score: %.4f",
            len(window_features),
            float(np.mean(window_scores)) if window_scores else float("nan"),
        )
        return self._report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_windows(
        self, n_samples: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate (train_indices, test_indices) pairs."""
        cfg = self.config
        windows: List[Tuple[np.ndarray, np.ndarray]] = []
        train_start = 0
        train_end = cfg.initial_train_size

        while train_end + cfg.embargo_periods + cfg.test_size <= n_samples:
            test_start = train_end + cfg.embargo_periods
            test_end = test_start + cfg.test_size

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, min(test_end, n_samples))

            windows.append((train_idx, test_idx))

            # Advance
            if cfg.window_type == "sliding":
                train_start += cfg.test_size
            train_end += cfg.test_size

        return windows

    @staticmethod
    def _build_stability_report(
        window_features: List[List[int]],
        n_features: int,
        feature_names: List[str],
    ) -> pd.DataFrame:
        """Build a stability table counting how often each feature is selected."""
        counts = np.zeros(n_features, dtype=int)
        n_windows = len(window_features)

        for selected in window_features:
            for idx in selected:
                if 0 <= idx < n_features:
                    counts[idx] += 1

        df = pd.DataFrame({
            "feature_name": feature_names,
            "selection_count": counts,
            "selection_rate": counts / max(n_windows, 1),
        })
        df.index.name = "feature_index"
        df = df.sort_values("selection_rate", ascending=False)
        return df

    @property
    def report(self) -> WalkForwardReport:
        """Access the most recent fit report."""
        if self._report is None:
            raise RuntimeError("Call fit() before accessing report.")
        return self._report

    def export_report(self, path: str) -> None:
        """Save the stability report as CSV.

        Parameters
        ----------
        path : str
            Destination file path.
        """
        self.report.stability_df.to_csv(path)
        logger.info("Stability report saved to %s", path)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    rng = np.random.default_rng(0)
    n, p = 600, 30
    X = rng.standard_normal((n, p))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    cfg = WalkForwardConfig(
        window_type="expanding",
        initial_train_size=200,
        test_size=50,
        embargo_periods=5,
        n_features_to_select=10,
        selection_method="mutual_info",
        task="classification",
        reselect_threshold=0.05,
        regime_detection=True,
    )

    wf = WalkForwardSelector(cfg)
    report = wf.fit(X, y)

    print(f"\nWindows completed : {report.n_windows}")
    print(f"Mean OOS AUC     : {np.mean(report.window_scores):.4f}")
    print(f"\nConsensus features (>=50% windows): {report.consensus_features(0.5)}")
    print("\nStability table (top 10):")
    print(report.stability_df.head(10).to_string())
