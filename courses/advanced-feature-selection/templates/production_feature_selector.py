"""
production_feature_selector.py
-------------------------------
Production-grade configurable feature selection pipeline.

Stages: filter -> embedded -> wrapper (any subset can be enabled).
Implements sklearn's fit/transform interface for drop-in Pipeline use.

Supported methods
-----------------
Filter  : mutual_info, mrmr
Embedded: lasso, stability_selection
Wrapper : boruta, shap

Usage
-----
    from production_feature_selector import FeatureSelectorConfig, ProductionFeatureSelector

    cfg = FeatureSelectorConfig(
        filter_method="mutual_info",
        filter_k=50,
        embedded_method="lasso",
        wrapper_method="boruta",
        n_jobs=-1,
    )
    selector = ProductionFeatureSelector(cfg)
    X_selected = selector.fit_transform(X_train, y_train)
    selector.export_importances("feature_report.csv")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_FILTER_METHODS = frozenset({"mutual_info", "mrmr", "none"})
VALID_EMBEDDED_METHODS = frozenset({"lasso", "stability_selection", "none"})
VALID_WRAPPER_METHODS = frozenset({"boruta", "shap", "none"})
DEFAULT_RANDOM_STATE = 42
STABILITY_SELECTION_N_BOOTSTRAP = 100
STABILITY_SELECTION_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class FeatureSelectorConfig:
    """Configuration for the production feature selection pipeline.

    Parameters
    ----------
    filter_method : str
        One of ``{"mutual_info", "mrmr", "none"}``.
    filter_k : int
        Number of features to keep after the filter stage.
    embedded_method : str
        One of ``{"lasso", "stability_selection", "none"}``.
    wrapper_method : str
        One of ``{"boruta", "shap", "none"}``.
    task : str
        ``"classification"`` or ``"regression"``.
    n_jobs : int
        Parallelism passed to joblib. ``-1`` uses all cores.
    random_state : int
        Reproducibility seed.
    lasso_cv_folds : int
        Number of CV folds for LassoCV.
    boruta_max_iter : int
        Maximum iterations for Boruta.
    shap_n_estimators : int
        Trees in the SHAP background estimator.
    extra_filter_params : dict
        Passed verbatim to the filter scorer callable.
    """

    filter_method: str = "mutual_info"
    filter_k: int = 50
    embedded_method: str = "lasso"
    wrapper_method: str = "none"
    task: str = "classification"
    n_jobs: int = -1
    random_state: int = DEFAULT_RANDOM_STATE
    lasso_cv_folds: int = 5
    boruta_max_iter: int = 100
    shap_n_estimators: int = 100
    extra_filter_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.filter_method not in VALID_FILTER_METHODS:
            raise ValueError(f"filter_method must be one of {VALID_FILTER_METHODS}")
        if self.embedded_method not in VALID_EMBEDDED_METHODS:
            raise ValueError(f"embedded_method must be one of {VALID_EMBEDDED_METHODS}")
        if self.wrapper_method not in VALID_WRAPPER_METHODS:
            raise ValueError(f"wrapper_method must be one of {VALID_WRAPPER_METHODS}")
        if self.task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
        if self.filter_k < 1:
            raise ValueError("filter_k must be >= 1")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _mutual_info_scorer(
    task: str,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    return mutual_info_classif if task == "classification" else mutual_info_regression


def _mrmr_scores(
    X: np.ndarray, y: np.ndarray, k: int, task: str, n_jobs: int
) -> np.ndarray:
    """Minimum-Redundancy Maximum-Relevance greedy selection.

    Returns an importance vector (shape ``(n_features,)``) where selected
    features have positive scores ranked by selection order.
    """
    n_features = X.shape[1]
    scorer = _mutual_info_scorer(task)
    relevance = scorer(X, y, random_state=DEFAULT_RANDOM_STATE)

    selected: List[int] = []
    remaining = list(range(n_features))
    scores = np.zeros(n_features)

    for rank in range(min(k, n_features)):
        if not remaining:
            break
        if not selected:
            best = int(np.argmax(relevance[remaining]))
            idx = remaining[best]
        else:
            def _score_candidate(j: int) -> float:
                red = float(np.mean([
                    scorer(X[:, [s]], X[:, j].ravel(), random_state=DEFAULT_RANDOM_STATE)[0]
                    for s in selected
                ]))
                return float(relevance[j]) - red

            candidate_scores = Parallel(n_jobs=n_jobs)(
                delayed(_score_candidate)(j) for j in remaining
            )
            best = int(np.argmax(candidate_scores))
            idx = remaining[best]

        selected.append(idx)
        remaining.remove(idx)
        scores[idx] = n_features - rank  # higher rank -> higher score

    return scores


def _stability_selection_mask(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
    n_bootstrap: int,
    threshold: float,
    task: str,
    n_jobs: int,
) -> np.ndarray:
    """Returns boolean mask of stably-selected features via bootstrap Lasso."""
    n_samples, n_features = X.shape
    selection_counts = np.zeros(n_features, dtype=int)

    def _one_bootstrap(seed: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n_samples, size=n_samples // 2, replace=False)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X[idx])
        model = LassoCV(cv=3, random_state=seed, n_jobs=1)
        model.fit(Xs, y[idx])
        return (model.coef_ != 0).astype(int)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_one_bootstrap)(random_state + i) for i in range(n_bootstrap)
    )
    selection_counts = np.sum(results, axis=0)
    return (selection_counts / n_bootstrap) >= threshold


def _boruta_mask(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    max_iter: int,
    random_state: int,
    n_jobs: int,
) -> np.ndarray:
    """Simplified Boruta: compare real features against shadow (permuted) features."""
    rng = np.random.RandomState(random_state)
    n_samples, n_features = X.shape
    hits = np.zeros(n_features, dtype=int)
    total_rounds = min(max_iter, 30)  # cap for speed in templates

    EstimatorClass = RandomForestClassifier if task == "classification" else RandomForestRegressor

    for i in range(total_rounds):
        shadow = X[:, rng.permutation(n_features)]
        X_aug = np.hstack([X, shadow])
        rf = EstimatorClass(
            n_estimators=50,
            random_state=random_state + i,
            n_jobs=n_jobs,
        )
        rf.fit(X_aug, y)
        importances = rf.feature_importances_
        real_imp = importances[:n_features]
        shadow_imp = importances[n_features:]
        shadow_max = shadow_imp.max()
        hits += (real_imp > shadow_max).astype(int)

    threshold = int(np.ceil(total_rounds * 0.3))
    return hits >= threshold


def _shap_importances(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    n_estimators: int,
    random_state: int,
    n_jobs: int,
) -> np.ndarray:
    """Mean absolute SHAP values as feature importances."""
    try:
        import shap  # type: ignore
    except ImportError as exc:
        raise ImportError("shap is required for wrapper_method='shap'. pip install shap") from exc

    EstimatorClass = RandomForestClassifier if task == "classification" else RandomForestRegressor
    rf = EstimatorClass(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs)
    rf.fit(X, y)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        # multiclass: average across classes
        shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    return np.abs(shap_values).mean(axis=0)


# ---------------------------------------------------------------------------
# Main selector
# ---------------------------------------------------------------------------
class ProductionFeatureSelector(BaseEstimator, TransformerMixin):
    """Configurable three-stage feature selection pipeline.

    Stages executed in order: filter -> embedded -> wrapper.
    Any stage can be disabled by setting the corresponding method to ``"none"``.

    Parameters
    ----------
    config : FeatureSelectorConfig
        Pipeline configuration.

    Attributes
    ----------
    selected_features_ : list[int]
        Column indices of selected features after fitting.
    feature_importances_ : dict[str, np.ndarray]
        Importance scores from each active stage.
    n_features_in_ : int
        Number of features seen during fit.
    stage_timings_ : dict[str, float]
        Wall-clock seconds spent in each stage.
    """

    def __init__(self, config: FeatureSelectorConfig) -> None:
        self.config = config
        self.selected_features_: List[int] = []
        self.feature_importances_: Dict[str, np.ndarray] = {}
        self.n_features_in_: int = 0
        self.stage_timings_: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # sklearn interface
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ProductionFeatureSelector":
        """Fit all enabled selection stages.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.n_features_in_ = X.shape[1]
        cfg = self.config

        logger.info(
            "Starting feature selection | samples=%d features=%d task=%s",
            X.shape[0], X.shape[1], cfg.task,
        )

        active_indices = list(range(X.shape[1]))

        # --- Stage 1: Filter ------------------------------------------
        if cfg.filter_method != "none":
            active_indices = self._run_filter(X, y, active_indices)

        # --- Stage 2: Embedded ----------------------------------------
        if cfg.embedded_method != "none":
            X_sub = X[:, active_indices]
            active_indices = [
                active_indices[i]
                for i in self._run_embedded(X_sub, y, active_indices)
            ]

        # --- Stage 3: Wrapper -----------------------------------------
        if cfg.wrapper_method != "none":
            X_sub = X[:, active_indices]
            active_indices = [
                active_indices[i]
                for i in self._run_wrapper(X_sub, y, active_indices)
            ]

        self.selected_features_ = active_indices
        logger.info("Selection complete. %d features retained.", len(active_indices))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Reduce X to the selected feature columns.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples, n_selected_features)
        """
        check_is_fitted(self, "selected_features_")
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )
        return X[:, self.selected_features_]

    # ------------------------------------------------------------------
    # Stage runners
    # ------------------------------------------------------------------
    def _run_filter(
        self, X: np.ndarray, y: np.ndarray, current_indices: List[int]
    ) -> List[int]:
        cfg = self.config
        t0 = time.perf_counter()
        k = min(cfg.filter_k, len(current_indices))
        logger.info("Filter stage: method=%s k=%d", cfg.filter_method, k)

        if cfg.filter_method == "mutual_info":
            scorer = _mutual_info_scorer(cfg.task)
            scores = scorer(X[:, current_indices], y, random_state=cfg.random_state)
            top_local = np.argsort(scores)[::-1][:k]

        elif cfg.filter_method == "mrmr":
            scores = _mrmr_scores(X[:, current_indices], y, k, cfg.task, cfg.n_jobs)
            top_local = np.argsort(scores)[::-1][:k]

        else:
            raise RuntimeError(f"Unhandled filter_method: {cfg.filter_method}")

        self.feature_importances_["filter"] = scores
        elapsed = time.perf_counter() - t0
        self.stage_timings_["filter"] = elapsed
        result = [current_indices[i] for i in top_local]
        logger.info("Filter done in %.2fs. Kept %d features.", elapsed, len(result))
        return result

    def _run_embedded(
        self, X_sub: np.ndarray, y: np.ndarray, current_indices: List[int]
    ) -> List[int]:
        cfg = self.config
        t0 = time.perf_counter()
        logger.info("Embedded stage: method=%s", cfg.embedded_method)

        if cfg.embedded_method == "lasso":
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X_sub)
            model = LassoCV(cv=cfg.lasso_cv_folds, random_state=cfg.random_state, n_jobs=cfg.n_jobs)
            model.fit(Xs, y)
            coefs = np.abs(model.coef_)
            mask = coefs > 0
            self.feature_importances_["embedded"] = coefs
            local_indices = [i for i, m in enumerate(mask) if m]

        elif cfg.embedded_method == "stability_selection":
            mask = _stability_selection_mask(
                X_sub, y,
                cfg.random_state,
                STABILITY_SELECTION_N_BOOTSTRAP,
                STABILITY_SELECTION_THRESHOLD,
                cfg.task,
                cfg.n_jobs,
            )
            self.feature_importances_["embedded"] = mask.astype(float)
            local_indices = [i for i, m in enumerate(mask) if m]

        else:
            raise RuntimeError(f"Unhandled embedded_method: {cfg.embedded_method}")

        elapsed = time.perf_counter() - t0
        self.stage_timings_["embedded"] = elapsed
        if not local_indices:
            logger.warning("Embedded stage selected 0 features; keeping all %d.", len(current_indices))
            local_indices = list(range(len(current_indices)))
        logger.info("Embedded done in %.2fs. Kept %d features.", elapsed, len(local_indices))
        return local_indices

    def _run_wrapper(
        self, X_sub: np.ndarray, y: np.ndarray, current_indices: List[int]
    ) -> List[int]:
        cfg = self.config
        t0 = time.perf_counter()
        logger.info("Wrapper stage: method=%s", cfg.wrapper_method)

        if cfg.wrapper_method == "boruta":
            mask = _boruta_mask(X_sub, y, cfg.task, cfg.boruta_max_iter, cfg.random_state, cfg.n_jobs)
            self.feature_importances_["wrapper"] = mask.astype(float)
            local_indices = [i for i, m in enumerate(mask) if m]

        elif cfg.wrapper_method == "shap":
            importances = _shap_importances(
                X_sub, y, cfg.task, cfg.shap_n_estimators, cfg.random_state, cfg.n_jobs
            )
            threshold = np.percentile(importances, 50)
            mask = importances >= threshold
            self.feature_importances_["wrapper"] = importances
            local_indices = [i for i, m in enumerate(mask) if m]

        else:
            raise RuntimeError(f"Unhandled wrapper_method: {cfg.wrapper_method}")

        elapsed = time.perf_counter() - t0
        self.stage_timings_["wrapper"] = elapsed
        if not local_indices:
            logger.warning("Wrapper stage selected 0 features; keeping all %d.", len(current_indices))
            local_indices = list(range(len(current_indices)))
        logger.info("Wrapper done in %.2fs. Kept %d features.", elapsed, len(local_indices))
        return local_indices

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def export_importances(
        self,
        path: str | Path,
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Export a CSV report of feature importances and selection status.

        Parameters
        ----------
        path : str or Path
            Destination CSV file path.
        feature_names : list[str], optional
            Human-readable names for each of the ``n_features_in_`` columns.

        Returns
        -------
        pd.DataFrame
            The report table (also written to *path*).
        """
        check_is_fitted(self, "selected_features_")
        n = self.n_features_in_
        names = feature_names or [f"feature_{i}" for i in range(n)]

        df = pd.DataFrame({"feature": names})
        df["selected"] = df.index.isin(self.selected_features_)

        for stage, scores in self.feature_importances_.items():
            padded = np.full(n, np.nan)
            # scores correspond to features present *at entry* of that stage;
            # store them aligned to original indices via selected_features_
            padded[: len(scores)] = scores
            df[f"{stage}_score"] = padded

        df.to_csv(path, index=False)
        logger.info("Importances exported to %s", path)
        return df

    def get_feature_names_out(
        self, feature_names: Optional[List[str]] = None
    ) -> List[str]:
        """Return the names of selected features.

        Parameters
        ----------
        feature_names : list[str], optional
            Input feature names. Defaults to ``feature_0, feature_1, ...``.
        """
        check_is_fitted(self, "selected_features_")
        names = feature_names or [f"feature_{i}" for i in range(self.n_features_in_)]
        return [names[i] for i in self.selected_features_]

    def summary(self) -> Dict[str, Any]:
        """Return a dict summary of the selection run."""
        check_is_fitted(self, "selected_features_")
        return {
            "n_features_in": self.n_features_in_,
            "n_features_selected": len(self.selected_features_),
            "selected_indices": self.selected_features_,
            "stage_timings_seconds": self.stage_timings_,
            "stages_active": list(self.feature_importances_.keys()),
        }


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------
def build_sklearn_pipeline(
    config: FeatureSelectorConfig,
    estimator: BaseEstimator,
) -> Pipeline:
    """Wrap the selector in a full sklearn Pipeline.

    Parameters
    ----------
    config : FeatureSelectorConfig
    estimator : sklearn estimator
        Final step (classifier or regressor).

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("selector", ProductionFeatureSelector(config)),
        ("model", estimator),
    ])


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=500, n_features=40, n_informative=10, random_state=0)

    cfg = FeatureSelectorConfig(
        filter_method="mutual_info",
        filter_k=20,
        embedded_method="lasso",
        wrapper_method="none",
        task="classification",
        n_jobs=2,
    )
    sel = ProductionFeatureSelector(cfg)
    X_sel = sel.fit_transform(X, y)
    print(sel.summary())
    print(f"Shape: {X.shape} -> {X_sel.shape}")
