"""
high_dimensional_screener.py
-----------------------------
Feature screening for high-dimensional problems (p >> n).

Pipeline
--------
1. SIS  : Sure Independence Screening — rank features by marginal correlation
          (or distance correlation / mutual information) and keep a moderate set.
2. ISIS : Iterative SIS — repeatedly screen residuals to handle multicollinearity.
3. Stability selection layer — bootstrap the screening to produce stable subsets.
4. Post-selection inference — debiased Lasso confidence intervals for selected features.

Designed to handle p > 10,000 features efficiently via vectorised NumPy ops
and optional chunked distance-correlation computation.

Usage
-----
    from high_dimensional_screener import ScreenerConfig, HighDimensionalScreener

    cfg = ScreenerConfig(
        criterion="mi",
        sis_keep=500,
        isis_iterations=3,
        stability_n_bootstrap=100,
        stability_threshold=0.6,
        compute_intervals=True,
    )
    screener = HighDimensionalScreener(cfg)
    screener.fit(X, y)

    print(screener.selected_features_)           # stable final selection
    print(screener.confidence_intervals_)        # debiased CIs
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNK_SIZE = 1000              # features per chunk for distance correlation
MIN_SIS_KEEP = 10             # lower bound on features kept by SIS
DEBIASED_LAMBDA_RATIO = 0.1   # lambda for debiased lasso = lambda_cv * ratio


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class ScreenerConfig:
    """Configuration for high-dimensional feature screening.

    Parameters
    ----------
    criterion : str
        Screening criterion: ``"correlation"``, ``"dcor"`` (distance correlation),
        or ``"mi"`` (mutual information).
    sis_keep : int
        Number of features to retain after SIS. Rule of thumb: n / log(n).
    task : str
        ``"classification"`` or ``"regression"``.
    isis_iterations : int
        Number of ISIS rounds (0 = SIS only).
    isis_keep_per_iter : int
        Additional features added per ISIS iteration.
    stability_n_bootstrap : int
        Bootstrap draws for stability selection (0 = skip).
    stability_threshold : float
        Minimum selection rate to declare a feature stable.
    compute_intervals : bool
        Whether to compute debiased Lasso confidence intervals.
    alpha_ci : float
        Confidence level for intervals (e.g. 0.05 -> 95% CI).
    n_jobs : int
        Joblib parallelism for bootstrap loops.
    random_state : int
        Reproducibility seed.
    chunk_size : int
        Feature chunk size for distance correlation (memory control).
    """

    criterion: str = "correlation"
    sis_keep: int = 200
    task: str = "regression"
    isis_iterations: int = 2
    isis_keep_per_iter: int = 50
    stability_n_bootstrap: int = 100
    stability_threshold: float = 0.6
    compute_intervals: bool = True
    alpha_ci: float = 0.05
    n_jobs: int = -1
    random_state: int = 42
    chunk_size: int = CHUNK_SIZE

    def __post_init__(self) -> None:
        valid_criteria = {"correlation", "dcor", "mi"}
        if self.criterion not in valid_criteria:
            raise ValueError(f"criterion must be one of {valid_criteria}")
        if self.task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
        if not 0 < self.stability_threshold < 1:
            raise ValueError("stability_threshold must be in (0, 1)")
        if self.sis_keep < MIN_SIS_KEEP:
            raise ValueError(f"sis_keep must be >= {MIN_SIS_KEEP}")


# ---------------------------------------------------------------------------
# Screening criteria
# ---------------------------------------------------------------------------
def _pearson_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Absolute Pearson correlation of each feature with y."""
    y_c = y - y.mean()
    X_c = X - X.mean(axis=0)
    denom = np.sqrt((X_c ** 2).sum(axis=0)) * np.sqrt((y_c ** 2).sum())
    denom = np.where(denom == 0, 1e-12, denom)
    return np.abs((X_c * y_c[:, None]).sum(axis=0) / denom)


def _dcor_chunk(X_chunk: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Distance correlation for a chunk of features (vectorised)."""
    n = len(y)
    scores = np.zeros(X_chunk.shape[1])

    # Precompute y distance matrix once per chunk call
    dy = np.abs(y[:, None] - y[None, :])
    dy -= dy.mean(axis=0)[None, :]
    dy -= dy.mean(axis=1)[:, None]
    dy += dy.mean()
    dcov_yy = float(np.sqrt(np.maximum((dy ** 2).mean(), 0.0)))

    for j in range(X_chunk.shape[1]):
        xj = X_chunk[:, j]
        dx = np.abs(xj[:, None] - xj[None, :])
        dx -= dx.mean(axis=0)[None, :]
        dx -= dx.mean(axis=1)[:, None]
        dx += dx.mean()
        dcov_xx = float(np.sqrt(np.maximum((dx ** 2).mean(), 0.0)))
        dcov_xy = float(np.sqrt(np.maximum((dx * dy).mean(), 0.0)))
        denom = dcov_xx * dcov_yy
        scores[j] = dcov_xy / denom if denom > 1e-12 else 0.0

    return scores


def _distance_correlation_scores(
    X: np.ndarray, y: np.ndarray, chunk_size: int, n_jobs: int
) -> np.ndarray:
    """Distance correlation for all features; chunked for memory efficiency."""
    n_feat = X.shape[1]
    chunks = [
        X[:, start: start + chunk_size]
        for start in range(0, n_feat, chunk_size)
    ]
    results = Parallel(n_jobs=n_jobs)(
        delayed(_dcor_chunk)(chunk, y) for chunk in chunks
    )
    return np.concatenate(results)


def _mutual_info_scores(X: np.ndarray, y: np.ndarray, task: str, seed: int) -> np.ndarray:
    """Mutual information scores (handles classification and regression)."""
    if task == "classification":
        return mutual_info_classif(X, y, random_state=seed)
    return mutual_info_regression(X, y, random_state=seed)


def _compute_scores(
    X: np.ndarray,
    y: np.ndarray,
    criterion: str,
    task: str,
    chunk_size: int,
    n_jobs: int,
    random_state: int,
) -> np.ndarray:
    """Dispatch to the chosen screening criterion."""
    if criterion == "correlation":
        return _pearson_scores(X, y.astype(float))
    if criterion == "dcor":
        return _distance_correlation_scores(X, y.astype(float), chunk_size, n_jobs)
    if criterion == "mi":
        return _mutual_info_scores(X, y, task, random_state)
    raise ValueError(f"Unknown criterion: {criterion}")


# ---------------------------------------------------------------------------
# SIS / ISIS
# ---------------------------------------------------------------------------
def _sis(
    X: np.ndarray,
    y: np.ndarray,
    keep: int,
    criterion: str,
    task: str,
    chunk_size: int,
    n_jobs: int,
    random_state: int,
    candidate_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run one SIS pass.

    Parameters
    ----------
    candidate_indices : np.ndarray, optional
        Restrict search to these original column indices (for ISIS).

    Returns
    -------
    selected : np.ndarray
        Original column indices of top-keep features.
    scores : np.ndarray
        Full score vector aligned to candidate_indices.
    """
    if candidate_indices is not None:
        X_sub = X[:, candidate_indices]
    else:
        X_sub = X
        candidate_indices = np.arange(X.shape[1])

    scores = _compute_scores(X_sub, y, criterion, task, chunk_size, n_jobs, random_state)
    keep = min(keep, len(candidate_indices))
    top_local = np.argsort(scores)[::-1][:keep]
    return candidate_indices[top_local], scores


def _isis(
    X: np.ndarray,
    y: np.ndarray,
    initial_selected: np.ndarray,
    n_iterations: int,
    keep_per_iter: int,
    criterion: str,
    task: str,
    chunk_size: int,
    n_jobs: int,
    random_state: int,
) -> np.ndarray:
    """Iterative SIS: screen residuals to capture features missed by marginal screening.

    Parameters
    ----------
    initial_selected : np.ndarray
        Feature indices from the SIS stage.

    Returns
    -------
    np.ndarray
        Union of initial_selected and features found in ISIS rounds.
    """
    selected = set(initial_selected.tolist())
    remaining = np.array([j for j in range(X.shape[1]) if j not in selected])

    for it in range(n_iterations):
        if len(remaining) == 0:
            break

        # Regress y on current selected features to get residuals
        X_sel = X[:, list(selected)]
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_sel)
        model = LassoCV(cv=3, random_state=random_state, n_jobs=n_jobs)
        model.fit(X_s, y)
        residuals = y - model.predict(X_s)

        # Screen remaining features against residuals
        new_idx, _ = _sis(
            X, residuals, keep_per_iter, criterion, task,
            chunk_size, n_jobs, random_state, candidate_indices=remaining
        )

        selected.update(new_idx.tolist())
        remaining = np.array([j for j in range(X.shape[1]) if j not in selected])
        logger.debug("ISIS iteration %d: total selected = %d", it + 1, len(selected))

    return np.array(sorted(selected))


# ---------------------------------------------------------------------------
# Stability selection
# ---------------------------------------------------------------------------
def _one_bootstrap_screen(
    X: np.ndarray,
    y: np.ndarray,
    keep: int,
    criterion: str,
    task: str,
    chunk_size: int,
    n_jobs: int,
    seed: int,
) -> np.ndarray:
    """One bootstrap draw: subsample half the data and run SIS."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(y), size=len(y) // 2, replace=False)
    selected, _ = _sis(X[idx], y[idx], keep, criterion, task, chunk_size, 1, seed)
    return selected


def _stability_screen(
    X: np.ndarray,
    y: np.ndarray,
    keep: int,
    n_bootstrap: int,
    threshold: float,
    criterion: str,
    task: str,
    chunk_size: int,
    n_jobs: int,
    random_state: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Run stability selection on top of SIS.

    Returns
    -------
    stable_indices : np.ndarray
        Feature indices with selection rate >= threshold.
    stability_df : pd.DataFrame
        Per-feature selection rates.
    """
    n_feat = X.shape[1]
    counts = np.zeros(n_feat, dtype=int)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_one_bootstrap_screen)(
            X, y, keep, criterion, task, chunk_size, 1, random_state + i
        )
        for i in range(n_bootstrap)
    )
    for selected in results:
        counts[selected] += 1

    rates = counts / n_bootstrap
    stability_df = pd.DataFrame({
        "feature_index": np.arange(n_feat),
        "selection_count": counts,
        "selection_rate": rates,
    }).sort_values("selection_rate", ascending=False).reset_index(drop=True)

    stable = np.where(rates >= threshold)[0]
    return stable, stability_df


# ---------------------------------------------------------------------------
# Post-selection inference: debiased Lasso
# ---------------------------------------------------------------------------
def _debiased_lasso_intervals(
    X: np.ndarray,
    y: np.ndarray,
    selected: np.ndarray,
    alpha_ci: float,
    random_state: int,
    lambda_ratio: float = DEBIASED_LAMBDA_RATIO,
) -> pd.DataFrame:
    """Compute debiased Lasso confidence intervals for selected features.

    Uses the one-step debiased estimator: beta_d = beta_lasso + Theta @ X.T(y - X @ beta_lasso) / n
    where Theta is an approximate inverse of X.T @ X / n.

    Parameters
    ----------
    selected : np.ndarray
        Indices of selected features (in original X space).
    alpha_ci : float
        Significance level (e.g. 0.05 for 95% CI).

    Returns
    -------
    pd.DataFrame with columns: feature_index, coef, se, ci_lower, ci_upper, z_score, p_value
    """
    if len(selected) == 0:
        return pd.DataFrame(columns=["feature_index", "coef", "se", "ci_lower", "ci_upper"])

    X_sel = X[:, selected]
    n, p = X_sel.shape
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_sel)

    # Fit Lasso with a gentle lambda
    lasso_cv = LassoCV(cv=3, random_state=random_state)
    lasso_cv.fit(Xs, y)
    lam = lasso_cv.alpha_ * lambda_ratio
    lasso = Lasso(alpha=lam, random_state=random_state)
    lasso.fit(Xs, y)
    beta = lasso.coef_

    # Debiased correction via nodewise regression (column-by-column)
    residuals_model = y - lasso.predict(Xs)
    n_feat = Xs.shape[1]
    debiased_coefs = np.zeros(n_feat)
    variances = np.zeros(n_feat)

    for j in range(n_feat):
        Xj = Xs[:, j]
        Xrest = np.delete(Xs, j, axis=1)
        node_model = LassoCV(cv=3, random_state=random_state)
        node_model.fit(Xrest, Xj)
        zj = Xj - node_model.predict(Xrest)
        zj_dot_xj = float(zj @ Xj)
        if abs(zj_dot_xj) < 1e-12:
            debiased_coefs[j] = beta[j]
            variances[j] = np.inf
            continue
        debiased_coefs[j] = beta[j] + float(zj @ residuals_model) / (n * zj_dot_xj / n)
        sigma2 = float(np.var(residuals_model))
        variances[j] = sigma2 * float((zj ** 2).sum()) / (n * (zj_dot_xj / n)) ** 2

    se = np.sqrt(np.maximum(variances, 0.0))
    z_crit = stats.norm.ppf(1 - alpha_ci / 2)
    z_scores = np.where(se > 0, debiased_coefs / se, 0.0)
    p_values = 2 * stats.norm.sf(np.abs(z_scores))

    # Back-scale from standardised to original units
    coef_orig = debiased_coefs / scaler.scale_
    se_orig = se / scaler.scale_

    df = pd.DataFrame({
        "feature_index": selected,
        "coef": coef_orig,
        "se": se_orig,
        "ci_lower": coef_orig - z_crit * se_orig,
        "ci_upper": coef_orig + z_crit * se_orig,
        "z_score": z_scores,
        "p_value": p_values,
    })
    return df


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class HighDimensionalScreener:
    """Feature screener for p >> n problems.

    Pipeline: SIS -> (ISIS) -> (stability selection) -> (debiased intervals).

    Parameters
    ----------
    config : ScreenerConfig

    Attributes
    ----------
    selected_features_ : np.ndarray
        Final selected feature indices.
    stability_df_ : pd.DataFrame
        Bootstrap selection rates (populated if stability_n_bootstrap > 0).
    confidence_intervals_ : pd.DataFrame
        Debiased Lasso intervals (populated if compute_intervals=True).
    sis_scores_ : np.ndarray
        Raw SIS screening scores for all features.
    stage_timings_ : dict[str, float]
        Wall-clock time per stage.
    """

    def __init__(self, config: ScreenerConfig) -> None:
        self.config = config
        self.selected_features_: np.ndarray = np.array([], dtype=int)
        self.stability_df_: Optional[pd.DataFrame] = None
        self.confidence_intervals_: Optional[pd.DataFrame] = None
        self.sis_scores_: Optional[np.ndarray] = None
        self.stage_timings_: Dict[str, float] = {}
        self._n_features_in: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HighDimensionalScreener":
        """Run the full screening pipeline.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Supports p > 10,000.
        y : np.ndarray of shape (n_samples,)

        Returns
        -------
        self
        """
        cfg = self.config
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        n, p = X.shape
        self._n_features_in = p

        logger.info(
            "HighDimensionalScreener: n=%d p=%d criterion=%s", n, p, cfg.criterion
        )

        # --- Stage 1: SIS -------------------------------------------
        t0 = time.perf_counter()
        sis_keep = min(cfg.sis_keep, p)
        sis_selected, sis_scores = _sis(
            X, y, sis_keep, cfg.criterion, cfg.task,
            cfg.chunk_size, cfg.n_jobs, cfg.random_state,
        )
        self.sis_scores_ = sis_scores
        self.stage_timings_["sis"] = time.perf_counter() - t0
        logger.info("SIS: kept %d / %d features (%.2fs)", len(sis_selected), p, self.stage_timings_["sis"])

        current_selected = sis_selected

        # --- Stage 2: ISIS ------------------------------------------
        if cfg.isis_iterations > 0:
            t0 = time.perf_counter()
            current_selected = _isis(
                X, y, current_selected,
                cfg.isis_iterations, cfg.isis_keep_per_iter,
                cfg.criterion, cfg.task,
                cfg.chunk_size, cfg.n_jobs, cfg.random_state,
            )
            self.stage_timings_["isis"] = time.perf_counter() - t0
            logger.info(
                "ISIS: expanded to %d features (%.2fs)",
                len(current_selected), self.stage_timings_["isis"],
            )

        # --- Stage 3: Stability selection ---------------------------
        if cfg.stability_n_bootstrap > 0:
            t0 = time.perf_counter()
            # Run stability selection only on the ISIS-reduced candidate set
            X_candidates = X[:, current_selected]
            stable_local, stab_df = _stability_screen(
                X_candidates, y,
                min(cfg.sis_keep, len(current_selected)),
                cfg.stability_n_bootstrap,
                cfg.stability_threshold,
                cfg.criterion, cfg.task,
                cfg.chunk_size, cfg.n_jobs, cfg.random_state,
            )
            # Re-map local indices back to original column space
            stab_df["feature_index"] = current_selected[stab_df["feature_index"].values]
            self.stability_df_ = stab_df
            current_selected = current_selected[stable_local]
            self.stage_timings_["stability"] = time.perf_counter() - t0
            logger.info(
                "Stability: %d stable features (threshold=%.2f, %.2fs)",
                len(current_selected), cfg.stability_threshold, self.stage_timings_["stability"],
            )

        self.selected_features_ = current_selected

        # --- Stage 4: Post-selection inference ----------------------
        if cfg.compute_intervals and len(current_selected) > 0:
            t0 = time.perf_counter()
            self.confidence_intervals_ = _debiased_lasso_intervals(
                X, y, current_selected, cfg.alpha_ci, cfg.random_state
            )
            self.stage_timings_["debiased_ci"] = time.perf_counter() - t0
            logger.info(
                "Debiased CIs computed for %d features (%.2fs)",
                len(current_selected), self.stage_timings_["debiased_ci"],
            )

        logger.info(
            "Screening complete. Final selection: %d features. Total time: %.2fs",
            len(self.selected_features_),
            sum(self.stage_timings_.values()),
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Reduce X to the selected columns.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples, n_selected)
        """
        if not hasattr(self, '_n_features_in') or self._n_features_in == 0:
            raise RuntimeError("Call fit() first.")
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self._n_features_in:
            raise ValueError(f"Expected {self._n_features_in} features, got {X.shape[1]}")
        return X[:, self.selected_features_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def summary(self) -> Dict[str, object]:
        """Return a summary dict of the screening run."""
        return {
            "n_features_in": self._n_features_in,
            "n_features_selected": len(self.selected_features_),
            "selected_indices": self.selected_features_.tolist(),
            "stage_timings_seconds": self.stage_timings_,
            "criterion": self.config.criterion,
        }

    def top_features(self, n: int = 20) -> pd.DataFrame:
        """Return top-n features by SIS score.

        Parameters
        ----------
        n : int

        Returns
        -------
        pd.DataFrame with columns: feature_index, sis_score, selected
        """
        if self.sis_scores_ is None:
            raise RuntimeError("Call fit() first.")
        top_idx = np.argsort(self.sis_scores_)[::-1][:n]
        return pd.DataFrame({
            "feature_index": top_idx,
            "sis_score": self.sis_scores_[top_idx],
            "selected": np.isin(top_idx, self.selected_features_),
        })


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    rng = np.random.default_rng(42)
    n, p = 200, 2000      # p >> n regime
    X = rng.standard_normal((n, p))
    true_coef = np.zeros(p)
    true_coef[[0, 5, 10, 50, 100]] = [2, -1.5, 1, -2, 1.8]
    y = X @ true_coef + rng.standard_normal(n) * 0.5

    cfg = ScreenerConfig(
        criterion="correlation",
        sis_keep=50,
        task="regression",
        isis_iterations=2,
        isis_keep_per_iter=20,
        stability_n_bootstrap=30,   # small for demo
        stability_threshold=0.5,
        compute_intervals=True,
        n_jobs=2,
    )
    screener = HighDimensionalScreener(cfg)
    screener.fit(X, y)

    print("\nSummary:", screener.summary())
    print("\nTop 10 by SIS score:")
    print(screener.top_features(10).to_string(index=False))
    if screener.confidence_intervals_ is not None:
        print("\nDebiased CIs:")
        print(screener.confidence_intervals_.to_string(index=False))
