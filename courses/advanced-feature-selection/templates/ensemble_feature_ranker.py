"""
ensemble_feature_ranker.py
--------------------------
Multi-method consensus feature ranking with stability metrics.

Aggregation methods
-------------------
- borda        : Borda count (position-based voting)
- rank_average : Mean rank across all methods
- weighted     : Weighted rank average (caller supplies weights)
- kemeny       : Kemeny optimal (minimise Kendall distance to consensus)

Stability metrics
-----------------
- Kuncheva's index (KI) : normalised index comparing selected sets
- Jaccard similarity    : intersection-over-union across bootstrap folds

Usage
-----
    from ensemble_feature_ranker import RankerConfig, EnsembleFeatureRanker

    cfg = RankerConfig(
        methods=["mutual_info", "lasso", "random_forest", "spearman"],
        aggregation="borda",
        n_features_to_select=20,
        bootstrap_n=50,
    )
    ranker = EnsembleFeatureRanker(cfg)
    ranker.fit(X_train, y_train)

    print(ranker.ranking_df_)          # ranked table with confidence intervals
    print(ranker.stability_metrics_)   # Kuncheva index and Jaccard
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_METHODS = frozenset({"mutual_info", "lasso", "random_forest", "spearman"})
VALID_AGGREGATIONS = frozenset({"borda", "rank_average", "weighted", "kemeny"})
KEMENY_MAX_FEATURES = 50  # Kemeny is O(p^2); cap for tractability


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class RankerConfig:
    """Configuration for the ensemble feature ranker.

    Parameters
    ----------
    methods : list[str]
        Selection methods to combine. Subset of
        ``{"mutual_info", "lasso", "random_forest", "spearman"}``.
    aggregation : str
        Consensus strategy: ``"borda"``, ``"rank_average"``,
        ``"weighted"``, ``"kemeny"``.
    method_weights : list[float], optional
        Weights for ``"weighted"`` aggregation (same order as ``methods``).
        If None, equal weights are used.
    n_features_to_select : int
        Number of top features to consider selected.
    task : str
        ``"classification"`` or ``"regression"``.
    bootstrap_n : int
        Bootstrap draws for confidence interval estimation. 0 = skip.
    bootstrap_frac : float
        Fraction of samples per bootstrap draw (0 < frac <= 1).
    n_jobs : int
        Joblib parallelism.
    random_state : int
        Reproducibility seed.
    rf_n_estimators : int
        Trees in Random Forest method.
    lasso_cv_folds : int
        CV folds for LassoCV.
    """

    methods: List[str] = field(default_factory=lambda: ["mutual_info", "lasso", "random_forest"])
    aggregation: str = "borda"
    method_weights: Optional[List[float]] = None
    n_features_to_select: int = 20
    task: str = "classification"
    bootstrap_n: int = 50
    bootstrap_frac: float = 0.8
    n_jobs: int = -1
    random_state: int = 42
    rf_n_estimators: int = 100
    lasso_cv_folds: int = 5

    def __post_init__(self) -> None:
        unknown = set(self.methods) - VALID_METHODS
        if unknown:
            raise ValueError(f"Unknown methods: {unknown}. Valid: {VALID_METHODS}")
        if self.aggregation not in VALID_AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {VALID_AGGREGATIONS}")
        if self.task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
        if self.method_weights is not None:
            if len(self.method_weights) != len(self.methods):
                raise ValueError("method_weights length must match methods length")
        if not (0 < self.bootstrap_frac <= 1):
            raise ValueError("bootstrap_frac must be in (0, 1]")


# ---------------------------------------------------------------------------
# Individual method scorers
# ---------------------------------------------------------------------------
def _score_mutual_info(
    X: np.ndarray, y: np.ndarray, task: str, seed: int
) -> np.ndarray:
    scorer = mutual_info_classif if task == "classification" else mutual_info_regression
    return scorer(X, y, random_state=seed)


def _score_lasso(
    X: np.ndarray, y: np.ndarray, task: str, cv_folds: int, seed: int, n_jobs: int
) -> np.ndarray:
    scaler = StandardScaler()
    xs = scaler.fit_transform(X)
    model = LassoCV(cv=cv_folds, random_state=seed, n_jobs=n_jobs)
    model.fit(xs, y)
    return np.abs(model.coef_)


def _score_random_forest(
    X: np.ndarray, y: np.ndarray, task: str, n_estimators: int, seed: int, n_jobs: int
) -> np.ndarray:
    est_class = RandomForestClassifier if task == "classification" else RandomForestRegressor
    rf = est_class(n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs)
    rf.fit(X, y)
    return rf.feature_importances_


def _score_spearman(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Absolute Spearman rank correlation with y."""
    n_feat = X.shape[1]
    scores = np.zeros(n_feat)
    y_rank = stats.rankdata(y)
    for j in range(n_feat):
        x_rank = stats.rankdata(X[:, j])
        corr, _ = stats.spearmanr(x_rank, y_rank)
        scores[j] = abs(corr)
    return scores


def _run_method(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    cfg: RankerConfig,
) -> np.ndarray:
    """Dispatch to one scoring method; return raw importance scores."""
    if method == "mutual_info":
        return _score_mutual_info(X, y, cfg.task, cfg.random_state)
    if method == "lasso":
        return _score_lasso(X, y, cfg.task, cfg.lasso_cv_folds, cfg.random_state, cfg.n_jobs)
    if method == "random_forest":
        return _score_random_forest(X, y, cfg.task, cfg.rf_n_estimators, cfg.random_state, cfg.n_jobs)
    if method == "spearman":
        return _score_spearman(X, y)
    raise ValueError(f"Unknown method: {method}")


def _scores_to_ranks(scores: np.ndarray) -> np.ndarray:
    """Convert importance scores to rank array (rank 1 = best)."""
    order = np.argsort(scores)[::-1]
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


# ---------------------------------------------------------------------------
# Aggregation methods
# ---------------------------------------------------------------------------
def _borda_aggregate(rank_matrix: np.ndarray) -> np.ndarray:
    """Borda count: lower rank -> more points.

    Parameters
    ----------
    rank_matrix : np.ndarray of shape (n_methods, n_features)
        Each row is a rank vector for one method.

    Returns
    -------
    np.ndarray of shape (n_features,)
        Borda scores (higher = better).
    """
    n_feat = rank_matrix.shape[1]
    # Points = n_feat - rank (rank 1 gets n_feat-1 points)
    borda = (n_feat - rank_matrix).sum(axis=0)
    return borda.astype(float)


def _rank_average_aggregate(rank_matrix: np.ndarray) -> np.ndarray:
    """Mean rank across methods (lower = better; invert for consistency)."""
    mean_ranks = rank_matrix.mean(axis=0)
    return -mean_ranks  # negate so higher is better


def _weighted_aggregate(rank_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted rank average."""
    w = weights / weights.sum()
    weighted_ranks = (rank_matrix * w[:, None]).sum(axis=0)
    return -weighted_ranks  # negate: lower weighted rank is better


def _kemeny_aggregate(rank_matrix: np.ndarray, max_features: int) -> np.ndarray:
    """Approximate Kemeny optimal ranking via pairwise preference matrix.

    Computes the number of methods that prefer feature i over feature j
    for each pair, then scores features by their total pairwise wins.

    Capped at max_features for tractability.
    """
    n_methods, n_feat = rank_matrix.shape
    cap = min(n_feat, max_features)

    # Work on a truncated subset for efficiency
    avg_rank = rank_matrix.mean(axis=0)
    top_idx = np.argsort(avg_rank)[:cap]
    sub_ranks = rank_matrix[:, top_idx]  # (n_methods, cap)

    pairwise_wins = np.zeros(cap, dtype=float)
    for i in range(cap):
        for j in range(cap):
            if i == j:
                continue
            wins = int((sub_ranks[:, i] < sub_ranks[:, j]).sum())
            pairwise_wins[i] += wins

    full_score = -avg_rank.copy()  # default for features outside top-cap
    full_score[top_idx] = pairwise_wins
    return full_score


def _aggregate_ranks(
    rank_matrix: np.ndarray,
    aggregation: str,
    weights: Optional[np.ndarray],
) -> np.ndarray:
    """Dispatch to aggregation strategy; return consensus score (higher = better)."""
    if aggregation == "borda":
        return _borda_aggregate(rank_matrix)
    if aggregation == "rank_average":
        return _rank_average_aggregate(rank_matrix)
    if aggregation == "weighted":
        w = weights if weights is not None else np.ones(rank_matrix.shape[0])
        return _weighted_aggregate(rank_matrix, w)
    if aggregation == "kemeny":
        return _kemeny_aggregate(rank_matrix, KEMENY_MAX_FEATURES)
    raise ValueError(f"Unknown aggregation: {aggregation}")


# ---------------------------------------------------------------------------
# Stability metrics
# ---------------------------------------------------------------------------
def kuncheva_index(
    selected_a: List[int],
    selected_b: List[int],
    n_total: int,
) -> float:
    """Kuncheva's Index between two selected subsets.

    KI = (|A ∩ B| - k²/n) / (k - k²/n)
    where k = |A| = |B| (assumed equal), n = total features.

    Returns NaN if k == 0 or k == n.
    """
    k = len(selected_a)
    if k == 0 or k == n_total:
        return float("nan")
    intersection = len(set(selected_a) & set(selected_b))
    expected = k * k / n_total
    denom = k - expected
    if abs(denom) < 1e-12:
        return float("nan")
    return (intersection - expected) / denom


def jaccard_similarity(selected_a: List[int], selected_b: List[int]) -> float:
    """Jaccard similarity between two feature sets."""
    sa, sb = set(selected_a), set(selected_b)
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


def _pairwise_stability(
    bootstrap_selections: List[List[int]], n_total: int
) -> Dict[str, float]:
    """Compute mean KI and Jaccard across all bootstrap pairs."""
    kis, jaccards = [], []
    for a, b in itertools.combinations(bootstrap_selections, 2):
        kis.append(kuncheva_index(a, b, n_total))
        jaccards.append(jaccard_similarity(a, b))
    valid_ki = [v for v in kis if not np.isnan(v)]
    return {
        "mean_kuncheva_index": float(np.mean(valid_ki)) if valid_ki else float("nan"),
        "mean_jaccard": float(np.mean(jaccards)) if jaccards else float("nan"),
        "n_pairs": len(kis),
    }


# ---------------------------------------------------------------------------
# Bootstrap CI helpers
# ---------------------------------------------------------------------------
def _one_bootstrap_rank(
    X: np.ndarray,
    y: np.ndarray,
    cfg: RankerConfig,
    seed: int,
) -> Tuple[np.ndarray, List[int]]:
    """One bootstrap draw: return consensus scores and top-k selection."""
    rng = np.random.RandomState(seed)
    n = len(y)
    idx = rng.choice(n, size=int(n * cfg.bootstrap_frac), replace=True)
    X_b, y_b = X[idx], y[idx]

    rank_matrix = np.stack([
        _scores_to_ranks(_run_method(m, X_b, y_b, cfg))
        for m in cfg.methods
    ])
    w = np.array(cfg.method_weights) if cfg.method_weights else None
    consensus_scores = _aggregate_ranks(rank_matrix, cfg.aggregation, w)
    top_k = list(np.argsort(consensus_scores)[::-1][:cfg.n_features_to_select])
    return consensus_scores, top_k


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class EnsembleFeatureRanker:
    """Multi-method consensus feature ranking.

    Parameters
    ----------
    config : RankerConfig

    Attributes
    ----------
    ranking_df_ : pd.DataFrame
        Full ranking with consensus scores and bootstrap CIs.
    method_rank_matrix_ : np.ndarray of shape (n_methods, n_features)
        Raw rank from each individual method.
    method_scores_ : dict[str, np.ndarray]
        Raw importance scores per method.
    selected_features_ : list[int]
        Indices of top-k features by consensus.
    stability_metrics_ : dict[str, float]
        Mean KI and Jaccard (populated when bootstrap_n > 0).
    """

    def __init__(self, config: RankerConfig) -> None:
        self.config = config
        self.ranking_df_: Optional[pd.DataFrame] = None
        self.method_rank_matrix_: Optional[np.ndarray] = None
        self.method_scores_: Dict[str, np.ndarray] = {}
        self.selected_features_: List[int] = []
        self.stability_metrics_: Dict[str, float] = {}
        self._n_features: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "EnsembleFeatureRanker":
        """Fit all methods and compute consensus ranking.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)
        feature_names : list[str], optional

        Returns
        -------
        self
        """
        cfg = self.config
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        n_samples, n_features = X.shape
        self._n_features = n_features

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        logger.info(
            "EnsembleFeatureRanker: n=%d p=%d methods=%s aggregation=%s",
            n_samples, n_features, cfg.methods, cfg.aggregation,
        )

        # --- Compute per-method scores on full data ---
        for method in cfg.methods:
            scores = _run_method(method, X, y, cfg)
            self.method_scores_[method] = scores
            logger.debug("Method %s complete.", method)

        # --- Build rank matrix ---
        rank_matrix = np.stack([
            _scores_to_ranks(self.method_scores_[m]) for m in cfg.methods
        ])
        self.method_rank_matrix_ = rank_matrix

        # --- Aggregate ---
        w = np.array(cfg.method_weights) if cfg.method_weights else None
        consensus_scores = _aggregate_ranks(rank_matrix, cfg.aggregation, w)
        consensus_ranks = _scores_to_ranks(consensus_scores)

        self.selected_features_ = list(
            np.argsort(consensus_scores)[::-1][:cfg.n_features_to_select]
        )

        # --- Bootstrap CIs ---
        ci_lower = np.full(n_features, np.nan)
        ci_upper = np.full(n_features, np.nan)
        bootstrap_selections: List[List[int]] = []

        if cfg.bootstrap_n > 0:
            logger.info("Running %d bootstrap draws...", cfg.bootstrap_n)
            results = Parallel(n_jobs=cfg.n_jobs)(
                delayed(_one_bootstrap_rank)(X, y, cfg, cfg.random_state + i)
                for i in range(cfg.bootstrap_n)
            )
            boot_scores = np.stack([r[0] for r in results])  # (n_bootstrap, n_features)
            bootstrap_selections = [r[1] for r in results]

            ci_lower = np.percentile(boot_scores, 2.5, axis=0)
            ci_upper = np.percentile(boot_scores, 97.5, axis=0)

            self.stability_metrics_ = _pairwise_stability(bootstrap_selections, n_features)
            logger.info("Stability: KI=%.4f Jaccard=%.4f",
                        self.stability_metrics_.get("mean_kuncheva_index", float("nan")),
                        self.stability_metrics_.get("mean_jaccard", float("nan")))

        # --- Build ranking DataFrame ---
        method_rank_cols = {
            f"rank_{m}": rank_matrix[i] for i, m in enumerate(cfg.methods)
        }
        method_score_cols = {
            f"score_{m}": self.method_scores_[m] for m in cfg.methods
        }

        df = pd.DataFrame({
            "feature_name": feature_names,
            "consensus_rank": consensus_ranks,
            "consensus_score": consensus_scores,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "selected": np.arange(n_features, dtype=int),
            **method_rank_cols,
            **method_score_cols,
        })
        df["selected"] = df.index.isin(self.selected_features_)
        df = df.sort_values("consensus_rank").reset_index(drop=True)
        df.index.name = "original_index"
        self.ranking_df_ = df

        logger.info("Ranking complete. Top feature: %s", df["feature_name"].iloc[0])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Reduce X to the top-k consensus features.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples, n_features_to_select)
        """
        if not self.selected_features_:
            raise RuntimeError("Call fit() first.")
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self._n_features:
            raise ValueError(f"Expected {self._n_features} features, got {X.shape[1]}")
        return X[:, self.selected_features_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def export(self, path: str) -> None:
        """Save the ranking DataFrame to CSV.

        Parameters
        ----------
        path : str
            Destination CSV file.
        """
        if self.ranking_df_ is None:
            raise RuntimeError("Call fit() first.")
        self.ranking_df_.to_csv(path)
        logger.info("Ranking exported to %s", path)

    def agreement_matrix(self) -> pd.DataFrame:
        """Pairwise rank agreement (Spearman correlation) between methods.

        Returns
        -------
        pd.DataFrame of shape (n_methods, n_methods)
        """
        if self.method_rank_matrix_ is None:
            raise RuntimeError("Call fit() first.")
        n_methods = len(self.config.methods)
        mat = np.zeros((n_methods, n_methods))
        ranks = self.method_rank_matrix_
        for i in range(n_methods):
            for j in range(n_methods):
                corr, _ = stats.spearmanr(ranks[i], ranks[j])
                mat[i, j] = corr
        return pd.DataFrame(mat, index=self.config.methods, columns=self.config.methods)

    def rank_distribution(self) -> pd.DataFrame:
        """Per-feature rank statistics across bootstrap draws.

        Returns empty DataFrame if bootstrap_n == 0.
        """
        if self.ranking_df_ is None:
            raise RuntimeError("Call fit() first.")
        if all(np.isnan(self.ranking_df_["ci_lower"])):
            return pd.DataFrame()
        return self.ranking_df_[["feature_name", "consensus_rank", "ci_lower", "ci_upper", "selected"]].copy()

    def summary(self) -> Dict[str, object]:
        """Return a plain dict summary."""
        return {
            "n_features_in": self._n_features,
            "n_features_selected": len(self.selected_features_),
            "methods": self.config.methods,
            "aggregation": self.config.aggregation,
            "stability_metrics": self.stability_metrics_,
            "top_10_features": (
                self.ranking_df_["feature_name"].head(10).tolist()
                if self.ranking_df_ is not None else []
            ),
        }


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=400, n_features=30, n_informative=8,
        n_redundant=4, random_state=0,
    )
    feature_names = [f"feat_{i:02d}" for i in range(X.shape[1])]

    cfg = RankerConfig(
        methods=["mutual_info", "lasso", "random_forest", "spearman"],
        aggregation="borda",
        n_features_to_select=10,
        task="classification",
        bootstrap_n=20,         # small for demo
        bootstrap_frac=0.8,
        n_jobs=2,
    )

    ranker = EnsembleFeatureRanker(cfg)
    ranker.fit(X, y, feature_names=feature_names)

    print("\n--- Top 10 features ---")
    print(ranker.ranking_df_[["feature_name", "consensus_rank", "ci_lower", "ci_upper", "selected"]].head(10).to_string(index=False))

    print("\n--- Stability metrics ---")
    print(ranker.stability_metrics_)

    print("\n--- Method agreement matrix ---")
    print(ranker.agreement_matrix().round(3).to_string())

    print("\n--- Summary ---")
    print(ranker.summary())
