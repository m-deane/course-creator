"""
meta_learning_recommender.py
-----------------------------
Recommend the best feature selection method for a given dataset based on
computed meta-features.

Meta-features extracted:
  - n_samples, n_features, aspect_ratio (n/p)
  - feature_correlation: mean absolute pairwise Pearson correlation
  - class_imbalance: normalised entropy of class distribution
  - noise_level: estimated via mean feature-target MI entropy ratio
  - intrinsic_dimensionality: fraction of PCA components for 95% variance

Rule-based recommender decision logic (grounded in published benchmarks):
  - Low n, high p, high correlation → Stability Selection / Lasso
  - High n, moderate p, low correlation → mRMR or Random Forest importance
  - Time-series structure (sequential) → Granger Causality
  - Need FDR control → Knockoffs
  - High n, many redundant features → mRMR
  - Multi-objective / embedded → NSGA-II if p > 50
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_wine,
    make_classification,
    make_regression,
)
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import entropy


# ---------------------------------------------------------------------------
# Meta-feature extraction
# ---------------------------------------------------------------------------

def extract_meta_features(
    X: np.ndarray,
    y: np.ndarray,
    task: str = "classification",
) -> dict[str, float]:
    """Compute a meta-feature vector describing dataset characteristics.

    Parameters
    ----------
    X : feature matrix (n_samples, n_features)
    y : target vector
    task : 'classification' or 'regression'

    Returns
    -------
    meta : dict of meta-feature name → float value
    """
    n, p = X.shape
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Basic shape
    aspect_ratio = n / max(p, 1)

    # Feature correlation (mean absolute pairwise Pearson)
    if p > 1:
        corr_matrix = np.corrcoef(X_scaled, rowvar=False)
        upper_tri = corr_matrix[np.triu_indices(p, k=1)]
        mean_abs_corr = float(np.mean(np.abs(upper_tri)))
    else:
        mean_abs_corr = 0.0

    # Class imbalance (normalised entropy for classification, 1.0 for regression)
    if task == "classification":
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        max_entropy = np.log(len(classes)) if len(classes) > 1 else 1.0
        class_balance = float(entropy(probs) / max_entropy)
        n_classes = len(classes)
    else:
        class_balance = 1.0
        n_classes = 0

    # Noise level proxy: 1 - max normalised MI over all features
    if task == "classification":
        mi = mutual_info_classif(X_scaled, y, random_state=42)
    else:
        mi = mutual_info_regression(X_scaled, y, random_state=42)
    max_mi = mi.max() if mi.max() > 0 else 1e-9
    noise_level = float(1.0 - max_mi / (max_mi + 1.0))  # maps to [0, 1]

    # Intrinsic dimensionality: fraction of features needed for 95% variance
    pca = PCA(n_components=min(n, p), random_state=42)
    pca.fit(X_scaled)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_components_95 = int(np.searchsorted(cumvar, 0.95)) + 1
    intrinsic_dim_ratio = float(n_components_95 / p)

    return {
        "n_samples": float(n),
        "n_features": float(p),
        "aspect_ratio": aspect_ratio,          # n/p — high = data-rich
        "mean_abs_correlation": mean_abs_corr, # 0=independent, 1=fully collinear
        "class_balance": class_balance,        # 0=imbalanced, 1=balanced
        "noise_level": noise_level,            # 0=informative, 1=noisy
        "intrinsic_dim_ratio": intrinsic_dim_ratio,  # fraction of features useful
        "n_classes": float(n_classes),
    }


# ---------------------------------------------------------------------------
# Rule-based recommender
# ---------------------------------------------------------------------------

METHODS = {
    "stability_selection": "Stability Selection (Meinshausen-Bühlmann)",
    "mrmr": "Minimum Redundancy Maximum Relevance (mRMR)",
    "knockoffs": "Model-X Knockoff Filter (FDR-controlled)",
    "lasso_cv": "Lasso with cross-validated regularisation",
    "random_forest_importance": "Random Forest feature importance",
    "granger_causality": "Granger Causality (time-series only)",
    "nsga2": "NSGA-II Multi-objective Selection",
    "rfe": "Recursive Feature Elimination (RFE)",
}


def recommend_method(
    meta: dict[str, float],
    fdr_control_required: bool = False,
    time_series: bool = False,
    multi_objective: bool = False,
) -> tuple[str, str, list[str]]:
    """Apply decision rules to recommend a feature selection method.

    Parameters
    ----------
    meta : dict from extract_meta_features
    fdr_control_required : caller needs statistical FDR guarantee
    time_series : data is sequential (time series)
    multi_objective : caller wants accuracy-vs-count tradeoff

    Returns
    -------
    primary_method : recommended method key
    rationale : explanation string
    alternatives : list of alternative method keys
    """
    n = meta["n_samples"]
    p = meta["n_features"]
    aspect = meta["aspect_ratio"]
    corr = meta["mean_abs_correlation"]
    noise = meta["noise_level"]
    idr = meta["intrinsic_dim_ratio"]

    # Hard constraints override heuristics
    if time_series:
        return (
            "granger_causality",
            "Time-series data: Granger causality captures temporal dependencies.",
            ["lasso_cv", "stability_selection"],
        )

    if fdr_control_required:
        return (
            "knockoffs",
            "Statistical FDR control required: knockoff filter gives guaranteed FDR bound.",
            ["stability_selection", "lasso_cv"],
        )

    if multi_objective:
        return (
            "nsga2",
            "Multi-objective trade-off requested: NSGA-II explores accuracy vs. sparsity Pareto front.",
            ["mrmr", "random_forest_importance"],
        )

    # Heuristic rules
    # Very high dimensional, small sample → regularised / stability
    if aspect < 1.0:   # p >> n
        return (
            "stability_selection",
            f"High-dimensional (n/p = {aspect:.2f}): stability selection controls false positives in p >> n regime.",
            ["lasso_cv", "knockoffs"],
        )

    # High collinearity → mRMR to filter redundant features
    if corr > 0.5 and idr < 0.4:
        return (
            "mrmr",
            f"High inter-feature correlation ({corr:.2f}) with low intrinsic dimensionality: "
            "mRMR explicitly removes redundancy.",
            ["stability_selection", "lasso_cv"],
        )

    # Data-rich, moderate dimensionality → Random Forest importance (fast, non-parametric)
    if aspect > 20 and p <= 100:
        return (
            "random_forest_importance",
            f"Large sample size (n/p = {aspect:.1f}), moderate p: random forest importance is "
            "efficient and handles non-linearities.",
            ["mrmr", "rfe"],
        )

    # High noise, moderate n/p → Lasso (promotes zero coefficients)
    if noise > 0.6:
        return (
            "lasso_cv",
            f"High noise level ({noise:.2f}): Lasso CV aggressively zeros irrelevant features.",
            ["stability_selection", "random_forest_importance"],
        )

    # Default: mRMR is robust across many settings
    return (
        "mrmr",
        "General case: mRMR balances relevance and redundancy without distributional assumptions.",
        ["random_forest_importance", "lasso_cv"],
    )


# ---------------------------------------------------------------------------
# Demo — apply to multiple datasets
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    datasets = [
        ("Breast Cancer (classification)", *load_breast_cancer(return_X_y=True), "classification", {}),
        ("Diabetes (regression)", *load_diabetes(return_X_y=True), "regression", {}),
        ("Wine (multi-class)", *load_wine(return_X_y=True), "classification", {}),
        (
            "High-dim synthetic (p>>n)",
            *make_classification(n_samples=80, n_features=200, n_informative=10,
                                  n_redundant=40, random_state=42),
            "classification",
            {},
        ),
        (
            "Noisy regression",
            *make_regression(n_samples=300, n_features=30, n_informative=5,
                              noise=50, random_state=42),
            "regression",
            {},
        ),
    ]

    print("=" * 70)
    print("Meta-Learning Feature Selection Recommender")
    print("=" * 70)

    for dataset_name, X, y, task, flags in datasets:
        meta = extract_meta_features(X, y, task=task)
        primary, rationale, alts = recommend_method(meta, **flags)

        print(f"\nDataset : {dataset_name}")
        print(f"  n={int(meta['n_samples'])}, p={int(meta['n_features'])}, "
              f"n/p={meta['aspect_ratio']:.1f}, "
              f"corr={meta['mean_abs_correlation']:.2f}, "
              f"noise={meta['noise_level']:.2f}, "
              f"idr={meta['intrinsic_dim_ratio']:.2f}")
        print(f"  Recommendation : {METHODS[primary]}")
        print(f"  Rationale      : {rationale}")
        print(f"  Alternatives   : {[METHODS[a] for a in alts]}")

    # FDR control example
    print("\n" + "-" * 70)
    print("Scenario: FDR control required")
    X_fdr, y_fdr = load_breast_cancer(return_X_y=True)
    meta_fdr = extract_meta_features(X_fdr, y_fdr, task="classification")
    primary, rationale, alts = recommend_method(meta_fdr, fdr_control_required=True)
    print(f"  Recommendation : {METHODS[primary]}")
    print(f"  Rationale      : {rationale}")
