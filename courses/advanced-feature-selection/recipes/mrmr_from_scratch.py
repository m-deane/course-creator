"""
mrmr_from_scratch.py
--------------------
Minimum Redundancy Maximum Relevance (mRMR) feature selection.

Two variants:
  - MID: maximise I(xi; y) - mean(I(xi; xj))          (difference)
  - MIQ: maximise I(xi; y) / mean(I(xi; xj))           (quotient)

Reference: Peng et al. 2005, "Feature Selection Based on Mutual Information".
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer


# ---------------------------------------------------------------------------
# Mutual information helpers
# ---------------------------------------------------------------------------

def _compute_relevance(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Return I(xi; y) for every feature column using sklearn's estimator."""
    return mutual_info_classif(X, y, discrete_features=False, random_state=random_state)


def _compute_redundancy(X: np.ndarray, selected_idx: list[int], candidate_idx: int,
                        random_state: int = 42) -> float:
    """Return mean I(xi; xj) between candidate and all already-selected features.

    Mutual information between two continuous variables is estimated by
    treating the second variable as a pseudo-target.
    """
    if not selected_idx:
        return 0.0

    mi_values = []
    for s_idx in selected_idx:
        # mutual_info_classif treats the second arg as discrete target;
        # discretise it to approximate I(xi_candidate ; xi_selected).
        discretiser = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
        y_proxy = discretiser.fit_transform(X[:, s_idx].reshape(-1, 1)).ravel().astype(int)
        mi = mutual_info_classif(
            X[:, candidate_idx].reshape(-1, 1),
            y_proxy,
            discrete_features=False,
            random_state=random_state,
        )[0]
        mi_values.append(mi)

    return float(np.mean(mi_values))


# ---------------------------------------------------------------------------
# mRMR selector
# ---------------------------------------------------------------------------

def mrmr_select(
    X: np.ndarray,
    y: np.ndarray,
    n_features: int = 10,
    variant: str = "MID",
    random_state: int = 42,
) -> list[int]:
    """Select features via mRMR greedy forward search.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    y : array of shape (n_samples,) — class labels
    n_features : number of features to select
    variant : "MID" (difference) or "MIQ" (quotient)
    random_state : reproducibility seed

    Returns
    -------
    selected : list of column indices in selection order
    """
    if variant not in ("MID", "MIQ"):
        raise ValueError("variant must be 'MID' or 'MIQ'")

    n_total = X.shape[1]
    n_features = min(n_features, n_total)

    relevance = _compute_relevance(X, y, random_state=random_state)
    selected: list[int] = []
    remaining = list(range(n_total))

    for step in range(n_features):
        scores = {}
        for cand in remaining:
            rel = relevance[cand]
            red = _compute_redundancy(X, selected, cand, random_state=random_state)

            if variant == "MID":
                scores[cand] = rel - red
            else:
                # Avoid division by zero; add small epsilon
                scores[cand] = rel / (red + 1e-9)

        best = max(scores, key=scores.__getitem__)
        selected.append(best)
        remaining.remove(best)
        print(f"  step {step + 1:>2}: selected feature {best:>3}  "
              f"(score={scores[best]:.4f}, relevance={relevance[best]:.4f})")

    return selected


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Load breast cancer — 30 features, binary classification
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = np.array(data.feature_names)

    print("=" * 60)
    print("mRMR — MID variant (top 10 features)")
    print("=" * 60)
    selected_mid = mrmr_select(X, y, n_features=10, variant="MID")

    print("\n" + "=" * 60)
    print("mRMR — MIQ variant (top 10 features)")
    print("=" * 60)
    selected_miq = mrmr_select(X, y, n_features=10, variant="MIQ")

    # Summary table
    results = pd.DataFrame({
        "rank": range(1, 11),
        "MID_feature": feature_names[selected_mid],
        "MIQ_feature": feature_names[selected_miq],
    })
    print("\nSide-by-side ranking:")
    print(results.to_string(index=False))
