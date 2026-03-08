"""
stability_selection.py
----------------------
Meinshausen-Bühlmann stability selection with randomised Lasso.

Algorithm:
  1. Repeat B times:
       a. Draw a 50% subsample (without replacement).
       b. Fit Lasso at each lambda on the subsample with random feature
          scaling (multiplied by uniform weights in [0.5, 1]).
       c. Record which features have non-zero coefficients.
  2. Compute selection probability P_hat(j, lambda) for each feature.
  3. Stable features: max over lambda of P_hat(j, lambda) > threshold.

Reference: Meinshausen & Bühlmann 2010, "Stability Selection".
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Core stability selection
# ---------------------------------------------------------------------------

def stability_selection(
    X: np.ndarray,
    y: np.ndarray,
    n_bootstraps: int = 100,
    alpha_min: float = 0.01,
    alpha_max: float = 1.0,
    n_alphas: int = 30,
    threshold: float = 0.6,
    random_weight_low: float = 0.5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run stability selection.

    Parameters
    ----------
    X : standardised feature matrix (n_samples, n_features)
    y : target vector (n_samples,)
    n_bootstraps : number of subsampling iterations
    alpha_min, alpha_max : regularisation range for Lasso
    n_alphas : number of lambda values in the grid
    threshold : stability probability cutoff (default 0.6)
    random_weight_low : lower bound for random feature scaling weight
    random_state : seed

    Returns
    -------
    stability_scores : max selection probability per feature  (n_features,)
    selection_matrix : P_hat[feature, alpha_idx]             (n_features, n_alphas)
    alphas : regularisation values used                      (n_alphas,)
    """
    rng = np.random.default_rng(random_state)
    n_samples, n_features = X.shape
    half_n = n_samples // 2
    alphas = np.logspace(np.log10(alpha_max), np.log10(alpha_min), n_alphas)

    # selection_counts[feature, alpha_idx] counts how many times selected
    selection_counts = np.zeros((n_features, n_alphas), dtype=float)

    for b in range(n_bootstraps):
        # 50% subsample without replacement
        idx = rng.choice(n_samples, size=half_n, replace=False)
        X_sub, y_sub = X[idx], y[idx]

        # Randomised feature scaling: divide columns by random weights in [low, 1]
        weights = rng.uniform(random_weight_low, 1.0, size=n_features)
        X_sub_scaled = X_sub / weights  # equivalent to amplifying penalty per feature

        for a_idx, alpha in enumerate(alphas):
            lasso = Lasso(alpha=alpha, max_iter=2000, fit_intercept=True)
            lasso.fit(X_sub_scaled, y_sub)
            selected = lasso.coef_ != 0
            selection_counts[:, a_idx] += selected

    selection_matrix = selection_counts / n_bootstraps          # probabilities
    stability_scores = selection_matrix.max(axis=1)             # max over lambda
    return stability_scores, selection_matrix, alphas


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_stability_paths(
    selection_matrix: np.ndarray,
    alphas: np.ndarray,
    feature_names: list[str],
    threshold: float = 0.6,
    top_n: int = 10,
    save_path: str | None = None,
) -> None:
    """Plot stability paths for the top_n most stable features."""
    stability_scores = selection_matrix.max(axis=1)
    top_idx = np.argsort(stability_scores)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.tab10
    for rank, feat_idx in enumerate(top_idx):
        ax.plot(
            alphas,
            selection_matrix[feat_idx],
            label=feature_names[feat_idx],
            color=cmap(rank % 10),
            linewidth=1.8,
        )

    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.5,
               label=f"threshold = {threshold}")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Lasso regularisation (alpha)", fontsize=11)
    ax.set_ylabel("Selection probability", fontsize=11)
    ax.set_title("Stability Paths", fontsize=13)
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Stability path saved to {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Diabetes dataset — 10 features, regression
    data = load_diabetes()
    X_raw, y = data.data, data.target
    feature_names = list(data.feature_names)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    THRESHOLD = 0.6

    print("Running stability selection (100 bootstrap subsamples) …")
    scores, sel_matrix, alphas = stability_selection(
        X, y,
        n_bootstraps=100,
        alpha_min=0.01,
        alpha_max=1.0,
        n_alphas=30,
        threshold=THRESHOLD,
        random_state=42,
    )

    results = pd.DataFrame({
        "feature": feature_names,
        "stability_score": scores,
        "selected": scores >= THRESHOLD,
    }).sort_values("stability_score", ascending=False)

    print(f"\nStability scores (threshold = {THRESHOLD}):")
    print(results.to_string(index=False))

    selected = results[results["selected"]]["feature"].tolist()
    print(f"\nStable features: {selected}")

    plot_stability_paths(
        sel_matrix, alphas, feature_names,
        threshold=THRESHOLD,
        save_path="/home/user/course-creator/courses/advanced-feature-selection/recipes/stability_paths.png",
    )
