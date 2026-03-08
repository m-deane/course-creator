"""
Module 1: Statistical Filter Methods — Self-Check Exercises
===========================================================

Three exercises to consolidate understanding from Guides 01-03 and Notebooks 01-03.
These exercises are ungraded and self-directed. Run the self-check functions at the
end of each section to verify your implementation.

Exercises:
  1. Custom filter selector using distance correlation
  2. Relief-based feature ranker for classification
  3. Comparison of filter rankings across three metrics

Prerequisites:
  numpy, scipy, sklearn, pandas (all standard; no extra installs needed)

Run this file directly:
  python 01_filter_methods_exercises.py
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cdist
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)


# =============================================================================
# SHARED UTILITIES
# Used by all three exercises. Do not modify.
# =============================================================================

def load_and_scale(dataset_loader):
    """Load a sklearn dataset and return (X_scaled_df, y_array, feature_names)."""
    data = dataset_loader()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    scaler = StandardScaler()
    X_sc = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_sc, y, list(X.columns)


def evaluate_features(X: pd.DataFrame, y: np.ndarray, feature_names: list,
                       cv: int = 5) -> dict:
    """
    Evaluate a feature set using cross-validated logistic regression.

    Returns
    -------
    dict with keys 'mean_cv_acc', 'std_cv_acc', 'n_features'
    """
    clf = LogisticRegression(max_iter=2000, random_state=42)
    scores = cross_val_score(clf, X[feature_names], y, cv=cv, scoring="accuracy")
    return {
        "mean_cv_acc": scores.mean(),
        "std_cv_acc": scores.std(),
        "n_features": len(feature_names),
    }


def print_section(title: str) -> None:
    width = max(len(title) + 4, 60)
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


# =============================================================================
# EXERCISE 1: Distance Correlation Filter Selector
# =============================================================================
#
# Task:
#   Implement a filter feature selector that ranks features using distance
#   correlation (dCor) with the target variable. Then build a selector that
#   returns the top-k features by dCor score.
#
# Why distance correlation?
#   dCor detects arbitrary nonlinear dependencies and handles multivariate
#   features natively. Unlike Pearson r, dCor(X,Y) = 0 iff X and Y are
#   independent (for continuous variables).
#
# Requirements:
#   1. distance_covariance_squared(X, Y) -> float
#        Compute dCov²(X, Y) via doubly-centred pairwise distance matrices.
#   2. distance_correlation(X, Y) -> float
#        Compute dCor(X, Y) = sqrt(dCov²(X,Y) / sqrt(dVar²(X) * dVar²(Y))).
#        Return 0.0 when denominator is near zero.
#   3. dcor_filter(X_df, y, k) -> list of str
#        Rank all features by dCor with y, return top-k feature names.
#
# Self-check:
#   - dCor of a variable with itself should equal 1.0 (exactly)
#   - dCor of independent variables should be near 0.0
#   - dCor should detect the quadratic relationship Y = X² + ε
#     while Pearson r should be near zero
#   - Selected features should improve over random baseline when evaluated
# =============================================================================

print_section("Exercise 1: Distance Correlation Filter Selector")


def distance_covariance_squared(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute squared distance covariance dCov²(X, Y).

    Algorithm:
      1. Compute pairwise Euclidean distance matrices A (from X) and B (from Y).
      2. Doubly centre each: A_kl = a_kl - row_mean_k - col_mean_l + grand_mean
      3. dCov²(X, Y) = mean(A_centred * B_centred)

    Parameters
    ----------
    X : np.ndarray of shape (n,) or (n, p)
    Y : np.ndarray of shape (n,) or (n, q)

    Returns
    -------
    float >= 0
    """
    # TODO: implement this function
    # Hint 1: reshape 1-D inputs to 2-D before calling cdist
    # Hint 2: double centering = subtract row_mean - col_mean + grand_mean
    # Hint 3: return (A_cent * B_cent).mean()
    raise NotImplementedError("Implement distance_covariance_squared")


def distance_correlation(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute distance correlation dCor(X, Y) in [0, 1].

    dCor(X, Y) = sqrt(dCov²(X,Y) / sqrt(dVar²(X) * dVar²(Y)))

    Returns 0.0 when denominator is near zero (either variable has zero variance).

    Parameters
    ----------
    X : np.ndarray of shape (n,) or (n, p)
    Y : np.ndarray of shape (n,) or (n, q)

    Returns
    -------
    float in [0, 1]
    """
    # TODO: call distance_covariance_squared three times (XY, XX, YY)
    # Hint: dVar²(X) = dCov²(X, X)
    # Hint: return 0.0 if sqrt(dVar²(X) * dVar²(Y)) < 1e-10
    raise NotImplementedError("Implement distance_correlation")


def dcor_filter(X_df: pd.DataFrame, y: np.ndarray, k: int) -> list:
    """
    Filter feature selector using distance correlation.

    Rank all features in X_df by dCor(feature, y), return top-k feature names.

    Parameters
    ----------
    X_df : pd.DataFrame, shape (n_samples, n_features)
    y    : np.ndarray, shape (n_samples,) — target (continuous or discrete)
    k    : int — number of features to select

    Returns
    -------
    list of str — top-k feature names sorted by dCor descending
    """
    # TODO: compute dCor for each feature, sort, return top k names
    raise NotImplementedError("Implement dcor_filter")


# ------- Self-Check 1 -------

def selfcheck_exercise_1():
    print("\n[Self-Check 1] Distance Correlation Filter")

    # Property 1: dCor(X, X) = 1.0
    x_test = np.random.randn(200)
    self_dcor = distance_correlation(x_test, x_test)
    assert abs(self_dcor - 1.0) < 0.01, (
        f"dCor(X,X) should be 1.0, got {self_dcor:.4f}"
    )
    print(f"  PASS  dCor(X, X) = {self_dcor:.4f} (expected ~1.0)")

    # Property 2: dCor near zero for independent variables
    x_a = np.random.randn(300)
    y_a = np.random.randn(300)
    ind_dcor = distance_correlation(x_a, y_a)
    assert ind_dcor < 0.15, (
        f"dCor(X, Y_independent) should be near 0, got {ind_dcor:.4f}"
    )
    print(f"  PASS  dCor(X, Y_indep) = {ind_dcor:.4f} (expected < 0.15)")

    # Property 3: dCor detects quadratic while Pearson r ≈ 0
    x_q = np.random.uniform(-2, 2, 500)
    y_q = x_q ** 2 + np.random.normal(0, 0.3, 500)
    quad_dcor = distance_correlation(x_q, y_q)
    quad_pearson = abs(pearsonr(x_q, y_q)[0])
    assert quad_dcor > 0.3, (
        f"dCor should detect Y=X² relationship (>0.3), got {quad_dcor:.4f}"
    )
    assert quad_pearson < 0.2, (
        f"Pearson r should be near zero for Y=X², got {quad_pearson:.4f}"
    )
    print(f"  PASS  Quadratic: dCor={quad_dcor:.4f} (>0.3) | Pearson={quad_pearson:.4f} (<0.2)")

    # Property 4: filter returns the right number of features
    X_bc, y_bc, _ = load_and_scale(load_breast_cancer)
    k = 7
    selected = dcor_filter(X_bc, y_bc.astype(float), k)
    assert len(selected) == k, f"Should return {k} features, got {len(selected)}"
    assert len(set(selected)) == k, "Returned features should be unique"
    assert all(f in X_bc.columns for f in selected), "All names must be valid feature names"
    print(f"  PASS  dcor_filter returns {k} unique, valid feature names")

    # Property 5: dCor selected features perform above a random baseline
    result = evaluate_features(X_bc, y_bc, selected)
    random_features = np.random.choice(X_bc.columns, k, replace=False).tolist()
    random_result = evaluate_features(X_bc, y_bc, random_features)
    assert result["mean_cv_acc"] >= random_result["mean_cv_acc"] - 0.02, (
        f"dCor features ({result['mean_cv_acc']:.4f}) should not be much worse "
        f"than random ({random_result['mean_cv_acc']:.4f})"
    )
    print(f"  PASS  dCor top-{k}: acc={result['mean_cv_acc']:.4f} vs "
          f"random={random_result['mean_cv_acc']:.4f}")
    print("  Exercise 1: ALL CHECKS PASSED")


# =============================================================================
# EXERCISE 2: Relief-Based Feature Ranker
# =============================================================================
#
# Task:
#   Implement a ReliefF feature ranker for binary classification. ReliefF assigns
#   weights to features based on how well they discriminate between instances of
#   different classes while being consistent within the same class.
#
# Algorithm (for binary classification):
#   Initialise: W[f] = 0 for all features f
#   For m random instances x_i (or all if n_iterations is None):
#     Find k nearest hits H_j (same class as x_i)
#     Find k nearest misses M_j (opposite class)
#     For each feature f:
#       W[f] -= (1/mk) * sum_j diff(f, x_i, H_j)   [penalise within-class difference]
#       W[f] += (1/mk) * sum_j diff(f, x_i, M_j)   [reward between-class difference]
#   where diff(f, a, b) = |a[f] - b[f]| (features normalised to [0,1])
#
# Requirements:
#   1. relieff_binary(X_df, y, k, n_iterations) -> pd.Series
#        Compute ReliefF weights for a binary classification problem.
#        Return a Series of feature weights indexed by feature name, sorted descending.
#   2. relief_select(X_df, y, k, n_features) -> list of str
#        Return the top n_features features by ReliefF weight.
#
# Self-check:
#   - On the XOR dataset (two informative features + noise), both informative
#     features should receive positive weights
#   - Noise features should receive weights near zero
#   - Selected features should outperform pure MI ranking on the XOR dataset
# =============================================================================

print_section("Exercise 2: ReliefF Feature Ranker")


def relieff_binary(X_df: pd.DataFrame, y: np.ndarray,
                    k: int = 10, n_iterations: int = None,
                    random_state: int = 42) -> pd.Series:
    """
    ReliefF feature weighting for binary classification.

    Parameters
    ----------
    X_df : pd.DataFrame, shape (n_samples, n_features)
        All features should be numeric. Will be normalised to [0,1] internally.
    y : np.ndarray of shape (n_samples,) — binary labels (0 or 1)
    k : int — number of nearest hits and misses per reference instance
    n_iterations : int or None
        Number of reference instances to sample. If None, use all instances.
    random_state : int

    Returns
    -------
    pd.Series
        Feature weights, indexed by feature name, sorted descending.
        Positive weight = feature helps discrimination.
        Negative weight = feature hurts discrimination.
    """
    # TODO: implement ReliefF
    #
    # Step 1: normalise X_df to [0, 1] using MinMaxScaler
    # Step 2: separate instances by class (y==0, y==1)
    # Step 3: sample n_iterations reference instances (or use all)
    # Step 4: for each reference instance:
    #           find k nearest hits using L2 distance (or L-inf for speed)
    #           find k nearest misses
    #           update W for each feature
    # Step 5: return sorted pd.Series
    #
    # Hint: Use np.linalg.norm(X_arr[i] - X_class, axis=1) to compute distances
    #       from instance i to all instances of one class.
    # Hint: np.argpartition(dists, k)[:k] gives k smallest without full sort.
    raise NotImplementedError("Implement relieff_binary")


def relief_select(X_df: pd.DataFrame, y: np.ndarray,
                   k_nn: int = 10, n_features: int = 10,
                   n_iterations: int = None,
                   random_state: int = 42) -> list:
    """
    Select top-n_features features by ReliefF weight.

    Parameters
    ----------
    X_df : pd.DataFrame
    y : np.ndarray — binary labels
    k_nn : int — nearest neighbours for ReliefF
    n_features : int — how many features to return
    n_iterations : int or None

    Returns
    -------
    list of str — top feature names by weight
    """
    # TODO: call relieff_binary, return top n_features by weight
    raise NotImplementedError("Implement relief_select")


# ------- Self-Check 2 -------

def selfcheck_exercise_2():
    print("\n[Self-Check 2] ReliefF Feature Ranker")

    # Build XOR dataset: informative features A, B; noise features C0..C9
    n = 800
    rng = np.random.default_rng(42)
    A = rng.integers(0, 2, n).astype(float)
    B = rng.integers(0, 2, n).astype(float)
    Y_xor = np.bitwise_xor(A.astype(int), B.astype(int))
    noise = rng.standard_normal((n, 10))
    X_xor = pd.DataFrame(
        np.column_stack([A, B, noise]),
        columns=["A", "B"] + [f"noise_{i}" for i in range(10)]
    )

    weights = relieff_binary(X_xor, Y_xor, k=10, n_iterations=200)
    assert isinstance(weights, pd.Series), "relieff_binary must return a pd.Series"
    assert set(weights.index) == set(X_xor.columns), (
        "Series index must contain all feature names"
    )

    w_A = weights["A"]
    w_B = weights["B"]
    assert w_A > 0, f"Feature A (XOR-informative) should have positive weight, got {w_A:.4f}"
    assert w_B > 0, f"Feature B (XOR-informative) should have positive weight, got {w_B:.4f}"
    print(f"  PASS  XOR features have positive weights: A={w_A:.4f}, B={w_B:.4f}")

    noise_weights = weights[[f"noise_{i}" for i in range(10)]]
    assert noise_weights.mean() < w_A and noise_weights.mean() < w_B, (
        "Noise features should have lower average weight than informative features"
    )
    print(f"  PASS  Noise features have lower avg weight ({noise_weights.mean():.4f}) "
          f"than informative features")

    # relief_select should return A and B in top features
    selected = relief_select(X_xor, Y_xor, k_nn=10, n_features=4, n_iterations=200)
    assert "A" in selected or "B" in selected, (
        f"At least one of A, B should be selected. Got: {selected}"
    )
    print(f"  PASS  relief_select top-4: {selected} (contains informative feature)")

    # Compare with pure MI: ReliefF should find XOR features; MI should not
    mi_scores = mutual_info_classif(X_xor, Y_xor, random_state=42)
    mi_series = pd.Series(mi_scores, index=X_xor.columns)
    mi_top4 = mi_series.nlargest(4).index.tolist()

    mi_finds_informative = "A" in mi_top4 or "B" in mi_top4
    relief_finds_informative = "A" in selected or "B" in selected

    if not mi_finds_informative and relief_finds_informative:
        print("  PASS  ReliefF finds XOR features; MI does not — interaction detected!")
    elif mi_finds_informative:
        print(f"  INFO  MI also finds informative feature(s): {mi_top4}")
    else:
        print("  INFO  Neither method finds both XOR features — try more iterations")

    print("  Exercise 2: ALL CHECKS PASSED")


# =============================================================================
# EXERCISE 3: Filter Ranking Comparison
# =============================================================================
#
# Task:
#   Build a comparison pipeline that applies three filter measures to a dataset
#   and produces:
#     a) Per-measure feature rankings
#     b) A consensus ranking (mean of ranks across measures)
#     c) A performance evaluation of top-k features per measure vs consensus
#
# The three measures to use:
#   - Mutual Information (sklearn: mutual_info_classif)
#   - Distance Correlation (from Exercise 1)
#   - Absolute Spearman rank correlation
#
# Requirements:
#   1. compute_filter_rankings(X_df, y) -> pd.DataFrame
#        Return a DataFrame with columns ['MI', 'dCor', 'Spearman'],
#        index = feature names, values = ranks (1 = most relevant).
#   2. consensus_ranking(rank_df) -> pd.Series
#        Return mean rank per feature, sorted ascending (best first).
#   3. compare_filters(X_df, y, k) -> pd.DataFrame
#        Run all three measures + consensus, evaluate top-k features from each,
#        return a DataFrame with columns ['method', 'n_features', 'mean_cv_acc'].
#
# Self-check:
#   - All methods should outperform random feature selection
#   - Consensus ranking should perform within 1% of the best individual measure
#   - The agreement matrix (Spearman r between per-measure rank lists) should
#     reveal which pairs of measures agree most
# =============================================================================

print_section("Exercise 3: Filter Ranking Comparison")


def compute_filter_rankings(X_df: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """
    Compute per-measure feature rankings.

    Parameters
    ----------
    X_df : pd.DataFrame, shape (n_samples, n_features)
    y : np.ndarray — target labels

    Returns
    -------
    pd.DataFrame with columns ['MI', 'dCor', 'Spearman']
        Values are ranks (1 = most relevant feature by that measure).
        Index = feature names.
    """
    # TODO:
    # 1. Compute MI scores using mutual_info_classif
    # 2. Compute dCor scores using distance_correlation (from Exercise 1)
    # 3. Compute absolute Spearman r with target for each feature
    # 4. Rank each score series (ascending=False so highest score -> rank 1)
    # 5. Return DataFrame of ranks
    #
    # Hint: pd.Series.rank(ascending=False) gives rank 1 to the largest value
    raise NotImplementedError("Implement compute_filter_rankings")


def consensus_ranking(rank_df: pd.DataFrame) -> pd.Series:
    """
    Compute consensus ranking as mean of per-measure ranks.

    Parameters
    ----------
    rank_df : pd.DataFrame — ranks per measure (from compute_filter_rankings)

    Returns
    -------
    pd.Series — mean rank per feature, sorted ascending (best first)
    """
    # TODO: return rank_df.mean(axis=1).sort_values(ascending=True)
    raise NotImplementedError("Implement consensus_ranking")


def compare_filters(X_df: pd.DataFrame, y: np.ndarray, k: int = 8) -> pd.DataFrame:
    """
    Run all three measures + consensus, evaluate top-k features from each.

    Parameters
    ----------
    X_df : pd.DataFrame
    y    : np.ndarray
    k    : int — number of features to select per method

    Returns
    -------
    pd.DataFrame with columns ['method', 'features', 'mean_cv_acc', 'std_cv_acc']
    """
    # TODO:
    # 1. Call compute_filter_rankings(X_df, y)
    # 2. Call consensus_ranking(rank_df) -> get top-k consensus features
    # 3. For each method (MI, dCor, Spearman) + consensus: get top-k features
    # 4. For each, call evaluate_features and record results
    # 5. Also include a random baseline
    # 6. Return results as DataFrame
    raise NotImplementedError("Implement compare_filters")


# ------- Self-Check 3 -------

def selfcheck_exercise_3():
    print("\n[Self-Check 3] Filter Ranking Comparison")

    X_bc, y_bc, _ = load_and_scale(load_breast_cancer)
    k = 8

    # Test compute_filter_rankings
    rank_df = compute_filter_rankings(X_bc, y_bc)
    assert isinstance(rank_df, pd.DataFrame), "compute_filter_rankings must return DataFrame"
    assert set(rank_df.columns) == {'MI', 'dCor', 'Spearman'}, (
        f"Columns should be ['MI', 'dCor', 'Spearman'], got {list(rank_df.columns)}"
    )
    assert set(rank_df.index) == set(X_bc.columns), (
        "Index should contain all feature names"
    )
    # Ranks should be integers 1 to n_features
    for col in rank_df.columns:
        assert rank_df[col].min() == 1, f"Minimum rank should be 1 for {col}"
        assert rank_df[col].max() == len(X_bc.columns), f"Max rank should equal n_features for {col}"
    print(f"  PASS  compute_filter_rankings returns valid rank DataFrame")

    # Test consensus_ranking
    consensus = consensus_ranking(rank_df)
    assert isinstance(consensus, pd.Series), "consensus_ranking must return pd.Series"
    assert consensus.index[0] == consensus.sort_values().index[0], (
        "consensus_ranking should be sorted ascending (best feature first)"
    )
    print(f"  PASS  consensus_ranking returns sorted Series")
    print(f"        Top consensus feature: '{consensus.index[0]}'")

    # Test compare_filters
    comparison = compare_filters(X_bc, y_bc, k=k)
    assert isinstance(comparison, pd.DataFrame), "compare_filters must return DataFrame"
    assert 'method' in comparison.columns, "Results must have 'method' column"
    assert 'mean_cv_acc' in comparison.columns, "Results must have 'mean_cv_acc' column"
    print(f"  PASS  compare_filters returns DataFrame with {len(comparison)} rows")

    # All filter methods should outperform random
    random_mask = comparison['method'].str.lower().str.contains('random')
    filter_mask = ~random_mask
    if random_mask.any() and filter_mask.any():
        random_acc = comparison.loc[random_mask, 'mean_cv_acc'].mean()
        filter_acc = comparison.loc[filter_mask, 'mean_cv_acc'].mean()
        assert filter_acc >= random_acc - 0.02, (
            f"Filter methods ({filter_acc:.4f}) should outperform random ({random_acc:.4f})"
        )
        print(f"  PASS  Filter methods avg acc {filter_acc:.4f} >= random acc {random_acc:.4f} - 0.02")

    # Consensus should be within 1% of best method
    if filter_mask.any():
        consensus_mask = comparison['method'].str.lower().str.contains('consensus')
        if consensus_mask.any():
            best_acc = comparison.loc[filter_mask, 'mean_cv_acc'].max()
            consensus_acc = comparison.loc[consensus_mask, 'mean_cv_acc'].max()
            assert consensus_acc >= best_acc - 0.01, (
                f"Consensus ({consensus_acc:.4f}) should be within 1% of best ({best_acc:.4f})"
            )
            print(f"  PASS  Consensus acc {consensus_acc:.4f} within 1% of best {best_acc:.4f}")

    # Measure agreement analysis
    print("\n  Measure agreement (Spearman r between rank lists):")
    for m1 in rank_df.columns:
        for m2 in rank_df.columns:
            if m1 < m2:
                r, _ = spearmanr(rank_df[m1], rank_df[m2])
                print(f"    {m1} vs {m2}: ρ = {r:.3f}")

    print(f"\n  Performance by method:")
    print(comparison[['method', 'mean_cv_acc', 'std_cv_acc']].to_string(index=False))
    print("  Exercise 3: ALL CHECKS PASSED")


# =============================================================================
# MAIN: Run all self-checks
# =============================================================================

if __name__ == "__main__":
    print("\nModule 1: Statistical Filter Methods — Self-Check Exercises")
    print("Complete each TODO, then run this file. Checks print PASS or raise AssertionError.\n")

    exercises = [
        ("Exercise 1: Distance Correlation Filter", selfcheck_exercise_1),
        ("Exercise 2: ReliefF Feature Ranker",      selfcheck_exercise_2),
        ("Exercise 3: Filter Ranking Comparison",   selfcheck_exercise_3),
    ]

    passed = 0
    failed = 0

    for name, check_fn in exercises:
        try:
            check_fn()
            passed += 1
        except NotImplementedError as e:
            print(f"\n  [TODO] {name}: not yet implemented ({e})")
            failed += 1
        except AssertionError as e:
            print(f"\n  [FAIL] {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"\n  [ERROR] {name}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} remaining")
    if failed == 0:
        print("  All exercises complete. Move to Module 2.")
    else:
        print(f"  {failed} exercise(s) still need implementation.")
    print("=" * 60)
