"""
Module 0: Feature Selection Landscape — Self-Check Exercises
=============================================================

Three exercises to consolidate understanding from Guides 01-03 and Notebooks 01-02.
These exercises are ungraded and self-directed. Run the self-check functions at the
end of each section to verify your implementation.

Exercises:
  1. Method cost estimator — compute and compare wall-clock estimates for each
     selection family given a problem specification
  2. Benchmark selector — implement a greedy univariate filter baseline and compare
     it against a random baseline on a real dataset
  3. Selection method fingerprinter — given a function that selects features, identify
     which family it most likely belongs to by observing its scaling behaviour

Prerequisites:
  numpy, scipy, sklearn, pandas, time (all standard; no extra installs needed)

Run this file directly:
  python 01_landscape_exercises.py
"""

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)


# =============================================================================
# SHARED UTILITIES
# =============================================================================

def print_section(title: str) -> None:
    width = max(len(title) + 4, 60)
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def make_benchmark_dataset(n_samples: int = 1000, n_features: int = 50,
                            n_informative: int = 10, random_state: int = 42):
    """
    Create a labelled classification dataset for benchmarking.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)
    feature_names : list of str
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=5,
        n_clusters_per_class=2,
        random_state=random_state,
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    feature_names = [f"f{i:03d}" for i in range(n_features)]
    return X, y, feature_names


# =============================================================================
# EXERCISE 1: Method Cost Estimator
# =============================================================================
#
# Task:
#   Build a cost estimator that calculates the *number of model evaluations*
#   required by each selection family for a given problem.
#
#   Your estimator should compute:
#     - Filter (univariate):   0 model evaluations (no model used)
#     - Forward Selection:     k * p model evaluations (approximately)
#     - RFE:                   (p - k) model evaluations
#     - RFECV (step=s, folds=f): f * ceil((p - k) / s) evaluations
#     - Lasso path (100 lambdas, f folds): f * 100 evaluations
#     - GA (G generations, P population, f folds): G * P * f evaluations
#
#   Then build a wall-clock estimator that multiplies by T_model (in seconds)
#   and returns a human-readable duration string.
#
# Requirements:
#   1. model_evaluations(method, p, k, f, s, G, P) -> int
#        Return the number of model evaluations for the given method.
#        Valid method names: 'filter', 'forward', 'rfe', 'rfecv', 'lasso', 'ga'
#   2. wall_clock_estimate(method, T_model_sec, p, k, f, s, G, P) -> str
#        Return a human-readable string like "3 min 20 sec" or "< 1 sec".
#   3. cost_table(p, k, T_model_sec, f, s, G, P) -> pd.DataFrame
#        Return a DataFrame with columns ['method', 'model_evals', 'estimated_time']
#        for all six methods.
#
# Self-check:
#   - Forward selection with p=100, k=10 should give 1,000 evaluations
#   - GA with G=50, P=30, f=5 should give 7,500 evaluations
#   - Filter should always give 0 evaluations
#   - wall_clock_estimate with T_model=0 should be "< 1 sec" for all methods
# =============================================================================

print_section("Exercise 1: Method Cost Estimator")

import math


def model_evaluations(method: str, p: int = 100, k: int = 10,
                       f: int = 5, s: int = 1, G: int = 100, P: int = 50) -> int:
    """
    Compute the number of model evaluations for a given feature selection method.

    Parameters
    ----------
    method : str
        One of: 'filter', 'forward', 'rfe', 'rfecv', 'lasso', 'ga'
    p : int
        Total number of features
    k : int
        Number of features to select
    f : int
        Number of CV folds (used by rfecv, lasso, ga)
    s : int
        Step size for rfecv (features removed per iteration)
    G : int
        Number of generations (GA only)
    P : int
        Population size (GA only)

    Returns
    -------
    int : number of model fits (0 for filter)
    """
    # TODO: implement this function
    #
    # method == 'filter':  return 0
    # method == 'forward': return k * p  (approximate: sum_{j=0}^{k-1} (p-j) ≈ k*p)
    # method == 'rfe':     return p - k
    # method == 'rfecv':   return f * math.ceil((p - k) / s)
    # method == 'lasso':   return f * 100   (100 lambda values on the path)
    # method == 'ga':      return G * P * f
    #
    # Hint: use math.ceil for rfecv
    raise NotImplementedError("Implement model_evaluations")


def wall_clock_estimate(method: str, T_model_sec: float, p: int = 100,
                         k: int = 10, f: int = 5, s: int = 1,
                         G: int = 100, P: int = 50) -> str:
    """
    Estimate the wall-clock time for a given method.

    Parameters
    ----------
    method : str
    T_model_sec : float
        Time for a single model train in seconds
    Other parameters: same as model_evaluations

    Returns
    -------
    str : human-readable duration, e.g. "3 min 20 sec", "< 1 sec", "2.4 hrs", "1.3 days"
    """
    # TODO: compute total_seconds = model_evaluations(...) * T_model_sec
    # Then convert to a readable string:
    #   < 1 sec      → "< 1 sec"
    #   < 60 sec     → "{n} sec"
    #   < 3600 sec   → "{m} min {s} sec"
    #   < 86400 sec  → "{h:.1f} hrs"
    #   else         → "{d:.1f} days"
    raise NotImplementedError("Implement wall_clock_estimate")


def cost_table(p: int = 100, k: int = 10, T_model_sec: float = 1.0,
               f: int = 5, s: int = 1, G: int = 100, P: int = 50) -> pd.DataFrame:
    """
    Return a cost comparison table for all six methods.

    Parameters
    ----------
    p, k, T_model_sec, f, s, G, P : see model_evaluations / wall_clock_estimate

    Returns
    -------
    pd.DataFrame with columns ['method', 'model_evals', 'estimated_time']
    """
    # TODO: call model_evaluations and wall_clock_estimate for each of:
    # ['filter', 'forward', 'rfe', 'rfecv', 'lasso', 'ga']
    # Return a DataFrame with one row per method.
    raise NotImplementedError("Implement cost_table")


# ------- Self-Check 1 -------

def selfcheck_exercise_1():
    print("\n[Self-Check 1] Method Cost Estimator")

    # filter: always 0
    assert model_evaluations('filter', p=500, k=20) == 0, (
        "Filter method should require 0 model evaluations"
    )
    print("  PASS  filter → 0 model evaluations")

    # forward selection: k * p
    evals_fwd = model_evaluations('forward', p=100, k=10)
    assert evals_fwd == 1000, (
        f"Forward selection: p=100, k=10 → expected 1000, got {evals_fwd}"
    )
    print(f"  PASS  forward: p=100, k=10 → {evals_fwd} evaluations")

    # GA
    evals_ga = model_evaluations('ga', G=50, P=30, f=5)
    assert evals_ga == 7500, (
        f"GA: G=50, P=30, f=5 → expected 7500, got {evals_ga}"
    )
    print(f"  PASS  ga: G=50, P=30, f=5 → {evals_ga} evaluations")

    # rfecv
    evals_rfecv = model_evaluations('rfecv', p=100, k=10, f=5, s=5)
    expected_rfecv = 5 * math.ceil((100 - 10) / 5)
    assert evals_rfecv == expected_rfecv, (
        f"rfecv: p=100, k=10, f=5, s=5 → expected {expected_rfecv}, got {evals_rfecv}"
    )
    print(f"  PASS  rfecv: p=100, k=10, f=5, s=5 → {evals_rfecv} evaluations")

    # lasso: f * 100
    evals_lasso = model_evaluations('lasso', f=5)
    assert evals_lasso == 500, (
        f"lasso: f=5 → expected 500, got {evals_lasso}"
    )
    print(f"  PASS  lasso: f=5 → {evals_lasso} evaluations")

    # wall_clock: T_model=0 → "< 1 sec" for all
    for m in ['filter', 'forward', 'rfe', 'rfecv', 'lasso', 'ga']:
        result = wall_clock_estimate(m, T_model_sec=0.0)
        assert result == "< 1 sec", (
            f"T_model=0 should give '< 1 sec' for all methods, got '{result}' for {m}"
        )
    print("  PASS  wall_clock_estimate with T_model=0 → '< 1 sec' for all")

    # cost_table has correct columns
    df = cost_table(p=100, k=10, T_model_sec=1.0, f=5, s=5, G=50, P=30)
    assert isinstance(df, pd.DataFrame), "cost_table must return pd.DataFrame"
    assert set(['method', 'model_evals', 'estimated_time']).issubset(df.columns), (
        f"Expected columns ['method', 'model_evals', 'estimated_time'], got {list(df.columns)}"
    )
    assert len(df) == 6, f"Expected 6 rows (one per method), got {len(df)}"
    print(f"  PASS  cost_table returns {len(df)}-row DataFrame")
    print("\n  Cost table (p=100, k=10, T_model=1s, f=5, s=5, G=50, P=30):")
    print(df.to_string(index=False))
    print("  Exercise 1: ALL CHECKS PASSED")


# =============================================================================
# EXERCISE 2: Greedy Univariate Filter Baseline
# =============================================================================
#
# Task:
#   Implement a greedy univariate filter selector and evaluate it against a
#   random baseline on the breast cancer dataset.
#
#   A greedy univariate filter:
#     1. Computes MI(feature, y) for each feature
#     2. Selects the top-k features by MI score
#     3. Trains a logistic regression on those features
#
# Requirements:
#   1. mi_filter_select(X, y, k) -> np.ndarray
#        Return the indices of the top-k features by MI with y.
#        X is a 2D array, y is 1D.
#   2. random_select(X, k, random_state) -> np.ndarray
#        Return k randomly chosen feature indices.
#   3. evaluate_selection(X, y, indices, cv=5) -> dict
#        Return {'mean_cv_acc': float, 'std_cv_acc': float, 'n_features': int}
#        using cross-validated LogisticRegression.
#   4. filter_vs_random_comparison(X, y, k_values) -> pd.DataFrame
#        For each k in k_values, compute mean_cv_acc for MI filter and random.
#        Return DataFrame with columns ['k', 'method', 'mean_cv_acc'].
#
# Self-check:
#   - MI filter should outperform random for all tested k values on average
#   - MI scores should be non-negative
#   - Top-k selection should return exactly k unique indices
# =============================================================================

print_section("Exercise 2: Greedy Univariate Filter Baseline")


def mi_filter_select(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """
    Select the top-k features by mutual information with y.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)
    k : int — number of features to select

    Returns
    -------
    np.ndarray of shape (k,) — indices of selected features (sorted ascending)
    """
    # TODO:
    # 1. Compute MI scores using mutual_info_classif(X, y, random_state=42)
    # 2. Use np.argsort(scores)[::-1][:k] to get top-k indices
    # 3. Return sorted indices (np.sort)
    raise NotImplementedError("Implement mi_filter_select")


def random_select(X: np.ndarray, k: int, random_state: int = 42) -> np.ndarray:
    """
    Select k features at random (without replacement).

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    k : int
    random_state : int

    Returns
    -------
    np.ndarray of shape (k,) — randomly chosen feature indices, sorted ascending
    """
    # TODO: use np.random.default_rng(random_state).choice(n_features, k, replace=False)
    # Return sorted indices.
    raise NotImplementedError("Implement random_select")


def evaluate_selection(X: np.ndarray, y: np.ndarray, indices: np.ndarray,
                        cv: int = 5) -> dict:
    """
    Evaluate a feature subset using cross-validated logistic regression.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)
    indices : np.ndarray of int — feature column indices to use
    cv : int

    Returns
    -------
    dict with keys 'mean_cv_acc', 'std_cv_acc', 'n_features'
    """
    # TODO:
    # 1. Subset X to X[:, indices]
    # 2. Train LogisticRegression(max_iter=2000, random_state=42) with cross_val_score
    # 3. Return the dict
    raise NotImplementedError("Implement evaluate_selection")


def filter_vs_random_comparison(X: np.ndarray, y: np.ndarray,
                                  k_values: list = None) -> pd.DataFrame:
    """
    Compare MI filter against random selection at multiple k values.

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray
    k_values : list of int — subset sizes to evaluate. Defaults to [3, 5, 10, 15, 20].

    Returns
    -------
    pd.DataFrame with columns ['k', 'method', 'mean_cv_acc']
    """
    if k_values is None:
        k_values = [3, 5, 10, 15, 20]

    # TODO: for each k in k_values:
    #   - call mi_filter_select, evaluate_selection → record ('mi_filter', k, acc)
    #   - call random_select, evaluate_selection → record ('random', k, acc)
    # Return DataFrame from records.
    raise NotImplementedError("Implement filter_vs_random_comparison")


# ------- Self-Check 2 -------

def selfcheck_exercise_2():
    print("\n[Self-Check 2] Greedy Univariate Filter Baseline")

    data = load_breast_cancer()
    X = data.data
    y = data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    n_features = X.shape[1]

    # Test mi_filter_select
    k = 10
    idx = mi_filter_select(X, y, k)
    assert len(idx) == k, f"Expected {k} features, got {len(idx)}"
    assert len(set(idx)) == k, "Indices must be unique"
    assert all(0 <= i < n_features for i in idx), "Indices must be valid column indices"
    print(f"  PASS  mi_filter_select returns {k} valid unique indices")
    print(f"        Selected indices: {sorted(idx)}")

    # Test random_select
    ridx = random_select(X, k)
    assert len(ridx) == k, f"Expected {k} features, got {len(ridx)}"
    assert len(set(ridx)) == k, "Random indices must be unique"
    print(f"  PASS  random_select returns {k} unique indices")

    # Test evaluate_selection
    result = evaluate_selection(X, y, idx)
    assert 'mean_cv_acc' in result, "Result must contain 'mean_cv_acc'"
    assert 0.5 < result['mean_cv_acc'] <= 1.0, (
        f"mean_cv_acc should be > 0.5, got {result['mean_cv_acc']:.4f}"
    )
    print(f"  PASS  evaluate_selection: acc={result['mean_cv_acc']:.4f}")

    # Test filter_vs_random_comparison
    k_vals = [5, 10, 15]
    df = filter_vs_random_comparison(X, y, k_vals)
    assert isinstance(df, pd.DataFrame), "Must return pd.DataFrame"
    assert 'method' in df.columns, "Must have 'method' column"
    assert 'mean_cv_acc' in df.columns, "Must have 'mean_cv_acc' column"
    assert len(df) == len(k_vals) * 2, (
        f"Expected {len(k_vals) * 2} rows, got {len(df)}"
    )
    print(f"  PASS  filter_vs_random_comparison returns {len(df)}-row DataFrame")

    # MI filter should outperform random on average
    filter_rows = df[df['method'] == 'mi_filter']
    random_rows = df[df['method'] == 'random']
    if len(filter_rows) > 0 and len(random_rows) > 0:
        avg_filter = filter_rows['mean_cv_acc'].mean()
        avg_random = random_rows['mean_cv_acc'].mean()
        assert avg_filter >= avg_random - 0.02, (
            f"MI filter ({avg_filter:.4f}) should outperform random ({avg_random:.4f})"
        )
        print(f"  PASS  MI filter avg acc {avg_filter:.4f} >= random avg {avg_random:.4f} - 0.02")

    print("\n  Comparison table:")
    print(df.to_string(index=False))
    print("  Exercise 2: ALL CHECKS PASSED")


# =============================================================================
# EXERCISE 3: Selection Method Fingerprinter
# =============================================================================
#
# Task:
#   Characterise an unknown selection method by measuring how its runtime scales
#   with p (number of features). Use this scaling signature to classify it as one
#   of three families: filter, wrapper, or evolutionary.
#
#   The classification rule (for runtime measured over p = [50, 100, 200]):
#     - If runtime roughly doubles when p doubles → O(p) → filter
#     - If runtime scales super-linearly (>3x when p doubles) → O(p * T_m) → wrapper
#     - If runtime is roughly constant as p grows → O(T_m only) → embedded
#
# Requirements:
#   1. measure_runtime(selector_fn, X, y, n_repeats=1) -> float
#        Time how long it takes to call selector_fn(X, y) and return mean seconds.
#   2. scaling_ratio(selector_fn, X_small, X_large, y_small, y_large) -> float
#        Compute runtime_large / runtime_small (the scaling ratio as p doubles).
#   3. classify_by_scaling(ratio) -> str
#        Return 'filter', 'wrapper', or 'embedded' based on the ratio:
#          ratio < 2.5   → 'filter' (roughly linear in p)
#          ratio < 6.0   → 'wrapper' (super-linear but bounded)
#          else          → 'evolutionary' (very expensive per p step)
#   4. fingerprint_selector(selector_fn, n_samples=200) -> dict
#        Run the full fingerprint: create datasets at p=50 and p=100, measure
#        runtimes, compute ratio, classify. Return dict with keys:
#          'p_small', 'p_large', 'time_small', 'time_large', 'ratio', 'family'
#
# Provided reference selectors (you will use these in the self-check):
#   - filter_selector(X, y): fast MI-based filter — should classify as 'filter'
#   - wrapper_selector(X, y): greedy forward selection — should classify as 'wrapper'
# =============================================================================

print_section("Exercise 3: Selection Method Fingerprinter")


def filter_selector(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Reference selector: fast MI filter (O(p*n)). Returns top-10 indices."""
    scores = mutual_info_classif(X, y, random_state=42)
    return np.argsort(scores)[::-1][:10]


def wrapper_selector(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Reference selector: greedy forward selection using LR (O(k*p*T_m))."""
    n_features = X.shape[1]
    selected = []
    remaining = list(range(n_features))
    clf = LogisticRegression(max_iter=500, random_state=42)

    for _ in range(min(5, n_features)):
        best_score = -np.inf
        best_feat = None
        for feat in remaining:
            candidate = selected + [feat]
            scores = cross_val_score(clf, X[:, candidate], y, cv=3, scoring='accuracy')
            if scores.mean() > best_score:
                best_score = scores.mean()
                best_feat = feat
        if best_feat is not None:
            selected.append(best_feat)
            remaining.remove(best_feat)

    return np.array(selected)


def measure_runtime(selector_fn, X: np.ndarray, y: np.ndarray,
                     n_repeats: int = 1) -> float:
    """
    Measure mean wall-clock time of selector_fn(X, y) in seconds.

    Parameters
    ----------
    selector_fn : callable (X, y) -> indices
    X : np.ndarray
    y : np.ndarray
    n_repeats : int — number of timed runs; return the mean

    Returns
    -------
    float : mean runtime in seconds
    """
    # TODO:
    # 1. For n_repeats iterations: record time.perf_counter() before and after
    #    calling selector_fn(X, y)
    # 2. Return the mean elapsed time
    raise NotImplementedError("Implement measure_runtime")


def scaling_ratio(selector_fn, X_small: np.ndarray, X_large: np.ndarray,
                   y_small: np.ndarray, y_large: np.ndarray) -> float:
    """
    Compute the runtime ratio when feature count doubles.

    Parameters
    ----------
    selector_fn : callable (X, y) -> indices
    X_small : np.ndarray with fewer features
    X_large : np.ndarray with more features (approximately double)
    y_small, y_large : np.ndarray — matching labels

    Returns
    -------
    float : time(X_large) / time(X_small)
    """
    # TODO: measure_runtime for both; divide. Guard against div-by-zero (use max(t_small, 1e-6))
    raise NotImplementedError("Implement scaling_ratio")


def classify_by_scaling(ratio: float) -> str:
    """
    Classify a selection method family based on runtime scaling ratio.

    Parameters
    ----------
    ratio : float — runtime_large / runtime_small (when p doubles)

    Returns
    -------
    str : one of 'filter', 'wrapper', 'evolutionary'
    """
    # TODO: apply the threshold rules from the docstring above
    raise NotImplementedError("Implement classify_by_scaling")


def fingerprint_selector(selector_fn, n_samples: int = 200) -> dict:
    """
    Characterise a selector by its runtime scaling signature.

    Parameters
    ----------
    selector_fn : callable (X, y) -> indices
    n_samples : int — number of rows in the test datasets

    Returns
    -------
    dict with keys: 'p_small', 'p_large', 'time_small', 'time_large', 'ratio', 'family'
    """
    # TODO:
    # 1. Create X_small (n_samples x 50) and X_large (n_samples x 100) using
    #    make_classification with matching n_samples and random_state=42
    # 2. Call scaling_ratio
    # 3. Call classify_by_scaling
    # 4. Return the fingerprint dict
    #
    # Hint: make_classification(n_samples, n_features, n_informative=min(10, n_features//5), random_state=42)
    raise NotImplementedError("Implement fingerprint_selector")


# ------- Self-Check 3 -------

def selfcheck_exercise_3():
    print("\n[Self-Check 3] Selection Method Fingerprinter")

    # Test measure_runtime
    t = measure_runtime(filter_selector, *make_benchmark_dataset(n_samples=100, n_features=20))
    assert isinstance(t, float) and t >= 0, f"measure_runtime must return a non-negative float, got {t}"
    print(f"  PASS  measure_runtime on small dataset: {t:.4f}s")

    # Test classify_by_scaling with explicit ratios
    assert classify_by_scaling(1.5) == 'filter', "ratio=1.5 → filter"
    assert classify_by_scaling(4.0) == 'wrapper', "ratio=4.0 → wrapper"
    assert classify_by_scaling(8.0) == 'evolutionary', "ratio=8.0 → evolutionary"
    print("  PASS  classify_by_scaling: 1.5→filter, 4.0→wrapper, 8.0→evolutionary")

    # Fingerprint the reference filter selector
    fp_filter = fingerprint_selector(filter_selector, n_samples=300)
    assert 'family' in fp_filter, "fingerprint_selector must return dict with 'family' key"
    assert fp_filter['family'] == 'filter', (
        f"filter_selector should classify as 'filter', got '{fp_filter['family']}'"
    )
    print(f"  PASS  filter_selector fingerprint: ratio={fp_filter['ratio']:.2f}, "
          f"family='{fp_filter['family']}'")

    # Fingerprint the reference wrapper selector
    fp_wrapper = fingerprint_selector(wrapper_selector, n_samples=100)
    assert fp_wrapper['family'] == 'wrapper', (
        f"wrapper_selector should classify as 'wrapper', got '{fp_wrapper['family']}'"
    )
    print(f"  PASS  wrapper_selector fingerprint: ratio={fp_wrapper['ratio']:.2f}, "
          f"family='{fp_wrapper['family']}'")

    print("  Exercise 3: ALL CHECKS PASSED")


# =============================================================================
# MAIN: Run all self-checks
# =============================================================================

if __name__ == "__main__":
    print("\nModule 0: Feature Selection Landscape — Self-Check Exercises")
    print("Complete each TODO, then run this file. Checks print PASS or raise AssertionError.\n")

    exercises = [
        ("Exercise 1: Method Cost Estimator",            selfcheck_exercise_1),
        ("Exercise 2: Greedy Univariate Filter Baseline", selfcheck_exercise_2),
        ("Exercise 3: Selection Method Fingerprinter",   selfcheck_exercise_3),
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
        print("  All exercises complete. Move to Module 1.")
    else:
        print(f"  {failed} exercise(s) still need implementation.")
    print("=" * 60)
