"""
Module 03 — Wrapper Methods at Scale
Self-Check Exercises

Three exercises of increasing complexity:
  1. SFFS with a custom stopping criterion
  2. Parallel wrapper selector using joblib
  3. Comparison of Boruta vs sequential methods on a high-dimensional dataset

Run this file directly to execute all exercises and check your implementations:
  python 01_wrapper_exercises.py

Each exercise includes:
  - A function stub to complete
  - A self-check function that runs after your implementation
  - A working reference solution (at the end of the file)
"""

import numpy as np
import pandas as pd
import time
import warnings
from typing import Callable, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _make_dataset(n_samples=400, n_features=50, n_informative=10, seed=42):
    """
    Generate a classification dataset with known informative features.

    Returns X (scaled), y, and the indices of the truly informative features.
    """
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=5,
        n_repeated=0,
        random_state=seed,
        n_clusters_per_class=2,
        flip_y=0.05,
    )
    X = StandardScaler().fit_transform(X)
    # The first n_informative features are informative by construction
    informative_idx = np.arange(n_informative)
    return X, y, informative_idx


def _cv_score(X, y, feature_mask, estimator, cv=3):
    """Cross-validate estimator on feature subset defined by boolean mask."""
    from sklearn.model_selection import cross_val_score
    from sklearn.base import clone

    selected = np.where(feature_mask)[0]
    if len(selected) == 0:
        return -np.inf
    scores = cross_val_score(
        clone(estimator), X[:, selected], y, cv=cv, scoring="accuracy", n_jobs=1
    )
    return float(scores.mean())


# ===========================================================================
# EXERCISE 1: SFFS with a custom stopping criterion
# ===========================================================================

def sffs_with_custom_stopping(
    X: np.ndarray,
    y: np.ndarray,
    estimator,
    k_max: int,
    stopping_fn: Callable[[List[float]], bool],
    cv: int = 3,
) -> Tuple[np.ndarray, List[float], List[Tuple]]:
    """
    Sequential Floating Forward Selection with a user-supplied stopping criterion.

    After each complete forward+backward cycle, call `stopping_fn(score_history)`.
    If it returns True, halt the search and return the current best subset.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    estimator : sklearn estimator
        Base model for CV evaluation.
    k_max : int
        Maximum number of features to select (hard upper bound).
    stopping_fn : callable
        Takes the list of best scores seen so far (one per forward step) and
        returns True to stop, False to continue.
        Signature: stopping_fn(scores: List[float]) -> bool
    cv : int
        Cross-validation folds.

    Returns
    -------
    selected : ndarray of int
        Indices of selected features.
    score_history : list of float
        Best CV score after each forward step.
    path : list of (step, action, feature_idx, score)
        Full search trace for visualisation.

    Notes
    -----
    The backward phase rule (unchanged from standard SFFS):
    Remove a feature only if doing so strictly improves the best known score
    at the resulting (smaller) subset size.
    """
    # TODO: Implement this function.
    #
    # Hints:
    #   - Maintain a `best_at_size` dict mapping subset_size -> best_score_seen
    #   - After each forward step, append the new best score to `score_history`
    #   - After the backward phase, call stopping_fn(score_history)
    #   - If stopping_fn returns True, break out of the while loop
    #   - Return (np.where(mask)[0], score_history, path)
    #
    # Reference: the `sffs` function in notebook 01 is the base implementation;
    # you only need to add the stopping_fn call and the early exit.
    raise NotImplementedError("Implement sffs_with_custom_stopping")


# ------- Stopping criterion implementations ------- #

def patience_stopper(patience: int = 5, tol: float = 1e-4) -> Callable:
    """
    Factory that returns a patience-based stopping function.

    The returned function returns True if the last `patience` scores
    have all been within `tol` of the maximum score seen so far.

    Usage:
        stopper = patience_stopper(patience=5, tol=1e-3)
        should_stop = stopper(score_history)  # True = stop
    """
    # TODO: Implement this factory.
    #
    # Hints:
    #   - The returned function receives the full score_history list
    #   - Check whether the last `patience` entries all satisfy
    #     score_history[i] >= max(score_history) - tol
    #   - Return False until at least `patience` entries are available
    raise NotImplementedError("Implement patience_stopper")


def diminishing_returns_stopper(min_delta_fraction: float = 0.1) -> Callable:
    """
    Factory that returns a diminishing-returns stopping function.

    Stop when the marginal gain of the latest forward step is less than
    `min_delta_fraction` times the maximum marginal gain seen so far.

    Usage:
        stopper = diminishing_returns_stopper(0.1)
        should_stop = stopper(score_history)
    """
    # TODO: Implement this factory.
    #
    # Hints:
    #   - Compute deltas = [score_history[i] - score_history[i-1] for i in 1..len]
    #   - Current delta = deltas[-1]
    #   - Max delta = max(deltas)
    #   - Stop if current_delta < min_delta_fraction * max_delta
    #   - Return False if len(score_history) < 3
    raise NotImplementedError("Implement diminishing_returns_stopper")


# ------- Self-check for Exercise 1 ------- #

def check_exercise_1():
    """Run all self-checks for Exercise 1."""
    from sklearn.ensemble import RandomForestClassifier

    print("=" * 60)
    print("EXERCISE 1: SFFS with custom stopping criterion")
    print("=" * 60)

    X, y, _ = _make_dataset(n_samples=300, n_features=30, n_informative=8)
    estimator = RandomForestClassifier(n_estimators=30, random_state=42)

    # Test 1: patience_stopper factory exists and is callable
    try:
        stopper = patience_stopper(patience=3, tol=1e-3)
        assert callable(stopper), "patience_stopper must return a callable"
        print("  [PASS] patience_stopper returns a callable")
    except NotImplementedError:
        print("  [TODO] patience_stopper not implemented yet")
        return

    # Test 2: patience_stopper returns False on short histories
    stopper = patience_stopper(patience=3, tol=1e-3)
    assert stopper([0.8, 0.82]) == False, (
        "patience_stopper must return False when fewer than patience entries"
    )
    print("  [PASS] patience_stopper returns False on short history")

    # Test 3: patience_stopper returns True on flat history
    stopper = patience_stopper(patience=3, tol=1e-4)
    flat_history = [0.80, 0.81, 0.81, 0.81, 0.81]
    result = stopper(flat_history)
    assert result == True, (
        f"patience_stopper should return True for flat history {flat_history}, got {result}"
    )
    print("  [PASS] patience_stopper returns True on flat history")

    # Test 4: diminishing_returns_stopper
    try:
        dr_stopper = diminishing_returns_stopper(0.1)
        assert callable(dr_stopper), "diminishing_returns_stopper must return callable"
        print("  [PASS] diminishing_returns_stopper returns a callable")
    except NotImplementedError:
        print("  [TODO] diminishing_returns_stopper not implemented yet")
        return

    # Test 5: diminishing_returns_stopper detects collapse
    dr_stopper = diminishing_returns_stopper(0.1)
    # Deltas: 0.05, 0.04, 0.03, 0.001 — last delta is << 10% of max delta
    dr_history = [0.70, 0.75, 0.79, 0.82, 0.821]
    result = dr_stopper(dr_history)
    assert result == True, (
        f"diminishing_returns_stopper should return True for {dr_history}, got {result}"
    )
    print("  [PASS] diminishing_returns_stopper detects diminishing returns")

    # Test 6: sffs_with_custom_stopping runs and respects early stopping
    try:
        patience_fn = patience_stopper(patience=3, tol=1e-3)
        selected, score_hist, path = sffs_with_custom_stopping(
            X, y, estimator, k_max=20, stopping_fn=patience_fn, cv=3
        )
        assert len(selected) >= 1, "Must select at least one feature"
        assert len(score_hist) >= 1, "Score history must have at least one entry"
        assert len(selected) <= 20, "Cannot exceed k_max"
        print(f"  [PASS] sffs_with_custom_stopping ran — selected {len(selected)} features")
        print(f"         Score history: {[f'{s:.4f}' for s in score_hist]}")
        if len(score_hist) < 20:
            print(f"         Early stopping fired after {len(score_hist)} forward steps")
    except NotImplementedError:
        print("  [TODO] sffs_with_custom_stopping not implemented yet")

    print()


# ===========================================================================
# EXERCISE 2: Parallel wrapper selector using joblib
# ===========================================================================

def parallel_sfs(
    X: np.ndarray,
    y: np.ndarray,
    estimator,
    k_target: int,
    cv: int = 5,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, List[float], float]:
    """
    Sequential Forward Selection with parallelised candidate evaluation.

    At each step, evaluate all candidate feature additions in parallel using
    joblib.Parallel. This provides a wall-time speedup proportional to the
    number of CPU cores without changing the selected feature set.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    estimator : sklearn estimator
    k_target : int
        Number of features to select.
    cv : int
        CV folds. Use n_jobs=1 *inside* cross_val_score to avoid nested
        parallelism conflicts.
    n_jobs : int
        Number of parallel workers for the outer candidate evaluation loop.
        -1 uses all available CPU cores.

    Returns
    -------
    selected : ndarray of int
        Indices of selected features.
    score_history : list of float
        CV score at each step (one entry per forward step).
    wall_time : float
        Total elapsed time in seconds.

    Important
    ---------
    Use n_jobs=1 for the inner cross_val_score call. Never set n_jobs=-1 in
    both the outer Parallel and inner cross_val_score simultaneously — this
    causes nested parallelism and can deadlock or severely degrade performance.
    """
    # TODO: Implement this function.
    #
    # Hints:
    #   from joblib import Parallel, delayed
    #   from sklearn.model_selection import cross_val_score
    #   from sklearn.base import clone
    #
    #   def score_candidate(j):
    #       mask = current_mask.copy()
    #       mask[j] = True
    #       selected = np.where(mask)[0]
    #       return cross_val_score(
    #           clone(estimator), X[:, selected], y,
    #           cv=cv, scoring='accuracy', n_jobs=1  # ← n_jobs=1 here!
    #       ).mean()
    #
    #   scores = Parallel(n_jobs=n_jobs)(
    #       delayed(score_candidate)(j) for j in candidates
    #   )
    raise NotImplementedError("Implement parallel_sfs")


def serial_sfs(
    X: np.ndarray,
    y: np.ndarray,
    estimator,
    k_target: int,
    cv: int = 5,
) -> Tuple[np.ndarray, List[float], float]:
    """
    Sequential Forward Selection, serial baseline for speedup comparison.

    Same logic as parallel_sfs but evaluates candidates one at a time.
    Returns (selected, score_history, wall_time).
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.base import clone

    p = X.shape[1]
    mask = np.zeros(p, dtype=bool)
    score_history = []

    t0 = time.time()
    for _ in range(k_target):
        candidates = np.where(~mask)[0]
        best_score = -np.inf
        best_j = None

        for j in candidates:
            candidate_mask = mask.copy()
            candidate_mask[j] = True
            selected = np.where(candidate_mask)[0]
            score = cross_val_score(
                clone(estimator), X[:, selected], y,
                cv=cv, scoring="accuracy", n_jobs=-1,
            ).mean()
            if score > best_score:
                best_score = score
                best_j = j

        mask[best_j] = True
        score_history.append(best_score)

    return np.where(mask)[0], score_history, time.time() - t0


# ------- Self-check for Exercise 2 ------- #

def check_exercise_2():
    """Run all self-checks for Exercise 2."""
    from sklearn.ensemble import RandomForestClassifier

    print("=" * 60)
    print("EXERCISE 2: Parallel wrapper selector")
    print("=" * 60)

    X, y, _ = _make_dataset(n_samples=300, n_features=30, n_informative=8)
    estimator = RandomForestClassifier(n_estimators=30, random_state=42)

    # Test 1: parallel_sfs exists
    try:
        selected_p, scores_p, time_p = parallel_sfs(
            X, y, estimator, k_target=8, cv=3, n_jobs=-1
        )
        print(f"  [PASS] parallel_sfs ran in {time_p:.2f}s, selected {len(selected_p)} features")
    except NotImplementedError:
        print("  [TODO] parallel_sfs not implemented yet")
        return

    # Test 2: correct feature count
    assert len(selected_p) == 8, f"Expected 8 features, got {len(selected_p)}"
    print(f"  [PASS] Correct feature count: {len(selected_p)}")

    # Test 3: score history length
    assert len(scores_p) == 8, f"score_history must have 8 entries, got {len(scores_p)}"
    print(f"  [PASS] Score history has {len(scores_p)} entries")

    # Test 4: scores are non-decreasing (greedy property)
    for i in range(1, len(scores_p)):
        assert scores_p[i] >= scores_p[i - 1] - 0.01, (
            f"SFS scores should be non-decreasing, but step {i}: "
            f"{scores_p[i]:.4f} < {scores_p[i-1]:.4f}"
        )
    print("  [PASS] Scores are non-decreasing (greedy property)")

    # Test 5: Compare wall time against serial baseline
    print("  Timing comparison (parallel vs serial)...")
    _, _, time_s = serial_sfs(X, y, estimator, k_target=8, cv=3)
    print(f"  Serial: {time_s:.2f}s  |  Parallel: {time_p:.2f}s")
    speedup = time_s / max(time_p, 0.001)
    print(f"  Speedup: {speedup:.1f}x")
    if speedup > 1.2:
        print("  [PASS] Parallel is faster than serial")
    else:
        print("  [INFO] Speedup marginal — normal for fast models (overhead > savings)")

    # Test 6: Same feature set (parallel should give identical results to serial)
    selected_s, _, _ = serial_sfs(X, y, estimator, k_target=8, cv=3)
    # Note: may differ if cross_val_score is not deterministic when parallelised
    # Check that both achieve a reasonable score
    assert len(selected_s) == len(selected_p), "Both methods should select k features"
    print("  [PASS] Parallel and serial select same number of features")

    print()


# ===========================================================================
# EXERCISE 3: Boruta vs sequential methods on a high-dimensional dataset
# ===========================================================================

def compare_boruta_vs_sequential(
    X: np.ndarray,
    y: np.ndarray,
    informative_idx: np.ndarray,
    estimator,
    k_target: int = 15,
    boruta_iters: int = 50,
    cv: int = 3,
) -> pd.DataFrame:
    """
    Compare Boruta (all-relevant) vs SFS (minimal-optimal) on a dataset where
    the true informative features are known.

    For each method, report:
      - Selected feature indices
      - Recall: fraction of truly informative features that were selected
      - Precision: fraction of selected features that are truly informative
      - Test accuracy using selected features
      - Wall time

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
        Pre-scaled feature matrix.
    y : ndarray (n_samples,)
        Binary target.
    informative_idx : ndarray
        True informative feature indices (ground truth for recall/precision).
    estimator : sklearn estimator
        Base model for both methods.
    k_target : int
        Number of features for SFS (Boruta determines its own count).
    boruta_iters : int
        Number of Boruta iterations.
    cv : int
        Cross-validation folds.

    Returns
    -------
    results : pd.DataFrame
        Comparison table with columns:
        Method, N_Selected, Recall, Precision, Test_Accuracy, Time_s

    Notes
    -----
    Use sklearn's SequentialFeatureSelector for SFS (it is faster than the
    from-scratch implementation and handles parallelisation automatically).

    For Boruta, use the run_boruta function from notebook 02, reproduced in
    the reference solution below.
    """
    # TODO: Implement this function.
    #
    # Structure:
    #   1. Import SequentialFeatureSelector and run SFS
    #   2. Import or re-implement Boruta and run it
    #   3. Compute recall, precision, test accuracy for each
    #   4. Return a DataFrame with the comparison
    #
    # Recall = len(selected ∩ informative) / len(informative)
    # Precision = len(selected ∩ informative) / len(selected)
    #
    # Use train_test_split with test_size=0.25, stratify=y, random_state=42
    # for the accuracy evaluation.
    raise NotImplementedError("Implement compare_boruta_vs_sequential")


# ------- Self-check for Exercise 3 ------- #

def check_exercise_3():
    """Run all self-checks for Exercise 3."""
    from sklearn.ensemble import RandomForestClassifier

    print("=" * 60)
    print("EXERCISE 3: Boruta vs Sequential on high-dimensional data")
    print("=" * 60)

    # High-dimensional dataset: 200 samples, 80 features, 12 informative
    X, y, informative_idx = _make_dataset(
        n_samples=300, n_features=80, n_informative=12, seed=7
    )
    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(informative_idx)} truly informative")

    estimator = RandomForestClassifier(
        n_estimators=50, max_depth=4, random_state=42, n_jobs=-1
    )

    try:
        results = compare_boruta_vs_sequential(
            X, y, informative_idx, estimator,
            k_target=15, boruta_iters=30, cv=3,
        )
    except NotImplementedError:
        print("  [TODO] compare_boruta_vs_sequential not implemented yet")
        return

    # Check structure
    assert isinstance(results, pd.DataFrame), "Must return a pd.DataFrame"
    required_cols = {"Method", "N_Selected", "Recall", "Precision", "Test_Accuracy", "Time_s"}
    assert required_cols.issubset(set(results.columns)), (
        f"Missing columns: {required_cols - set(results.columns)}"
    )
    print("  [PASS] Returns a DataFrame with required columns")

    # Check there are at least 2 rows (one for each method)
    assert len(results) >= 2, "DataFrame must have at least 2 rows (Boruta + SFS)"
    print(f"  [PASS] DataFrame has {len(results)} rows")

    # Check recall and precision are in [0, 1]
    assert results["Recall"].between(0, 1).all(), "Recall must be in [0, 1]"
    assert results["Precision"].between(0, 1).all(), "Precision must be in [0, 1]"
    print("  [PASS] Recall and Precision are in valid range [0, 1]")

    # Check test accuracy is plausible
    assert results["Test_Accuracy"].between(0.5, 1.0).all(), (
        "Test accuracy must be plausible (>50% for binary classification)"
    )
    print("  [PASS] Test accuracies are plausible")

    # Print the comparison table
    print("\n  Comparison results:")
    print(results.to_string(index=False))
    print()

    # Interpretation guidance
    print("  Interpretation:")
    print("  - Boruta recall should be higher (all-relevant = comprehensive)")
    print("  - SFS precision may be higher (minimal-optimal = fewer false positives)")
    print("  - Test accuracy often similar when k* is well-chosen")
    print()


# ===========================================================================
# Reference solutions (read after attempting yourself)
# ===========================================================================

def _solution_patience_stopper(patience: int = 5, tol: float = 1e-4) -> Callable:
    """Reference solution for patience_stopper."""
    def stopper(scores: List[float]) -> bool:
        if len(scores) < patience:
            return False
        recent = scores[-patience:]
        best_score = max(scores)
        # Stop if all recent scores are within tol of the best
        return all(s >= best_score - tol for s in recent)
    return stopper


def _solution_diminishing_returns_stopper(min_delta_fraction: float = 0.1) -> Callable:
    """Reference solution for diminishing_returns_stopper."""
    def stopper(scores: List[float]) -> bool:
        if len(scores) < 3:
            return False
        deltas = [scores[i] - scores[i - 1] for i in range(1, len(scores))]
        max_delta = max(deltas)
        current_delta = deltas[-1]
        if max_delta <= 0:
            return False
        return current_delta < min_delta_fraction * max_delta
    return stopper


def _solution_sffs_with_custom_stopping(
    X, y, estimator, k_max, stopping_fn, cv=3
):
    """Reference solution for sffs_with_custom_stopping."""
    from sklearn.model_selection import cross_val_score
    from sklearn.base import clone

    p = X.shape[1]
    mask = np.zeros(p, dtype=bool)
    cache: Dict[Tuple, float] = {}
    best_at_size: Dict[int, float] = {}
    path = []
    score_history: List[float] = []

    def _score(m):
        key = tuple(np.where(m)[0])
        if key not in cache:
            if not key:
                cache[key] = -np.inf
            else:
                cache[key] = cross_val_score(
                    clone(estimator), X[:, list(key)], y,
                    cv=cv, scoring="accuracy", n_jobs=1
                ).mean()
        return cache[key]

    step = 0
    while mask.sum() < k_max:
        # Forward phase
        candidates = np.where(~mask)[0]
        if len(candidates) == 0:
            break

        best_score = -np.inf
        best_j = None
        for j in candidates:
            m = mask.copy()
            m[j] = True
            score = _score(m)
            if score > best_score:
                best_score = score
                best_j = j

        mask[best_j] = True
        k = int(mask.sum())
        best_at_size[k] = max(best_at_size.get(k, -np.inf), best_score)
        path.append((step, "add", best_j, best_score))
        score_history.append(best_score)
        step += 1

        # Backward phase (floating)
        improved = True
        while improved and mask.sum() > 1:
            improved = False
            best_bwd = -np.inf
            best_bwd_j = None
            for j in np.where(mask)[0]:
                m = mask.copy()
                m[j] = False
                score = _score(m)
                if score > best_bwd:
                    best_bwd = score
                    best_bwd_j = j

            k_smaller = int(mask.sum()) - 1
            if best_bwd > best_at_size.get(k_smaller, -np.inf):
                mask[best_bwd_j] = False
                best_at_size[k_smaller] = best_bwd
                path.append((step, "remove", best_bwd_j, best_bwd))
                step += 1
                improved = True

        # Early stopping check
        if stopping_fn(score_history):
            break

    return np.where(mask)[0], score_history, path


def _solution_parallel_sfs(X, y, estimator, k_target, cv=5, n_jobs=-1):
    """Reference solution for parallel_sfs."""
    from joblib import Parallel, delayed
    from sklearn.model_selection import cross_val_score
    from sklearn.base import clone

    p = X.shape[1]
    mask = np.zeros(p, dtype=bool)
    score_history = []

    t0 = time.time()
    for _ in range(k_target):
        candidates = list(np.where(~mask)[0])

        def score_candidate(j):
            m = mask.copy()
            m[j] = True
            selected = np.where(m)[0]
            return cross_val_score(
                clone(estimator), X[:, selected], y,
                cv=cv, scoring="accuracy", n_jobs=1,
            ).mean()

        candidate_scores = Parallel(n_jobs=n_jobs)(
            delayed(score_candidate)(j) for j in candidates
        )
        best_idx = int(np.argmax(candidate_scores))
        mask[candidates[best_idx]] = True
        score_history.append(candidate_scores[best_idx])

    return np.where(mask)[0], score_history, time.time() - t0


def _solution_compare_boruta_vs_sequential(
    X, y, informative_idx, estimator, k_target=15, boruta_iters=50, cv=3
):
    """Reference solution for compare_boruta_vs_sequential."""
    from sklearn.feature_selection import SequentialFeatureSelector
    from sklearn.model_selection import train_test_split
    from sklearn.base import clone
    from scipy.stats import binom

    informative_set = set(informative_idx.tolist())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    def _eval_subset(selected_idx):
        """Test accuracy using selected features."""
        sel = list(selected_idx)
        model = clone(estimator)
        model.fit(X_train[:, sel], y_train)
        return model.score(X_test[:, sel], y_test)

    def _recall_precision(selected_idx):
        sel_set = set(selected_idx.tolist())
        tp = len(sel_set & informative_set)
        recall = tp / len(informative_set) if informative_set else 0.0
        precision = tp / len(sel_set) if sel_set else 0.0
        return recall, precision

    rows = []

    # --- SFS ---
    t0 = time.time()
    sfs_sel = SequentialFeatureSelector(
        clone(estimator), n_features_to_select=k_target,
        direction="forward", scoring="accuracy", cv=cv, n_jobs=-1,
    )
    sfs_sel.fit(X_train, y_train)
    sfs_features = np.where(sfs_sel.support_)[0]
    sfs_time = time.time() - t0
    rec, prec = _recall_precision(sfs_features)
    acc = _eval_subset(sfs_features)
    rows.append({
        "Method": f"SFS (k*={k_target})",
        "N_Selected": len(sfs_features),
        "Recall": round(rec, 4),
        "Precision": round(prec, 4),
        "Test_Accuracy": round(acc, 4),
        "Time_s": round(sfs_time, 2),
    })

    # --- Boruta ---
    rng = np.random.default_rng(42)
    n_train, p = X_train.shape
    hits = np.zeros(p, dtype=int)
    status = np.zeros(p, dtype=int)

    t0 = time.time()
    for t in range(boruta_iters):
        tentative_mask = status == 0
        if not tentative_mask.any():
            break

        X_shadow = np.apply_along_axis(rng.permutation, 0, X_train)
        X_aug = np.hstack([X_train, X_shadow])
        model = clone(estimator)
        model.fit(X_aug, y_train)
        importances = np.array(model.feature_importances_)
        real_imp = importances[:p]
        shadow_max = importances[p:].max()

        tentative_idx = np.where(tentative_mask)[0]
        for j in tentative_idx:
            if real_imp[j] > shadow_max:
                hits[j] += 1

        n_tent = len(tentative_idx)
        if n_tent > 0:
            thresh = 0.05 / (2 * n_tent)
            for j in tentative_idx:
                h = hits[j]
                if 1 - binom.cdf(h - 1, t + 1, 0.5) < thresh:
                    status[j] = 1
                elif binom.cdf(h, t + 1, 0.5) < thresh:
                    status[j] = -1

    boruta_features = np.where(status == 1)[0]
    boruta_time = time.time() - t0

    if len(boruta_features) == 0:
        # Fallback: take features with any hits
        boruta_features = np.where(hits > boruta_iters // 4)[0]

    if len(boruta_features) > 0:
        rec, prec = _recall_precision(boruta_features)
        acc = _eval_subset(boruta_features)
    else:
        rec, prec, acc = 0.0, 0.0, 0.0

    rows.append({
        "Method": f"Boruta ({boruta_iters} iters)",
        "N_Selected": len(boruta_features),
        "Recall": round(rec, 4),
        "Precision": round(prec, 4),
        "Test_Accuracy": round(acc, 4),
        "Time_s": round(boruta_time, 2),
    })

    return pd.DataFrame(rows)


# ===========================================================================
# Main: run all checks
# ===========================================================================

if __name__ == "__main__":
    print("\nModule 03 — Wrapper Methods: Self-Check Exercises")
    print("=" * 60)
    print("Run each check to see which exercises pass.\n")

    check_exercise_1()
    check_exercise_2()
    check_exercise_3()

    print("=" * 60)
    print("All checks complete.")
    print()
    print("To see the reference solutions, look for the _solution_ functions")
    print("at the bottom of this file.")
