"""
Module 10 — Ensemble & Hybrid Feature Selection: Self-Check Exercises
=====================================================================

Three exercises covering the core skills from this module:

Exercise 1: Borda count rank aggregation
    Implement Borda count from scratch, test on synthetic rankings.

Exercise 2: Hybrid filter-GA pipeline
    Build a configurable filter → GA cascade pipeline and compare
    it to a direct GA on the full feature space.

Exercise 3: Simple meta-learning recommender
    Compute dataset meta-features and train a classifier to recommend
    the best feature selector.

Run this file directly to check all exercises:
    python exercises/01_ensemble_exercises.py

Each exercise has:
  - A problem description
  - A stub to complete
  - Auto-tests that print PASS or FAIL with helpful messages
"""

import numpy as np
import pandas as pd
import random
import warnings
from collections import Counter
from itertools import combinations

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Shared utilities used across exercises
# ---------------------------------------------------------------------------

def make_synthetic_dataset(n: int = 500, p: int = 50,
                             n_informative: int = 10,
                             random_state: int = 42) -> tuple:
    """Generate a reproducible classification dataset."""
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=n,
        n_features=p,
        n_informative=n_informative,
        n_redundant=5,
        flip_y=0.05,
        random_state=random_state
    )
    return X, y


# ===========================================================================
# EXERCISE 1: Borda Count Rank Aggregation
# ===========================================================================

def exercise_1_description():
    """
    Exercise 1: Implement Borda Count Rank Aggregation
    ---------------------------------------------------
    Implement `borda_count_aggregation(rankings)` that takes a list of rankings
    (each ranking is a list of feature indices from best to worst) and returns
    the aggregate ranking as a list of feature indices (best first).

    Borda count rules:
    - Feature ranked 1st out of p features receives (p - 1) points
    - Feature ranked 2nd receives (p - 2) points
    - Feature ranked last receives 0 points
    - Sum points across all M rankings
    - Final aggregate ranking: sort by total points (descending)

    Example:
        p = 4 features (0, 1, 2, 3)
        Method A ranking: [2, 0, 1, 3]  (feature 2 is best, feature 3 is worst)
        Method B ranking: [0, 2, 3, 1]
        Method C ranking: [2, 1, 0, 3]

        Borda scores from Method A: {2: 3, 0: 2, 1: 1, 3: 0}
        Borda scores from Method B: {0: 3, 2: 2, 3: 1, 1: 0}
        Borda scores from Method C: {2: 3, 1: 2, 0: 1, 3: 0}

        Total Borda scores: {2: 8, 0: 6, 1: 3, 3: 1}
        Aggregate ranking: [2, 0, 1, 3]
    """
    print(exercise_1_description.__doc__)


def borda_count_aggregation(rankings: list[list[int]]) -> list[int]:
    """
    Aggregate multiple feature rankings using Borda count.

    Parameters
    ----------
    rankings : list of lists, shape (M, p)
        M rankings of p features. Each inner list contains feature indices
        ordered from best (index 0) to worst (index p-1).
        All rankings must contain exactly the same p features.

    Returns
    -------
    list of int
        Feature indices sorted from highest to lowest aggregate Borda score.
        Length equals p (complete ranking of all features).

    Raises
    ------
    ValueError
        If rankings is empty or contains rankings of different lengths.
    """
    # YOUR CODE HERE
    # ---------------
    # Hint: Create a scores dict initialised to 0 for each feature.
    # Iterate over rankings; for each ranking, add (p - 1 - rank_idx) points
    # to the feature at position rank_idx.
    # Sort features by score in descending order.
    raise NotImplementedError("Implement borda_count_aggregation")


def test_exercise_1():
    """Auto-tests for Exercise 1: Borda Count Aggregation."""
    print("\n" + "=" * 60)
    print("EXERCISE 1: Borda Count Rank Aggregation")
    print("=" * 60)

    # --- Test 1.1: Basic correctness on the example from the docstring ---
    rankings = [
        [2, 0, 1, 3],   # Method A
        [0, 2, 3, 1],   # Method B
        [2, 1, 0, 3],   # Method C
    ]
    result = borda_count_aggregation(rankings)
    expected = [2, 0, 1, 3]

    assert result == expected, (
        f"Test 1.1 FAILED: Expected {expected}, got {result}. "
        f"Check that you are scoring: feature at rank 0 gets p-1 points, "
        f"feature at rank p-1 gets 0 points."
    )
    print("Test 1.1 PASSED: Basic correctness on 3-method, 4-feature example.")

    # --- Test 1.2: Single ranking (should return input unchanged) ---
    single_ranking = [4, 2, 0, 1, 3]
    result_single = borda_count_aggregation([single_ranking])
    assert result_single == single_ranking, (
        f"Test 1.2 FAILED: Single ranking should pass through unchanged. "
        f"Expected {single_ranking}, got {result_single}."
    )
    print("Test 1.2 PASSED: Single ranking passes through unchanged.")

    # --- Test 1.3: All methods agree (unanimous ranking) ---
    unanimous = [[0, 1, 2, 3, 4]] * 5
    result_unanimous = borda_count_aggregation(unanimous)
    assert result_unanimous == [0, 1, 2, 3, 4], (
        f"Test 1.3 FAILED: When all methods agree, aggregate should match. "
        f"Expected [0, 1, 2, 3, 4], got {result_unanimous}."
    )
    print("Test 1.3 PASSED: Unanimous agreement produces expected ranking.")

    # --- Test 1.4: Output is a complete ranking (all features present) ---
    np.random.seed(42)
    p = 20
    n_methods = 7
    rand_rankings = [np.random.permutation(p).tolist() for _ in range(n_methods)]
    result_rand = borda_count_aggregation(rand_rankings)

    assert isinstance(result_rand, list), (
        "Test 1.4 FAILED: Output must be a list."
    )
    assert len(result_rand) == p, (
        f"Test 1.4 FAILED: Output length must equal p={p}, got {len(result_rand)}."
    )
    assert set(result_rand) == set(range(p)), (
        f"Test 1.4 FAILED: Output must contain all {p} feature indices."
    )
    print(f"Test 1.4 PASSED: Output is a complete ranking of all {p} features.")

    # --- Test 1.5: Diverse rankings reduce variance vs individual methods ---
    # Create 5 noisy rankings of a dataset with a known "true" top feature (0)
    # Borda count should rank feature 0 at the top more reliably than any single method
    true_ranking = list(range(15))   # feature 0 is truly best
    noisy_rankings = []
    rng = random.Random(7)
    for _ in range(10):
        # Perturb the true ranking by swapping adjacent elements
        ranking = true_ranking.copy()
        for _ in range(5):  # 5 random swaps
            i = rng.randint(0, 13)
            ranking[i], ranking[i+1] = ranking[i+1], ranking[i]
        noisy_rankings.append(ranking)

    aggregate = borda_count_aggregation(noisy_rankings)
    # Feature 0 (best in the true ranking) should appear in the top-3 of the aggregate
    assert aggregate.index(0) < 3, (
        f"Test 1.5 FAILED: Feature 0 (truly best) should be in top-3 of aggregate, "
        f"but got rank {aggregate.index(0) + 1}. Borda count should smooth out noise."
    )
    print(f"Test 1.5 PASSED: Borda count correctly identifies best feature "
          f"(rank {aggregate.index(0) + 1}) despite noisy individual rankings.")

    print("\nAll Exercise 1 tests PASSED.")
    return True


# ===========================================================================
# EXERCISE 2: Hybrid Filter-GA Pipeline
# ===========================================================================

def exercise_2_description():
    """
    Exercise 2: Hybrid Filter → GA Pipeline
    ----------------------------------------
    Implement `HybridFilterGA`, a configurable two-stage feature selection
    pipeline:

    Stage 1 (Filter): Score all features with mutual information.
                      Keep the top `filter_k` features.

    Stage 2 (GA):     Run a binary GA on the `filter_k` candidate features.
                      Fitness = 5-fold cross-validated balanced accuracy of
                      a GradientBoostingClassifier.
                      Return the final selected feature indices.

    The class should have:
    - __init__(filter_k, pop_size, n_generations, random_state)
    - fit(X, y) -> self       (runs both stages, stores results)
    - transform(X) -> X_sel   (selects the final features)
    - fit_transform(X, y) -> X_sel

    Attributes after fitting:
    - selected_features_  : list of final feature indices (into original X)
    - filter_candidates_  : list of feature indices after filter stage
    - best_fitness_       : float, best CV score achieved by GA
    - fitness_history_    : list of best fitness per generation
    """
    print(exercise_2_description.__doc__)


class HybridFilterGA:
    """
    Two-stage feature selection: Filter → GA.

    Parameters
    ----------
    filter_k : int
        Number of features to keep after the filter stage (MI-based).
    pop_size : int
        GA population size.
    n_generations : int
        Number of GA generations.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(self, filter_k: int = 30, pop_size: int = 20,
                 n_generations: int = 30, random_state: int = 42):
        self.filter_k = filter_k
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.random_state = random_state

        # Attributes set by fit()
        self.selected_features_ = None
        self.filter_candidates_ = None
        self.best_fitness_ = None
        self.fitness_history_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'HybridFilterGA':
        """
        Fit the two-stage pipeline on (X, y).

        Stage 1: Score features with mutual_info_classif.
                 Set self.filter_candidates_ = top filter_k indices.

        Stage 2: Run a binary GA on the filter_k candidates.
                 Fitness = cross_val_score with GradientBoostingClassifier, cv=5.
                 Set self.selected_features_, self.best_fitness_,
                 self.fitness_history_.

        Returns self.
        """
        # YOUR CODE HERE
        # ---------------
        # Stage 1: filter
        # Stage 2: GA
        raise NotImplementedError("Implement HybridFilterGA.fit()")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return X with only the selected features."""
        if self.selected_features_ is None:
            raise RuntimeError("Call fit() before transform().")
        return X[:, self.selected_features_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


def test_exercise_2():
    """Auto-tests for Exercise 2: Hybrid Filter-GA Pipeline."""
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    print("\n" + "=" * 60)
    print("EXERCISE 2: Hybrid Filter-GA Pipeline")
    print("=" * 60)

    X, y = make_synthetic_dataset(n=400, p=50, n_informative=10, random_state=42)

    # --- Test 2.1: fit() runs without error ---
    pipeline = HybridFilterGA(filter_k=20, pop_size=10, n_generations=10,
                               random_state=42)
    try:
        pipeline.fit(X, y)
    except NotImplementedError:
        print("Test 2.1 FAILED: HybridFilterGA.fit() not implemented yet.")
        return False
    except Exception as e:
        print(f"Test 2.1 FAILED: fit() raised an unexpected exception: {e}")
        return False
    print("Test 2.1 PASSED: fit() runs without error.")

    # --- Test 2.2: filter_candidates_ has correct size ---
    assert pipeline.filter_candidates_ is not None, (
        "Test 2.2 FAILED: filter_candidates_ is None after fit()."
    )
    assert len(pipeline.filter_candidates_) == 20, (
        f"Test 2.2 FAILED: filter_candidates_ should have 20 features, "
        f"got {len(pipeline.filter_candidates_)}."
    )
    assert all(0 <= i < 50 for i in pipeline.filter_candidates_), (
        "Test 2.2 FAILED: filter_candidates_ contains invalid feature indices."
    )
    print(f"Test 2.2 PASSED: filter_candidates_ has {len(pipeline.filter_candidates_)} features.")

    # --- Test 2.3: selected_features_ is a non-empty subset of filter_candidates_ ---
    assert pipeline.selected_features_ is not None, (
        "Test 2.3 FAILED: selected_features_ is None after fit()."
    )
    assert len(pipeline.selected_features_) > 0, (
        "Test 2.3 FAILED: selected_features_ is empty."
    )
    assert set(pipeline.selected_features_).issubset(set(pipeline.filter_candidates_)), (
        "Test 2.3 FAILED: selected_features_ must be a subset of filter_candidates_. "
        "The GA should select from the pre-screened candidates only."
    )
    print(f"Test 2.3 PASSED: selected_features_ ({len(pipeline.selected_features_)} features) "
          f"is a subset of filter_candidates_ (20 features).")

    # --- Test 2.4: transform() returns correct shape ---
    X_selected = pipeline.transform(X)
    assert X_selected.shape == (len(X), len(pipeline.selected_features_)), (
        f"Test 2.4 FAILED: transform() output shape {X_selected.shape} != "
        f"expected ({len(X)}, {len(pipeline.selected_features_)})."
    )
    print(f"Test 2.4 PASSED: transform() returns shape {X_selected.shape}.")

    # --- Test 2.5: best_fitness_ is a reasonable accuracy value ---
    assert isinstance(pipeline.best_fitness_, float), (
        "Test 2.5 FAILED: best_fitness_ must be a float."
    )
    assert 0.5 <= pipeline.best_fitness_ <= 1.0, (
        f"Test 2.5 FAILED: best_fitness_ = {pipeline.best_fitness_:.4f} is not in [0.5, 1.0]. "
        "This suggests the fitness function is returning implausible values."
    )
    print(f"Test 2.5 PASSED: best_fitness_ = {pipeline.best_fitness_:.4f} (valid range).")

    # --- Test 2.6: fitness_history_ shows convergence ---
    assert pipeline.fitness_history_ is not None, (
        "Test 2.6 FAILED: fitness_history_ is None."
    )
    assert len(pipeline.fitness_history_) == 10, (
        f"Test 2.6 FAILED: fitness_history_ length should equal n_generations=10, "
        f"got {len(pipeline.fitness_history_)}."
    )
    # Fitness should be non-decreasing (GA with elitism)
    hist = pipeline.fitness_history_
    assert all(hist[i] <= hist[i+1] + 0.01 for i in range(len(hist) - 1)), (
        "Test 2.6 FAILED: fitness_history_ is not monotonically non-decreasing. "
        "Use elitism to ensure the best solution is never lost."
    )
    print(f"Test 2.6 PASSED: fitness_history_ shows convergence "
          f"({hist[0]:.3f} → {hist[-1]:.3f}).")

    # --- Test 2.7: filter pre-screening reduces GA cost ---
    # The filter_k=20 pipeline should finish faster than filter_k=50 (no filtering)
    import time

    t0 = time.perf_counter()
    pipeline_filtered = HybridFilterGA(filter_k=20, pop_size=10,
                                        n_generations=10, random_state=42)
    pipeline_filtered.fit(X, y)
    t_filtered = time.perf_counter() - t0

    t0 = time.perf_counter()
    pipeline_full = HybridFilterGA(filter_k=50, pop_size=10,
                                    n_generations=10, random_state=42)
    pipeline_full.fit(X, y)
    t_full = time.perf_counter() - t0

    print(f"Test 2.7: Filtered ({pipeline_filtered.filter_k} features): {t_filtered:.2f}s | "
          f"No filter (50 features): {t_full:.2f}s")
    # We just check both finish; timing is environment-dependent
    print("Test 2.7 PASSED: Both pipeline configurations completed successfully.")

    print("\nAll Exercise 2 tests PASSED.")
    return True


# ===========================================================================
# EXERCISE 3: Simple Meta-Learning Recommender
# ===========================================================================

def exercise_3_description():
    """
    Exercise 3: Simple Meta-Learning Recommender
    ---------------------------------------------
    Implement `SimpleMetaRecommender` that:

    1. compute_meta_features(X, y) -> dict
       Computes three meta-features:
       - log_ratio_n_p : log(1 + n/p)
       - mean_abs_correlation : mean |Pearson r| between all feature pairs
                                (use a sample of max 50 features for speed)
       - frac_zero_mi : fraction of features with MI < 0.01 with target
                         (use a sample of max 30 features)

    2. fit(datasets_list) -> self
       datasets_list = list of {'X': ..., 'y': ..., 'best_selector': ...}
       Computes meta-features for each dataset,
       trains a RandomForestClassifier to predict 'best_selector'.

    3. recommend(X, y, top_n=3) -> list of (selector_name, probability) tuples
       Computes meta-features for (X, y),
       returns top_n selector recommendations sorted by probability.
    """
    print(exercise_3_description.__doc__)


class SimpleMetaRecommender:
    """
    Meta-learning recommender: predict best feature selector from dataset properties.

    Training:
        fit(datasets_list) takes a list of {'X', 'y', 'best_selector'} dicts.

    Inference:
        recommend(X, y) returns ranked list of (selector_name, probability) tuples.
    """

    META_FEATURE_NAMES = ['log_ratio_n_p', 'mean_abs_correlation', 'frac_zero_mi']

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.meta_model = None
        self.label_encoder = None
        self._is_fitted = False

    def compute_meta_features(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Compute three meta-features for dataset (X, y).

        Returns
        -------
        dict with keys: 'log_ratio_n_p', 'mean_abs_correlation', 'frac_zero_mi'
        """
        # YOUR CODE HERE
        # ---------------
        # Hint 1: log_ratio_n_p = np.log1p(n / p)
        # Hint 2: mean_abs_correlation: np.abs(np.corrcoef(X_sample.T))
        #         Use only the upper triangle (k=1) to avoid self-correlation.
        # Hint 3: frac_zero_mi: use sklearn's mutual_info_classif on a sample of features.
        raise NotImplementedError("Implement compute_meta_features()")

    def fit(self, datasets_list: list[dict]) -> 'SimpleMetaRecommender':
        """
        Train the meta-model on a list of (dataset, best_selector) pairs.

        Parameters
        ----------
        datasets_list : list of dicts, each with keys 'X', 'y', 'best_selector'
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder

        # YOUR CODE HERE
        # ---------------
        # For each dataset in datasets_list:
        #   - compute_meta_features(d['X'], d['y'])
        #   - record d['best_selector']
        # Encode labels with LabelEncoder
        # Train RandomForestClassifier on the meta-features
        # Store in self.meta_model, self.label_encoder
        raise NotImplementedError("Implement SimpleMetaRecommender.fit()")

    def recommend(self, X: np.ndarray, y: np.ndarray,
                   top_n: int = 3) -> list[tuple[str, float]]:
        """
        Recommend top_n selectors for a new dataset.

        Returns
        -------
        list of (selector_name, probability) tuples, sorted by probability desc.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        # YOUR CODE HERE
        # ---------------
        # Compute meta-features for (X, y)
        # Get probability predictions from meta_model
        # Map probabilities to class names via label_encoder
        # Return top_n sorted by probability (descending)
        raise NotImplementedError("Implement SimpleMetaRecommender.recommend()")


def test_exercise_3():
    """Auto-tests for Exercise 3: Simple Meta-Learning Recommender."""
    from sklearn.datasets import load_breast_cancer, load_wine, load_iris
    from sklearn.datasets import make_classification

    print("\n" + "=" * 60)
    print("EXERCISE 3: Simple Meta-Learning Recommender")
    print("=" * 60)

    # --- Test 3.1: compute_meta_features() returns correct keys ---
    X_test, y_test = make_synthetic_dataset(n=200, p=20)
    recommender = SimpleMetaRecommender(random_state=42)

    try:
        mf = recommender.compute_meta_features(X_test, y_test)
    except NotImplementedError:
        print("Test 3.1 FAILED: compute_meta_features() not implemented yet.")
        return False
    except Exception as e:
        print(f"Test 3.1 FAILED: compute_meta_features() raised: {e}")
        return False

    for key in SimpleMetaRecommender.META_FEATURE_NAMES:
        assert key in mf, (
            f"Test 3.1 FAILED: meta-feature '{key}' missing from output dict. "
            f"Got keys: {list(mf.keys())}"
        )
    print(f"Test 3.1 PASSED: All 3 meta-feature keys present: {list(mf.keys())}.")

    # --- Test 3.2: meta-feature values are in valid ranges ---
    assert mf['log_ratio_n_p'] > 0, (
        f"Test 3.2 FAILED: log_ratio_n_p={mf['log_ratio_n_p']:.4f} should be > 0. "
        "n/p ratio with n > p always gives log(1 + n/p) > 0."
    )
    assert 0 <= mf['mean_abs_correlation'] <= 1, (
        f"Test 3.2 FAILED: mean_abs_correlation={mf['mean_abs_correlation']:.4f} "
        "must be in [0, 1]. Check you are taking absolute value of correlations."
    )
    assert 0 <= mf['frac_zero_mi'] <= 1, (
        f"Test 3.2 FAILED: frac_zero_mi={mf['frac_zero_mi']:.4f} must be in [0, 1]."
    )
    print(f"Test 3.2 PASSED: Meta-feature values are in valid ranges: "
          f"log_ratio_n_p={mf['log_ratio_n_p']:.3f}, "
          f"mean_abs_corr={mf['mean_abs_correlation']:.3f}, "
          f"frac_zero_mi={mf['frac_zero_mi']:.3f}.")

    # --- Test 3.3: High-p, noisy dataset has higher frac_zero_mi ---
    # A dataset with mostly noise features should have high frac_zero_mi
    X_noisy, y_noisy = make_classification(n_samples=300, n_features=50,
                                            n_informative=3, flip_y=0.05,
                                            random_state=42)
    mf_noisy = recommender.compute_meta_features(X_noisy, y_noisy)
    X_clean, y_clean = make_classification(n_samples=300, n_features=10,
                                            n_informative=8, flip_y=0.01,
                                            random_state=42)
    mf_clean = recommender.compute_meta_features(X_clean, y_clean)
    assert mf_noisy['frac_zero_mi'] > mf_clean['frac_zero_mi'], (
        f"Test 3.3 FAILED: Noisy dataset ({mf_noisy['frac_zero_mi']:.3f}) should have "
        f"higher frac_zero_mi than clean dataset ({mf_clean['frac_zero_mi']:.3f})."
    )
    print(f"Test 3.3 PASSED: Noisy dataset has higher frac_zero_mi "
          f"({mf_noisy['frac_zero_mi']:.3f} > {mf_clean['frac_zero_mi']:.3f}).")

    # --- Test 3.4: fit() trains successfully ---
    bc = load_breast_cancer()
    wn = load_wine()
    ir = load_iris()

    training_datasets = [
        {'X': bc.data, 'y': bc.target, 'best_selector': 'rf_importance'},
        {'X': wn.data, 'y': wn.target, 'best_selector': 'mi'},
        {'X': ir.data, 'y': ir.target, 'best_selector': 'lasso'},
        {'X': X_noisy, 'y': y_noisy, 'best_selector': 'rf_importance'},
        {'X': X_clean, 'y': y_clean, 'best_selector': 'mi'},
    ]

    try:
        recommender.fit(training_datasets)
    except NotImplementedError:
        print("Test 3.4 FAILED: SimpleMetaRecommender.fit() not implemented yet.")
        return False
    except Exception as e:
        print(f"Test 3.4 FAILED: fit() raised: {e}")
        return False

    assert recommender._is_fitted, (
        "Test 3.4 FAILED: _is_fitted should be True after fit()."
    )
    assert recommender.meta_model is not None, (
        "Test 3.4 FAILED: meta_model is None after fit()."
    )
    assert recommender.label_encoder is not None, (
        "Test 3.4 FAILED: label_encoder is None after fit()."
    )
    print("Test 3.4 PASSED: fit() trains successfully on 5 training datasets.")

    # --- Test 3.5: recommend() returns valid output ---
    X_new, y_new = make_synthetic_dataset(n=300, p=30, n_informative=8)
    try:
        recs = recommender.recommend(X_new, y_new, top_n=2)
    except NotImplementedError:
        print("Test 3.5 FAILED: recommend() not implemented yet.")
        return False
    except Exception as e:
        print(f"Test 3.5 FAILED: recommend() raised: {e}")
        return False

    assert isinstance(recs, list), (
        "Test 3.5 FAILED: recommend() must return a list."
    )
    assert len(recs) == 2, (
        f"Test 3.5 FAILED: top_n=2 requested but got {len(recs)} recommendations."
    )
    for sel_name, prob in recs:
        assert isinstance(sel_name, str), (
            f"Test 3.5 FAILED: Selector name must be a string, got {type(sel_name)}."
        )
        assert 0 <= prob <= 1, (
            f"Test 3.5 FAILED: Probability {prob:.4f} is not in [0, 1]."
        )
    # Probabilities should be sorted descending
    probs = [r[1] for r in recs]
    assert probs == sorted(probs, reverse=True), (
        f"Test 3.5 FAILED: Recommendations must be sorted by probability descending. "
        f"Got probabilities: {probs}."
    )
    print(f"Test 3.5 PASSED: recommend() returns {len(recs)} recommendations "
          f"sorted by probability: {[(n, f'{p:.3f}') for n, p in recs]}.")

    # --- Test 3.6: Recommender prefers RF for noisy high-p dataset ---
    # High frac_zero_mi should push toward RF/Boruta (which handle noise well)
    X_highnoise, y_highnoise = make_classification(
        n_samples=500, n_features=100, n_informative=5, flip_y=0.08,
        random_state=13
    )
    recs_noise = recommender.recommend(X_highnoise, y_highnoise, top_n=3)
    top_name = recs_noise[0][0]
    print(f"Test 3.6 INFO: For high-noise dataset, top recommendation = '{top_name}'.")
    # We don't assert a specific answer (too few training datasets),
    # but the recommendation should be one of the known selectors
    known_selectors = {'mi', 'lasso', 'rf_importance', 'rf_boruta_approx', 'elastic_net'}
    assert top_name in known_selectors, (
        f"Test 3.6 FAILED: Recommended selector '{top_name}' is not in known selectors "
        f"{known_selectors}."
    )
    print(f"Test 3.6 PASSED: Recommendation '{top_name}' is a valid selector name.")

    print("\nAll Exercise 3 tests PASSED.")
    return True


# ===========================================================================
# Run all exercises
# ===========================================================================

if __name__ == '__main__':
    print("Module 10 — Ensemble & Hybrid Feature Selection: Self-Check Exercises")
    print("=" * 70)
    print()
    print("Run each test block after implementing the corresponding function.")
    print("Tests print PASSED or FAILED with helpful error messages.\n")

    all_passed = True

    try:
        ex1_ok = test_exercise_1()
        all_passed = all_passed and ex1_ok
    except NotImplementedError:
        print("Exercise 1 not yet implemented — skipping tests.\n")
        all_passed = False
    except AssertionError as e:
        print(f"ASSERTION FAILED: {e}\n")
        all_passed = False

    try:
        ex2_ok = test_exercise_2()
        all_passed = all_passed and ex2_ok
    except NotImplementedError:
        print("Exercise 2 not yet implemented — skipping tests.\n")
        all_passed = False
    except AssertionError as e:
        print(f"ASSERTION FAILED: {e}\n")
        all_passed = False

    try:
        ex3_ok = test_exercise_3()
        all_passed = all_passed and ex3_ok
    except NotImplementedError:
        print("Exercise 3 not yet implemented — skipping tests.\n")
        all_passed = False
    except AssertionError as e:
        print(f"ASSERTION FAILED: {e}\n")
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL EXERCISES PASSED — Module 10 exercises complete.")
    else:
        print("Some exercises have not passed yet. Review the error messages above.")
    print("=" * 70)
