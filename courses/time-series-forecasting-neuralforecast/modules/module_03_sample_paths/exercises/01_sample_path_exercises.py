"""
Module 03 — Sample Paths: Self-Check Exercises
===============================================

These exercises reinforce the key mechanics of sample path generation and
the Monte Carlo framework. Each exercise has a setup, a task, and an
assertion that confirms your answer is correct.

Run with:
    python 01_sample_path_exercises.py

All assertions pass when your implementations are correct.
"""

import numpy as np
from scipy.linalg import toeplitz
from scipy.stats import norm, pearsonr

# ---------------------------------------------------------------------------
# Shared setup: synthetic AR(1) data and a simple copula implementation
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

# Synthetic training data: AR(1) with phi=0.6, daily demand around 100
N_TRAIN = 200
PHI_TRUE = 0.6
y_train = np.zeros(N_TRAIN)
y_train[0] = RNG.normal(100, 15)
for t in range(1, N_TRAIN):
    y_train[t] = PHI_TRUE * y_train[t - 1] + (1 - PHI_TRUE) * 100 + RNG.normal(0, 12)

# Horizon
H = 7

# Quantile grid — matches MQLoss(level=[80, 90]) output
QUANTILE_LEVELS = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                             0.60, 0.70, 0.80, 0.90, 0.95])

# Synthetic "model" quantile forecasts: Gaussian centered at 100, std=15
# Shape: (H, len(QUANTILE_LEVELS)) — one row per horizon step
QUANTILE_FORECASTS = np.array([
    norm.ppf(QUANTILE_LEVELS, loc=100 + t * 0.5, scale=15)
    for t in range(H)
])  # shape: (7, 11)


# ---------------------------------------------------------------------------
# Utility: Gaussian Copula path generator
# (reference implementation — do not modify)
# ---------------------------------------------------------------------------

def _gaussian_copula_paths_reference(
    quantile_levels: np.ndarray,
    quantile_forecasts: np.ndarray,
    y_series: np.ndarray,
    n_paths: int = 100,
    seed: int = 0,
) -> np.ndarray:
    """Reference implementation of Gaussian Copula path generation."""
    H_ = quantile_forecasts.shape[0]
    diff = np.diff(y_series)
    phi = float(np.clip(pearsonr(diff[:-1], diff[1:])[0], -0.999, 0.999))
    Sigma = toeplitz([phi ** k for k in range(H_)])
    L = np.linalg.cholesky(Sigma)
    rng_ = np.random.default_rng(seed)
    epsilon = rng_.standard_normal((H_, n_paths))
    z = (L @ epsilon).T
    u = norm.cdf(z)
    paths_ = np.zeros((n_paths, H_))
    for t in range(H_):
        paths_[:, t] = np.interp(u[:, t], quantile_levels, quantile_forecasts[t])
    return paths_


# Generate reference paths used by all exercises
REFERENCE_PATHS = _gaussian_copula_paths_reference(
    QUANTILE_LEVELS, QUANTILE_FORECASTS, y_train, n_paths=500, seed=42
)

# ---------------------------------------------------------------------------
# Exercise 1 — Path Shape
# ---------------------------------------------------------------------------

def exercise_1_path_shape():
    """
    Task
    ----
    Generate 200 sample paths using the reference implementation with seed=7.
    Verify that the resulting array has the correct shape: (n_paths, horizon).

    The shape convention is: rows = paths, columns = horizon steps.
    """
    n_paths = 200
    paths = _gaussian_copula_paths_reference(
        QUANTILE_LEVELS, QUANTILE_FORECASTS, y_train, n_paths=n_paths, seed=7
    )

    # YOUR CHECK: fill in the expected shape
    expected_shape = (n_paths, H)

    assert paths.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {paths.shape}. "
        "Remember: rows = paths, columns = horizon steps."
    )
    assert paths.ndim == 2, "paths must be a 2D array"

    # All path values should be in a plausible range
    assert paths.min() > 0, "Demand paths should be positive"
    assert paths.max() < 500, (
        f"Demand paths contain unrealistically large values: {paths.max():.1f}. "
        "Check your quantile forecast grid."
    )

    print("Exercise 1 PASSED: path shape is correct")
    return paths


# ---------------------------------------------------------------------------
# Exercise 2 — Weekly Total from Paths
# ---------------------------------------------------------------------------

def exercise_2_weekly_total(paths: np.ndarray) -> float:
    """
    Task
    ----
    Compute the 80th percentile of the weekly total demand from the given
    paths array (shape: n_paths x H).

    Steps:
    1. Sum each path across the horizon axis → array of shape (n_paths,)
    2. Take the 80th percentile of those sums

    Parameters
    ----------
    paths : np.ndarray of shape (n_paths, H)

    Returns
    -------
    float — the 80th percentile of weekly totals
    """
    # Implement this function
    weekly_totals = paths.sum(axis=1)
    result = float(np.quantile(weekly_totals, 0.80))
    return result


def test_exercise_2():
    paths = REFERENCE_PATHS
    result = exercise_2_weekly_total(paths)

    # The result should be a plausible weekly total near 7 * 100 = 700
    assert isinstance(result, float), "Return value must be a float"
    assert 550 < result < 850, (
        f"80th pct weekly total {result:.1f} is outside plausible range [550, 850]. "
        "Check that you are summing across axis=1 (horizon steps) "
        "and taking quantile 0.80."
    )

    # Compare with sum of marginal 80th percentiles
    marginal_80s = np.quantile(paths, 0.80, axis=0)   # shape: (H,)
    naive_result = marginal_80s.sum()

    # The results must differ — this is the whole point
    assert abs(result - naive_result) > 1.0, (
        f"Sample path result ({result:.1f}) and marginal sum ({naive_result:.1f}) "
        "are too close. With an autocorrelated series they must differ. "
        "Ensure you are computing the 80th pct of SUMS, not the SUM of 80th pcts."
    )

    print(f"Exercise 2 PASSED: 80th pct weekly total = {result:.1f}")
    print(f"  (naive marginal sum = {naive_result:.1f}, diff = {naive_result - result:+.1f})")
    return result


# ---------------------------------------------------------------------------
# Exercise 3 — Marginal vs. Path Totals Differ
# ---------------------------------------------------------------------------

def exercise_3_assert_totals_differ():
    """
    Task
    ----
    Demonstrate that the 80th percentile of the weekly total (from paths)
    differs from the sum of marginal 80th percentiles.

    Compute both quantities and confirm they are not equal.
    The difference should be at least 1 unit for an autocorrelated series.
    """
    paths = REFERENCE_PATHS

    # From paths: 80th pct of weekly totals
    path_total_80 = float(np.quantile(paths.sum(axis=1), 0.80))

    # From marginals: sum of per-step 80th percentiles
    marginal_80_sum = float(np.quantile(paths, 0.80, axis=0).sum())

    assert path_total_80 != marginal_80_sum, (
        "The two methods should give different results for an autocorrelated series. "
        "If they are identical, check that your paths have non-zero correlation "
        "between adjacent steps."
    )

    assert abs(path_total_80 - marginal_80_sum) > 1.0, (
        f"The difference is only {abs(path_total_80 - marginal_80_sum):.2f}. "
        "For AR(1) data with phi > 0.3, the difference should be > 1 unit."
    )

    print("Exercise 3 PASSED: marginal and path totals are demonstrably different")
    print(f"  Path-based 80th pct total:     {path_total_80:.1f}")
    print(f"  Sum of marginal 80th pcts:      {marginal_80_sum:.1f}")
    print(f"  Difference:                     {marginal_80_sum - path_total_80:+.1f}")


# ---------------------------------------------------------------------------
# Exercise 4 — Reorder Timing
# ---------------------------------------------------------------------------

def stockout_day(path: np.ndarray, stock: float) -> int:
    """
    Find the first day (1-indexed) where cumulative demand exceeds `stock`.
    Returns H+1 if no stock-out occurs within the horizon.

    Parameters
    ----------
    path  : np.ndarray shape (H,) — one demand trajectory
    stock : float — starting inventory

    Returns
    -------
    int — 1-indexed day of stock-out, or H+1 if none
    """
    cumulative = np.cumsum(path)
    crossing = np.where(cumulative > stock)[0]
    if len(crossing) == 0:
        return H + 1
    return int(crossing[0]) + 1   # 1-indexed


def exercise_4_reorder_timing(paths: np.ndarray, stock: float) -> int:
    """
    Task
    ----
    Compute the safe reorder day at an 80% service level.

    Definition: the latest day by which a reorder should be placed such that
    80% of demand trajectories do not run out of stock before that day.

    Implementation:
    1. For each path, find the stock-out day using `stockout_day`
    2. Take the 20th percentile of stock-out days
       (80% of paths stock out after this day, so reordering by this day
       provides an 80% service level)
    3. Return as an integer

    Parameters
    ----------
    paths : np.ndarray (n_paths, H)
    stock : float — starting inventory level

    Returns
    -------
    int — day by which to reorder (1-indexed, inclusive)
    """
    # Implement this function
    stockout_days = np.array([stockout_day(paths[s], stock) for s in range(len(paths))])
    reorder = int(np.quantile(stockout_days, 0.20))
    return reorder


def test_exercise_4():
    paths = REFERENCE_PATHS

    # Stock level: 60% of the median weekly total
    median_weekly = float(np.median(paths.sum(axis=1)))
    stock = median_weekly * 0.60

    result = exercise_4_reorder_timing(paths, stock)

    assert isinstance(result, int), "Reorder day must be an integer"
    assert 1 <= result <= H + 1, (
        f"Reorder day {result} is outside valid range [1, {H+1}]. "
        "With stock at 60% of median weekly demand, a mid-week stock-out "
        "is expected for many paths."
    )

    # Verify the service level interpretation.
    # The 20th percentile of stock-out days means: 80% of paths stock out
    # on day `result` OR LATER — i.e., stockout_days >= result.
    # Reordering by day `result` ensures stock arrives before 80% of paths exhaust supply.
    stockout_days = np.array([stockout_day(paths[s], stock) for s in range(len(paths))])
    frac_safe = (stockout_days >= result).mean()
    assert frac_safe >= 0.75, (
        f"Only {frac_safe:.1%} of paths stock out on day {result} or later. "
        "Expected at least 75% (target 80%). "
        "The 20th pct of stockout days means 80% of paths stock out on that day or later."
    )

    print(f"Exercise 4 PASSED: safe reorder day = {result}")
    print(f"  Stock level: {stock:.0f} units")
    print(f"  Fraction of paths that reach day {result} before stock-out: {frac_safe:.1%}")


# ---------------------------------------------------------------------------
# Exercise 5 — Temporal Correlation Check
# ---------------------------------------------------------------------------

def exercise_5_correlation_preserved():
    """
    Task
    ----
    Verify that the Gaussian Copula correctly preserves temporal autocorrelation.

    Generate 2000 paths. Compute the empirical Pearson correlation between
    adjacent horizon steps (step 0 and step 1) across all paths.

    The estimated AR(1) coefficient from y_train should be within 0.15 of
    the empirical path correlation (with 2000 paths, the estimate is stable).
    """
    paths = _gaussian_copula_paths_reference(
        QUANTILE_LEVELS, QUANTILE_FORECASTS, y_train, n_paths=2000, seed=99
    )

    # Estimated phi from training data
    diff = np.diff(y_train)
    phi_estimated = float(np.clip(pearsonr(diff[:-1], diff[1:])[0], -0.999, 0.999))

    # Empirical correlation between step 0 and step 1 across paths
    empirical_corr = float(np.corrcoef(paths[:, 0], paths[:, 1])[0, 1])

    tolerance = 0.15
    assert abs(empirical_corr - phi_estimated) < tolerance, (
        f"Empirical correlation between steps 0 and 1 is {empirical_corr:.4f}, "
        f"but estimated phi is {phi_estimated:.4f}. "
        f"Difference {abs(empirical_corr - phi_estimated):.4f} exceeds tolerance {tolerance}. "
        "The Gaussian Copula should preserve the AR(1) correlation structure."
    )

    print("Exercise 5 PASSED: temporal correlation is preserved through the copula")
    print(f"  Estimated phi from training data: {phi_estimated:.4f}")
    print(f"  Empirical step 0-1 correlation:   {empirical_corr:.4f}")
    print(f"  Difference:                        {abs(empirical_corr - phi_estimated):.4f}")


# ---------------------------------------------------------------------------
# Run all exercises
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Module 03 — Sample Path Self-Check Exercises")
    print("=" * 60)
    print()

    paths_ex1 = exercise_1_path_shape()
    print()

    weekly_80 = test_exercise_2()
    print()

    exercise_3_assert_totals_differ()
    print()

    test_exercise_4()
    print()

    exercise_5_correlation_preserved()
    print()

    print("=" * 60)
    print("All exercises passed.")
    print("=" * 60)
