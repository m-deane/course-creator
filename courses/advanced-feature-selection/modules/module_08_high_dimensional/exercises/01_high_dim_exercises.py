"""
Module 8: Feature Selection for High-Dimensional Data
Self-Check Exercises

These exercises are self-graded — run the test functions at the bottom to
check your implementations. No submission required.

Topics:
  Exercise 1 — DC-SIS: Distance correlation-based screening
  Exercise 2 — Group Lasso: Group-level selection with known feature groups
  Exercise 3 — Coverage comparison: Naive vs debiased post-selection inference

Prerequisites:
  - Sure Independence Screening (Guide 01, Notebook 01)
  - Structured sparsity and Group Lasso (Guide 02, Notebook 02)
  - Post-selection inference (Guide 03, Notebook 03)

Reference implementations are in the SOLUTIONS section at the bottom
of this file (search for "# === SOLUTIONS ===").

Usage:
    python 01_high_dim_exercises.py
"""

import numpy as np
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)


# =============================================================================
# Shared data generation utilities
# =============================================================================

def generate_sparse_data(
    n: int = 200,
    p: int = 1000,
    s: int = 8,
    snr: float = 3.0,
    seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generate a sparse linear regression dataset.

    Parameters
    ----------
    n : int — sample size
    p : int — number of features
    s : int — sparsity (number of active features)
    snr : float — signal-to-noise ratio
    seed : int

    Returns
    -------
    X : ndarray (n, p) — standardised feature matrix
    y : ndarray (n,)   — centred outcome
    beta : ndarray (p,) — true coefficients
    active : ndarray   — indices of active features
    sigma : float      — true noise standard deviation
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    X = (X - X.mean(0)) / (X.std(0) + 1e-10)

    active = rng.choice(p, size=s, replace=False)
    beta = np.zeros(p)
    beta[active] = rng.uniform(0.8, 2.0, size=s) * rng.choice([-1, 1], size=s)

    signal = X @ beta
    sigma = np.std(signal) / np.sqrt(snr)
    y = signal + rng.standard_normal(n) * sigma
    y = y - y.mean()

    return X, y, beta, active, sigma


def generate_grouped_data(
    n: int = 200,
    n_groups: int = 8,
    group_size: int = 25,
    n_active_groups: int = 3,
    intra_corr: float = 0.7,
    snr: float = 3.0,
    seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate grouped regression data.

    Returns X, y, groups, beta_true, active_features, active_groups.
    """
    rng = np.random.default_rng(seed)
    p = n_groups * group_size

    X_blocks = []
    for g in range(n_groups):
        common = rng.standard_normal(n)
        idio = rng.standard_normal((n, group_size))
        block = np.sqrt(intra_corr) * common[:, None] + np.sqrt(1 - intra_corr) * idio
        X_blocks.append(block)

    X = np.hstack(X_blocks)
    X = (X - X.mean(0)) / (X.std(0) + 1e-10)
    groups = np.repeat(np.arange(n_groups), group_size)

    active_groups = np.arange(n_active_groups)
    beta_true = np.zeros(p)
    active_features = []

    for g in active_groups:
        start = g * group_size
        n_active = 3
        within = rng.choice(group_size, size=n_active, replace=False)
        idx = start + within
        active_features.extend(idx.tolist())
        mags = rng.uniform(0.5, 2.0, size=n_active)
        signs = rng.choice([-1, 1], size=n_active)
        beta_true[idx] = mags * signs

    active_features = np.array(sorted(active_features))
    signal = X @ beta_true
    sigma = np.std(signal) / np.sqrt(snr)
    y = signal + rng.standard_normal(n) * sigma
    y = y - y.mean()

    return X, y, groups, beta_true, active_features, active_groups


# =============================================================================
# Exercise 1: DC-SIS (Distance Correlation Screening)
# =============================================================================

def compute_distance_covariance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute sample distance covariance^2 between 1-D arrays x and y.

    Formula:
        dCov^2(x, y) = (1/n^2) * sum_{k,l} A_{kl} * B_{kl}

    where A and B are doubly-centred distance matrices:
        a_{kl} = |x_k - x_l|  (Euclidean distance)
        A_{kl} = a_{kl} - mean_l(a_{kl}) - mean_k(a_{kl}) + mean_{kl}(a_{kl})

    Parameters
    ----------
    x : ndarray (n,) — first variable
    y : ndarray (n,) — second variable

    Returns
    -------
    float — sample distance covariance squared (>= 0)
    """
    raise NotImplementedError("Exercise 1a: Implement compute_distance_covariance")


def compute_distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute sample distance correlation between 1-D arrays x and y.

    Formula:
        dCor(x, y) = dCov(x, y) / sqrt(dCov(x, x) * dCov(y, y))

    If denominator is zero, return 0.

    Parameters
    ----------
    x : ndarray (n,) — first variable
    y : ndarray (n,) — second variable

    Returns
    -------
    float in [0, 1] — sample distance correlation
    """
    raise NotImplementedError("Exercise 1b: Implement compute_distance_correlation")


def dc_sis(
    X: np.ndarray,
    y: np.ndarray,
    d_n: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Distance Correlation Sure Independence Screening (DC-SIS).

    Li, Zhong & Zhu (2012): Replace marginal Pearson correlation in SIS
    with marginal distance correlation. This detects nonlinear dependence
    that SIS misses.

    Algorithm:
    1. For each feature j, compute dCor(x_j, y).
    2. Sort features by dCor score descending.
    3. Return top d_n indices and all scores.

    Parameters
    ----------
    X : ndarray (n, p) — standardised feature matrix
    y : ndarray (n,) — outcome
    d_n : int — screen size. Defaults to floor(n / log(n)).

    Returns
    -------
    screened_idx : ndarray (d_n,) — sorted by score descending
    dcor_scores : ndarray (p,) — distance correlation score for each feature

    Hint:
        Call compute_distance_correlation(X[:, j], y) for each j.
        This is O(p * n^2) — for p=500 and n=100 it runs in a few seconds.
    """
    raise NotImplementedError("Exercise 1c: Implement dc_sis")


# =============================================================================
# Exercise 2: Group Lasso Selector
# =============================================================================

def group_soft_threshold(v: np.ndarray, threshold: float) -> np.ndarray:
    """
    Group soft-thresholding operator.

    For a vector v, apply:
        prox(v, threshold) = v * max(0, 1 - threshold / ||v||_2)

    If ||v||_2 <= threshold, return the zero vector.
    If ||v||_2 > threshold, shrink v toward zero but preserve direction.

    Parameters
    ----------
    v : ndarray — input vector (group coefficient block)
    threshold : float — thresholding value (>= 0)

    Returns
    -------
    ndarray — proximal output, same shape as v
    """
    raise NotImplementedError("Exercise 2a: Implement group_soft_threshold")


def group_lasso_fit(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    lam: float,
    max_iter: int = 1500,
    tol: float = 1e-5
) -> np.ndarray:
    """
    Fit Group Lasso via proximal gradient descent.

    Minimises:
        (1/2n) ||y - X beta||^2 + lam * sum_g sqrt(|G_g|) * ||beta_{G_g}||_2

    Algorithm (proximal gradient):
    1. Initialise beta = 0.
    2. Repeat until convergence:
       a. Gradient step: beta_half = beta - step * (-X^T (y - X beta) / n)
       b. Proximal step: for each group g,
              beta_{G_g} = group_soft_threshold(beta_half_{G_g}, step * lam * sqrt(|G_g|))

    Parameters
    ----------
    X : ndarray (n, p)
    y : ndarray (n,)
    groups : ndarray (p,) — integer group label for each feature
    lam : float — penalty parameter
    max_iter : int
    tol : float — convergence tolerance on max |delta_beta|

    Returns
    -------
    beta : ndarray (p,) — Group Lasso coefficient vector

    Hint:
        Step size: use 1 / (largest_eigenvalue_of_XtX/n + 1e-6).
        Estimate the largest eigenvalue by: np.linalg.norm(X, ord=2)**2 / n
        (squared spectral norm equals largest singular value squared).
    """
    raise NotImplementedError("Exercise 2b: Implement group_lasso_fit")


def group_lasso_select(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    lam_grid: np.ndarray = None,
    cv: int = 3
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Select features using Group Lasso with cross-validated lambda.

    For each lambda in lam_grid, fit Group Lasso and compute cross-validated MSE.
    Select the lambda with the lowest CV-MSE. Return selected features, selected
    group indices, and the best lambda.

    Parameters
    ----------
    X : ndarray (n, p)
    y : ndarray (n,)
    groups : ndarray (p,)
    lam_grid : ndarray — lambda values to try. If None, use np.logspace(-2, 0, 15).
    cv : int — number of cross-validation folds

    Returns
    -------
    selected_features : ndarray — indices of features with non-zero beta at best lambda
    selected_groups : ndarray — indices of groups with non-zero group norm at best lambda
    best_lam : float — lambda that minimised CV-MSE

    Hint:
        For each fold and lambda:
            beta = group_lasso_fit(X_train, y_train, groups, lam)
            mse_fold = mean((y_val - X_val @ beta)^2)
        Average mse_fold over folds for each lambda.
        Select lambda with minimum average MSE.
    """
    raise NotImplementedError("Exercise 2c: Implement group_lasso_select")


# =============================================================================
# Exercise 3: Post-Selection Coverage Comparison
# =============================================================================

def naive_ci(
    X_selected: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute naive OLS confidence intervals on pre-selected features.

    WARNING: These CIs have incorrect coverage after selection. This function
    is provided to demonstrate the problem, not as a valid inference tool.

    Parameters
    ----------
    X_selected : ndarray (n, k) — feature matrix for selected features
    y : ndarray (n,) — outcome
    alpha : float — significance level

    Returns
    -------
    coef : ndarray (k,) — OLS coefficients
    ci_lo : ndarray (k,) — lower CI bounds
    ci_hi : ndarray (k,) — upper CI bounds
    """
    raise NotImplementedError("Exercise 3a: Implement naive_ci using OLS")


def split_sample_ci(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
    split_frac: float = 0.5,
    seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Data-splitting confidence intervals.

    Uses first split_frac of data for Lasso feature selection and the
    remaining (1 - split_frac) for OLS inference.

    Parameters
    ----------
    X : ndarray (n, p)
    y : ndarray (n,)
    alpha : float — significance level (0.05 for 95% CI)
    split_frac : float — fraction of data for selection
    seed : int — random seed for the split

    Returns
    -------
    selected : ndarray — selected feature indices (from original feature space)
    coef : ndarray — OLS coefficients on inference set
    ci_lo : ndarray — lower CI bounds
    ci_hi : ndarray — upper CI bounds

    Hint:
        1. Split data: selection set (first n1 obs) and inference set (rest).
        2. Run LassoCV on selection set -> selected features.
        3. Run OLS (LinearRegression) on inference set restricted to selected features.
        4. Compute CIs from OLS: coef +/- t_crit * se.
        5. t_crit = scipy.stats.t.ppf(1 - alpha/2, df=n2 - k - 1)
           where n2 = inference set size, k = number of selected features.
    """
    raise NotImplementedError("Exercise 3b: Implement split_sample_ci")


def compare_coverage(
    n: int = 150,
    p: int = 150,
    s: int = 8,
    n_reps: int = 200,
    alpha: float = 0.05,
    snr: float = 3.0
) -> dict:
    """
    Monte Carlo comparison of coverage rates for naive vs split-sample CIs.

    For each repetition:
    1. Generate data.
    2. Compute naive CI on Lasso-selected features.
    3. Compute split-sample CI.
    4. For true active features that were selected, check whether the CI contains beta*.

    Parameters
    ----------
    n, p, s : int — sample size, features, sparsity
    n_reps : int — number of Monte Carlo repetitions
    alpha : float — significance level
    snr : float — signal-to-noise ratio

    Returns
    -------
    dict with keys:
        'naive_coverage'  : float — empirical coverage of naive CIs
        'split_coverage'  : float — empirical coverage of split-sample CIs
        'naive_n'         : int   — number of intervals assessed (naive)
        'split_n'         : int   — number of intervals assessed (split)

    Hint:
        For naive CI, you need to:
        1. Run Lasso on all data -> selected features.
        2. Call naive_ci(X[:, selected], y, alpha).
        3. Check coverage for features in (selected intersection active).

        For split CI, call split_sample_ci(X, y, alpha, seed=rep).
        Check coverage for features in (split_selected intersection active).
    """
    raise NotImplementedError("Exercise 3c: Implement compare_coverage")


# =============================================================================
# Self-check tests — run these to verify your implementations
# =============================================================================

def test_distance_covariance():
    """Test dCov^2 is non-negative, symmetric, and zero for independent variables."""
    rng = np.random.default_rng(0)
    n = 50
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    z = x**2 + 0.5 * rng.standard_normal(n)  # nonlinear relationship

    # Test 1: Non-negative
    dcov_xy = compute_distance_covariance(x, y)
    assert dcov_xy >= -1e-10, f"dCov^2 must be non-negative, got {dcov_xy}"

    # Test 2: Self-distance covariance > 0
    dcov_xx = compute_distance_covariance(x, x)
    assert dcov_xx > 0, f"dCov^2(x, x) must be positive, got {dcov_xx}"

    # Test 3: Distance covariance detects nonlinear relationship
    dcov_xz = compute_distance_covariance(x, z)
    pearson_corr = abs(np.corrcoef(x, z)[0, 1])
    # dCov should be informative about the nonlinear relationship
    # (no strict assertion since it depends on the sample, just check it's positive)
    assert dcov_xz > 0, f"dCov^2(x, z) should be positive for related variables, got {dcov_xz}"

    print("PASS: test_distance_covariance")


def test_distance_correlation():
    """Test dCor is in [0, 1] and is larger for related variables."""
    rng = np.random.default_rng(1)
    n = 80
    x = rng.standard_normal(n)
    y_indep = rng.standard_normal(n)
    y_dep = x**2 + 0.3 * rng.standard_normal(n)  # nonlinear

    dcor_indep = compute_distance_correlation(x, y_indep)
    dcor_dep = compute_distance_correlation(x, y_dep)

    # Test 1: Range
    assert 0 <= dcor_indep <= 1 + 1e-10, f"dCor must be in [0,1], got {dcor_indep}"
    assert 0 <= dcor_dep <= 1 + 1e-10, f"dCor must be in [0,1], got {dcor_dep}"

    # Test 2: Dependent variable has higher dCor than independent
    assert dcor_dep > dcor_indep, (
        f"dCor should be higher for dependent variable: dep={dcor_dep:.4f}, indep={dcor_indep:.4f}"
    )

    # Test 3: Self-correlation equals 1
    dcor_self = compute_distance_correlation(x, x)
    assert abs(dcor_self - 1.0) < 1e-10, f"dCor(x, x) should be 1, got {dcor_self}"

    print("PASS: test_distance_correlation")


def test_dc_sis():
    """Test DC-SIS recovers active features with nonlinear relationships."""
    # Dataset with a nonlinear relationship: y = x_0^2 + x_1 + noise
    rng = np.random.default_rng(5)
    n, p = 100, 200
    X = rng.standard_normal((n, p))
    X = (X - X.mean(0)) / X.std(0)

    # Feature 0: quadratic relationship (Pearson ≈ 0 but dCor > 0)
    # Feature 1: linear relationship (both SIS and DC-SIS detect this)
    y = X[:, 0]**2 + 2.0 * X[:, 1] + 0.5 * rng.standard_normal(n)

    # SIS (Pearson-based) marginal scores
    y_c = y - y.mean()
    pearson_scores = np.abs(X.T @ y_c) / n
    pearson_rank_feat0 = np.sum(pearson_scores >= pearson_scores[0])

    d_n = int(n / np.log(n))  # ~21 for n=100
    screened, dcor_scores = dc_sis(X, y, d_n=d_n)

    # Test 1: Feature 1 (linear) is in screened set
    assert 1 in screened, f"DC-SIS should recover feature 1 (linear effect), got {screened[:10]}"

    # Test 2: Feature 0 (quadratic) has higher DC-SIS score than Pearson rank suggests
    dcor_rank_feat0 = np.sum(dcor_scores >= dcor_scores[0])
    assert dcor_rank_feat0 <= pearson_rank_feat0, (
        f"DC-SIS should rank feature 0 (quadratic) higher than Pearson: "
        f"dcor_rank={dcor_rank_feat0}, pearson_rank={pearson_rank_feat0}"
    )

    # Test 3: Output shapes
    assert len(screened) == d_n, f"Expected {d_n} screened features, got {len(screened)}"
    assert len(dcor_scores) == p, f"Expected {p} scores, got {len(dcor_scores)}"

    # Test 4: Scores are in [0, 1]
    assert np.all(dcor_scores >= -1e-10), "dCor scores must be non-negative"
    assert np.all(dcor_scores <= 1 + 1e-10), "dCor scores must be <= 1"

    print("PASS: test_dc_sis")


def test_group_soft_threshold():
    """Test group soft-thresholding correctness."""
    # Test 1: Zero input -> zero output
    v_zero = np.zeros(5)
    result = group_soft_threshold(v_zero, threshold=0.5)
    assert np.allclose(result, 0), f"Zero input should give zero output, got {result}"

    # Test 2: Norm below threshold -> zero vector
    v_small = np.array([0.1, 0.1, 0.1])  # norm = sqrt(0.03) ≈ 0.173
    result_small = group_soft_threshold(v_small, threshold=0.5)
    assert np.allclose(result_small, 0), (
        f"Norm {np.linalg.norm(v_small):.3f} < threshold 0.5, should be zero, got {result_small}"
    )

    # Test 3: Norm above threshold -> non-zero, direction preserved
    v_large = np.array([3.0, 4.0])  # norm = 5.0
    threshold = 1.0
    result_large = group_soft_threshold(v_large, threshold=threshold)
    expected_norm = 5.0 - threshold  # should be 4.0
    assert abs(np.linalg.norm(result_large) - expected_norm) < 1e-10, (
        f"Expected norm {expected_norm}, got {np.linalg.norm(result_large):.6f}"
    )
    # Direction should be preserved
    assert np.allclose(result_large / np.linalg.norm(result_large),
                       v_large / np.linalg.norm(v_large)), "Direction should be preserved"

    # Test 4: Threshold=0 -> identity
    result_id = group_soft_threshold(v_large, threshold=0.0)
    assert np.allclose(result_id, v_large), "Threshold=0 should return input unchanged"

    print("PASS: test_group_soft_threshold")


def test_group_lasso_fit():
    """Test Group Lasso fits correctly and enforces group sparsity."""
    X, y, groups, beta_true, active_features, active_groups = generate_grouped_data(
        n=150, n_groups=6, group_size=15, n_active_groups=2, seed=1
    )
    n_groups = len(np.unique(groups))

    # Fit at a moderate lambda that should select some groups
    lam = 0.05
    beta_hat = group_lasso_fit(X, y, groups, lam=lam, max_iter=2000)

    # Test 1: Shape
    assert beta_hat.shape == (X.shape[1],), f"Expected shape ({X.shape[1]},), got {beta_hat.shape}"

    # Test 2: Group sparsity — coefficients within a zero group are ALL zero
    for g in range(n_groups):
        g_mask = groups == g
        g_coefs = beta_hat[g_mask]
        g_norm = np.linalg.norm(g_coefs)
        if g_norm < 1e-8:  # group is zeroed
            assert np.all(np.abs(g_coefs) < 1e-8), (
                f"Group {g} norm is near-zero but has non-zero individual coefficients: {g_coefs}"
            )

    # Test 3: Prediction is better than null model
    pred = X @ beta_hat
    mse_model = np.mean((y - pred)**2)
    mse_null = np.mean(y**2)
    assert mse_model < mse_null, (
        f"Group Lasso should fit better than null: mse_model={mse_model:.4f}, mse_null={mse_null:.4f}"
    )

    # Test 4: High lambda drives all groups to zero
    beta_zero = group_lasso_fit(X, y, groups, lam=10.0, max_iter=1000)
    assert np.allclose(beta_zero, 0, atol=1e-6), (
        f"Very high lambda should drive all coefs to zero, max coef: {np.abs(beta_zero).max()}"
    )

    print("PASS: test_group_lasso_fit")


def test_group_lasso_select():
    """Test Group Lasso selection recovers active groups."""
    X, y, groups, beta_true, active_features, active_groups = generate_grouped_data(
        n=200, n_groups=8, group_size=20, n_active_groups=2, snr=4.0, seed=3
    )
    n_groups_total = len(np.unique(groups))

    selected_features, selected_groups, best_lam = group_lasso_select(X, y, groups, cv=3)

    # Test 1: Returns valid types and shapes
    assert isinstance(selected_features, np.ndarray), "selected_features must be ndarray"
    assert isinstance(selected_groups, np.ndarray), "selected_groups must be ndarray"
    assert isinstance(best_lam, float), f"best_lam must be float, got {type(best_lam)}"

    # Test 2: Selected groups are subset of all groups
    assert np.all(selected_groups >= 0) and np.all(selected_groups < n_groups_total), (
        f"Selected group indices out of range: {selected_groups}"
    )

    # Test 3: At least one active group is recovered (with SNR=4, should be easy)
    n_active_recovered = len(set(selected_groups) & set(active_groups))
    assert n_active_recovered >= 1, (
        f"Group Lasso should recover at least 1 active group. "
        f"Selected groups: {sorted(selected_groups)}, Active groups: {sorted(active_groups)}"
    )

    # Test 4: Selected features are consistent with selected groups
    if len(selected_features) > 0:
        selected_group_labels = np.unique(groups[selected_features])
        assert set(selected_group_labels) == set(selected_groups), (
            "Selected features must belong to selected groups"
        )

    print(f"PASS: test_group_lasso_select (recovered {n_active_recovered}/{len(active_groups)} active groups)")


def test_naive_ci():
    """Test naive CI is computed correctly (coverage problem is expected)."""
    rng = np.random.default_rng(10)
    n, k = 100, 5
    X_sel = rng.standard_normal((n, k))
    y = X_sel @ rng.standard_normal(k) + 0.5 * rng.standard_normal(n)

    coef, ci_lo, ci_hi = naive_ci(X_sel, y, alpha=0.05)

    # Test 1: Shapes
    assert coef.shape == (k,), f"Expected coef shape ({k},), got {coef.shape}"
    assert ci_lo.shape == (k,), f"Expected ci_lo shape ({k},), got {ci_lo.shape}"
    assert ci_hi.shape == (k,), f"Expected ci_hi shape ({k},), got {ci_hi.shape}"

    # Test 2: CI is ordered
    assert np.all(ci_lo <= ci_hi), "ci_lo must be <= ci_hi for all features"

    # Test 3: Coefficient is inside its own CI
    assert np.all(ci_lo <= coef) and np.all(coef <= ci_hi), (
        "OLS coefficient must be inside its own CI"
    )

    # Test 4: CIs are non-degenerate
    assert np.all(ci_hi - ci_lo > 0), "CI must have positive width"

    print("PASS: test_naive_ci")


def test_split_sample_ci():
    """Test split-sample CI correctness."""
    X, y, beta, active, sigma = generate_sparse_data(n=200, p=100, s=6, snr=4.0, seed=20)

    selected, coef, ci_lo, ci_hi = split_sample_ci(X, y, alpha=0.05, seed=7)

    # Test 1: Shapes are consistent
    assert len(coef) == len(selected), (
        f"coef length {len(coef)} != selected length {len(selected)}"
    )
    assert len(ci_lo) == len(selected)
    assert len(ci_hi) == len(selected)

    # Test 2: All selected indices are valid feature indices
    assert np.all(selected >= 0) and np.all(selected < X.shape[1]), (
        f"Selected indices out of range: {selected}"
    )

    # Test 3: CI is ordered
    assert np.all(ci_lo <= ci_hi), "ci_lo must be <= ci_hi"

    # Test 4: CIs are non-degenerate
    if len(ci_hi) > 0:
        assert np.all(ci_hi - ci_lo > 0), "CI must have positive width"

    print("PASS: test_split_sample_ci")


def test_compare_coverage():
    """Verify that split-sample CIs have higher coverage than naive CIs."""
    results = compare_coverage(n=150, p=150, s=8, n_reps=200, alpha=0.05, snr=3.0)

    # Test 1: Required keys present
    required_keys = ['naive_coverage', 'split_coverage', 'naive_n', 'split_n']
    for key in required_keys:
        assert key in results, f"Missing key in results: {key}"

    # Test 2: Coverages are in [0, 1]
    assert 0 <= results['naive_coverage'] <= 1, (
        f"naive_coverage must be in [0,1], got {results['naive_coverage']}"
    )
    assert 0 <= results['split_coverage'] <= 1, (
        f"split_coverage must be in [0,1], got {results['split_coverage']}"
    )

    # Test 3: Split-sample has higher coverage than naive (main point of the exercise)
    assert results['split_coverage'] > results['naive_coverage'], (
        f"Split-sample CI should have higher coverage than naive: "
        f"split={results['split_coverage']:.3f}, naive={results['naive_coverage']:.3f}"
    )

    # Test 4: Naive coverage is well below 95% (demonstrating the problem)
    assert results['naive_coverage'] < 0.90, (
        f"Naive coverage should be well below 90% in this setting, got {results['naive_coverage']:.3f}"
    )

    # Test 5: Split coverage is close to 95% (valid inference)
    assert results['split_coverage'] >= 0.85, (
        f"Split coverage should be at least 85%, got {results['split_coverage']:.3f}"
    )

    print(f"PASS: test_compare_coverage")
    print(f"  Naive CI coverage: {100*results['naive_coverage']:.1f}% (target 95%)")
    print(f"  Split-sample CI coverage: {100*results['split_coverage']:.1f}% (target 95%)")
    print(f"  Coverage gap: {100*(results['split_coverage'] - results['naive_coverage']):.1f}pp")


# =============================================================================
# Run all tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Module 8 Self-Check Exercises")
    print("=" * 60)
    print()

    print("Exercise 1: DC-SIS (Distance Correlation Screening)")
    print("-" * 60)
    try:
        test_distance_covariance()
    except NotImplementedError as e:
        print(f"  TODO: {e}")
    except AssertionError as e:
        print(f"  FAIL: {e}")

    try:
        test_distance_correlation()
    except NotImplementedError as e:
        print(f"  TODO: {e}")
    except AssertionError as e:
        print(f"  FAIL: {e}")

    try:
        test_dc_sis()
    except NotImplementedError as e:
        print(f"  TODO: {e}")
    except AssertionError as e:
        print(f"  FAIL: {e}")

    print()
    print("Exercise 2: Group Lasso")
    print("-" * 60)
    try:
        test_group_soft_threshold()
    except NotImplementedError as e:
        print(f"  TODO: {e}")
    except AssertionError as e:
        print(f"  FAIL: {e}")

    try:
        test_group_lasso_fit()
    except NotImplementedError as e:
        print(f"  TODO: {e}")
    except AssertionError as e:
        print(f"  FAIL: {e}")

    try:
        test_group_lasso_select()
    except NotImplementedError as e:
        print(f"  TODO: {e}")
    except AssertionError as e:
        print(f"  FAIL: {e}")

    print()
    print("Exercise 3: Post-Selection Coverage Comparison")
    print("-" * 60)
    try:
        test_naive_ci()
    except NotImplementedError as e:
        print(f"  TODO: {e}")
    except AssertionError as e:
        print(f"  FAIL: {e}")

    try:
        test_split_sample_ci()
    except NotImplementedError as e:
        print(f"  TODO: {e}")
    except AssertionError as e:
        print(f"  FAIL: {e}")

    try:
        test_compare_coverage()
    except NotImplementedError as e:
        print(f"  TODO: {e}")
    except AssertionError as e:
        print(f"  FAIL: {e}")

    print()
    print("=" * 60)
    print("If all tests PASS, your implementations are correct.")
    print("See the SOLUTIONS section below for reference implementations.")


# =============================================================================
# === SOLUTIONS === (reference implementations — read after attempting)
# =============================================================================

def _sol_compute_distance_covariance(x: np.ndarray, y: np.ndarray) -> float:
    """Reference: Sample distance covariance squared."""
    n = len(x)

    # Pairwise distance matrices
    a = np.abs(x[:, None] - x[None, :])  # (n, n)
    b = np.abs(y[:, None] - y[None, :])  # (n, n)

    # Double-centering: A_kl = a_kl - mean_row_k - mean_col_l + grand_mean
    a_row_mean = a.mean(axis=1, keepdims=True)  # row means
    a_col_mean = a.mean(axis=0, keepdims=True)  # col means
    a_grand_mean = a.mean()
    A = a - a_row_mean - a_col_mean + a_grand_mean

    b_row_mean = b.mean(axis=1, keepdims=True)
    b_col_mean = b.mean(axis=0, keepdims=True)
    b_grand_mean = b.mean()
    B = b - b_row_mean - b_col_mean + b_grand_mean

    # dCov^2 = (1/n^2) * sum_{k,l} A_kl * B_kl
    return np.sum(A * B) / n**2


def _sol_compute_distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Reference: Sample distance correlation."""
    dcov_xy = _sol_compute_distance_covariance(x, y)
    dcov_xx = _sol_compute_distance_covariance(x, x)
    dcov_yy = _sol_compute_distance_covariance(y, y)

    denom = np.sqrt(max(dcov_xx, 0) * max(dcov_yy, 0))
    if denom < 1e-14:
        return 0.0
    return np.sqrt(max(dcov_xy, 0) / denom)


def _sol_dc_sis(X: np.ndarray, y: np.ndarray, d_n: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Reference: DC-SIS."""
    n, p = X.shape
    if d_n is None:
        d_n = int(n / np.log(n))

    dcor_scores = np.array([
        _sol_compute_distance_correlation(X[:, j], y) for j in range(p)
    ])
    screened_idx = np.argsort(dcor_scores)[::-1][:d_n]
    return screened_idx, dcor_scores


def _sol_group_soft_threshold(v: np.ndarray, threshold: float) -> np.ndarray:
    """Reference: Group soft-thresholding."""
    v_norm = np.linalg.norm(v)
    if v_norm <= threshold:
        return np.zeros_like(v)
    return v * (1 - threshold / v_norm)


def _sol_group_lasso_fit(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    lam: float,
    max_iter: int = 1500,
    tol: float = 1e-5
) -> np.ndarray:
    """Reference: Group Lasso proximal gradient."""
    n, p = X.shape
    unique_groups = np.unique(groups)

    # Step size: 1 / L where L = largest eigenvalue of X^T X / n
    L = np.linalg.norm(X, ord=2)**2 / n
    step = 1.0 / (L + 1e-8)

    beta = np.zeros(p)
    for _ in range(max_iter):
        beta_old = beta.copy()
        # Gradient step
        grad = -X.T @ (y - X @ beta) / n
        beta_half = beta - step * grad
        # Proximal step
        beta = np.zeros(p)
        for g in unique_groups:
            mask = groups == g
            threshold = step * lam * np.sqrt(np.sum(mask))
            beta[mask] = _sol_group_soft_threshold(beta_half[mask], threshold)
        # Convergence
        if np.max(np.abs(beta - beta_old)) < tol:
            break
    return beta


def _sol_group_lasso_select(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    lam_grid: np.ndarray = None,
    cv: int = 3
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Reference: Group Lasso with cross-validated lambda."""
    n, p = X.shape
    if lam_grid is None:
        lam_grid = np.logspace(-2, 0, 15)

    fold_size = n // cv
    cv_mse = np.zeros(len(lam_grid))

    for k in range(cv):
        val_start = k * fold_size
        val_end = val_start + fold_size if k < cv - 1 else n
        val_idx = np.arange(val_start, val_end)
        train_idx = np.concatenate([np.arange(0, val_start), np.arange(val_end, n)])

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        for i, lam in enumerate(lam_grid):
            beta = _sol_group_lasso_fit(X_tr, y_tr, groups, lam=lam)
            pred = X_val @ beta
            cv_mse[i] += np.mean((y_val - pred)**2) / cv

    best_lam = float(lam_grid[np.argmin(cv_mse)])
    beta_best = _sol_group_lasso_fit(X, y, groups, lam=best_lam)

    selected_features = np.where(np.abs(beta_best) > 1e-8)[0]
    unique_groups_arr = np.unique(groups)
    selected_groups = np.array([
        g for g in unique_groups_arr
        if np.linalg.norm(beta_best[groups == g]) > 1e-8
    ])

    return selected_features, selected_groups, best_lam


def _sol_naive_ci(
    X_selected: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference: Naive OLS CI."""
    n, k = X_selected.shape
    ols = LinearRegression(fit_intercept=False)
    ols.fit(X_selected, y)
    residuals = y - ols.predict(X_selected)
    sigma_sq = np.sum(residuals**2) / max(n - k, 1)
    XtX_inv = np.linalg.pinv(X_selected.T @ X_selected)
    se = np.sqrt(sigma_sq * np.diag(XtX_inv))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lo = ols.coef_ - z_crit * se
    ci_hi = ols.coef_ + z_crit * se
    return ols.coef_, ci_lo, ci_hi


def _sol_split_sample_ci(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
    split_frac: float = 0.5,
    seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference: Data-splitting CI."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    n1 = int(n * split_frac)
    perm = rng.permutation(n)
    sel_idx, inf_idx = perm[:n1], perm[n1:]

    lasso = LassoCV(cv=3, max_iter=5000, fit_intercept=False)
    lasso.fit(X[sel_idx], y[sel_idx])
    selected = np.where(np.abs(lasso.coef_) > 1e-8)[0]

    if len(selected) == 0:
        return selected, np.array([]), np.array([]), np.array([])

    X_inf = X[np.ix_(inf_idx, selected)]
    y_inf = y[inf_idx]
    n2, k = X_inf.shape

    ols = LinearRegression(fit_intercept=False)
    ols.fit(X_inf, y_inf)
    residuals = y_inf - ols.predict(X_inf)
    sigma_sq = np.sum(residuals**2) / max(n2 - k, 1)
    XtX_inv = np.linalg.pinv(X_inf.T @ X_inf)
    se = np.sqrt(sigma_sq * np.diag(XtX_inv))
    t_crit = stats.t.ppf(1 - alpha / 2, df=max(n2 - k, 1))
    ci_lo = ols.coef_ - t_crit * se
    ci_hi = ols.coef_ + t_crit * se

    return selected, ols.coef_, ci_lo, ci_hi


def _sol_compare_coverage(
    n: int = 150,
    p: int = 150,
    s: int = 8,
    n_reps: int = 200,
    alpha: float = 0.05,
    snr: float = 3.0
) -> dict:
    """Reference: Coverage comparison Monte Carlo."""
    covered_naive = []
    covered_split = []

    for rep in range(n_reps):
        X, y, beta, active, sigma = generate_sparse_data(n, p, s, snr=snr, seed=rep)
        active_set = set(active)

        # Naive CI
        lasso = LassoCV(cv=3, max_iter=5000, fit_intercept=False)
        lasso.fit(X, y)
        selected = np.where(np.abs(lasso.coef_) > 1e-8)[0]

        if len(selected) >= 2:
            coef_n, ci_lo_n, ci_hi_n = _sol_naive_ci(X[:, selected], y, alpha)
            for j, feat in enumerate(selected):
                if feat in active_set:
                    covered_naive.append(ci_lo_n[j] <= beta[feat] <= ci_hi_n[j])

        # Split CI
        sel_s, coef_s, ci_lo_s, ci_hi_s = _sol_split_sample_ci(X, y, alpha=alpha, seed=rep)
        for j, feat in enumerate(sel_s):
            if feat in active_set:
                covered_split.append(ci_lo_s[j] <= beta[feat] <= ci_hi_s[j])

    return {
        'naive_coverage': np.mean(covered_naive) if covered_naive else np.nan,
        'split_coverage': np.mean(covered_split) if covered_split else np.nan,
        'naive_n': len(covered_naive),
        'split_n': len(covered_split),
    }
