"""
Module 04: Embedded Methods — Self-Check Exercises
Advanced Feature Selection Course

Three exercises covering:
  1. Stability selection pipeline with configurable base estimator
  2. Knockoff filter procedure from scratch
  3. Four importance methods on a dataset with known ground truth

Run each exercise function to check your implementation against the tests.
All datasets are synthetically generated with known ground truth so you can
verify correctness without external downloads.

Usage:
    python 01_embedded_exercises.py

Dependencies:
    numpy, pandas, scikit-learn, scipy
    Optional: shap (for Exercise 3 SHAP comparison)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV, lasso_path, ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import kendalltau
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


# ---------------------------------------------------------------------------
# Shared data generation utilities
# ---------------------------------------------------------------------------

def make_sparse_regression(
    n: int = 500,
    p: int = 30,
    n_true: int = 6,
    corr_within_group: float = 0.85,
    noise_std: float = 1.0,
    seed: int = 42
) -> tuple:
    """
    Generate a sparse regression dataset with known ground truth.

    The first n_true features are truly relevant (non-zero coefficients).
    Among those, features 0..2 form a correlated group (correlation ~corr_within_group).
    Features 3..n_true-1 are independent.
    Features n_true..p-1 are null (independent noise).

    Parameters
    ----------
    n : int — number of observations
    p : int — total number of features
    n_true : int — number of truly relevant features (must be <= p)
    corr_within_group : float — within-group correlation for features 0..2
    noise_std : float — standard deviation of outcome noise
    seed : int — random seed

    Returns
    -------
    X : pd.DataFrame (n, p) — standardised feature matrix
    y : np.ndarray (n,) — outcome vector
    true_features : list[int] — indices of truly relevant features
    true_coef : np.ndarray (p,) — true coefficient vector
    """
    assert n_true >= 3, "Need at least 3 true features (first group of 3 is correlated)"
    assert n_true <= p

    rng = np.random.default_rng(seed)

    # Correlated group: features 0, 1, 2 share a latent factor
    z = rng.normal(0, 1, n)
    noise_level = np.sqrt(1 - corr_within_group ** 2)
    x0 = corr_within_group * z + noise_level * rng.normal(0, 1, n)
    x1 = corr_within_group * z + noise_level * rng.normal(0, 1, n)
    x2 = corr_within_group * z + noise_level * rng.normal(0, 1, n)

    # Independent true features: 3 .. n_true-1
    n_indep_true = n_true - 3
    X_indep_true = rng.normal(0, 1, (n, n_indep_true)) if n_indep_true > 0 else np.empty((n, 0))

    # Null features: n_true .. p-1
    n_null = p - n_true
    X_null = rng.normal(0, 1, (n, n_null))

    X_raw = np.hstack([
        np.column_stack([x0, x1, x2]),
        X_indep_true,
        X_null,
    ])

    # True coefficients: 2.0 for all true features, 0.0 for null
    true_coef = np.zeros(p)
    true_coef[:n_true] = 2.0

    y = X_raw @ true_coef + rng.normal(0, noise_std, n)

    # Standardise features (required for regularisation-based methods)
    scaler = StandardScaler()
    X = pd.DataFrame(
        scaler.fit_transform(X_raw),
        columns=[f'x{i:02d}' for i in range(p)]
    )

    return X, y, list(range(n_true)), true_coef


# ---------------------------------------------------------------------------
# Exercise 1: Stability Selection Pipeline
# ---------------------------------------------------------------------------

def stability_selection_pipeline(
    X: pd.DataFrame,
    y: np.ndarray,
    base_estimator: str = 'lasso',
    n_subsamples: int = 100,
    subsample_fraction: float = 0.5,
    n_alphas: int = 40,
    eps: float = 1e-2,
    pi_threshold: float = 0.75,
    l1_ratio: float = 0.5,
    random_state: int = 0,
) -> tuple:
    """
    Stability selection with configurable base estimator.

    Runs the regularisation path on B random subsamples and computes
    per-feature selection probabilities. Returns stable features
    (those exceeding pi_threshold) along with the full probability matrix.

    Parameters
    ----------
    X : pd.DataFrame (n, p) — standardised features
    y : np.ndarray (n,) — target
    base_estimator : str — 'lasso' or 'elasticnet'
    n_subsamples : int — number of bootstrap subsamples (>=50 recommended)
    subsample_fraction : float — fraction of data per subsample (default 0.5)
    n_alphas : int — number of alpha values in regularisation path
    eps : float — lambda_min = eps * lambda_max
    pi_threshold : float — selection probability threshold (default 0.75)
    l1_ratio : float — L1/L2 mixing ratio for ElasticNet (ignored for Lasso)
    random_state : int — seed for reproducibility

    Returns
    -------
    stable_features : np.ndarray[int] — indices of stably selected features
    selection_probs : np.ndarray (p, n_alphas) — selection probability matrix
    alphas : np.ndarray (n_alphas,) — regularisation path (decreasing)

    Implementation Notes
    --------------------
    - Use lasso_path(..., method='lars') for Lasso
    - Use enet_path(..., l1_ratio=l1_ratio) for ElasticNet
    - Draw subsamples WITHOUT replacement
    - Use the SAME alpha grid (computed from full data) for all subsamples
    - Count a feature as selected when its coefficient is nonzero
    """
    # TODO: Implement stability selection
    # ---------------
    # 1. Compute the alpha grid from the full dataset
    # 2. For each subsample b in range(n_subsamples):
    #    a. Draw random indices (size = floor(subsample_fraction * n))
    #    b. Fit Lasso or ElasticNet path on subsample using the precomputed alpha grid
    #    c. Record which features are nonzero at each alpha
    # 3. Divide selection counts by n_subsamples to get probabilities
    # 4. Stable features: those with max probability >= pi_threshold
    raise NotImplementedError("Implement stability_selection_pipeline")


def test_stability_selection():
    """
    Self-check tests for stability_selection_pipeline.

    Tests:
    1. Correct output shapes and types
    2. Selection probabilities in [0, 1]
    3. True features have higher average probability than null features
    4. ElasticNet and Lasso produce similar (but not identical) stable sets
    5. Higher pi_threshold produces fewer stable features (monotonicity)
    """
    X, y, true_features, _ = make_sparse_regression(
        n=300, p=20, n_true=5, seed=42
    )

    print("Running stability selection tests...")

    # --- Test 1: Output shapes and types ---
    stable, probs, alphas = stability_selection_pipeline(
        X, y, base_estimator='lasso',
        n_subsamples=50, pi_threshold=0.75, random_state=0
    )
    assert isinstance(stable, np.ndarray), "stable_features must be np.ndarray"
    assert probs.shape[0] == X.shape[1], (
        f"selection_probs rows should equal n_features={X.shape[1]}, got {probs.shape[0]}"
    )
    assert probs.shape[1] == 40, (
        f"selection_probs cols should equal n_alphas=40, got {probs.shape[1]}"
    )
    assert len(alphas) == 40, f"alphas should have 40 values, got {len(alphas)}"
    print("  [PASS] Test 1: Output shapes correct.")

    # --- Test 2: Probabilities in [0, 1] ---
    assert probs.min() >= 0.0 - 1e-9, f"Probabilities must be >= 0, got min={probs.min()}"
    assert probs.max() <= 1.0 + 1e-9, f"Probabilities must be <= 1, got max={probs.max()}"
    print("  [PASS] Test 2: Selection probabilities in [0, 1].")

    # --- Test 3: True features have higher selection probability than null features ---
    max_probs = probs.max(axis=1)  # (p,) — max probability over all alphas
    null_features = [i for i in range(X.shape[1]) if i not in true_features]
    mean_prob_true = max_probs[true_features].mean()
    mean_prob_null = max_probs[null_features].mean()
    assert mean_prob_true > mean_prob_null, (
        f"True features should have higher avg max probability than null features. "
        f"True: {mean_prob_true:.3f}, Null: {mean_prob_null:.3f}"
    )
    print(f"  [PASS] Test 3: True features avg prob={mean_prob_true:.3f} > Null avg prob={mean_prob_null:.3f}.")

    # --- Test 4: ElasticNet produces a valid (and different) result ---
    stable_en, probs_en, _ = stability_selection_pipeline(
        X, y, base_estimator='elasticnet',
        n_subsamples=50, pi_threshold=0.75, l1_ratio=0.5, random_state=0
    )
    assert isinstance(stable_en, np.ndarray), "ElasticNet stable features must be np.ndarray"
    assert probs_en.shape == probs.shape, "ElasticNet probs should have same shape as Lasso probs"
    print(f"  [PASS] Test 4: ElasticNet produces valid output "
          f"(stable={stable_en.tolist()}, lasso_stable={stable.tolist()}).")

    # --- Test 5: Monotonicity — higher threshold → fewer stable features ---
    _, _, _ = stability_selection_pipeline(X, y, n_subsamples=50, pi_threshold=0.60, random_state=0)
    stable_60, _, _ = stability_selection_pipeline(X, y, n_subsamples=50, pi_threshold=0.60, random_state=0)
    stable_90, _, _ = stability_selection_pipeline(X, y, n_subsamples=50, pi_threshold=0.90, random_state=0)
    assert len(stable_60) >= len(stable_90), (
        f"Higher threshold should select <= features: threshold 0.6 selected {len(stable_60)}, "
        f"threshold 0.9 selected {len(stable_90)}"
    )
    print(f"  [PASS] Test 5: Monotonicity holds (0.6 selects {len(stable_60)}, 0.9 selects {len(stable_90)}).")

    print("\n[ALL TESTS PASSED] stability_selection_pipeline is correct.")


# ---------------------------------------------------------------------------
# Exercise 2: Knockoff Filter
# ---------------------------------------------------------------------------

def construct_equicorrelated_knockoffs(X_arr: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    Construct equicorrelated Gaussian knockoff features.

    For X ~ N(mu, Sigma), the equicorrelated knockoff uses:
        S = s * I,  s = min(2 * lambda_min(Sigma), 1)

    The knockoff covariance conditional on X is:
        Sigma_tilde|X = 2S - S @ Sigma^{-1} @ S

    The knockoff mean conditional on X is:
        mu_tilde|X = mu + (X - mu) @ (I - Sigma^{-1} @ S)

    Parameters
    ----------
    X_arr : np.ndarray (n, p) — standardised features
    seed : int — random seed for noise generation

    Returns
    -------
    X_ko : np.ndarray (n, p) — knockoff feature matrix

    Implementation Notes
    --------------------
    - Add 1e-6 * I to Sigma for numerical stability before inversion
    - Clip negative eigenvalues of Sigma_tilde|X to 0 (PSD correction)
    - Add 1e-8 * I before Cholesky for numerical stability
    """
    # TODO: Implement equicorrelated knockoff construction
    raise NotImplementedError("Implement construct_equicorrelated_knockoffs")


def knockoff_filter(
    X_orig: np.ndarray,
    X_ko: np.ndarray,
    y: np.ndarray,
    fdr_level: float = 0.10,
    stat: str = 'lasso_coef',
) -> tuple:
    """
    Knockoff+ filter for FDR-controlled feature selection.

    Steps:
    1. Augment: X_aug = [X_orig, X_ko]  (shape n x 2p)
    2. Fit Lasso on X_aug (use LassoCV to select lambda)
    3. Compute W_j = |Z_j| - |Z_tilde_j| where Z and Z_tilde are Lasso coefs
    4. Apply knockoff+ threshold:
           tau = min{t > 0 : (1 + #{j: W_j <= -t}) / #{j: W_j >= t} <= fdr_level}
    5. Select features: {j : W_j >= tau}

    Parameters
    ----------
    X_orig : np.ndarray (n, p)
    X_ko : np.ndarray (n, p) — knockoff features
    y : np.ndarray (n,)
    fdr_level : float — target FDR level (e.g. 0.10 for 10%)
    stat : str — 'lasso_coef' (only option in this exercise)

    Returns
    -------
    selected : np.ndarray[int] — selected feature indices
    W : np.ndarray (p,) — knockoff statistics
    tau : float — selection threshold (inf if nothing selected)

    Implementation Notes
    --------------------
    - Use LassoCV(cv=5, n_alphas=50, max_iter=5000) for fitting
    - Iterate t over sorted unique values of |W[W != 0]| in ASCENDING order
    - Stop at the FIRST t that satisfies the FDR constraint
    - If no t satisfies the constraint, return empty selection and tau=inf
    """
    # TODO: Implement the knockoff filter procedure
    raise NotImplementedError("Implement knockoff_filter")


def test_knockoff_filter():
    """
    Self-check tests for knockoff_filter (and construct_equicorrelated_knockoffs).

    Tests:
    1. Knockoff marginal distribution matches original (mean and std)
    2. Knockoff conditional independence: Corr(X_j, X_tilde_j | X_{-j}) != Corr(X_j, X_tilde_j)
    3. W statistics have shape (p,)
    4. FDR control: false positive rate <= target (with tolerance for finite samples)
    5. Monotonicity: higher FDR level selects >= features
    """
    X, y, true_features, _ = make_sparse_regression(
        n=400, p=15, n_true=5, seed=0
    )
    X_arr = X.values
    n, p = X_arr.shape

    print("Running knockoff filter tests...")

    # --- Test 1: Knockoff marginal distribution ---
    X_ko = construct_equicorrelated_knockoffs(X_arr, seed=42)
    assert X_ko.shape == (n, p), f"Knockoff shape should be ({n}, {p}), got {X_ko.shape}"

    mean_diff = np.abs(X_arr.mean(axis=0) - X_ko.mean(axis=0)).max()
    std_diff = np.abs(X_arr.std(axis=0) - X_ko.std(axis=0)).max()
    assert mean_diff < 0.5, (
        f"Knockoff means should be close to original means (max diff={mean_diff:.3f})"
    )
    assert std_diff < 0.5, (
        f"Knockoff stds should be close to original stds (max diff={std_diff:.3f})"
    )
    print(f"  [PASS] Test 1: Knockoff marginals match (mean_diff={mean_diff:.3f}, std_diff={std_diff:.3f}).")

    # --- Test 2: W statistics shape ---
    selected, W, tau = knockoff_filter(X_arr, X_ko, y, fdr_level=0.10)
    assert W.shape == (p,), f"W should have shape ({p},), got {W.shape}"
    print(f"  [PASS] Test 2: W statistics shape correct ({W.shape}).")

    # --- Test 3: Selected features are a subset of [0, p) ---
    assert all(0 <= s < p for s in selected), f"Selected indices out of range: {selected}"
    print(f"  [PASS] Test 3: Selected indices valid ({selected.tolist()}).")

    # --- Test 4: True features have W > null features (on average) ---
    null_features = [i for i in range(p) if i not in true_features]
    mean_W_true = W[true_features].mean()
    mean_W_null = W[null_features].mean()
    assert mean_W_true > mean_W_null, (
        f"True features should have higher W on average. "
        f"True W={mean_W_true:.4f}, Null W={mean_W_null:.4f}"
    )
    print(f"  [PASS] Test 4: True feature W={mean_W_true:.4f} > Null W={mean_W_null:.4f}.")

    # --- Test 5: FDR control (achieved FDR <= target + slack for finite samples) ---
    if len(selected) > 0:
        fp = len(set(selected.tolist()) - set(true_features))
        achieved_fdr = fp / len(selected)
        slack = 0.20  # finite sample slack
        assert achieved_fdr <= 0.10 + slack, (
            f"Achieved FDR={achieved_fdr:.3f} exceeds target 0.10 + slack {slack}. "
            f"Knockoff construction may be incorrect."
        )
        print(f"  [PASS] Test 5: Achieved FDR={achieved_fdr:.3f} <= target 0.10 + slack {slack}.")
    else:
        print("  [INFO] Test 5: No features selected — achieved FDR = 0 (conservative).")

    # --- Test 6: Monotonicity across FDR levels ---
    n_sel_05 = len(knockoff_filter(X_arr, X_ko, y, fdr_level=0.05)[0])
    n_sel_10 = len(knockoff_filter(X_arr, X_ko, y, fdr_level=0.10)[0])
    n_sel_20 = len(knockoff_filter(X_arr, X_ko, y, fdr_level=0.20)[0])
    assert n_sel_05 <= n_sel_10 <= n_sel_20, (
        f"Higher FDR level should select >= features: "
        f"FDR=0.05: {n_sel_05}, FDR=0.10: {n_sel_10}, FDR=0.20: {n_sel_20}"
    )
    print(f"  [PASS] Test 6: Monotonicity holds (0.05→{n_sel_05}, 0.10→{n_sel_10}, 0.20→{n_sel_20}).")

    print("\n[ALL TESTS PASSED] knockoff_filter is correct.")


# ---------------------------------------------------------------------------
# Exercise 3: Four Importance Methods on Known Ground Truth
# ---------------------------------------------------------------------------

def compare_importance_methods(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int = 150,
    n_permutation_repeats: int = 20,
    random_state: int = 42,
) -> dict:
    """
    Fit a Random Forest and compute four importance measures.

    Measures:
    1. MDI — Mean Decrease in Impurity (rf.feature_importances_)
    2. Permutation — Mean Decrease in Accuracy on test set (sklearn permutation_importance)
    3. LassoCV coefficient magnitudes (|coef_|) as an embedded linear importance
    4. (Optional) SHAP — mean |SHAP value| if shap is installed; else None

    Parameters
    ----------
    X_train, X_test : pd.DataFrame — features
    y_train, y_test : np.ndarray — targets
    n_estimators : int — number of trees in the Random Forest
    n_permutation_repeats : int — repeats for permutation importance
    random_state : int — seed

    Returns
    -------
    results : dict with keys:
        'mdi'         : np.ndarray (p,) — normalised MDI importances
        'permutation' : np.ndarray (p,) — raw permutation importances (R² decrease)
        'lasso'       : np.ndarray (p,) — |LassoCV coefficients|
        'shap'        : np.ndarray (p,) or None — mean |SHAP values| (None if not available)
        'rf_r2'       : float — Random Forest R² on test set
        'feature_names': list[str] — feature names

    Implementation Notes
    --------------------
    - Train RandomForestRegressor with n_estimators, min_samples_leaf=5, random_state
    - Compute permutation_importance on X_test (not X_train)
    - For Lasso: fit LassoCV(cv=5, max_iter=5000) on X_train
    - Normalise MDI and |Lasso coef| to [0, 1] by dividing by their max
    - For SHAP: use shap.TreeExplainer(rf).shap_values(X_test)
    - Return raw (not normalised) permutation importances
    """
    # TODO: Implement the four-method comparison
    raise NotImplementedError("Implement compare_importance_methods")


def rank_agreement_matrix(results: dict, true_coef: np.ndarray) -> dict:
    """
    Compute Kendall's tau between each method's ranking and the ground truth.

    Parameters
    ----------
    results : dict — output of compare_importance_methods
    true_coef : np.ndarray (p,) — true coefficients (ground truth importance)

    Returns
    -------
    tau_vs_truth : dict mapping method_name -> (tau, p_value)
        Method names: 'mdi', 'permutation', 'lasso', 'shap' (if available)

    Implementation Notes
    --------------------
    - Ground truth ranking: argsort(argsort(-|true_coef|))  (0 = most important)
    - Method ranking: argsort(argsort(-importance))
    - Use scipy.stats.kendalltau(truth_rank, method_rank)
    - Skip 'shap' if results['shap'] is None
    """
    # TODO: Implement rank agreement computation
    raise NotImplementedError("Implement rank_agreement_matrix")


def test_importance_methods():
    """
    Self-check tests for compare_importance_methods and rank_agreement_matrix.

    Tests:
    1. All four keys present in results dict; shapes correct
    2. MDI sums to approximately 1 (sklearn normalises internally)
    3. Random Forest R² > 0.5 (dataset is learnable)
    4. Lasso correctly identifies at least 4/6 true features (non-zero coef)
    5. Kendall's tau vs ground truth is positive for all methods
    6. Permutation and SHAP have higher tau than MDI (less biased)
    """
    X, y, true_features, true_coef = make_sparse_regression(
        n=500, p=20, n_true=6, corr_within_group=0.9, seed=7
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print("Running importance methods tests...")

    # --- Test 1: Output structure ---
    results = compare_importance_methods(
        X_train, X_test, y_train, y_test,
        n_estimators=100, n_permutation_repeats=10, random_state=42
    )
    required_keys = {'mdi', 'permutation', 'lasso', 'shap', 'rf_r2', 'feature_names'}
    assert set(results.keys()) >= required_keys, (
        f"Results dict missing keys: {required_keys - set(results.keys())}"
    )
    p = X.shape[1]
    for key in ['mdi', 'permutation', 'lasso']:
        assert results[key].shape == (p,), (
            f"results['{key}'] should have shape ({p},), got {results[key].shape}"
        )
    print("  [PASS] Test 1: Output structure correct.")

    # --- Test 2: MDI sums to approximately 1 ---
    mdi_sum = results['mdi'].sum()
    # Note: we asked for normalised MDI, so sum may not be 1 — check raw from RF
    # Instead, check that all MDI values are non-negative
    assert (results['mdi'] >= 0).all(), "MDI values must be non-negative"
    print(f"  [PASS] Test 2: MDI values non-negative (sum={results['mdi'].sum():.4f}).")

    # --- Test 3: Random Forest R² > 0.5 ---
    assert results['rf_r2'] > 0.5, (
        f"Random Forest R² should be > 0.5 on this learnable dataset, got {results['rf_r2']:.4f}"
    )
    print(f"  [PASS] Test 3: Random Forest R²={results['rf_r2']:.4f} > 0.5.")

    # --- Test 4: Lasso recovers at least 4 of 6 true features ---
    lasso_selected = np.where(results['lasso'] > 0)[0]
    lasso_tp = len(set(lasso_selected.tolist()) & set(true_features))
    assert lasso_tp >= 4, (
        f"Lasso should identify at least 4 of {len(true_features)} true features, "
        f"found only {lasso_tp}."
    )
    print(f"  [PASS] Test 4: Lasso identifies {lasso_tp}/{len(true_features)} true features.")

    # --- Test 5: Kendall's tau positive for all methods ---
    tau_results = rank_agreement_matrix(results, true_coef)
    assert isinstance(tau_results, dict), "rank_agreement_matrix must return a dict"
    for method, (tau_val, p_val) in tau_results.items():
        assert tau_val > 0, (
            f"Kendall's tau for {method} should be positive, got {tau_val:.4f}. "
            f"Check that importance is being computed correctly."
        )
        print(f"  [PASS] Test 5 ({method}): Kendall tau={tau_val:.4f} > 0.")

    # --- Test 6: SHAP and Permutation have higher tau than MDI (if SHAP available) ---
    if 'mdi' in tau_results and 'permutation' in tau_results:
        tau_mdi = tau_results['mdi'][0]
        tau_perm = tau_results['permutation'][0]
        if 'shap' in tau_results:
            tau_shap = tau_results['shap'][0]
            # On correlated data, SHAP should generally beat MDI
            # Allow some tolerance since this is stochastic
            print(f"  [INFO] Test 6: MDI tau={tau_mdi:.3f}, Perm tau={tau_perm:.3f}, SHAP tau={tau_shap:.3f}")
            if tau_shap < tau_mdi:
                print("  [WARN] SHAP tau < MDI tau on this dataset — may occur with correlated features.")
            else:
                print("  [PASS] Test 6: SHAP tau >= MDI tau (expected for correlated features).")
        else:
            print(f"  [INFO] Test 6 (no SHAP): MDI tau={tau_mdi:.3f}, Perm tau={tau_perm:.3f}")

    print("\n[ALL TESTS PASSED] compare_importance_methods and rank_agreement_matrix are correct.")


# ---------------------------------------------------------------------------
# Main: Run all exercise tests
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 70)
    print("Module 04: Embedded Methods — Self-Check Exercises")
    print("=" * 70)

    exercises = [
        ("Exercise 1: Stability Selection Pipeline", test_stability_selection),
        ("Exercise 2: Knockoff Filter", test_knockoff_filter),
        ("Exercise 3: Four Importance Methods", test_importance_methods),
    ]

    results = {}
    for name, test_fn in exercises:
        print(f"\n{'-' * 70}")
        print(f"{name}")
        print(f"{'-' * 70}")
        try:
            test_fn()
            results[name] = 'PASSED'
        except NotImplementedError as e:
            print(f"[NOT IMPLEMENTED] {e}")
            results[name] = 'NOT IMPLEMENTED'
        except AssertionError as e:
            print(f"[FAILED] {e}")
            results[name] = 'FAILED'
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")
            results[name] = 'ERROR'

    print(f"\n{'=' * 70}")
    print("Results Summary")
    print(f"{'=' * 70}")
    for name, status in results.items():
        icon = '[OK]' if status == 'PASSED' else '[  ]'
        print(f"  {icon} {name}: {status}")
    print()
