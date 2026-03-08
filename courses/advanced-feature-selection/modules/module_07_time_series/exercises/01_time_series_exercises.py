"""
Module 7: Feature Selection for Time Series — Self-Check Exercises

Three advanced exercises covering:
1. Nonlinear Granger causality via kernel ridge regression + permutation test
2. Purged CV splitter compatible with sklearn cross_val_score
3. Feature drift monitor with PSI, KS, and Wasserstein distance

Run each exercise section independently. Assertions at the end of each section
verify correctness — read the assertion messages carefully if they fail.

Reference: de Prado, M.L. (2018). Advances in Financial Machine Learning. Wiley.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import ks_2samp, wasserstein_distance, f as f_dist
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Shared test data (used across all three exercises)
# =============================================================================

def _make_test_data(T: int = 300, n_features: int = 15, seed: int = 42):
    """
    Generate multivariate time series for exercise testing.

    Returns:
        target   : pd.Series of T observations
        features : pd.DataFrame of T x n_features
        causal   : list of feature names that truly Granger-cause target
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-31", periods=T, freq="ME")

    # Target: AR(1)
    target_arr = np.zeros(T)
    target_arr[0] = rng.normal(0, 0.5)
    for t in range(1, T):
        target_arr[t] = 0.4 * target_arr[t - 1] + rng.normal(0, 0.5)

    target = pd.Series(target_arr, index=dates, name="target")

    # Features
    feat_arr = np.zeros((T, n_features))
    causal_indices = [0, 1, 2, 3]  # first 4 features Granger-cause target (linearly)
    nonlinear_causal = [4, 5]  # nonlinear Granger causality (quadratic lag effect)

    for j in range(n_features):
        phi = rng.uniform(0.2, 0.6)
        feat_arr[0, j] = rng.normal(0, 1)
        for t in range(1, T):
            feat_arr[t, j] = phi * feat_arr[t - 1, j] + rng.normal(0, 1)
            if j in causal_indices:
                feat_arr[t, j] += 0.25 * target_arr[t - 1]
            elif j in nonlinear_causal:
                # Nonlinear: feature depends on squared lagged target
                feat_arr[t, j] += 0.30 * target_arr[t - 1] ** 2 - 0.15

    feat_names = [f"F{j+1:02d}" for j in range(n_features)]
    features = pd.DataFrame(feat_arr, index=dates, columns=feat_names)
    causal_names = [feat_names[j] for j in causal_indices + nonlinear_causal]

    return target, features, causal_names


print("Generating test data...")
TARGET, FEATURES, CAUSAL_FEATURES = _make_test_data()
print(f"Target: {len(TARGET)} observations")
print(f"Features: {FEATURES.shape[1]}")
print(f"True causal features: {CAUSAL_FEATURES}")


# =============================================================================
# Exercise 1: Nonlinear Granger Causality via Kernel Ridge Regression
# =============================================================================
print("\n" + "=" * 70)
print("EXERCISE 1: Nonlinear Granger Causality (Kernel Ridge Regression)")
print("=" * 70)

"""
Task:
    Implement kernel_granger_causality() that tests whether a feature
    nonlinearly Granger-causes the target using:
    1. Kernel ridge regression (KernelRidge with rbf kernel)
    2. Permutation testing for significance (no asymptotic assumptions)

    The test compares:
    - Restricted model: predict target from its own lagged values only
    - Unrestricted model: predict target from own lags + feature lags

    Significance: permute feature lags and measure how often the permuted
    model does as well as or better than the true model.

    Why kernel ridge? It can capture nonlinear relationships that the
    standard linear Granger F-test misses entirely.
"""


def make_lag_matrix(arr: np.ndarray, lag: int) -> np.ndarray:
    """
    Construct lag matrix from a 1-D array.

    Parameters
    ----------
    arr : np.ndarray, shape (T,)
    lag : int — number of lags to stack

    Returns
    -------
    np.ndarray, shape (T - lag, lag)
        Each row contains [arr[t-lag], ..., arr[t-1]] for t = lag, ..., T-1.
    """
    rows = []
    for t in range(lag, len(arr)):
        rows.append(arr[t - lag : t])
    return np.array(rows)


def kernel_granger_causality(
    target: pd.Series,
    feature: pd.Series,
    lag: int = 3,
    kernel: str = "rbf",
    alpha: float = 0.5,
    n_permutations: int = 200,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Nonlinear Granger causality test using kernel ridge regression
    and permutation testing.

    Parameters
    ----------
    target : pd.Series
        Target variable Y (stationary).
    feature : pd.Series
        Candidate feature X (stationary).
    lag : int
        Number of lags for both target and feature history.
    kernel : str
        Kernel type for KernelRidge ('rbf', 'polynomial', 'linear').
    alpha : float
        Regularisation strength for KernelRidge.
    n_permutations : int
        Number of permutation replicates for p-value.
    cv_folds : int
        Cross-validation folds for MSE estimation.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict with keys:
        'test_stat'       : float — fractional reduction in MSE (0 = no improvement)
        'pvalue'          : float — permutation p-value
        'mse_restricted'  : float — CV MSE of restricted model (Y lags only)
        'mse_unrestricted': float — CV MSE of unrestricted model (Y + X lags)
        'significant'     : bool — True if pvalue < 0.05
    """
    rng = np.random.default_rng(random_state)

    # Step 1: Align series and drop NaN
    data = pd.concat([target.rename("y"), feature.rename("x")], axis=1).dropna()
    y_arr = data["y"].values
    x_arr = data["x"].values

    # Step 2: Build lag matrices
    y_lags = make_lag_matrix(y_arr, lag)   # shape (T-lag, lag)
    x_lags = make_lag_matrix(x_arr, lag)   # shape (T-lag, lag)
    y_target = y_arr[lag:]                  # shape (T-lag,)

    # Step 3: Restricted model — KernelRidge on Y lags only
    model_r = KernelRidge(kernel=kernel, alpha=alpha)
    cv_scores_r = cross_val_score(
        model_r, y_lags, y_target,
        cv=cv_folds, scoring="neg_mean_squared_error"
    )
    mse_r = -cv_scores_r.mean()

    # Step 4: Unrestricted model — KernelRidge on Y lags + X lags
    Xy = np.hstack([y_lags, x_lags])
    model_u = KernelRidge(kernel=kernel, alpha=alpha)
    cv_scores_u = cross_val_score(
        model_u, Xy, y_target,
        cv=cv_folds, scoring="neg_mean_squared_error"
    )
    mse_u = -cv_scores_u.mean()

    # Step 5: Observed test statistic — fractional MSE reduction
    obs_stat = (mse_r - mse_u) / (mse_r + 1e-12)

    # Step 6: Permutation test
    # Under H0 (X does not cause Y), permuting X lags should not change MSE
    null_stats = []
    for _ in range(n_permutations):
        x_lags_perm = rng.permutation(x_lags)
        Xy_perm = np.hstack([y_lags, x_lags_perm])
        scores_perm = cross_val_score(
            KernelRidge(kernel=kernel, alpha=alpha),
            Xy_perm, y_target,
            cv=cv_folds, scoring="neg_mean_squared_error"
        )
        mse_perm = -scores_perm.mean()
        null_stats.append((mse_r - mse_perm) / (mse_r + 1e-12))

    null_stats = np.array(null_stats)
    p_value = float(np.mean(null_stats >= obs_stat))

    return {
        "test_stat": float(obs_stat),
        "pvalue": p_value,
        "mse_restricted": float(mse_r),
        "mse_unrestricted": float(mse_u),
        "significant": p_value < 0.05,
    }


# --- Run Exercise 1 ---
print("\nTesting kernel Granger causality on F05 (nonlinearly causal) vs F10 (noise):")
print("(This may take ~30 seconds due to cross-validation...)")

result_causal = kernel_granger_causality(TARGET, FEATURES["F05"], lag=3,
                                          n_permutations=100, random_state=42)
result_noise = kernel_granger_causality(TARGET, FEATURES["F10"], lag=3,
                                         n_permutations=100, random_state=42)

print(f"\nF05 (nonlinearly causal):")
print(f"  Test stat: {result_causal['test_stat']:.4f}")
print(f"  p-value:   {result_causal['pvalue']:.3f}")
print(f"  Significant: {result_causal['significant']}")

print(f"\nF10 (noise):")
print(f"  Test stat: {result_noise['test_stat']:.4f}")
print(f"  p-value:   {result_noise['pvalue']:.3f}")
print(f"  Significant: {result_noise['significant']}")

# Assertions
assert isinstance(result_causal, dict), "Return type must be dict"
assert all(k in result_causal for k in ["test_stat", "pvalue", "mse_restricted",
                                         "mse_unrestricted", "significant"]), \
    "Missing keys in result dict"
assert 0.0 <= result_causal["pvalue"] <= 1.0, \
    f"p-value must be in [0, 1], got {result_causal['pvalue']}"
assert result_causal["mse_unrestricted"] <= result_causal["mse_restricted"] + 0.5, \
    "Unrestricted model MSE should generally be <= restricted (within tolerance)"
assert result_causal["test_stat"] >= -1.0, "Test stat should not be extremely negative"

print("\nExercise 1 assertions passed.")
print("Note: Permutation test has variance — for larger n_permutations,")
print("F05 should be more consistently significant than F10.")


# =============================================================================
# Exercise 2: Purged CV Splitter Compatible with sklearn cross_val_score
# =============================================================================
print("\n" + "=" * 70)
print("EXERCISE 2: Sklearn-Compatible Purged Cross-Validation Splitter")
print("=" * 70)

"""
Task:
    Build PurgedKFold — a class compatible with sklearn's cross_val_score
    that implements purged walk-forward cross-validation.

    Requirements:
    1. Inherit from sklearn BaseEstimator (or just implement the CV protocol)
    2. Implement split(X, y=None, groups=None) that yields (train, test) index arrays
    3. Implement get_n_splits(X=None, y=None, groups=None) -> int
    4. The groups parameter passes label_ends (end index of each observation's label span)
    5. When groups is provided, purge training observations that overlap with test

    Usage must be compatible with:
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(estimator, X, y, cv=PurgedKFold(...), groups=label_ends)

    Why this matters: sklearn's built-in TimeSeriesSplit does not support purging.
    This class makes de Prado's purged CV drop-in compatible with any sklearn workflow.
"""


class PurgedKFold:
    """
    Sklearn-compatible purged walk-forward cross-validator.

    Implements the de Prado (2018) purging and embargo protocol
    as an sklearn cross-validation splitter.

    Parameters
    ----------
    n_splits : int
        Number of walk-forward folds.
    embargo_pct : float
        Embargo size as fraction of total observations (default 0.01).

    Notes
    -----
    Pass label_ends as the groups parameter to cross_val_score.
    label_ends[i] = index of the last time step used to form observation i's label.
    For point-in-time labels: label_ends[i] = i.
    For rolling 12-month labels: label_ends[i] = i + 11.
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(
        self,
        X,
        y=None,
        groups=None,
    ):
        """
        Generate (train_indices, test_indices) pairs.

        Parameters
        ----------
        X : array-like, shape (n_samples, ...)
        y : ignored
        groups : array-like of int, optional
            groups[i] = label end index for observation i.
            Used for purging. If None: no purging (standard walk-forward).

        Yields
        ------
        train_idx : np.ndarray of int
        test_idx  : np.ndarray of int
        """
        n = len(X) if not isinstance(X, pd.DataFrame) else len(X)
        test_size = max(1, n // (self.n_splits + 1))
        embargo_size = max(1, int(n * self.embargo_pct))

        first_test_start = n - test_size * self.n_splits

        for fold in range(self.n_splits):
            test_start = first_test_start + fold * test_size
            test_end = min(test_start + test_size, n)
            test_idx = np.arange(test_start, test_end)

            train_candidates = np.arange(0, test_start)

            if groups is not None:
                # Purge: remove training obs whose label end >= test_start
                label_ends_arr = np.asarray(groups)
                purge_mask = label_ends_arr[train_candidates] < test_start
                train_candidates = train_candidates[purge_mask]

            # Embargo: remove the last embargo_size observations before test
            embargo_start = max(0, test_start - embargo_size)
            embargo_zone = np.arange(embargo_start, test_start)
            train_idx = train_candidates[~np.isin(train_candidates, embargo_zone)]

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splits."""
        return self.n_splits


# --- Run Exercise 2 ---
print("\nTesting PurgedKFold with sklearn cross_val_score:")

# Prepare feature matrix and target
X_test = FEATURES.values
y_test = TARGET.values

# Point-in-time labels (no overlap): label_ends[i] = i
label_ends_pit = np.arange(len(X_test))

# Rolling 12-month labels (overlap): label_ends[i] = min(i + 11, len - 1)
label_ends_roll = np.minimum(np.arange(len(X_test)) + 11, len(X_test) - 1)

# Test with point-in-time labels
purged_cv = PurgedKFold(n_splits=5, embargo_pct=0.01)

# Verify split structure
print("\nVerifying purged split structure (point-in-time labels):")
for fold, (train_idx, test_idx) in enumerate(purged_cv.split(X_test, groups=label_ends_pit)):
    assert len(train_idx) > 0, f"Fold {fold}: empty training set"
    assert len(test_idx) > 0, f"Fold {fold}: empty test set"
    assert train_idx.max() < test_idx.min(), \
        f"Fold {fold}: training indices must all precede test indices"
    print(f"  Fold {fold+1}: train [{train_idx[0]}, {train_idx[-1]}] ({len(train_idx)} obs), "
          f"test [{test_idx[0]}, {test_idx[-1]}] ({len(test_idx)} obs)")

# Verify with rolling labels — training set should be smaller (purged)
print("\nComparing training set sizes: point-in-time vs rolling labels:")
pit_train_sizes = []
roll_train_sizes = []
for (t_pit, _), (t_roll, _) in zip(
    purged_cv.split(X_test, groups=label_ends_pit),
    purged_cv.split(X_test, groups=label_ends_roll),
):
    pit_train_sizes.append(len(t_pit))
    roll_train_sizes.append(len(t_roll))

for fold in range(purged_cv.n_splits):
    assert roll_train_sizes[fold] <= pit_train_sizes[fold], \
        (f"Fold {fold}: rolling labels should produce smaller/equal training set "
         f"due to purging ({roll_train_sizes[fold]} vs {pit_train_sizes[fold]})")
    print(f"  Fold {fold+1}: PIT train={pit_train_sizes[fold]}, "
          f"Rolling (purged) train={roll_train_sizes[fold]}")

# Test with sklearn cross_val_score
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_test)

scores_purged = cross_val_score(
    Ridge(alpha=1.0),
    X_scaled, y_test,
    cv=purged_cv,
    groups=label_ends_pit,
    scoring="neg_mean_squared_error",
)
print(f"\ncross_val_score with PurgedKFold:")
print(f"  Scores: {scores_purged.round(4)}")
print(f"  Mean CV MSE: {-scores_purged.mean():.4f}")

# Assertions
assert purged_cv.get_n_splits() == 5, "get_n_splits() must return n_splits"
assert len(scores_purged) == 5, "cross_val_score should return one score per fold"
assert all(np.isfinite(scores_purged)), "All CV scores must be finite"

print("\nExercise 2 assertions passed.")
print("PurgedKFold is compatible with sklearn cross_val_score.")


# =============================================================================
# Exercise 3: Feature Drift Monitor
# =============================================================================
print("\n" + "=" * 70)
print("EXERCISE 3: Feature Drift Monitor with PSI, KS, and Wasserstein")
print("=" * 70)

"""
Task:
    Build FeatureDriftMonitor — a class that monitors all features for
    distribution drift between a reference period and a current period.

    The monitor computes three drift statistics for each feature:
    1. PSI (Population Stability Index) — bin-based distribution comparison
    2. KS statistic (Kolmogorov-Smirnov test) — sup-norm distributional distance
    3. Wasserstein-1 distance (Earth Mover's Distance) — optimal transport distance

    And produces a single re-selection recommendation: True if any trigger fires.

    Triggers:
    - PSI > psi_threshold (default 0.20)
    - KS p-value < alpha (default 0.05)
    - Wasserstein distance > wasserstein_threshold (default: 2x reference std)

    Usage:
        monitor = FeatureDriftMonitor(psi_threshold=0.20, ks_alpha=0.05)
        monitor.fit(reference_data)   # establish reference distribution
        report = monitor.check(current_data)  # compare current vs reference
        features_to_reselect = report[report['reselect']]['feature'].tolist()
"""


class FeatureDriftMonitor:
    """
    Monitor feature distributions for drift relative to a reference period.

    Uses three complementary statistics:
    - PSI: coarse, bin-based, robust to outliers
    - KS test: formal p-value, sensitive to all distributional differences
    - Wasserstein-1: sensitive to tail shifts and heavy-tailed changes

    Parameters
    ----------
    psi_threshold : float
        Flag drift if PSI > psi_threshold (default 0.20).
    ks_alpha : float
        Flag drift if KS p-value < ks_alpha (default 0.05).
    wasserstein_multiplier : float
        Flag drift if Wasserstein > multiplier * reference_std (default 2.0).
    n_bins : int
        Number of quantile bins for PSI (default 10).
    """

    def __init__(
        self,
        psi_threshold: float = 0.20,
        ks_alpha: float = 0.05,
        wasserstein_multiplier: float = 2.0,
        n_bins: int = 10,
    ):
        self.psi_threshold = psi_threshold
        self.ks_alpha = ks_alpha
        self.wasserstein_multiplier = wasserstein_multiplier
        self.n_bins = n_bins
        self._reference = None  # pd.DataFrame of reference data
        self._ref_stds = None   # per-feature reference std for Wasserstein threshold

    def fit(self, reference_data: pd.DataFrame) -> "FeatureDriftMonitor":
        """
        Establish reference distribution from historical data.

        Parameters
        ----------
        reference_data : pd.DataFrame, shape (n_ref_samples, n_features)
            Feature values in the reference (training) period.

        Returns
        -------
        self
        """
        self._reference = reference_data.copy()
        self._ref_stds = reference_data.std()
        return self

    @staticmethod
    def _compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
        """Population Stability Index."""
        eps = 1e-10
        breakpoints = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        ref_counts, _ = np.histogram(reference, bins=breakpoints)
        cur_counts, _ = np.histogram(current, bins=breakpoints)

        ref_pct = ref_counts / (len(reference) + eps) + eps
        cur_pct = cur_counts / (len(current) + eps) + eps

        return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    def check(self, current_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compare current feature distributions to reference.

        Parameters
        ----------
        current_data : pd.DataFrame, shape (n_current_samples, n_features)
            Feature values in the current (monitoring) period.
            Must have the same columns as reference_data.

        Returns
        -------
        pd.DataFrame with columns:
            feature           : str   — feature name
            psi               : float — Population Stability Index
            psi_flag          : bool  — True if psi > psi_threshold
            ks_statistic      : float — KS test statistic
            ks_pvalue         : float — KS test p-value
            ks_flag           : bool  — True if ks_pvalue < ks_alpha
            wasserstein       : float — Wasserstein-1 distance
            wasserstein_flag  : bool  — True if wasserstein > threshold
            reselect          : bool  — True if ANY flag is True
        """
        if self._reference is None:
            raise RuntimeError("Call fit() before check().")

        common_cols = self._reference.columns.intersection(current_data.columns)
        records = []

        for col in common_cols:
            ref = self._reference[col].dropna().values
            cur = current_data[col].dropna().values

            if len(ref) < 5 or len(cur) < 5:
                continue

            # PSI
            psi = self._compute_psi(ref, cur, n_bins=self.n_bins)
            psi_flag = psi > self.psi_threshold

            # KS test
            ks_stat, ks_p = ks_2samp(ref, cur)
            ks_flag = ks_p < self.ks_alpha

            # Wasserstein-1 distance
            w_dist = wasserstein_distance(ref, cur)
            ref_std = float(self._ref_stds.get(col, 1.0))
            w_threshold = self.wasserstein_multiplier * max(ref_std, 1e-6)
            w_flag = w_dist > w_threshold

            records.append({
                "feature": col,
                "psi": psi,
                "psi_flag": psi_flag,
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_p,
                "ks_flag": ks_flag,
                "wasserstein": w_dist,
                "wasserstein_flag": w_flag,
                "reselect": psi_flag or ks_flag or w_flag,
            })

        return pd.DataFrame(records).sort_values("psi", ascending=False).reset_index(drop=True)

    def summary(self, current_data: pd.DataFrame) -> dict:
        """
        High-level summary of drift detection results.

        Returns
        -------
        dict with:
            n_features_checked : int
            n_flagged_psi      : int
            n_flagged_ks       : int
            n_flagged_wass     : int
            n_reselect         : int
            features_to_reselect : list of str
            reselect_triggered : bool
        """
        report = self.check(current_data)
        return {
            "n_features_checked": len(report),
            "n_flagged_psi": report["psi_flag"].sum(),
            "n_flagged_ks": report["ks_flag"].sum(),
            "n_flagged_wass": report["wasserstein_flag"].sum(),
            "n_reselect": report["reselect"].sum(),
            "features_to_reselect": report[report["reselect"]]["feature"].tolist(),
            "reselect_triggered": report["reselect"].any(),
        }


# --- Run Exercise 3 ---
print("\nTesting FeatureDriftMonitor:")

# Reference period: first 150 observations
# Current period with drift: last 150 observations with mean shift in some features
n_ref = 150
reference_data = FEATURES.iloc[:n_ref].copy()

# Current data: introduce drift in features F01-F03 (shift mean by 2 std)
current_data = FEATURES.iloc[n_ref:].copy()
ref_std = reference_data[["F01", "F02", "F03"]].std()
current_data_drifted = current_data.copy()
current_data_drifted[["F01", "F02", "F03"]] += 3.0 * ref_std  # significant shift

# No-drift case: same distribution as reference (bootstrap)
rng_check = np.random.default_rng(99)
current_data_nodrift = reference_data.sample(frac=0.8, replace=True,
                                              random_state=99).reset_index(drop=True)
current_data_nodrift.index = FEATURES.index[n_ref: n_ref + len(current_data_nodrift)]

# Create and fit monitor
monitor = FeatureDriftMonitor(psi_threshold=0.20, ks_alpha=0.05, wasserstein_multiplier=2.0)
monitor.fit(reference_data)

# Check drifted data
report_drift = monitor.check(current_data_drifted)
summary_drift = monitor.summary(current_data_drifted)

print("\nDrifted data (F01-F03 shifted by 3 std):")
print(report_drift[["feature", "psi", "psi_flag", "ks_pvalue", "ks_flag",
                     "wasserstein", "wasserstein_flag", "reselect"]].to_string(index=False))

print(f"\nSummary (drifted):")
for k, v in summary_drift.items():
    print(f"  {k}: {v}")

# Check no-drift data
report_nodrift = monitor.check(current_data_nodrift)
summary_nodrift = monitor.summary(current_data_nodrift)

print(f"\nNo-drift data summary:")
print(f"  Features flagged for re-selection: {summary_nodrift['n_reselect']}")
print(f"  Re-selection triggered: {summary_nodrift['reselect_triggered']}")

# Assertions
assert isinstance(monitor._reference, pd.DataFrame), \
    "fit() must store reference data in self._reference"
assert isinstance(report_drift, pd.DataFrame), "check() must return a DataFrame"
assert "reselect" in report_drift.columns, "report must have 'reselect' column"
assert "feature" in report_drift.columns, "report must have 'feature' column"

# Drifted features (F01-F03) should be flagged
drifted_flagged = set(report_drift[report_drift["reselect"]]["feature"].tolist())
drifted_target = {"F01", "F02", "F03"}
assert len(drifted_flagged & drifted_target) >= 2, \
    (f"At least 2 of F01, F02, F03 should be flagged for drift. "
     f"Got {drifted_flagged}. Check PSI and KS implementations.")

assert summary_drift["reselect_triggered"], \
    "Re-selection should be triggered for drifted data"

assert len(report_drift) == FEATURES.shape[1], \
    f"Report should have one row per feature ({FEATURES.shape[1]}), got {len(report_drift)}"

# PSI values should be larger for drifted features
psi_drifted = report_drift[report_drift["feature"].isin(["F01", "F02", "F03"])]["psi"].mean()
psi_stable = report_drift[~report_drift["feature"].isin(["F01", "F02", "F03"])]["psi"].mean()
assert psi_drifted > psi_stable, \
    (f"Drifted features should have higher PSI than stable ones. "
     f"Got psi_drifted={psi_drifted:.3f}, psi_stable={psi_stable:.3f}")

print("\nExercise 3 assertions passed.")
print("FeatureDriftMonitor correctly detects distributional drift in F01-F03.")


# =============================================================================
# Final Summary
# =============================================================================
print("\n" + "=" * 70)
print("ALL EXERCISES COMPLETE")
print("=" * 70)
print("""
Module 7 Self-Check Summary:

Exercise 1 — Nonlinear Granger Causality:
  Implemented kernel_granger_causality() using KernelRidge + permutation test.
  Key learning: nonlinear Granger causality detects relationships the linear
  F-test misses (e.g., quadratic lag effects, regime-switching dynamics).

Exercise 2 — Purged CV Splitter:
  Implemented PurgedKFold compatible with sklearn cross_val_score.
  Key learning: purging removes training observations whose label windows
  overlap the test period — critical for rolling-window features.

Exercise 3 — Feature Drift Monitor:
  Implemented FeatureDriftMonitor with PSI, KS test, and Wasserstein distance.
  Key learning: PSI > 0.20 is a standard industry trigger for model review;
  KS test provides formal p-values; Wasserstein distance is sensitive to tails.

Next steps:
  - Apply kernel Granger to financial time series (Module 9: Causal Methods)
  - Integrate PurgedKFold into walk-forward backtesting pipelines
  - Deploy FeatureDriftMonitor in a production monitoring loop (Module 11)
""")
