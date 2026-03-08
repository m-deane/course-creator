"""
Module 11: Production Feature Selection Pipelines — Self-Check Exercises

Three exercises covering the core production engineering patterns:

  Exercise 1: Universal selector transformer — wrap any selection method
               in a sklearn-compatible fit/transform interface
  Exercise 2: PSI drift monitor — compute PSI for all features and
               emit structured alerts when thresholds are crossed
  Exercise 3: MLflow logging wrapper — log any selection experiment with
               full provenance in a single function call

Run with:
    python 01_production_exercises.py

Each exercise has:
  - A function stub to implement
  - A test function that checks your implementation
  - Clear error messages telling you exactly what to fix
"""

import numpy as np
import pandas as pd
import json
import hashlib
import datetime
import pathlib
import warnings
warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils import resample
from sklearn.utils.validation import check_is_fitted
from sklearn.datasets import fetch_california_housing, make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp


# =============================================================================
# Shared data setup — used by all exercises
# =============================================================================

def load_data():
    """Load and split California housing as a binary classification problem."""
    housing = fetch_california_housing(as_frame=True)
    X = housing.frame.drop(columns=['MedHouseVal'])
    y = (housing.frame['MedHouseVal'] > housing.frame['MedHouseVal'].median()).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


# =============================================================================
# Exercise 1: Universal Selector Transformer
# =============================================================================

class UniversalSelectorTransformer(BaseEstimator, TransformerMixin):
    """
    Wrap any sklearn selector in a Pipeline-compatible fit/transform interface.

    This transformer generalises the pattern from Notebook 01 to work
    with ANY sklearn selector that exposes fit(X, y) and get_support().

    Parameters
    ----------
    selector : sklearn selector
        Any object implementing fit(X, y) and get_support().
        Examples: SelectKBest, SelectFromModel, RFE, VarianceThreshold.
    verbose : bool
        If True, print the selected feature names after fitting.

    Attributes (set by fit)
    -----------------------
    support_ : ndarray of bool, shape (n_features,)
        Boolean mask of selected features.
    feature_names_in_ : ndarray of str, shape (n_features,)
        Feature names from training data (or generated names for arrays).
    n_features_in_ : int
        Total number of input features.
    n_features_selected_ : int
        Number of features selected.

    Methods
    -------
    fit(X, y)
    transform(X)
    get_feature_names_out()
    get_support()          ← returns self.support_ (for Pipeline compatibility)
    """

    def __init__(self, selector, verbose: bool = False):
        self.selector = selector
        self.verbose  = verbose

    def fit(self, X, y):
        """
        Fit the selector on training data and record the selection mask.

        Steps:
          1. Store feature names:
             - If X is a DataFrame, use X.columns
             - Otherwise, generate names: ['x0', 'x1', ..., 'xN']
          2. Clone the selector to avoid state sharing across fit calls
          3. Fit the cloned selector on X and y
          4. Store support_ (boolean mask from get_support())
          5. Store n_features_in_ and n_features_selected_
          6. If verbose=True, print selected feature names
          7. Return self

        Example
        -------
        >>> sel = UniversalSelectorTransformer(SelectKBest(k=3), verbose=True)
        >>> sel.fit(X_train, y_train)
        Selected 3 / 8 features: ['MedInc', 'Latitude', 'Longitude']
        """
        # TODO: Implement
        raise NotImplementedError(
            "Implement fit:\n"
            "  1. Store feature names in self.feature_names_in_\n"
            "  2. Clone self.selector and fit it on X, y\n"
            "  3. Store self.support_ from get_support()\n"
            "  4. Store self.n_features_in_ and self.n_features_selected_\n"
            "  5. If self.verbose, print selected features\n"
            "  6. Return self"
        )

    def transform(self, X):
        """
        Apply the fitted selection mask to new data.

        - If X is a DataFrame: return X.iloc[:, self.support_]
        - If X is an ndarray: return X[:, self.support_]
        - Call check_is_fitted(self, 'support_') at the start for a clear error
          if transform is called before fit.

        Example
        -------
        >>> X_selected = sel.transform(X_test)
        >>> X_selected.shape
        (412, 3)
        """
        # TODO: Implement
        raise NotImplementedError(
            "Implement transform:\n"
            "  1. check_is_fitted(self, 'support_')\n"
            "  2. Return X.iloc[:, self.support_] for DataFrame\n"
            "     or X[:, self.support_] for ndarray"
        )

    def get_feature_names_out(self, input_features=None):
        """
        Return the names of selected features.

        Returns
        -------
        ndarray of str
            self.feature_names_in_[self.support_]
        """
        # TODO: Implement
        raise NotImplementedError(
            "Implement get_feature_names_out:\n"
            "  1. check_is_fitted(self, 'support_')\n"
            "  2. Return self.feature_names_in_[self.support_]"
        )

    def get_support(self):
        """Return the boolean selection mask (for Pipeline compatibility)."""
        check_is_fitted(self, 'support_')
        return self.support_


def test_exercise_1():
    """
    Test the UniversalSelectorTransformer implementation.

    Checks:
      - Works with SelectKBest (filter method)
      - Works with SelectFromModel (embedded method)
      - Works with both DataFrame and ndarray inputs
      - check_is_fitted prevents transform before fit
      - get_feature_names_out returns correct names
      - n_features_selected_ is accurate
      - Pipeline integration works end-to-end
    """
    print("Testing Exercise 1: UniversalSelectorTransformer")
    print("-" * 50)

    X_train, X_test, y_train, y_test = load_data()
    n_original = X_train.shape[1]

    # --- Test 1a: SelectKBest wrapper ---
    kbest_transformer = UniversalSelectorTransformer(
        selector=SelectKBest(score_func=f_classif, k=5),
        verbose=True,
    )
    kbest_transformer.fit(X_train, y_train)

    assert hasattr(kbest_transformer, 'support_'), \
        "Missing support_ attribute after fit. Set self.support_ in fit()."
    assert kbest_transformer.support_.dtype == bool, \
        f"support_ must be a boolean array, got dtype={kbest_transformer.support_.dtype}"
    assert kbest_transformer.support_.sum() == 5, \
        f"Expected 5 features selected, got {kbest_transformer.support_.sum()}"
    assert kbest_transformer.n_features_in_       == n_original, \
        f"n_features_in_ should be {n_original}, got {kbest_transformer.n_features_in_}"
    assert kbest_transformer.n_features_selected_ == 5, \
        f"n_features_selected_ should be 5, got {kbest_transformer.n_features_selected_}"

    # --- Test 1b: transform output shape ---
    X_selected_df  = kbest_transformer.transform(X_train)
    X_selected_arr = kbest_transformer.transform(X_train.values)
    assert X_selected_df.shape  == (len(X_train), 5), \
        f"DataFrame transform: expected shape ({len(X_train)}, 5), got {X_selected_df.shape}"
    assert X_selected_arr.shape == (len(X_train), 5), \
        f"ndarray transform: expected shape ({len(X_train)}, 5), got {X_selected_arr.shape}"

    # --- Test 1c: feature names ---
    selected_names = kbest_transformer.get_feature_names_out()
    assert len(selected_names) == 5, \
        f"get_feature_names_out should return 5 names, got {len(selected_names)}"
    assert all(name in X_train.columns for name in selected_names), \
        "get_feature_names_out returned names not in original columns"

    # --- Test 1d: check_is_fitted prevents premature transform ---
    fresh_transformer = UniversalSelectorTransformer(SelectKBest(k=3))
    try:
        fresh_transformer.transform(X_test)
        # If we reach here, check_is_fitted was not called
        assert False, (
            "transform should raise NotFittedError before fit is called. "
            "Add check_is_fitted(self, 'support_') at the start of transform()."
        )
    except Exception as e:
        # NotFittedError or similar — expected
        assert 'fitted' in str(e).lower() or 'not' in str(e).lower(), \
            f"Expected a NotFittedError-like message, got: {e}"

    # --- Test 1e: works with SelectFromModel ---
    rf_transformer = UniversalSelectorTransformer(
        selector=SelectFromModel(
            estimator=RandomForestClassifier(n_estimators=50, random_state=42),
            threshold='mean',
        )
    )
    rf_transformer.fit(X_train, y_train)
    X_rf_selected = rf_transformer.transform(X_test)
    assert X_rf_selected.shape[0] == len(X_test), \
        f"Number of rows must be preserved. Expected {len(X_test)}, got {X_rf_selected.shape[0]}"
    assert 1 <= X_rf_selected.shape[1] <= n_original, \
        f"Number of columns must be between 1 and {n_original}"

    # --- Test 1f: end-to-end Pipeline integration ---
    pipeline = Pipeline([
        ('scaler',   StandardScaler()),
        ('selector', UniversalSelectorTransformer(SelectKBest(k=5))),
        ('model',    GradientBoostingClassifier(n_estimators=50, random_state=42)),
    ])

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
    assert scores.mean() > 0.8, \
        f"Pipeline CV AUC should be > 0.80, got {scores.mean():.4f}. Check that transform returns correct shape."

    print("  All tests passed!")
    print(f"  SelectKBest: {kbest_transformer.n_features_selected_} features selected")
    print(f"  SelectFromModel: {rf_transformer.n_features_selected_} features selected")
    print(f"  Pipeline CV AUC: {scores.mean():.4f}")
    print()


# =============================================================================
# Exercise 2: PSI Drift Monitor
# =============================================================================

class PSIDriftMonitor:
    """
    Monitor feature distribution drift using Population Stability Index.

    Computes PSI for each feature against a reference distribution and
    emits structured alerts when per-feature or aggregate thresholds are crossed.

    Parameters
    ----------
    n_bins : int
        Number of bins for PSI computation. Bins are percentile-based,
        defined on the reference distribution.
    psi_warning_threshold : float
        PSI > this triggers a WARNING alert (default: 0.10).
    psi_critical_threshold : float
        PSI > this triggers a CRITICAL alert (default: 0.20).
    fraction_warning : float
        Trigger aggregate WARNING if > this fraction of features are in WARNING or CRITICAL.
    fraction_critical : float
        Trigger aggregate CRITICAL if > this fraction of features are CRITICAL.

    Methods
    -------
    fit(reference_df)
        Learn the reference distribution (compute and store bin edges per feature).
    check(current_df)
        Compute PSI against reference for all features. Returns a report dict.
    """

    def __init__(
        self,
        n_bins: int = 10,
        psi_warning_threshold: float = 0.10,
        psi_critical_threshold: float = 0.20,
        fraction_warning: float = 0.20,
        fraction_critical: float = 0.20,
    ):
        self.n_bins                  = n_bins
        self.psi_warning_threshold   = psi_warning_threshold
        self.psi_critical_threshold  = psi_critical_threshold
        self.fraction_warning        = fraction_warning
        self.fraction_critical       = fraction_critical

    def fit(self, reference_df: pd.DataFrame):
        """
        Store the reference distribution bin edges for each feature.

        For each column in reference_df, compute percentile-based bin edges:
          bin_edges = np.percentile(col_values, np.linspace(0, 100, n_bins + 1))
          bin_edges[0]  -= 1e-10   # include minimum
          bin_edges[-1] += 1e-10   # include maximum

        Store in self.bin_edges_: dict mapping column name → bin_edges array.
        Also store self.feature_names_: list of column names in fit order.

        Steps:
          1. Validate that reference_df is a DataFrame
          2. For each column: compute and store bin edges
          3. Return self

        Example
        -------
        >>> monitor = PSIDriftMonitor()
        >>> monitor.fit(X_train)
        >>> monitor.bin_edges_['MedInc'].shape
        (11,)
        """
        # TODO: Implement
        raise NotImplementedError(
            "Implement fit:\n"
            "  1. Validate reference_df is a DataFrame\n"
            "  2. For each column, compute percentile-based bin edges:\n"
            "       edges = np.percentile(col, np.linspace(0, 100, n_bins+1))\n"
            "       edges[0] -= 1e-10; edges[-1] += 1e-10\n"
            "  3. Store in self.bin_edges_ = {col: edges, ...}\n"
            "  4. Store self.feature_names_ = list(reference_df.columns)\n"
            "  5. Return self"
        )

    def _compute_psi_one(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bin_edges: np.ndarray,
        eps: float = 1e-6,
    ) -> float:
        """
        Compute PSI for one feature using pre-computed bin edges.

        Algorithm:
          1. Histogram both reference and current using bin_edges
          2. Convert counts to proportions
          3. Add eps to avoid log(0), renormalise
          4. Return sum((cur - ref) * log(cur / ref))

        This helper is called by check() for each feature.
        """
        # TODO: Implement
        raise NotImplementedError(
            "Implement _compute_psi_one:\n"
            "  ref_counts = np.histogram(reference, bins=bin_edges)[0]\n"
            "  cur_counts = np.histogram(current,   bins=bin_edges)[0]\n"
            "  ref_props = (ref_counts / ref_counts.sum()) + eps\n"
            "  cur_props = (cur_counts / cur_counts.sum()) + eps\n"
            "  ref_props /= ref_props.sum()\n"
            "  cur_props /= cur_props.sum()\n"
            "  return float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))"
        )

    def check(self, current_df: pd.DataFrame) -> dict:
        """
        Compute PSI for all features and return a structured report.

        Returns
        -------
        dict with keys:
          'feature_psi'       : dict mapping feature name → PSI value
          'feature_status'    : dict mapping feature name → 'ok'/'warning'/'critical'
          'n_ok'              : int
          'n_warning'         : int
          'n_critical'        : int
          'aggregate_status'  : 'ok' | 'warning' | 'critical'
          'trigger_reselect'  : bool (True if aggregate CRITICAL threshold crossed)
          'alerts'            : list of str — human-readable alert messages

        Aggregate status logic:
          - 'critical' if n_critical / n_features >= fraction_critical
          - 'warning'  if (n_warning + n_critical) / n_features >= fraction_warning
          - 'ok'       otherwise

        trigger_reselect is True when aggregate_status == 'critical'.

        Example
        -------
        >>> report = monitor.check(X_test)
        >>> report['aggregate_status']
        'ok'
        >>> report['trigger_reselect']
        False
        """
        check_is_fitted(self, 'bin_edges_')

        # Validate that current_df has the same features as reference
        missing = set(self.feature_names_) - set(current_df.columns)
        if missing:
            raise ValueError(
                f"current_df is missing features from reference: {missing}"
            )

        # TODO: Implement
        raise NotImplementedError(
            "Implement check:\n"
            "  1. For each feature, compute PSI using _compute_psi_one\n"
            "  2. Classify each as 'ok', 'warning', or 'critical'\n"
            "  3. Count n_ok, n_warning, n_critical\n"
            "  4. Compute aggregate_status based on fraction thresholds\n"
            "  5. Build alerts list for features in warning/critical\n"
            "  6. Set trigger_reselect = (aggregate_status == 'critical')\n"
            "  7. Return the report dict"
        )


def test_exercise_2():
    """
    Test the PSIDriftMonitor implementation.

    Checks:
      - fit stores bin edges for all features
      - check returns correct structure
      - stable data → all 'ok', no trigger
      - heavily drifted data → 'critical' features, trigger fires
      - PSI values are non-negative
      - alerts list contains strings for drifted features
    """
    print("Testing Exercise 2: PSIDriftMonitor")
    print("-" * 50)

    X_train, X_test, y_train, y_test = load_data()

    monitor = PSIDriftMonitor(
        n_bins=10,
        psi_warning_threshold=0.10,
        psi_critical_threshold=0.20,
        fraction_warning=0.25,
        fraction_critical=0.25,
    )

    # --- Test 2a: fit ---
    monitor.fit(X_train)

    assert hasattr(monitor, 'bin_edges_'), \
        "Missing bin_edges_ after fit. Create self.bin_edges_ = {} in fit."
    assert hasattr(monitor, 'feature_names_'), \
        "Missing feature_names_ after fit."
    assert set(monitor.bin_edges_.keys()) == set(X_train.columns), \
        "bin_edges_ must have one entry per feature column."
    for col, edges in monitor.bin_edges_.items():
        assert len(edges) == 11, \
            f"Expected 11 bin edges (10 bins) for '{col}', got {len(edges)}"

    # --- Test 2b: check on undrifted data (X_test same distribution as X_train) ---
    report_stable = monitor.check(X_test)

    required_keys = [
        'feature_psi', 'feature_status', 'n_ok', 'n_warning', 'n_critical',
        'aggregate_status', 'trigger_reselect', 'alerts',
    ]
    for key in required_keys:
        assert key in report_stable, \
            f"Missing key '{key}' in report dict. check() must return all required keys."

    # PSI values must be non-negative
    for feat, psi_val in report_stable['feature_psi'].items():
        assert psi_val >= 0, f"PSI for '{feat}' is negative ({psi_val:.4f}). PSI must be >= 0."

    # Undrifted data should have low PSI
    assert report_stable['aggregate_status'] in ('ok', 'warning'), \
        f"X_test has same distribution as X_train — expected 'ok' or 'warning', " \
        f"got '{report_stable['aggregate_status']}'"

    assert isinstance(report_stable['trigger_reselect'], bool), \
        "trigger_reselect must be a bool"

    assert isinstance(report_stable['alerts'], list), \
        "alerts must be a list"

    # --- Test 2c: check on heavily drifted data ---
    # Shift all features by 5 standard deviations — should trigger CRITICAL
    X_drifted = X_test.copy()
    for col in X_drifted.columns:
        X_drifted[col] = X_drifted[col] + 5 * X_train[col].std()

    report_drifted = monitor.check(X_drifted)

    n_critical = report_drifted['n_critical']
    assert n_critical >= 1, \
        f"Heavily drifted data (5-sigma shift) should have at least 1 critical feature. " \
        f"Got n_critical={n_critical}."

    # At least one alert for critical features
    assert len(report_drifted['alerts']) >= 1, \
        "Expected at least 1 alert message for drifted data."

    # All alert entries must be strings
    for alert in report_drifted['alerts']:
        assert isinstance(alert, str), f"alerts must contain strings, got {type(alert)}"

    print("  All tests passed!")
    print(f"  Stable data:  n_critical={report_stable['n_critical']}, "
          f"status={report_stable['aggregate_status']}")
    print(f"  Drifted data: n_critical={report_drifted['n_critical']}, "
          f"status={report_drifted['aggregate_status']}, "
          f"trigger={report_drifted['trigger_reselect']}")
    print()


# =============================================================================
# Exercise 3: MLflow Logging Wrapper
# =============================================================================

def log_selection_experiment(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    selector,
    selector_name: str,
    selector_params: dict,
    experiment_name: str = 'feature-selection-production',
    tags: dict | None = None,
    output_dir: str = 'exercise_mlflow_outputs',
) -> dict:
    """
    Log a complete feature selection experiment to MLflow.

    This function wraps the entire selection process in a single MLflow run
    with full provenance logging. It should be usable as a drop-in wrapper
    around any selection method without modifying the selection logic itself.

    Parameters
    ----------
    X_train, y_train : training data
    X_val, y_val     : validation data (used only for metric evaluation, NOT selection)
    selector         : sklearn selector (implements fit(X, y) and get_support())
    selector_name    : human-readable name (e.g. 'stability_selection')
    selector_params  : dict of selector hyperparameters to log
    experiment_name  : MLflow experiment name
    tags             : additional MLflow tags (e.g. {'triggered_by': 'psi_drift'})
    output_dir       : local directory for temporary artefact files

    Returns
    -------
    dict with keys:
        'run_id'            : str (MLflow run ID, first 8 chars)
        'selected_features' : list of str
        'n_selected'        : int
        'val_roc_auc'       : float
        'dataset_hash'      : str (SHA-256 of X_train)

    Implementation steps:
    ---------------------
    1. Set the MLflow tracking URI and experiment:
          mlflow.set_tracking_uri('sqlite:///mlflow_tracking.db')
          mlflow.set_experiment(experiment_name)

    2. Compute dataset_hash = SHA-256 of X_train (use hash_dataframe below)

    3. Open a new MLflow run: `with mlflow.start_run(tags=tags or {}) as run:`

    4. Inside the run, log the following PARAMETERS:
          selector_name
          dataset_hash
          n_features_total     (X_train.shape[1])
          n_train              (len(X_train))
          selection_date       (today as ISO-8601 string)
          For each k, v in selector_params: log as 'selector.{k}'

    5. Standardise X_train and X_val using StandardScaler (fit on X_train only)

    6. Fit the selector on scaled X_train

    7. Extract:
          support        = selector.get_support()
          selected_names = X_train.columns[support].tolist()
          n_selected     = len(selected_names)

    8. Log the following METRICS:
          n_features_selected
          selection_ratio     (n_selected / X_train.shape[1])
          val_roc_auc         (train model on scaled X_train[support], score on scaled X_val[support])

    9. Save selected_features as JSON and log as an MLflow ARTEFACT:
          - File: {output_dir}/selected_features.json
          - Content: {'selected_features': selected_names, 'n_selected': n_selected,
                      'selector': selector_name, 'dataset_hash': dataset_hash}
          - mlflow.log_artifact(str(features_file))

    10. Return the result dict

    Notes:
    - Train a GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
      on the selected features to compute val_roc_auc.
    - Use roc_auc_score(y_val, model.predict_proba(X_val_selected)[:, 1]) for the AUC.
    - The mlflow import is already at the top — do not re-import it inside this function.

    Example
    -------
    >>> result = log_selection_experiment(
    ...     X_train, y_train, X_val, y_val,
    ...     selector=SelectKBest(k=5),
    ...     selector_name='selectkbest',
    ...     selector_params={'k': 5},
    ... )
    >>> result['n_selected']
    5
    >>> 0.80 < result['val_roc_auc'] < 1.0
    True
    """
    # TODO: Implement
    raise NotImplementedError(
        "Implement log_selection_experiment following the 10 steps in the docstring."
    )


def hash_dataframe(df: pd.DataFrame) -> str:
    """Deterministic SHA-256 hash of a DataFrame (helper for Exercise 3)."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()[:16]


def test_exercise_3():
    """
    Test the log_selection_experiment implementation.

    Checks:
      - Return type and required keys
      - n_selected matches selector k parameter
      - val_roc_auc is a valid AUC score (between 0 and 1)
      - MLflow params are logged (selector_name, n_features_total)
      - MLflow metrics are logged (n_features_selected, val_roc_auc)
      - Artefact file exists and contains correct JSON structure
      - Two runs with different selectors produce different run IDs
    """
    try:
        import mlflow
    except ImportError:
        print("  MLflow not installed. Install with: pip install mlflow")
        print("  Skipping Exercise 3 tests.")
        return

    print("Testing Exercise 3: log_selection_experiment")
    print("-" * 50)

    X_train, X_test, y_train, y_test = load_data()
    X_train_small = X_train.iloc[:500]  # faster for testing
    y_train_small = y_train.iloc[:500]
    X_val = X_test.iloc[:200]
    y_val = y_test.iloc[:200]

    # --- Test 3a: Basic call ---
    result = log_selection_experiment(
        X_train_small, y_train_small.values, X_val, y_val.values,
        selector=SelectKBest(score_func=f_classif, k=4),
        selector_name='selectkbest_k4',
        selector_params={'score_func': 'f_classif', 'k': 4},
        tags={'test': 'exercise_3'},
    )

    required_keys = ['run_id', 'selected_features', 'n_selected', 'val_roc_auc', 'dataset_hash']
    for key in required_keys:
        assert key in result, \
            f"Missing key '{key}' in return dict. Return dict must have: {required_keys}"

    assert result['n_selected'] == 4, \
        f"Expected 4 features (k=4), got {result['n_selected']}"

    assert 0.5 < result['val_roc_auc'] < 1.0, \
        f"val_roc_auc must be between 0.5 and 1.0, got {result['val_roc_auc']:.4f}"

    assert isinstance(result['run_id'], str) and len(result['run_id']) >= 8, \
        "run_id must be a string of at least 8 characters"

    assert isinstance(result['dataset_hash'], str) and len(result['dataset_hash']) > 0, \
        "dataset_hash must be a non-empty string"

    assert len(result['selected_features']) == 4, \
        f"selected_features must contain 4 names, got {len(result['selected_features'])}"

    # --- Test 3b: MLflow records are queryable ---
    mlflow.set_tracking_uri('sqlite:///mlflow_tracking.db')
    client = mlflow.tracking.MlflowClient()

    # Find the run we just created
    experiment = client.get_experiment_by_name('feature-selection-production')
    if experiment is not None:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=['attribute.start_time DESC'],
            max_results=1,
        )
        if runs:
            latest_run = runs[0]
            params = latest_run.data.params
            metrics = latest_run.data.metrics

            assert 'selector_name' in params, \
                "MLflow params must include 'selector_name'. Use mlflow.log_param('selector_name', ...)."
            assert 'n_features_total' in params, \
                "MLflow params must include 'n_features_total'."
            assert 'n_features_selected' in metrics, \
                "MLflow metrics must include 'n_features_selected'. Use mlflow.log_metric(...)."
            assert 'val_roc_auc' in metrics, \
                "MLflow metrics must include 'val_roc_auc'."
            assert abs(metrics['val_roc_auc'] - result['val_roc_auc']) < 1e-4, \
                f"val_roc_auc in MLflow ({metrics['val_roc_auc']:.4f}) must match " \
                f"return dict ({result['val_roc_auc']:.4f})"

    # --- Test 3c: Artefact file ---
    output_dir = pathlib.Path('exercise_mlflow_outputs')
    artefact_file = output_dir / 'selected_features.json'
    assert artefact_file.exists(), \
        f"Expected artefact file at {artefact_file}. Create it with features_file.write_text(json.dumps(...))"

    with open(artefact_file) as f:
        artefact = json.load(f)

    for key in ['selected_features', 'n_selected', 'selector', 'dataset_hash']:
        assert key in artefact, \
            f"Artefact JSON must contain '{key}'. Got keys: {list(artefact.keys())}"

    # --- Test 3d: Different selectors produce different run IDs ---
    result2 = log_selection_experiment(
        X_train_small, y_train_small.values, X_val, y_val.values,
        selector=SelectKBest(score_func=f_classif, k=3),
        selector_name='selectkbest_k3',
        selector_params={'score_func': 'f_classif', 'k': 3},
    )
    assert result['run_id'] != result2['run_id'], \
        "Two separate runs must have different run IDs"

    print("  All tests passed!")
    print(f"  Run 1: {result['run_id']} — {result['n_selected']} features, "
          f"AUC={result['val_roc_auc']:.4f}")
    print(f"  Run 2: {result2['run_id']} — {result2['n_selected']} features, "
          f"AUC={result2['val_roc_auc']:.4f}")
    print()


# =============================================================================
# Main runner
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Module 11: Production Feature Selection — Self-Check Exercises")
    print("=" * 60)
    print()

    exercises = [
        ("Exercise 1: Universal Selector Transformer", test_exercise_1),
        ("Exercise 2: PSI Drift Monitor",              test_exercise_2),
        ("Exercise 3: MLflow Logging Wrapper",         test_exercise_3),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for name, test_fn in exercises:
        try:
            test_fn()
            passed += 1
        except NotImplementedError as e:
            print(f"  NOT IMPLEMENTED: {e}")
            print(f"  → Implement the function and re-run.")
            print()
            skipped += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            print()
            failed += 1
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            print()
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed | {failed} failed | {skipped} not implemented")
    if passed == 3:
        print("All exercises complete! You have mastered production feature selection.")
    elif passed > 0:
        print(f"Good progress! {3 - passed} exercise(s) remaining.")
    else:
        print("Start with Exercise 1 and work through in order.")
    print("=" * 60)
