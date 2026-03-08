"""
Module 9: Causal Feature Selection — Self-Check Exercises

Three self-check exercises:
  1. Implement a simple ICP procedure from scratch using linear regression
     and environment splitting.
  2. Run FCI (handling latent confounders) and compare with PC on the same
     dataset.
  3. Build a distribution-shift test bench comparing causal vs predictive
     feature sets.

Run all exercises:
    python 01_causal_exercises.py

Run a single exercise:
    python 01_causal_exercises.py --exercise 1

References:
  - Peters, Bühlmann, Meinshausen (2016). JRSS-B 78(5), 947–1012.
  - Spirtes, Glymour, Scheines (2000). Causation, Prediction, and Search.
  - Pearl (2009). Causality (2nd ed.).
"""

import argparse
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


# ══════════════════════════════════════════════════════════════════════════════
# Shared data generation utilities
# ══════════════════════════════════════════════════════════════════════════════

def make_icp_dataset(n_per_env: int = 300,
                     n_train_envs: int = 4,
                     seed: int = 42) -> dict:
    """
    Generate a dataset with known causal structure for ICP testing.

    True causal model: Y = 2*X1 + 1.5*X2 + noise

    X3, X4 are spuriously correlated via a hidden confounder H.
    H has different strength in each environment, making the correlation
    between X3, X4 and Y environment-specific.

    Parameters
    ----------
    n_per_env : int
        Samples per training environment.
    n_train_envs : int
        Number of training environments.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys:
        'X_train', 'y_train', 'env_labels'  — training data
        'X_test', 'y_test'                  — test data (shifted environment)
        'feature_names', 'true_causal'
    """
    rng = np.random.RandomState(seed)

    # Training environments: alternating hidden confounder strength
    env_h_strengths = [2.0, -1.5, 1.0, -2.0][:n_train_envs]
    X_parts, y_parts, env_parts = [], [], []

    for e, h_str in enumerate(env_h_strengths):
        H = h_str * rng.randn(n_per_env)
        x1 = rng.randn(n_per_env)
        x2 = rng.randn(n_per_env)
        x3 = H + 0.4 * rng.randn(n_per_env)          # Spurious
        x4 = 0.8 * H + 0.3 * rng.randn(n_per_env)    # Spurious
        x5 = rng.randn(n_per_env)                      # Noise
        y = 2.0 * x1 + 1.5 * x2 + 0.3 * H + 0.5 * rng.randn(n_per_env)
        X_parts.append(np.column_stack([x1, x2, x3, x4, x5]))
        y_parts.append(y)
        env_parts.append(np.full(n_per_env, e))

    X_train = np.vstack(X_parts)
    y_train = np.concatenate(y_parts)
    env_labels = np.concatenate(env_parts)

    # Test environment: reversed hidden confounder (severe shift)
    H_test = -2.5 * rng.randn(n_per_env)
    x1t = rng.randn(n_per_env)
    x2t = rng.randn(n_per_env)
    x3t = H_test + 0.4 * rng.randn(n_per_env)
    x4t = 0.8 * H_test + 0.3 * rng.randn(n_per_env)
    x5t = rng.randn(n_per_env)
    y_test = 2.0 * x1t + 1.5 * x2t + 0.3 * H_test + 0.5 * rng.randn(n_per_env)
    X_test = np.column_stack([x1t, x2t, x3t, x4t, x5t])

    feature_names = ['X1_causal', 'X2_causal', 'X3_spurious', 'X4_spurious', 'X5_noise']

    # Standardise
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    return {
        'X_train': X_train_sc,
        'y_train': y_train,
        'env_labels': env_labels,
        'X_test': X_test_sc,
        'y_test': y_test,
        'feature_names': feature_names,
        'true_causal': ['X1_causal', 'X2_causal'],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Exercise 1: ICP from Scratch
# ══════════════════════════════════════════════════════════════════════════════

def exercise_1_icp_from_scratch():
    """
    Exercise 1: Implement a simple ICP procedure from scratch.

    Task:
    -----
    Complete the functions below:
    1. `test_invariance_simple`: Test whether residuals from Y ~ X_S are
       invariant across environments using a two-stage test (F-test + Levene).
    2. `run_icp_exhaustive`: For a small feature set (p <= 8), exhaustively
       test all subsets and return the ICP estimate (intersection of accepted sets).
    3. `run_icp_greedy`: For larger p, use greedy forward selection guided
       by invariance tests.

    Then run ICP on the provided dataset and verify it selects the causal features.
    """
    print("=" * 65)
    print("Exercise 1: ICP Implementation from Scratch")
    print("=" * 65)

    data = make_icp_dataset(n_per_env=400, n_train_envs=4)
    X = data['X_train']
    y = data['y_train']
    env_labels = data['env_labels']
    feature_names = data['feature_names']
    true_causal = data['true_causal']

    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(np.unique(env_labels))} environments")
    print(f"True causal features: {true_causal}")

    # ── YOUR IMPLEMENTATION ────────────────────────────────────────────────

    def test_invariance_simple(y: np.ndarray,
                                X_S: np.ndarray,
                                env_labels: np.ndarray,
                                alpha: float = 0.05) -> dict:
        """
        Test H0(S): residuals from Y ~ X_S are invariant across environments.

        Two-stage test:
        - Stage 1: F-test comparing pooled regression (same coefficients) vs
                   per-environment regressions (different coefficients).
        - Stage 2: Levene's test for equal residual variances.

        Reject H0(S) if either stage rejects (Bonferroni: multiply smaller p by 2).

        Parameters
        ----------
        y : array, shape (n,)
        X_S : array, shape (n, |S|)
        env_labels : array, shape (n,)
        alpha : float

        Returns
        -------
        dict:
            'invariant': bool — True if H0(S) not rejected
            'p_value': float — combined p-value
            'p_coef': float — p-value for coefficient stability
            'p_var': float — p-value for variance equality
        """
        # TODO: Implement the two-stage invariance test
        #
        # Hint — coefficient F-test:
        #   rss_pooled = residuals from LinearRegression fit on full data
        #   rss_separate = sum of residuals from per-environment fits
        #   F = ((rss_pooled - rss_separate) / (p_s * (E-1))) /
        #       (rss_separate / (n - E*(p_s+1)))
        #   p_coef = 1 - stats.f.cdf(F, p_s*(E-1), n-E*(p_s+1))
        #
        # Hint — Levene's test:
        #   p_var = stats.levene(*residuals_per_env).pvalue
        #
        # Hint — combined p-value:
        #   p_combined = min(min(p_coef, p_var) * 2, 1.0)  # Bonferroni

        raise NotImplementedError("Implement test_invariance_simple()")

    def run_icp_exhaustive(y: np.ndarray,
                            X: np.ndarray,
                            env_labels: np.ndarray,
                            feature_names: list,
                            alpha: float = 0.05) -> dict:
        """
        Exhaustive ICP: test all 2^p subsets and return the intersection of
        accepted sets.

        Only use this for p <= 8 (intractable for larger p).

        Parameters
        ----------
        y, X, env_labels, feature_names : standard inputs
        alpha : significance level

        Returns
        -------
        dict:
            'accepted_sets': list of accepted subsets (as frozensets of names)
            'icp_estimate': list of feature names in intersection
        """
        # TODO: Implement exhaustive ICP
        #
        # Hint: iterate over all non-empty subsets of {0, 1, ..., p-1}
        #   from itertools import combinations
        #   for size in range(1, p+1):
        #     for subset in combinations(range(p), size):
        #       test H0(subset) using test_invariance_simple
        #
        # Collect accepted_sets = [frozenset of names for accepted subsets]
        # Compute intersection = set.intersection(*accepted_sets)

        raise NotImplementedError("Implement run_icp_exhaustive()")

    def run_icp_greedy(y: np.ndarray,
                        X: np.ndarray,
                        env_labels: np.ndarray,
                        feature_names: list,
                        alpha: float = 0.05,
                        max_features: int = 10) -> list:
        """
        Greedy forward ICP: at each step add the feature that:
        1. Maintains invariance of the current set + new feature
        2. Most reduces the pooled residual sum of squares

        Stop when no feature can be added while maintaining invariance.

        Parameters
        ----------
        y, X, env_labels, feature_names : standard inputs
        alpha : significance level
        max_features : maximum features to select

        Returns
        -------
        list : selected feature names
        """
        # TODO: Implement greedy ICP
        #
        # Hint:
        #   selected_idx = []
        #   remaining_idx = list(range(p))
        #   for step in range(max_features):
        #     best_feature, best_rss = None, inf
        #     for j in remaining_idx:
        #       candidate = selected_idx + [j]
        #       result = test_invariance_simple(y, X[:, candidate], env_labels, alpha)
        #       if result['invariant']:
        #         compute RSS from pooled fit on X[:, candidate]
        #         if rss < best_rss: best_feature = j, best_rss = rss
        #     if best_feature is None: break
        #     selected_idx.append(best_feature), remaining_idx.remove(best_feature)

        raise NotImplementedError("Implement run_icp_greedy()")

    # ── Run and validate ──────────────────────────────────────────────────

    print("\nRunning invariance tests on individual features...")
    for j, fname in enumerate(feature_names):
        result = test_invariance_simple(y, X[:, [j]], env_labels, alpha=0.05)
        print(f"  {fname:15s}: invariant={result['invariant']}, "
              f"p={result['p_value']:.4f}")

    print("\nRunning exhaustive ICP (p=5, all 31 subsets)...")
    try:
        icp_result = run_icp_exhaustive(y, X, env_labels, feature_names, alpha=0.05)
        icp_estimate = icp_result['icp_estimate']
        n_accepted = len(icp_result['accepted_sets'])
    except NotImplementedError:
        print("  Falling back to greedy ICP...")
        icp_estimate = run_icp_greedy(y, X, env_labels, feature_names, alpha=0.05)
        n_accepted = None

    print(f"  ICP estimate: {icp_estimate}")
    if n_accepted is not None:
        print(f"  Accepted sets: {n_accepted}")

    # Validation
    icp_set = set(icp_estimate)
    true_set = set(true_causal)
    tp = len(icp_set & true_set)
    fp = len(icp_set - true_set)
    fn = len(true_set - icp_set)

    print(f"\nValidation vs ground truth:")
    print(f"  True positives: {tp}  (causal features correctly selected)")
    print(f"  False positives: {fp}  (spurious features incorrectly selected)")
    print(f"  False negatives: {fn}  (causal features missed)")

    check_exercise_1(icp_estimate, true_causal, feature_names)


def check_exercise_1(icp_estimate: list, true_causal: list, feature_names: list):
    """Auto-check for Exercise 1."""
    print("\n--- Exercise 1 Auto-Check ---")
    passed = True

    if not icp_estimate:
        print("FAIL: ICP returned no features. Check your implementation.")
        return

    icp_set = set(icp_estimate)
    true_set = set(true_causal)

    fp = icp_set - true_set
    fn = true_set - icp_set

    if fp:
        print(f"PARTIAL: False positives found: {fp}")
        print("  ICP should exclude spurious features X3_spurious, X4_spurious.")
        print("  Check that your F-test correctly detects coefficient instability.")
        passed = False

    if fn:
        print(f"PARTIAL: False negatives found: {fn}")
        print("  ICP missed some true causal features.")
        print("  Check that your invariance test accepts the causal feature set.")
        passed = False

    if passed:
        print("PASS: ICP correctly identified the causal feature set.")
        print(f"  Selected: {sorted(icp_set)}")
        print(f"  True causal: {sorted(true_set)}")
    else:
        print(f"  Selected: {sorted(icp_set)}")
        print(f"  True causal: {sorted(true_set)}")

    print()


# ══════════════════════════════════════════════════════════════════════════════
# Exercise 2: FCI vs PC Comparison
# ══════════════════════════════════════════════════════════════════════════════

def exercise_2_fci_vs_pc():
    """
    Exercise 2: Run FCI (handling latent confounders) and compare with PC.

    Task:
    -----
    1. Generate a dataset where latent confounders exist between observed variables.
    2. Run PC algorithm (assumes causal sufficiency).
    3. Run FCI algorithm (handles latent confounders).
    4. Compare the discovered graphs: what edges does PC add incorrectly?
    5. Extract features adjacent to the target in both graphs and compare.

    Key difference:
    - PC produces spurious edges between variables connected via latent confounders
    - FCI uses bidirected edges (X <-> Y) to represent latent common causes
    - Feature sets from FCI are more conservative (exclude latent-confounder paths)
    """
    print("=" * 65)
    print("Exercise 2: FCI vs PC — Handling Latent Confounders")
    print("=" * 65)

    try:
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.search.ConstraintBased.FCI import fci
        from causallearn.utils.cit import fisherz
    except ImportError:
        print("ERROR: causal-learn not installed.")
        print("Install with: pip install causal-learn")
        return

    # ── Generate dataset with latent confounder ────────────────────────────
    # We observe: X1, X2, X3, Y, Z1, Z2
    # Hidden: H (confounder between X3 and Y)
    # True causal graph:
    #   X1 → Y  (direct cause)
    #   X2 → Y  (direct cause)
    #   H → X3  (H is hidden)
    #   H → Y   (H is hidden — confounds X3-Y relationship)
    #   Z1, Z2 are independent noise

    rng = np.random.RandomState(99)
    n = 1500

    H = rng.randn(n)                              # Hidden confounder
    X1 = rng.randn(n)                             # Observed, causal
    X2 = rng.randn(n)                             # Observed, causal
    X3 = 1.5 * H + 0.3 * rng.randn(n)            # Confounded by H
    Z1 = rng.randn(n)                             # Independent noise
    Z2 = rng.randn(n)                             # Independent noise
    Y = 2.0 * X1 + 1.5 * X2 + 0.8 * H + 0.5 * rng.randn(n)  # Y also affected by H

    # Observed data (H not included — latent)
    X_obs = np.column_stack([X1, X2, X3, Z1, Z2, Y])
    obs_names = ['X1', 'X2', 'X3', 'Z1', 'Z2', 'Y']
    target_idx = 5  # Y

    # Standardise
    scaler = StandardScaler()
    X_obs_sc = scaler.fit_transform(X_obs)

    print(f"\nObserved variables: {obs_names}")
    print("Hidden: H (confounder between X3 and Y)")
    print("True causal parents of Y: {X1, X2} (X3 is spurious via H)")

    # ── YOUR IMPLEMENTATION ────────────────────────────────────────────────

    def run_pc(data, variable_names, target_variable, alpha=0.05):
        """
        Run PC algorithm and return:
        - adjacency matrix
        - set of features adjacent to target_variable (in MB(Y))

        Parameters
        ----------
        data : np.ndarray, shape (n, p)
        variable_names : list of str
        target_variable : str
        alpha : float

        Returns
        -------
        dict:
            'graph': adjacency matrix (cg.G.graph)
            'mb_features': list of feature names in Markov blanket of target
            'edge_list': list of (source, target, type) tuples
        """
        # TODO: Implement PC run
        # Hint:
        #   cg = pc(data, alpha=alpha, indep_test=fisherz, stable=True)
        #   adj = cg.G.graph
        #   Extract features adjacent to target_variable index

        raise NotImplementedError("Implement run_pc()")

    def run_fci(data, variable_names, target_variable, alpha=0.05):
        """
        Run FCI algorithm and return:
        - PAG (partial ancestral graph)
        - set of features adjacent to target_variable

        FCI uses richer edge marks:
        - adj[i,j] == -1 and adj[j,i] == 1: i -> j
        - adj[i,j] == 1 and adj[j,i] == -1: i <- j
        - adj[i,j] == 2 and adj[j,i] == 2: i <-> j (hidden common cause)
        - adj[i,j] == 1 and adj[j,i] == 1: i -- j (undirected)

        Parameters
        ----------
        data, variable_names, target_variable, alpha : same as run_pc()

        Returns
        -------
        dict:
            'graph': adjacency matrix (G.graph from FCI)
            'mb_features': list of feature names adjacent to target
                           (conservative: include both -> and <-> edges)
            'bidirected_edges': list of (X, Y) pairs with hidden confounder
        """
        # TODO: Implement FCI run
        # Hint:
        #   G, edges = fci(data, independence_test_method=fisherz, alpha=alpha)
        #   adj = G.graph
        #   For feature selection:
        #     - Include X if X -> target (adj[X_idx, target_idx] == -1 and adj[target_idx, X_idx] == 1)
        #     - Include X if X <-> target (bidirected — hidden common cause — include for prediction)
        #     - Exclude X if only X <- target (X is an effect of target, not a feature)

        raise NotImplementedError("Implement run_fci()")

    def compare_graphs(pc_result, fci_result, variable_names):
        """
        Compare PC and FCI results.
        Print edges that differ between the two algorithms.
        """
        # TODO: Compare adjacency matrices from PC and FCI
        # Highlight:
        # 1. Edges in PC but not FCI (spurious edges from latent confounders)
        # 2. Bidirected edges in FCI (latent confounder indicators)
        # 3. Features selected by PC but not FCI (due to latent confounders)
        # 4. Features selected by FCI but not PC

        raise NotImplementedError("Implement compare_graphs()")

    # ── Run and validate ──────────────────────────────────────────────────

    print("\nRunning PC algorithm (assumes causal sufficiency)...")
    try:
        pc_result = run_pc(X_obs_sc, obs_names, 'Y', alpha=0.05)
        print(f"PC features for Y: {pc_result['mb_features']}")
    except NotImplementedError:
        print("  PC not yet implemented. Showing expected output:")
        print("  Expected PC features: likely includes X3 (spurious via H)")
        pc_result = None

    print("\nRunning FCI algorithm (handles latent confounders)...")
    try:
        fci_result = run_fci(X_obs_sc, obs_names, 'Y', alpha=0.05)
        print(f"FCI features for Y: {fci_result['mb_features']}")
        print(f"FCI bidirected edges (latent confounders): {fci_result.get('bidirected_edges', [])}")
    except NotImplementedError:
        print("  FCI not yet implemented. Showing expected output:")
        print("  Expected FCI features: {X1, X2} (excludes X3, shows X3 <-> Y)")
        fci_result = None

    if pc_result is not None and fci_result is not None:
        print("\nComparing PC and FCI:")
        try:
            compare_graphs(pc_result, fci_result, obs_names)
        except NotImplementedError:
            print("  compare_graphs() not yet implemented.")

    check_exercise_2(pc_result, fci_result, obs_names)


def check_exercise_2(pc_result, fci_result, variable_names):
    """Auto-check for Exercise 2."""
    print("\n--- Exercise 2 Auto-Check ---")

    if pc_result is None or fci_result is None:
        print("INCOMPLETE: Implement run_pc() and run_fci() to run the check.")
        return

    pc_features = set(pc_result.get('mb_features', []))
    fci_features = set(fci_result.get('mb_features', []))

    print(f"PC features for Y:  {sorted(pc_features)}")
    print(f"FCI features for Y: {sorted(fci_features)}")

    # Key check: PC should include X3 (spurious), FCI should not
    if 'X3' in pc_features and 'X3' not in fci_features:
        print("PASS: FCI correctly excluded X3 (spurious via hidden H), PC incorrectly included it.")
    elif 'X3' not in pc_features and 'X3' not in fci_features:
        print("PASS: Both algorithms excluded X3 (may be due to strong CI test).")
    elif 'X3' in fci_features:
        print("NOTE: FCI included X3. If via bidirected edge (X3 <-> Y), this is correct —")
        print("      it signals a hidden confounder. For pure prediction, X3 can be included.")
    else:
        print("PARTIAL: Review your FCI implementation.")

    # Check: both should include X1, X2
    for x in ['X1', 'X2']:
        pc_has = x in pc_features
        fci_has = x in fci_features
        status = "PASS" if (pc_has and fci_has) else "PARTIAL"
        print(f"{status}: {x} — PC: {'included' if pc_has else 'missing'}, "
              f"FCI: {'included' if fci_has else 'missing'}")

    print()


# ══════════════════════════════════════════════════════════════════════════════
# Exercise 3: Distribution Shift Test Bench
# ══════════════════════════════════════════════════════════════════════════════

def exercise_3_shift_test_bench():
    """
    Exercise 3: Build a distribution-shift test bench comparing causal
    vs predictive feature sets.

    Task:
    -----
    Complete the `ShiftTestBench` class with the following methods:
    1. `fit`: Run all feature selection methods on training data.
    2. `evaluate_at_shift`: Given a shift severity (0=none to 1=severe),
       generate test data at that shift and compute R² for each method.
    3. `run`: Run evaluation at multiple shift levels and return a DataFrame.
    4. `plot`: Plot performance degradation curves.

    The test bench should:
    - Accept multiple feature selection methods as input (dict of name → feature list)
    - Generate test data at shift levels [0.0, 0.25, 0.5, 0.75, 1.0]
    - Return a tidy DataFrame with columns: shift_strength, method, n_features, r2
    - Plot performance curves with causal methods highlighted in blue
    """
    print("=" * 65)
    print("Exercise 3: Distribution Shift Test Bench")
    print("=" * 65)

    data = make_icp_dataset(n_per_env=300, n_train_envs=3)
    X_train = data['X_train']
    y_train = data['y_train']
    env_labels = data['env_labels']
    feature_names = data['feature_names']
    true_causal = data['true_causal']

    print(f"\nTraining data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"True causal features: {true_causal}")

    # ── YOUR IMPLEMENTATION ────────────────────────────────────────────────

    class ShiftTestBench:
        """
        Distribution shift test bench for comparing causal vs predictive
        feature sets.

        Usage:
        ------
        bench = ShiftTestBench(X_train, y_train, env_labels, feature_names,
                               causal_method_names=['ICP'])
        bench.fit()
        results_df = bench.run(shift_levels=[0.0, 0.25, 0.5, 0.75, 1.0])
        bench.plot(results_df)
        """

        def __init__(self,
                     X_train: np.ndarray,
                     y_train: np.ndarray,
                     env_labels: np.ndarray,
                     feature_names: list,
                     alpha_icp: float = 0.05,
                     causal_method_names: list = None):
            """
            Parameters
            ----------
            X_train : array, shape (n, p) — standardised training features
            y_train : array, shape (n,) — training target
            env_labels : array, shape (n,) — environment labels for ICP
            feature_names : list — feature names corresponding to X columns
            alpha_icp : float — significance level for ICP invariance tests
            causal_method_names : list — method names to highlight as causal
            """
            self.X_train = X_train
            self.y_train = y_train
            self.env_labels = env_labels
            self.feature_names = feature_names
            self.alpha_icp = alpha_icp
            self.causal_method_names = causal_method_names or ['ICP']
            self.feature_sets = {}   # populated by fit()
            self.models = {}         # fitted models, populated by fit()

        def _run_icp(self) -> list:
            """
            Run greedy ICP on training data.

            Returns list of selected feature names.
            """
            # TODO: Implement ICP selection
            # Hint: use the test_invariance and greedy loop from Exercise 1
            # or a simplified version. You may also copy from Notebook 02.
            raise NotImplementedError("Implement _run_icp() in ShiftTestBench")

        def _run_lasso(self) -> list:
            """Run LassoCV and return selected feature names."""
            # TODO: Implement Lasso selection
            # Hint:
            #   lasso = LassoCV(cv=5).fit(self.X_train, self.y_train)
            #   selected = [self.feature_names[i] for i in np.where(np.abs(lasso.coef_) > 1e-4)[0]]
            raise NotImplementedError("Implement _run_lasso() in ShiftTestBench")

        def _run_rf_importance(self) -> list:
            """Run Random Forest and return top-importance feature names."""
            # TODO: Implement RF importance selection
            # Hint:
            #   from sklearn.ensemble import RandomForestRegressor
            #   rf = RandomForestRegressor(n_estimators=200).fit(...)
            #   Select features with importance > mean
            raise NotImplementedError("Implement _run_rf_importance() in ShiftTestBench")

        def fit(self):
            """
            Run all feature selection methods on training data.
            Fit a GradientBoostingRegressor for each feature set.
            Store in self.feature_sets and self.models.
            """
            # TODO: Call _run_icp, _run_lasso, _run_rf_importance
            # Store results in self.feature_sets (dict: method_name -> feature_list)
            # Also add 'All features' and 'True causal' as reference sets
            # For each feature set, fit a GradientBoostingRegressor
            # Store fitted models in self.models (dict: method_name -> fitted model)
            raise NotImplementedError("Implement fit() in ShiftTestBench")

        def _generate_shifted_test(self, shift_strength: float,
                                    n_test: int = 400,
                                    seed: int = 100) -> tuple:
            """
            Generate test data at a given shift severity.

            Parameters
            ----------
            shift_strength : float
                0.0 = same distribution as training
                1.0 = maximum shift (hidden confounder fully reversed)
            n_test : int
                Number of test samples.
            seed : int

            Returns
            -------
            X_test : array, shape (n_test, p) — standardised test features
            y_test : array, shape (n_test,)
            """
            # TODO: Generate test data at specified shift_strength
            # Hint: adapt make_icp_dataset() logic
            # The key: H_test_mean = 1.5 - shift_strength * 4.0
            # (interpolate from training distribution to reversed distribution)
            #
            # Use the SAME scaler that was fit on training data
            # (passed as argument or stored in self)
            raise NotImplementedError("Implement _generate_shifted_test() in ShiftTestBench")

        def evaluate_at_shift(self, shift_strength: float) -> dict:
            """
            Evaluate all feature sets at a given shift level.

            Returns dict: method_name -> {'r2': float, 'n_features': int}
            """
            # TODO: Generate test data, then for each method:
            # 1. Get the feature indices for this method's feature set
            # 2. Extract those columns from X_test
            # 3. Predict with self.models[method_name]
            # 4. Compute R² against y_test
            raise NotImplementedError("Implement evaluate_at_shift() in ShiftTestBench")

        def run(self, shift_levels: list = None) -> pd.DataFrame:
            """
            Run full evaluation across all shift levels.

            Parameters
            ----------
            shift_levels : list of float
                Default: [0.0, 0.25, 0.5, 0.75, 1.0]

            Returns
            -------
            pd.DataFrame with columns: shift_strength, method, n_features, r2
            """
            if shift_levels is None:
                shift_levels = [0.0, 0.25, 0.5, 0.75, 1.0]

            # TODO: Call evaluate_at_shift for each level, collect into DataFrame
            raise NotImplementedError("Implement run() in ShiftTestBench")

        def plot(self, results_df: pd.DataFrame):
            """
            Plot performance degradation curves.

            Causal methods (in self.causal_method_names) should be plotted
            with thicker lines and distinct colours. Predictive methods
            with thinner dashed lines.
            """
            import matplotlib.pyplot as plt

            # TODO: Plot R² vs shift_strength for each method
            # Hint:
            #   for method, group in results_df.groupby('method'):
            #     is_causal = method in self.causal_method_names
            #     linewidth = 2.5 if is_causal else 1.5
            #     linestyle = '-' if is_causal else '--'
            #     ax.plot(group['shift_strength'], group['r2'],
            #             label=method, linewidth=linewidth, linestyle=linestyle)
            raise NotImplementedError("Implement plot() in ShiftTestBench")

    # ── Run and validate ──────────────────────────────────────────────────

    print("\nInitialising ShiftTestBench...")
    bench = ShiftTestBench(
        X_train=X_train,
        y_train=y_train,
        env_labels=env_labels,
        feature_names=feature_names,
        causal_method_names=['ICP', 'True causal'],
    )

    try:
        bench.fit()
        print(f"Feature sets fitted: {list(bench.feature_sets.keys())}")
        print("\nFeature sets:")
        for method, features in bench.feature_sets.items():
            print(f"  {method:20s}: {features}")

        results_df = bench.run(shift_levels=[0.0, 0.25, 0.5, 0.75, 1.0])
        print("\nResults DataFrame:")
        print(results_df.to_string(index=False))

        bench.plot(results_df)

        check_exercise_3(bench, results_df)

    except NotImplementedError as e:
        print(f"\nNot yet implemented: {e}")
        print("\nExpected output when complete:")
        print("  ICP should degrade less than Lasso/RF under shift")
        print("  Example: ICP R²=0.85→0.78 (9% drop) vs Lasso R²=0.88→0.42 (52% drop)")
        print()


def check_exercise_3(bench, results_df):
    """Auto-check for Exercise 3."""
    print("\n--- Exercise 3 Auto-Check ---")

    if results_df is None or len(results_df) == 0:
        print("INCOMPLETE: run() returned empty DataFrame.")
        return

    required_cols = {'shift_strength', 'method', 'r2'}
    if not required_cols.issubset(results_df.columns):
        missing = required_cols - set(results_df.columns)
        print(f"FAIL: Missing columns: {missing}")
        return

    # Check ICP exists and degrades less than Lasso/RF
    methods = results_df['method'].unique().tolist()
    print(f"Methods in results: {methods}")

    if 'ICP' not in methods:
        print("FAIL: 'ICP' method not found in results.")
        return

    if 'Lasso' not in methods:
        print("PARTIAL: 'Lasso' method not found. Add it to compare against ICP.")

    # Check degradation
    icp_data = results_df[results_df['method'] == 'ICP'].sort_values('shift_strength')
    r2_low = icp_data[icp_data['shift_strength'] == icp_data['shift_strength'].min()]['r2'].values
    r2_high = icp_data[icp_data['shift_strength'] == icp_data['shift_strength'].max()]['r2'].values

    if len(r2_low) > 0 and len(r2_high) > 0:
        icp_degrad = r2_low[0] - r2_high[0]
        print(f"ICP R² degradation (low→high shift): {icp_degrad:.4f}")

        if 'Lasso' in methods:
            lasso_data = results_df[results_df['method'] == 'Lasso'].sort_values('shift_strength')
            lasso_low = lasso_data['r2'].values[0]
            lasso_high = lasso_data['r2'].values[-1]
            lasso_degrad = lasso_low - lasso_high
            print(f"Lasso R² degradation:               {lasso_degrad:.4f}")

            if icp_degrad < lasso_degrad:
                print("PASS: ICP degrades less than Lasso under distribution shift.")
            else:
                print("NOTE: ICP degraded as much as Lasso. Verify your shift data generation.")

    print()


# ══════════════════════════════════════════════════════════════════════════════
# Main runner
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Module 9 Causal Feature Selection — Self-Check Exercises'
    )
    parser.add_argument(
        '--exercise', type=int, choices=[1, 2, 3],
        help='Run a specific exercise (1, 2, or 3). Omit to run all.'
    )
    args = parser.parse_args()

    exercises = {
        1: exercise_1_icp_from_scratch,
        2: exercise_2_fci_vs_pc,
        3: exercise_3_shift_test_bench,
    }

    if args.exercise:
        exercises[args.exercise]()
    else:
        for ex_num, ex_func in exercises.items():
            try:
                ex_func()
            except Exception as e:
                print(f"\nExercise {ex_num} raised an error: {type(e).__name__}: {e}")
                print("Implement the NotImplementedError placeholders to proceed.\n")


if __name__ == '__main__':
    main()
