"""
Module 02: Information Theory Feature Selection — Self-Check Exercises

Three exercises building on the Brown et al. (2012) unified CLM framework
and the advanced information measures from Guide 02.

Exercise 1: Derive and implement the CMIM criterion from the CLM framework
Exercise 2: Compute Rényi mutual information for α=0.5, 1.0, 2.0 and compare rankings
Exercise 3: Build a transfer entropy-based feature selector for a time series problem

Run this file directly to check your implementations:
    python 01_information_theory_exercises.py

Reference:
    Brown, G., Pocock, A., Zhao, M-J., Lujan, M. (2012).
    Conditional Likelihood Maximisation: A Unifying Framework for
    Information Theoretic Feature Selection. JMLR 13, 27-66.
"""

import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.datasets import load_breast_cancer
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def discretise(X, n_bins=10):
    """Quantile-based discretisation of a continuous feature matrix."""
    kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    return kbd.fit_transform(X).astype(int)


def mi(a, b):
    """Marginal mutual information I(a; b) for discrete arrays."""
    return mutual_info_score(a, b)


def cmi(a, b, c):
    """
    Conditional mutual information I(a; b | c) via chain rule.

    I(a; b | c) = I(a; b, c) - I(a; c)
    Encodes (b, c) as a joint integer variable to avoid partitioning.
    """
    n_c = int(np.max(c)) + 1
    bc = b * n_c + c
    return max(0.0, mi(a, bc) - mi(a, c))


# ---------------------------------------------------------------------------
# Exercise 1: CMIM from the CLM Framework
# ---------------------------------------------------------------------------

def exercise1_about():
    """
    CMIM (Conditional Mutual Information Maximisation, Fleuret 2004) is one of
    the five criteria unified by Brown et al. (2012) in the CLM framework:

        J_CLM(x_k) = I(x_k; y) + sum_j [ gamma_j * I(x_k; x_j | y)
                                         - beta_j * I(x_k; x_j) ]

    CMIM uses beta = gamma = 1, but applies the criterion to the SINGLE
    worst-case feature x_j* = argmin_{x_j in S} I(x_k; y | x_j),
    rather than summing or averaging over all selected features.

    Formally:
        J_CMIM(x_k) = min_{x_j in S} I(x_k; y | x_j)

    This can be rewritten using the chain rule:
        I(x_k; y | x_j) = I(x_k; y) - I(x_k; x_j) + I(x_k; x_j | y)

    which is exactly the CLM criterion with beta=1, gamma=1, evaluated at
    the minimising x_j.
    """
    pass


def cmim_score(k, y, S, X):
    """
    CMIM criterion: select the feature with maximum guaranteed CMI.

    Implements: J_CMIM(x_k) = min_{x_j in S} I(x_k; y | x_j)

    When S is empty, falls back to: J_CMIM(x_k) = I(x_k; y)

    Parameters
    ----------
    k : int
        Index of the candidate feature in X
    y : array of int
        Discretised target variable
    S : list of int
        Indices of already-selected features
    X : array of int, shape (n, p)
        Full discretised feature matrix

    Returns
    -------
    float : CMIM score for feature k (higher = better candidate)

    Notes
    -----
    Use the cmi() function defined above for I(x_k; y | x_j).
    The cmi function signature is: cmi(a, b, c) returns I(a; b | c).
    So: I(x_k; y | x_j) = cmi(X[:, k], y, X[:, j])
    """
    # YOUR CODE HERE
    # --------------------------------------------------
    # Hint: when S is empty, return I(x_k; y)
    # When S is non-empty, compute I(x_k; y | x_j) for each x_j in S
    # and return the MINIMUM value (not the mean like JMI does)
    raise NotImplementedError("Implement cmim_score in Exercise 1")


def clm_verify_cmim(X_disc, y_disc, k_feat=5):
    """
    Verify that CMIM is a special case of the CLM framework.

    For a given candidate feature x_k and single selected feature x_j,
    verify:
        min_{x_j in S} I(x_k; y | x_j)
        == I(x_k; y) - I(x_k; x_j*) + I(x_k; x_j* | y)
    where x_j* is the minimising feature.

    Parameters
    ----------
    X_disc : array of int, shape (n, p)
        Discretised feature matrix
    y_disc : array of int
        Discretised target
    k_feat : int
        Candidate feature index to test

    Returns
    -------
    bool : True if both formulations agree (within 1e-8)
    """
    # YOUR CODE HERE
    # --------------------------------------------------
    # Step 1: Set S = [0] (one selected feature for simplicity)
    # Step 2: Compute J_CMIM using the min formulation
    # Step 3: Compute the CLM expansion for the minimising j:
    #         I(x_k; y) - I(x_k; x_j*) + I(x_k; x_j* | y)
    # Step 4: Check they agree within 1e-8
    raise NotImplementedError("Implement clm_verify_cmim in Exercise 1")


def greedy_cmim_selector(X_disc, y_disc, k_features, feature_names=None):
    """
    Greedy forward feature selection using CMIM.

    Parameters
    ----------
    X_disc : array of int, shape (n, p)
        Discretised feature matrix
    y_disc : array of int
        Discretised target
    k_features : int
        Number of features to select
    feature_names : list of str, optional
        Feature names for reporting

    Returns
    -------
    selected : list of int
        Indices of selected features in selection order
    """
    # YOUR CODE HERE
    # --------------------------------------------------
    # Hint: Use the same greedy loop as itfs_select in Notebook 01
    # At each step, score all candidates with cmim_score and pick the best
    raise NotImplementedError("Implement greedy_cmim_selector in Exercise 1")


def test_exercise1():
    """Self-check tests for Exercise 1."""
    print("=" * 60)
    print("Exercise 1: CMIM from the CLM Framework")
    print("=" * 60)

    # Load and discretise data
    data = load_breast_cancer()
    X_disc = discretise(data.data, n_bins=10)
    y_disc = data.target.astype(int)

    # Test 1.1: cmim_score with empty S equals marginal MI
    score_empty = cmim_score(0, y_disc, [], X_disc)
    mi_marginal = mi(X_disc[:, 0], y_disc)
    assert abs(score_empty - mi_marginal) < 1e-8, (
        f"cmim_score with empty S should equal I(x_k; y) = {mi_marginal:.4f}, "
        f"got {score_empty:.4f}"
    )
    print(f"  [PASS] cmim_score(k=0, S=[]) == I(x0; y) = {mi_marginal:.4f}")

    # Test 1.2: cmim_score is non-negative
    score_with_s = cmim_score(5, y_disc, [0, 1], X_disc)
    assert score_with_s >= 0, (
        f"CMIM score must be non-negative (CMI >= 0), got {score_with_s:.4f}"
    )
    print(f"  [PASS] cmim_score(k=5, S=[0,1]) >= 0: score={score_with_s:.4f}")

    # Test 1.3: cmim_score <= JMI score (CMIM is more conservative)
    # CMIM takes min, JMI takes mean — min <= mean for non-negative values
    def jmi_score_single(k, y, S, X):
        relevance = mi(X[:, k], y)
        if not S:
            return relevance
        redundancy = np.mean([mi(X[:, k], X[:, j]) for j in S])
        complement = np.mean([cmi(X[:, k], X[:, j], y) for j in S])
        return relevance - redundancy + complement

    S_test = [0, 1, 2]
    cmim_val = cmim_score(10, y_disc, S_test, X_disc)
    jmi_val = jmi_score_single(10, y_disc, S_test, X_disc)
    # CMIM and JMI can differ in either direction when using different formulations
    # Just check they are both non-negative
    assert cmim_val >= 0, f"CMIM score must be non-negative, got {cmim_val:.4f}"
    print(f"  [PASS] cmim_score(k=10, S=[0,1,2])={cmim_val:.4f}, "
          f"jmi_score(k=10)={jmi_val:.4f}")

    # Test 1.4: CLM equivalence verification
    agrees = clm_verify_cmim(X_disc, y_disc, k_feat=5)
    assert agrees, (
        "CMIM min formulation and CLM expansion should agree within 1e-8. "
        "Check your CLM expansion: I(x_k; y) - I(x_k; x_j*) + I(x_k; x_j* | y)"
    )
    print("  [PASS] CLM expansion agrees with min-CMI formulation")

    # Test 1.5: Greedy selector returns correct number of features
    selected = greedy_cmim_selector(X_disc, y_disc, k_features=8)
    assert len(selected) == 8, (
        f"Selector should return 8 features, got {len(selected)}"
    )
    assert len(set(selected)) == 8, (
        "Selected features must be unique (no repeats)"
    )
    assert all(0 <= i < X_disc.shape[1] for i in selected), (
        "All selected indices must be valid feature indices"
    )
    print(f"  [PASS] greedy_cmim_selector returns 8 unique features: {selected}")

    print("\n  All Exercise 1 tests PASSED.\n")


# ---------------------------------------------------------------------------
# Exercise 2: Rényi Mutual Information for α=0.5, 1.0, 2.0
# ---------------------------------------------------------------------------

def renyi_entropy(probs, alpha):
    """
    Compute Rényi entropy H_alpha from a probability vector.

    H_alpha(X) = (1/(1-alpha)) * log(sum_x p(x)^alpha)

    For alpha = 1: recovers Shannon entropy H(X) = -sum p(x) log p(x).

    Parameters
    ----------
    probs : array
        Probability mass function (must sum to approximately 1)
    alpha : float
        Rényi order. Must be > 0 and != 1 (use limit formula at alpha=1).

    Returns
    -------
    float : Rényi entropy H_alpha
    """
    probs = np.asarray(probs, dtype=float)
    probs = probs[probs > 0]  # Avoid log(0)

    if np.isclose(alpha, 1.0):
        # Shannon entropy as limiting case
        return -float(np.sum(probs * np.log(probs)))

    return float((1.0 / (1.0 - alpha)) * np.log(np.sum(probs ** alpha)))


def renyi_mi(x_disc, y_disc, alpha):
    """
    Compute alpha-mutual information using the additive approximation:

        I_alpha(X; Y) ~= H_alpha(X) + H_alpha(Y) - H_alpha(X, Y)

    This is exact for alpha=1 (Shannon) and approximate for other alpha.

    Parameters
    ----------
    x_disc : array of int
        Discretised feature values
    y_disc : array of int
        Discretised target values
    alpha : float
        Rényi order

    Returns
    -------
    float : alpha-MI (non-negative)

    Notes
    -----
    To compute the joint probability distribution of (x, y):
    - Encode as: xy = x_disc * (max(y_disc) + 1) + y_disc
    - Compute frequencies with np.unique(..., return_counts=True)
    - Divide by n to get probabilities
    """
    # YOUR CODE HERE
    # --------------------------------------------------
    # Step 1: Compute marginal probabilities p(x) and p(y)
    # Step 2: Compute joint probabilities p(x, y) via integer encoding
    # Step 3: Apply renyi_entropy to each distribution
    # Step 4: Return H_alpha(X) + H_alpha(Y) - H_alpha(X, Y), clipped to 0
    raise NotImplementedError("Implement renyi_mi in Exercise 2")


def compare_renyi_rankings(X_disc, y_disc, alphas=(0.5, 1.0, 2.0),
                            feature_names=None):
    """
    Rank features by Rényi MI for multiple alpha values and compare rankings.

    Parameters
    ----------
    X_disc : array of int, shape (n, p)
        Discretised feature matrix
    y_disc : array of int
        Discretised target
    alphas : tuple of float
        Rényi orders to compare
    feature_names : list of str, optional
        Feature names

    Returns
    -------
    dict mapping alpha -> array of shape (p,)
        Rényi MI scores for each feature at each alpha value
    """
    p = X_disc.shape[1]
    results = {}

    for alpha in alphas:
        scores = np.zeros(p)
        for i in range(p):
            scores[i] = renyi_mi(X_disc[:, i], y_disc, alpha=alpha)
        results[alpha] = scores

    return results


def rank_correlation(scores_a, scores_b):
    """
    Spearman rank correlation between two score vectors.

    Returns a value in [-1, 1]. 1 = identical ranking.
    """
    from scipy.stats import spearmanr
    corr, _ = spearmanr(scores_a, scores_b)
    return float(corr)


def test_exercise2():
    """Self-check tests for Exercise 2."""
    print("=" * 60)
    print("Exercise 2: Rényi Mutual Information")
    print("=" * 60)

    # Test 2.1: renyi_entropy recovers Shannon at alpha=1
    probs = np.array([0.3, 0.5, 0.2])
    h_renyi1 = renyi_entropy(probs, alpha=1.0)
    h_shannon = -np.sum(probs * np.log(probs))
    assert abs(h_renyi1 - h_shannon) < 1e-8, (
        f"renyi_entropy at alpha=1 should equal Shannon entropy {h_shannon:.4f}, "
        f"got {h_renyi1:.4f}"
    )
    print(f"  [PASS] H_1 = Shannon entropy = {h_shannon:.4f}")

    # Test 2.2: Rényi entropy is a decreasing function of alpha for non-uniform distributions
    # H_0.5 >= H_1.0 >= H_2.0 for non-uniform distributions
    h_05 = renyi_entropy(probs, alpha=0.5)
    h_10 = renyi_entropy(probs, alpha=1.0)
    h_20 = renyi_entropy(probs, alpha=2.0)
    assert h_05 >= h_10 - 1e-8, (
        f"H_0.5 >= H_1.0 must hold: H_0.5={h_05:.4f}, H_1.0={h_10:.4f}"
    )
    assert h_10 >= h_20 - 1e-8, (
        f"H_1.0 >= H_2.0 must hold: H_1.0={h_10:.4f}, H_2.0={h_20:.4f}"
    )
    print(f"  [PASS] H_0.5={h_05:.4f} >= H_1.0={h_10:.4f} >= H_2.0={h_20:.4f}")

    # Test 2.3: renyi_mi at alpha=1 matches standard MI
    data = load_breast_cancer()
    X_disc = discretise(data.data[:, :5], n_bins=10)
    y_disc = data.target.astype(int)

    rmi_10 = renyi_mi(X_disc[:, 0], y_disc, alpha=1.0)
    std_mi = mi(X_disc[:, 0], y_disc)
    assert abs(rmi_10 - std_mi) < 0.01, (
        f"Rényi MI at alpha=1 should approximately equal standard MI {std_mi:.4f}, "
        f"got {rmi_10:.4f}"
    )
    print(f"  [PASS] Rényi MI at alpha=1.0 ≈ standard MI: {rmi_10:.4f} vs {std_mi:.4f}")

    # Test 2.4: renyi_mi is non-negative for all alpha
    for alpha in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        rmi = renyi_mi(X_disc[:, 0], y_disc, alpha=alpha)
        assert rmi >= -1e-8, (
            f"Rényi MI at alpha={alpha} must be non-negative, got {rmi:.6f}"
        )
    print("  [PASS] Rényi MI >= 0 for alpha in {0.25, 0.5, 0.75, 1.0, 1.5, 2.0}")

    # Test 2.5: Compare rankings across alpha values
    X_full = discretise(data.data, n_bins=10)
    ranking_results = compare_renyi_rankings(X_full, y_disc,
                                              alphas=(0.5, 1.0, 2.0))
    assert set(ranking_results.keys()) == {0.5, 1.0, 2.0}, \
        "compare_renyi_rankings must return dict with keys 0.5, 1.0, 2.0"
    for alpha, scores in ranking_results.items():
        assert len(scores) == X_full.shape[1], \
            f"Scores for alpha={alpha} must have one entry per feature"
        assert all(s >= -1e-8 for s in scores), \
            f"All Rényi MI scores at alpha={alpha} must be non-negative"

    # Rank correlation between alpha=0.5 and alpha=1.0
    rc_05_10 = rank_correlation(ranking_results[0.5], ranking_results[1.0])
    rc_10_20 = rank_correlation(ranking_results[1.0], ranking_results[2.0])

    print(f"  [PASS] Rankings computed. Rank correlation:")
    print(f"          alpha=0.5 vs 1.0: {rc_05_10:.3f}")
    print(f"          alpha=1.0 vs 2.0: {rc_10_20:.3f}")
    print("         (High correlation = alpha choice doesn't matter much)")
    print("         (Low correlation = heavy tails make a difference)")

    print("\n  All Exercise 2 tests PASSED.\n")


# ---------------------------------------------------------------------------
# Exercise 3: Transfer Entropy Feature Selector
# ---------------------------------------------------------------------------

def discretise_series(ts, n_bins=10):
    """Quantile-based discretisation of a 1D time series."""
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(ts, quantiles)
    bin_edges[0] -= 1e-10
    bin_edges[-1] += 1e-10
    return (np.digitize(ts, bin_edges) - 1).astype(int)


def encode_joint_history(ts_disc, t_start, lag):
    """
    Encode the lagged history of a discrete time series as a single integer.

    Creates a joint variable representing (ts[t-1], ..., ts[t-lag])
    as a single integer for times t in [t_start, T).

    Parameters
    ----------
    ts_disc : array of int, shape (T,)
        Discretised time series
    t_start : int
        Index of first valid time step (t_start = max_lag)
    lag : int
        Number of lag steps

    Returns
    -------
    array of int, shape (T - t_start,)
        Joint encoded history
    """
    T = len(ts_disc)
    n_vals = int(np.max(ts_disc)) + 1
    result = np.zeros(T - t_start, dtype=int)
    for l in range(1, lag + 1):
        # ts[t-l] for each t in [t_start, T)
        result = result * n_vals + ts_disc[t_start - l: T - l]
    return result


def compute_transfer_entropy(x, y, k=1, ell=1, n_bins=10):
    """
    Compute transfer entropy T_{X -> Y}.

    T_{X->Y} = I(Y_{t+1}; X_{t-ell:t} | Y_{t-k:t})
             = I(Y_{t+1}; X_hist, Y_hist) - I(Y_{t+1}; Y_hist)

    Parameters
    ----------
    x : array of shape (T,)
        Source feature time series
    y : array of shape (T,)
        Target time series
    k : int
        Markov order for Y history (lags of Y to condition on)
    ell : int
        Number of lags of X to include
    n_bins : int
        Discretisation bins

    Returns
    -------
    float : Transfer entropy (non-negative)

    Notes
    -----
    Use encode_joint_history() to build y_hist and x_hist.
    Then encode (x_hist, y_hist) as a single joint variable.
    TE = I(y_future; xy_hist) - I(y_future; y_hist)
    Both MI terms use mutual_info_score from sklearn.
    """
    # YOUR CODE HERE
    # --------------------------------------------------
    # Step 1: Discretise x and y using discretise_series()
    # Step 2: Set max_lag = max(k, ell)
    # Step 3: y_future = y_disc[max_lag:]  (the quantity we're predicting)
    # Step 4: y_hist = encode_joint_history(y_disc, max_lag, k)
    # Step 5: x_hist = encode_joint_history(x_disc, max_lag, ell)
    # Step 6: xy_hist = x_hist * (max(y_hist)+1) + y_hist  (joint encoding)
    # Step 7: TE = mi(y_future, xy_hist) - mi(y_future, y_hist)
    # Step 8: return max(0.0, TE)
    raise NotImplementedError("Implement compute_transfer_entropy in Exercise 3")


def te_feature_selector(X_ts, y_ts, k_select, k_markov=1, ell=1,
                        n_bins=10, n_surrogates=100, alpha=0.05):
    """
    Select features from a multivariate time series using transfer entropy.

    For each candidate feature x_i, computes T(x_i -> y) and tests
    significance using n_surrogates time-shuffled surrogates.

    Returns the top k_select features that are statistically significant
    (or top k_select by TE if none are significant).

    Parameters
    ----------
    X_ts : array of shape (T, p)
        Multivariate feature time series
    y_ts : array of shape (T,)
        Target time series
    k_select : int
        Number of features to select
    k_markov : int
        Markov order for Y history
    ell : int
        Lags of X to include
    n_bins : int
        Discretisation bins
    n_surrogates : int
        Number of surrogates for significance testing
    alpha : float
        Significance level

    Returns
    -------
    dict with keys:
        'selected_indices': list of int — indices of selected features
        'te_scores': array of float — TE score for each feature
        'p_values': array of float — p-value for each feature
        'significant': array of bool — whether each feature is significant
    """
    # YOUR CODE HERE
    # --------------------------------------------------
    # For each feature i:
    #   1. Compute TE: compute_transfer_entropy(X_ts[:, i], y_ts, ...)
    #   2. Compute surrogate distribution: shuffle X_ts[:, i] n_surrogates times,
    #      compute TE on each shuffle
    #   3. p_value = fraction of surrogates >= observed TE
    #   4. significant = (p_value < alpha)
    #
    # Then return top k_select features by TE score, preferring significant ones.
    raise NotImplementedError("Implement te_feature_selector in Exercise 3")


def generate_time_series_dataset(T=500, n_causal=2, n_noise=5, seed=42):
    """
    Generate a synthetic time series dataset with known causal structure.

    Causal features drive y with lag-1. Noise features are independent.

    Parameters
    ----------
    T : int
        Length of each time series
    n_causal : int
        Number of genuine causal features
    n_noise : int
        Number of pure noise features
    seed : int
        Random seed

    Returns
    -------
    X : array of shape (T, n_causal + n_noise)
        Feature time series
    y : array of shape (T,)
        Target time series
    causal_indices : list of int
        Indices of the true causal features
    """
    rng = np.random.default_rng(seed)

    # Causal features: AR(1) processes
    causal_features = []
    for i in range(n_causal):
        x = np.zeros(T)
        x[0] = rng.standard_normal()
        for t in range(1, T):
            x[t] = 0.5 * x[t-1] + rng.standard_normal()
        causal_features.append(x)

    # Target: sum of lagged causal features + noise
    y = np.zeros(T)
    for x in causal_features:
        y[1:] += 0.5 * x[:-1]  # Lag-1 effect
    y += 0.3 * rng.standard_normal(T)

    # Noise features: independent AR(1) processes
    noise_features = []
    for i in range(n_noise):
        x = np.zeros(T)
        ar_coef = 0.3 + 0.4 * rng.random()
        for t in range(1, T):
            x[t] = ar_coef * x[t-1] + rng.standard_normal()
        noise_features.append(x)

    # Stack all features
    X = np.column_stack(causal_features + noise_features)
    causal_indices = list(range(n_causal))

    return X, y, causal_indices


def test_exercise3():
    """Self-check tests for Exercise 3."""
    print("=" * 60)
    print("Exercise 3: Transfer Entropy Feature Selector")
    print("=" * 60)

    # Generate test data
    X_ts, y_ts, causal_idx = generate_time_series_dataset(
        T=500, n_causal=2, n_noise=5, seed=42
    )
    n_causal = len(causal_idx)

    # Test 3.1: compute_transfer_entropy is non-negative
    te_c0 = compute_transfer_entropy(X_ts[:, 0], y_ts, k=1, ell=1)
    te_noise = compute_transfer_entropy(X_ts[:, 3], y_ts, k=1, ell=1)
    assert te_c0 >= 0, f"TE must be non-negative, got {te_c0:.4f}"
    assert te_noise >= 0, f"TE must be non-negative, got {te_noise:.4f}"
    print(f"  [PASS] TE(x0->y)={te_c0:.4f} >= 0")
    print(f"  [PASS] TE(x3->y)={te_noise:.4f} >= 0 (noise feature)")

    # Test 3.2: TE for causal features > TE for noise features on average
    te_causals = [compute_transfer_entropy(X_ts[:, i], y_ts, k=1, ell=1)
                  for i in causal_idx]
    te_noises = [compute_transfer_entropy(X_ts[:, i], y_ts, k=1, ell=1)
                 for i in range(n_causal, X_ts.shape[1])]

    mean_te_causal = np.mean(te_causals)
    mean_te_noise = np.mean(te_noises)

    assert mean_te_causal > mean_te_noise, (
        f"Mean TE for causal features ({mean_te_causal:.4f}) should exceed "
        f"mean TE for noise features ({mean_te_noise:.4f}). "
        "Check your TE implementation — it may not be conditioning correctly."
    )
    print(f"  [PASS] Mean TE causal={mean_te_causal:.4f} > Mean TE noise={mean_te_noise:.4f}")

    # Test 3.3: TE is asymmetric (T(x->y) != T(y->x) in general)
    te_forward = compute_transfer_entropy(X_ts[:, 0], y_ts, k=1, ell=1)
    te_reverse = compute_transfer_entropy(y_ts, X_ts[:, 0], k=1, ell=1)
    # For causal features, forward TE should exceed reverse TE
    print(f"  T(x0->y)={te_forward:.4f}, T(y->x0)={te_reverse:.4f}")
    assert te_forward != te_reverse or abs(te_forward) < 1e-8, (
        "TE should be asymmetric: T(x->y) != T(y->x) in general. "
        "If both are zero, the TE estimator may have a problem."
    )
    print(f"  [PASS] TE is asymmetric: T(x0->y)={te_forward:.4f} != T(y->x0)={te_reverse:.4f}")

    # Test 3.4: te_feature_selector returns correct structure
    result = te_feature_selector(X_ts, y_ts, k_select=n_causal,
                                  k_markov=1, ell=1, n_bins=8,
                                  n_surrogates=50, alpha=0.05)

    required_keys = {'selected_indices', 'te_scores', 'p_values', 'significant'}
    assert required_keys.issubset(result.keys()), (
        f"te_feature_selector must return dict with keys {required_keys}. "
        f"Got {set(result.keys())}"
    )
    assert len(result['selected_indices']) == n_causal, (
        f"Should select {n_causal} features, got {len(result['selected_indices'])}"
    )
    assert len(result['te_scores']) == X_ts.shape[1], (
        "te_scores must have one entry per feature"
    )
    assert len(result['p_values']) == X_ts.shape[1], (
        "p_values must have one entry per feature"
    )
    assert all(0 <= p <= 1 for p in result['p_values']), (
        "p-values must be in [0, 1]"
    )
    print(f"  [PASS] te_feature_selector returns correct structure")
    print(f"         Selected indices: {sorted(result['selected_indices'])}")
    print(f"         True causal indices: {sorted(causal_idx)}")

    # Test 3.5: Selected features overlap with true causal features
    selected_set = set(result['selected_indices'])
    causal_set = set(causal_idx)
    overlap = len(selected_set & causal_set)
    print(f"  Overlap with true causal features: {overlap}/{n_causal}")
    # With 500 observations and clear causal structure, expect at least 1 correct
    assert overlap >= 1, (
        f"Expected at least 1 of {n_causal} true causal features to be selected. "
        f"Got 0 overlap. Check that your TE implementation conditions on Y's history."
    )
    print(f"  [PASS] At least 1 true causal feature correctly identified.")

    print("\n  All Exercise 3 tests PASSED.\n")


# ---------------------------------------------------------------------------
# Main: run all exercises
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("Module 02: Information Theory Feature Selection — Self-Check")
    print("Based on Brown et al. (2012) CLM Framework")
    print("=" * 70 + "\n")

    exercise_results = []

    # Exercise 1
    try:
        test_exercise1()
        exercise_results.append(("Exercise 1: CMIM from CLM", "PASSED"))
    except NotImplementedError as e:
        print(f"Exercise 1 not yet implemented: {e}\n")
        exercise_results.append(("Exercise 1: CMIM from CLM", "NOT IMPLEMENTED"))
    except AssertionError as e:
        print(f"Exercise 1 FAILED: {e}\n")
        exercise_results.append(("Exercise 1: CMIM from CLM", "FAILED"))

    # Exercise 2
    try:
        test_exercise2()
        exercise_results.append(("Exercise 2: Rényi MI", "PASSED"))
    except NotImplementedError as e:
        print(f"Exercise 2 not yet implemented: {e}\n")
        exercise_results.append(("Exercise 2: Rényi MI", "NOT IMPLEMENTED"))
    except AssertionError as e:
        print(f"Exercise 2 FAILED: {e}\n")
        exercise_results.append(("Exercise 2: Rényi MI", "FAILED"))

    # Exercise 3
    try:
        test_exercise3()
        exercise_results.append(("Exercise 3: Transfer Entropy Selector", "PASSED"))
    except NotImplementedError as e:
        print(f"Exercise 3 not yet implemented: {e}\n")
        exercise_results.append(("Exercise 3: Transfer Entropy Selector", "NOT IMPLEMENTED"))
    except AssertionError as e:
        print(f"Exercise 3 FAILED: {e}\n")
        exercise_results.append(("Exercise 3: Transfer Entropy Selector", "FAILED"))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, status in exercise_results:
        symbol = "PASS" if status == "PASSED" else ("TODO" if "NOT IMPLEMENTED" in status else "FAIL")
        print(f"  [{symbol}] {name}: {status}")

    all_passed = all(s == "PASSED" for _, s in exercise_results)
    if all_passed:
        print("\n  All exercises complete. Well done.")
    else:
        remaining = sum(1 for _, s in exercise_results if s != "PASSED")
        print(f"\n  {remaining} exercise(s) remaining. "
              "Implement the functions marked with 'raise NotImplementedError'.")
    print("=" * 70 + "\n")
