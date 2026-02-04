"""
Module 04: Advanced Extensions
Self-Check Exercises (Ungraded)

Complete these exercises to test your understanding.
Run: python exercises.py
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# EXERCISE 1: Rolling Window Estimation
# ============================================================================

def exercise_1():
    """
    Implement rolling-window DFM estimation to detect parameter changes.
    """
    print("="*70)
    print("EXERCISE 1: Rolling Window Estimation")
    print("="*70)

    # Generate data with structural break
    np.random.seed(42)
    T = 200
    dates = pd.date_range('2005-01-01', periods=T, freq='M')

    # Factor with break at t=100
    factor = np.random.randn(T)
    for t in range(1, T):
        factor[t] = 0.7 * factor[t-1] + np.random.randn()

    # Data with loading break
    loading_pre = 0.8
    loading_post = 0.4
    X = np.zeros((T, 3))
    for t in range(T):
        loading = loading_pre if t < 100 else loading_post
        X[t, 0] = loading * factor[t] + 0.3 * np.random.randn()
        X[t, 1] = 0.6 * factor[t] + 0.3 * np.random.randn()
        X[t, 2] = 0.5 * factor[t] + 0.3 * np.random.randn()

    data = pd.DataFrame(X, index=dates, columns=['Var1', 'Var2', 'Var3'])

    # YOUR CODE HERE
    def rolling_window_loadings(data, window_size=60, k_factors=1):
        """
        TODO: Estimate DFM on rolling windows and extract loading evolution.

        Returns:
        -------
        loading_history : list of arrays
            Loading matrices for each window
        dates : list
            Dates for each estimation
        """
        # SOLUTION (hidden - try first!)
        loading_history = []
        dates_history = []

        for t in range(window_size, len(data)):
            window = data.iloc[t-window_size:t]

            try:
                dfm = DynamicFactor(window, k_factors=k_factors, factor_order=1)
                dfm_res = dfm.fit(maxiter=500, disp=False)

                loadings = dfm_res.params['loading']
                loading_history.append(loadings)
                dates_history.append(data.index[t-1])
            except:
                continue

        return loading_history, dates_history

    # Test your function
    loadings, dates = rolling_window_loadings(data, window_size=60)

    print(f"\n✓ Estimated {len(loadings)} rolling windows")

    # Check if structural break detected
    loading_var1_early = np.mean([l[0] for l in loadings[:20]])
    loading_var1_late = np.mean([l[0] for l in loadings[-20:]])

    print(f"\nVar1 loading evolution:")
    print(f"  Early period: {loading_var1_early:.3f}")
    print(f"  Late period: {loading_var1_late:.3f}")
    print(f"  Change: {loading_var1_late - loading_var1_early:.3f}")

    if abs(loading_var1_late - loading_var1_early) > 0.2:
        print("\n✓ PASS: Structural break detected!")
    else:
        print("\n✗ Break not detected - check window size")

    print()


# ============================================================================
# EXERCISE 2: Two-Step Estimation
# ============================================================================

def exercise_2():
    """
    Implement two-step DFM estimation (PCA + VAR).
    """
    print("="*70)
    print("EXERCISE 2: Two-Step Estimation")
    print("="*70)

    # Generate large dataset
    np.random.seed(123)
    T, N, r = 150, 50, 3

    # True factors
    F_true = np.random.randn(T, r)
    for t in range(1, T):
        F_true[t] = 0.7 * F_true[t-1] + 0.3 * np.random.randn(r)

    # Loadings and data
    Lambda_true = np.random.randn(N, r) * 0.5
    X = F_true @ Lambda_true.T + 0.4 * np.random.randn(T, N)

    # YOUR CODE HERE
    def two_step_estimation(X, k_factors=3):
        """
        TODO: Implement two-step estimation

        Step 1: PCA to get factors and loadings
        Step 2: Estimate VAR on factors

        Returns:
        -------
        factors : array (T, r)
        loadings : array (N, r)
        var_coef : array (r, r)
            VAR(1) coefficient matrix
        """
        # SOLUTION (hidden - try first!)
        from sklearn.decomposition import PCA
        from statsmodels.tsa.api import VAR

        # Step 1: PCA
        pca = PCA(n_components=k_factors)
        factors = pca.fit_transform(X)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

        # Step 2: VAR
        var_model = VAR(factors)
        var_res = var_model.fit(maxlags=1, ic=None)
        var_coef = var_res.params[1:].T

        return factors, loadings, var_coef

    # Test your function
    factors_est, loadings_est, var_coef = two_step_estimation(X, k_factors=r)

    print(f"\n✓ Two-step estimation complete")
    print(f"  Factors shape: {factors_est.shape}")
    print(f"  Loadings shape: {loadings_est.shape}")
    print(f"  VAR coefficients:\n{var_coef.round(2)}")

    # Check factor correlation (account for sign/rotation indeterminacy)
    from scipy.linalg import orthogonal_procrustes
    R, _ = orthogonal_procrustes(factors_est, F_true)
    factors_aligned = factors_est @ R

    corr = np.corrcoef(factors_aligned[:, 0], F_true[:, 0])[0, 1]
    print(f"\n  Factor 1 correlation with truth: {corr:.3f}")

    if corr > 0.7:
        print("\n✓ PASS: Factors reasonably estimated!")
    else:
        print("\n✗ Check your PCA implementation")

    print()


# ============================================================================
# EXERCISE 3: Sparse Loadings with LASSO
# ============================================================================

def exercise_3():
    """
    Implement sparse factor model with LASSO-regularized loadings.
    """
    print("="*70)
    print("EXERCISE 3: Sparse Loadings")
    print("="*70)

    # Generate sparse data
    np.random.seed(42)
    T, N, r = 100, 20, 2

    F = np.random.randn(T, r)

    # Sparse loadings: only first 5 variables load on each factor
    Lambda_true = np.zeros((N, r))
    Lambda_true[:5, 0] = np.random.randn(5) * 0.8
    Lambda_true[5:10, 1] = np.random.randn(5) * 0.8

    X = F @ Lambda_true.T + 0.3 * np.random.randn(T, N)

    # YOUR CODE HERE
    def sparse_factor_estimation(X, k_factors=2, alpha=0.1):
        """
        TODO: Estimate factor loadings with LASSO sparsity.

        Returns:
        -------
        factors : array (T, r)
        sparse_loadings : array (N, r)
            Many entries should be zero
        """
        # SOLUTION (hidden - try first!)
        from sklearn.decomposition import PCA
        from sklearn.linear_model import Lasso

        # Initialize factors with PCA
        pca = PCA(n_components=k_factors)
        factors = pca.fit_transform(X)

        # LASSO for each variable
        sparse_loadings = np.zeros((X.shape[1], k_factors))

        for i in range(X.shape[1]):
            lasso = Lasso(alpha=alpha, fit_intercept=False)
            lasso.fit(factors, X[:, i])
            sparse_loadings[i] = lasso.coef_

        return factors, sparse_loadings

    # Test your function
    factors, loadings_sparse = sparse_factor_estimation(X, k_factors=r, alpha=0.05)

    # Measure sparsity
    sparsity = (loadings_sparse == 0).sum() / loadings_sparse.size * 100

    print(f"\n✓ Sparse estimation complete")
    print(f"  Sparsity: {sparsity:.1f}% of loadings are zero")
    print(f"\nSparse loading matrix:")
    print(loadings_sparse.round(2))

    # Check if recovered true sparsity pattern
    true_nonzero = np.sum(Lambda_true != 0)
    est_nonzero = np.sum(loadings_sparse != 0)

    print(f"\n  True nonzero loadings: {true_nonzero}")
    print(f"  Estimated nonzero: {est_nonzero}")

    if abs(true_nonzero - est_nonzero) <= 5:
        print("\n✓ PASS: Sparsity pattern approximately recovered!")
    else:
        print("\n✗ Try adjusting alpha parameter")

    print()


# ============================================================================
# EXERCISE 4: Mixed-Frequency Aggregation
# ============================================================================

def exercise_4():
    """
    Implement temporal aggregation for mixed-frequency models.
    """
    print("="*70)
    print("EXERCISE 4: Mixed-Frequency Aggregation")
    print("="*70)

    # Monthly factors
    np.random.seed(42)
    dates_monthly = pd.date_range('2020-01-01', periods=24, freq='M')
    factors_monthly = pd.DataFrame({
        'Factor1': np.random.randn(24),
        'Factor2': np.random.randn(24)
    }, index=dates_monthly)

    # YOUR CODE HERE
    def aggregate_to_quarterly(factors_monthly, method='average'):
        """
        TODO: Aggregate monthly factors to quarterly.

        Parameters:
        ----------
        method : str
            'average' (for flow variables) or 'last' (for stock variables)

        Returns:
        -------
        factors_quarterly : pd.DataFrame
        """
        # SOLUTION (hidden - try first!)
        if method == 'average':
            factors_quarterly = factors_monthly.resample('Q').mean()
        elif method == 'last':
            factors_quarterly = factors_monthly.resample('Q').last()
        else:
            raise ValueError("method must be 'average' or 'last'")

        return factors_quarterly

    # Test your function
    factors_q_avg = aggregate_to_quarterly(factors_monthly, method='average')
    factors_q_last = aggregate_to_quarterly(factors_monthly, method='last')

    print(f"\n✓ Aggregation complete")
    print(f"  Monthly observations: {len(factors_monthly)}")
    print(f"  Quarterly observations: {len(factors_q_avg)}")

    print(f"\nAveraged quarterly factors:")
    print(factors_q_avg)

    print(f"\nLast-observation quarterly factors:")
    print(factors_q_last)

    # Validation
    # Q1 2020 average should equal mean of Jan, Feb, Mar
    q1_manual = factors_monthly.iloc[:3].mean()
    q1_function = factors_q_avg.iloc[0]

    if np.allclose(q1_manual, q1_function):
        print("\n✓ PASS: Aggregation correct!")
    else:
        print("\n✗ Check your aggregation logic")

    print()


# ============================================================================
# EXERCISE 5: Factor Number Selection
# ============================================================================

def exercise_5():
    """
    Implement information criteria for selecting number of factors.
    """
    print("="*70)
    print("EXERCISE 5: Factor Number Selection")
    print("="*70)

    # Generate data with r=3 true factors
    np.random.seed(123)
    T, N, r_true = 150, 30, 3

    F_true = np.random.randn(T, r_true)
    Lambda_true = np.random.randn(N, r_true)
    X = F_true @ Lambda_true.T + 0.5 * np.random.randn(T, N)

    # YOUR CODE HERE
    def select_factors_bic(X, max_factors=10):
        """
        TODO: Select number of factors using BIC.

        Returns:
        -------
        optimal_r : int
            Number of factors that minimizes BIC
        bic_values : dict
            BIC for each number of factors
        """
        # SOLUTION (hidden - try first!)
        from sklearn.decomposition import PCA

        T, N = X.shape
        bic_values = {}

        for r in range(1, max_factors + 1):
            pca = PCA(n_components=r)
            pca.fit(X)

            # Reconstruct
            X_recon = pca.transform(X) @ pca.components_

            # MSE
            mse = np.mean((X - X_recon)**2)

            # Number of parameters
            n_params = N * r + r**2  # Loadings + VAR coefficients

            # BIC
            bic = T * np.log(mse) + n_params * np.log(T)
            bic_values[r] = bic

        optimal_r = min(bic_values, key=bic_values.get)

        return optimal_r, bic_values

    # Test your function
    r_optimal, bic_vals = select_factors_bic(X, max_factors=8)

    print(f"\n✓ Factor selection complete")
    print(f"\nBIC values:")
    for r, bic in bic_vals.items():
        marker = " ← OPTIMAL" if r == r_optimal else ""
        print(f"  {r} factors: BIC = {bic:.1f}{marker}")

    print(f"\n  Selected: {r_optimal} factors")
    print(f"  True: {r_true} factors")

    if r_optimal == r_true:
        print("\n✓ PASS: Correct number of factors selected!")
    elif abs(r_optimal - r_true) <= 1:
        print("\n✓ PASS: Within 1 of true value (acceptable)")
    else:
        print("\n✗ Selection off - check BIC calculation")

    print()


# ============================================================================
# Run All Exercises
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*13 + "MODULE 04: SELF-CHECK EXERCISES" + " "*24 + "║")
    print("╚" + "="*68 + "╝")
    print("\nComplete these exercises to test your understanding.")
    print("Solutions are provided after each exercise.\n")

    try:
        exercise_1()
    except Exception as e:
        print(f"Exercise 1 error: {e}\n")

    try:
        exercise_2()
    except Exception as e:
        print(f"Exercise 2 error: {e}\n")

    try:
        exercise_3()
    except Exception as e:
        print(f"Exercise 3 error: {e}\n")

    try:
        exercise_4()
    except Exception as e:
        print(f"Exercise 4 error: {e}\n")

    try:
        exercise_5()
    except Exception as e:
        print(f"Exercise 5 error: {e}\n")

    print("="*70)
    print("All exercises complete! Review any errors above.")
    print("="*70)
