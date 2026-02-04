"""
Module 03: Applications - Nowcasting & Forecasting
Self-Check Exercises (Ungraded)

Complete these exercises to test your understanding.
Run: python exercises.py
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from sklearn.linear_regression import LinearRegression
from scipy import stats

# ============================================================================
# EXERCISE 1: Build a Complete Nowcasting Function
# ============================================================================

def exercise_1():
    """
    Build a nowcasting function that takes monthly data and produces GDP nowcast.

    Requirements:
    - Estimate DFM with 3 factors
    - Extract quarterly factors
    - Build bridge equation
    - Return nowcast + standard error
    """
    print("="*70)
    print("EXERCISE 1: Nowcasting Function")
    print("="*70)

    # Provided data
    np.random.seed(42)
    dates = pd.date_range('2000-01-01', '2023-12-01', freq='M')
    T = len(dates)

    factor = np.zeros(T)
    for t in range(1, T):
        factor[t] = 0.7 * factor[t-1] + np.random.randn()

    monthly_data = pd.DataFrame({
        'IP': 0.8 * factor + 0.5 * np.random.randn(T),
        'Employment': 0.7 * factor + 0.6 * np.random.randn(T),
        'Sales': 0.6 * factor + 0.7 * np.random.randn(T)
    }, index=dates)

    quarterly_factor = pd.Series(factor, index=dates).resample('Q').mean()
    gdp = 2.0 + 1.5 * quarterly_factor + 0.3 * np.random.randn(len(quarterly_factor))

    # YOUR CODE HERE
    def nowcast_gdp(monthly_data, gdp_history, n_factors=3):
        """
        TODO: Implement nowcasting function

        Returns:
        -------
        nowcast : float
        std_error : float
        """
        # Step 1: Estimate DFM
        # Step 2: Extract factors
        # Step 3: Bridge equation
        # Step 4: Produce nowcast

        # SOLUTION (hidden - try first!)
        dfm = DynamicFactor(monthly_data, k_factors=n_factors, factor_order=1)
        dfm_res = dfm.fit(maxiter=500, disp=False)

        factors_monthly = dfm_res.factors.filtered
        factors_quarterly = factors_monthly.resample('Q').mean()

        common_idx = factors_quarterly.index.intersection(gdp_history.index)
        X_bridge = factors_quarterly.loc[common_idx]
        y_bridge = gdp_history.loc[common_idx]

        bridge = LinearRegression()
        bridge.fit(X_bridge, y_bridge)

        current_factors = factors_monthly.iloc[-3:].mean().values.reshape(1, -1)
        nowcast = bridge.predict(current_factors)[0]

        residuals = y_bridge - bridge.predict(X_bridge)
        std_error = np.std(residuals)

        return nowcast, std_error

    # Test your function
    nowcast, se = nowcast_gdp(monthly_data, gdp)

    print(f"\nYour nowcast: {nowcast:.2f}% ± {1.96*se:.2f}%")
    print(f"Actual Q4 2023 GDP: {gdp.iloc[-1]:.2f}%")
    print(f"\nError: {abs(nowcast - gdp.iloc[-1]):.2f}pp")

    if abs(nowcast - gdp.iloc[-1]) < 1.0:
        print("✓ PASS: Nowcast within 1pp of actual!")
    else:
        print("✗ Check your implementation")

    print()


# ============================================================================
# EXERCISE 2: Diebold-Mariano Test Implementation
# ============================================================================

def exercise_2():
    """
    Implement the Diebold-Mariano test for comparing forecast accuracy.
    """
    print("="*70)
    print("EXERCISE 2: Diebold-Mariano Test")
    print("="*70)

    # Two competing forecasts
    np.random.seed(123)
    T = 100
    actuals = np.random.randn(T)

    # Model A: slightly better
    forecasts_A = actuals + 0.5 * np.random.randn(T)

    # Model B: slightly worse
    forecasts_B = actuals + 0.7 * np.random.randn(T)

    errors_A = actuals - forecasts_A
    errors_B = actuals - forecasts_B

    # YOUR CODE HERE
    def diebold_mariano_test(errors_a, errors_b, h=1):
        """
        TODO: Implement DM test

        Returns:
        -------
        dm_stat : float
            DM test statistic
        p_value : float
            Two-sided p-value
        """
        # SOLUTION (hidden - try first!)
        d = errors_a**2 - errors_b**2
        T = len(d)

        d_bar = np.mean(d)
        var_d = np.var(d, ddof=1) / T

        dm_stat = d_bar / np.sqrt(var_d)

        # Harvey correction
        dm_stat *= np.sqrt((T + 1 - 2*h + h*(h-1)/T) / T)

        p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

        return dm_stat, p_value

    # Test your function
    dm_stat, p_val = diebold_mariano_test(errors_A, errors_B)

    print(f"\nDM Statistic: {dm_stat:.3f}")
    print(f"P-value: {p_val:.4f}")

    if p_val < 0.05:
        winner = "Model A" if dm_stat < 0 else "Model B"
        print(f"✓ {winner} is significantly better (p < 0.05)")
    else:
        print("• No significant difference")

    # Check implementation
    expected_dm = d.mean() / np.sqrt(np.var(d, ddof=1) / T)
    if abs(dm_stat - expected_dm) < 0.1:
        print("\n✓ PASS: DM statistic correctly computed!")
    else:
        print("\n✗ Check your DM statistic calculation")

    print()


# ============================================================================
# EXERCISE 3: Kalman Filter with Missing Data
# ============================================================================

def exercise_3():
    """
    Implement one iteration of Kalman filter update with missing observations.
    """
    print("="*70)
    print("EXERCISE 3: Kalman Filter with Missing Data")
    print("="*70)

    # Setup
    np.random.seed(42)
    r, N = 2, 5

    F_pred = np.array([1.0, -0.5])  # Predicted factor
    P_pred = np.array([[1.2, 0.1], [0.1, 0.8]])  # Predicted covariance

    Lambda = np.random.randn(N, r)
    Sigma_e = 0.5 * np.eye(N)

    X_obs = np.array([2.3, np.nan, -1.1, np.nan, 0.8])  # Some missing

    # YOUR CODE HERE
    def kalman_update_missing(F_pred, P_pred, X_t, Lambda, Sigma_e):
        """
        TODO: Implement Kalman update with missing observations

        Returns:
        -------
        F_filt : array (r,)
            Updated factor estimate
        P_filt : array (r, r)
            Updated covariance
        """
        # SOLUTION (hidden - try first!)
        obs_mask = ~np.isnan(X_t)
        obs_idx = np.where(obs_mask)[0]

        if len(obs_idx) == 0:
            return F_pred, P_pred

        X_obs = X_t[obs_idx]
        Lambda_obs = Lambda[obs_idx, :]
        Sigma_obs = Sigma_e[np.ix_(obs_idx, obs_idx)]

        v = X_obs - Lambda_obs @ F_pred
        S = Lambda_obs @ P_pred @ Lambda_obs.T + Sigma_obs
        K = P_pred @ Lambda_obs.T @ np.linalg.inv(S)

        F_filt = F_pred + K @ v
        P_filt = (np.eye(len(F_pred)) - K @ Lambda_obs) @ P_pred

        return F_filt, P_filt

    # Test your function
    F_filt, P_filt = kalman_update_missing(F_pred, P_pred, X_obs, Lambda, Sigma_e)

    print(f"\nPredicted factor: {F_pred}")
    print(f"Updated factor: {F_filt}")
    print(f"\nObservations used: {np.sum(~np.isnan(X_obs))}/5")

    # Check dimensions
    if F_filt.shape == (r,) and P_filt.shape == (r, r):
        print("\n✓ PASS: Correct dimensions!")
    else:
        print("\n✗ Check your dimensions")

    # Check that update moved estimate
    if not np.allclose(F_filt, F_pred):
        print("✓ PASS: Factor estimate updated (not just prediction)!")
    else:
        print("✗ Factor didn't update - check your Kalman gain")

    print()


# ============================================================================
# EXERCISE 4: Forecast Evaluation Metrics
# ============================================================================

def exercise_4():
    """
    Compute multiple forecast evaluation metrics.
    """
    print("="*70)
    print("EXERCISE 4: Forecast Evaluation Metrics")
    print("="*70)

    # Provided forecasts and actuals
    np.random.seed(42)
    T = 50
    actuals = np.random.randn(T)
    forecasts = actuals + 0.5 * np.random.randn(T)

    # YOUR CODE HERE
    def compute_all_metrics(y_true, y_pred):
        """
        TODO: Compute MSE, MAE, RMSE, Bias, Directional Accuracy

        Returns:
        -------
        metrics : dict
        """
        # SOLUTION (hidden - try first!)
        errors = y_true - y_pred

        metrics = {
            'MSE': np.mean(errors**2),
            'RMSE': np.sqrt(np.mean(errors**2)),
            'MAE': np.mean(np.abs(errors)),
            'Bias': np.mean(errors),
            'Directional_Accuracy': np.mean(np.sign(y_pred) == np.sign(y_true)) * 100
        }

        return metrics

    # Test your function
    metrics = compute_all_metrics(actuals, forecasts)

    print("\nYour metrics:")
    for name, value in metrics.items():
        print(f"  {name:25s}: {value:.4f}")

    # Validation checks
    checks_passed = 0

    if 0 <= metrics['MSE'] <= 10:
        checks_passed += 1

    if metrics['RMSE'] == np.sqrt(metrics['MSE']):
        checks_passed += 1

    if 0 <= metrics['Directional_Accuracy'] <= 100:
        checks_passed += 1

    print(f"\n✓ Passed {checks_passed}/3 validation checks")

    if checks_passed == 3:
        print("✓ PASS: All metrics computed correctly!")
    else:
        print("✗ Check your metric calculations")

    print()


# ============================================================================
# EXERCISE 5: Publication Lag Simulation
# ============================================================================

def exercise_5():
    """
    Impose realistic publication lags on data.
    """
    print("="*70)
    print("EXERCISE 5: Publication Lag Simulation")
    print("="*70)

    # Create sample data
    dates = pd.date_range('2020-01-01', periods=24, freq='M')
    data = pd.DataFrame({
        'IP': np.random.randn(24),
        'Employment': np.random.randn(24),
        'Sales': np.random.randn(24)
    }, index=dates)

    # Publication lags (in days)
    lag_days = {'IP': 15, 'Employment': 5, 'Sales': 12}

    # YOUR CODE HERE
    def impose_lags(data_df, lag_dict):
        """
        TODO: Set last N observations to NaN based on publication lags

        Parameters:
        ----------
        data_df : pd.DataFrame
        lag_dict : dict {column: days_delay}

        Returns:
        -------
        data_ragged : pd.DataFrame with NaNs for delayed data
        """
        # SOLUTION (hidden - try first!)
        data_ragged = data_df.copy()

        for col, days in lag_dict.items():
            months_lag = max(0, days // 30)
            if months_lag > 0:
                data_ragged.iloc[-months_lag:, data_ragged.columns.get_loc(col)] = np.nan

        return data_ragged

    # Test your function
    data_ragged = impose_lags(data, lag_days)

    print("\nOriginal data (last 3 months):")
    print(data.tail(3).to_string())

    print("\nWith publication lags (last 3 months):")
    print(data_ragged.tail(3).to_string())

    # Check that some data is missing
    n_missing = data_ragged.isnull().sum().sum()

    if n_missing > 0:
        print(f"\n✓ PASS: {n_missing} observations marked as missing!")
    else:
        print("\n✗ No data marked as missing - check your implementation")

    print()


# ============================================================================
# Run All Exercises
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "MODULE 03: SELF-CHECK EXERCISES" + " "*22 + "║")
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
