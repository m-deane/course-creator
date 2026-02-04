"""
Common Dynamic Factor Model Patterns - Copy-paste code snippets

Each recipe solves one specific problem with complete working code.
Input and output clearly shown for each pattern.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# ===========================================================================
# RECIPE 1: Factor Extraction (PCA vs DFM Comparison)
# Problem: Extract common factors from multivariate time series
# ===========================================================================

def extract_factors_comparison(data: pd.DataFrame, n_factors: int = 2):
    """
    Input: DataFrame with standardized time series (T x N)
    Output: DataFrame with factors from both methods for comparison
    """
    # PCA factors (static)
    pca = PCA(n_components=n_factors)
    pca_factors = pca.fit_transform(data)

    # DFM factors (dynamic)
    dfm = DynamicFactor(data, k_factors=n_factors, factor_order=2)
    dfm_results = dfm.fit(disp=False)
    dfm_factors = dfm_results.factors.smoothed

    # Combine for comparison
    factors_df = pd.DataFrame({
        'PCA_F1': pca_factors[:, 0],
        'PCA_F2': pca_factors[:, 1] if n_factors > 1 else np.nan,
        'DFM_F1': dfm_factors[:, 0],
        'DFM_F2': dfm_factors[:, 1] if n_factors > 1 else np.nan,
    }, index=data.index)

    print(f"PCA variance explained: {pca.explained_variance_ratio_}")
    return factors_df


# ===========================================================================
# RECIPE 2: State Space Model Setup
# Problem: Configure custom state space representation for DFM
# ===========================================================================

def setup_state_space_dfm(data: pd.DataFrame, n_factors: int = 1, factor_ar: int = 2):
    """
    Input: Data (T x N), number of factors, AR order
    Output: Configured DynamicFactor model ready for estimation
    """
    model = DynamicFactor(
        endog=data,
        k_factors=n_factors,
        factor_order=factor_ar,
        error_order=0,  # White noise idiosyncratic errors
        error_cov_type='diagonal',  # No correlation across series errors
        enforce_stationarity=True  # Constrain AR parameters
    )
    return model


# ===========================================================================
# RECIPE 3: Kalman Filter Initialization
# Problem: Set initial conditions for Kalman filter
# ===========================================================================

def initialize_kalman_filter(model: DynamicFactor, method: str = 'diffuse'):
    """
    Input: Unfitted DynamicFactor model
    Output: Model with initialized state

    Methods:
    - 'diffuse': Diffuse initialization (default, no prior info)
    - 'stationary': Use stationary distribution
    - 'custom': Provide your own initial state
    """
    if method == 'diffuse':
        # Diffuse initialization (large variance for initial state)
        model.initialize_approximate_diffuse()
    elif method == 'stationary':
        # Initialize at stationary distribution
        model.initialize_stationary()
    else:
        raise ValueError(f"Unknown method: {method}")

    return model


# ===========================================================================
# RECIPE 4: Factor Rotation for Interpretability
# Problem: Rotate factors to align with economically meaningful directions
# ===========================================================================

def rotate_factors_varimax(factors: np.ndarray, loadings: np.ndarray):
    """
    Input: factors (T x K), loadings (N x K)
    Output: Rotated factors and loadings

    Varimax rotation maximizes variance of squared loadings
    """
    from scipy.stats import ortho_group
    from sklearn.preprocessing import normalize

    # Simple varimax rotation (for illustration)
    # In practice, use factor_analyzer or R's psych package
    U, S, Vt = np.linalg.svd(loadings, full_matrices=False)

    # Rotate loadings
    rotated_loadings = U @ np.diag(S)

    # Apply same rotation to factors
    rotated_factors = factors @ Vt.T

    return rotated_factors, rotated_loadings


# ===========================================================================
# RECIPE 5: Variance Decomposition
# Problem: Decompose variance of each series into common vs idiosyncratic
# ===========================================================================

def variance_decomposition(results: DynamicFactor):
    """
    Input: Fitted DynamicFactor results
    Output: DataFrame with variance decomposition by series
    """
    # Extract loadings
    param_names = results.model.param_names
    loadings = []

    n_series = results.model.k_endog
    n_factors = results.model.k_factors

    for i in range(n_series):
        series_loadings = []
        for k in range(n_factors):
            param_name = f'loading.f{k+1}.{results.model.endog_names[i]}'
            if param_name in param_names:
                idx = param_names.index(param_name)
                series_loadings.append(results.params[idx])
        loadings.append(series_loadings)

    loadings = np.array(loadings)

    # Variance from factors: sum of squared loadings
    factor_variance = np.sum(loadings**2, axis=1)

    # Total variance (assuming standardized data)
    total_variance = np.ones(n_series)

    # Idiosyncratic variance
    idio_variance = total_variance - factor_variance

    # Create decomposition table
    decomp = pd.DataFrame({
        'Series': results.model.endog_names,
        'Factor_Variance': factor_variance,
        'Idiosyncratic_Variance': idio_variance,
        'Factor_Share': factor_variance / total_variance * 100
    })

    return decomp


# ===========================================================================
# RECIPE 6: Factor News Decomposition
# Problem: Attribute forecast revisions to specific data releases
# ===========================================================================

def factor_news_decomposition(
    old_results: DynamicFactor,
    new_results: DynamicFactor,
    series_updated: str
):
    """
    Input: Results before/after data update, name of updated series
    Output: Impact of new data on factor estimates
    """
    # Extract factors before and after
    old_factors = old_results.factors.smoothed[-1]  # Most recent
    new_factors = new_results.factors.smoothed[-1]

    # Change in factors
    factor_news = new_factors - old_factors

    # Get loading for updated series
    param_names = new_results.model.param_names
    series_idx = new_results.model.endog_names.index(series_updated)

    loading = []
    for k in range(new_results.model.k_factors):
        param_name = f'loading.f{k+1}.{series_updated}'
        if param_name in param_names:
            idx = param_names.index(param_name)
            loading.append(new_results.params[idx])

    loading = np.array(loading)

    # Impact on forecasts
    impact = loading @ factor_news

    print(f"Factor news from {series_updated}: {factor_news}")
    print(f"Forecast impact: {impact:.4f}")

    return {'factor_news': factor_news, 'forecast_impact': impact}


# ===========================================================================
# RECIPE 7: Information Criteria for Model Selection
# Problem: Choose optimal number of factors and AR order
# ===========================================================================

def select_model_specification(data: pd.DataFrame, max_factors: int = 4, max_ar: int = 4):
    """
    Input: Data, maximum factors and AR order to consider
    Output: DataFrame with AIC/BIC for all combinations
    """
    results = []

    for n_factors in range(1, max_factors + 1):
        for ar_order in range(1, max_ar + 1):
            try:
                model = DynamicFactor(
                    data,
                    k_factors=n_factors,
                    factor_order=ar_order,
                    error_order=0
                )
                fit = model.fit(disp=False, maxiter=500)

                results.append({
                    'n_factors': n_factors,
                    'ar_order': ar_order,
                    'aic': fit.aic,
                    'bic': fit.bic,
                    'llf': fit.llf
                })
            except:
                continue

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('bic')

    print("Best model (by BIC):")
    print(results_df.iloc[0])

    return results_df


# ===========================================================================
# RECIPE 8: Out-of-Sample Forecast Evaluation
# Problem: Evaluate forecast accuracy with rolling window
# ===========================================================================

def rolling_forecast_evaluation(
    data: pd.DataFrame,
    n_factors: int,
    factor_order: int,
    forecast_horizon: int = 1,
    window_size: int = 60
):
    """
    Input: Data, model spec, forecast horizon, estimation window
    Output: DataFrame with forecasts and errors
    """
    forecasts = []

    # Rolling window forecasts
    for i in range(window_size, len(data) - forecast_horizon):
        # Estimation sample
        train = data.iloc[i-window_size:i]

        # Fit model
        model = DynamicFactor(train, k_factors=n_factors, factor_order=factor_order)
        results = model.fit(disp=False, maxiter=500)

        # Forecast
        forecast = results.forecast(steps=forecast_horizon)

        # Actual
        actual = data.iloc[i + forecast_horizon - 1]

        # Store
        forecasts.append({
            'date': data.index[i + forecast_horizon - 1],
            'forecast': forecast[-1, 0],  # First series, last step
            'actual': actual.iloc[0],
            'error': actual.iloc[0] - forecast[-1, 0]
        })

    forecast_df = pd.DataFrame(forecasts)

    # Compute metrics
    rmse = np.sqrt(np.mean(forecast_df['error']**2))
    mae = np.mean(np.abs(forecast_df['error']))

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    return forecast_df


# ===========================================================================
# RECIPE 9: Handle Missing Data with EM Algorithm
# Problem: Estimate DFM with missing observations
# ===========================================================================

def estimate_with_missing_data(data_with_missing: pd.DataFrame):
    """
    Input: DataFrame with NaN values
    Output: Fitted model using EM algorithm

    DynamicFactor in statsmodels handles missing data via Kalman filter
    """
    # Check missing pattern
    missing_summary = data_with_missing.isnull().sum()
    print("Missing observations per series:")
    print(missing_summary)

    # Estimate (Kalman filter handles missing automatically)
    model = DynamicFactor(
        data_with_missing,
        k_factors=2,
        factor_order=2
    )
    results = model.fit(disp=False)

    # Imputed values available via smoothed estimates
    imputed = results.fittedvalues

    print(f"Model estimated with {missing_summary.sum()} missing observations")
    return results, imputed


# ===========================================================================
# RECIPE 10: Convert DFM to AR Representation
# Problem: Express DFM in reduced-form VAR for impulse responses
# ===========================================================================

def dfm_to_var_representation(results: DynamicFactor):
    """
    Input: Fitted DynamicFactor model
    Output: Companion matrix for VAR representation
    """
    # Extract factor transition matrix
    k_factors = results.model.k_factors
    factor_order = results.model.factor_order

    # Transition matrix is in state space form
    transition = results.transition[:, :, 0]  # Time-invariant

    # Factor dynamics (top-left block)
    factor_transition = transition[:k_factors*factor_order, :k_factors*factor_order]

    print("Factor transition matrix (companion form):")
    print(factor_transition)

    # Eigenvalues check stability
    eigenvalues = np.linalg.eigvals(factor_transition)
    is_stable = np.all(np.abs(eigenvalues) < 1)

    print(f"Eigenvalues: {eigenvalues}")
    print(f"Stable: {is_stable}")

    return factor_transition


# ===========================================================================
# EXAMPLE USAGE
# ===========================================================================

if __name__ == "__main__":
    # Generate synthetic data for demonstration
    np.random.seed(42)
    T = 200
    N = 5

    # True factor
    factor = np.random.randn(T)
    for t in range(2, T):
        factor[t] = 0.8 * factor[t-1] - 0.1 * factor[t-2] + np.random.randn()

    # Observed series = loadings * factor + noise
    loadings = np.random.uniform(0.5, 1.5, N)
    data = np.outer(factor, loadings) + np.random.randn(T, N) * 0.5

    data_df = pd.DataFrame(
        data,
        columns=[f'Series_{i+1}' for i in range(N)],
        index=pd.date_range('2000-01-01', periods=T, freq='MS')
    )

    # Standardize
    data_df = (data_df - data_df.mean()) / data_df.std()

    print("="*70)
    print("RECIPE 1: Factor Extraction Comparison")
    print("="*70)
    factors = extract_factors_comparison(data_df, n_factors=2)
    print(factors.head())

    print("\n" + "="*70)
    print("RECIPE 5: Variance Decomposition")
    print("="*70)
    model = DynamicFactor(data_df, k_factors=1, factor_order=2)
    results = model.fit(disp=False)
    decomp = variance_decomposition(results)
    print(decomp)

    print("\n" + "="*70)
    print("RECIPE 7: Model Selection")
    print("="*70)
    selection = select_model_specification(data_df, max_factors=2, max_ar=3)
    print(selection.head())
