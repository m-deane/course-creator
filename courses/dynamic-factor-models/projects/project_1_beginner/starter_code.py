"""
GDP Nowcasting Model - Starter Code
====================================

This file provides a working skeleton for building a real-time GDP nowcasting
system using dynamic factor models.

YOUR TASK: Fill in the TODOs to complete the implementation.

Run: python starter_code.py
Expected output: GDP nowcast with visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import warnings
warnings.filterwarnings('ignore')

# Try importing optional dependencies
try:
    from fredapi import Fred
    HAS_FRED = True
except ImportError:
    HAS_FRED = False
    print("Warning: fredapi not installed. Using sample data.")

from sklearn.decomposition import PCA
from statsmodels.tsa.api import VAR
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

# Configuration
CONFIG_FILE = "config.yaml"
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

# Create directories
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# SECTION 1: DATA PIPELINE
# ============================================================================

def load_config():
    """Load configuration from YAML file."""
    if Path(CONFIG_FILE).exists():
        with open(CONFIG_FILE, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Default configuration
        return {
            'fred': {'api_key': 'YOUR_API_KEY'},
            'model': {
                'n_factors': 3,
                'factor_lags': 2,
                'target': 'GDPC1',
            },
            'indicators': get_default_indicators()
        }


def get_default_indicators():
    """Return list of default FRED series for nowcasting."""
    return [
        # Real activity
        'INDPRO',      # Industrial Production Index
        'PAYEMS',      # Total Nonfarm Payrolls
        'HOUST',       # Housing Starts
        'RETAILX',     # Retail Sales Ex Auto
        'CUMFNS',      # Capacity Utilization

        # Prices
        'CPIAUCSL',    # CPI All Urban
        'PCEPI',       # PCE Price Index
        'PPIFGS',      # PPI Finished Goods

        # Financial
        'GS10',        # 10-Year Treasury
        'FEDFUNDS',    # Federal Funds Rate
        'SP500',       # S&P 500

        # Labor market
        'UNRATE',      # Unemployment Rate
        'AWHMAN',      # Average Weekly Hours Manufacturing
        'ICSA',        # Initial Jobless Claims
    ]


def fetch_fred_data(api_key, series_list, start_date='2000-01-01'):
    """
    Fetch data from FRED API.

    Parameters:
    -----------
    api_key : str
        FRED API key
    series_list : list
        List of FRED series codes
    start_date : str
        Start date in 'YYYY-MM-DD' format

    Returns:
    --------
    pd.DataFrame
        Data with dates as index, series as columns
    """
    if not HAS_FRED:
        print("Loading sample data (fredapi not available)")
        return load_sample_data()

    # TODO: Implement FRED data fetching
    # 1. Initialize Fred client with API key
    # 2. Loop over series_list and fetch each series
    # 3. Combine into single DataFrame
    # 4. Handle missing values and mismatched frequencies

    fred = Fred(api_key=api_key)
    data = pd.DataFrame()

    print(f"Fetching {len(series_list)} series from FRED...")
    for series in series_list:
        try:
            series_data = fred.get_series(series, observation_start=start_date)
            data[series] = series_data
            print(f"  ✓ {series}")
        except Exception as e:
            print(f"  ✗ {series}: {str(e)}")

    return data


def load_sample_data():
    """Load sample data for testing when FRED API unavailable."""
    # Generate synthetic data for demonstration
    dates = pd.date_range('2000-01-01', '2024-08-01', freq='MS')
    n_series = 14

    # Simulate correlated data (3 factors)
    factors = np.random.randn(len(dates), 3).cumsum(axis=0)
    loadings = np.random.randn(n_series, 3)
    data = factors @ loadings.T + np.random.randn(len(dates), n_series) * 0.5

    df = pd.DataFrame(data, index=dates, columns=get_default_indicators())
    return df


def transform_series(data, transformation_codes=None):
    """
    Apply FRED-MD transformation codes.

    Transformation codes:
    1 = no transformation
    2 = first difference
    3 = second difference
    4 = log
    5 = log difference
    6 = second log difference

    Parameters:
    -----------
    data : pd.DataFrame
        Raw data
    transformation_codes : dict
        Mapping from series name to transformation code

    Returns:
    --------
    pd.DataFrame
        Transformed data
    """
    if transformation_codes is None:
        # Default: most series are log-differenced (growth rates)
        transformation_codes = {col: 5 for col in data.columns}
        # Except rates (already in percent)
        for col in ['GS10', 'FEDFUNDS', 'UNRATE']:
            if col in data.columns:
                transformation_codes[col] = 2  # First difference

    # TODO: Implement transformations
    # Hint: Use np.log(), pd.diff()

    transformed = pd.DataFrame(index=data.index)

    for col in data.columns:
        code = transformation_codes.get(col, 1)
        series = data[col].copy()

        if code == 1:  # No transformation
            transformed[col] = series
        elif code == 2:  # First difference
            transformed[col] = series.diff()
        elif code == 3:  # Second difference
            transformed[col] = series.diff().diff()
        elif code == 4:  # Log
            transformed[col] = np.log(series)
        elif code == 5:  # Log difference (growth rate)
            transformed[col] = np.log(series).diff()
        elif code == 6:  # Second log difference
            transformed[col] = np.log(series).diff().diff()

    return transformed


def standardize_data(data):
    """Standardize each series to mean 0, std 1."""
    return (data - data.mean()) / data.std()


# ============================================================================
# SECTION 2: FACTOR MODEL
# ============================================================================

class StockWatsonFactorModel:
    """
    Dynamic Factor Model using Stock-Watson PCA approach.

    Two-step estimation:
    1. Extract factors via PCA
    2. Estimate factor dynamics via VAR
    """

    def __init__(self, n_factors=3, factor_lags=2):
        self.n_factors = n_factors
        self.factor_lags = factor_lags
        self.pca = None
        self.var_model = None
        self.loadings = None
        self.explained_variance = None

    def fit(self, X):
        """
        Estimate factor model.

        Parameters:
        -----------
        X : pd.DataFrame, shape (T, N)
            Standardized data (T time periods, N series)
        """
        print(f"\nEstimating {self.n_factors}-factor model...")

        # Step 1: Extract factors via PCA
        self.pca = PCA(n_components=self.n_factors)
        factors = self.pca.fit_transform(X.dropna())

        self.loadings = self.pca.components_.T  # N x r
        self.explained_variance = self.pca.explained_variance_ratio_

        # Convert to DataFrame with dates
        self.factors = pd.DataFrame(
            factors,
            index=X.dropna().index,
            columns=[f'Factor_{i+1}' for i in range(self.n_factors)]
        )

        print(f"  Variance explained: {self.explained_variance.sum():.1%}")
        for i, var in enumerate(self.explained_variance):
            print(f"    Factor {i+1}: {var:.1%}")

        # Step 2: Estimate factor dynamics (VAR)
        if self.factor_lags > 0:
            print(f"\nEstimating VAR({self.factor_lags}) for factors...")
            self.var_model = VAR(self.factors)
            self.var_result = self.var_model.fit(maxlags=self.factor_lags)
            print(f"  VAR estimated successfully")

        return self

    def predict_factors(self, steps=1):
        """Forecast factors h steps ahead."""
        if self.var_model is None:
            raise ValueError("Must fit model first")

        forecast = self.var_result.forecast(
            self.factors.values[-self.factor_lags:],
            steps=steps
        )

        return pd.DataFrame(
            forecast,
            columns=self.factors.columns
        )

    def get_factor_interpretation(self, series_names, top_k=5):
        """
        Interpret factors by top-loading series.

        Returns dict mapping factor names to top-k series.
        """
        interpretations = {}

        for i in range(self.n_factors):
            # Get loadings for this factor
            factor_loadings = pd.Series(
                self.loadings[:, i],
                index=series_names
            ).abs().sort_values(ascending=False)

            interpretations[f'Factor_{i+1}'] = factor_loadings.head(top_k)

        return interpretations


# ============================================================================
# SECTION 3: NOWCASTING
# ============================================================================

class GDPNowcaster:
    """
    GDP Nowcasting model using bridge equations.

    Methodology:
    1. Extract monthly factors from high-frequency indicators
    2. Aggregate factors to quarterly frequency
    3. Regress quarterly GDP on lagged quarterly factors
    """

    def __init__(self, factor_model):
        self.factor_model = factor_model
        self.bridge_model = None

    def fit(self, gdp_quarterly, factors_monthly):
        """
        Estimate bridge equation: GDP_t = α + β'*factors_{t-1} + ε_t

        Parameters:
        -----------
        gdp_quarterly : pd.Series
            Quarterly GDP growth rates
        factors_monthly : pd.DataFrame
            Monthly factors
        """
        print("\nEstimating bridge equation...")

        # TODO: Aggregate monthly factors to quarterly
        # Hint: Use resample('Q').mean() or take last month of quarter

        factors_quarterly = factors_monthly.resample('Q').mean()

        # Align GDP and factors
        # TODO: Create lagged factors and merge with GDP

        # Simple approach: current quarter GDP depends on current quarter factors
        data = pd.concat([gdp_quarterly, factors_quarterly], axis=1).dropna()

        y = data.iloc[:, 0]  # GDP
        X = data.iloc[:, 1:]  # Factors
        X = sm.add_constant(X)  # Add intercept

        # Estimate OLS
        self.bridge_model = OLS(y, X).fit()

        print(f"  R²: {self.bridge_model.rsquared:.3f}")
        print(f"  RMSE: {np.sqrt(self.bridge_model.mse_resid):.3f}")

        return self

    def nowcast(self, current_factors):
        """
        Generate nowcast for current quarter.

        Parameters:
        -----------
        current_factors : pd.Series or pd.DataFrame
            Factor values for current quarter

        Returns:
        --------
        dict with 'nowcast', 'lower', 'upper' (90% CI)
        """
        if self.bridge_model is None:
            raise ValueError("Must fit bridge equation first")

        # Prepare factors for prediction
        if isinstance(current_factors, pd.Series):
            current_factors = current_factors.to_frame().T

        X = sm.add_constant(current_factors)

        # Point forecast
        prediction = self.bridge_model.get_prediction(X)
        nowcast = prediction.predicted_mean[0]

        # Confidence interval
        ci = prediction.conf_int(alpha=0.10)  # 90% CI

        return {
            'nowcast': nowcast,
            'lower': ci[0, 0],
            'upper': ci[0, 1],
            'std_error': prediction.se_mean[0]
        }


# ============================================================================
# SECTION 4: VISUALIZATION
# ============================================================================

def plot_factors(factor_model, data_dates):
    """Plot estimated factors over time."""
    fig, axes = plt.subplots(factor_model.n_factors, 1, figsize=(12, 8))

    for i in range(factor_model.n_factors):
        ax = axes[i] if factor_model.n_factors > 1 else axes
        factor_data = factor_model.factors.iloc[:, i]

        ax.plot(factor_data.index, factor_data, linewidth=1.5)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_ylabel(f'Factor {i+1}')
        ax.grid(True, alpha=0.3)

        # Add shaded recession periods (optional)
        # TODO: Add NBER recession shading

    axes[-1].set_xlabel('Date')
    plt.suptitle('Estimated Factors', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_loadings_heatmap(factor_model, series_names):
    """Plot factor loadings as heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    loadings_df = pd.DataFrame(
        factor_model.loadings,
        index=series_names,
        columns=[f'Factor {i+1}' for i in range(factor_model.n_factors)]
    )

    sns.heatmap(loadings_df, cmap='RdBu_r', center=0,
                annot=True, fmt='.2f', ax=ax)
    ax.set_title('Factor Loadings', fontsize=14, fontweight='bold')

    return fig


def plot_nowcast_evolution(nowcasts, gdp_actual):
    """
    Plot how nowcast evolves as new data arrives.

    Parameters:
    -----------
    nowcasts : pd.DataFrame
        Columns: 'vintage_date', 'nowcast', 'lower', 'upper'
    gdp_actual : pd.Series
        Actual GDP growth rates
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot historical GDP
    ax.plot(gdp_actual.index, gdp_actual, 'k-', linewidth=2,
            label='Actual GDP', alpha=0.7)

    # Plot nowcast evolution
    if 'vintage_date' in nowcasts.columns:
        for _, row in nowcasts.iterrows():
            ax.plot(row['target_quarter'], row['nowcast'], 'o',
                   color='red', alpha=0.5)
    else:
        # Single nowcast
        ax.plot(nowcasts.index, nowcasts['nowcast'], 'ro',
               markersize=10, label='Nowcast')
        ax.fill_between(nowcasts.index, nowcasts['lower'], nowcasts['upper'],
                       alpha=0.3, color='red', label='90% CI')

    ax.set_xlabel('Date')
    ax.set_ylabel('GDP Growth Rate (%)')
    ax.set_title('GDP Nowcast', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main nowcasting workflow."""

    print("=" * 70)
    print("GDP NOWCASTING MODEL")
    print("=" * 70)

    # 1. Load configuration
    config = load_config()

    # 2. Fetch data
    print("\n[1/5] Fetching data...")
    data_raw = fetch_fred_data(
        api_key=config['fred']['api_key'],
        series_list=config['indicators'],
        start_date='2000-01-01'
    )

    print(f"  Data shape: {data_raw.shape}")
    print(f"  Date range: {data_raw.index[0]} to {data_raw.index[-1]}")

    # 3. Transform and standardize
    print("\n[2/5] Transforming data...")
    data_transformed = transform_series(data_raw)
    data_std = standardize_data(data_transformed)

    print(f"  Transformed shape: {data_std.shape}")
    print(f"  Missing values: {data_std.isna().sum().sum()}")

    # 4. Estimate factor model
    print("\n[3/5] Estimating factor model...")
    factor_model = StockWatsonFactorModel(
        n_factors=config['model']['n_factors'],
        factor_lags=config['model']['factor_lags']
    )
    factor_model.fit(data_std)

    # Interpret factors
    print("\nFactor interpretation:")
    interpretations = factor_model.get_factor_interpretation(data_std.columns)
    for factor_name, top_series in interpretations.items():
        print(f"\n  {factor_name}:")
        for series, loading in top_series.items():
            print(f"    {series}: {loading:.3f}")

    # 5. Estimate bridge equation and nowcast
    print("\n[4/5] Nowcasting GDP...")

    # Fetch quarterly GDP (TODO: Replace with actual FRED data)
    # For now, simulate quarterly GDP
    gdp_quarterly = pd.Series(
        np.random.randn(len(data_std) // 3) * 0.5 + 2.0,
        index=pd.date_range(data_std.index[0], data_std.index[-1], freq='Q')
    )

    nowcaster = GDPNowcaster(factor_model)
    nowcaster.fit(gdp_quarterly, factor_model.factors)

    # Generate nowcast for most recent quarter
    current_factors = factor_model.factors.resample('Q').mean().iloc[-1]
    nowcast_result = nowcaster.nowcast(current_factors)

    print(f"\n{'=' * 70}")
    print(f"NOWCAST RESULT")
    print(f"{'=' * 70}")
    print(f"  Estimate: {nowcast_result['nowcast']:.2f}%")
    print(f"  90% CI:   [{nowcast_result['lower']:.2f}%, {nowcast_result['upper']:.2f}%]")
    print(f"  Std Error: {nowcast_result['std_error']:.2f}%")

    # 6. Create visualizations
    print("\n[5/5] Creating visualizations...")

    # Plot factors
    fig_factors = plot_factors(factor_model, data_std.index)
    fig_factors.savefig(OUTPUT_DIR / 'factors.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR / 'factors.png'}")

    # Plot loadings
    fig_loadings = plot_loadings_heatmap(factor_model, data_std.columns)
    fig_loadings.savefig(OUTPUT_DIR / 'loadings.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR / 'loadings.png'}")

    # Plot nowcast
    nowcast_df = pd.DataFrame([nowcast_result],
                              index=[gdp_quarterly.index[-1]])
    fig_nowcast = plot_nowcast_evolution(nowcast_df, gdp_quarterly)
    fig_nowcast.savefig(OUTPUT_DIR / 'nowcast.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR / 'nowcast.png'}")

    print(f"\n{'=' * 70}")
    print("COMPLETE! Check the 'output/' folder for results.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
