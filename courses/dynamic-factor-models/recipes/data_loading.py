"""
Data Loading Patterns for Dynamic Factor Models - Copy-paste recipes

Ready-to-use code for loading and preparing economic/financial data
from common sources (FRED, Yahoo Finance, CSV, etc.)
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict
import warnings

warnings.filterwarnings('ignore')


# ===========================================================================
# RECIPE 1: Load Data from FRED (Federal Reserve Economic Data)
# Problem: Download multiple macroeconomic series from FRED API
# ===========================================================================

def load_fred_data(
    series_ids: List[str],
    start_date: str = '2000-01-01',
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Input: List of FRED series IDs, date range
    Output: DataFrame with series as columns, dates as index

    Example series IDs:
    - GDPC1: Real GDP
    - INDPRO: Industrial Production Index
    - PAYEMS: Total Nonfarm Payrolls
    - UNRATE: Unemployment Rate
    - CPIAUCSL: Consumer Price Index
    """
    try:
        from pandas_datareader import data as pdr

        df = pdr.DataReader(
            series_ids,
            'fred',
            start=start_date,
            end=end_date
        )

        print(f"Loaded {len(series_ids)} series from FRED")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        print(f"Observations: {len(df)}")

        return df

    except ImportError:
        print("ERROR: Install pandas-datareader with: pip install pandas-datareader")
        raise
    except Exception as e:
        print(f"ERROR loading FRED data: {e}")
        print("Check series IDs are valid at: https://fred.stlouisfed.org/")
        raise


# ===========================================================================
# RECIPE 2: Load Financial Data from Yahoo Finance
# Problem: Download stock prices, indices, or ETF data
# ===========================================================================

def load_yahoo_finance_data(
    tickers: List[str],
    start_date: str = '2010-01-01',
    end_date: Optional[str] = None,
    price_type: str = 'Adj Close'
) -> pd.DataFrame:
    """
    Input: List of tickers, date range, price type
    Output: DataFrame with prices

    Common tickers:
    - '^GSPC': S&P 500
    - '^DJI': Dow Jones
    - '^VIX': Volatility Index
    - 'SPY': S&P 500 ETF
    """
    try:
        import yfinance as yf

        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False
        )

        # Extract specific price type
        if len(tickers) == 1:
            df = data[[price_type]].copy()
            df.columns = tickers
        else:
            df = data[price_type].copy()

        print(f"Loaded {len(tickers)} tickers from Yahoo Finance")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")

        return df

    except ImportError:
        print("ERROR: Install yfinance with: pip install yfinance")
        raise
    except Exception as e:
        print(f"ERROR loading Yahoo Finance data: {e}")
        raise


# ===========================================================================
# RECIPE 3: Load CSV with Proper Date Parsing
# Problem: Import local CSV file with time series data
# ===========================================================================

def load_csv_timeseries(
    filepath: str,
    date_column: str = 'Date',
    date_format: Optional[str] = None
) -> pd.DataFrame:
    """
    Input: CSV file path, date column name, optional date format
    Output: DataFrame with datetime index

    Date format examples:
    - '%Y-%m-%d': 2020-01-15
    - '%m/%d/%Y': 01/15/2020
    - '%Y%m%d': 20200115
    """
    df = pd.read_csv(filepath, parse_dates=[date_column])

    # Set date as index
    df = df.set_index(date_column)

    # Ensure proper datetime
    if date_format:
        df.index = pd.to_datetime(df.index, format=date_format)
    else:
        df.index = pd.to_datetime(df.index)

    # Sort by date
    df = df.sort_index()

    print(f"Loaded CSV: {filepath}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Columns: {list(df.columns)}")

    return df


# ===========================================================================
# RECIPE 4: Handle Missing Values (Multiple Strategies)
# Problem: Clean missing data before DFM estimation
# ===========================================================================

def handle_missing_values(
    df: pd.DataFrame,
    method: str = 'interpolate',
    max_consecutive: int = 3
) -> pd.DataFrame:
    """
    Input: DataFrame with missing values, imputation method
    Output: DataFrame with missing values handled

    Methods:
    - 'interpolate': Linear interpolation
    - 'forward_fill': Carry forward last observation
    - 'backward_fill': Use next observation
    - 'mean': Replace with column mean
    - 'drop': Drop rows with any missing
    """
    df_clean = df.copy()

    # Report missing values
    missing = df_clean.isnull().sum()
    if missing.any():
        print("Missing values before cleaning:")
        print(missing[missing > 0])
    else:
        print("No missing values detected")
        return df_clean

    if method == 'interpolate':
        df_clean = df_clean.interpolate(
            method='linear',
            limit=max_consecutive,
            limit_direction='both'
        )

    elif method == 'forward_fill':
        df_clean = df_clean.fillna(method='ffill', limit=max_consecutive)

    elif method == 'backward_fill':
        df_clean = df_clean.fillna(method='bfill', limit=max_consecutive)

    elif method == 'mean':
        df_clean = df_clean.fillna(df_clean.mean())

    elif method == 'drop':
        df_clean = df_clean.dropna()

    else:
        raise ValueError(f"Unknown method: {method}")

    # Check remaining missing
    remaining_missing = df_clean.isnull().sum()
    if remaining_missing.any():
        print(f"\nWarning: Still have missing values after {method}:")
        print(remaining_missing[remaining_missing > 0])

    print(f"\nCleaning complete. Shape: {df_clean.shape}")

    return df_clean


# ===========================================================================
# RECIPE 5: Align Mixed Frequency Data (Monthly + Quarterly)
# Problem: Combine monthly indicators with quarterly target (e.g., GDP)
# ===========================================================================

def align_mixed_frequencies(
    monthly_data: pd.DataFrame,
    quarterly_data: pd.DataFrame,
    quarterly_method: str = 'end'
) -> pd.DataFrame:
    """
    Input: Monthly and quarterly DataFrames
    Output: Combined DataFrame at monthly frequency

    quarterly_method:
    - 'end': Quarterly value at end of quarter
    - 'start': Quarterly value at start of quarter
    - 'interpolate': Spread quarterly evenly across months
    """
    # Resample quarterly to monthly
    if quarterly_method == 'end':
        # Quarterly values appear at quarter end
        quarterly_monthly = quarterly_data.resample('MS').last()

    elif quarterly_method == 'start':
        # Quarterly values appear at quarter start
        quarterly_monthly = quarterly_data.resample('MS').first()

    elif quarterly_method == 'interpolate':
        # Interpolate quarterly linearly to monthly
        quarterly_monthly = quarterly_data.resample('MS').interpolate(method='linear')

    else:
        raise ValueError(f"Unknown method: {quarterly_method}")

    # Combine
    combined = pd.concat([monthly_data, quarterly_monthly], axis=1)

    # Align dates
    combined = combined.sort_index()

    print(f"Combined {len(monthly_data.columns)} monthly + {len(quarterly_data.columns)} quarterly series")
    print(f"Result shape: {combined.shape}")

    return combined


# ===========================================================================
# RECIPE 6: Standardize Data (Essential for DFM)
# Problem: Normalize series to zero mean, unit variance
# ===========================================================================

def standardize_data(
    df: pd.DataFrame,
    method: str = 'zscore',
    save_params: bool = False
) -> pd.DataFrame:
    """
    Input: Raw data DataFrame
    Output: Standardized DataFrame

    Methods:
    - 'zscore': (x - mean) / std
    - 'minmax': (x - min) / (max - min)
    - 'robust': (x - median) / IQR
    """
    if method == 'zscore':
        mean = df.mean()
        std = df.std()
        df_standardized = (df - mean) / std

        if save_params:
            return df_standardized, {'mean': mean, 'std': std}

    elif method == 'minmax':
        min_val = df.min()
        max_val = df.max()
        df_standardized = (df - min_val) / (max_val - min_val)

        if save_params:
            return df_standardized, {'min': min_val, 'max': max_val}

    elif method == 'robust':
        median = df.median()
        q75 = df.quantile(0.75)
        q25 = df.quantile(0.25)
        iqr = q75 - q25
        df_standardized = (df - median) / iqr

        if save_params:
            return df_standardized, {'median': median, 'iqr': iqr}

    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"Standardized using {method} method")
    return df_standardized


# ===========================================================================
# RECIPE 7: Apply Transformations (Log, Diff, Growth Rates)
# Problem: Transform non-stationary series before DFM
# ===========================================================================

def apply_transformations(
    df: pd.DataFrame,
    transformations: Dict[str, str]
) -> pd.DataFrame:
    """
    Input: DataFrame, dict mapping column names to transformations
    Output: Transformed DataFrame

    Transformations:
    - 'log': Natural logarithm
    - 'diff': First difference
    - 'log_diff': Log difference (growth rate)
    - 'pct_change': Percentage change
    - 'none': No transformation
    """
    df_transformed = pd.DataFrame(index=df.index)

    for col in df.columns:
        transform = transformations.get(col, 'none')

        if transform == 'log':
            df_transformed[col] = np.log(df[col])

        elif transform == 'diff':
            df_transformed[col] = df[col].diff()

        elif transform == 'log_diff':
            df_transformed[col] = np.log(df[col]).diff()

        elif transform == 'pct_change':
            df_transformed[col] = df[col].pct_change() * 100

        elif transform == 'none':
            df_transformed[col] = df[col]

        else:
            raise ValueError(f"Unknown transformation: {transform}")

    # Drop NaN from differencing
    df_transformed = df_transformed.dropna()

    print("Applied transformations:")
    for col, trans in transformations.items():
        if col in df.columns:
            print(f"  {col}: {trans}")

    return df_transformed


# ===========================================================================
# RECIPE 8: Load Complete Macro Dataset (Ready-to-Use)
# Problem: Get full macro dataset for DFM in one function call
# ===========================================================================

def load_complete_macro_dataset(
    start_date: str = '2000-01-01',
    include_financial: bool = True
) -> pd.DataFrame:
    """
    Input: Start date, whether to include financial indicators
    Output: Complete standardized dataset ready for DFM

    Includes:
    - Real activity: Industrial production, employment, retail sales
    - Prices: CPI, PPI
    - Housing: Housing starts, permits
    - Financial (optional): Stock indices, interest rates
    """
    # Real activity indicators (monthly)
    real_activity = [
        'INDPRO',   # Industrial Production Index
        'PAYEMS',   # Total Nonfarm Payrolls
        'RETAIL',   # Retail and Food Services Sales
        'HOUST',    # Housing Starts
    ]

    # Price indicators (monthly)
    prices = [
        'CPIAUCSL',  # Consumer Price Index
        'PPIACO',    # Producer Price Index
    ]

    # Financial indicators (monthly)
    financial = [
        'GS10',      # 10-Year Treasury Rate
        'TB3MS',     # 3-Month Treasury Bill
    ]

    # Combine series
    series_ids = real_activity + prices
    if include_financial:
        series_ids += financial

    # Load from FRED
    df = load_fred_data(series_ids, start_date=start_date)

    # Apply standard transformations
    transformations = {
        'INDPRO': 'log_diff',    # Growth rate
        'PAYEMS': 'log_diff',    # Growth rate
        'RETAIL': 'log_diff',    # Growth rate
        'HOUST': 'log',          # Log level
        'CPIAUCSL': 'log_diff',  # Inflation
        'PPIACO': 'log_diff',    # Inflation
        'GS10': 'diff',          # Change in rate
        'TB3MS': 'diff',         # Change in rate
    }

    df = apply_transformations(df, transformations)

    # Handle missing
    df = handle_missing_values(df, method='interpolate')

    # Standardize
    df = standardize_data(df, method='zscore')

    print("\n" + "="*70)
    print("COMPLETE MACRO DATASET READY")
    print("="*70)
    print(f"Shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    return df


# ===========================================================================
# RECIPE 9: Save and Load Processed Data
# Problem: Cache processed data to avoid re-downloading
# ===========================================================================

def save_processed_data(df: pd.DataFrame, filepath: str = 'processed_data.csv'):
    """Save processed DataFrame to CSV"""
    df.to_csv(filepath)
    print(f"Saved to: {filepath}")


def load_processed_data(filepath: str = 'processed_data.csv') -> pd.DataFrame:
    """Load previously processed data"""
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"Loaded from: {filepath}")
    print(f"Shape: {df.shape}")
    return df


# ===========================================================================
# RECIPE 10: Create Lagged Features for Forecasting
# Problem: Create lagged variables for forecasting models
# ===========================================================================

def create_lagged_features(
    df: pd.DataFrame,
    lags: List[int] = [1, 2, 3],
    target_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Input: DataFrame, list of lags, optional target column
    Output: DataFrame with lagged features

    If target_column specified, only lags for that column are created
    """
    df_lagged = df.copy()

    columns_to_lag = [target_column] if target_column else df.columns

    for col in columns_to_lag:
        for lag in lags:
            df_lagged[f'{col}_lag{lag}'] = df[col].shift(lag)

    # Drop rows with NaN from lagging
    df_lagged = df_lagged.dropna()

    print(f"Created lags {lags} for {len(columns_to_lag)} column(s)")
    print(f"Result shape: {df_lagged.shape}")

    return df_lagged


# ===========================================================================
# EXAMPLE USAGE
# ===========================================================================

if __name__ == "__main__":
    print("="*70)
    print("EXAMPLE 1: Load FRED Data")
    print("="*70)

    fred_series = ['INDPRO', 'PAYEMS', 'UNRATE']
    df_fred = load_fred_data(fred_series, start_date='2010-01-01')
    print(df_fred.head())

    print("\n" + "="*70)
    print("EXAMPLE 2: Handle Missing Values")
    print("="*70)

    # Introduce some missing values for demonstration
    df_missing = df_fred.copy()
    df_missing.iloc[10:15, 0] = np.nan

    df_clean = handle_missing_values(df_missing, method='interpolate')
    print(f"Missing before: {df_missing.isnull().sum().sum()}")
    print(f"Missing after: {df_clean.isnull().sum().sum()}")

    print("\n" + "="*70)
    print("EXAMPLE 3: Standardize Data")
    print("="*70)

    df_standardized = standardize_data(df_clean, method='zscore')
    print("\nStandardized data statistics:")
    print(df_standardized.describe())

    print("\n" + "="*70)
    print("EXAMPLE 4: Complete Macro Dataset")
    print("="*70)

    df_macro = load_complete_macro_dataset(
        start_date='2015-01-01',
        include_financial=True
    )
    print(df_macro.tail())
