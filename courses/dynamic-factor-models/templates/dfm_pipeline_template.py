"""
Dynamic Factor Model Pipeline Template - Copy and customize for your use case

Works with: statsmodels, pandas, pandas-datareader
Time to working: 5-10 minutes

This template provides an end-to-end DFM pipeline:
1. Data loading and preprocessing
2. Model estimation
3. Factor extraction
4. Forecasting
5. Results visualization and export

Usage:
    python dfm_pipeline_template.py
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from statsmodels.tsa.statespace.tools import diff

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION - CUSTOMIZE THESE
# ============================================================================

CONFIG = {
    # Data settings
    'data_source': 'fred',  # TODO: Change to 'csv', 'fred', or 'custom'
    'fred_series': [
        'INDPRO',      # Industrial Production Index
        'PAYEMS',      # Total Nonfarm Payrolls
        'RETAIL',      # Retail Sales
        'HOUST',       # Housing Starts
        'UNRATE',      # Unemployment Rate
        'CPIAUCSL',    # Consumer Price Index
    ],
    'start_date': '2000-01-01',
    'end_date': None,  # None for latest available

    # Model settings
    'n_factors': 1,    # TODO: Number of latent factors (typically 1-3)
    'factor_order': 2, # TODO: AR order for factor dynamics (typically 1-4)
    'error_order': 0,  # AR order for idiosyncratic errors (0 for white noise)

    # Preprocessing
    'standardize': True,
    'handle_missing': 'interpolate',  # 'interpolate', 'drop', or 'forward_fill'

    # Forecasting
    'forecast_horizon': 12,  # TODO: Months ahead to forecast

    # Output
    'output_dir': './dfm_results',
    'save_plots': True,
    'save_factors': True,
    'save_forecasts': True,
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(config: Dict) -> pd.DataFrame:
    """
    Load data from configured source.

    Returns:
        DataFrame with datetime index and series as columns
    """
    logger.info("Loading data...")

    if config['data_source'] == 'fred':
        try:
            from pandas_datareader import data as pdr

            df = pdr.DataReader(
                config['fred_series'],
                'fred',
                start=config['start_date'],
                end=config['end_date']
            )
            logger.info(f"Loaded {len(df)} observations for {len(df.columns)} series from FRED")
            return df

        except ImportError:
            logger.error("pandas-datareader not installed. Install with: pip install pandas-datareader")
            raise
        except Exception as e:
            logger.error(f"Failed to load FRED data: {e}")
            raise

    elif config['data_source'] == 'csv':
        # TODO: Customize CSV loading
        csv_path = './data/macro_data.csv'  # TODO: Update path
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded {len(df)} observations from CSV")
        return df

    else:
        raise ValueError(f"Unknown data source: {config['data_source']}")


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def preprocess_data(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Clean, transform, and standardize data.

    Returns:
        Preprocessed DataFrame ready for modeling
    """
    logger.info("Preprocessing data...")

    df_clean = df.copy()

    # Handle missing values
    if config['handle_missing'] == 'interpolate':
        df_clean = df_clean.interpolate(method='linear', limit_direction='both')
    elif config['handle_missing'] == 'drop':
        df_clean = df_clean.dropna()
    elif config['handle_missing'] == 'forward_fill':
        df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')

    # Log missing value summary
    missing = df_clean.isnull().sum()
    if missing.any():
        logger.warning(f"Remaining missing values:\n{missing[missing > 0]}")

    # Standardize (zero mean, unit variance)
    if config['standardize']:
        df_clean = (df_clean - df_clean.mean()) / df_clean.std()
        logger.info("Data standardized")

    logger.info(f"Preprocessed shape: {df_clean.shape}")
    return df_clean


# ============================================================================
# MODEL ESTIMATION
# ============================================================================

def estimate_dfm(data: pd.DataFrame, config: Dict) -> DynamicFactor:
    """
    Estimate Dynamic Factor Model.

    Returns:
        Fitted DynamicFactor model
    """
    logger.info("Estimating Dynamic Factor Model...")
    logger.info(f"Factors: {config['n_factors']}, Factor AR: {config['factor_order']}, Error AR: {config['error_order']}")

    try:
        # Initialize model
        model = DynamicFactor(
            endog=data,
            k_factors=config['n_factors'],
            factor_order=config['factor_order'],
            error_order=config['error_order'],
            error_cov_type='diagonal'  # Diagonal covariance for idiosyncratic errors
        )

        # Estimate parameters using MLE
        results = model.fit(maxiter=1000, disp=False)

        logger.info("Model estimation complete")
        logger.info(f"Log-likelihood: {results.llf:.2f}")
        logger.info(f"AIC: {results.aic:.2f}")
        logger.info(f"BIC: {results.bic:.2f}")

        return results

    except Exception as e:
        logger.error(f"Model estimation failed: {e}")
        logger.error("Try reducing factor_order or n_factors, or check for data issues")
        raise


# ============================================================================
# FACTOR EXTRACTION
# ============================================================================

def extract_factors(results: DynamicFactor) -> pd.DataFrame:
    """
    Extract smoothed factors from fitted model.

    Returns:
        DataFrame with extracted factors
    """
    logger.info("Extracting factors...")

    # Get smoothed factors (using full sample information)
    factors = results.factors.smoothed

    # Create DataFrame with proper index
    factor_df = pd.DataFrame(
        factors,
        index=results.data.dates,
        columns=[f'Factor_{i+1}' for i in range(factors.shape[1])]
    )

    logger.info(f"Extracted {factors.shape[1]} factors")
    return factor_df


# ============================================================================
# FORECASTING
# ============================================================================

def generate_forecasts(results: DynamicFactor, horizon: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate out-of-sample forecasts.

    Returns:
        Tuple of (forecasts, confidence_intervals)
    """
    logger.info(f"Generating {horizon}-step ahead forecasts...")

    try:
        # Generate forecasts
        forecast = results.forecast(steps=horizon)

        # Get forecast object with confidence intervals
        forecast_obj = results.get_forecast(steps=horizon)
        forecast_df = forecast_obj.summary_frame(alpha=0.05)  # 95% CI

        # Create forecast index
        last_date = results.data.dates[-1]
        if isinstance(last_date, pd.Timestamp):
            freq = pd.infer_freq(results.data.dates)
            forecast_index = pd.date_range(
                start=last_date + pd.tseries.frequencies.to_offset(freq),
                periods=horizon,
                freq=freq
            )
        else:
            forecast_index = range(len(results.data.dates), len(results.data.dates) + horizon)

        # Organize forecasts
        forecasts = pd.DataFrame(
            forecast,
            index=forecast_index,
            columns=results.model.endog_names
        )

        logger.info("Forecasts generated successfully")
        return forecasts, forecast_df

    except Exception as e:
        logger.error(f"Forecast generation failed: {e}")
        raise


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(
    data: pd.DataFrame,
    factors: pd.DataFrame,
    forecasts: pd.DataFrame,
    results: DynamicFactor,
    output_dir: Path
) -> None:
    """
    Create diagnostic plots and save to output directory.
    """
    logger.info("Creating visualization plots...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Extracted Factors
    fig, axes = plt.subplots(factors.shape[1], 1, figsize=(12, 3*factors.shape[1]))
    if factors.shape[1] == 1:
        axes = [axes]

    for i, col in enumerate(factors.columns):
        axes[i].plot(factors.index, factors[col], linewidth=1.5)
        axes[i].set_title(f'{col}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Date')
        axes[i].grid(alpha=0.3)
        axes[i].axhline(y=0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'factors.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_dir / 'factors.png'}")
    plt.close()

    # Plot 2: Factor Loadings
    loadings = results.params[results.model.param_names.index('loading.f1.INDPRO'):
                              results.model.param_names.index('loading.f1.INDPRO') + len(data.columns)]

    fig, ax = plt.subplots(figsize=(10, 6))
    loadings_series = pd.Series(loadings.values, index=data.columns)
    loadings_series.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title('Factor Loadings', fontsize=14, fontweight='bold')
    ax.set_xlabel('Loading Value')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'loadings.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_dir / 'loadings.png'}")
    plt.close()

    # Plot 3: Forecasts (first 3 series)
    n_series_to_plot = min(3, len(forecasts.columns))
    fig, axes = plt.subplots(n_series_to_plot, 1, figsize=(12, 4*n_series_to_plot))
    if n_series_to_plot == 1:
        axes = [axes]

    for i, col in enumerate(forecasts.columns[:n_series_to_plot]):
        # Historical data
        axes[i].plot(data.index, data[col], label='Historical', linewidth=1.5)

        # Forecasts
        axes[i].plot(forecasts.index, forecasts[col],
                    label='Forecast', linewidth=1.5, linestyle='--', color='red')

        axes[i].set_title(f'{col} - Historical vs Forecast', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Date')
        axes[i].legend()
        axes[i].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'forecasts.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_dir / 'forecasts.png'}")
    plt.close()


# ============================================================================
# EXPORT RESULTS
# ============================================================================

def export_results(
    factors: pd.DataFrame,
    forecasts: pd.DataFrame,
    results: DynamicFactor,
    config: Dict,
    output_dir: Path
) -> None:
    """
    Export factors, forecasts, and model summary to CSV.
    """
    logger.info("Exporting results...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Export factors
    if config['save_factors']:
        factors_path = output_dir / 'extracted_factors.csv'
        factors.to_csv(factors_path)
        logger.info(f"Saved: {factors_path}")

    # Export forecasts
    if config['save_forecasts']:
        forecasts_path = output_dir / 'forecasts.csv'
        forecasts.to_csv(forecasts_path)
        logger.info(f"Saved: {forecasts_path}")

    # Export model summary
    summary_path = output_dir / 'model_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(str(results.summary()))
    logger.info(f"Saved: {summary_path}")

    # Export parameters
    params_path = output_dir / 'model_parameters.csv'
    params_df = pd.DataFrame({
        'parameter': results.params.index,
        'value': results.params.values,
        'std_error': results.bse.values,
        'p_value': results.pvalues.values
    })
    params_df.to_csv(params_path, index=False)
    logger.info(f"Saved: {params_path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(config: Dict) -> Dict:
    """
    Execute end-to-end DFM pipeline.

    Returns:
        Dictionary containing all results
    """
    logger.info("="*70)
    logger.info("DYNAMIC FACTOR MODEL PIPELINE")
    logger.info("="*70)

    # Setup output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Load data
        data = load_data(config)

        # Step 2: Preprocess
        data_clean = preprocess_data(data, config)

        # Step 3: Estimate model
        results = estimate_dfm(data_clean, config)

        # Step 4: Extract factors
        factors = extract_factors(results)

        # Step 5: Generate forecasts
        forecasts, forecast_intervals = generate_forecasts(
            results,
            config['forecast_horizon']
        )

        # Step 6: Visualize
        if config['save_plots']:
            plot_results(data_clean, factors, forecasts, results, output_dir)

        # Step 7: Export results
        export_results(factors, forecasts, results, config, output_dir)

        logger.info("="*70)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Results saved to: {output_dir.absolute()}")
        logger.info("="*70)

        return {
            'data': data_clean,
            'model': results,
            'factors': factors,
            'forecasts': forecasts,
            'forecast_intervals': forecast_intervals
        }

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run the pipeline with configured settings
    pipeline_results = run_pipeline(CONFIG)

    # TODO: Add custom post-processing here
    # Example: Compare forecasts with actuals, compute accuracy metrics, etc.

    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"\nExtracted Factors:")
    print(pipeline_results['factors'].describe())
    print(f"\nForecasts (first 3 series):")
    print(pipeline_results['forecasts'].iloc[:, :3].describe())
