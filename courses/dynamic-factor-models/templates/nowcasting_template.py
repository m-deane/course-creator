"""
Real-Time Nowcasting Template - Copy and customize for your use case

Works with: statsmodels, pandas, pandas-datareader
Time to working: 10-15 minutes

This template implements a production-ready nowcasting system:
1. Handles mixed-frequency data (monthly + quarterly)
2. Updates with new data releases incrementally
3. Caches intermediate results for speed
4. Produces nowcasts with confidence intervals
5. Tracks nowcast revisions over time

Usage:
    python nowcasting_template.py
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION - CUSTOMIZE THESE
# ============================================================================

CONFIG = {
    # Target variable (typically quarterly GDP)
    'target_series': 'GDPC1',  # TODO: Real GDP (quarterly)
    'target_name': 'GDP Growth',

    # High-frequency indicators (monthly)
    'monthly_indicators': [
        'INDPRO',      # Industrial Production
        'PAYEMS',      # Nonfarm Payrolls
        'RETAIL',      # Retail Sales
        'HOUST',       # Housing Starts
    ],

    # Model settings
    'n_factors': 2,
    'factor_order': 2,
    'estimation_window': 60,  # Months for rolling window (0 = expanding)

    # Nowcasting settings
    'transform_target': 'growth',  # 'growth', 'level', 'log'
    'update_frequency': 'monthly',  # How often to update nowcast

    # Caching for speed
    'cache_dir': './nowcast_cache',
    'use_cache': True,

    # Output
    'output_dir': './nowcast_results',
    'track_revisions': True,
}


# ============================================================================
# DATA MANAGEMENT
# ============================================================================

class NowcastDataManager:
    """
    Manages data loading, updating, and alignment for nowcasting.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.cache_dir = Path(config['cache_dir'])
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_latest_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Load latest data from FRED, using cache if available.
        """
        cache_file = self.cache_dir / 'latest_data.pkl'

        # Check cache
        if not force_refresh and self.config['use_cache'] and cache_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age.days < 1:  # Cache valid for 1 day
                logger.info("Loading data from cache...")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)

        # Fetch fresh data
        logger.info("Fetching latest data from FRED...")
        try:
            from pandas_datareader import data as pdr

            # Load target (quarterly)
            target = pdr.DataReader(
                self.config['target_series'],
                'fred',
                start='2000-01-01'
            )

            # Load indicators (monthly)
            indicators = pdr.DataReader(
                self.config['monthly_indicators'],
                'fred',
                start='2000-01-01'
            )

            # Combine
            data = pd.concat([target, indicators], axis=1)

            # Cache for next time
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)

            logger.info(f"Loaded data through {data.index[-1].strftime('%Y-%m-%d')}")
            return data

        except ImportError:
            logger.error("pandas-datareader not installed. Install with: pip install pandas-datareader")
            raise
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def align_mixed_frequencies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Align quarterly target with monthly indicators.

        Strategy: Interpolate quarterly data to monthly, keeping
        track of which values are observed vs interpolated.
        """
        logger.info("Aligning mixed-frequency data...")

        # Resample everything to monthly frequency
        monthly_data = data.resample('MS').last()

        # Forward-fill quarterly values within the quarter
        # (quarterly values are only observed at quarter-end)
        target_col = self.config['target_series']
        monthly_data[f'{target_col}_observed'] = ~monthly_data[target_col].isnull()
        monthly_data[target_col] = monthly_data[target_col].fillna(method='ffill')

        logger.info(f"Aligned to monthly frequency: {len(monthly_data)} observations")
        return monthly_data

    def transform_target(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform target variable (growth rates, log, etc).
        """
        target_col = self.config['target_series']
        transform = self.config['transform_target']

        if transform == 'growth':
            # Quarter-over-quarter growth rate
            data[f'{target_col}_transformed'] = (
                data[target_col].pct_change(periods=3) * 100  # Quarterly growth in %
            )
        elif transform == 'log':
            data[f'{target_col}_transformed'] = np.log(data[target_col])
        else:  # level
            data[f'{target_col}_transformed'] = data[target_col]

        logger.info(f"Applied transformation: {transform}")
        return data

    def prepare_modeling_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Complete data preparation pipeline.
        """
        # Load
        data = self.load_latest_data(force_refresh)

        # Align frequencies
        data = self.align_mixed_frequencies(data)

        # Transform target
        data = self.transform_target(data)

        # Handle missing in indicators
        indicator_cols = self.config['monthly_indicators']
        data[indicator_cols] = data[indicator_cols].interpolate(method='linear')

        # Standardize
        data[indicator_cols] = (
            (data[indicator_cols] - data[indicator_cols].mean()) /
            data[indicator_cols].std()
        )

        return data


# ============================================================================
# NOWCASTING ENGINE
# ============================================================================

class NowcastEngine:
    """
    Core nowcasting engine with incremental updates.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.model_cache = {}
        self.nowcast_history = []

    def estimate_model(
        self,
        data: pd.DataFrame,
        as_of_date: Optional[pd.Timestamp] = None
    ) -> DynamicFactor:
        """
        Estimate DFM on data available as of a specific date.
        """
        if as_of_date is None:
            as_of_date = data.index[-1]

        # Subset data to estimation window
        estimation_data = data.loc[:as_of_date].copy()

        if self.config['estimation_window'] > 0:
            estimation_data = estimation_data.iloc[-self.config['estimation_window']:]

        # Prepare endog (indicators + target)
        target_col = f"{self.config['target_series']}_transformed"
        indicator_cols = self.config['monthly_indicators']
        endog = estimation_data[indicator_cols + [target_col]]

        # Drop rows with any missing target values for estimation
        endog_clean = endog.dropna()

        logger.info(f"Estimating model with {len(endog_clean)} observations (as of {as_of_date.strftime('%Y-%m-%d')})")

        # Estimate DFM
        model = DynamicFactor(
            endog=endog_clean,
            k_factors=self.config['n_factors'],
            factor_order=self.config['factor_order'],
            error_order=0,
            error_cov_type='diagonal'
        )

        results = model.fit(maxiter=500, disp=False)
        logger.info(f"Model AIC: {results.aic:.2f}, BIC: {results.bic:.2f}")

        return results

    def generate_nowcast(
        self,
        results: DynamicFactor,
        data: pd.DataFrame,
        target_date: pd.Timestamp
    ) -> Tuple[float, float, float]:
        """
        Generate nowcast for target_date using all available indicator data.

        Returns:
            Tuple of (nowcast, lower_bound, upper_bound)
        """
        # Get data up to target date
        data_subset = data.loc[:target_date].copy()

        target_col = f"{self.config['target_series']}_transformed"
        indicator_cols = self.config['monthly_indicators']

        # Check if target is already observed
        is_observed = data_subset[f"{self.config['target_series']}_observed"].iloc[-1]

        if is_observed:
            # No nowcast needed - value already observed
            actual = data_subset[target_col].iloc[-1]
            return actual, actual, actual

        # Create "missing" target for nowcasting
        # Use indicators to fill in missing target value
        endog = data_subset[indicator_cols + [target_col]].copy()

        # Set target to missing for dates we want to nowcast
        last_observed_idx = data_subset[
            data_subset[f"{self.config['target_series']}_observed"]
        ].index[-1]

        endog.loc[last_observed_idx:, target_col] = np.nan

        # Update model with new data (Kalman filter)
        updated_results = results.apply(endog, copy_initialization=False)

        # Extract smoothed estimate of missing target
        smoothed = updated_results.fittedvalues[target_col].iloc[-1]

        # Approximate confidence interval
        # (using standard error of filtered state)
        std_error = np.sqrt(updated_results.filtered_state_cov[-1, -1, -1])
        lower = smoothed - 1.96 * std_error
        upper = smoothed + 1.96 * std_error

        return smoothed, lower, upper

    def run_nowcast_cycle(
        self,
        data: pd.DataFrame,
        estimation_date: pd.Timestamp,
        target_date: pd.Timestamp
    ) -> Dict:
        """
        Complete nowcast cycle: estimate model, generate nowcast, track revision.
        """
        logger.info(f"Nowcasting {target_date.strftime('%Y-%m')} as of {estimation_date.strftime('%Y-%m-%d')}")

        # Estimate model with data available as of estimation_date
        results = self.estimate_model(data, as_of_date=estimation_date)

        # Generate nowcast
        nowcast, lower, upper = self.generate_nowcast(results, data, target_date)

        # Check if actual is available (for revision tracking)
        target_col = f"{self.config['target_series']}_transformed"
        actual = data.loc[target_date, target_col] if target_date in data.index else None

        nowcast_record = {
            'estimation_date': estimation_date,
            'target_date': target_date,
            'nowcast': nowcast,
            'lower_bound': lower,
            'upper_bound': upper,
            'actual': actual,
            'error': actual - nowcast if actual is not None else None
        }

        self.nowcast_history.append(nowcast_record)

        logger.info(f"Nowcast: {nowcast:.2f} [{lower:.2f}, {upper:.2f}]")
        if actual is not None:
            logger.info(f"Actual: {actual:.2f}, Error: {actual - nowcast:.2f}")

        return nowcast_record


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_nowcast_evolution(
    nowcast_history: List[Dict],
    output_dir: Path
) -> None:
    """
    Plot how nowcast evolves as new data arrives.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(nowcast_history)

    # Group by target date
    target_dates = df['target_date'].unique()

    fig, axes = plt.subplots(
        len(target_dates), 1,
        figsize=(12, 4*len(target_dates))
    )

    if len(target_dates) == 1:
        axes = [axes]

    for i, target_date in enumerate(target_dates):
        target_df = df[df['target_date'] == target_date].sort_values('estimation_date')

        ax = axes[i]

        # Plot nowcast evolution
        ax.plot(
            target_df['estimation_date'],
            target_df['nowcast'],
            marker='o',
            linewidth=2,
            label='Nowcast'
        )

        # Confidence bands
        ax.fill_between(
            target_df['estimation_date'],
            target_df['lower_bound'],
            target_df['upper_bound'],
            alpha=0.3
        )

        # Actual value if available
        if target_df['actual'].notna().any():
            actual_value = target_df['actual'].iloc[-1]
            ax.axhline(
                y=actual_value,
                color='red',
                linestyle='--',
                linewidth=2,
                label=f'Actual: {actual_value:.2f}'
            )

        ax.set_title(
            f'Nowcast Evolution for {target_date.strftime("%Y-%m")}',
            fontsize=12,
            fontweight='bold'
        )
        ax.set_xlabel('Estimation Date')
        ax.set_ylabel('Nowcast Value')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'nowcast_evolution.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_dir / 'nowcast_evolution.png'}")
    plt.close()


def plot_nowcast_accuracy(
    nowcast_history: List[Dict],
    output_dir: Path
) -> None:
    """
    Plot nowcast errors and accuracy metrics.
    """
    df = pd.DataFrame(nowcast_history)
    df_with_actuals = df[df['actual'].notna()].copy()

    if len(df_with_actuals) == 0:
        logger.warning("No actual values available yet for accuracy assessment")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Nowcast vs Actual
    axes[0].scatter(
        df_with_actuals['actual'],
        df_with_actuals['nowcast'],
        alpha=0.6,
        s=100
    )
    axes[0].plot(
        [df_with_actuals['actual'].min(), df_with_actuals['actual'].max()],
        [df_with_actuals['actual'].min(), df_with_actuals['actual'].max()],
        'r--',
        linewidth=2,
        label='Perfect Nowcast'
    )
    axes[0].set_xlabel('Actual', fontsize=12)
    axes[0].set_ylabel('Nowcast', fontsize=12)
    axes[0].set_title('Nowcast vs Actual', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot 2: Error distribution
    axes[1].hist(df_with_actuals['error'], bins=15, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Nowcast Error', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Error Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)

    # Add RMSE
    rmse = np.sqrt(np.mean(df_with_actuals['error']**2))
    mae = np.mean(np.abs(df_with_actuals['error']))
    axes[1].text(
        0.05, 0.95,
        f'RMSE: {rmse:.3f}\nMAE: {mae:.3f}',
        transform=axes[1].transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=10
    )

    plt.tight_layout()
    plt.savefig(output_dir / 'nowcast_accuracy.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_dir / 'nowcast_accuracy.png'}")
    plt.close()


# ============================================================================
# MAIN NOWCASTING SYSTEM
# ============================================================================

def run_nowcasting_system(config: Dict) -> pd.DataFrame:
    """
    Execute complete nowcasting system.

    Returns:
        DataFrame with nowcast history
    """
    logger.info("="*70)
    logger.info("REAL-TIME NOWCASTING SYSTEM")
    logger.info("="*70)

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize components
    data_manager = NowcastDataManager(config)
    engine = NowcastEngine(config)

    # Prepare data
    data = data_manager.prepare_modeling_data(force_refresh=False)

    # TODO: Customize nowcasting schedule
    # Example: Nowcast current quarter using data available at different points

    # Get current quarter
    last_date = data.index[-1]
    current_quarter_end = pd.Timestamp(last_date.year, ((last_date.month-1)//3 + 1)*3, 1)

    # Simulate nowcasts as new monthly data arrives
    # (In production, this would run on a schedule as new data is released)

    logger.info("\nSimulating nowcast updates as new data arrives...")

    # Example: Generate nowcasts for current quarter using data from last 3 months
    for months_back in [2, 1, 0]:
        estimation_date = last_date - pd.DateOffset(months=months_back)

        if estimation_date >= current_quarter_end:
            continue

        engine.run_nowcast_cycle(
            data=data,
            estimation_date=estimation_date,
            target_date=current_quarter_end
        )

    # Create visualizations
    if engine.nowcast_history:
        plot_nowcast_evolution(engine.nowcast_history, output_dir)
        plot_nowcast_accuracy(engine.nowcast_history, output_dir)

        # Export nowcast history
        history_df = pd.DataFrame(engine.nowcast_history)
        history_path = output_dir / 'nowcast_history.csv'
        history_df.to_csv(history_path, index=False)
        logger.info(f"Saved: {history_path}")

    logger.info("="*70)
    logger.info("NOWCASTING COMPLETE")
    logger.info(f"Results saved to: {output_dir.absolute()}")
    logger.info("="*70)

    return pd.DataFrame(engine.nowcast_history)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run nowcasting system
    nowcast_results = run_nowcasting_system(CONFIG)

    # TODO: Add custom post-processing
    # Example: Send alerts if nowcast deviates significantly from expectations

    if not nowcast_results.empty:
        print("\n" + "="*70)
        print("LATEST NOWCASTS")
        print("="*70)
        print(nowcast_results[['estimation_date', 'target_date', 'nowcast', 'lower_bound', 'upper_bound']].tail())

        # Performance summary
        if nowcast_results['actual'].notna().any():
            print("\n" + "="*70)
            print("ACCURACY METRICS")
            print("="*70)
            errors = nowcast_results['error'].dropna()
            print(f"RMSE: {np.sqrt(np.mean(errors**2)):.3f}")
            print(f"MAE:  {np.mean(np.abs(errors)):.3f}")
            print(f"Mean Error: {np.mean(errors):.3f}")
