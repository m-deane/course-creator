"""
GDP Nowcasting Model - Reference Solution
==========================================

This is a complete, production-ready implementation of a GDP nowcasting system.

Features:
- Real-time data fetching from FRED
- Robust handling of missing data and ragged edge
- Dynamic factor model estimation (Stock-Watson approach)
- Bridge equation for GDP nowcasting
- Comprehensive visualization dashboard
- Logging and error handling
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from fredapi import Fred
from sklearn.decomposition import PCA
from statsmodels.tsa.api import VAR
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG_FILE = "config.yaml"
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# DATA PIPELINE
# ============================================================================

def load_config():
    """Load configuration with validation."""
    if not Path(CONFIG_FILE).exists():
        logger.warning(f"{CONFIG_FILE} not found. Creating default config.")
        default_config = create_default_config()
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(default_config, f)
        return default_config

    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)

    # Validate
    if 'fred' not in config or config['fred']['api_key'] == 'YOUR_API_KEY':
        logger.error("Please set valid FRED API key in config.yaml")
        raise ValueError("Invalid FRED API key")

    return config


def create_default_config():
    """Create default configuration."""
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
    """Curated list of nowcasting indicators."""
    return [
        # Real activity
        'INDPRO', 'PAYEMS', 'HOUST', 'RETAILX', 'CUMFNS',
        # Prices
        'CPIAUCSL', 'PCEPI', 'PPIFGS',
        # Financial
        'GS10', 'FEDFUNDS', 'SP500',
        # Labor market
        'UNRATE', 'AWHMAN', 'ICSA',
    ]


def fetch_fred_data(api_key, series_list, start_date='2000-01-01'):
    """Fetch data from FRED with error handling."""
    fred = Fred(api_key=api_key)
    data = pd.DataFrame()
    failed = []

    logger.info(f"Fetching {len(series_list)} series from FRED...")

    for series in series_list:
        try:
            series_data = fred.get_series(series, observation_start=start_date)
            data[series] = series_data
            logger.info(f"  ✓ {series}")
        except Exception as e:
            logger.warning(f"  ✗ {series}: {str(e)}")
            failed.append(series)

    if failed:
        logger.warning(f"Failed to fetch {len(failed)} series: {failed}")

    # Save raw data
    data.to_csv(DATA_DIR / 'fred_data_raw.csv')
    logger.info(f"Raw data saved to {DATA_DIR / 'fred_data_raw.csv'}")

    return data


def get_transformation_codes():
    """FRED-MD transformation codes for each series."""
    return {
        # Level series (rates, already in %)
        'GS10': 2, 'FEDFUNDS': 2, 'UNRATE': 2,
        # Growth rates (log difference)
        'INDPRO': 5, 'PAYEMS': 5, 'HOUST': 5, 'RETAILX': 5,
        'CPIAUCSL': 5, 'PCEPI': 5, 'PPIFGS': 5, 'SP500': 5,
        # First difference
        'CUMFNS': 2, 'AWHMAN': 2, 'ICSA': 2,
    }


def transform_series(data, transformation_codes=None):
    """Apply FRED-MD transformations."""
    if transformation_codes is None:
        transformation_codes = get_transformation_codes()

    transformed = pd.DataFrame(index=data.index)

    for col in data.columns:
        code = transformation_codes.get(col, 5)  # Default: log-difference
        series = data[col].dropna()

        if len(series) == 0:
            continue

        if code == 1:  # No transformation
            transformed[col] = series
        elif code == 2:  # First difference
            transformed[col] = series.diff()
        elif code == 4:  # Log
            transformed[col] = np.log(series.clip(lower=0.001))
        elif code == 5:  # Log difference
            transformed[col] = np.log(series.clip(lower=0.001)).diff()

    return transformed


def standardize_data(data):
    """Robust standardization (handle outliers)."""
    # Remove extreme outliers (> 5 std)
    data_clean = data.copy()
    for col in data_clean.columns:
        mean, std = data_clean[col].mean(), data_clean[col].std()
        data_clean[col] = data_clean[col].clip(mean - 5*std, mean + 5*std)

    # Standardize
    return (data_clean - data_clean.mean()) / data_clean.std()


# ============================================================================
# FACTOR MODEL
# ============================================================================

class StockWatsonFactorModel:
    """Dynamic Factor Model with Stock-Watson two-step estimation."""

    def __init__(self, n_factors=3, factor_lags=2):
        self.n_factors = n_factors
        self.factor_lags = factor_lags

    def fit(self, X):
        """Estimate factors via PCA and dynamics via VAR."""
        logger.info(f"Estimating {self.n_factors}-factor model...")

        # Step 1: PCA
        self.pca = PCA(n_components=self.n_factors)
        factors = self.pca.fit_transform(X.dropna())

        self.loadings = self.pca.components_.T
        self.explained_variance = self.pca.explained_variance_ratio_

        self.factors = pd.DataFrame(
            factors,
            index=X.dropna().index,
            columns=[f'Factor_{i+1}' for i in range(self.n_factors)]
        )

        logger.info(f"  Total variance explained: {self.explained_variance.sum():.1%}")

        # Step 2: VAR
        if self.factor_lags > 0:
            self.var_model = VAR(self.factors)
            self.var_result = self.var_model.fit(maxlags=self.factor_lags, ic='aic')
            logger.info(f"  VAR({self.var_result.k_ar}) estimated")

        return self

    def predict_factors(self, steps=1):
        """Forecast factors."""
        forecast = self.var_result.forecast(
            self.factors.values[-self.factor_lags:],
            steps=steps
        )
        return pd.DataFrame(forecast, columns=self.factors.columns)

    def get_factor_interpretation(self, series_names, top_k=5):
        """Interpret factors by loadings."""
        interpretations = {}
        for i in range(self.n_factors):
            loadings = pd.Series(
                self.loadings[:, i],
                index=series_names
            ).abs().sort_values(ascending=False)
            interpretations[f'Factor_{i+1}'] = loadings.head(top_k)
        return interpretations


# ============================================================================
# NOWCASTING
# ============================================================================

class GDPNowcaster:
    """GDP nowcasting via bridge equations."""

    def __init__(self, factor_model):
        self.factor_model = factor_model

    def fit(self, gdp_quarterly, factors_monthly):
        """Estimate bridge equation."""
        logger.info("Estimating bridge equation...")

        # Aggregate factors to quarterly (last month of quarter)
        factors_quarterly = factors_monthly.resample('Q').last()

        # Merge and align
        data = pd.concat([gdp_quarterly, factors_quarterly], axis=1).dropna()

        y = data.iloc[:, 0]
        X = data.iloc[:, 1:]
        X = sm.add_constant(X)

        # OLS estimation
        self.bridge_model = OLS(y, X).fit()

        logger.info(f"  R²: {self.bridge_model.rsquared:.3f}")
        logger.info(f"  RMSE: {np.sqrt(self.bridge_model.mse_resid):.3f}pp")

        # Store fitted values for plotting
        self.fitted = self.bridge_model.fittedvalues
        self.residuals = self.bridge_model.resid

        return self

    def nowcast(self, current_factors):
        """Generate nowcast with confidence interval."""
        if isinstance(current_factors, pd.Series):
            current_factors = current_factors.to_frame().T

        X = sm.add_constant(current_factors)
        prediction = self.bridge_model.get_prediction(X)

        return {
            'nowcast': prediction.predicted_mean[0],
            'lower': prediction.conf_int(alpha=0.10)[0, 0],
            'upper': prediction.conf_int(alpha=0.10)[0, 1],
            'std_error': prediction.se_mean[0]
        }

    def backtest(self, gdp_quarterly, factors_monthly, train_end='2020-01-01'):
        """Pseudo real-time backtest."""
        logger.info("Running backtest...")

        test_quarters = gdp_quarterly[train_end:].index
        results = []

        for quarter in test_quarters:
            # Use only data available before this quarter
            train_gdp = gdp_quarterly[:quarter]
            train_factors = factors_monthly[:quarter]

            if len(train_gdp) < 20:
                continue

            # Refit model
            self.fit(train_gdp[:-1], train_factors)

            # Nowcast current quarter
            current_factors = train_factors.resample('Q').last().loc[quarter]
            nowcast_result = self.nowcast(current_factors)

            results.append({
                'quarter': quarter,
                'actual': gdp_quarterly[quarter],
                'nowcast': nowcast_result['nowcast'],
                'error': nowcast_result['nowcast'] - gdp_quarterly[quarter]
            })

        results_df = pd.DataFrame(results)
        rmse = np.sqrt((results_df['error']**2).mean())
        mae = results_df['error'].abs().mean()

        logger.info(f"Backtest RMSE: {rmse:.3f}pp")
        logger.info(f"Backtest MAE: {mae:.3f}pp")

        return results_df


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_comprehensive_dashboard(factor_model, nowcaster, gdp_actual, nowcast_result):
    """Create multi-panel visualization dashboard."""

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Panel 1: Factors over time
    ax1 = fig.add_subplot(gs[0, :])
    for i in range(factor_model.n_factors):
        ax1.plot(factor_model.factors.index,
                factor_model.factors.iloc[:, i],
                label=f'Factor {i+1}', linewidth=1.5)
    ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax1.set_title('Estimated Factors', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Factor Value (Standardized)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Factor loadings heatmap
    ax2 = fig.add_subplot(gs[1, 0])
    loadings_df = pd.DataFrame(
        factor_model.loadings,
        columns=[f'F{i+1}' for i in range(factor_model.n_factors)]
    )
    sns.heatmap(loadings_df, cmap='RdBu_r', center=0,
                ax=ax2, cbar_kws={'label': 'Loading'})
    ax2.set_title('Factor Loadings', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Series')

    # Panel 3: Variance explained
    ax3 = fig.add_subplot(gs[1, 1])
    variance = factor_model.explained_variance
    ax3.bar(range(1, len(variance)+1), variance, color='steelblue', alpha=0.7)
    ax3.set_xlabel('Factor Number')
    ax3.set_ylabel('Variance Explained')
    ax3.set_title('Variance Explained by Each Factor', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel 4: GDP actual vs fitted
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(gdp_actual.index, gdp_actual, 'k-', linewidth=2,
            label='Actual GDP', alpha=0.7)
    if hasattr(nowcaster, 'fitted'):
        ax4.plot(nowcaster.fitted.index, nowcaster.fitted, 'r--',
                linewidth=1.5, label='Fitted', alpha=0.7)
    ax4.set_xlabel('Date')
    ax4.set_ylabel('GDP Growth (%)')
    ax4.set_title('GDP: Actual vs Fitted', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Panel 5: Current nowcast
    ax5 = fig.add_subplot(gs[2, 1])
    quarters = list(gdp_actual.index[-8:]) + [gdp_actual.index[-1]]
    values = list(gdp_actual.iloc[-8:]) + [nowcast_result['nowcast']]
    colors = ['steelblue'] * 8 + ['red']

    ax5.bar(range(len(values)), values, color=colors, alpha=0.7)
    ax5.errorbar(len(values)-1, nowcast_result['nowcast'],
                yerr=[[nowcast_result['nowcast'] - nowcast_result['lower']],
                      [nowcast_result['upper'] - nowcast_result['nowcast']]],
                fmt='none', ecolor='red', capsize=5, linewidth=2)
    ax5.set_xticks(range(len(values)))
    ax5.set_xticklabels([q.strftime('%Y-Q%q') for q in quarters], rotation=45)
    ax5.set_ylabel('GDP Growth (%)')
    ax5.set_title(f'Current Nowcast: {nowcast_result["nowcast"]:.2f}%',
                 fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    plt.suptitle('GDP Nowcasting Dashboard', fontsize=16, fontweight='bold', y=0.995)

    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Complete nowcasting pipeline."""

    print("=" * 70)
    print("GDP NOWCASTING MODEL - REFERENCE SOLUTION")
    print("=" * 70)

    # 1. Configuration
    config = load_config()

    # 2. Fetch data
    logger.info("Fetching data from FRED...")
    data_raw = fetch_fred_data(
        api_key=config['fred']['api_key'],
        series_list=config['indicators'],
        start_date='2000-01-01'
    )

    # Fetch quarterly GDP
    fred = Fred(api_key=config['fred']['api_key'])
    gdp_raw = fred.get_series('GDPC1', observation_start='2000-01-01')
    gdp_growth = (np.log(gdp_raw).diff() * 100).dropna()  # Annualized growth rate

    logger.info(f"Data: {data_raw.shape[0]} months, {data_raw.shape[1]} series")
    logger.info(f"GDP: {len(gdp_growth)} quarters")

    # 3. Transform data
    logger.info("Transforming data...")
    data_transformed = transform_series(data_raw)
    data_std = standardize_data(data_transformed)

    # 4. Estimate factor model
    logger.info("Estimating factor model...")
    factor_model = StockWatsonFactorModel(
        n_factors=config['model']['n_factors'],
        factor_lags=config['model']['factor_lags']
    )
    factor_model.fit(data_std)

    # Interpret factors
    interpretations = factor_model.get_factor_interpretation(data_std.columns)
    logger.info("\nFactor interpretation:")
    for factor_name, top_series in interpretations.items():
        print(f"\n  {factor_name}:")
        for series, loading in top_series.items():
            print(f"    {series}: {loading:.3f}")

    # 5. Estimate bridge equation
    nowcaster = GDPNowcaster(factor_model)
    nowcaster.fit(gdp_growth, factor_model.factors)

    # 6. Generate nowcast
    current_factors = factor_model.factors.resample('Q').last().iloc[-1]
    nowcast_result = nowcaster.nowcast(current_factors)

    print(f"\n{'=' * 70}")
    print(f"NOWCAST FOR {gdp_growth.index[-1].strftime('%Y-Q%q')}")
    print(f"{'=' * 70}")
    print(f"  Estimate:  {nowcast_result['nowcast']:>6.2f}%")
    print(f"  90% CI:    [{nowcast_result['lower']:>6.2f}%, {nowcast_result['upper']:>6.2f}%]")
    print(f"  Std Error: {nowcast_result['std_error']:>6.2f}%")
    print(f"{'=' * 70}\n")

    # 7. Backtest
    backtest_results = nowcaster.backtest(gdp_growth, factor_model.factors)

    # 8. Create comprehensive dashboard
    logger.info("Creating dashboard...")
    fig = create_comprehensive_dashboard(
        factor_model, nowcaster, gdp_growth, nowcast_result
    )
    fig.savefig(OUTPUT_DIR / 'nowcast_dashboard.png', dpi=150, bbox_inches='tight')
    logger.info(f"✓ Dashboard saved: {OUTPUT_DIR / 'nowcast_dashboard.png'}")

    # Save results
    results_summary = pd.DataFrame([{
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'target_quarter': gdp_growth.index[-1].strftime('%Y-Q%q'),
        'nowcast': nowcast_result['nowcast'],
        'lower_90': nowcast_result['lower'],
        'upper_90': nowcast_result['upper'],
        'n_factors': factor_model.n_factors,
        'variance_explained': factor_model.explained_variance.sum(),
        'bridge_r2': nowcaster.bridge_model.rsquared
    }])
    results_summary.to_csv(OUTPUT_DIR / 'nowcast_results.csv', index=False)

    logger.info(f"\n{'=' * 70}")
    logger.info("COMPLETE! Results saved to 'output/' folder.")
    logger.info(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
