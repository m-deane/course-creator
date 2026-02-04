"""
Multi-Country Factor Model - Starter Code
==========================================

Hierarchical factor model for international macroeconomic data.

Structure:
- Level 1: Global factors (affect all countries)
- Level 2: Regional/country-specific factors

Run: python starter_code.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from statsmodels.tsa.api import VAR
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

COUNTRIES = {
    'US': {
        'gdp': 'GDPC1',
        'cpi': 'CPIAUCSL',
        'unemployment': 'UNRATE',
        'ip': 'INDPRO'
    },
    'Germany': {
        'gdp': 'CLVMNACSCAB1GQDE',
        'cpi': 'DEUCPIALLMINMEI',
        'unemployment': 'LRUNTTTTDEQ156S',
        'ip': 'DEUPRO NOSTSAM'
    },
    'Japan': {
        'gdp': 'JPNRGDPEXP',
        'cpi': 'JPNCPIALLMINMEI',
        'unemployment': 'LRUNTTTTJPQ156S',
        'ip': 'JPNPROINDMISMEI'
    },
    # TODO: Add more countries (UK, China, Canada, etc.)
}

REGIONAL_GROUPS = {
    'Americas': ['US', 'Canada'],
    'Europe': ['Germany', 'France', 'UK', 'Italy'],
    'Asia': ['Japan', 'China', 'South Korea']
}


# ============================================================================
# DATA PIPELINE
# ============================================================================

def fetch_international_data(countries, start_date='2000-01-01'):
    """
    Fetch data for multiple countries.

    TODO: Implement this function
    - Loop over countries and indicators
    - Create multi-index DataFrame (country, date)
    - Handle missing/discontinued series
    """
    print("Fetching international data...")

    # Placeholder: Generate synthetic data
    dates = pd.date_range(start_date, '2024-01-01', freq='Q')
    data_list = []

    for country, indicators in countries.items():
        for ind_name, ind_code in indicators.items():
            # Simulate data
            series = np.random.randn(len(dates)).cumsum() + 100
            for date, value in zip(dates, series):
                data_list.append({
                    'country': country,
                    'indicator': ind_name,
                    'date': date,
                    'value': value
                })

    df = pd.DataFrame(data_list)
    df_pivot = df.pivot_table(
        values='value',
        index='date',
        columns=['country', 'indicator']
    )

    return df_pivot


def transform_and_standardize(data):
    """
    Transform to growth rates and standardize.

    TODO: Apply country-specific transformations
    """
    # Growth rates (log difference)
    data_growth = np.log(data).diff()

    # Standardize
    data_std = (data_growth - data_growth.mean()) / data_growth.std()

    return data_std.dropna()


# ============================================================================
# HIERARCHICAL FACTOR MODEL
# ============================================================================

class HierarchicalFactorModel:
    """
    Two-level factor model:
    y_{it} = Λ_global · f_global + Λ_regional · f_regional + ε_{it}
    """

    def __init__(self, n_global=2, n_regional=1):
        self.n_global = n_global
        self.n_regional = n_regional

    def fit(self, panel_data, regional_groups=None):
        """
        Estimate hierarchical factor model.

        Steps:
        1. Extract global factors from full panel
        2. Compute residuals
        3. Extract regional factors from residuals (by region)
        4. Estimate factor dynamics

        TODO: Implement each step
        """
        print(f"\nEstimating hierarchical model...")
        print(f"  Global factors: {self.n_global}")
        print(f"  Regional factors per region: {self.n_regional}")

        # Step 1: Global factors
        # TODO: Run PCA on full panel
        pca_global = PCA(n_components=self.n_global)
        self.global_factors = pd.DataFrame(
            pca_global.fit_transform(panel_data.dropna()),
            index=panel_data.dropna().index,
            columns=[f'Global_{i+1}' for i in range(self.n_global)]
        )
        self.global_loadings = pca_global.components_.T
        self.global_variance = pca_global.explained_variance_ratio_

        print(f"  Global variance explained: {self.global_variance.sum():.1%}")

        # Step 2: Regional factors
        # TODO: Extract from residuals by region
        self.regional_factors = {}
        self.regional_loadings = {}

        # Placeholder: Skip regional factors for now
        # In full implementation: loop over regional_groups

        return self

    def forecast(self, steps=4):
        """
        Forecast global and regional factors.

        TODO: Fit VAR to factors, generate forecast
        """
        # Combine all factors
        all_factors = self.global_factors  # Add regional if implemented

        # VAR forecast
        var_model = VAR(all_factors)
        var_result = var_model.fit(maxlags=2)

        forecast = var_result.forecast(
            all_factors.values[-2:],
            steps=steps
        )

        return pd.DataFrame(
            forecast,
            columns=all_factors.columns
        )


# ============================================================================
# VARIANCE DECOMPOSITION
# ============================================================================

def decompose_variance(model, panel_data):
    """
    For each country-indicator, compute:
    - Variance from global factors
    - Variance from regional factors
    - Residual variance (country-specific)

    TODO: Implement variance decomposition
    Returns: DataFrame with columns [global_%, regional_%, country_%]
    """
    print("\nVariance decomposition...")

    # Reconstruct fitted values
    fitted_global = model.global_factors @ model.global_loadings.T

    # Compute variance components
    total_var = panel_data.var()
    global_var = fitted_global.var()

    decomp = pd.DataFrame({
        'global_pct': (global_var / total_var * 100),
        'country_pct': ((total_var - global_var) / total_var * 100)
    })

    return decomp


# ============================================================================
# SPILLOVER ANALYSIS
# ============================================================================

def estimate_spillovers(country_factors, horizon=4):
    """
    Compute spillover index (Diebold-Yilmaz).

    Method: Forecast Error Variance Decomposition from VAR.

    TODO: Full implementation
    1. Estimate VAR for country-level factors
    2. Compute generalized impulse responses
    3. Variance decomposition → spillover matrix
    """
    print("\nEstimating spillovers...")

    # Placeholder: Random spillover matrix
    n = len(country_factors.columns)
    spillover_matrix = np.random.rand(n, n)
    spillover_matrix = spillover_matrix / spillover_matrix.sum(axis=0)

    return pd.DataFrame(
        spillover_matrix,
        index=country_factors.columns,
        columns=country_factors.columns
    )


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_global_factors(factors):
    """Plot global factors over time."""
    fig, axes = plt.subplots(factors.shape[1], 1, figsize=(12, 6))

    for i, col in enumerate(factors.columns):
        ax = axes if factors.shape[1] == 1 else axes[i]
        ax.plot(factors.index, factors[col], linewidth=1.5)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Date')
    plt.suptitle('Global Factors', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_variance_decomposition(decomp):
    """Stacked bar chart of variance decomposition."""
    fig, ax = plt.subplots(figsize=(12, 6))

    decomp.plot(kind='bar', stacked=True, ax=ax,
                color=['steelblue', 'coral'])

    ax.set_ylabel('Variance Explained (%)')
    ax.set_xlabel('Country-Indicator')
    ax.set_title('Variance Decomposition: Global vs Country-Specific',
                 fontsize=14, fontweight='bold')
    ax.legend(['Global', 'Country-Specific'])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig


def plot_spillover_matrix(spillover_matrix):
    """Heatmap of spillover matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(spillover_matrix, annot=True, fmt='.2f',
                cmap='YlOrRd', ax=ax, vmin=0, vmax=1)

    ax.set_title('Cross-Country Spillover Matrix',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Shock From')
    ax.set_ylabel('Impact On')

    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("MULTI-COUNTRY HIERARCHICAL FACTOR MODEL")
    print("=" * 70)

    # 1. Fetch data
    print("\n[1/5] Fetching data...")
    data = fetch_international_data(COUNTRIES)
    print(f"  Panel shape: {data.shape}")

    # 2. Transform and standardize
    print("\n[2/5] Transforming data...")
    data_std = transform_and_standardize(data)
    print(f"  Standardized shape: {data_std.shape}")

    # 3. Estimate hierarchical model
    print("\n[3/5] Estimating model...")
    model = HierarchicalFactorModel(n_global=2, n_regional=1)
    model.fit(data_std, REGIONAL_GROUPS)

    # 4. Variance decomposition
    print("\n[4/5] Variance decomposition...")
    decomp = decompose_variance(model, data_std)
    print("\nAverage variance decomposition:")
    print(f"  Global:  {decomp['global_pct'].mean():.1f}%")
    print(f"  Country: {decomp['country_pct'].mean():.1f}%")

    # 5. Spillover analysis
    print("\n[5/5] Spillover analysis...")
    # For spillover, need country-level factors
    # Placeholder: Use global factors
    spillovers = estimate_spillovers(model.global_factors)
    print(f"  Total spillover index: {spillovers.values.sum():.1%}")

    # 6. Visualizations
    print("\nCreating visualizations...")

    fig1 = plot_global_factors(model.global_factors)
    fig1.savefig('output/global_factors.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: output/global_factors.png")

    fig2 = plot_variance_decomposition(decomp)
    fig2.savefig('output/variance_decomposition.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: output/variance_decomposition.png")

    fig3 = plot_spillover_matrix(spillovers)
    fig3.savefig('output/spillover_matrix.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: output/spillover_matrix.png")

    print("\n" + "=" * 70)
    print("COMPLETE! Check 'output/' folder for results.")
    print("=" * 70)


if __name__ == "__main__":
    import os
    os.makedirs('output', exist_ok=True)
    main()
