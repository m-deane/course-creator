# Recipes - Dynamic Factor Models

Copy-paste code patterns for common DFM tasks. Each recipe is self-contained and ready to use in your projects.

## Available Recipes

### `common_patterns.py` - Core DFM Operations

**10 essential patterns for working with Dynamic Factor Models**

| Recipe | What It Does | Lines of Code |
|--------|-------------|---------------|
| 1. Factor Extraction Comparison | Compare PCA vs DFM factors | < 20 |
| 2. State Space Model Setup | Configure DFM state space | < 10 |
| 3. Kalman Filter Initialization | Set initial conditions | < 15 |
| 4. Factor Rotation | Apply varimax rotation | < 15 |
| 5. Variance Decomposition | Decompose common vs idiosyncratic variance | < 20 |
| 6. Factor News Decomposition | Attribute forecast revisions to data releases | < 20 |
| 7. Model Selection | Choose optimal factors and AR order via IC | < 20 |
| 8. Rolling Forecast Evaluation | Out-of-sample forecast accuracy | < 20 |
| 9. Handle Missing Data | Estimate DFM with NaN values | < 15 |
| 10. Convert to VAR | Express DFM in reduced-form VAR | < 15 |

**Example usage:**
```python
from recipes.common_patterns import extract_factors_comparison

# Compare PCA and DFM factors
factors_df = extract_factors_comparison(data, n_factors=2)
print(factors_df.head())
```

---

### `data_loading.py` - Data Preparation Patterns

**10 practical recipes for loading and preparing time series data**

| Recipe | What It Does | Use Case |
|--------|-------------|----------|
| 1. Load FRED Data | Download from Federal Reserve API | Macro indicators |
| 2. Load Yahoo Finance | Download stock/index data | Financial indicators |
| 3. Load CSV Time Series | Import local CSV with dates | Custom data |
| 4. Handle Missing Values | Interpolate, forward-fill, drop | Data cleaning |
| 5. Align Mixed Frequencies | Combine monthly + quarterly data | Nowcasting |
| 6. Standardize Data | Z-score, min-max, robust scaling | Pre-processing |
| 7. Apply Transformations | Log, diff, growth rates | Stationarity |
| 8. Load Complete Macro Dataset | Ready-to-use macro dataset | Quick start |
| 9. Save/Load Processed Data | Cache processed data | Speed |
| 10. Create Lagged Features | Build lagged variables | Forecasting |

**Example usage:**
```python
from recipes.data_loading import load_fred_data, standardize_data

# Load and prepare data
data = load_fred_data(['INDPRO', 'PAYEMS', 'UNRATE'], start_date='2010-01-01')
data_std = standardize_data(data, method='zscore')
```

**Quick start - complete macro dataset in one line:**
```python
from recipes.data_loading import load_complete_macro_dataset

# Get standardized, transformed macro dataset ready for DFM
data = load_complete_macro_dataset(start_date='2000-01-01', include_financial=True)
# Returns: Industrial production, employment, retail, housing, prices, rates
```

---

### `troubleshooting.md` - Error Solutions

**Comprehensive guide to fixing common DFM errors**

Quick reference for:
- ❌ Convergence issues → Solutions for max iterations errors
- ❌ Singular matrix errors → Fix multicollinearity problems
- ❌ Missing data handling → Strategies for NaN values
- ❌ Identification problems → Ensure unique factor loadings
- ❌ Numerical instability → Prevent overflow/underflow
- ❌ Forecast errors → Fix constant or nonsensical forecasts
- ❌ Memory issues → Optimize for large datasets
- ❌ Parameter interpretation → Make economic sense of results

**Example:**
```markdown
Problem: "Maximum iterations reached"

Solution 1: Reduce model complexity
model = DynamicFactor(data, k_factors=1, factor_order=2)

Solution 2: Standardize data first
data_std = (data - data.mean()) / data.std()
```

---

## Quick Start Guide

### 1. Factor Extraction (2 minutes)

```python
import pandas as pd
from recipes.common_patterns import extract_factors_comparison
from recipes.data_loading import load_complete_macro_dataset

# Load ready-to-use macro data
data = load_complete_macro_dataset(start_date='2010-01-01')

# Extract and compare factors
factors = extract_factors_comparison(data, n_factors=2)

# Plot
import matplotlib.pyplot as plt
factors.plot(figsize=(12, 6))
plt.title('PCA vs DFM Factors')
plt.show()
```

### 2. Model Selection (3 minutes)

```python
from recipes.common_patterns import select_model_specification
from recipes.data_loading import load_complete_macro_dataset

# Load data
data = load_complete_macro_dataset(start_date='2010-01-01')

# Find optimal model specification
results = select_model_specification(data, max_factors=3, max_ar=3)

print("Best model by BIC:")
print(results.head(1))
```

### 3. Nowcasting Setup (5 minutes)

```python
from recipes.data_loading import load_fred_data, align_mixed_frequencies
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

# Load monthly indicators
monthly = load_fred_data(['INDPRO', 'PAYEMS'], start_date='2010-01-01')

# Load quarterly GDP
quarterly = load_fred_data(['GDPC1'], start_date='2010-01-01')

# Align frequencies
data = align_mixed_frequencies(monthly, quarterly, quarterly_method='end')

# Estimate DFM for nowcasting
model = DynamicFactor(data.dropna(), k_factors=1, factor_order=2)
results = model.fit(disp=False)

# Nowcast current quarter
nowcast = results.forecast(steps=3)
print(f"GDP nowcast: {nowcast[-1, -1]:.2f}")
```

---

## Common Workflows

### Workflow 1: Basic DFM Analysis

```python
# 1. Load data
from recipes.data_loading import load_complete_macro_dataset
data = load_complete_macro_dataset(start_date='2000-01-01')

# 2. Select model
from recipes.common_patterns import select_model_specification
model_specs = select_model_specification(data, max_factors=3, max_ar=3)
best_spec = model_specs.iloc[0]

# 3. Estimate
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
model = DynamicFactor(
    data,
    k_factors=int(best_spec['n_factors']),
    factor_order=int(best_spec['ar_order'])
)
results = model.fit(disp=True)

# 4. Analyze
from recipes.common_patterns import variance_decomposition
decomp = variance_decomposition(results)
print(decomp)
```

### Workflow 2: Forecasting with DFM

```python
# 1. Load and prepare
from recipes.data_loading import load_fred_data, standardize_data
data = load_fred_data(['INDPRO', 'PAYEMS', 'UNRATE'], start_date='2010-01-01')
data = standardize_data(data)

# 2. Evaluate forecast accuracy
from recipes.common_patterns import rolling_forecast_evaluation
forecast_results = rolling_forecast_evaluation(
    data,
    n_factors=1,
    factor_order=2,
    forecast_horizon=3,
    window_size=60
)

# 3. Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(forecast_results['date'], forecast_results['actual'], label='Actual')
plt.plot(forecast_results['date'], forecast_results['forecast'], label='Forecast')
plt.legend()
plt.show()
```

### Workflow 3: Handling Missing Data

```python
# 1. Load data with missing values
from recipes.data_loading import load_fred_data
data = load_fred_data(['INDPRO', 'PAYEMS', 'RETAIL'], start_date='2020-01-01')

# 2. Check missing pattern
print(data.isnull().sum())

# 3. Handle missing
from recipes.data_loading import handle_missing_values
data_clean = handle_missing_values(data, method='interpolate', max_consecutive=3)

# 4. Estimate (or use EM directly)
from recipes.common_patterns import estimate_with_missing_data
results, imputed = estimate_with_missing_data(data)  # Uses Kalman filter
```

---

## Installation

All recipes require:
```bash
pip install statsmodels pandas numpy matplotlib
```

For data loading:
```bash
pip install pandas-datareader yfinance
```

Optional for advanced recipes:
```bash
pip install scikit-learn scipy joblib
```

---

## Recipe Format

Each recipe follows this pattern:

```python
def recipe_name(input_data: pd.DataFrame, **kwargs):
    """
    Input: Clear description of expected input
    Output: Clear description of output

    Brief explanation of what the recipe does
    """
    # Problem statement as comment
    # Solution in < 20 lines
    # Clear input → output

    return result
```

---

## Tips for Using Recipes

1. **Copy entire function** - Each recipe is self-contained
2. **Check input format** - Read docstring for expected data structure
3. **Customize parameters** - Modify kwargs for your use case
4. **Combine recipes** - Chain multiple recipes for complex workflows
5. **Check troubleshooting** - If errors occur, see `troubleshooting.md`

---

## Contributing Your Own Recipes

Good recipes are:
- ✅ Self-contained (< 20 lines)
- ✅ Solve one specific problem
- ✅ Have clear input/output
- ✅ Include problem statement as comment
- ✅ Work with copy-paste (no mocks)

---

## Next Steps

1. **Explore recipes:** Run `python common_patterns.py` to see examples
2. **Try data loading:** Run `python data_loading.py` for data prep demos
3. **Use in your project:** Import and use in your scripts
4. **Read troubleshooting:** Bookmark `troubleshooting.md` for quick reference

For complete end-to-end pipelines, see `../templates/`
