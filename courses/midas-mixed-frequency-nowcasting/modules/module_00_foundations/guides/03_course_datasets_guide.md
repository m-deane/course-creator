# Course Datasets Guide

> **Reading time:** ~20 min | **Module:** 00 — Foundations | **Prerequisites:** None (entry point)


## In Brief


<div class="callout-key">

**Key Concept Summary:** This course uses real macroeconomic and financial data throughout. All datasets are sourced from FRED (Federal Reserve Bank of St. Louis) and Yahoo Finance. CSV fallbacks are provided in each module's

</div>

This course uses real macroeconomic and financial data throughout. All datasets are sourced from FRED (Federal Reserve Bank of St. Louis) and Yahoo Finance. CSV fallbacks are provided in each module's `resources/` directory so notebooks run offline.

## Key Insight

<div class="callout-insight">

**Insight:** The mixed-frequency approach preserves within-period dynamics that aggregation destroys. This is especially valuable when the timing of high-frequency movements carries economic information.

</div>


Working with real data from the beginning exposes the data challenges that make mixed-frequency modeling interesting: revision history, ragged edges, missing observations, and structural breaks. Synthetic data hides these complexities.

---

## Primary Data Sources

<div class="callout-warning">

**Warning:** Be cautious about extrapolating MIDAS performance from stable periods to crisis periods. The relationship between high-frequency indicators and the low-frequency target can shift dramatically during regime changes.

</div>


### FRED (Federal Reserve Economic Data)

FRED is the primary source for macroeconomic data. The St. Louis Fed maintains over 800,000 series from 107 sources. All series used in this course are publicly available.

**API Access:**


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# Install: pip install fredapi
from fredapi import Fred

# Get a free API key at https://fred.stlouisfed.org/docs/api/api_key.html
fred = Fred(api_key='your_key_here')

# Or set the environment variable FRED_API_KEY
import os
fred = Fred(api_key=os.environ.get('FRED_API_KEY'))
```

</div>
</div>

**Fallback (no API key required):**


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import pandas as pd

# All CSV files are in each module's resources/ directory
gdp = pd.read_csv('resources/gdp_quarterly.csv', index_col=0, parse_dates=True)
ip = pd.read_csv('resources/industrial_production_monthly.csv', index_col=0, parse_dates=True)
```

</div>
</div>

---

## Series Reference

### Quarterly Series (Target Variables)

| FRED ID | Description | Units | Start |
|---------|-------------|-------|-------|
| `GDPC1` | Real GDP | Billions 2017 USD, SAAR | 1947-Q1 |
| `GDPC1` (pct chg) | Real GDP Growth | Quarter-over-quarter % | 1947-Q2 |
| `PCECC96` | Real Personal Consumption | Billions 2017 USD, SAAR | 1947-Q1 |
| `GPDIC1` | Real Gross Private Investment | Billions 2017 USD, SAAR | 1947-Q1 |

**How to download:**


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import pandas as pd
from fredapi import Fred

fred = Fred(api_key='your_key_here')

# Real GDP, quarterly
gdp_raw = fred.get_series('GDPC1', observation_start='2000-01-01')
gdp_growth = gdp_raw.pct_change() * 100  # quarter-over-quarter growth rate

# Note: FRED returns pandas Series with DatetimeIndex
print(f"GDP series: {len(gdp_growth)} observations")
print(f"Date range: {gdp_growth.index[0]} to {gdp_growth.index[-1]}")
print(f"Last value: {gdp_growth.iloc[-1]:.3f}%")
```

</div>
</div>

### Monthly Series (Primary Regressors)

| FRED ID | Description | Units | Frequency | Start |
|---------|-------------|-------|-----------|-------|
| `INDPRO` | Industrial Production Index | Index 2017=100 | Monthly | 1919-01 |
| `PAYEMS` | Nonfarm Payrolls | Thousands of persons | Monthly | 1939-01 |
| `RSAFS` | Advance Retail Sales | Millions USD | Monthly | 1992-01 |
| `UNRATE` | Unemployment Rate | Percent | Monthly | 1948-01 |
| `CPIAUCSL` | CPI All Items | Index 1982-84=100 | Monthly | 1947-01 |
| `HOUST` | Housing Starts | Thousands of units, SAAR | Monthly | 1959-01 |
| `ISMMFG` | ISM Manufacturing PMI | Index | Monthly | 1948-01 |

**How to download:**

```python
# Industrial Production
ip_raw = fred.get_series('INDPRO', observation_start='2000-01-01')
ip_growth = ip_raw.pct_change() * 100  # month-over-month growth

# Nonfarm Payrolls (change in thousands)
payrolls = fred.get_series('PAYEMS', observation_start='2000-01-01')
payrolls_chg = payrolls.diff()  # monthly change

# Handle the alignment: monthly series needs to be mapped to quarterly periods
# We'll do this in the notebooks using pandas Period indexing
ip_quarterly_index = ip_growth.resample('QE').last().index
print(f"IP available through: {ip_growth.index[-1]}")
```

### Daily/Weekly Series (High-Frequency Regressors)

| ID | Source | Description | Frequency |
|----|--------|-------------|-----------|
| `^GSPC` | Yahoo Finance | S&P 500 Index | Daily |
| `T10Y2Y` | FRED | 10Y-2Y Treasury Spread | Daily |
| `DCOILWTICO` | FRED | WTI Crude Oil Price | Daily |
| `DGS10` | FRED | 10-Year Treasury Yield | Daily |
| `VIXCLS` | FRED | CBOE VIX | Daily |
| `DEXUSEU` | FRED | USD/EUR Exchange Rate | Daily |

**How to download:**

```python
import yfinance as yf

# S&P 500 daily returns
sp500 = yf.download('^GSPC', start='2000-01-01', progress=False)
sp500_returns = sp500['Adj Close'].pct_change().dropna() * 100

# FRED daily series
vix = fred.get_series('VIXCLS', observation_start='2000-01-01')
spread_10y2y = fred.get_series('T10Y2Y', observation_start='2000-01-01')

print(f"S&P 500: {len(sp500_returns)} daily observations")
print(f"VIX: {len(vix)} daily observations")
```

---

## Data Quality and Preprocessing

### Missing Values

FRED data contains `NaN` for non-business days and for dates before series begin. Standard preprocessing:

```python
def preprocess_fred_series(series, freq='M', method='ffill'):
    """
    Clean a FRED series for use in MIDAS models.

    Parameters
    ----------
    series : pd.Series
        Raw FRED series with DatetimeIndex.
    freq : str
        Target frequency ('Q', 'M', 'D').
    method : str
        Method for filling missing values ('ffill', 'bfill', 'interpolate').

    Returns
    -------
    clean : pd.Series
        Preprocessed series.
    """
    # Convert to Period index for clean frequency handling
    if freq == 'Q':
        clean = series.resample('QE').last()
    elif freq == 'M':
        clean = series.resample('ME').last()
    else:
        clean = series.copy()

    # Fill forward (most appropriate for financial/macro series)
    if method == 'ffill':
        clean = clean.ffill()
    elif method == 'interpolate':
        clean = clean.interpolate(method='linear')

    # Remove leading NaNs
    clean = clean.loc[clean.first_valid_index():]

    return clean
```

### Vintage vs. Revised Data

An important distinction for realistic evaluation:

- **Revised data (current vintage):** What FRED serves by default — the most recent, fully revised values. Use for developing models.
- **Real-time (vintage) data:** What was available at a specific point in time. Use for honest backtesting.

```python
# FRED-MD (McCracken & Ng) provides vintage data
# Available at: https://research.stlouisfed.org/econ/mccracken/fred-databases/

# For this course, we use current-vintage data and note the caveat:
# "In-sample fit will overstate real-time performance due to data revisions."
```

---

## CSV Fallback Files

All CSV files use the following convention:

- **Index column:** `date` (ISO 8601 format: YYYY-MM-DD or YYYY-MM-01 for monthly)
- **Value column:** Series-specific name (e.g., `gdp_growth`, `ip_growth`)
- **Coverage:** 2000-01-01 through 2024-12-31

### Loading CSV Fallbacks

```python
import pandas as pd
import os

def load_series(series_name, data_dir='resources', use_fred=True, fred_api_key=None):
    """
    Load a data series, falling back to CSV if FRED is unavailable.

    Parameters
    ----------
    series_name : str
        FRED series ID (e.g., 'GDPC1') or local name (e.g., 'gdp_quarterly').
    data_dir : str
        Directory containing CSV fallback files.
    use_fred : bool
        Attempt FRED download first.
    fred_api_key : str or None
        FRED API key. If None, checks FRED_API_KEY environment variable.

    Returns
    -------
    series : pd.Series
    """
    # Map FRED IDs to local CSV filenames
    csv_map = {
        'GDPC1': 'gdp_quarterly.csv',
        'INDPRO': 'industrial_production_monthly.csv',
        'PAYEMS': 'payrolls_monthly.csv',
        '^GSPC': 'sp500_daily.csv',
        'T10Y2Y': 'treasury_spread_daily.csv',
        'UNRATE': 'unemployment_monthly.csv',
        'VIXCLS': 'vix_daily.csv',
    }

    if use_fred and fred_api_key is not None:
        try:
            from fredapi import Fred
            fred = Fred(api_key=fred_api_key)
            return fred.get_series(series_name, observation_start='2000-01-01')
        except Exception as e:
            print(f"FRED download failed ({e}), falling back to CSV.")

    # Fall back to local CSV
    csv_name = csv_map.get(series_name, f"{series_name.lower()}.csv")
    csv_path = os.path.join(data_dir, csv_name)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV fallback not found: {csv_path}")

    series = pd.read_csv(csv_path, index_col=0, parse_dates=True).squeeze()
    return series
```

---

## Module-by-Module Dataset Usage

| Module | Primary Datasets | Frequency | Purpose |
|--------|-----------------|-----------|---------|
| 00 | GDP, IP | Q, M | Visualization, information loss demo |
| 01 | GDP growth, IP growth | Q, M | MIDAS regression basics |
| 02 | GDP growth, IP, Payrolls | Q, M | Estimation, model selection |
| 03 | GDP, IP, Payrolls, Retail Sales | Q, M | Nowcasting with ragged edge |
| 04 | GDP, IP, Payrolls, Retail, VIX, Spread | Q, M, D | DFM + Factor-MIDAS |

---

## Constructing the MIDAS Data Matrix

A critical step in any MIDAS notebook is constructing the aligned data matrix that maps high-frequency observations to low-frequency time periods.

```python
import pandas as pd
import numpy as np

def build_midas_matrix(y_low, x_high, n_lags, freq_ratio):
    """
    Build the MIDAS data matrix aligning high-frequency regressors
    to low-frequency observations.

    Parameters
    ----------
    y_low : pd.Series
        Low-frequency dependent variable (length T).
    x_high : pd.Series
        High-frequency regressor (length T * freq_ratio).
    n_lags : int
        Number of high-frequency lags to include.
    freq_ratio : int
        Frequency ratio (3 for monthly/quarterly, 65 for daily/quarterly).

    Returns
    -------
    Y : np.ndarray, shape (T,)
        Dependent variable.
    X : np.ndarray, shape (T, n_lags)
        MIDAS regressor matrix. X[t, j] = x_{m*t - j} in high-frequency time.
    dates : pd.DatetimeIndex
        Dates for Y observations.
    """
    T = len(y_low)
    Y = y_low.values
    dates = y_low.index

    X = np.full((T, n_lags), np.nan)
    x_vals = x_high.values

    for t in range(T):
        # Map quarterly index t to end of quarter in high-frequency time
        hf_end = (t + 1) * freq_ratio - 1  # last high-freq index in quarter t

        for j in range(n_lags):
            hf_idx = hf_end - j
            if 0 <= hf_idx < len(x_vals):
                X[t, j] = x_vals[hf_idx]

    # Drop rows with NaNs
    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(Y)
    return Y[valid], X[valid], dates[valid]
```

---

## Common Data Pitfalls in This Course

**Pitfall 1: Misaligned frequencies.** When combining quarterly and monthly series, ensure dates align to the same quarter-end convention. Use `pd.Period` for unambiguous alignment.

```python
# Convert monthly series to quarterly Period for alignment
ip_monthly = pd.read_csv('resources/industrial_production_monthly.csv',
                          index_col=0, parse_dates=True).squeeze()
ip_quarterly_periods = ip_monthly.index.to_period('Q')
```

**Pitfall 2: Using pct_change vs. diff.** GDP and IP are typically expressed as percentage growth rates (pct_change), while payrolls changes are expressed in thousands (diff).

**Pitfall 3: NaN at series start after differencing.** After pct_change() or diff(), the first observation is NaN. Always call .dropna() before alignment.

**Pitfall 4: FRED series multiplied by 100.** Some FRED series return decimal fractions; others return percentages. Check units before modeling.

---

## Connections

- **Builds on:** Guide 01 (mixed-frequency problem), Guide 02 (aggregation strategies)
- **Used in:** Every notebook in Modules 00–04
- **Related to:** Real-time data analysis, vintage data, data revisions

---

## Practice Problems

<div class="callout-danger">

**Danger:** Never use future information when constructing the high-frequency regressor matrix. In a real-time nowcasting context, you only have data up to the current date -- using the full quarter of monthly data when nowcasting mid-quarter is a look-ahead bias that invalidates your results.

</div>


1. Download the INDPRO series from FRED (or load from CSV). Convert to monthly growth rates. Resample to quarterly frequency using three different aggregation methods (last, mean, sum). Plot all three. In which quarter does the choice of aggregation method matter most?

2. The S&P 500 has approximately 65 trading days per calendar quarter. If you want to use 4 quarters of daily data as MIDAS lags, how many high-frequency lags do you need? Write out the expression for the total number of parameters in an unrestricted MIDAS model with this setup.

3. Identify a structural break in the IP growth series. How would you handle this break in a MIDAS regression?

---

## Further Reading

- FRED Documentation: https://fred.stlouisfed.org/docs/api/fred/
- McCracken, M., & Ng, S. (2016). "FRED-MD: A monthly database for macroeconomic research." *Journal of Business & Economic Statistics.*
- Ghysels, E., & Qian, H. (2019). "Estimating MIDAS regressions via OLS with polynomial parameter profiling." *Econometrics and Statistics.*


---

## Cross-References

<a class="link-card" href="./01_mixed_frequency_problem_guide.md">
  <div class="link-card-title">01 Mixed Frequency Problem</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_mixed_frequency_problem_slides.md">
  <div class="link-card-title">01 Mixed Frequency Problem — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./02_traditional_solutions_guide.md">
  <div class="link-card-title">02 Traditional Solutions</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_traditional_solutions_slides.md">
  <div class="link-card-title">02 Traditional Solutions — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

