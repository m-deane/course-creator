# Project 2: Multi-Country Factor Model (Intermediate)

## What You'll Build

A **global macroeconomic monitoring system** that extracts common factors from multiple countries' economic indicators, distinguishing between global shocks, regional factors, and country-specific dynamics.

**End result:** A hierarchical factor model that:
- Identifies global factors (affect all countries)
- Extracts regional factors (US, Europe, Asia)
- Estimates country-specific components
- Tracks international spillovers and contagion
- Forecasts cross-country correlations

**Difficulty:** Intermediate
**Time:** 6-8 hours
**Prerequisites:** Project 1 completed, understanding of panel data, VAR models

---

## Learning Goals

1. **Hierarchical Factor Models**
   - Two-level factor structure (global + country)
   - Identification in multi-level models
   - Variance decomposition across levels

2. **Panel Data Techniques**
   - Cross-sectional and time-series dimensions
   - Balancing vs. unbalanced panels
   - Country-specific transformations

3. **Spillover Analysis**
   - Measuring cross-country linkages
   - Shock decomposition (global vs. idiosyncratic)
   - Contagion vs. interdependence

4. **International Economics**
   - Distinguishing global business cycles from local
   - Trade linkages and factor loadings
   - Financial market integration

---

## Project Structure

```
project_2_intermediate/
├── README.md
├── starter_code.py        # Your starting point
├── solution.py            # Reference implementation
├── deploy.md              # Deployment guide
├── data/
│   ├── country_indicators/  # Data by country
│   └── global_factors/      # Estimated factors
└── output/
    ├── global_factors.csv
    ├── variance_decomposition.png
    └── spillover_network.html
```

---

## Data Sources

### Countries Covered
- **US:** GDP, CPI, unemployment, industrial production
- **Eurozone:** Same indicators for Germany, France, Italy, Spain
- **Asia:** Japan, China, South Korea
- **Other:** UK, Canada, Australia

### FRED International Series
```python
COUNTRY_INDICATORS = {
    'US': ['GDPC1', 'CPIAUCSL', 'UNRATE', 'INDPRO'],
    'Germany': ['CLVMNACSCAB1GQDE', 'DEUCPIALLMINMEI', ...],
    'Japan': ['JPNRGDPEXP', 'JPNCPIALLMINMEI', ...],
    # ... (15-20 countries total)
}
```

---

## Tasks

### Phase 1: Data Pipeline (2-3 hours)

**Task 1.1:** Fetch multi-country data
```python
def fetch_international_data(countries, indicators):
    """
    Fetch data for multiple countries from FRED.
    Return: Multi-index DataFrame (country, date)
    """
    pass
```

**Task 1.2:** Harmonize frequencies and transformations
```python
def harmonize_panel(data, target_frequency='Q'):
    """
    Convert all series to common frequency.
    Handle: Monthly + quarterly data
    """
    pass
```

**Task 1.3:** Balance the panel
```python
def balance_panel(data, method='interpolate'):
    """
    Handle missing data across countries.
    Methods: 'interpolate', 'drop', 'EM'
    """
    pass
```

### Phase 2: Hierarchical Factor Model (2-3 hours)

**Task 2.1:** Estimate two-level factor model
```python
class HierarchicalFactorModel:
    """
    Level 1: Global factors (affect all countries)
    Level 2: Regional factors (affect groups of countries)
    """

    def __init__(self, n_global=2, n_regional=3):
        pass

    def fit(self, panel_data, country_groups):
        """
        1. Extract global factors from full panel
        2. Extract regional factors from residuals
        3. Estimate factor dynamics (VAR)
        """
        pass
```

**Task 2.2:** Variance decomposition
```python
def decompose_variance(model, country_data):
    """
    For each country-series, compute:
    - % variance from global factors
    - % variance from regional factors
    - % variance from country-specific
    """
    pass
```

**Task 2.3:** Test factor structure
```python
def test_global_vs_country(data):
    """
    Test H0: All variation is country-specific (no global factors)
    Use: Likelihood ratio test
    """
    pass
```

### Phase 3: Spillover Analysis (2-3 hours)

**Task 3.1:** Estimate spillover network
```python
def estimate_spillovers(country_factors):
    """
    Generalized Forecast Error Variance Decomposition.
    Returns: N x N spillover matrix
    """
    pass
```

**Task 3.2:** Identify crisis transmission
```python
def detect_contagion(factors, crisis_dates):
    """
    Compare factor correlations:
    - Normal times vs. crisis times
    - Test for structural break in correlations
    """
    pass
```

**Task 3.3:** Forecast cross-country correlations
```python
def forecast_correlations(model, horizon=4):
    """
    Forecast future correlation matrix.
    Useful for international portfolio allocation.
    """
    pass
```

### Phase 4: Visualization Dashboard (1-2 hours)

**Task 4.1:** Global factor plot
```python
def plot_global_factors(factors, crisis_periods):
    """
    Time series of global factors with:
    - Recession shading
    - Crisis annotations (2008, COVID, etc.)
    """
    pass
```

**Task 4.2:** Variance decomposition heatmap
```python
def plot_variance_decomposition(decomposition):
    """
    Stacked bar chart: global vs regional vs country
    Across countries and variables
    """
    pass
```

**Task 4.3:** Interactive spillover network
```python
def create_spillover_network(spillover_matrix):
    """
    Interactive network graph (Plotly):
    - Nodes = countries
    - Edges = spillover strength
    - Color = regions
    """
    pass
```

---

## Success Criteria

Your model is successful if:

1. **Global factors are interpretable:**
   - Factor 1: Global growth (GDP indicators load strongly)
   - Factor 2: Global inflation (price indicators load strongly)

2. **Variance decomposition makes sense:**
   - Global factors explain 30-50% of variance (large, open economies)
   - Regional factors explain 10-30%
   - Country factors explain remainder

3. **Spillover network shows known relationships:**
   - Strong US → everyone spillovers
   - Regional clustering (Europe, Asia)
   - Financial spillovers stronger than real economy

4. **Forecasting performance:**
   - Multi-country model outperforms country-by-country models
   - RMSE improvement of 10-20% for small open economies

---

## Example Output

```
=== HIERARCHICAL FACTOR MODEL ===

Data: 15 countries, 4 variables each, 80 quarters (2000-2020)

Global Factors (2):
  Factor 1: Global Growth
    - Top loadings: US GDP (0.85), Germany GDP (0.78), China IP (0.72)
    - Variance explained: 35%
  Factor 2: Global Inflation
    - Top loadings: US CPI (0.81), Japan CPI (0.65), UK CPI (0.70)
    - Variance explained: 18%

Regional Factors (3):
  Americas: 12% variance
  Europe: 15% variance
  Asia: 10% variance

Variance Decomposition (average across countries):
  Global:  42%
  Regional: 23%
  Country: 35%

Spillover Analysis:
  US → Rest of World: 28% (largest transmitter)
  China → Asia: 22%
  Germany → Europe: 19%

  Total Spillover Index: 65% (high integration)
```

---

## Extensions

### Extension 1: Time-Varying Integration
- Allow factor loadings to change over time
- Measure increasing/decreasing globalization
- Test pre- vs post-2008 integration

### Extension 2: Financial Contagion
- Add financial variables (stock returns, bond spreads)
- Detect crisis transmission channels
- Early warning indicators

### Extension 3: Trade-Weighted Factors
- Weight countries by trade linkages
- Construct country-specific external factors
- Test whether trade explains factor structure

### Extension 4: Real-Time Nowcasting
- Nowcast GDP for all countries simultaneously
- Use global factors to improve country forecasts
- Handle different publication lags across countries

---

## Key References

### Foundational Papers
- **Kose, Otrok & Whiteman (2003):** "International Business Cycles: World, Region, and Country-Specific Factors." *American Economic Review*
- **Stock & Watson (2005):** "Understanding Changes in International Business Cycle Dynamics." *JEconomic Perspectives*

### Spillover Methods
- **Diebold & Yilmaz (2012):** "Better to Give than to Receive: Predictive Directional Measurement of Volatility Spillovers." *International Journal of Forecasting*
- **Pesaran & Shin (1998):** "Generalized Impulse Response Analysis in Linear Multivariate Models." *Economics Letters*

### Applications
- **ECB:** Uses multi-country factor models for Eurozone nowcasting
- **IMF:** Global spillover reports use GVAR (Global VAR) models

---

## Getting Help

**Common Issues:**

1. **Unbalanced panel:** Countries have different data availability
   - **Solution:** Use EM algorithm to handle missing data

2. **Too many parameters:** Model doesn't converge
   - **Solution:** Reduce number of factors, increase sample size

3. **Loadings not interpretable:** Mixed factor meanings
   - **Solution:** Try varimax rotation, impose sign restrictions

4. **Multicollinearity:** Global and regional factors overlap
   - **Solution:** Use orthogonal extraction, test factor structure

---

## Deliverables

1. **Code:** `solution.py` with complete pipeline
2. **Report:** 2-page summary of findings
   - Global vs regional variance decomposition
   - Spillover network visualization
   - Economic interpretation of factors
3. **Dashboard:** Interactive multi-country monitor
4. **Presentation:** 10-min presentation of key insights

---

Good luck! This project will give you experience with real-world international macroeconomic analysis used by central banks and international organizations.
