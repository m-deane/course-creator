# Project 1: GDP Nowcasting Model (Beginner)

## What You'll Build

A real-time **GDP nowcasting system** that estimates current-quarter GDP growth before official data is released, using high-frequency indicators from the FRED-MD database.

**End result:** A working nowcasting model that:
- Ingests monthly macro indicators from FRED
- Extracts 3 latent factors (real activity, inflation, financial conditions)
- Produces GDP nowcasts with uncertainty bands
- Updates automatically as new data arrives
- Generates a visualization dashboard

**Difficulty:** Beginner
**Time:** 4-6 hours
**Prerequisites:** Basic Python, pandas, understanding of factor models

---

## Learning Goals

By completing this project, you will:

1. **Data Engineering**
   - Fetch real-time data from FRED API
   - Handle mixed-frequency data (monthly indicators → quarterly GDP)
   - Deal with ragged edge (different publication lags)
   - Transform and standardize macro time series

2. **Factor Model Implementation**
   - Estimate dynamic factor model via PCA (Stock-Watson approach)
   - Extract factors and estimate factor dynamics (VAR)
   - Construct bridge equation for GDP nowcasting
   - Validate model using real-time vintages

3. **Real-World Application**
   - Implement nowcasting workflow (fetch → estimate → predict → visualize)
   - Track nowcast evolution as data arrives
   - Decompose GDP forecast into factor contributions
   - Generate professional visualization dashboard

4. **Production Practices**
   - Modular code structure (separate data, model, visualization)
   - Configuration management (YAML config file)
   - Error handling for missing/delayed data
   - Logging and monitoring

---

## Project Structure

```
project_1_beginner/
├── README.md              # This file
├── starter_code.py        # Working foundation (extend this)
├── solution.py            # Reference implementation
├── deploy.md              # Deployment instructions
├── config.yaml            # Configuration (API keys, parameters)
├── data/                  # Downloaded data (gitignored)
├── output/                # Nowcasts and plots
└── tests/                 # Unit tests
```

---

## Getting Started

### 1. Environment Setup

```bash
# Create environment
conda create -n gdp-nowcast python=3.11
conda activate gdp-nowcast

# Install dependencies
pip install pandas numpy statsmodels matplotlib seaborn
pip install fredapi pyyaml plotly
```

### 2. Get FRED API Key

1. Sign up at https://fred.stlouisfed.org/
2. Get API key: https://fredapi.stlouisfed.org/docs/api/api_key.html
3. Create `config.yaml`:

```yaml
fred:
  api_key: "YOUR_API_KEY_HERE"

model:
  n_factors: 3
  factor_lags: 2
  target: "GDP"
  indicators:
    - "INDPRO"      # Industrial Production
    - "PAYEMS"      # Employment
    - "CPIAUCSL"    # CPI
    - "HOUST"       # Housing Starts
    # ... (30-50 indicators total)

nowcast:
  current_quarter: "2024-Q3"
  update_frequency: "weekly"
```

### 3. Run Starter Code

```bash
python starter_code.py
```

**Expected output:**
- Downloads FRED-MD data
- Estimates 3-factor model
- Produces initial GDP nowcast
- Generates plot: `output/nowcast_evolution.png`

---

## Your Tasks

### Phase 1: Data Pipeline (Beginner)

**Task 1.1:** Fetch FRED-MD data
```python
def fetch_fred_md(api_key):
    """Download FRED-MD monthly macroeconomic database."""
    # TODO: Implement data download
    # Hint: Use fredapi.Fred() and pd.DataFrame
    pass
```

**Task 1.2:** Handle transformations
```python
def transform_series(data, transformation_codes):
    """Apply FRED-MD transformation codes (1=none, 2=diff, 4=log, 5=log-diff)."""
    # TODO: Implement transformations
    # Hint: Use pd.diff() and np.log()
    pass
```

**Task 1.3:** Create ragged-edge dataset
```python
def create_ragged_edge(data, vintage_date):
    """Simulate real-time data availability as of vintage_date."""
    # TODO: Mask data that wouldn't be available yet
    # Hint: Different series have different publication lags
    pass
```

### Phase 2: Factor Model (Intermediate)

**Task 2.1:** Extract factors via PCA
```python
def extract_factors(data, n_factors=3):
    """Extract factors using Stock-Watson PCA approach."""
    # TODO: Standardize data, run PCA, return factors and loadings
    pass
```

**Task 2.2:** Estimate factor dynamics
```python
def estimate_factor_var(factors, lags=2):
    """Fit VAR model to estimated factors."""
    # TODO: Use statsmodels.tsa.api.VAR
    pass
```

**Task 2.3:** Build bridge equation
```python
def estimate_bridge_equation(gdp_quarterly, factors_monthly):
    """Estimate relationship between monthly factors and quarterly GDP."""
    # TODO: Aggregate factors to quarterly, regress GDP on lagged factors
    pass
```

### Phase 3: Nowcasting (Advanced)

**Task 3.1:** Produce nowcast
```python
def nowcast_gdp(model, current_data):
    """Generate current-quarter GDP nowcast."""
    # TODO: Extract factors, forecast using bridge equation
    pass
```

**Task 3.2:** Decompose contributions
```python
def decompose_nowcast(nowcast, factor_contributions):
    """Break down nowcast into factor contributions."""
    # TODO: Show how much each factor contributes to forecast
    pass
```

**Task 3.3:** Track nowcast evolution
```python
def track_evolution(model, data_vintages):
    """Show how nowcast changes as new data arrives."""
    # TODO: Recompute nowcast for each vintage date
    pass
```

### Phase 4: Visualization (Polish)

**Task 4.1:** Create nowcast dashboard
```python
def create_dashboard(nowcasts, uncertainty_bands):
    """Professional visualization with Plotly."""
    # TODO: Time series plot with confidence bands
    # Include: historical GDP, nowcast, official forecasts
    pass
```

**Task 4.2:** Factor interpretation plot
```python
def plot_factor_loadings(loadings, series_names):
    """Heatmap showing which series load on which factors."""
    pass
```

---

## Checkpoints

### Checkpoint 1: Data Working (30% complete)
- [ ] FRED API connection successful
- [ ] Data transformations applied correctly
- [ ] Ragged edge simulation working
- [ ] Plot raw data and verify no lookahead bias

### Checkpoint 2: Model Estimated (60% complete)
- [ ] PCA extracts 3 interpretable factors
- [ ] Factor VAR shows reasonable dynamics
- [ ] Bridge equation has R² > 0.5
- [ ] In-sample fit looks good

### Checkpoint 3: Nowcast Produced (80% complete)
- [ ] Real-time nowcast generated
- [ ] Uncertainty quantification included
- [ ] Decomposition into factors working
- [ ] Evolution tracking implemented

### Checkpoint 4: Production Ready (100% complete)
- [ ] Dashboard looks professional
- [ ] Code is modular and well-documented
- [ ] Error handling for edge cases
- [ ] Unit tests pass
- [ ] Ready for deployment

---

## Success Criteria

Your nowcasting model is successful if:

1. **Accuracy:** RMSE < 1.0pp on historical backtests (2010-2023)
2. **Timeliness:** Nowcast available 30 days before GDP release
3. **Stability:** Nowcast doesn't jump wildly with each new data point
4. **Interpretability:** Factors have clear economic meaning
5. **Robustness:** Handles missing data, outliers, data revisions

---

## Extensions (Optional)

Want to go further? Try these:

### Extension 1: Real-Time Vintages
- Use ALFRED database (real-time data archives)
- Backtest using actual data available on each date
- Measure nowcast accuracy vs. first-release GDP

### Extension 2: High-Frequency Indicators
- Add weekly data (initial jobless claims)
- Add daily data (stock prices, yields)
- Handle mixed daily/monthly/quarterly frequencies

### Extension 3: Machine Learning Enhancements
- Try LASSO for variable selection before PCA
- Random forest for bridge equation
- Ensemble of multiple nowcasting models

### Extension 4: Deploy as API
- Flask/FastAPI web service
- Automatic daily updates
- Email alerts when nowcast changes significantly

---

## Example Output

After completing the project, running:
```bash
python solution.py --quarter 2024-Q3
```

Should produce:

**Console output:**
```
=== GDP Nowcasting Model ===
Current quarter: 2024-Q3
Data vintage: 2024-08-15

Downloading FRED-MD data... ✓
Transforming series... ✓
Extracting 3 factors... ✓

Factor interpretation:
  Factor 1: Real Activity (60% variance)
    - Top loadings: INDPRO, PAYEMS, HOUST
  Factor 2: Inflation (20% variance)
    - Top loadings: CPIAUCSL, PCEPI, CPI_CORE
  Factor 3: Financial Conditions (12% variance)
    - Top loadings: GS10, BAA, SP500

Estimating factor VAR(2)... ✓
Estimating bridge equation... ✓
  In-sample R²: 0.73
  RMSE: 0.65pp

=== NOWCAST ===
Q3 2024 GDP Growth: 2.8%  [90% CI: 1.5% to 4.1%]

Contributions:
  Real Activity Factor:     +2.1%
  Inflation Factor:         +0.3%
  Financial Conditions:     +0.4%

Dashboard saved to: output/nowcast_dashboard.html
```

**Visualization:** Interactive Plotly dashboard showing:
- Historical GDP (black line)
- Current nowcast (red dot)
- 90% confidence band (gray shading)
- Evolution of nowcast over current quarter (blue line)
- Factor contributions (stacked bar chart)

---

## Resources

### Data Sources
- **FRED-MD:** https://research.stlouisfed.org/econ/mccracken/fred-databases/
- **ALFRED:** https://alfred.stlouisfed.org/ (real-time vintages)
- **FRED API Docs:** https://fred.stlouisfed.org/docs/api/

### Key Papers
- **Stock & Watson (2002):** "Forecasting Using Principal Components"
- **Giannone et al. (2008):** "Nowcasting: The Real-Time Informational Content of Macroeconomic Data"
- **Bańbura et al. (2013):** "Now-Casting and the Real-Time Data Flow"

### Code Examples
- **statsmodels state-space:** https://www.statsmodels.org/stable/statespace.html
- **NY Fed nowcasting:** https://www.newyorkfed.org/research/policy/nowcast

---

## Getting Help

**Stuck?** Check:
1. `starter_code.py` has helpful comments and structure
2. `solution.py` for reference implementation (but try yourself first!)
3. Course discussion forum
4. Office hours

**Common issues:**
- **API key error:** Make sure `config.yaml` has valid FRED API key
- **Missing data:** Some FRED series discontinued - check `transformation_codes.csv`
- **Singular matrix:** Too many factors for amount of data - reduce `n_factors`
- **Poor forecasts:** Try different indicator sets, more data preprocessing

---

Good luck! Remember: The goal is a **working prototype**, not perfection. Focus on end-to-end functionality first, then refine.
