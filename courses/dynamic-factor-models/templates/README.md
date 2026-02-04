# Production Templates - Dynamic Factor Models

Production-ready Python templates for building DFM applications. Each template works out-of-the-box with clear TODO markers for customization.

## Available Templates

### 1. `dfm_pipeline_template.py` - End-to-End DFM Pipeline
**What it does:**
Complete pipeline from data loading to forecasting with visualization and export.

**Use when:**
- Building a complete DFM analysis from scratch
- Need a structured workflow for regular analysis
- Want production-grade logging and error handling

**Quick start:**
```python
python dfm_pipeline_template.py
```

**Customize:**
- `CONFIG['fred_series']` - Add your data series
- `CONFIG['n_factors']` - Number of latent factors
- `CONFIG['factor_order']` - AR order for factor dynamics
- `CONFIG['forecast_horizon']` - Forecast steps ahead

**Time to working:** 5-10 minutes

**Outputs:**
- `dfm_results/extracted_factors.csv` - Time series of factors
- `dfm_results/forecasts.csv` - Out-of-sample forecasts
- `dfm_results/factors.png` - Factor plots
- `dfm_results/loadings.png` - Factor loading chart
- `dfm_results/model_summary.txt` - Full estimation results

---

### 2. `nowcasting_template.py` - Real-Time Nowcasting System
**What it does:**
Production nowcasting system with incremental updates, mixed-frequency data handling, and revision tracking.

**Use when:**
- Building a real-time economic indicator tracker
- Combining monthly indicators with quarterly targets (e.g., GDP)
- Need to track how nowcasts evolve as new data arrives

**Quick start:**
```python
python nowcasting_template.py
```

**Customize:**
- `CONFIG['target_series']` - Quarterly variable to nowcast (default: GDP)
- `CONFIG['monthly_indicators']` - High-frequency predictors
- `CONFIG['n_factors']` - Number of factors
- `CONFIG['use_cache']` - Enable/disable data caching

**Time to working:** 10-15 minutes

**Outputs:**
- `nowcast_results/nowcast_history.csv` - Nowcast evolution over time
- `nowcast_results/nowcast_evolution.png` - How nowcasts change with new data
- `nowcast_results/nowcast_accuracy.png` - Accuracy metrics

**Key features:**
- Handles missing data automatically via Kalman filter
- Caches data for fast updates
- Tracks forecast revisions
- Produces confidence intervals

---

## Installation Requirements

Both templates require:
```bash
pip install statsmodels pandas numpy matplotlib pandas-datareader
```

Optional (for nowcasting):
```bash
pip install yfinance  # For financial data
```

---

## Usage Pattern

All templates follow this structure:

1. **Configuration Section** - Customize parameters here
   ```python
   CONFIG = {
       'data_source': 'fred',
       'n_factors': 2,
       # ... more settings
   }
   ```

2. **TODO Markers** - Look for these to customize:
   ```python
   # TODO: Update this value
   'fred_series': ['INDPRO', 'PAYEMS'],  # TODO: Add your series
   ```

3. **Run It** - Execute directly:
   ```python
   if __name__ == "__main__":
       results = run_pipeline(CONFIG)
   ```

4. **Import and Extend** - Use as module:
   ```python
   from dfm_pipeline_template import run_pipeline, estimate_dfm

   # Customize config
   my_config = CONFIG.copy()
   my_config['n_factors'] = 3

   # Run
   results = run_pipeline(my_config)
   ```

---

## Template Design Principles

Each template:
- ✅ Works out-of-the-box with `python template.py`
- ✅ Uses real data sources (FRED, Yahoo Finance)
- ✅ Includes production patterns (logging, error handling, config)
- ✅ Has clear TODO markers for customization
- ✅ Exports results in standard formats (CSV, PNG)
- ✅ Includes comprehensive docstrings
- ✅ Handles edge cases (missing data, convergence issues)

---

## Next Steps

1. **Start with basic pipeline:** Run `dfm_pipeline_template.py` with default settings
2. **Customize for your data:** Update `CONFIG` with your series
3. **Add domain logic:** Extend functions with your specific needs
4. **Deploy:** Add to your production workflow

For copy-paste code snippets, see `../recipes/`

For troubleshooting, see `../recipes/troubleshooting.md`
