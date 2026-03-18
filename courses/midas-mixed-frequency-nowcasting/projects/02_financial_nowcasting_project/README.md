# Project 02: Live Financial Nowcasting Dashboard

## Overview

Build an end-to-end financial nowcasting system that tracks realised volatility, VaR, and commodity prices in real time. The system runs daily, produces updated forecasts, and displays them in an interactive dashboard. This is an open-ended project — the specifications below define minimum requirements, but strong submissions will add novel features.

Unlike Project 01, this project intentionally leaves architecture decisions to you. The requirements describe *what* must be delivered, not *how* to implement it.

---

## System Requirements

### Minimum Viable System

Your system must:

1. Produce a daily realised volatility (RV) forecast for a major equity index (e.g. S&P 500)
2. Derive a daily 1% and 5% VaR from the RV forecast
3. Produce a 5-day-ahead crude oil price directional forecast (up/down)
4. Update all three forecasts daily using new market data
5. Display all three in a dashboard with at least the following panels:
   - Current RV forecast with 80% prediction interval
   - 30-day rolling RV chart (forecast vs realised)
   - VaR backtest: forecast VaR vs actual daily returns (violation plot)
   - Crude oil directional accuracy over the last 60 trading days

### Dashboard Technology

Choose one:
- **Streamlit** (`pip install streamlit`) — recommended for simplicity
- **Plotly Dash** (`pip install dash`) — recommended for interactivity
- **Jupyter Voila** — recommended if you prefer notebook-based development

Minimum: the dashboard must run locally (`streamlit run app.py` or equivalent).

---

## Deliverable Specifications

### Deliverable A: Data Layer (4 hours)

**Files**: `data/`, `src/data_loader.py`

1. Implement daily download of S&P 500 returns (Yahoo Finance: `^GSPC`)
2. Implement daily download of crude oil prices (Yahoo Finance: `CL=F` or `DCOILWTICO` from FRED)
3. Store all data in a local SQLite database or Parquet files
4. Implement `get_returns(ticker, start, end)` and `get_prices(ticker, start, end)`
5. Implement a CSV fallback for each series in case the download fails

**Minimum data requirement**: 5 years of daily data (2018-present or use the CSV resources from `modules/module_06_financial_applications/resources/`).

### Deliverable B: MIDAS-RV Model (4 hours)

**Files**: `src/rv_model.py`, `notebooks/rv_analysis.ipynb`

Implement the MIDAS-RV model from Module 06:

$$\text{RV}_{m+1} = \alpha + \beta \sum_{k=1}^{K} w(k;\,\theta) \cdot \text{RV}_{m-k+1} + \varepsilon_{m+1}$$

1. Compute monthly realised volatility from daily squared returns
2. Fit MIDAS-RV by NLS with Beta polynomial weights (K=22)
3. Compare to HAR-RV (daily/weekly/monthly RV components)
4. Evaluate with QLIKE loss function: $L = \text{RV}/\hat{\text{RV}} - \log(\text{RV}/\hat{\text{RV}}) - 1$
5. Generate prediction intervals via residual bootstrap (B=200)

**Required analysis in `rv_analysis.ipynb`**:
- Beta weight profile vs HAR weights
- Rolling 12-month QLIKE: MIDAS-RV vs HAR-RV
- Mincer-Zarnowitz regression: regress realised RV on forecast RV
- Period analysis: how did your model perform during COVID-19 (Feb–Apr 2020)?

### Deliverable C: MIDAS-VaR System (3 hours)

**Files**: `src/var_model.py`

Derive daily VaR from the monthly MIDAS-RV forecast:

$$\text{VaR}_{d,\alpha} = -z_\alpha \cdot \frac{\hat{\sigma}_{\text{monthly}}}{\sqrt{22}}$$

1. Implement `compute_var(rv_forecast, alpha)` for both 1% and 5% levels
2. Implement `backtest_var(returns, var_series)` returning violation rate, Kupiec LR, Christoffersen LR
3. Apply Basel III traffic light classification:
   - Green: 0–4 violations per 250 trading days
   - Yellow: 5–9 violations
   - Red: 10+ violations
4. Plot the violation clustering chart: daily returns coloured by VaR breach status

**Acceptance criteria**:
- Your 1% VaR should produce between 0 and 8 violations per 250 days (Green or Yellow zone)
- Kupiec test p-value > 0.05 (cannot reject correct unconditional coverage)

### Deliverable D: Crude Oil Directional Model (3 hours)

**Files**: `src/oil_model.py`

Build a model to forecast whether crude oil prices will be higher or lower 5 trading days from now:

1. **Features**: 5-day, 10-day, and 20-day log return (momentum); realised volatility; S&P 500 5-day return; VIX proxy (use 20-day historical vol of S&P 500 if VIX not available)
2. **Model**: Choose logistic regression, gradient boosting, or MIDAS-X — justify your choice
3. **Evaluation**: Directional accuracy, AUC-ROC curve, confusion matrix
4. **Baseline**: Naive momentum (if 5-day return > 0, predict up)

**Acceptance criteria**:
- Directional accuracy > 50% (beat the naive random baseline)
- AUC-ROC > 0.55 on OOS period
- Model documented: why did you choose this model over the alternatives?

### Deliverable E: Dashboard (4 hours)

**Files**: `app.py` (or `dashboard.py`), `src/dashboard_components.py`

Build the interactive dashboard with at least four panels:

**Panel 1 — Realised Volatility Tracker**
- Time series: 30-day rolling realised RV (annualised %)
- Forecast: next-month MIDAS-RV forecast with 80% prediction band
- Annotation: COVID spike, GFC spike if visible in your sample

**Panel 2 — VaR Monitor**
- Daily returns as bar chart
- VaR levels (1% and 5%) as horizontal lines
- Violations highlighted in red
- Basel III zone displayed as a coloured badge (Green/Yellow/Red)

**Panel 3 — Crude Oil Directional**
- Directional accuracy over trailing 60 trading days (rolling)
- Most recent prediction: up or down arrow with probability
- Model confidence (predicted probability, not just class label)

**Panel 4 — System Health**
- Last update timestamp
- Number of trading days since last VaR violation
- Rolling QLIKE (last 12 months) vs MIDAS-RV backtest baseline
- Alert if any model has not updated in >24 hours

**Dashboard must**:
- Load in <5 seconds on a local machine
- Update data on page refresh (or on a button press)
- Display a clear error message if data download fails (do not crash)

### Deliverable F: Technical Report (2 hours)

**File**: `report.md`

Write a 700–1000 word technical report:

#### F.1 Model Selection Rationale (150 words)
Why MIDAS-RV over GARCH? Why your chosen oil model over alternatives? What assumptions are you making?

#### F.2 Backtesting Results (200 words)
Quantitative summary of RV QLIKE, VaR coverage, oil directional accuracy. Include the most important numbers as a table.

#### F.3 Regime Analysis (200 words)
How did your models perform across at least two distinct market regimes? Analyse a low-vol period (e.g. 2017), a high-vol period (e.g. 2020), and a recovery period (e.g. 2021).

#### F.4 Production Concerns (150 words)
What are the three most important risks to monitor if this system were deployed for production use? Include: data quality, model stability, and one domain-specific risk.

#### F.5 Limitations and Extensions (150 words)
What would you add with another week of development? Which limitation of the current model concerns you most?

---

## Directory Structure

```
02_financial_nowcasting_project/
├── README.md                         (this file)
├── app.py                            (dashboard entry point)
├── report.md
├── src/
│   ├── data_loader.py
│   ├── rv_model.py
│   ├── var_model.py
│   ├── oil_model.py
│   └── dashboard_components.py
├── notebooks/
│   └── rv_analysis.ipynb
├── data/
│   ├── sp500_returns.db             (or .csv)
│   └── crude_oil_prices.db         (or .csv)
└── tests/
    ├── test_rv_model.py
    └── test_var_model.py
```

---

## Scoring Criteria (self-assessment)

This project is evaluated on four dimensions:

| Dimension | Minimum (pass) | Target | Stretch |
|-----------|---------------|--------|---------|
| Data reliability | CSV fallback works | Live data + fallback | Real-time intraday (yfinance) |
| RV model | HAR-RV only | MIDAS-RV + QLIKE eval | Beta NLS + bootstrap intervals |
| VaR system | Point VaR only | Backtested + Kupiec | Christoffersen + Basel zone |
| Dashboard | Static charts | Interactive + update | Auto-refresh + alerts |
| Report | Model description | Results + regime analysis | Economic interpretation + extensions |

**Minimum to complete the project**: pass all five minimum thresholds.
**Target**: reach the target level on at least three dimensions.

---

## Resources

- `modules/module_06_financial_applications/notebooks/01_midas_rv_sp500.ipynb` — MIDAS-RV implementation reference
- `modules/module_06_financial_applications/notebooks/03_var_mixed_frequency.ipynb` — VaR backtest reference
- `modules/module_06_financial_applications/resources/` — SP500 and crude oil CSV fallback data
- `templates/midas_regression_template.py` — `MIDASModel` class with NLS Beta weights
- `templates/data_ingestion_template.py` — `YahooFetcher` and `FREDFetcher` with caching
- `modules/module_08_production_systems/guides/02_monitoring_reporting_guide.md` — health report format

---

## Extension Challenges

Strong submissions will include at least one:

1. **Intraday RV**: compute 5-minute intraday RV using tick data from a free provider (e.g. Polygon.io free tier). Compare to daily-squared-return RV.
2. **Multi-asset VaR**: extend the system to cover a two-asset portfolio (equities + crude oil). Compute portfolio VaR with correlation using DCC-GARCH or a simplified copula.
3. **Sentiment feature**: incorporate a sentiment proxy (e.g. AAII bullish-bearish spread, available free from FRED: `AAII`) as an additional predictor in the oil directional model.
4. **Alert system**: implement an email or Slack alert that fires when the VaR model moves from Green to Yellow zone or when the oil model directional accuracy drops below 50% for 10 consecutive trading days.
5. **Model comparison page**: add a fifth dashboard panel that runs a live DM test between MIDAS-RV and HAR-RV on the trailing 252 trading days and displays whether the difference is statistically significant.
