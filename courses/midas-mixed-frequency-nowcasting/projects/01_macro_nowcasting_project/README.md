# Project 01: Macro Nowcasting System

## Overview

Build a complete nowcasting pipeline for three macroeconomic targets — GDP growth, CPI inflation, and nonfarm payrolls — and produce a brief analytical report comparing your models to published central bank forecasts.

This is a guided project. Each deliverable is specified in detail. Complete all deliverables in order; later deliverables depend on earlier ones.

---

## Learning Goals

By completing this project you will demonstrate:

- End-to-end pipeline construction (data → features → estimation → evaluation → reporting)
- Correct pseudo-real-time evaluation methodology
- News decomposition analysis for communication
- Multi-target modelling with appropriate MIDAS specifications
- Written economic analysis linking model outputs to real-world events

---

## Data Sources

Use real FRED data via the `data_ingestion_template.py` if you have a FRED API key.
Without an API key, use the synthetic CSV resources provided in `modules/module_07_macro_applications/resources/` as starting data and extend them with your own synthetic generation.

**Required series**:

| Series | FRED ID | Frequency | Notes |
|--------|---------|-----------|-------|
| Real GDP growth | GDPC1 | Quarterly | Target 1 |
| CPI All Items | CPIAUCSL | Monthly | Target 2 |
| Nonfarm Payrolls | PAYEMS | Monthly | Target 3 |
| ISM Manufacturing PMI | NAPM | Monthly | Predictor |
| Industrial Production | INDPRO | Monthly | Predictor |
| Retail Sales | RETAILSL | Monthly | Predictor |
| Initial Claims | ICSA | Weekly | Predictor for payrolls |
| Crude oil price | DCOILWTICO | Daily | Predictor for CPI |
| ADP private payrolls | ADPWNUSXNSA | Monthly | Predictor for payrolls |

---

## Deliverable 1: Data Pipeline (2 hours)

**File**: `deliverable_01_data_pipeline.py`

Build a `DataPipeline` class that:

1. Loads all nine series (from FRED or synthetic fallback)
2. Stores them in a SQLite vintage database (use the `VintageDatabase` from `templates/nowcasting_pipeline.py`)
3. Implements `get_dataset(as_of_date)` that returns a dict of all series as of the given date
4. Handles missing series gracefully (log a warning, continue with remaining series)

**Verification**:
```python
pipeline = DataPipeline(db_path="data/project01_vintages.db")
dataset = pipeline.get_dataset(as_of_date="2023-10-15")
assert "PAYEMS" in dataset
assert "GDPC1" in dataset
assert len(dataset["PAYEMS"]) > 40  # at least 10 years of monthly data
```

---

## Deliverable 2: GDP Nowcasting Model (2 hours)

**File**: `deliverable_02_gdp_nowcast.ipynb`

Build a quarterly GDP nowcasting model:

1. **Specification**: ElasticNet MIDAS with PAYEMS, INDPRO, RETAILSL, NAPM (4 indicators, 3 monthly lags each)
2. **Ragged-edge**: Carry-forward fill for all indicators; simulate 3 forecast dates per quarter
3. **Estimation**: ElasticNetCV with TimeSeriesSplit(n_splits=5)
4. **Evaluation**: Expanding-window OOS, minimum 8 evaluation quarters
5. **Benchmark**: AR(1) on GDP growth

**Required outputs**:
- Nowcast evolution chart for the most recent quarter in your data
- Expanding-window RMSE plot with AR(1) benchmark line
- News decomposition waterfall for the last nowcast revision
- Printed DM test result (ElasticNet vs AR1)

**Acceptance criteria**:
- OOS RMSE < AR(1) RMSE
- At least 8 OOS evaluation periods
- News decomposition contributions sum to the observed revision (within floating-point tolerance)

---

## Deliverable 3: Inflation Nowcasting Model (1.5 hours)

**File**: `deliverable_03_inflation_nowcast.ipynb`

Build a monthly CPI nowcasting model using the bottom-up approach:

1. **Energy CPI**: MIDAS with daily crude oil returns, K=22 lags, Beta weights estimated by NLS
2. **Core (AR proxy)**: AR(1) on core CPI (CPI minus energy and food)
3. **Aggregate**: `CPI_hat = 0.07 * energy_hat + 0.93 * core_hat` (approximate weights)

**Required outputs**:
- Beta weight profile for the energy MIDAS
- Comparison chart: full-CPI MIDAS vs bottom-up vs AR(1)
- RMSE and MAE table for all three approaches
- One-paragraph interpretation: which component drives CPI forecast accuracy?

**Acceptance criteria**:
- Bottom-up approach clearly outperforms simple AR(1)
- Beta weights are monotone-decreasing or hump-shaped (economically plausible)
- All three approaches evaluated on the same OOS period

---

## Deliverable 4: Labour Market Nowcasting Model (1.5 hours)

**File**: `deliverable_04_labour_nowcast.ipynb`

Build a monthly payrolls nowcasting model:

1. **MIDAS-W**: Monthly payrolls with weekly claims, K=4 weekly lags
2. **Enhanced**: Add ADP private payrolls (monthly lag 1) as an additional predictor
3. **Evaluation**: Expanding-window OOS, compare MIDAS-W vs enhanced vs AR(1)

**Required outputs**:
- Beta weight profile for the claims-to-payrolls MIDAS-W
- Scatter plot: actual payrolls vs predicted payrolls with R² annotation
- RMSE table with 95% DM test p-value vs AR(1)
- Economic interpretation: does the ADP addition improve accuracy significantly?

**Acceptance criteria**:
- Negative estimated coefficient on claims (high claims → lower payrolls)
- MIDAS-W outperforms AR(1) on RMSE
- DM test result correctly computed with Newey-West variance

---

## Deliverable 5: Model Monitoring Dashboard (1 hour)

**File**: `deliverable_05_monitoring.py`

Build a monitoring script that:

1. Loads OOS errors from Deliverables 2–4
2. Computes rolling 8-quarter RMSE for each model
3. Runs the bias t-test for each model
4. Runs CUSUM test on GDP nowcasting errors
5. Generates the plain-text health report (use `generate_health_report()` from `modules/module_08_production_systems/guides/02_monitoring_reporting_guide.md`)

**Required outputs**:
- Console: health report for each of the three targets
- File `reports/monitoring_summary.txt`: the three health reports concatenated

**Acceptance criteria**:
- Script runs end-to-end without errors
- Health report includes RMSE, MAE, bias, CUSUM status, and re-estimation recommendation

---

## Deliverable 6: Analytical Report (2 hours)

**File**: `deliverable_06_report.md`

Write a 600–900 word report with the following sections:

### 6.1 Executive Summary (100 words)
One-paragraph summary of results across all three models.

### 6.2 GDP Nowcasting
- Which indicators contributed most to forecast revisions (from news decomposition)?
- How does your RMSE compare to published FRBNY Nowcast numbers (typically 0.3–0.6 percentage points for GDP)?
- What limitations does your simplified approach have vs the NY Fed's full DFM?

### 6.3 Inflation Nowcasting
- Is the bottom-up approach worth the complexity vs a simple AR?
- Which component (energy vs core) drives the majority of CPI forecast errors?
- What would you add if you had access to PPI sub-components?

### 6.4 Labour Market Nowcasting
- Does the claims-to-payrolls relationship appear stable over your sample?
- How did the 2020 COVID shock affect the claims-payrolls model?
- What is the marginal value of ADP vs claims alone?

### 6.5 Monitoring and Production Readiness
- Are any of your models biased? Which trigger (if any) would fire on your data?
- What would you need to do to deploy this system for live publication?

---

## Grading Rubric (self-assessment)

Rate yourself on each dimension before submitting for peer review.

| Dimension | 1 (basic) | 3 (proficient) | 5 (expert) |
|-----------|-----------|----------------|-----------|
| Data pipeline | Loads some series | All series, vintage DB | Real-time safe, handles failures |
| GDP model | AR benchmark only | ElasticNet + eval | News decomp + DM test + evolution chart |
| Inflation model | Single CPI AR | Bottom-up two components | Daily oil MIDAS + beta weights + correct aggregation |
| Labour model | Simple regression | MIDAS-W with claims | Enhanced + DM test + economic interpretation |
| Monitoring | RMSE only | RMSE + bias test | CUSUM + trigger + health report |
| Report | Description of methods | Results with numbers | Economic interpretation + limitations |

**Target**: score 3+ on every dimension before considering the project complete.

---

## Directory Structure

Organise your project files as follows:

```
01_macro_nowcasting_project/
├── README.md                          (this file)
├── deliverable_01_data_pipeline.py
├── deliverable_02_gdp_nowcast.ipynb
├── deliverable_03_inflation_nowcast.ipynb
├── deliverable_04_labour_nowcast.ipynb
├── deliverable_05_monitoring.py
├── deliverable_06_report.md
├── data/
│   └── project01_vintages.db
└── reports/
    └── monitoring_summary.txt
```

---

## Resources

- `templates/nowcasting_pipeline.py` — pipeline architecture
- `templates/midas_regression_template.py` — MIDAS model fitting
- `templates/forecast_evaluation_template.py` — expanding-window evaluation and DM test
- `templates/data_ingestion_template.py` — FRED data loading
- `modules/module_07_macro_applications/` — all macro nowcasting guides and notebooks
- `modules/module_08_production_systems/` — monitoring and production guides
- `recipes/ragged_edge_handling.py` — publication-lag-aware data availability
- `recipes/forecast_combination.py` — ensemble methods if you want to extend beyond Deliverable 4

---

## Extension Challenges

If you complete all six deliverables and want to push further:

1. **Extension A**: Add a Nelson-Siegel term structure factor (yield curve level/slope/curvature) as a predictor in the GDP model. Does it add value?
2. **Extension B**: Replace ElasticNet MIDAS with an XGBoost model using the flat-stacked feature engineering from Module 05 Notebook 03. Run a DM test comparing the two.
3. **Extension C**: Implement a forecast combination of all three target-specific models and evaluate whether the combined model is more accurate than any individual model.
4. **Extension D**: Build a simple web dashboard (Flask or Streamlit) that displays the latest nowcast for all three targets with their evolution charts.
