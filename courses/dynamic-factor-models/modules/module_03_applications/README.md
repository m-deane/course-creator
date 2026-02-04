# Module 03: Applications - Nowcasting & Forecasting

## Overview

This module transitions from theory to practice, focusing on the primary real-world applications of Dynamic Factor Models: **nowcasting** (estimating current-quarter GDP before official releases), **forecasting** (predicting future values), and **handling missing data** (dealing with publication lags and ragged edges). You'll build production-ready nowcasting systems using real FRED data and learn evaluation techniques used by central banks and policy institutions.

**Duration:** 2 weeks
**Effort:** 8-10 hours per week

## Learning Objectives

By the end of this module, you will be able to:

1. **Construct** a real-time GDP nowcasting system using mixed-frequency data from FRED
2. **Handle** missing data, publication lags, and ragged-edge datasets using Kalman filtering
3. **Evaluate** forecasts using proper scoring rules, real-time vintages, and out-of-sample tests
4. **Implement** factor-augmented forecasting models that outperform univariate benchmarks
5. **Diagnose** forecast failures and model misspecification
6. **Build** production pipelines for automated nowcasting updates

## Why This Matters

**Real-world impact:**
- Federal Reserve uses DFM-based nowcasting (NY Fed Nowcast, Atlanta Fed GDPNow)
- ECB and Bank of England rely on factor models for policy decisions
- Financial institutions pay for real-time GDP estimates (hours matter for trading)
- Central banks publish nowcast uncertainty bands alongside point forecasts

**Key insight:** The value of DFMs isn't just accuracy—it's **timeliness**. A 90% accurate forecast today beats a 95% accurate forecast next month.

## Module Contents

### Guides (Detailed Concept Explanations)

1. **[Nowcasting](guides/01_nowcasting.md)** - Real-time economic monitoring with DFMs
   - Bridge equations vs state-space nowcasting
   - Ragged-edge data handling
   - News vs uncertainty decomposition
   - Production implementation patterns

2. **[Forecast Evaluation](guides/02_forecasting_evaluation.md)** - Rigorous forecast assessment
   - Proper scoring rules (MSE, MAE, CRPS)
   - Real-time vs revised data evaluation
   - Diebold-Mariano tests for forecast comparison
   - Forecast combination strategies

3. **[Missing Data Handling](guides/03_missing_data.md)** - Dealing with incomplete observations
   - Kalman filter with missing observations
   - Publication lag patterns
   - Interpolation vs model-based imputation
   - Nowcasting with arbitrary missingness

4. **[Cheatsheet](guides/cheatsheet.md)** - Quick reference for module concepts

### Notebooks (15-Minute Hands-On)

1. **[GDP Nowcasting](notebooks/01_gdp_nowcasting.ipynb)** - Build a working nowcasting system
   - Download real-time FRED data (IP, employment, sales, surveys)
   - Estimate DFM with Kalman filter
   - Produce GDP nowcast with confidence bands
   - Decompose nowcast revisions (news analysis)

2. **[Forecast Evaluation](notebooks/02_forecast_evaluation.ipynb)** - Evaluate nowcast performance
   - Out-of-sample backtesting framework
   - Compare DFM vs AR vs Blue Chip consensus
   - Compute Diebold-Mariano statistics
   - Visualize forecast errors over business cycle

3. **[Missing Data Handling](notebooks/03_missing_data_handling.ipynb)** - Handle ragged edges
   - Simulate publication lag patterns
   - Implement Kalman filter with missing obs
   - Compare to naive forward-fill
   - Quantify value of timely indicators

### Exercises (Self-Check, Ungraded)

**[exercises.py](exercises/exercises.py)** - Practice problems with instant feedback
- Build nowcasting function from scratch
- Implement Diebold-Mariano test
- Handle mixed-frequency ragged edges
- Decompose forecast errors

### Resources

- **[Additional Readings](resources/additional_readings.md)** - Curated papers and documentation
- **[Figures](resources/figures/)** - Visual assets and diagrams

## Prerequisites

**From previous modules:**
- State-space representation (Module 2)
- Kalman filter (Module 2)
- Dynamic factor model estimation (Module 2)

**Additional skills:**
- Familiarity with FRED database structure
- Understanding of publication lags in macroeconomic data
- Basic knowledge of forecast evaluation metrics

## Key Datasets

All notebooks use real data from FRED (Federal Reserve Economic Data):

- **GDP:** Quarterly real GDP growth (advance, 2nd, 3rd estimates + revisions)
- **Monthly indicators:**
  - Industrial Production (IP)
  - Employment (Payrolls, unemployment)
  - Retail sales
  - PMI surveys
  - Housing starts
- **Real-time vintages:** ALFRED API for as-available data

## Practical Skills You'll Gain

1. **Data Engineering**
   - Handle FRED API authentication and rate limits
   - Align mixed-frequency data (monthly → quarterly)
   - Manage real-time vintage datasets
   - Build ragged-edge data matrices

2. **Model Production**
   - Automated data refresh pipelines
   - Kalman filter with arbitrary missing patterns
   - Confidence interval construction
   - Nowcast decomposition (what moved the needle?)

3. **Evaluation & Communication**
   - Rigorous backtest frameworks
   - Fan chart visualization
   - News vs uncertainty attribution
   - Model documentation for stakeholders

## Success Criteria

You're ready to move on when you can:

- [ ] Build a GDP nowcast that updates as new data arrives
- [ ] Explain why your nowcast changed from last week (news decomposition)
- [ ] Evaluate forecast accuracy using multiple metrics
- [ ] Handle datasets with arbitrary missing patterns
- [ ] Compare your nowcast to Fed nowcasts (directionally aligned)

## Connections

**Builds on:**
- Module 02: Kalman filter, state-space methods
- Module 01: Factor extraction with PCA

**Prepares for:**
- Module 04: Mixed-frequency extensions (MIDAS)
- Module 05: Factor-augmented regression (FAVAR)
- Capstone: Production nowcasting system

## Estimated Time Breakdown

| Activity | Time |
|----------|------|
| Read guides | 3 hours |
| Complete notebooks | 4 hours |
| Self-check exercises | 2 hours |
| Additional readings (optional) | 2-3 hours |

## Getting Started

1. Read **[Nowcasting Guide](guides/01_nowcasting.md)** to understand the framework
2. Work through **[GDP Nowcasting Notebook](notebooks/01_gdp_nowcasting.ipynb)** to see it in action
3. Complete **[Forecast Evaluation](notebooks/02_forecast_evaluation.ipynb)** to assess performance
4. Tackle **[Missing Data Handling](notebooks/03_missing_data_handling.ipynb)** for robust implementation
5. Test yourself with **[exercises.py](exercises/exercises.py)**

## Common Questions

**Q: Why not just use the latest GDP estimate?**
A: Official GDP is released with a 4-6 week lag and revised for years. Markets and policymakers need timely estimates.

**Q: How do DFMs compare to Fed nowcasts?**
A: Similar methodology! NY Fed Nowcast is a DFM. We'll compare our results to theirs.

**Q: What if my nowcast is way off?**
A: Expected during structural breaks (COVID, financial crisis). We'll diagnose failures in forecast evaluation.

**Q: Do I need FRED API credentials?**
A: Recommended for high-frequency updates, but notebooks include cached data for offline work.

---

**Next Module:** Advanced Extensions (Time-Varying Parameters, Mixed Frequency, Large Datasets)
