# Module 03: Nowcasting with MIDAS

**Mixed-Frequency Models: MIDAS Regression and Nowcasting**

## Overview

This module applies the MIDAS estimation and inference framework to real-time GDP nowcasting. The core challenge is the "ragged edge" — the current quarter's MIDAS data matrix is incomplete because some monthly releases have not yet occurred. We implement three vintage points (1-month, 2-month, 3-month) and evaluate forecast accuracy using expanding-window cross-validation.

## Learning Objectives

1. Explain the nowcasting problem: ragged edge, publication calendar, vintage points
2. Build MIDAS matrices for each of three nowcast vintages (h_missing = 0, 1, 2)
3. Apply direct MIDAS at each vintage point using separate theta estimates
4. Evaluate RMSE by vintage and compare to AR(1) and equal-weight benchmarks
5. Construct forecast evolution charts showing nowcast revisions
6. Quantify parameter stability over the expanding window
7. Use Monte Carlo simulation to measure ragged-edge estimation bias

## Directory Structure

```
module_03_nowcasting/
├── guides/
│   ├── 01_nowcasting_problem_guide.md   # Ragged edge, publication calendar, vintages
│   ├── 01_nowcasting_problem_slides.md  # 13-slide companion deck
│   ├── 02_direct_vs_iterated_guide.md   # Direct vs. iterated strategy; MIDAS-AR
│   └── 02_direct_vs_iterated_slides.md  # 13-slide companion deck
├── notebooks/
│   ├── 01_gdp_nowcast.ipynb             # Complete nowcast workflow, RMSE by vintage
│   ├── 02_ragged_edge_simulation.ipynb  # Monte Carlo ragged-edge bias analysis
│   └── 03_forecast_evolution.ipynb      # Parameter stability, revision analysis
├── exercises/
│   └── 01_nowcasting_self_check.py      # Self-check exercises (5 topics)
└── resources/
    (uses module_00_foundations/resources/ CSVs)
```

## Key Concepts

### The Ragged Edge

At any point within a quarter, some monthly observations are available and some are not:

```
Quarter Q, at the 2-month vintage (h_missing=1):
  j=0 (March IP):  MISSING  <- most recent month not yet released
  j=1 (February):  available
  j=2 (January):   available
  j=3..j=11 (prior quarters): all available
```

The backward-shift strategy drops the missing lag and uses K_eff = K - h_missing lags.

### Nowcast Update Formula

When a new monthly IP figure arrives:

$$\Delta\hat{y}_Q = \hat{\beta} \cdot \hat{w}_0 \cdot x_{IP,\text{new}}$$

A +1% IP surprise updates the GDP nowcast by $\hat{\beta} \cdot \hat{w}_0 \approx 0.1\%$ for typical parameters.

### RMSE by Vintage

More monthly data always improves accuracy:
$$\text{RMSE}_{h=0} \leq \text{RMSE}_{h=1} \leq \text{RMSE}_{h=2}$$

## Data Requirements

Uses CSV files from `module_00_foundations/resources/`:
- `gdp_quarterly.csv` — quarterly GDP growth
- `industrial_production_monthly.csv` — monthly IP growth

## Connections

- **Builds on:** Module 01 (MIDAS fundamentals), Module 02 (estimation and inference)
- **Leads to:** Module 04 (Dynamic Factor Models — many indicators simultaneously)
- **Related to:** Giannone-Reichlin-Small (2008), Bańbura-Rünstler (2011)
