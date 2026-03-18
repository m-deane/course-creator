# Module 07 — Macroeconomic Applications

## Overview

This module covers the three core macroeconomic nowcasting applications: GDP, inflation, and the labour market. Each application demonstrates a distinct aspect of MIDAS: multi-indicator aggregation for GDP, component-level bottom-up forecasting for inflation, and weekly-to-monthly bridging for the labour market. The module also covers the institutional frameworks of the NY Fed and ECB, real-time data challenges, and rigorous evaluation methodology.

## Learning Outcomes

After completing this module, you will be able to:

1. Design and implement a simplified GDP nowcasting pipeline with ragged-edge handling
2. Compute news decomposition to explain nowcast revisions from individual data releases
3. Build a bottom-up inflation nowcasting model using daily commodity prices
4. Implement the MIDAS-W (weekly-to-monthly) model for payrolls nowcasting
5. Apply pseudo-real-time evaluation with proper vintage data treatment
6. Describe how the NY Fed and ECB approach real-time GDP nowcasting

## Contents

### Guides

| File | Topic |
|------|-------|
| `guides/01_gdp_nowcasting_practice_guide.md` | NY Fed/ECB frameworks, real-time data, ragged edges |
| `guides/01_gdp_nowcasting_practice_slides.md` | 15-slide companion deck with mermaid architecture diagrams |
| `guides/02_inflation_labour_guide.md` | Inflation components, energy CPI MIDAS, claims-payrolls MIDAS-W |
| `guides/02_inflation_labour_slides.md` | Companion deck |

### Notebooks

| File | Description | Time |
|------|-------------|------|
| `notebooks/01_simplified_nyfed_nowcast.ipynb` | Multi-indicator GDP nowcast with ragged-edge simulation | 15 min |
| `notebooks/02_inflation_nowcasting.ipynb` | Bottom-up CPI nowcast with daily oil prices | 12 min |
| `notebooks/03_labour_market_nowcasting.ipynb` | Weekly claims → monthly payrolls MIDAS-W | 12 min |

### Exercises

| File | Description |
|------|-------------|
| `exercises/01_macro_self_check.py` | Four tasks: ragged edge, GDP news decomposition, energy CPI, claims-payrolls |

## Prerequisites

- Module 01: MIDAS Fundamentals
- Module 02: Estimation and Inference
- Module 05: Regularized MIDAS (for multi-indicator models)

## Key Concepts

**Ragged edge**: At any forecast date, different monthly indicators have different latest available vintage. Handle with carry-forward, zero-fill, or AR projection.

**Vintage data**: Economic data is revised. Use data as it appeared at the forecast date for fair evaluation. ALFRED database provides historical vintages.

**Publication calendar**: Deterministic schedule of when each release is available. Mandatory for real-time nowcasting systems.

**News decomposition**: Revision to nowcast = sum of (coefficient × surprise for each new release). Traces how each data release contributed to forecast revisions.

**MIDAS-W**: Monthly target (payrolls) with weekly predictor (claims). K=4 lags, Beta weights estimated by NLS. Negative coefficient: high claims → lower payrolls.

**Bottom-up inflation**: Forecast energy and food CPI separately using MIDAS; use AR for core services; aggregate with CPI weights.

## Connection to Other Modules

- **Module 06 (Financial)**: Same multi-frequency structure applied to financial risk variables
- **Module 08 (Production)**: How to operationalise these macro nowcasting models in a production pipeline
- **Module 04 (DFMs)**: DFMs provide a more formal alternative to MIDAS for large-scale macro nowcasting

## Quick Start

```python
# Run the self-check exercise
python exercises/01_macro_self_check.py

# Recommended notebook order:
# 1. notebooks/01_simplified_nyfed_nowcast.ipynb  (start here)
# 2. notebooks/02_inflation_nowcasting.ipynb
# 3. notebooks/03_labour_market_nowcasting.ipynb
```
