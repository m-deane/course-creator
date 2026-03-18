# Module 06 — Financial Applications

## Overview

This module applies MIDAS to three core financial risk management problems: realised volatility forecasting, mixed-frequency VaR, and commodity price nowcasting. The canonical MIDAS-RV model of Ghysels, Santa-Clara, and Valkanov (2005) directly motivated mixed-frequency regression in empirical finance. The module covers estimation, evaluation with QLIKE loss, VaR backtesting, and multi-frequency commodity fundamentals.

## Learning Outcomes

After completing this module, you will be able to:

1. Formulate and estimate the MIDAS-RV model for realised volatility forecasting
2. Interpret Beta polynomial weight shapes (geometric decay, hump-shaped, uniform)
3. Evaluate volatility forecasts with QLIKE loss and Mincer-Zarnowitz regression
4. Implement daily VaR from MIDAS-RV monthly forecasts
5. Backtest VaR models with Kupiec and Christoffersen tests
6. Build multi-frequency MIDAS-X models for commodity price nowcasting

## Contents

### Guides

| File | Topic |
|------|-------|
| `guides/01_midas_rv_guide.md` | MIDAS-RV theory, Beta weights, NLS estimation, HAR-RV comparison |
| `guides/01_midas_rv_slides.md` | Slide deck companion (15 slides with LaTeX equations) |
| `guides/02_mixed_freq_risk_guide.md` | Mixed-frequency VaR, term structure, commodity fundamentals |
| `guides/02_mixed_freq_risk_slides.md` | Slide deck companion |

### Notebooks

| File | Description | Time |
|------|-------------|------|
| `notebooks/01_midas_rv_sp500.ipynb` | MIDAS-RV for S&P 500 RV with HAR-RV comparison | 15 min |
| `notebooks/02_commodity_nowcasting.ipynb` | Crude oil price nowcasting with multi-frequency features | 13 min |
| `notebooks/03_var_mixed_frequency.ipynb` | MIDAS-VaR with Kupiec and Christoffersen backtests | 13 min |

### Exercises

| File | Description |
|------|-------------|
| `exercises/01_financial_self_check.py` | Four tasks: Beta weights, NLS estimation, Kupiec test, QLIKE comparison |

### Resources

| File | Description |
|------|-------------|
| `resources/sp500_returns.csv` | S&P 500 daily returns 2004-2023 |
| `resources/crude_oil_daily.csv` | Crude oil daily prices and returns 2004-2023 |
| `resources/macro_monthly.csv` | Monthly macro indicators (INDPRO, UNRATE, credit spreads, VIX) |

## Prerequisites

- Module 01: MIDAS Fundamentals
- Module 02: Estimation and Inference (NLS estimation)
- Basic understanding of financial returns and volatility

## Key Concepts

**MIDAS-RV**: Forecast monthly RV from daily squared returns using Beta polynomial weights. Estimated by NLS. Log transformation is essential.

**QLIKE loss**: The appropriate evaluation metric for volatility forecasts. Proxy-robust (Patton 2011) and asymmetric — penalises under-prediction more than MSE.

**HAR-RV**: Standard benchmark (Corsi 2009) — hard to beat at short horizons; MIDAS-RV gains are largest at quarterly and annual horizons.

**Kupiec test**: Unconditional coverage — is the violation rate equal to alpha?

**Christoffersen test**: Independence — are violations serially uncorrelated (no clustering)?

**Beta weight patterns:**
- $\theta_1=1, \theta_2=5$: Geometric decay (most common empirically for equity RV)
- $\theta_1=\theta_2=1$: Uniform (all lags equal)
- $\theta_1=2, \theta_2=5$: Hump-shaped (some intermediate lag matters most)

## Quick Start

```python
# Run the self-check exercise
python exercises/01_financial_self_check.py

# Or open notebooks:
# 1. notebooks/01_midas_rv_sp500.ipynb  (start here)
# 2. notebooks/02_commodity_nowcasting.ipynb
# 3. notebooks/03_var_mixed_frequency.ipynb
```

## Connection to Other Modules

- **Module 05 (ML Extensions)**: ML approaches to volatility forecasting (Random Forest-RV, XGBoost-RV)
- **Module 07 (Macro)**: Same MIDAS-X framework applied to GDP and inflation
- **Module 08 (Production)**: How to operationalise MIDAS-RV in a live risk system
