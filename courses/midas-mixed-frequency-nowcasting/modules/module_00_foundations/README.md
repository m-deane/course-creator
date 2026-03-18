# Module 00: Foundations and the Mixed-Frequency Problem

## Overview

This module establishes the conceptual and technical foundation for the entire course. You will understand why mixed-frequency data is challenging, what traditional solutions offer and sacrifice, and how to access and structure the datasets used throughout.

## Learning Objectives

By the end of this module, you will be able to:

1. Explain why economic data arrives at different frequencies and why this creates a statistical problem
2. Quantify information loss from temporal aggregation
3. Describe the limitations of bridge equations and interpolation
4. Access quarterly GDP, monthly industrial production, and daily S&P 500 data from FRED and Yahoo Finance
5. Build a MIDAS data matrix aligning high-frequency regressors to low-frequency target observations
6. Identify the ragged-edge problem in real-time nowcasting

## Structure

```
module_00_foundations/
├── guides/
│   ├── 01_mixed_frequency_problem_guide.md     # Why frequencies differ and why it matters
│   ├── 01_mixed_frequency_problem_slides.md    # 16-slide companion deck
│   ├── 02_traditional_solutions_guide.md       # Aggregation, Chow-Lin, bridge equations
│   ├── 02_traditional_solutions_slides.md      # 16-slide companion deck
│   ├── 03_course_datasets_guide.md             # FRED, Yahoo Finance, CSV fallbacks
│   └── 03_course_datasets_slides.md            # 16-slide companion deck
├── notebooks/
│   ├── 01_environment_setup.ipynb              # Install, verify, download data (~10 min)
│   └── 02_mixed_frequency_exploration.ipynb   # Visualize and quantify info loss (~15 min)
├── exercises/
│   └── 01_mixed_freq_self_check.py            # 5-exercise self-check
└── resources/
    ├── gdp_quarterly.csv                       # Real GDP growth, 2000Q1–2024Q4
    ├── industrial_production_monthly.csv       # IP growth, 2000-01–2024-12
    └── sp500_daily.csv                         # S&P 500 returns, sample periods
```

## Prerequisites

- Python 3.9+ with pandas, numpy, matplotlib, scipy, statsmodels
- Basic familiarity with time series data in pandas
- Understanding of OLS regression

## Getting Started

1. Complete `notebooks/01_environment_setup.ipynb` first — it verifies your setup and downloads data
2. Read `guides/01_mixed_frequency_problem_guide.md`
3. Complete `notebooks/02_mixed_frequency_exploration.ipynb`
4. Read guides 02 and 03
5. Run `exercises/01_mixed_freq_self_check.py` to confirm understanding

## Key Concepts Introduced

**Mixed-frequency problem:** The challenge of combining time series observed at different frequencies (quarterly GDP, monthly IP, daily returns) in a single regression model.

**Temporal aggregation:** Collapsing high-frequency observations to low frequency (averaging, summing, end-of-period). Loses within-period timing information.

**Bridge equation:** Two-step procedure: (1) quarterly regression on aggregated monthly data, (2) forecast missing months for nowcasting. Compounds estimation errors across steps.

**Chow-Lin interpolation:** Inferring synthetic high-frequency values consistent with observed low-frequency aggregates. Creates generated-regressors problem in subsequent models.

**Ragged edge:** The incomplete data pattern at the edge of the sample — within the current quarter, only some monthly observations have been released.

**MIDAS data matrix:** The $(T_L \times n_{\text{lags}})$ matrix aligning high-frequency lags to low-frequency observations. Fundamental structure for all MIDAS estimators.

## Data Summary

| Series | Source | Frequency | Coverage | CSV File |
|--------|--------|-----------|----------|----------|
| Real GDP Growth | FRED: GDPC1 | Quarterly | 2000Q1–2024Q4 | `gdp_quarterly.csv` |
| Industrial Production Growth | FRED: INDPRO | Monthly | 2000-01–2024-12 | `industrial_production_monthly.csv` |
| S&P 500 Returns | Yahoo: ^GSPC | Daily | Sample periods | `sp500_daily.csv` |

## Connection to Course Arc

This module sets up the problem. Module 01 introduces the MIDAS solution. Module 02 covers estimation. Module 03 applies MIDAS to real-time nowcasting. Module 04 extends to dynamic factor models.

The information loss quantified in this module (within-quarter variance discarded by aggregation) is exactly what MIDAS recovers — and the empirical improvement in R-squared that you'll measure in Module 01 is the payoff for that recovery.
