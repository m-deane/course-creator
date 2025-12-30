# Module 2: Commodity Market Data and Features

## Overview

This module covers the practical aspects of acquiring, cleaning, and transforming commodity data into features suitable for Bayesian modeling. You'll work with real data from EIA, USDA, and market sources to build a robust data pipeline.

**Time Estimate:** 6-8 hours

## Learning Objectives

By completing this module, you will:
1. **Retrieve** commodity price and fundamental data from public APIs
2. **Process** time series data handling missing values and outliers
3. **Decompose** seasonality patterns in commodity prices
4. **Engineer** features from term structure and fundamental data
5. **Build** reproducible data pipelines for forecasting workflows

## Module Contents

### Guides
- `01_data_sources.md` - Comprehensive guide to commodity data APIs
- `02_seasonality_analysis.md` - Decomposition methods for commodities
- `03_feature_engineering.md` - Creating predictive features

### Notebooks
- `01_data_retrieval_apis.ipynb` - Fetching data from EIA, USDA, yfinance
- `02_data_cleaning_pipeline.ipynb` - Handling missing data and outliers
- `03_seasonality_decomposition.ipynb` - STL, Fourier, and calendar features
- `04_term_structure_features.ipynb` - Roll yield, curve shape, spreads

### Assessments
- `quiz.md` - Data concepts (15 questions)
- `mini_project_rubric.md` - Build a commodity data pipeline

## Key Concepts

### Data Source Hierarchy

```
Primary Sources (Official)
├── EIA.gov - Energy data (petroleum, natural gas, coal)
├── USDA.gov - Agricultural data (WASDE, crop progress)
├── LME.com - Base metals inventories
└── CFTC.gov - Commitment of Traders positioning

Secondary Sources (Aggregated)
├── Yahoo Finance - Futures prices (free, delayed)
├── FRED - Macro variables
└── Quandl - Various commodity datasets
```

### Feature Categories

| Category | Examples | Use Case |
|----------|----------|----------|
| **Price-based** | Returns, realized vol, momentum | Technical signals |
| **Fundamental** | Inventory levels, production, demand | Supply/demand balance |
| **Seasonal** | Month, week of year, heating/cooling days | Cyclical patterns |
| **Term structure** | Roll yield, curve slope, calendar spreads | Storage/convenience |
| **Macro** | Dollar index, interest rates, GDP | Demand drivers |

### Seasonality Patterns

**Energy:**
- Natural gas: Winter heating, summer cooling peaks
- Gasoline: Summer driving season
- Heating oil: Q4-Q1 demand surge

**Agriculture:**
- Corn/Soybeans: Spring planting, fall harvest
- Wheat: Winter wheat harvest (June), spring wheat (August)
- Weather uncertainty peaks during growing season

**Metals:**
- Less pronounced seasonality
- Construction demand in spring
- Chinese New Year inventory effects

## Completion Criteria

- [ ] Data retrieval notebook runs successfully
- [ ] Pipeline handles missing EIA releases correctly
- [ ] Seasonality decomposition applied to chosen commodity
- [ ] Mini-project: Complete data pipeline for capstone

## Prerequisites

- Module 0-1 completed
- Python pandas proficiency
- Basic API concepts

---

*"Bad data leads to bad models. The data pipeline is the foundation of any forecasting system."*
