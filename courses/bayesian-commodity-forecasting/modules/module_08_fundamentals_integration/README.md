# Module 8: Fundamentals Integration and Forecast Combination

## Overview

This capstone module brings everything together: integrating fundamental supply/demand variables into Bayesian forecasting models, combining multiple models via Bayesian model averaging, and evaluating forecasts with proper scoring rules.

**Time Estimate:** 10-12 hours

## Learning Objectives

By completing this module, you will:
1. **Build** models that integrate fundamental variables (inventories, production)
2. **Apply** storage theory to inform model priors
3. **Combine** multiple forecasts using Bayesian model averaging
4. **Evaluate** forecasts using proper scoring rules (CRPS, log score)
5. **Construct** complete forecasting pipelines for commodity markets

## Module Contents

### Guides
- `01_storage_theory.md` - Economic foundations of commodity pricing
- `02_fundamental_variables.md` - Which fundamentals matter and why
- `03_bayesian_model_averaging.md` - Combining forecasts optimally
- `04_forecast_evaluation.md` - Proper scoring rules and calibration

### Notebooks
- `01_fundamental_regression.ipynb` - Bayesian regression with fundamentals
- `02_dynamic_coefficients.ipynb` - Time-varying fundamental relationships
- `03_model_averaging.ipynb` - BMA for forecast combination
- `04_forecast_evaluation.ipynb` - CRPS, log score, calibration checks
- `05_complete_pipeline.ipynb` - End-to-end forecasting system

### Assessments
- `quiz.md` - Fundamentals and evaluation concepts (15 questions)
- `final_project_integration.md` - Guidelines for capstone integration

## Key Concepts

### The Fundamental Pricing Framework

```
Supply-Demand Balance
        │
        ▼
    Inventories ◄────────────────┐
        │                        │
        ▼                        │
  Convenience Yield              │
        │                        │
        ▼                        │
    Spot Price ─────▶ Futures ───┘
        │              Price
        ▼
    Forecasts
```

### Storage Theory

The relationship between spot ($S$) and futures ($F$) prices:

$$F = S \cdot e^{(r + u - y)T}$$

Where:
- $r$: Risk-free rate
- $u$: Storage costs
- $y$: Convenience yield (benefit of holding physical)
- $T$: Time to maturity

**Key implication:** Inventory levels affect convenience yield, which affects the term structure.

### Fundamental Variables

| Variable | Source | Impact | Lag |
|----------|--------|--------|-----|
| **Crude Inventories** | EIA Weekly | Negative | 1 day |
| **Refinery Runs** | EIA Weekly | Positive | 1 day |
| **Production** | EIA Weekly | Negative | 1 day |
| **Imports/Exports** | EIA Weekly | Mixed | 1 day |
| **OPEC Production** | IEA Monthly | Negative | ~30 days |
| **GDP Growth** | BEA Quarterly | Positive | ~60 days |
| **Crop Conditions** | USDA Weekly | Negative | 1 day |

### Model Averaging Motivation

No single model dominates across all market conditions:

| Model | Bull Market | Bear Market | High Vol | Low Vol |
|-------|-------------|-------------|----------|---------|
| Random Walk | Poor | Poor | OK | OK |
| State Space | Good | Good | Good | Good |
| GP | Good | Good | Poor | Excellent |
| Regime Switch | Excellent | Excellent | Good | Poor |

**Solution:** Combine models with Bayesian Model Averaging.

### Bayesian Model Averaging

Given models $M_1, ..., M_K$:

$$P(y_{new} | y) = \sum_{k=1}^K P(y_{new} | M_k, y) \cdot P(M_k | y)$$

**Posterior model probability:**
$$P(M_k | y) = \frac{P(y | M_k) P(M_k)}{\sum_j P(y | M_j) P(M_j)}$$

**Stacking weights (alternative):**
Optimize weights to minimize cross-validated prediction error.

### Forecast Evaluation

**Proper Scoring Rules** are loss functions that are minimized when the forecaster reports their true belief.

| Scoring Rule | Formula | Measures |
|--------------|---------|----------|
| **Log Score** | $-\log p(y_{obs})$ | Density at observed value |
| **CRPS** | $\int (F(y) - \mathbf{1}_{y \geq y_{obs}})^2 dy$ | Full distribution fit |
| **Brier Score** | $(p - \mathbf{1}_{event})^2$ | Probability calibration |

**Why not MSE?**
MSE only evaluates point forecasts, ignoring uncertainty quantification.

## Commodity-Specific Considerations

### Energy Fundamentals

**Crude Oil:**
- EIA Weekly Petroleum Status Report (Wednesday)
- Cushing stocks (WTI delivery point)
- OPEC+ production decisions
- Refinery utilization

**Natural Gas:**
- EIA Weekly Storage Report (Thursday)
- Heating/Cooling Degree Days
- LNG export capacity

### Agricultural Fundamentals

**Grains:**
- USDA WASDE (monthly)
- Crop progress/conditions (weekly)
- Acreage intentions (March)
- Export inspections

### Look-Ahead Bias

**Critical:** Fundamentals are released with delays.

```
Timeline:
Day 0 (Friday): EIA covers data through this day
Day 5 (Wednesday): EIA report released
Day 5+: Market reacts

In backtesting, you can only use EIA data from Day 5 onward,
NOT for predictions on Days 0-4!
```

## Completion Criteria

- [ ] Fundamental regression model with proper lag structure
- [ ] Model averaging implementation with multiple models
- [ ] Forecast evaluation with CRPS and calibration checks
- [ ] Complete pipeline ready for capstone
- [ ] Quiz score ≥ 80%

## Prerequisites

- Modules 1-7 completed
- Understanding of commodity market fundamentals (Module 2)
- Multiple model implementations from earlier modules

---

*"The best forecasters don't pick winners—they combine models thoughtfully and evaluate honestly."*
