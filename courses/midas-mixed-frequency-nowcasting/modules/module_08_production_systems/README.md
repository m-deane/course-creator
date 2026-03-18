# Module 08 — Production Nowcasting Systems

## Overview

This module covers the engineering of production-grade nowcasting pipelines: how to move from a research notebook to a system that runs automatically, handles data revisions, detects model deterioration, and generates auditable forecast records. The module addresses the five-layer pipeline architecture, vintage data management, model monitoring, structural break detection, and decision frameworks for the full model lifecycle.

## Learning Outcomes

After completing this module, you will be able to:

1. Design and implement a five-layer nowcasting pipeline (scheduler, acquisition, features, estimation, publication)
2. Build a SQLite vintage database with immutable rows and pseudo-real-time as-of queries
3. Handle ragged-edge data with documented per-indicator fill strategies
4. Monitor rolling forecast accuracy and detect structural breaks using CUSUM
5. Operate re-estimation triggers based on calendar, performance, and stability criteria
6. Generate structured health reports and model comparison tables
7. Apply the production decision flowcharts to new nowcasting problems

## Contents

### Guides

| File | Topic |
|------|-------|
| `guides/01_pipeline_architecture_guide.md` | Five-layer architecture, vintage DB, ragged-edge fill, publication calendar |
| `guides/01_pipeline_architecture_slides.md` | 15-slide companion deck with mermaid pipeline and sequence diagrams |
| `guides/02_monitoring_reporting_guide.md` | Rolling metrics, bias test, CUSUM, re-estimation triggers, health reports |
| `guides/02_monitoring_reporting_slides.md` | Companion deck |
| `guides/03_decision_flowchart_guide.md` | Complete decision trees for specification, estimation, evaluation, maintenance |
| `guides/03_decision_flowchart_slides.md` | Flowchart companion deck — course synthesis slide |

### Notebooks

| File | Description | Time |
|------|-------------|------|
| `notebooks/01_nowcasting_pipeline.ipynb` | End-to-end pipeline: vintage DB, ragged-edge, ElasticNet, nowcast evolution | 15 min |
| `notebooks/02_monitoring_dashboard.ipynb` | Rolling metrics, CUSUM, trigger, model comparison, health report | 12 min |

### Exercises

| File | Description |
|------|-------------|
| `exercises/01_production_self_check.py` | Four tasks: vintage DB query, fill methods, CUSUM detection, trigger logic |

## Prerequisites

- Module 01: MIDAS Fundamentals
- Module 05: Regularized MIDAS (ElasticNet, CV design)
- Module 07: Macro Applications (ragged edge, news decomposition)

## Key Concepts

**Vintage data**: Every economic series is released in preliminary form, then revised. Storing one row per `(series_id, obs_date, vintage_date)` triple enables pseudo-real-time queries: "what did the model know on date X?" Rows are never modified; revisions create new rows.

**Publication calendar**: A table of `(series_id, release_date, pub_lag_days)` triples. The scheduler triggers pipeline runs from this calendar, not from a fixed clock.

**Ragged edge**: At any forecast date, different monthly indicators are available through different reference months. Carry-forward is the default fill; AR1 projection is used for low-persistence series.

**CUSUM test**: Cumulative sum of standardised recursive residuals. Detects parameter instability without requiring a pre-specified break date. Critical value for 5% level: $c_{0.05} = 0.948$.

**Re-estimation trigger**: Three independent rules — calendar (every 4 quarters), performance (RMSE > 120% of backtest for 2 periods), structural break (CUSUM crossing). Any one fires re-estimation.

**Health report**: Plain-text structured log generated after every pipeline run. Contains current nowcast, rolling metrics, bias test, CUSUM status, and re-estimation action.

**ForecastRecord**: Immutable dataclass storing every forecast: date, target, point estimate, prediction intervals, training obs count, model name, news decomposition. Enables full audit trail.

## Connection to Other Modules

- **Module 07 (Macro)**: Pipeline architecture operationalises the GDP/CPI/payrolls models from Module 07
- **Module 05 (ML)**: ElasticNetCV with TimeSeriesSplit is the default estimator in Layer 4
- **Module 06 (Financial)**: Same pipeline architecture applies to MIDAS-RV and MIDAS-VaR

## Quick Start

```python
# Run the self-check exercise
python exercises/01_production_self_check.py

# Recommended notebook order:
# 1. notebooks/01_nowcasting_pipeline.ipynb   (pipeline end-to-end)
# 2. notebooks/02_monitoring_dashboard.ipynb  (monitoring and reporting)
```
