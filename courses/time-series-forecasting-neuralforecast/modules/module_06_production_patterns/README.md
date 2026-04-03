# Module 6: Production Patterns & Integration

This module ties together everything from the course — point forecasting, probabilistic forecasting, sample paths, and explainability — into a single production-ready pipeline.

## Learning Objectives

By the end of this module you will be able to:

1. Build a `ForecastPipeline` class that wraps NeuralForecast from data ingestion through business decision output
2. Choose between NHITS and XLinear based on series length, feature richness, and interpretability requirements
3. Select a retraining strategy (sliding window vs expanding window) based on demand stability
4. Scale forecasting across hundreds of products using the global model pattern
5. Use `cross_validation` to make a defensible model selection decision
6. Implement production hardening: GPU checkpointing, experiment logging, fallback models, and sanity checks

## Module Contents

### Guides

| File | Topic | Time |
|---|---|---|
| `guides/01_production_pipeline.md` | End-to-end pipeline architecture, ForecastPipeline class, model selection, retraining strategies | 20 min |
| `guides/01_production_pipeline_slides.md` | Slide deck companion (12 slides) | 25 min |
| `guides/02_neuralforecast_patterns.md` | Custom losses, GPU checkpointing, multi-series scaling, logging, error handling | 25 min |
| `guides/02_neuralforecast_patterns_slides.md` | Slide deck companion (13 slides) | 25 min |

### Notebooks

| File | Topic | Time |
|---|---|---|
| `notebooks/01_end_to_end_pipeline.ipynb` | Complete pipeline on French Bakery data: load, train, forecast, simulate, explain, decide | 12 min |
| `notebooks/02_scaling_patterns.ipynb` | Global models, batch forecasting, cross-validation, profiling | 13 min |

### Exercises

| File | Topic |
|---|---|
| `exercises/01_production_exercises.py` | Build and verify ForecastPipeline with shape checks at every stage |

## Central Business Question

> **How many baguettes should we order for the week at an 80% service level?**

This module answers that question through the full pipeline: load data, train NHITS with MQLoss, generate quantile forecasts, and convert the 80th percentile to an order quantity.

## Pipeline Architecture

```
Raw Data
    → ingest()       — convert to nixtla format, validate duplicates and length
    → train()        — fit NHITS or XLinear with MQLoss
    → predict()      — quantile forecasts (H steps ahead)
    → simulate()     — sample paths for distributional analysis
    → explain()      — feature importances for stakeholder reporting
    → service_level_order()  — order quantity at target service level
```

## Key Decision Rules

**Model selection:**
- NHITS: series length > 200, many exogenous features, non-linear seasonal patterns
- XLinear: series length < 100, interpretability required, near-linear demand

**Retraining strategy:**
- Sliding window: demand shifts seasonally or with promotions (e.g., fashion, specialty items)
- Expanding window: stable demand where all history is informative (e.g., staple products)

## Prerequisites

- Module 2: Probabilistic Forecasting (MQLoss, quantile interpretation)
- Module 3: Sample Paths (simulate API)
- Module 4: Explainability (explain API)
- Module 5: XLinear (model selection context)

## Running the Exercises

```bash
# Self-check exercises (no Jupyter required)
python exercises/01_production_exercises.py

# Notebooks (requires Jupyter)
jupyter notebook notebooks/01_end_to_end_pipeline.ipynb
jupyter notebook notebooks/02_scaling_patterns.ipynb

# Render slides
npx @marp-team/marp-cli --html --theme-set ../../../../resources/themes/course-theme.css -- guides/01_production_pipeline_slides.md
npx @marp-team/marp-cli --html --theme-set ../../../../resources/themes/course-theme.css -- guides/02_neuralforecast_patterns_slides.md
```

## Data

All notebooks use the **French Bakery dataset** from the nixtla transfer learning repository. It is downloaded automatically on first run and cached locally. No manual data preparation is required.

Dataset URL:
`https://raw.githubusercontent.com/nixtla/transfer-learning-time-series/main/datasets/french_bakery/bakery.csv`
