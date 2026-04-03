# Module 0: Foundations & Prerequisites

**Course:** Modern Time Series Forecasting with NeuralForecast  
**Module:** 0 of 7  
**Estimated time:** 60–75 minutes total

---

## Learning Objectives

By completing this module you will be able to:

1. **Distinguish** point, probabilistic, and distributional forecasting approaches and explain when each is appropriate for a business decision
2. **Navigate** the neuralforecast ecosystem: the roles of `neuralforecast`, `datasetsforecast`, and `utilsforecast`, and how they compose into a forecasting pipeline
3. **Format** any time series dataset into the nixtla `(unique_id, ds, y)` contract required by all three libraries
4. **Train** a first neural forecasting model (NHITS with MQLoss) using the French Bakery dataset and produce a 7-day probabilistic forecast with prediction intervals
5. **Interpret** the forecast output DataFrame: column naming conventions, quantile columns, and how to merge with actual values for evaluation

---

## Module Contents

```
module_00_foundations/
├── guides/
│   ├── 01_forecasting_landscape.md         # Point vs probabilistic vs distributional
│   ├── 01_forecasting_landscape_slides.md  # Companion Marp deck (12 slides)
│   ├── 02_neuralforecast_ecosystem.md      # API walkthrough: fit/predict/cv/simulate/explain
│   └── 02_neuralforecast_ecosystem_slides.md  # Companion Marp deck (14 slides)
├── notebooks/
│   ├── 01_quickstart_neuralforecast.ipynb  # Train NHITS on French Bakery in <15 min
│   └── 02_exploring_datasets.ipynb         # EDA across 3 real datasets (<15 min)
├── exercises/
│   └── 01_setup_exercises.py               # Self-check: verify environment and API
└── README.md                               # This file
```

---

## Recommended Sequence

The materials are designed for two paths. Choose based on your preference.

### Path A: Code-first
1. Run `exercises/01_setup_exercises.py` — verify your environment (2 min)
2. Open `notebooks/01_quickstart_neuralforecast.ipynb` — train your first model (12 min)
3. Open `notebooks/02_exploring_datasets.ipynb` — explore real datasets (13 min)
4. Read `guides/01_forecasting_landscape.md` — deepen the conceptual foundation (15 min)
5. Read `guides/02_neuralforecast_ecosystem.md` — full API reference (15 min)

### Path B: Concepts-first
1. Read `guides/01_forecasting_landscape.md` (15 min)
2. View `guides/01_forecasting_landscape_slides.md` as a slide deck (optional)
3. Read `guides/02_neuralforecast_ecosystem.md` (15 min)
4. Open `notebooks/01_quickstart_neuralforecast.ipynb` (12 min)
5. Open `notebooks/02_exploring_datasets.ipynb` (13 min)
6. Run `exercises/01_setup_exercises.py` — confirm everything works (2 min)

---

## Prerequisites

- Python 3.9 or higher
- `pip install neuralforecast datasetsforecast utilsforecast`
- Basic pandas familiarity (DataFrames, groupby, plot)
- Internet connection for dataset downloads

No prior time series or deep learning experience is required. Module 0 builds from scratch.

---

## Key Concepts Introduced

| Concept | Introduced in |
|---|---|
| Point vs probabilistic vs distributional forecasting | Guide 01, Notebook 01 |
| CRPS scoring rule | Guide 01 |
| Calibration of prediction intervals | Guide 01, Slides 01 |
| nixtla `(unique_id, ds, y)` format | Guide 01, Notebook 02 |
| `NeuralForecast` wrapper class | Guide 02, Notebook 01 |
| NHITS architecture (high-level) | Guide 02, Notebook 01 |
| XLinear baseline model | Guide 02, Notebook 01 |
| `MQLoss` vs `MAE` — the loss-as-forecast-type pattern | Guide 02 |
| `.fit()` / `.predict()` / `.cross_validation()` | Guide 02, Notebook 01 |
| `.simulate()` for sample paths | Guide 02 |
| `.explain()` for SHAP-based attribution | Guide 02 |

---

## Datasets Used

| Dataset | Source | Use in this module |
|---|---|---|
| French Bakery Daily Sales | nixtla/transfer-learning-time-series | Primary training dataset |
| ETTm1 | zhouhaoyi/ETDataset | EDA in Notebook 02 |
| Blog traffic / M4 Weekly | nixtla / M4 competition | Scale demonstration in Notebook 02 |

All datasets are publicly available and load from URLs — no manual download required.

---

## What's Next

After completing Module 0, proceed to:

**Module 01: Point Forecasting**  
NHITS and NBEATS architectures, benchmark comparisons against ARIMA and ETS baselines, systematic hyperparameter selection, and evaluation with MASE and sMAPE.
