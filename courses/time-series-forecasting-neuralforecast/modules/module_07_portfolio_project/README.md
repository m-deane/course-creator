# Module 7: Portfolio Project — Demand Forecasting System

This is the capstone module. You build a complete, deployable demand forecasting system
from a dataset of your choice and produce a portfolio artifact demonstrating the full
forecasting stack from this course.

There are no grades, quizzes, or submission requirements. The output is yours.

---

## What You Produce

A complete demand forecasting system with six components:

1. A real time series dataset loaded in nixtla format
2. A trained NHITS or XLinear model with probabilistic output (MQLoss)
3. 200+ sample paths from the joint forecast distribution
4. A quantitative answer to a business decision question
5. An explainability report from `.explain()`
6. A stakeholder-facing summary with visualizations

---

## Suggested 4-Week Timeline

| Week | Focus | Key Question |
|------|-------|--------------|
| 1 | Data selection and EDA | Which dataset tells a business story I care about? |
| 2 | Model training and evaluation | Is my model calibrated? Does it beat the baseline? |
| 3 | Sample paths and business decision | What is the numeric answer to my business question? |
| 4 | Explainability and final summary | Can a non-specialist understand what I built? |

This is a suggested pace, not a hard deadline. Some learners complete this in two intensive
weekends. Others spread it over six weeks while working through domain context.

---

## How to Select a Dataset

A good dataset for this project has three properties:

**1. Operational relevance**
The forecast output should change what someone does. Forecasting demand for a product
no one buys, or a series with no seasonal structure, produces a technically correct but
practically useless result.

**2. Visible seasonal structure**
NHITS is designed for series with multiple seasonal periods. If your series has no
weekly, monthly, or annual seasonality, consider a different dataset.

**3. A natural business question**
The best business questions involve cost asymmetry: the penalty for being wrong in one
direction differs from the penalty for being wrong in the other. Inventory stockouts
cost more than overstock. Grid undersupply is more dangerous than overcapacity. Find
the asymmetry in your domain.

**Zero-ETL options via `datasetsforecast`:**

```python
# French Bakery — fastest to prototype
import pandas as pd
df = pd.read_csv(
    "https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/"
    "main/datasets/french_bakery_daily.csv",
    parse_dates=["ds"]
)

# M5 Walmart retail
from datasetsforecast.m5 import M5
df, *_ = M5.load("./data/")

# ETT Energy
from datasetsforecast.long_horizon import LongHorizon
df, *_ = LongHorizon.load("./data/", group="ETTh1")

# Australian Tourism
from datasetsforecast.long_horizon import LongHorizon
df, *_ = LongHorizon.load("./data/", group="Tourism")
```

---

## What to Deliver

The project is self-directed. There is no submission. What you produce is determined by
what you want to show. A strong portfolio piece typically includes:

- A Jupyter notebook with working code from data loading through stakeholder summary
- Three saved figures: EDA overview, sample paths fan chart, attribution bar chart
- A plain-language executive summary paragraph explaining the business answer

The weakest projects are ones where the business question is vague ("forecast demand")
and the results section says "the model performed well." The strongest projects are ones
where a non-technical reader can understand the business question, the answer, and what
to do differently because of the forecast.

---

## Files in This Module

| File | Purpose |
|------|---------|
| `guides/01_project_specification.md` | Full project requirements and milestone details |
| `guides/01_project_specification_slides.md` | 10-slide overview (open in Marp or VS Code) |
| `notebooks/01_project_starter.ipynb` | Working skeleton — run end-to-end on French Bakery, then adapt |
| `exercises/01_milestone_checklist.py` | Self-check script — validates all four milestones |

---

## Running the Milestone Checklist

After completing each milestone, save your key results to `project_artifacts.py` in the
`exercises/` directory and run:

```bash
cd exercises/
python 01_milestone_checklist.py
```

The script prints PASS or FAIL for each structural requirement with a clear explanation
of what to fix. A project is ready to present when every check prints PASS.

---

## Connections to Earlier Modules

Every module in this course feeds directly into this project:

| Module | What It Contributes |
|--------|---------------------|
| 01 Point Forecasting | NHITS architecture, `.fit()` / `.predict()` API |
| 02 Probabilistic Forecasting | MQLoss, CRPS, calibration verification |
| 03 Sample Paths | `.simulate()`, Monte Carlo business framework |
| 04 Explainability | `.explain()`, attribution interpretation |
| 05 XLinear | Alternative model — faster, often competitive |
| 06 Production Patterns | Deployment patterns for the finished system |

---

## After This Project

Strong directions for continuing work after the capstone:

- **Hierarchical reconciliation**: aggregate store-level forecasts to regional totals using
  `hierarchicalforecast`
- **Foundation model fine-tuning**: transfer learning on your dataset using Nixtla TimeGPT
- **Real-time inference**: wrap `NeuralForecast` in a FastAPI service with `/predict` and
  `/simulate` endpoints
- **Production monitoring**: track calibration drift using `utilsforecast` on rolling
  evaluation windows
