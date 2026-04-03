# Portfolio Project: Demand Forecasting System

## Overview

Build a complete demand forecasting system that trains a neural model, generates sample paths for probabilistic decision-making, and produces explainability reports for stakeholder communication.

This is a **portfolio piece**, not a graded assignment. The output is a deployable artifact you can showcase.

## Requirements

### 1. Data Selection
Choose a real time series dataset with at least 365 daily observations. Suggested options:
- **M5 Competition** — Walmart retail sales (hierarchical, daily)
- **Australian Tourism** — quarterly visitor nights by region
- **Energy Load** — hourly/daily electricity demand
- **French Bakery** — daily product sales (used in course)

### 2. Model Training
- Train NHITS or XLinear with MQLoss for probabilistic output
- Use cross_validation for honest evaluation
- Report MAE and MSE against a simple baseline

### 3. Sample Path Generation
- Generate 100+ sample paths using .simulate()
- Answer a specific business question using the joint distribution:
  - "How much inventory for the week at 80% service level?"
  - "When should I place a reorder?"
  - "What's my worst-case scenario this month?"
- Show that marginal quantiles give a different (wrong) answer

### 4. Explainability Report
- Use .explain() with IntegratedGradients
- Identify which features/lags drive the forecast
- Create visualizations: lag importance heatmap, feature attribution bar chart
- Write a 1-page stakeholder summary in plain language

### 5. Deliverables
- Jupyter notebook with complete pipeline
- Stakeholder summary (1 page, non-technical)
- README explaining how to run the project

## Suggested Timeline

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1 | Data selection and EDA | Notebook with data exploration |
| 2 | Model training and evaluation | Trained model, evaluation metrics |
| 3 | Sample paths and business decision | Business question answered |
| 4 | Explainability and presentation | Final notebook + stakeholder summary |

## Example Project Outline

```python
# 1. Load and prepare data
df = load_bakery_data()

# 2. Train with probabilistic loss
nf = NeuralForecast(models=[NHITS(h=7, loss=MQLoss(level=[80,90]))], freq="D")
nf.fit(df)

# 3. Generate sample paths
paths = nf.simulate(n_paths=100)

# 4. Answer business question
weekly_total_80 = np.quantile(paths.sum(axis=1), 0.8)
print(f"Stock {weekly_total_80:.0f} units for 80% service level")

# 5. Explain the forecast
fcsts, explanations = nf.explain(explainer="IntegratedGradients")
plot_attributions(explanations)
```
