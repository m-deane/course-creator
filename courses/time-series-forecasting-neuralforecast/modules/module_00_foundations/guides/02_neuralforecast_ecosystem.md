# The NeuralForecast Ecosystem: API, Models, and Workflows

## In Brief

This guide is a hands-on tour of the neuralforecast API. By the end, you will have run a complete fit → predict → cross-validation cycle, generated probabilistic sample paths, and inspected model explanations — all using the French Bakery daily sales dataset.

Start here: load the dataset and inspect its structure.

```python
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# French Bakery daily sales — real retail data, 8 series
url = (
    "https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/"
    "main/datasets/french_bakery_daily.csv"
)
df = pd.read_csv(url, parse_dates=['ds'])

print("Shape:", df.shape)
print("\nFirst rows:")
print(df.head())
print("\nSeries:")
print(df['unique_id'].value_counts())
```

Expected output:
```
Shape: (4382, 3)

First rows:
  unique_id          ds      y
0  baguette  2021-01-04  142.0
1  baguette  2021-01-05  118.0
...

Series:
baguette           547
pain au chocolat   547
croissant          547
...
```

---

## 1. The NeuralForecast API Overview

The `NeuralForecast` class is the entry point for everything. It wraps one or more model instances and provides a unified interface.

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MQLoss

# 1. Instantiate models
models = [
    NHITS(
        h=7,                    # forecast horizon: 7 days
        input_size=28,          # lookback window: 28 days
        loss=MQLoss(            # distributional loss
            quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]
        ),
        max_steps=500,          # training iterations
        scaler_type='standard', # normalize inputs
    )
]

# 2. Wrap in NeuralForecast
nf = NeuralForecast(models=models, freq='D')

# 3. The three core methods
nf.fit(df=train)                         # .fit()
forecasts = nf.predict()                 # .predict()
cv_results = nf.cross_validation(df=df)  # .cross_validation()
```

---

## 2. Key Models

neuralforecast ships over 20 neural models. For Module 0, two are essential:

### NHITS (Neural Hierarchical Interpolation for Time Series)

NHITS decomposes a time series into multiple frequency components using hierarchical interpolation. Each stack processes a different temporal resolution, then the outputs are summed.

**Why NHITS first:**
- State-of-the-art on M4 and M5 benchmarks
- Explicit multi-resolution decomposition
- Fast training (fully convolutional, no attention)
- Stable with default hyperparameters

**Critical hyperparameters:**

| Parameter | Default | Effect |
|---|---|---|
| `h` | required | Forecast horizon (steps ahead) |
| `input_size` | `2 * h` | Lookback window length |
| `max_steps` | 1000 | Training iterations |
| `scaler_type` | `'standard'` | Input normalization (`'robust'` for outliers) |
| `n_freq_downsample` | `[2, 1, 1]` | Downsampling per stack |
| `learning_rate` | `1e-3` | Adam optimizer LR |

### XLinear (Cross-Learning Linear)

A linear model trained across all series simultaneously. Extremely fast and interpretable. Use it as a baseline before any neural model.

```python
from neuralforecast.models import NHITS, XLinear

models = [
    XLinear(h=7, input_size=28),     # baseline
    NHITS(h=7, input_size=28, max_steps=300),  # neural model
]
nf = NeuralForecast(models=models, freq='D')
```

Running both at once produces side-by-side forecasts in a single DataFrame — column names distinguish models.

---

## 3. The `.fit()` → `.predict()` Workflow

### Preparing the Train/Test Split

neuralforecast's `.predict()` method generates out-of-sample forecasts for the next `h` steps after the last observed date. To evaluate on held-out data, split before fitting.

```python
# Split: hold out last 7 days per series for evaluation
horizon = 7
train = df.groupby('unique_id').apply(
    lambda x: x.iloc[:-horizon]
).reset_index(drop=True)

test = df.groupby('unique_id').apply(
    lambda x: x.iloc[-horizon:]
).reset_index(drop=True)

print(f"Train rows: {len(train):,}")
print(f"Test rows:  {len(test):,}")
```

### Fitting the Model

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MQLoss

nf = NeuralForecast(
    models=[
        NHITS(
            h=horizon,
            input_size=4 * horizon,
            loss=MQLoss(quantiles=[0.1, 0.5, 0.9]),
            max_steps=300,
            scaler_type='standard',
        )
    ],
    freq='D',
)

nf.fit(df=train)
```

### Generating Predictions

```python
forecasts = nf.predict()

print(forecasts.head())
#   unique_id          ds  NHITS-q-0.1  NHITS-q-0.5  NHITS-q-0.9
# 0  baguette  2022-11-14        98.4        131.2        164.0
# 1  baguette  2022-11-15        87.1        118.5        149.8
# ...
```

The output is a DataFrame in the nixtla format — `unique_id`, `ds`, and one column per quantile per model.

---

## 4. Visualizing Probabilistic Forecasts

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_forecast_with_intervals(train, forecasts, series_id, model='NHITS'):
    fig, ax = plt.subplots(figsize=(12, 5))

    # Historical data — last 60 days
    hist = train[train['unique_id'] == series_id].tail(60)
    ax.plot(hist['ds'], hist['y'], color='black', linewidth=1.2, label='Observed')

    # Forecast median
    fc = forecasts[forecasts['unique_id'] == series_id]
    ax.plot(fc['ds'], fc[f'{model}-q-0.5'],
            color='steelblue', linewidth=2, label='Median forecast')

    # 80% prediction interval
    ax.fill_between(
        fc['ds'],
        fc[f'{model}-q-0.1'],
        fc[f'{model}-q-0.9'],
        alpha=0.3,
        color='steelblue',
        label='80% interval',
    )

    ax.set_title(f'{series_id} — 7-Day NHITS Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily sales')
    ax.legend()
    plt.tight_layout()
    plt.show()

plot_forecast_with_intervals(train, forecasts, series_id='baguette')
```

---

## 5. The `.cross_validation()` Method

Cross-validation in time series requires special care: future data must never be used to train the model. neuralforecast implements **rolling-window cross-validation** automatically.

```python
cv_results = nf.cross_validation(
    df=df,
    n_windows=3,     # number of evaluation windows
    step_size=7,     # slide window forward 7 steps each time
)

print(cv_results.head())
#   unique_id          ds     cutoff  y  NHITS-q-0.1  NHITS-q-0.5  NHITS-q-0.9
# 0  baguette  2022-10-03  2022-09-26 ...
```

The `cutoff` column records the last training date for each window. Visualizing cutoffs helps verify that no data leakage occurred.

### Evaluating with CRPS

```python
from utilsforecast.losses import mqloss
from utilsforecast.evaluation import evaluate

# Compute quantile loss across all windows and series
scores = evaluate(
    df=cv_results,
    metrics=[mqloss],
    models=['NHITS'],
    target_col='y',
)
print(scores)
```

---

## 6. The `.simulate()` Method: Sample Paths

Sample paths are Monte Carlo draws from the predictive distribution. They answer the question: "What might actually happen?" rather than "What is the probability of each outcome?"

**Use cases:**
- Scenario analysis ("What if demand hits the 95th percentile for 3 consecutive weeks?")
- Supply chain simulation
- Risk capital calculation by running P&L through simulated demand trajectories

```python
# Generate 200 sample paths for the next 7 days
sample_paths = nf.predict(
    futr_df=None,
    quantiles=None,
    n_samples=200,     # number of Monte Carlo draws
)

# sample_paths shape: (n_series * horizon, n_samples + 2)
# Columns: unique_id, ds, sample_0, sample_1, ..., sample_199
print(sample_paths.shape)

# Plot fan chart for baguette
baguette_paths = sample_paths[sample_paths['unique_id'] == 'baguette']
sample_cols = [c for c in baguette_paths.columns if c.startswith('sample_')]

fig, ax = plt.subplots(figsize=(12, 5))
for col in sample_cols[:50]:   # plot 50 of 200 paths for clarity
    ax.plot(baguette_paths['ds'], baguette_paths[col],
            alpha=0.1, color='steelblue', linewidth=0.8)
ax.set_title('Baguette: 50 Simulated Demand Paths (next 7 days)')
ax.set_xlabel('Date')
ax.set_ylabel('Daily sales')
plt.tight_layout()
plt.show()
```

---

## 7. The `.explain()` Method: Feature Importance

`NeuralForecast.explain()` uses SHAP-based attribution to quantify how much each lagged input contributes to the forecast. This works with NHITS and NBEATS.

```python
# Compute SHAP-based explanations
explanations = nf.explain(df=train)

# explanations is a dict: {model_name: DataFrame}
# Columns: unique_id, ds, lag_1, lag_2, ..., lag_{input_size}
nhits_shap = explanations['NHITS']
print(nhits_shap.head())

# Aggregate importance across series
mean_importance = (
    nhits_shap
    .drop(columns=['unique_id', 'ds'])
    .abs()
    .mean()
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(10, 4))
mean_importance.head(14).plot(kind='bar', ax=ax)
ax.set_title('NHITS: Mean Absolute SHAP Value by Lag')
ax.set_xlabel('Lag')
ax.set_ylabel('Mean |SHAP|')
plt.tight_layout()
plt.show()
```

High SHAP values at lag 7 and lag 14 confirm that the model has learned weekly seasonality.

---

## 8. Losses in neuralforecast

The loss function is a first-class model component — passing a different loss object changes the forecast type without modifying any other code.

| Loss | Import | Output columns | Use case |
|---|---|---|---|
| `MAE` | `losses.pytorch` | Single `{model}` | Point forecast |
| `MSE` | `losses.pytorch` | Single `{model}` | Point forecast, penalizes large errors |
| `MQLoss` | `losses.pytorch` | `{model}-q-{q}` per quantile | Probabilistic forecast |
| `DistributionLoss` | `losses.pytorch` | Mean, std (or shape params) | Parametric distributional |
| `HuberMQLoss` | `losses.pytorch` | `{model}-q-{q}` per quantile | Robust quantile (outlier-tolerant) |

```python
from neuralforecast.losses.pytorch import MAE, MQLoss, HuberMQLoss

# Robust probabilistic forecast for noisy sales data
robust_model = NHITS(
    h=7,
    input_size=28,
    loss=HuberMQLoss(
        quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
        huber_delta=1.35,   # transition point between L1 and L2
    ),
    max_steps=300,
)
```

---

## 9. A Complete Pipeline in 30 Lines

```python
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, XLinear
from neuralforecast.losses.pytorch import MQLoss
from utilsforecast.losses import mqloss
from utilsforecast.evaluation import evaluate

# Load data
url = (
    "https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/"
    "main/datasets/french_bakery_daily.csv"
)
df = pd.read_csv(url, parse_dates=['ds'])

# Instantiate models
nf = NeuralForecast(
    models=[
        XLinear(h=7, input_size=28),
        NHITS(h=7, input_size=28,
              loss=MQLoss(quantiles=[0.1, 0.5, 0.9]),
              max_steps=300, scaler_type='standard'),
    ],
    freq='D',
)

# Cross-validate
cv = nf.cross_validation(df=df, n_windows=3, step_size=7)

# Score
scores = evaluate(df=cv, metrics=[mqloss],
                  models=['XLinear', 'NHITS'], target_col='y')
print(scores)
```

This 30-line pipeline trains two models, runs time-series cross-validation across three windows, and reports pinball loss — a production-ready workflow pattern.

---

## 10. Key Takeaways

- **NeuralForecast wraps models in a unified API.** `.fit()`, `.predict()`, and `.cross_validation()` work identically regardless of which model is inside.
- **NHITS is the default workhorse.** Multi-resolution decomposition, fast training, and stable defaults make it the right starting point.
- **XLinear is always your baseline.** Never report neural model results without first checking whether a linear model does the job.
- **The loss is the forecast type.** `MAE` → point forecast. `MQLoss` → probabilistic forecast. One argument changes.
- **`.simulate()` produces sample paths.** These are essential for supply chain simulation and scenario analysis.
- **`.explain()` surfaces lag importance.** Use it to verify that the model has learned expected patterns (weekly seasonality, trend) rather than noise.

---

## What's Next

Work through [Notebook 01: QuickStart](../notebooks/01_quickstart_neuralforecast.ipynb) to train NHITS on the French Bakery dataset interactively, then explore [Notebook 02: Exploring Datasets](../notebooks/02_exploring_datasets.ipynb) to understand EDA patterns across multiple dataset types.
