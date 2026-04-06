# NeuralForecast Production Patterns

> **Reading time:** ~16 min | **Module:** 6 — Production Patterns | **Prerequisites:** Modules 1-5

## In Brief

This guide covers the patterns that bridge the gap between a working notebook and a robust production system: custom losses, GPU checkpointing, multi-series batch processing, experiment logging, and graceful error handling with fallback models.

Each pattern is self-contained — copy the snippet, adapt the parameters, use it.


<span class="filename">example.py</span>
</div>
The following implementation builds on the approach above:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# Quickcheck: confirm your environment supports GPU training
import torch
print(f"PyTorch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
# Expected on a GPU instance: CUDA available: True
# Expected on CPU-only: CUDA available: False  (still works, just slower)
```

</div>
</div>

<div class="callout-key">

<strong>Key Concept:</strong> This guide covers the patterns that bridge the gap between a working notebook and a robust production system: custom losses, GPU checkpointing, multi-series batch processing, experiment logging, and graceful error handling with fallback models. Each pattern is self-contained — copy the snippet, adapt the parameters, use it.

</div>


---

## 1. Custom Losses

NeuralForecast ships `MQLoss` (quantile), `DistributionLoss` (full distribution), and `MAE`/`MSE` for point forecasting. For production, two customizations are common: asymmetric quantiles and scaled losses.

<div class="callout-insight">

<strong>Insight:</strong> NeuralForecast ships `MQLoss` (quantile), `DistributionLoss` (full distribution), and `MAE`/`MSE` for point forecasting.

</div>


### 1a. Asymmetric Quantile Selection

Stockout costs more than overstock. Express this by weighting upper quantiles.


<span class="filename">example.py</span>
</div>
The following implementation builds on the approach above:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MQLoss

# Standard symmetric quantiles
symmetric = MQLoss(quantiles=[0.1, 0.5, 0.9])

# Asymmetric: more resolution in the upper tail
# Useful when understocking is more costly than overstocking
asymmetric = MQLoss(quantiles=[0.5, 0.7, 0.8, 0.85, 0.9, 0.95])

# The loss function does not change — only the quantiles requested change.
# MQLoss trains N quantile heads simultaneously; more upper quantiles
# give finer-grained resolution where it matters most.

model = NHITS(
    h=7,
    input_size=21,
    loss=asymmetric,
    max_steps=300,
)
```

</div>
</div>

### 1b. DistributionLoss for Full Probabilistic Output

`DistributionLoss` fits a parametric distribution (Normal, StudentT, NegativeBinomial) instead of individual quantiles. Use this when you need sample paths or when demand follows a known distributional family.


<span class="filename">example.py</span>
</div>
The following implementation builds on the approach above:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
from neuralforecast.losses.pytorch import DistributionLoss

# Normal distribution — appropriate for continuous, symmetric demand
normal_loss = DistributionLoss(distribution="Normal", level=[80, 90])

# NegativeBinomial — appropriate for count data (units sold)
# Handles zero-inflated demand common in slow-moving SKUs
nb_loss = DistributionLoss(distribution="NegativeBinomial", level=[80, 90])

model_nb = NHITS(
    h=7,
    input_size=21,
    loss=nb_loss,
    max_steps=300,
)
```

</div>
</div>

**Choosing between MQLoss and DistributionLoss:**

| Criterion | MQLoss | DistributionLoss |
|---|---|---|
| Need specific quantiles | Yes | No — infer from fitted distribution |
| Need sample paths | No | Yes — sample directly from fitted dist. |
| Data is count/integer | Acceptable | Preferred (NegativeBinomial) |
| Calibration priority | Good | Excellent when family matches |

---

## 2. GPU Training and Checkpointing

### 2a. Selecting Device

<div class="callout-key">

<strong>Key Point:</strong> Selecting Device

NeuralForecast uses PyTorch Lightning, which auto-detects GPU.

</div>


NeuralForecast uses PyTorch Lightning, which auto-detects GPU. Override if needed.


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast import NeuralForecast

# Auto-detect: uses GPU if available, CPU otherwise
model_auto = NHITS(h=7, input_size=21, loss=MQLoss(), max_steps=500)

# Explicit GPU selection (multi-GPU machines)
model_gpu = NHITS(
    h=7,
    input_size=21,
    loss=MQLoss(),
    max_steps=500,
    accelerator="gpu",
    devices=1,          # use first GPU only
)

# Explicit CPU (for debugging — never for production with large datasets)
model_cpu = NHITS(
    h=7,
    input_size=21,
    loss=MQLoss(),
    max_steps=500,
    accelerator="cpu",
)
```

</div>
</div>

### 2b. Checkpointing During Training

Long training runs need checkpoints. If the process dies at step 450/500, you want to resume from step 400, not step 0.

```python
import os
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MQLoss

CHECKPOINT_DIR = "/tmp/nf_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

model = NHITS(
    h=7,
    input_size=21,
    loss=MQLoss(quantiles=[0.1, 0.5, 0.8, 0.9]),
    max_steps=500,
    # PyTorch Lightning checkpoint callback parameters
    # passed through as trainer kwargs
    trainer_kwargs={
        "enable_checkpointing": True,
        "default_root_dir": CHECKPOINT_DIR,
    },
)

# NeuralForecast saves checkpoints automatically.
# To resume from a checkpoint after a crash:
# model = NHITS.load_from_checkpoint("/tmp/nf_checkpoints/...")
```

### 2c. Early Stopping

Stop training when validation loss stops improving. Prevents overfitting and reduces training time on easy datasets.

```python
from pytorch_lightning.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor="ptl/val_loss",
    patience=10,        # stop if no improvement for 10 epochs
    mode="min",
    verbose=True,
)

model = NHITS(
    h=7,
    input_size=21,
    loss=MQLoss(),
    max_steps=1000,     # ceiling — early stopping fires before this
    val_check_steps=50, # validate every 50 steps
    trainer_kwargs={"callbacks": [early_stop]},
)
```

---

## 3. Handling Multiple Series at Scale

### 3a. The unique_id Pattern

<div class="callout-insight">

<strong>Insight:</strong> The unique_id Pattern

NeuralForecast trains a single global model across all series identified by `unique_id`.

</div>


NeuralForecast trains a single global model across all series identified by `unique_id`. This is the key scaling mechanism: one model call trains on thousands of products simultaneously.

```python
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MQLoss

# Construct a multi-series DataFrame
# Each (unique_id, ds) pair must be unique
def build_multi_series_df(series_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Build nixtla long format from a dict of {series_name: pd.Series with DatetimeIndex}.
    """
    frames = []
    for uid, series in series_dict.items():
        frame = pd.DataFrame({
            "unique_id": uid,
            "ds": series.index,
            "y": series.values,
        })
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(["unique_id", "ds"])


# Train one model on all series simultaneously
def train_global_model(df: pd.DataFrame, horizon: int = 7) -> NeuralForecast:
    """
    Train a global NHITS model across all unique_ids in df.

    Global models often outperform per-series models when series share patterns
    (e.g., all products in the same bakery follow the same weekly seasonality).
    """
    model = NHITS(
        h=horizon,
        input_size=3 * horizon,
        loss=MQLoss(quantiles=[0.1, 0.5, 0.8, 0.9]),
        max_steps=500,
    )
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df)
    return nf
```

### 3b. Batch Processing for Large Catalogs

When you have thousands of series, process in batches to manage memory.

```python
def forecast_in_batches(
    nf: NeuralForecast,
    all_series_df: pd.DataFrame,
    batch_size: int = 500,
) -> pd.DataFrame:
    """
    Generate forecasts for a large catalog in batches.

    Parameters
    ----------
    nf : NeuralForecast
        Trained NeuralForecast object.
    all_series_df : pd.DataFrame
        Full dataset in nixtla format.
    batch_size : int
        Number of unique series per batch.

    Returns
    -------
    pd.DataFrame
        All forecasts concatenated.
    """
    unique_ids = all_series_df["unique_id"].unique().tolist()
    all_forecasts = []

    for start in range(0, len(unique_ids), batch_size):
        batch_ids = unique_ids[start: start + batch_size]
        batch_df = all_series_df[all_series_df["unique_id"].isin(batch_ids)]

        batch_forecast = nf.predict(df=batch_df)
        all_forecasts.append(batch_forecast)

        print(f"Forecasted {min(start + batch_size, len(unique_ids))}/{len(unique_ids)} series")

    return pd.concat(all_forecasts, ignore_index=True)
```

### 3c. Memory-Efficient batch_size Tuning

`batch_size` in NeuralForecast refers to the gradient update batch, not the number of series. Tuning it controls GPU memory usage.

```python
import torch

def get_recommended_batch_size(n_series: int, horizon: int, input_size: int) -> int:
    """
    Heuristic for initial batch_size based on available GPU memory.

    Rule of thumb:
    - 16 GB GPU: batch_size = 32 for models with input_size > 100
    - 8 GB GPU: batch_size = 16
    - CPU only: batch_size = 8
    """
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if gpu_mem_gb >= 16:
            return 32
        elif gpu_mem_gb >= 8:
            return 16
        else:
            return 8
    return 8  # CPU fallback


# Use in model construction:
batch_size = get_recommended_batch_size(n_series=200, horizon=7, input_size=21)

model = NHITS(
    h=7,
    input_size=21,
    loss=MQLoss(),
    max_steps=500,
    batch_size=batch_size,
)
```

---

## 4. Experiment Logging

### 4a. Weights & Biases Integration

```python
# pip install wandb
import wandb
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MQLoss

# Initialize a wandb run before training
wandb.init(
    project="bakery-forecasting",
    name="nhits-mqloss-v1",
    config={
        "model": "NHITS",
        "horizon": 7,
        "input_size": 21,
        "loss": "MQLoss",
        "quantiles": [0.1, 0.5, 0.8, 0.9],
        "max_steps": 500,
    }
)

model = NHITS(
    h=7,
    input_size=21,
    loss=MQLoss(quantiles=[0.1, 0.5, 0.8, 0.9]),
    max_steps=500,
    # PyTorch Lightning logger — pass WandbLogger
    trainer_kwargs={
        "logger": True,   # uses default TensorBoard; replace with WandbLogger below
    },
)

# For full wandb integration, use the WandbLogger from pytorch_lightning:
# from pytorch_lightning.loggers import WandbLogger
# wandb_logger = WandbLogger(project="bakery-forecasting")
# trainer_kwargs={"logger": wandb_logger}
```

### 4b. TensorBoard Integration

```python
from pytorch_lightning.loggers import TensorBoardLogger

tb_logger = TensorBoardLogger(
    save_dir="/tmp/tb_logs",
    name="nhits_bakery",
)

model = NHITS(
    h=7,
    input_size=21,
    loss=MQLoss(),
    max_steps=500,
    trainer_kwargs={"logger": tb_logger},
)

# View logs: tensorboard --logdir /tmp/tb_logs
```

### 4c. Manual Metric Logging

For lightweight logging without a full experiment tracker:

```python
import json
import hashlib
from datetime import datetime

def log_training_run(
    model_config: dict,
    train_df: pd.DataFrame,
    val_metrics: dict,
    output_path: str = "/tmp/training_runs.jsonl",
) -> str:
    """
    Append a training run record to a JSONL log file.

    Returns the run_id for downstream reference.
    """
    run_id = hashlib.md5(
        (str(model_config) + datetime.utcnow().isoformat()).encode()
    ).hexdigest()[:8]

    record = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "model_config": model_config,
        "n_series": train_df["unique_id"].nunique(),
        "n_rows": len(train_df),
        "val_metrics": val_metrics,
    }

    with open(output_path, "a") as f:
        f.write(json.dumps(record) + "\n")

    return run_id
```

---

## 5. Error Handling and Fallback Models

Production systems must not fail silently. Two patterns work well: a try/except fallback to a simpler model, and a validation step that checks forecast reasonableness before serving.

### 5a. Fallback to XLinear on NHITS Failure

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, LinearRegressor
from neuralforecast.losses.pytorch import MQLoss
import pandas as pd


def train_with_fallback(
    df: pd.DataFrame,
    horizon: int = 7,
    freq: str = "D",
) -> tuple[NeuralForecast, str]:
    """
    Attempt to train NHITS. Fall back to LinearRegressor on failure.

    Returns (nf, model_name) so downstream code knows which model was used.
    """
    loss = MQLoss(quantiles=[0.1, 0.5, 0.8, 0.9])

    try:
        model = NHITS(h=horizon, input_size=3 * horizon, loss=loss, max_steps=300)
        nf = NeuralForecast(models=[model], freq=freq)
        nf.fit(df)
        return nf, "NHITS"

    except Exception as e:
        print(f"NHITS training failed: {e}. Falling back to LinearRegressor.")

        try:
            fallback = LinearRegressor(h=horizon, input_size=3 * horizon, loss=loss)
            nf = NeuralForecast(models=[fallback], freq=freq)
            nf.fit(df)
            return nf, "LinearRegressor"

        except Exception as e2:
            raise RuntimeError(
                f"Both NHITS and LinearRegressor failed. "
                f"NHITS error: {e}. LinearRegressor error: {e2}."
            ) from e2
```

### 5b. Forecast Sanity Checks

Reject forecasts that are implausible given historical data.

```python
def validate_forecast(
    forecast: pd.DataFrame,
    train_df: pd.DataFrame,
    median_col: str,
    max_multiplier: float = 5.0,
    min_value: float = 0.0,
) -> pd.DataFrame:
    """
    Reject series whose median forecast exceeds max_multiplier * historical median.

    Returns the forecast DataFrame with a 'status' column: 'ok' or 'rejected'.
    """
    hist_medians = (
        train_df.groupby("unique_id")["y"]
        .median()
        .rename("hist_median")
    )

    fc = forecast.merge(hist_medians, on="unique_id", how="left")
    fc_median = fc.groupby("unique_id")[median_col].median()

    status = {}
    for uid in fc["unique_id"].unique():
        hist_med = hist_medians.get(uid, 0)
        fc_med = fc_median.get(uid, 0)

        if fc_med < min_value:
            status[uid] = "rejected_negative"
        elif hist_med > 0 and fc_med > max_multiplier * hist_med:
            status[uid] = f"rejected_spike_{fc_med/hist_med:.1f}x"
        else:
            status[uid] = "ok"

    forecast["status"] = forecast["unique_id"].map(status)
    n_rejected = (forecast["status"] != "ok").any()
    if n_rejected:
        print(f"Warning: {(forecast['status'] != 'ok').sum()} rows rejected.")

    return forecast
```

---

## 6. Pattern Summary

| Pattern | When to Use | Key Parameter |
|---|---|---|
| Asymmetric quantiles | Asymmetric cost function | `quantiles` list |
| DistributionLoss | Need sample paths, count data | `distribution=` |
| Checkpointing | Training > 10 minutes | `enable_checkpointing=True` |
| Early stopping | Risk of overfitting | `patience=10` |
| Global model | Shared patterns across series | Single `NeuralForecast.fit()` |
| Batch forecasting | > 1000 series | `batch_size` loop |
| Fallback model | Mission-critical pipelines | Try/except chain |
| Forecast validation | Before serving predictions | `max_multiplier` threshold |

---

**Next:** Notebook `01_end_to_end_pipeline.ipynb` — run the complete pipeline on real bakery data.


## Practice Questions

**Question 1 — Conceptual:** Based on the concepts in this guide, explain in your own words why the core technique matters and when you would choose it over alternatives.

**Question 2 — Application:** Sketch out how you would apply the main concept from this guide to a real-world dataset or problem you have encountered. What would you need to watch out for?


---

## Cross-References

<a class="link-card" href="./02_neuralforecast_patterns.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Interactive slide deck covering the key concepts with visual examples.</div>
</a>

<a class="link-card" href="../notebooks/01_end_to_end_pipeline.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises and real data.</div>
</a>
