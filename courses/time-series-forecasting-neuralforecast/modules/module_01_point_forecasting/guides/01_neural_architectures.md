# Neural Forecasting Architectures

> **Reading time:** ~14 min | **Module:** 1 — Point Forecasting | **Prerequisites:** Module 0

## Start Here: Training NHITS in 60 Seconds

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

<div class="callout-insight">
<strong>Insight:</strong> example.py


The following implementation builds on the approach above:




Run this first.
</div>


The following implementation builds on the approach above:

```python
import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

# Load French Bakery data
url = "https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/french_bakery.csv"
df = pd.read_csv(url, parse_dates=["ds"])
# Keep just one product to start
bakery = df[df["unique_id"] == "Baguette"].copy()

# Train NHITS — 7-day forecast horizon, 28-day lookback
nf = NeuralForecast(
    models=[NHITS(h=7, input_size=28, max_steps=500)],
    freq="D"
)
nf.fit(bakery)
forecast = nf.predict()
print(forecast.head())
```
</div>

Run this first. The rest of the guide explains why it works.

---

## The Neural Forecasting Landscape

Before 2020, most production forecasters used classical methods: ARIMA, ETS, Prophet. These methods are interpretable and fast, but they struggle with:

<div class="callout-key">
<strong>Key Point:</strong> Before 2020, most production forecasters used classical methods: ARIMA, ETS, Prophet.
</div>


- Long-horizon multi-step forecasts (error compounds)
- Capturing non-linear seasonal interactions
- Scaling across hundreds of series simultaneously

Neural models address these weaknesses by learning flexible non-linear mappings from a window of historical values to a multi-step forecast output directly. This "sequence-to-sequence" framing trains a single model across all series, transferring patterns learned from one series to another.

### The Core Paradigm

```
Input window                Output forecast
[y_{t-H+1}, ..., y_t]  →   [y_{t+1}, ..., y_{t+h}]
```

Every neural forecasting model maps a fixed-length input window to a fixed-length forecast horizon. The models differ in *how* they learn this mapping.

---

## Architecture Overview

| Architecture | Family | Params | Speed | Best For |
|---|---|---|---|---|
| **MLP** | Feed-forward | ~10K | Very fast | Baselines, simple trends |
| **N-BEATS** | Residual MLP | ~2M | Fast | Interpretable decomposition |
| **NHITS** | Hierarchical MLP | ~800K | Fast | Long horizons, multi-frequency |
| **PatchTST** | Transformer | ~5M | Moderate | Long sequences, channel mixing |
| **DLinear / NLinear** | Linear | ~1K | Fastest | Surprisingly strong baseline |

<div class="callout-info">
<strong>Info:</strong> NHITS achieves state-of-the-art long-horizon accuracy with fewer parameters than Transformer models, by using structured inductive biases instead of raw capacity.
</div>


The table shows a key insight: **more parameters is not always better**. NHITS achieves state-of-the-art long-horizon accuracy with fewer parameters than Transformer models, by using structured inductive biases instead of raw capacity.

---

## MLP: The Baseline

A multi-layer perceptron (MLP) is the simplest neural forecaster. It flattens the input window into a vector, passes it through dense layers with non-linearities, and produces the forecast vector.

<div class="callout-warning">
<strong>Warning:</strong> A multi-layer perceptron (MLP) is the simplest neural forecaster.
</div>


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

The following implementation builds on the approach above:

```python
from neuralforecast.models import MLP

model = MLP(
    h=7,
    input_size=28,
    num_layers=2,
    hidden_size=256,
    max_steps=500
)
```
</div>

**Weakness**: MLPs treat each input time step as an independent feature. There is no structure that forces the model to reason about temporal order, trends, or seasonality — it must discover everything from data.

---

## N-BEATS: Neural Basis Expansion

N-BEATS (Neural Basis Expansion Analysis for Time Series, Oreshkin et al. 2019) introduces two innovations:

<div class="callout-insight">
<strong>Insight:</strong> N-BEATS (Neural Basis Expansion Analysis for Time Series, Oreshkin et al.
</div>


1. **Doubly residual stacking**: Each block produces a *backcast* (what it explains from the input) and a *forecast* (its contribution to the output). Blocks are chained so each block processes the residual not yet explained by previous blocks.

2. **Basis expansion**: In its interpretable variant, N-BEATS constrains the output of each stack to lie in a specific basis: polynomial (trend) or harmonic/Fourier (seasonality). This makes the architecture interpretable by construction.

```
Stack 1 (trend):       learns f(t) ≈ polynomial
Stack 2 (seasonality): learns f(t) ≈ sum of sinusoids
Residual connection:   next stack gets what's left unexplained
```

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

The following implementation builds on the approach above:

```python
from neuralforecast.models import NBEATS

model = NBEATS(
    h=7,
    input_size=28,
    stack_types=["trend", "seasonality"],
    n_blocks=[3, 3],
    max_steps=500
)
```
</div>

**Strength**: Excellent on M3/M4 competition benchmarks. Interpretable decomposition. Clean inductive bias.  
**Weakness**: The fixed basis can hurt when the signal does not cleanly decompose into trend + seasonality.

---

## NHITS: The Default Choice

NHITS (Neural Hierarchical Interpolation for Time Series, Challu et al. 2022, AAAI 2023) extends N-BEATS with two additional ideas designed specifically for long-horizon forecasting:

<div class="callout-key">
<strong>Key Point:</strong> NHITS (Neural Hierarchical Interpolation for Time Series, Challu et al.
</div>


### 1. Multi-Rate Signal Sampling

Each stack in NHITS applies a `MaxPool` operation with a different kernel size to the input window before feeding it to the MLP block. This means:

- Stack 1 (low expressiveness ratio): heavily pooled — sees a coarse, low-frequency summary
- Stack 2 (medium ratio): moderately pooled — sees medium-frequency patterns
- Stack 3 (high ratio): unpooled — sees fine-grained, high-frequency detail

This forces each stack to specialize in a different frequency band of the signal. Stack 1 handles trend; stack 2 handles weekly seasonality; stack 3 handles day-to-day variation.

### 2. Hierarchical Interpolation

Each stack does not predict all `h` output steps directly. Instead, it predicts a coarser grid — say `h/4` points — and upsamples to the full horizon using interpolation. The expressiveness ratio $r_s$ controls how coarse each stack's output is:

$$\hat{y}_{t+1:t+h} = \sum_{s=1}^{S} \text{Interp}_s\left(\text{MLP}_s\left(\text{Pool}_s(x)\right)\right)$$

Where `Interp_s` upsamples stack $s$'s coarse output to the full `h`-step horizon.

**Result**: Lower-frequency stacks contribute smooth, slowly-varying forecast components. Higher-frequency stacks layer in fine-grained detail. The final forecast is the additive combination.

### NHITS Architecture Diagram

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#e8f5e9", "primaryBorderColor": "#4caf50", "primaryTextColor": "#212121", "secondaryColor": "#e3f2fd", "tertiaryColor": "#fff8e1", "lineColor": "#757575", "fontFamily": "Inter, sans-serif", "fontSize": "14px"}}}%%
flowchart TD
    X["Input window\n[y_{t-H+1}, ..., y_t]"]
    X --> P1["MaxPool(kernel=4)\nCoarse view"]
    X --> P2["MaxPool(kernel=2)\nMedium view"]
    X --> P3["No pooling\nFine view"]
    
    P1 --> B1["MLP Block 1\nLow expressiveness"]
    P2 --> B2["MLP Block 2\nMed expressiveness"]
    P3 --> B3["MLP Block 3\nHigh expressiveness"]
    
    B1 --> F1["Forecast 1\nTrend component\n(interpolated from h/4 points)"]
    B2 --> F2["Forecast 2\nSeasonal component\n(interpolated from h/2 points)"]
    B3 --> F3["Forecast 3\nResidual detail\n(full h points)"]
    
    B1 --> R1["Backcast 1"]
    B2 --> R2["Backcast 2"]
    B3 --> R3["Backcast 3"]
    
    R1 -->|"subtract residual"| B2
    R2 -->|"subtract residual"| B3
    
    F1 & F2 & F3 --> SUM["Sum\nFinal forecast"]
```
</div>

### Why NHITS Is the Default Choice

1. **Accuracy**: +20% average improvement over Transformer-based models on long-horizon benchmarks (Challu et al. 2022).
2. **Speed**: 50x faster than the Informer Transformer.
3. **Parameter efficiency**: ~800K parameters vs 5M+ for PatchTST.
4. **Stability**: The multi-rate decomposition prevents the model from overfitting to spurious high-frequency noise.
5. **Practical**: Works well without extensive tuning. The default architecture handles most time series forecasting problems out of the box.


<div class="flow">
<div class="flow-step mint">1. Accuracy</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">2. Speed</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">3. Parameter efficiency</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. Stability</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step rose">5. Practical</div>
</div>

```python
from neuralforecast.models import NHITS

# Full NHITS configuration
model = NHITS(
    h=7,                        # forecast horizon
    input_size=28,              # 4x horizon — good rule of thumb
    stack_types=["identity"] * 3,
    n_blocks=[1, 1, 1],
    mlp_units=[[512, 512]] * 3,
    n_freq_downsample=[4, 2, 1],  # multi-rate pooling ratios
    dropout_prob_theta=0.0,
    learning_rate=1e-3,
    max_steps=1000,
    scaler_type="robust",       # handles outliers in bakery sales
    loss="MAE"                  # appropriate for counts/sales
)
```

---

## PatchTST: When to Use Transformers

PatchTST (Nie et al. 2023) divides the input time series into non-overlapping patches and treats each patch as a token in a Transformer encoder. This reduces the sequence length from `T` to `T/P` (where `P` is patch size), making attention computationally feasible.

<div class="callout-info">
<strong>Info:</strong> 2023) divides the input time series into non-overlapping patches and treats each patch as a token in a Transformer encoder.
</div>


```python
from neuralforecast.models import PatchTST

model = PatchTST(
    h=7,
    input_size=56,  # transformers benefit from longer context
    patch_len=7,    # each patch = one week
    stride=7,
    d_model=64,
    n_heads=4,
    max_steps=1000
)
```

**Use PatchTST when:**
- You have very long input sequences (120+ steps) where NHITS starts to lose signal
- You are working with multiple related series where cross-channel attention helps
- Computation budget allows for slower training

**Stick with NHITS when:**
- Horizon is 7–90 steps
- Fast iteration matters
- You want stable results without careful tuning

---

## Linear Models: The Surprising Baseline

A 2023 paper (Zeng et al.) showed that simple linear models — DLinear and NLinear — outperform many Transformer architectures on standard benchmarks. This sparked significant debate.

```python
from neuralforecast.models import DLinear

model = DLinear(
    h=7,
    input_size=28,
    max_steps=500
)
```

**Lesson**: Always benchmark against a linear model. If your neural model cannot beat DLinear on your dataset, the signal is likely not complex enough to justify the added complexity.

---

## Applying to French Bakery Data

The French Bakery dataset from Kaggle contains daily sales for multiple bakery products (Baguette, Pain au Chocolat, Croissant, etc.) from a French boulangerie over approximately 3 years. It captures:

- **Weekly seasonality**: strong weekend peaks for baguettes
- **Annual seasonality**: summer vacations, holidays
- **Trend**: gradual growth in overall sales
- **Irregularity**: weather effects, local events

This makes it ideal for testing multi-frequency models like NHITS.

```python
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, NBEATS, DLinear

url = "https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/french_bakery.csv"
df = pd.read_csv(url, parse_dates=["ds"])

print(df.head())
print(f"\nUnique products: {df['unique_id'].unique()}")
print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
print(f"Rows: {len(df)}")

# Train multiple models for comparison
models = [
    DLinear(h=7, input_size=28, max_steps=500),
    NBEATS(h=7, input_size=28, max_steps=500),
    NHITS(h=7, input_size=28, max_steps=500, scaler_type="robust"),
]

nf = NeuralForecast(models=models, freq="D")
nf.fit(df)
forecasts = nf.predict()
print(forecasts)
```

---

## Architecture Selection Guide

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#e8f5e9", "primaryBorderColor": "#4caf50", "primaryTextColor": "#212121", "secondaryColor": "#e3f2fd", "tertiaryColor": "#fff8e1", "lineColor": "#757575", "fontFamily": "Inter, sans-serif", "fontSize": "14px"}}}%%
flowchart TD
    Q1{Horizon length?}
    Q1 -->|"h <= 14"| Q2{Speed critical?}
    Q1 -->|"h = 15–90"| Q3{Long context needed?}
    Q1 -->|"h > 90"| NHITS2["NHITS\n(hierarchical interpolation\nscales to long horizons)"]
    
    Q2 -->|Yes| DL["DLinear\n(fastest, surprisingly strong)"]
    Q2 -->|No| NHITS1["NHITS\n(strong default)"]
    
    Q3 -->|Yes| PT["PatchTST\n(input_size 56-336)"]
    Q3 -->|No| NHITS1
```

For the bakery forecasting problem (h=7, daily data, multiple products), **NHITS is the right default**.

---

## Key Takeaways

1. Neural forecasters map a fixed input window directly to a multi-step forecast — no recursive error accumulation.
2. MLP and DLinear are fast baselines; always include them in comparisons.
3. N-BEATS adds interpretable decomposition via basis expansion and residual stacking.
4. NHITS extends N-BEATS with multi-rate pooling and hierarchical interpolation, making it the most accurate and efficient choice for most horizons.
5. PatchTST and other Transformers add cross-channel attention — useful for long sequences or multi-series problems, but at higher compute cost.
6. For French Bakery data (h=7, daily), use NHITS with `input_size=28`, `scaler_type="robust"`.

---

## Next Steps

- **Notebook 01**: Train NHITS on bakery data, evaluate with MAE and MSE
- **Guide 02**: Hyperparameter tuning — input_size, scaler_type, loss functions
- **Notebook 02**: Cross-validation for honest evaluation


## Practice Questions

**Question 1 — Conceptual:** Based on the concepts in this guide, explain in your own words why the core technique matters and when you would choose it over alternatives.

**Question 2 — Application:** Sketch out how you would apply the main concept from this guide to a real-world dataset or problem you have encountered. What would you need to watch out for?



---

## Cross-References

<a class="link-card" href="./01_neural_architectures.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Interactive slide deck covering the key concepts with visual examples.</div>
</a>

<a class="link-card" href="../notebooks/01_training_nhits.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises and real data.</div>
</a>
