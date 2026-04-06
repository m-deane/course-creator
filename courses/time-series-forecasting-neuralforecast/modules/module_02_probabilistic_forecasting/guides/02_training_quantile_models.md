# Training Quantile Models: MQLoss and the Pinball Loss

> **Reading time:** ~16 min | **Module:** 2 — Probabilistic Forecasting | **Prerequisites:** Module 1

## In Brief

NeuralForecast produces probabilistic forecasts by training with a loss function that directly optimizes quantile accuracy. The multi-quantile loss (MQLoss) is a sum of pinball losses — one per quantile level per time step. Understanding this loss function tells you exactly what the model is optimizing, what its output columns mean, and why it produces marginals rather than joint distributions.

<div class="callout-insight">

<strong>Insight:</strong> MQLoss trains the model to minimize a weighted asymmetric penalty. Overforecasting and underforecasting are penalized differently depending on the quantile level. This is what makes the output a calibrated quantile, not just a point estimate with noise.

</div>

<div class="callout-key">

<strong>Key Concept:</strong> NeuralForecast produces probabilistic forecasts by training with a loss function that directly optimizes quantile accuracy. The multi-quantile loss (MQLoss) is a sum of pinball losses — one per quantile level per time step.

</div>


---

## Start Here: Training with MQLoss

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MQLoss

# ── Load French Bakery data ──────────────────────────────────────────────────
url = "https://raw.githubusercontent.com/nicholasjmorales/French-Bakery-Daily-Transactional-Dataset/main/Bakery_sales.csv"
raw = pd.read_csv(url, parse_dates=["date"])

baguettes = (
    raw[raw["article"].str.upper().str.contains("BAGUETTE")]
    .groupby("date")["Quantity"]
    .sum()
    .reset_index()
    .rename(columns={"date": "ds", "Quantity": "y"})
    .assign(unique_id="bakery_total")
    .sort_values("ds")
    .reset_index(drop=True)
)

cutoff = baguettes["ds"].max() - pd.Timedelta(days=14)
train_df = baguettes[baguettes["ds"] <= cutoff]

# Train with three quantile levels: 50%, 80%, 90%
model = NHITS(
    h=7,
    input_size=28,
    loss=MQLoss(level=[50, 80, 90]),
    max_steps=500,
    random_seed=42,
)

nf = NeuralForecast(models=[model], freq="D")
nf.fit(df=train_df)
forecast = nf.predict()

print("Output columns:")
for col in forecast.columns:
    print(f"  {col}")
```

</div>

Expected output:
```
Output columns:
  unique_id
  ds
  NHITS        ← point forecast (mean / median depending on loss)
  NHITS-lo-50  ← 25th percentile
  NHITS-hi-50  ← 75th percentile
  NHITS-lo-80  ← 10th percentile
  NHITS-hi-80  ← 90th percentile
  NHITS-lo-90  ← 5th percentile
  NHITS-hi-90  ← 95th percentile
```

The `level` parameter controls the **coverage** of each prediction interval. `level=[80]` asks for an interval that should contain the true value 80% of the time. NeuralForecast symmetrically places this at the 10th and 90th percentiles.

---

## The Pinball Loss: What the Model Optimizes

The pinball loss (also called the quantile loss or check function) is an asymmetric loss function that, when minimized in expectation, produces the desired quantile of the distribution.

<div class="callout-key">

<strong>Key Point:</strong> The pinball loss (also called the quantile loss or check function) is an asymmetric loss function that, when minimized in expectation, produces the desired quantile of the distribution.

</div>


For quantile level $q \in (0, 1)$ and residual $u = y - \hat{y}$:

$$\rho_q(u) = \begin{cases} q \cdot u & \text{if } u \geq 0 \quad \text{(underprediction)} \\ (q - 1) \cdot u & \text{if } u < 0 \quad \text{(overprediction)} \end{cases}$$

Which simplifies to:

$$\rho_q(u) = u \cdot (q - \mathbf{1}[u < 0])$$

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>


```python
# ── Visualize the pinball loss for different quantiles ────────────────────────
def pinball_loss(u, q):
    """Pinball loss for quantile q and residual u = y - yhat."""
    return np.where(u >= 0, q * u, (q - 1) * u)

u_values = np.linspace(-3, 3, 500)
quantile_levels = [0.10, 0.25, 0.50, 0.75, 0.90]
colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: individual loss curves
ax = axes[0]
for q, color in zip(quantile_levels, colors):
    loss = pinball_loss(u_values, q)
    ax.plot(u_values, loss, color=color, linewidth=2, label=f"q = {q:.2f}")
ax.axvline(0, color="black", linestyle="--", alpha=0.4)
ax.set_xlabel("Residual u = y − ŷ")
ax.set_ylabel("Pinball loss")
ax.set_title("Pinball Loss for Different Quantiles")
ax.legend()
ax.set_ylim(-0.1, 2.5)
ax.grid(alpha=0.3)

# Right: asymmetry illustration for q=0.90
ax = axes[1]
q = 0.90
loss_pos = pinball_loss(u_values[u_values >= 0], q)
loss_neg = pinball_loss(u_values[u_values < 0], q)
ax.plot(u_values[u_values >= 0], loss_pos, color="#1f77b4", linewidth=3,
        label=f"Underprediction: slope = {q:.2f}")
ax.plot(u_values[u_values < 0], loss_neg, color="#d62728", linewidth=3,
        label=f"Overprediction: slope = {1-q:.2f}")
ax.axvline(0, color="black", linestyle="--", alpha=0.4)
ax.set_xlabel("Residual u = y − ŷ")
ax.set_ylabel("Pinball loss")
ax.set_title("Pinball Loss for q = 0.90\n(90% more costly to underpredict)")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Numeric illustration
print("Pinball loss for q=0.90 with residual +10 (underpredict by 10):")
print(f"  Loss = 0.90 × 10 = {0.90 * 10:.1f}")
print("\nPinball loss for q=0.90 with residual -10 (overpredict by 10):")
print(f"  Loss = (0.90 - 1) × (-10) = {(0.90 - 1) * (-10):.1f}")
print("\nFor q=0.90: underprediction is 9x more costly than overprediction.")
print("The optimal predictor is the 90th percentile — below which you're right 90% of the time.")
```

</div>

### Why This Produces the Quantile

The minimizer of the expected pinball loss $E[\rho_q(Y - c)]$ over constant $c$ is the $q$-quantile of $Y$.

**Intuition:** At $q = 0.90$, underpredicting costs 0.90 per unit while overpredicting costs only 0.10 per unit. The optimal predictor compensates by setting the threshold high enough that it underpredicts only 10% of the time — which is exactly the 90th percentile.

---

## Multi-Quantile Loss: Training Multiple Quantiles Simultaneously

MQLoss trains all requested quantile levels simultaneously in a single forward pass:

<div class="callout-insight">

<strong>Insight:</strong> MQLoss trains all requested quantile levels simultaneously in a single forward pass:

$$\mathcal{L}_{\text{MQ}} = \frac{1}{H \cdot |Q|} \sum_{t=1}^{H} \sum_{q \in Q} \rho_q\!\left(y_t - \hat{q}_t^q\ri...

</div>


$$\mathcal{L}_{\text{MQ}} = \frac{1}{H \cdot |Q|} \sum_{t=1}^{H} \sum_{q \in Q} \rho_q\!\left(y_t - \hat{q}_t^q\right)$$

where $Q$ is the set of quantile levels and $H$ is the forecast horizon.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>


```python
# ── Compare single vs multi-quantile training ─────────────────────────────────
from neuralforecast.losses.pytorch import QuantileLoss

# Single quantile (QuantileLoss)
model_single_q90 = NHITS(
    h=7,
    input_size=28,
    loss=QuantileLoss(q=0.9),
    max_steps=300,
    random_seed=42,
)

# Multiple quantiles (MQLoss) — more efficient, shared representation
model_multi = NHITS(
    h=7,
    input_size=28,
    loss=MQLoss(level=[50, 80, 90]),
    max_steps=300,
    random_seed=42,
)

# MQLoss trains one network that simultaneously outputs all quantiles.
# QuantileLoss trains a separate prediction for a single quantile.
# For multiple levels, MQLoss is more efficient and produces consistent quantiles
# (they won't cross because they share the same internal representation).
print("MQLoss level=[50, 80, 90] trains quantiles:")
print("  q ∈ {0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95}")
print("  (symmetric around each level: level=80 → q=0.10 and q=0.90)")
```

</div>

---

## Interpreting the Output Columns

NeuralForecast uses a consistent naming convention for quantile output columns:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>


```python
# ── Decode output column names ────────────────────────────────────────────────
print("Column naming convention: {model}-{lo|hi}-{level}")
print()

level_to_quantiles = {
    50: (0.25, 0.75),
    80: (0.10, 0.90),
    90: (0.05, 0.95),
}

for level, (lo_q, hi_q) in level_to_quantiles.items():
    print(f"  level={level}:")
    print(f"    NHITS-lo-{level} = {lo_q:.0%} quantile (lower bound)")
    print(f"    NHITS-hi-{level} = {hi_q:.0%} quantile (upper bound)")
    print(f"    Expected coverage: {level}% of actuals should fall within")
    print()

# Verify coverage on held-out data
test_df = baguettes[baguettes["ds"] > cutoff].merge(
    forecast[["ds", "NHITS-lo-80", "NHITS-hi-80", "NHITS-lo-90", "NHITS-hi-90"]],
    on="ds", how="inner"
)

for level in [80, 90]:
    inside = (
        (test_df["y"] >= test_df[f"NHITS-lo-{level}"]) &
        (test_df["y"] <= test_df[f"NHITS-hi-{level}"])
    ).mean()
    print(f"Empirical coverage at {level}%: {inside:.1%} (target: {level}%)")
```

</div>

---

## Visualizing Prediction Intervals: Fan Charts

Fan charts layer multiple prediction intervals to show the full uncertainty shape.

```python
# ── Build a fan chart with multiple quantile levels ───────────────────────────
# Train with more levels for a smoother fan
model_fan = NHITS(
    h=14,
    input_size=28,
    loss=MQLoss(level=[50, 60, 70, 80, 90]),
    max_steps=500,
    random_seed=42,
)

nf_fan = NeuralForecast(models=[model_fan], freq="D")
nf_fan.fit(df=train_df)
forecast_fan = nf_fan.predict()

fig, ax = plt.subplots(figsize=(14, 6))

# Layer intervals from widest to narrowest
levels = [90, 80, 70, 60, 50]
alphas = [0.15, 0.20, 0.25, 0.30, 0.35]
blues = ["#cfe2f3", "#9fc5e8", "#6fa8dc", "#3d85c8", "#1155cc"]

for level, alpha, color in zip(levels, alphas, blues):
    lo_col = f"NHITS-lo-{level}"
    hi_col = f"NHITS-hi-{level}"
    if lo_col in forecast_fan.columns:
        ax.fill_between(
            forecast_fan["ds"],
            forecast_fan[lo_col],
            forecast_fan[hi_col],
            alpha=alpha, color=color, label=f"{level}% interval"
        )

ax.plot(forecast_fan["ds"], forecast_fan["NHITS"], color="#1155cc",
        linewidth=2.5, label="Point forecast", zorder=5)

# Actuals
recent_actuals = baguettes[
    (baguettes["ds"] >= train_df["ds"].max() - pd.Timedelta(days=14)) &
    (baguettes["ds"] <= cutoff)
]
test_actuals = baguettes[baguettes["ds"] > cutoff].head(14)

ax.plot(recent_actuals["ds"], recent_actuals["y"], "k-", linewidth=1.5,
        alpha=0.6, label="Historical")
ax.plot(test_actuals["ds"], test_actuals["y"], "ko", markersize=5,
        label="Actual (test)")

ax.axvline(cutoff, color="red", linestyle="--", alpha=0.5, label="Train/test cutoff")
ax.set_title("NHITS Fan Chart: Marginal Prediction Intervals\n(5 quantile levels stacked)")
ax.set_xlabel("Date")
ax.set_ylabel("Baguettes sold")
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.show()
```

Verify that the quantiles are properly ordered (no crossings where a tighter interval is wider than a broader one):

```python
# ── Check: are the quantiles non-crossing? ────────────────────────────────────
# A well-trained MQLoss model should not have crossing quantiles
# (where a tighter interval is wider than a broader one)

crossings = (
    forecast_fan["NHITS-hi-80"] > forecast_fan["NHITS-hi-90"]
).sum()
print(f"Quantile crossings (hi-80 > hi-90): {crossings} / {len(forecast_fan)}")
print("A value of 0 means quantiles are properly ordered (no crossings).")
```

---

## Effect of level= Settings: More Levels = Smoother Uncertainty

```python
# ── Compare sparse vs dense quantile coverage ─────────────────────────────────
configs = {
    "2 levels [80, 90]": [80, 90],
    "4 levels [50, 70, 80, 90]": [50, 70, 80, 90],
    "6 levels [50, 60, 70, 80, 90, 95]": [50, 60, 70, 80, 90, 95],
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

for ax, (title, levels) in zip(axes, configs.items()):
    m = NHITS(
        h=7, input_size=28,
        loss=MQLoss(level=levels),
        max_steps=300, random_seed=42,
    )
    nf_temp = NeuralForecast(models=[m], freq="D")
    nf_temp.fit(df=train_df)
    fc = nf_temp.predict()

    # Plot the widest interval available
    lo_col = f"NHITS-lo-{max(levels)}"
    hi_col = f"NHITS-hi-{max(levels)}"
    ax.fill_between(fc["ds"], fc[lo_col], fc[hi_col],
                    alpha=0.3, color="steelblue")

    # Plot intermediate intervals if available
    for lvl in sorted(levels)[:-1]:
        lo = f"NHITS-lo-{lvl}"
        hi = f"NHITS-hi-{lvl}"
        if lo in fc.columns:
            ax.fill_between(fc["ds"], fc[lo], fc[hi], alpha=0.2, color="steelblue")

    ax.plot(fc["ds"], fc["NHITS"], color="steelblue", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.tick_params(axis="x", rotation=30)

axes[0].set_ylabel("Baguettes sold")
plt.suptitle("Effect of level= Settings: More Levels → Smoother Fan Chart", fontsize=12)
plt.tight_layout()
plt.show()

print("More quantile levels give a smoother visual representation of uncertainty.")
print("The computational cost increases linearly with the number of levels.")
print("For production use, level=[80, 90] is usually sufficient.")
```

---

## MQLoss vs DistributionLoss

NeuralForecast also supports parametric distribution losses. These assume a distributional form (e.g., Normal, Negative Binomial) and optimize its parameters.

```python
from neuralforecast.losses.pytorch import DistributionLoss

# Normal distribution loss
model_normal = NHITS(
    h=7,
    input_size=28,
    loss=DistributionLoss("Normal", level=[80, 90]),
    max_steps=300,
    random_seed=42,
)

# Negative Binomial — better for count data like bakery transactions
model_negbinom = NHITS(
    h=7,
    input_size=28,
    loss=DistributionLoss("NegativeBinomial", level=[80, 90]),
    max_steps=300,
    random_seed=42,
)
```

| Property | MQLoss | DistributionLoss |
|----------|--------|-----------------|
| **Assumption** | None (nonparametric) | Specific distribution family |
| **Output** | Specific quantile levels | Any quantile from fitted distribution |
| **Count data** | Works fine | NegativeBinomial is more natural |
| **Skewed data** | Naturally handles skew | Depends on family choice |
| **Crossing quantiles** | Possible if not careful | Impossible (derived from CDF) |
| **Training stability** | Generally stable | More sensitive to initialization |

For bakery data (positive counts, potentially right-skewed), `DistributionLoss("NegativeBinomial")` is often more appropriate. For general real-valued data, MQLoss is the safer default.

```python
# ── Compare coverage: MQLoss vs DistributionLoss ──────────────────────────────
models_compare = [
    NHITS(h=7, input_size=28, loss=MQLoss(level=[80, 90]),
          max_steps=300, random_seed=42),
    NHITS(h=7, input_size=28, loss=DistributionLoss("Normal", level=[80, 90]),
          max_steps=300, random_seed=42),
    NHITS(h=7, input_size=28, loss=DistributionLoss("NegativeBinomial", level=[80, 90]),
          max_steps=300, random_seed=42),
]

model_labels = ["MQLoss", "Normal", "NegBinomial"]

results = []
for model, label in zip(models_compare, model_labels):
    # Rename to avoid collision
    model_clone = model
    nf_temp = NeuralForecast(models=[model_clone], freq="D")
    nf_temp.fit(df=train_df)
    fc = nf_temp.predict()

    # Find the hi-80 column regardless of model name prefix
    hi80_col = [c for c in fc.columns if "hi-80" in c][0]
    lo80_col = [c for c in fc.columns if "lo-80" in c][0]

    test_merged = test_df[["ds", "y"]].merge(
        fc[["ds", lo80_col, hi80_col]], on="ds"
    )
    coverage = (
        (test_merged["y"] >= test_merged[lo80_col]) &
        (test_merged["y"] <= test_merged[hi80_col])
    ).mean()
    results.append({"model": label, "coverage_80": coverage})

results_df = pd.DataFrame(results)
print("\n80% Interval Empirical Coverage:")
print(results_df.to_string(index=False))
print("\nTarget: 80%. Models close to 80% are well-calibrated.")
```

---


<div class="compare">
<div class="compare-card">
<div class="header before">MQLoss</div>
<div class="body">

See detailed comparison in the table above.

</div>
</div>
<div class="compare-card">
<div class="header after">DistributionLoss</div>
<div class="body">

See detailed comparison in the table above.

</div>
</div>
</div>

## Why This Is Necessary But Not Sufficient

MQLoss gives you well-calibrated marginal quantiles. For single-step decisions, this is exactly what you need.

For multi-step decisions, you need to take one more step.

```python
# ── The preview: what MQLoss cannot do ───────────────────────────────────────
print("MQLoss CAN answer:")
print("  'What is the 80th percentile of Monday demand?'  →  forecast['NHITS-hi-80'][0]")
print("  'What is the 90th percentile of Wednesday demand?' → forecast['NHITS-hi-90'][2]")
print()
print("MQLoss CANNOT answer:")
print("  'What is the 80th percentile of weekly total demand?'")
print("  'What is the probability that total demand exceeds 2000 baguettes?'")
print("  'What inventory level guarantees 95% service level for the whole week?'")
print()
print("These require the joint distribution — i.e., sample paths.")
print("Module 3: Generating sample paths with NeuralForecast ConformalIntervals.")
```

---


## Practice Questions

**Question 1 — Conceptual:** Based on the concepts in this guide, explain in your own words why the core technique matters and when you would choose it over alternatives.

**Question 2 — Application:** Sketch out how you would apply the main concept from this guide to a real-world dataset or problem you have encountered. What would you need to watch out for?


## Connections


<div class="callout-info">

<strong>Info:</strong> This section maps how this guide connects to the broader course. Use these links to navigate related material.

</div>

### Builds on
- Guide 01: Marginal vs joint distributions — the conceptual foundation
- Module 1: NHITS point forecasting — the same model architecture

### Leads to
- **Notebook 02:** MQLoss deep dive — training different level settings, calibration checks
- **Module 3:** Sample paths — fixing the multi-step problem using conformal prediction

---

## Key Takeaways

1. **Pinball loss** is an asymmetric loss that penalizes over/underforecasting according to the desired quantile level.
2. **MQLoss** trains all requested quantile levels simultaneously, sharing a single model representation.
3. **Output column convention:** `{model}-lo-{level}` and `{model}-hi-{level}`, where level is the coverage percentage.
4. **More levels** produce a smoother fan chart but cost more computation — `level=[80, 90]` is a good default.
5. **MQLoss vs DistributionLoss:** MQLoss is nonparametric and robust; DistributionLoss is more appropriate when you know the data's distributional family.
6. **Marginal, not joint:** MQLoss produces per-step quantiles that are correct individually but cannot describe multi-period behavior.


<div class="flow">
<div class="flow-step mint">1. Pinball loss</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">2. MQLoss</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">3. Output column convention:</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. More levels</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step rose">5. MQLoss vs DistributionLoss:</div>
</div>


---

## Cross-References

<a class="link-card" href="./02_training_quantile_models.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Interactive slide deck covering the key concepts with visual examples.</div>
</a>

<a class="link-card" href="../notebooks/01_quantiles_not_enough.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises and real data.</div>
</a>
