# Interpreting Attribution Tensors

> **Reading time:** ~13 min | **Module:** 4 — Explainability | **Prerequisites:** Module 1

## In Brief

Running `.explain()` returns an `explanations` dictionary with attribution tensors. Raw tensors are not interpretable until you know what each dimension means. This guide maps every dimension, shows how to slice the tensors into business-relevant views, and covers three visualization techniques: heatmaps, waterfall plots, and bar charts.

Start by parsing the dictionary:


The following implementation builds on the approach above:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python

# After fitting and calling .explain()
fcsts_df, explanations = nf.explain(futr_df=futr_df, explainer="IntegratedGradients")

insample      = explanations["insample"]           # past lag attributions
futr_exog     = explanations["futr_exog"]          # exogenous feature attributions
baseline_pred = explanations["baseline_predictions"]  # baseline model output

print("insample shape:      ", insample.shape)
print("futr_exog shape:     ", futr_exog.shape)
print("baseline_pred shape: ", baseline_pred.shape)

# insample shape:       (1, 28, 1, 1, 56, 2)

# futr_exog shape:      (1, 28, 1, 1, 84, 2)

# baseline_pred shape:  (1, 28, 1, 1)
```

</div>
</div>

<div class="callout-key">

<strong>Key Concept:</strong> Running `.explain()` returns an `explanations` dictionary with attribution tensors. Raw tensors are not interpretable until you know what each dimension means.

</div>


---

## 1. The `insample` Tensor: Past Lag Attributions

### Shape Reference

<div class="callout-insight">

<strong>Insight:</strong> ### Shape Reference

`insample` has shape `[batch, horizon, series, output, input_size, 2]`.

</div>


`insample` has shape `[batch, horizon, series, output, input_size, 2]`.

| Dimension | Size | Meaning |
|---|---|---|
| `batch` | 1 | Number of prediction batches (usually 1 for single forecast) |
| `horizon` | $h$ | Forecast step (1 to $h$) |
| `series` | $S$ | Number of time series (1 for a single series) |
| `output` | 1 | Output dimension (1 for point forecast) |
| `input_size` | $L$ | Number of past lags fed to the model |
| `2` | 2 | `[value, attribution]` pair |

The last dimension always contains two values: the actual input value at that lag, and the attribution score assigned to that lag.

### Extracting Lag Attributions


The following implementation builds on the approach above:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np

# Select: batch=0, all horizon steps, series=0, output=0, all lags, attribution column (index 1)
lag_attributions = insample[0, :, 0, 0, :, 1]

# Shape: (horizon, input_size) = (28, 56)

# lag_attributions[h, l] = attribution of lag l for forecast step h

lag_values = insample[0, :, 0, 0, :, 0]

# Shape: (28, 56)

# lag_values[h, l] = actual value of lag l (what the model saw)

# Most important lag for each forecast step
most_important_lag = np.argmax(np.abs(lag_attributions), axis=1)
print("Most important lag per forecast step:", most_important_lag)

# Typical output: [0 0 0 0 1 0 ...] — lag 0 (most recent) is usually dominant
```

</div>
</div>

### What "Lag 0" Means

Lag 0 is the most recent observation before the forecast origin. Lag 1 is one period before that. In a model with `input_size=56` for a daily series, lag 55 is 55 days before the forecast start.

**Expected pattern for time series:** lag 0 (the most recent observation) typically receives the largest attribution. Attribution should decay as lags grow older. A model where distant lags consistently outrank recent lags is a signal worth investigating.

---

## 2. The `futr_exog` Tensor: Exogenous Feature Attributions

### Shape Reference

<div class="callout-key">

<strong>Key Point:</strong> ### Shape Reference

`futr_exog` has shape `[batch, horizon, series, output, input_size+horizon, n_features]`.

</div>


`futr_exog` has shape `[batch, horizon, series, output, input_size+horizon, n_features]`.

| Dimension | Size | Meaning |
|---|---|---|
| `batch` | 1 | Prediction batch |
| `horizon` | $h$ | Forecast step |
| `series` | $S$ | Number of time series |
| `output` | 1 | Output dimension |
| `input_size+horizon` | $L+h$ | Lookback window plus forecast horizon (full context window for future covariates) |
| `n_features` | $F$ | Number of exogenous features |

Future exogenous features span the full `input_size + horizon` window because the model can see past values of the feature (in the lookback) and future values (in the forecast horizon). The `published` column is known for past and future dates; same for `is_holiday`.

### Extracting Feature Attributions


The following implementation builds on the approach above:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python

# Feature index mapping (matches futr_exog_list order)
FEATURE_NAMES = ["published", "is_holiday"]
PUBLISHED_IDX = 0
IS_HOLIDAY_IDX = 1

# Select all forecast steps, all positions, specific feature
published_attr = futr_exog[0, :, 0, 0, :, PUBLISHED_IDX]
holiday_attr   = futr_exog[0, :, 0, 0, :, IS_HOLIDAY_IDX]

# Shape of each: (horizon, input_size+horizon) = (28, 84)

# Attribution for the published feature at forecast positions (last h positions)

# input_size = 56, horizon = 28, so positions 56..83 are the forecast window
input_size = 56
published_future_attr = published_attr[:, input_size:]

# Shape: (28, 28) — forecast_step × forecast_position

# Total attribution per feature, summed over positions and steps
published_total = published_attr.sum()
holiday_total   = holiday_attr.sum()
print(f"published total attribution: {published_total:.1f}")
print(f"is_holiday total attribution: {holiday_total:.1f}")
```

</div>
</div>

---

## 3. Visualization Technique 1: Heatmap of Lag Importances

A heatmap with forecast steps on the y-axis and lag positions on the x-axis reveals how the model's attention to historical data changes across the horizon.

<div class="callout-info">

<strong>Info:</strong> A heatmap with forecast steps on the y-axis and lag positions on the x-axis reveals how the model's attention to historical data changes across the horizon.

</div>


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_lag_attribution_heatmap(insample, title="Lag Attribution Heatmap"):
    """
    Visualize insample attributions as a heatmap.
    
    Parameters
    ----------
    insample : np.ndarray
        Shape [batch, horizon, series, output, input_size, 2]
    title : str
    """
    # Extract attribution scores: shape (horizon, input_size)
    attr = insample[0, :, 0, 0, :, 1]
    horizon, input_size = attr.shape
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    im = ax.imshow(
        attr,
        aspect="auto",
        cmap="RdBu_r",
        interpolation="nearest"
    )
    plt.colorbar(im, ax=ax, label="Attribution Score")
    
    ax.set_xlabel("Lag Position (0 = most recent)", fontsize=12)
    ax.set_ylabel("Forecast Step", fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Mark lag 0 and every 7th lag
    ax.set_xticks([0, 7, 14, 21, 28, 35, 42, 49, 55])
    ax.set_xticklabels(["lag 0", "lag 7", "lag 14", "lag 21",
                         "lag 28", "lag 35", "lag 42", "lag 49", "lag 55"])
    
    plt.tight_layout()
    return fig

fig = plot_lag_attribution_heatmap(insample, "NHITS Lag Attribution: Blog Traffic")
plt.savefig("../resources/lag_attribution_heatmap.png", dpi=150)
plt.show()
```

</div>
</div>

**Reading the heatmap:** dark red cells indicate lags that push the forecast up; dark blue cells push it down. A column of high-magnitude cells at lag 0 confirms the model relies heavily on recent history, which is expected and healthy for a daily series.

---

## 4. Visualization Technique 2: Waterfall Plot with SHAP

A waterfall plot shows how each feature contribution adds or subtracts from the baseline prediction to arrive at the final forecast. SHAP's built-in waterfall is the clearest way to communicate this to stakeholders.

<div class="callout-warning">

<strong>Warning:</strong> A waterfall plot shows how each feature contribution adds or subtracts from the baseline prediction to arrive at the final forecast.

</div>


```python
import shap

def build_shap_explanation(futr_exog, baseline_pred, feature_names, horizon_step=0):
    """
    Build a shap.Explanation object from NeuralForecast attribution tensors.
    
    Parameters
    ----------
    futr_exog : np.ndarray
        Shape [batch, horizon, series, output, input_size+horizon, n_features]
    baseline_pred : np.ndarray
        Shape [batch, horizon, series, output]
    feature_names : list of str
    horizon_step : int
        Which forecast step to visualize (0 = first step)
    
    Returns
    -------
    shap.Explanation
    """
    n_features = len(feature_names)
    
    # Sum attributions over position dimension for each feature
    # futr_exog[0, horizon_step, 0, 0, :, :] → shape (input_size+horizon, n_features)
    position_attr = futr_exog[0, horizon_step, 0, 0, :, :]  # (input_size+horizon, n_features)
    feature_attr = position_attr.sum(axis=0)                 # (n_features,)
    
    base_value = float(baseline_pred[0, horizon_step, 0, 0])
    
    explanation = shap.Explanation(
        values=feature_attr,
        base_values=base_value,
        feature_names=feature_names
    )
    return explanation

# Build explanation for forecast step 0
explanation = build_shap_explanation(
    futr_exog, baseline_pred,
    feature_names=["published", "is_holiday"],
    horizon_step=0
)

# Waterfall plot
shap.plots.waterfall(explanation, show=True)
```

**Reading the waterfall:** each bar shows one feature's contribution. Bars extending right (positive) increase the forecast above the baseline. Bars extending left (negative) decrease it. The final value is baseline + all contributions = forecast.

For blog traffic: if publishing an article (`published=1`) adds a large positive bar, it means the model learned that publication events drive traffic spikes — business intuition validated.

---

## 5. Visualization Technique 3: Bar Chart of Feature Importance

A summary bar chart aggregates attributions across all forecast steps and shows net impact per feature.

<div class="callout-insight">

<strong>Insight:</strong> A summary bar chart aggregates attributions across all forecast steps and shows net impact per feature.

</div>


```python
import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importance_bar(futr_exog, feature_names, title="Feature Attribution Summary"):
    """
    Bar chart of mean absolute attribution per feature, aggregated across horizon.
    
    Parameters
    ----------
    futr_exog : np.ndarray
        Shape [batch, horizon, series, output, input_size+horizon, n_features]
    feature_names : list of str
    title : str
    """
    # Sum over position dimension, then mean absolute value over horizon
    # Shape: (horizon, n_features)
    attr_by_horizon = futr_exog[0, :, 0, 0, :, :].sum(axis=1)
    mean_abs_attr = np.abs(attr_by_horizon).mean(axis=0)  # (n_features,)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bars = ax.barh(feature_names, mean_abs_attr, color=["#2196F3", "#FF9800"])
    
    ax.set_xlabel("Mean Absolute Attribution", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis="x")
    
    # Annotate bars
    for bar, val in zip(bars, mean_abs_attr):
        ax.text(
            val + 0.01 * mean_abs_attr.max(),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}",
            va="center",
            fontsize=11
        )
    
    plt.tight_layout()
    return fig

fig = plot_feature_importance_bar(
    futr_exog,
    feature_names=["published", "is_holiday"],
    title="Exogenous Feature Importance: Blog Traffic NHITS"
)
plt.savefig("../resources/feature_importance_bar.png", dpi=150)
plt.show()
```

---

## 6. Validating Model Behavior: Does It Make Business Sense?

Attributions are only useful if you check them against domain knowledge. Run through this checklist after generating attributions:

**Lag attribution checks:**
- Does lag 0 dominate? For most time series it should. If lag 30 dominates, investigate.
- Is there a weekly pattern? For a blog with 7-day seasonality, lags 7, 14, 21 should show elevated importance.
- Are attributions stable across forecast steps? Instability signals a poorly trained model.

**Exogenous feature checks:**
- Does `published` get positive attribution? Publishing drives traffic — positive attribution is expected.
- Does `is_holiday` get near-zero or negative attribution? Blog readership often drops on holidays.
- Are the magnitudes economically meaningful? If `published` adds 610 visitors to a baseline of 500, that is a 120% lift — high but plausible for a new article.

**Completeness check (Integrated Gradients only):**

```python

# Verify: insample attributions + futr_exog attributions ≈ prediction - baseline

# (Only holds approximately because insample covers the endogenous component)
forecast_step = 0
predicted_value = float(fcsts_df["NHITS"].iloc[forecast_step])
baseline_value  = float(baseline_pred[0, forecast_step, 0, 0])

total_insample_attr = insample[0, forecast_step, 0, 0, :, 1].sum()
total_exog_attr     = futr_exog[0, forecast_step, 0, 0, :, :].sum()

reconstructed = baseline_value + total_insample_attr + total_exog_attr
print(f"Predicted:     {predicted_value:.1f}")
print(f"Reconstructed: {reconstructed:.1f}")
print(f"Difference:    {abs(predicted_value - reconstructed):.2f}")

# Should be near zero for Integrated Gradients
```

---

## 7. Translating Attributions into Stakeholder Language

Raw attribution numbers are meaningless to a content manager or commodity trader. The translation step is where the analysis becomes useful.

**Template for a business narrative:**

> "Over the 28-day forecast horizon, the model attributes approximately [X] additional daily visitors to publishing activity (`published=1`). This is equivalent to a [Y]% lift over the baseline of [Z] visitors per day on non-publishing days. Holiday effects contribute a smaller and [positive/negative] [W] visitors, suggesting that [interpretation]."

For the blog traffic dataset:
- Baseline prediction: ~500 visitors/day
- `published` total attribution: ~610 additional visitors on days an article is published
- `is_holiday` attribution: ~15 fewer visitors, consistent with reduced readership on holiday days

This narrative can be included in a dashboard, a model card, or a presentation to model risk management.

---

## 8. Common Pitfalls

**Pitfall 1: Interpreting attribution sign as causal direction.** A positive attribution for `published` means publishing is associated with higher forecasts in this dataset. It does not prove causality. Confounders (e.g., articles are published on weekdays, which have higher traffic anyway) may be responsible.

**Pitfall 2: Ignoring the baseline.** The completeness guarantee is `sum(attributions) = f(x) - f(x')`. If your baseline is unusual, the attributions reflect deviation from that unusual baseline, not from "no signal."

**Pitfall 3: Averaging attributions across series.** For a multi-series model, averaging attributions across all series will cancel out heterogeneous patterns. Always examine single-series attributions before aggregating.

**Pitfall 4: Confusing insample and futr_exog attribution magnitudes.** The insample tensor captures the endogenous signal (past lags), which typically dominates. Do not compare raw magnitudes between insample and futr_exog without normalizing for the difference in tensor sizes.

---

## What's Next

- `notebooks/01_explain_api.ipynb` — Run `.explain()` end-to-end on synthetic blog traffic data, extract lag attributions, visualize
- `notebooks/02_attribution_visualization.ipynb` — Compare all three methods, generate waterfall and heatmap, write business narrative
- `exercises/01_explainability_exercises.py` — Self-check: verify dict keys, extract shapes, find the dominant lag


## Practice Questions

**Question 1 — Conceptual:** Based on the concepts in this guide, explain in your own words why the core technique matters and when you would choose it over alternatives.

**Question 2 — Application:** Sketch out how you would apply the main concept from this guide to a real-world dataset or problem you have encountered. What would you need to watch out for?


---

## Cross-References

<a class="link-card" href="./02_interpreting_attributions.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Interactive slide deck covering the key concepts with visual examples.</div>
</a>

<a class="link-card" href="../notebooks/01_explain_api.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises and real data.</div>
</a>
