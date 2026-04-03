"""
Recipe: Explain a Neural Forecast with Feature Attributions

Copy-paste pattern for producing explainability reports using
neuralforecast's .explain() API backed by captum.

Three methods available:
- IntegratedGradients: Best balance of speed and accuracy (recommended)
- InputXGradient: Fastest, but biased with ReLU activations
- ShapleyValueSampling: Most theoretically sound, but slowest
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MSE, MAE

# ---------------------------------------------------------------------------
# Step 1: Train model with exogenous features
# ---------------------------------------------------------------------------
# Replace with your own data
# df must have columns: unique_id, ds, y, plus any exogenous features

# models = [NHITS(
#     h=28,
#     input_size=56,
#     futr_exog_list=["feature_1", "feature_2"],  # future-known features
#     max_steps=1000,
#     loss=MSE(),
#     valid_loss=MAE(),
# )]
# nf = NeuralForecast(models=models, freq="D")
# nf.fit(df=train_df, val_size=28)

# ---------------------------------------------------------------------------
# Step 2: Generate attributions
# ---------------------------------------------------------------------------
# futr_df = ...  # DataFrame with future exogenous feature values

# fcsts_df, explanations = nf.explain(
#     futr_df=futr_df,
#     explainer="IntegratedGradients"
# )

# ---------------------------------------------------------------------------
# Step 3: Parse attribution tensors
# ---------------------------------------------------------------------------
# insample_attr = explanations["insample"]
# Shape: [batch, horizon, series, output, input_size, 2]
# The last dim [0] = y values, [1] = attributions

# futr_exog_attr = explanations["futr_exog"]
# Shape: [batch, horizon, series, output, input_size+horizon, features]

# ---------------------------------------------------------------------------
# Step 4: Visualize
# ---------------------------------------------------------------------------
def plot_lag_importance(insample_attr: np.ndarray, horizon_step: int = 0):
    """Plot the importance of each lag for a specific forecast step."""
    # Extract attributions for the first series, first output
    attr = insample_attr[0, horizon_step, 0, 0, :, 1]  # shape: (input_size,)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(attr)), np.abs(attr), color="steelblue")
    ax.set_xlabel("Lag Position (most recent = rightmost)")
    ax.set_ylabel("|Attribution|")
    ax.set_title(f"Lag Importance for Forecast Step {horizon_step + 1}")
    plt.tight_layout()
    plt.show()


# plot_lag_importance(insample_attr, horizon_step=0)
