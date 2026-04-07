"""
Recipe: Explain a Neural Forecast with Feature Attributions

IMPORTANT: NeuralForecast does not natively support model explainability.
There is no .explain() method in the NeuralForecast API.

For interpretability, use Captum with the underlying PyTorch models,
or use inherently interpretable models like NHITS which provide
basis function decompositions.

Three Captum methods available:
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
# Step 2: Generate attributions using Captum
# ---------------------------------------------------------------------------
# NeuralForecast does NOT have a .explain() method.
# Use Captum directly with the underlying PyTorch model.

# from captum.attr import IntegratedGradients
#
# pytorch_model = nf.models[0]  # Extract the trained PyTorch model
# ig = IntegratedGradients(pytorch_model)
#
# # You need to prepare model-specific input tensors.
# # See Captum docs for full usage: https://captum.ai/
# # attributions = ig.attribute(input_tensor, baselines=baseline_tensor)

# ---------------------------------------------------------------------------
# Step 3: Visualize attributions
# ---------------------------------------------------------------------------
def plot_feature_importance(attributions: np.ndarray, feature_names: list[str]):
    """Plot the importance of each feature based on Captum attributions."""
    # Average absolute attributions across samples
    mean_attr = np.abs(attributions).mean(axis=0)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(mean_attr)), mean_attr, color="steelblue")
    if feature_names:
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_ylabel("|Attribution|")
    ax.set_title("Feature Importance (Captum Attributions)")
    plt.tight_layout()
    plt.show()


# plot_feature_importance(attributions, feature_names=["feature_1", "feature_2"])
