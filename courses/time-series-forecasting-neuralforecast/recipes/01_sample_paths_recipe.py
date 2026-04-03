"""
Recipe: Generate Sample Paths for Inventory Decisions

Copy-paste pattern for answering multi-period business questions
using sample paths instead of marginal quantiles.

The key insight: "The sum of quantiles is NOT the quantiles of the sum."
"""

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MQLoss

# ---------------------------------------------------------------------------
# Step 1: Prepare data in nixtla format (unique_id, ds, y)
# ---------------------------------------------------------------------------
# Replace with your own data loading
from neuralforecast.utils import AirPassengersDF

df = AirPassengersDF

# ---------------------------------------------------------------------------
# Step 2: Train with probabilistic loss
# ---------------------------------------------------------------------------
horizon = 12
models = [NHITS(
    h=horizon,
    input_size=horizon * 2,
    max_steps=500,
    loss=MQLoss(level=[80, 90]),
    scaler_type="robust",
)]

nf = NeuralForecast(models=models, freq="MS")
nf.fit(df=df)

# ---------------------------------------------------------------------------
# Step 3: Generate sample paths (neuralforecast v3.1.6+)
# ---------------------------------------------------------------------------
# paths = nf.simulate(n_paths=100)  # shape: (n_paths, horizon)

# For older versions, manual Gaussian Copula approach:
forecasts = nf.predict()
print("Marginal forecast columns:", list(forecasts.columns))

# ---------------------------------------------------------------------------
# Step 4: Answer business questions with paths
# ---------------------------------------------------------------------------
# WRONG way (using marginal quantiles):
# weekly_80_wrong = forecasts["NHITS-hi-80"].sum()

# RIGHT way (using sample paths):
# path_totals = paths.sum(axis=1)
# weekly_80_correct = np.quantile(path_totals, 0.8)

# print(f"Wrong (sum of quantiles):    {weekly_80_wrong:.0f}")
# print(f"Correct (quantile of sums):  {weekly_80_correct:.0f}")
