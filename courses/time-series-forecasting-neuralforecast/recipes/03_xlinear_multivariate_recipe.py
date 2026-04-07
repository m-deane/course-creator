"""
Recipe: Multivariate Forecasting with DLinear

Copy-paste pattern for training DLinear on multivariate time series
where cross-variable associations improve forecasts.

DLinear architecture:
1. Embedding layer with global context tokens
2. Time-wise Gating Module (TGM) — temporal patterns
3. Variate-wise Gating Module (VGM) — cross-variable associations
4. Prediction head
"""

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import DLinear
from utilsforecast.losses import mae, mse

# ---------------------------------------------------------------------------
# Step 1: Load multivariate data (ETTm1 example)
# ---------------------------------------------------------------------------
# from datasetsforecast.long_horizon import LongHorizon
# Y_df, _, _ = LongHorizon.load(directory="./data", group="ETTm1")
# n_series = Y_df["unique_id"].nunique()  # 7 for ETTm1

# ---------------------------------------------------------------------------
# Step 2: Train DLinear
# ---------------------------------------------------------------------------
# horizon = 96
# models = [DLinear(
#     h=horizon,
#     input_size=horizon,           # Match input to horizon
#     n_series=n_series,            # Number of variables
#     hidden_size=512,
#     temporal_ff=256,              # Time-wise gating feedforward size
#     channel_ff=n_series * 3,      # Variate-wise gating feedforward size
#     head_dropout=0.5,
#     embed_dropout=0.2,
#     learning_rate=1e-4,
#     batch_size=32,
#     max_steps=2000,
# )]

# nf = NeuralForecast(models=models, freq=freq)

# ---------------------------------------------------------------------------
# Step 3: Cross-validate for honest evaluation
# ---------------------------------------------------------------------------
# val_size = horizon
# test_size = horizon

# preds = nf.cross_validation(
#     df=Y_df,
#     val_size=val_size,
#     test_size=test_size,
# )

# ---------------------------------------------------------------------------
# Step 4: Evaluate
# ---------------------------------------------------------------------------
# mae_score = mae(preds["y"], preds["DLinear"])
# mse_score = mse(preds["y"], preds["DLinear"])
# print(f"MAE: {mae_score:.4f}")
# print(f"MSE: {mse_score:.4f}")
