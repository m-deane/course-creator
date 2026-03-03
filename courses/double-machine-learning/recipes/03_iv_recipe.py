"""
Instrumental Variables with DML in 20 Lines
Problem: Estimate causal effect when treatment is endogenous (unobserved confounders)
"""

import numpy as np
import pandas as pd
from doubleml import DoubleMLData, DoubleMLPLIV
from sklearn.ensemble import RandomForestRegressor

# Simulate (replace with your data)
np.random.seed(42)
n, p = 3000, 20
X = np.random.randn(n, p)
Z = 0.5 * X[:, 0] + np.random.randn(n)  # Instrument
U = np.random.randn(n)  # Unobserved confounder
D = 0.6 * Z + 0.3 * X[:, 1] + 0.4 * U + np.random.randn(n) * 0.3
Y = 0.5 * D + 0.5 * X[:, 0] + 0.5 * U + np.random.randn(n) * 0.3

# Prepare data
cols = [f"X{i}" for i in range(p)]
df = pd.DataFrame(X, columns=cols)
df["Y"], df["D"], df["Z"] = Y, D, Z
dml_data = DoubleMLData(df, y_col="Y", d_cols="D", x_cols=cols, z_cols="Z")

# Fit PLIV
pliv = DoubleMLPLIV(dml_data,
                     ml_l=RandomForestRegressor(200, random_state=42),
                     ml_m=RandomForestRegressor(200, random_state=42),
                     ml_r=RandomForestRegressor(200, random_state=42),
                     n_folds=5)
pliv.fit()
print(pliv.summary)
