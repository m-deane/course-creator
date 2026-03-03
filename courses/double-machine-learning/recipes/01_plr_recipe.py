"""
Partially Linear Regression in 20 Lines
Problem: Estimate a causal effect with many controls using doubleml
"""

import numpy as np
import pandas as pd
from doubleml import DoubleMLData, DoubleMLPLR
from sklearn.ensemble import GradientBoostingRegressor

# Simulate (replace with your data)
np.random.seed(42)
n, p = 2000, 50
X = np.random.randn(n, p)
D = 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.randn(n) * 0.5
Y = 0.5 * D + X[:, 0] + 0.3 * X[:, 2] + np.random.randn(n)

# Prepare data
cols = [f"X{i}" for i in range(p)]
df = pd.DataFrame(X, columns=cols)
df["Y"], df["D"] = Y, D
dml_data = DoubleMLData(df, y_col="Y", d_cols="D", x_cols=cols)

# Fit and print
dml = DoubleMLPLR(dml_data,
                   ml_l=GradientBoostingRegressor(200, max_depth=5, random_state=42),
                   ml_m=GradientBoostingRegressor(200, max_depth=5, random_state=42),
                   n_folds=5)
dml.fit()
print(dml.summary)
