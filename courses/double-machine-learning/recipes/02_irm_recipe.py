"""
Interactive Regression Model in 20 Lines
Problem: Estimate ATE/ATTE for a binary treatment with many controls
"""

import numpy as np
import pandas as pd
from doubleml import DoubleMLData, DoubleMLIRM
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Simulate (replace with your data)
np.random.seed(42)
n, p = 3000, 30
X = np.random.randn(n, p)
propensity = 1 / (1 + np.exp(-(0.5 * X[:, 0] + 0.3 * X[:, 1])))
D = np.random.binomial(1, propensity)
Y = 1.5 * D + 0.5 * X[:, 0] + 0.3 * X[:, 2] + np.random.randn(n) * 0.5

# Prepare data
cols = [f"X{i}" for i in range(p)]
df = pd.DataFrame(X, columns=cols)
df["Y"], df["D"] = Y, D
dml_data = DoubleMLData(df, y_col="Y", d_cols="D", x_cols=cols)

# Fit ATE
irm = DoubleMLIRM(dml_data,
                   ml_g=RandomForestRegressor(200, random_state=42),
                   ml_m=RandomForestClassifier(200, random_state=42),
                   score="ATE", n_folds=5, trimming_threshold=0.05)
irm.fit()
print(irm.summary)
