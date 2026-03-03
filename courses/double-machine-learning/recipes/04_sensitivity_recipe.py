"""
Sensitivity Analysis for DML Estimates in 25 Lines
Problem: Check if your DML estimate is robust to nuisance model choice
"""

import numpy as np
import pandas as pd
from doubleml import DoubleMLData, DoubleMLPLR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV

# Simulate (replace with your data)
np.random.seed(42)
n, p = 2000, 50
X = np.random.randn(n, p)
D = 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.randn(n) * 0.5
Y = -0.5 * D + X[:, 0] + 0.3 * X[:, 2] + np.random.randn(n)

cols = [f"X{i}" for i in range(p)]
df = pd.DataFrame(X, columns=cols)
df["Y"], df["D"] = Y, D
dml_data = DoubleMLData(df, y_col="Y", d_cols="D", x_cols=cols)

# Run with multiple nuisance models
models = {
    "Lasso": (LassoCV(cv=3), LassoCV(cv=3)),
    "RF": (RandomForestRegressor(200, random_state=42), RandomForestRegressor(200, random_state=42)),
    "GBM": (GradientBoostingRegressor(200, max_depth=5, random_state=42),
            GradientBoostingRegressor(200, max_depth=5, random_state=42)),
}

results = []
for name, (ml_l, ml_m) in models.items():
    dml = DoubleMLPLR(dml_data, ml_l=ml_l, ml_m=ml_m, n_folds=5)
    dml.fit()
    ci = dml.confint()
    results.append({"model": name, "estimate": dml.coef[0], "se": dml.se[0],
                     "ci_low": ci.iloc[0, 0], "ci_high": ci.iloc[0, 1]})

sa_df = pd.DataFrame(results)
print(sa_df.to_string(index=False))

spread = sa_df["estimate"].max() - sa_df["estimate"].min()
print(f"\nRobust: {'YES' if spread < sa_df['se'].mean() else 'NO'} "
      f"(spread={spread:.4f}, avg SE={sa_df['se'].mean():.4f})")
