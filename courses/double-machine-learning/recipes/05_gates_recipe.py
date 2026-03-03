"""
GATES Visualization in 25 Lines
Problem: Visualise treatment effect heterogeneity with sorted group effects
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor

# Simulate (replace with your data)
np.random.seed(42)
n, p = 5000, 20
X = np.random.randn(n, p)
D = 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.randn(n) * 0.5
tau = 0.5 + 0.8 * X[:, 0]  # Heterogeneous effect
Y = tau * D + X[:, 0] + np.random.randn(n) * 0.5

# Estimate CATE
cf = CausalForestDML(
    model_y=GradientBoostingRegressor(200, max_depth=5, random_state=42),
    model_t=GradientBoostingRegressor(200, max_depth=5, random_state=42),
    n_estimators=200, random_state=42)
cf.fit(Y, D, X=X)
cate = cf.effect(X)

# GATES: sort by estimated CATE into 5 groups
sorted_idx = np.argsort(cate)
n_groups = 5
gates = []
for g in range(n_groups):
    start = g * (n // n_groups)
    end = (g + 1) * (n // n_groups) if g < n_groups - 1 else n
    idx = sorted_idx[start:end]
    gates.append({"quintile": g + 1, "mean_cate": np.mean(cate[idx]),
                   "true_cate": np.mean(tau[idx])})

gates_df = pd.DataFrame(gates)

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
w = 0.3
ax.bar(gates_df["quintile"] - w/2, gates_df["true_cate"], w,
       label="True", color="crimson", alpha=0.7)
ax.bar(gates_df["quintile"] + w/2, gates_df["mean_cate"], w,
       label="Estimated", color="steelblue", alpha=0.7)
ax.set_xlabel("CATE Quintile")
ax.set_ylabel("Group Average Treatment Effect")
ax.set_title("GATES: Sorted Group Effects")
ax.legend()
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.show()
print(gates_df.to_string(index=False))
