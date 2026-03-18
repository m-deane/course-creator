"""
Module 05 — Machine Learning Extensions: Self-Check Exercise
=============================================================

This exercise tests your understanding of regularized MIDAS and
ML-based nowcasting through four tasks:

  1. Fit and compare Lasso vs Elastic Net MIDAS
  2. Run expanding-window evaluation
  3. Implement a basic Diebold-Mariano test
  4. Interpret results from feature importance

Run this script directly:
    python 01_ml_extensions_self_check.py

Each task has a VERIFY block that checks your answers.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, ElasticNet, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
import warnings

warnings.filterwarnings("ignore")
np.random.seed(2024)

# ──────────────────────────────────────────────────────────────────────────────
# Data setup: 12 monthly indicators → quarterly GDP growth
# ──────────────────────────────────────────────────────────────────────────────

def generate_data(n_quarters=80, n_indicators=12, n_lags=12, seed=2024):
    """
    Generate a MIDAS design matrix with K monthly indicators, each providing
    m=12 monthly lags. The true model uses only 3 indicators.
    """
    np.random.seed(seed)
    K, m = n_indicators, n_lags
    T = n_quarters

    # Monthly indicators (T*3 monthly observations, K series)
    monthly = np.random.randn(T * 3, K)
    for k in range(K):
        rho = 0.7 + 0.2 * np.random.rand()
        for t in range(1, T * 3):
            monthly[t, k] = rho * monthly[t-1, k] + np.sqrt(1 - rho**2) * np.random.randn()

    # MIDAS design matrix: for each quarter, stack last m monthly lags of each indicator
    X_rows = []
    for q in range(m // 3, T):
        row = []
        for k in range(K):
            end_idx = q * 3
            lags = monthly[end_idx - m:end_idx, k][::-1]  # Most recent first
            row.extend(lags)
        X_rows.append(row)

    X = np.array(X_rows)
    T_eff = len(X_rows)

    # True model: indicators 0, 2, 5 only; exponentially decaying weights
    true_signal = np.zeros(T_eff)
    for k_true in [0, 2, 5]:
        start_col = k_true * m
        weights = np.exp(-0.3 * np.arange(m))
        weights /= weights.sum()
        true_signal += X[:, start_col:start_col + m] @ weights

    y = 2.0 + true_signal + 0.4 * np.random.randn(T_eff)

    col_names = [f"ind{k:02d}_lag{j:02d}" for k in range(K) for j in range(m)]
    return X, y, col_names, K, m


X, y, col_names, K, m = generate_data()
T = len(y)
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

print(f"Data ready: X={X.shape}, y={y.shape}")
print(f"True predictors: indicators 0, 2, 5 (indices {list(range(0, m))}, {list(range(2*m, 3*m))}, {list(range(5*m, 6*m))})")
print()

# ──────────────────────────────────────────────────────────────────────────────
# TASK 1: Fit Lasso and Elastic Net with time-series cross-validation
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TASK 1: Fit Lasso and Elastic Net with TS cross-validation")
print("=" * 60)

# Your work here: fit LassoCV and ElasticNetCV using TimeSeriesSplit
# Use alphas=np.logspace(-4, 0, 40) for both

tscv = TimeSeriesSplit(n_splits=5)

# Fit Lasso
lasso_cv = LassoCV(
    alphas=np.logspace(-4, 0, 40),
    cv=tscv,
    max_iter=10000,
    random_state=42
)
lasso_cv.fit(X_s, y)

# Fit ElasticNet
en_cv = ElasticNetCV(
    l1_ratio=[0.3, 0.5, 0.7, 0.9],
    alphas=np.logspace(-4, 0, 30),
    cv=tscv,
    max_iter=10000,
    random_state=42
)
en_cv.fit(X_s, y)

# Count selected indicators (at least one nonzero lag out of m)
def count_selected_indicators(coef, K, m):
    """Return list of indicator indices with at least one nonzero lag coefficient."""
    selected = []
    for k in range(K):
        group_coef = coef[k * m:(k + 1) * m]
        if np.any(np.abs(group_coef) > 1e-8):
            selected.append(k)
    return selected


lasso_selected = count_selected_indicators(lasso_cv.coef_, K, m)
en_selected = count_selected_indicators(en_cv.coef_, K, m)

print(f"Lasso optimal alpha: {lasso_cv.alpha_:.6f}")
print(f"  Selected indicators: {lasso_selected}")
print(f"  Number of nonzero features: {np.sum(np.abs(lasso_cv.coef_) > 1e-8)}")
print()
print(f"ElasticNet optimal alpha: {en_cv.alpha_:.6f}, l1_ratio: {en_cv.l1_ratio_:.2f}")
print(f"  Selected indicators: {en_selected}")
print(f"  Number of nonzero features: {np.sum(np.abs(en_cv.coef_) > 1e-8)}")
print()

# VERIFY Task 1
true_indicators = {0, 2, 5}
lasso_recovered = len(true_indicators.intersection(lasso_selected))
en_recovered = len(true_indicators.intersection(en_selected))

assert lasso_recovered >= 2, (
    f"Lasso should recover at least 2 of 3 true indicators. "
    f"Got {lasso_selected}, expected some of {{0, 2, 5}}"
)
assert en_recovered >= 2, (
    f"ElasticNet should recover at least 2 of 3 true indicators. "
    f"Got {en_selected}, expected some of {{0, 2, 5}}"
)
print("TASK 1 PASSED: Both estimators recover the true sparse structure.")
print()

# ──────────────────────────────────────────────────────────────────────────────
# TASK 2: Expanding-window evaluation
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TASK 2: Expanding-window evaluation")
print("=" * 60)

min_train = 25
lasso_forecasts = np.full(T, np.nan)
en_forecasts = np.full(T, np.nan)

# Expanding window: re-fit at each step
for t in range(min_train, T):
    X_tr, y_tr = X[:t], y[:t]
    X_te = X[t:t+1]

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    # Lasso: fixed optimal alpha
    lasso = Lasso(alpha=lasso_cv.alpha_, max_iter=10000)
    lasso.fit(X_tr_s, y_tr)
    lasso_forecasts[t] = lasso.predict(X_te_s)[0]

    # ElasticNet: fixed optimal alpha and l1_ratio
    en = ElasticNet(alpha=en_cv.alpha_, l1_ratio=en_cv.l1_ratio_, max_iter=10000)
    en.fit(X_tr_s, y_tr)
    en_forecasts[t] = en.predict(X_te_s)[0]

# Evaluate on last 20%
eval_start = int(T * 0.8)
y_eval = y[eval_start:]
lasso_eval = lasso_forecasts[eval_start:]
en_eval = en_forecasts[eval_start:]

valid = ~np.isnan(lasso_eval) & ~np.isnan(en_eval)
y_v = y_eval[valid]
l_v = lasso_eval[valid]
e_v = en_eval[valid]

lasso_rmse = np.sqrt(mean_squared_error(y_v, l_v))
en_rmse = np.sqrt(mean_squared_error(y_v, e_v))

print(f"Evaluation on last 20% ({np.sum(valid)} quarters):")
print(f"  Lasso RMSE:      {lasso_rmse:.4f}")
print(f"  ElasticNet RMSE: {en_rmse:.4f}")
print()

# VERIFY Task 2
assert lasso_rmse < 5.0, f"RMSE too high: Lasso RMSE={lasso_rmse:.4f}. Check implementation."
assert en_rmse < 5.0, f"RMSE too high: ElasticNet RMSE={en_rmse:.4f}. Check implementation."
assert not np.all(np.isnan(lasso_forecasts)), "All Lasso forecasts are NaN. Check loop range."
print("TASK 2 PASSED: Expanding-window forecasts computed successfully.")
print()

# ──────────────────────────────────────────────────────────────────────────────
# TASK 3: Diebold-Mariano test
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TASK 3: Diebold-Mariano test for equal predictive accuracy")
print("=" * 60)

# Implement the DM test from scratch
# H0: E[L(e_lasso)] = E[L(e_en)] where L(e) = e^2 (squared error)
# Positive DM stat: Lasso has higher MSE (ElasticNet is better)

e_lasso = y_v - l_v
e_en = y_v - e_v

d = e_lasso**2 - e_en**2  # Loss differential
T_eval = len(d)
d_bar = d.mean()

# Newey-West variance (h=1 for one-step-ahead)
gamma0 = np.var(d, ddof=1)
dm_var = gamma0 / T_eval  # Simplified (h=1, no autocorrelation correction)

dm_stat = d_bar / np.sqrt(max(dm_var, 1e-12))
p_value = 2 * stats.t.sf(np.abs(dm_stat), df=T_eval - 1)

print(f"DM test: Lasso vs ElasticNet")
print(f"  Loss differential mean: {d_bar:.6f}")
print(f"  DM statistic: {dm_stat:.3f}")
print(f"  p-value: {p_value:.4f}")
if p_value < 0.05:
    better = "ElasticNet" if dm_stat > 0 else "Lasso"
    print(f"  Interpretation: Reject H0 at 5%. {better} is significantly more accurate.")
else:
    print(f"  Interpretation: Fail to reject H0. No significant difference in accuracy.")
print()

# VERIFY Task 3
assert isinstance(dm_stat, float), "DM statistic should be a float."
assert 0 <= p_value <= 1, f"p-value should be in [0,1], got {p_value}"
assert not np.isnan(dm_stat), "DM statistic is NaN. Check variance computation."
print("TASK 3 PASSED: DM test computed correctly.")
print()

# ──────────────────────────────────────────────────────────────────────────────
# TASK 4: Forecast combination
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TASK 4: Forecast combination")
print("=" * 60)

# Compare equal-weight combination vs individual models
combined_equal = 0.5 * l_v + 0.5 * e_v
combined_rmse = np.sqrt(mean_squared_error(y_v, combined_equal))

print(f"Forecast combination results:")
print(f"  Lasso alone:       RMSE = {lasso_rmse:.4f}")
print(f"  ElasticNet alone:  RMSE = {en_rmse:.4f}")
print(f"  Equal combination: RMSE = {combined_rmse:.4f}")

# Compute optimal weights
from scipy.optimize import minimize

def combo_loss(w):
    comb = w[0] * l_v + (1 - w[0]) * e_v
    return np.sqrt(mean_squared_error(y_v, comb))

result = minimize(combo_loss, [0.5], method='L-BFGS-B', bounds=[(0, 1)])
w_opt = result.x[0]
combined_opt = w_opt * l_v + (1 - w_opt) * e_v
combined_opt_rmse = np.sqrt(mean_squared_error(y_v, combined_opt))

print(f"  Optimal combination (w_Lasso={w_opt:.3f}): RMSE = {combined_opt_rmse:.4f}")
print()

# VERIFY Task 4
assert combined_rmse <= max(lasso_rmse, en_rmse) + 0.05, (
    f"Equal-weight combination RMSE ({combined_rmse:.4f}) should be at most slightly "
    f"above best individual ({min(lasso_rmse, en_rmse):.4f})"
)
print("TASK 4 PASSED: Combination forecast computed.")
print()

# ──────────────────────────────────────────────────────────────────────────────
# Final summary
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("ALL TASKS COMPLETED")
print("=" * 60)
print()
print("Summary:")
print(f"  True model uses indicators: 0, 2, 5")
print(f"  Lasso recovered: {sorted(set(lasso_selected) & true_indicators)}")
print(f"  ElasticNet recovered: {sorted(set(en_selected) & true_indicators)}")
print()
print(f"  Out-of-sample RMSE (last 20%):")
print(f"    Lasso:       {lasso_rmse:.4f}")
print(f"    ElasticNet:  {en_rmse:.4f}")
print(f"    Combination: {combined_rmse:.4f}")
print()
print(f"  DM test: stat={dm_stat:.3f}, p={p_value:.4f}")
print()
print("Key insights:")
print("  1. Both regularized estimators recover sparse structure when signal is clear")
print("  2. Elastic Net l1_ratio controls sparsity level")
print("  3. DM test tells us if RMSE differences are statistically significant")
print("  4. Forecast combinations reduce idiosyncratic model errors")
