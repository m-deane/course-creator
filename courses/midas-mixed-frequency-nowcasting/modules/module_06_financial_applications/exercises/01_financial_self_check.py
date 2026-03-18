"""
Module 06 — Financial Applications: Self-Check Exercise
=========================================================

Four tasks covering MIDAS-RV, VaR backtesting, and forecast comparison.

  1. Compute Beta weights and verify they sum to 1
  2. Fit MIDAS-RV by NLS and check parameter plausibility
  3. Implement the Kupiec VaR backtest from scratch
  4. Compare QLIKE vs MSE model rankings

Run: python 01_financial_self_check.py
"""

import numpy as np
from scipy.optimize import minimize
from scipy import stats

np.random.seed(2024)

# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data: S&P 500-like daily returns and monthly RV
# ──────────────────────────────────────────────────────────────────────────────
MONTHS = 120  # 10 years
DAYS_PER_MONTH = 22
K = DAYS_PER_MONTH

# Generate GARCH-like daily squared returns
n_days = MONTHS * DAYS_PER_MONTH
sigma2 = np.zeros(n_days)
sq_ret = np.zeros(n_days)
sigma2[0] = 0.0001
for t in range(1, n_days):
    sigma2[t] = 0.000005 + 0.08 * sq_ret[t-1] + 0.88 * sigma2[t-1]
    sq_ret[t] = sigma2[t] * np.random.chi2(1)

daily_ret = np.sqrt(np.maximum(sq_ret, 0)) * np.random.choice([-1, 1], n_days)

# Monthly RV = sum of K daily squared returns
monthly_rv = np.array([sq_ret[m*K:(m+1)*K].sum() for m in range(MONTHS)])

# Build daily squared lag matrix for MIDAS
X_daily_sq = np.array([sq_ret[m*K:(m+1)*K][::-1] for m in range(MONTHS)])  # (T, K) lag1=most recent

print("Data ready:")
print(f"  Daily squared returns: {sq_ret.shape}")
print(f"  Monthly RV: {monthly_rv.shape}")
print(f"  Mean monthly RV: {monthly_rv.mean():.6f}")
print(f"  Mean annualised vol: {np.sqrt(monthly_rv.mean() * 252):.1%}")
print()

# ──────────────────────────────────────────────────────────────────────────────
# TASK 1: Compute Beta weights and verify properties
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TASK 1: Beta weights")
print("=" * 60)


def beta_weights(K, theta1, theta2):
    """Compute normalised Beta polynomial weights."""
    x = np.linspace(0.001, 0.999, K)
    unnorm = x**(theta1 - 1) * (1 - x)**(theta2 - 1)
    return unnorm / unnorm.sum()


# Test with several parameter combinations
configs = [(1.0, 5.0), (1.0, 1.0), (2.0, 5.0)]
for t1, t2 in configs:
    w = beta_weights(K, t1, t2)
    print(f"theta1={t1}, theta2={t2}: sum={w.sum():.6f}, max_lag={np.argmax(w)+1}, w[0]={w[0]:.4f}")

    # VERIFY: weights sum to 1
    assert abs(w.sum() - 1.0) < 1e-8, f"Weights don't sum to 1! Got {w.sum()}"

    # VERIFY: all weights are non-negative
    assert np.all(w >= 0), f"Negative weights found!"

print()
print("TASK 1 PASSED: Beta weights are valid (sum=1, non-negative).")
print()

# ──────────────────────────────────────────────────────────────────────────────
# TASK 2: Fit MIDAS-RV by NLS
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TASK 2: MIDAS-RV estimation by NLS")
print("=" * 60)

log_rv = np.log(monthly_rv + 1e-10)
log_d_sq = np.log(X_daily_sq + 1e-10)


def midas_rv_sse(params, X_log, y_log):
    mu, phi, t1, t2 = params
    if t1 <= 0 or t2 <= 0:
        return 1e10
    w = beta_weights(K, t1, t2)
    fitted = mu + phi * (X_log @ w)
    return np.sum((y_log - fitted)**2)


# Fit MIDAS-RV
best_sse = np.inf
best_params = None
for x0 in [[log_rv.mean(), 0.5, 1.0, 5.0], [log_rv.mean(), 0.7, 1.5, 3.0]]:
    res = minimize(midas_rv_sse, x0=x0,
                   args=(log_d_sq, log_rv),
                   method='L-BFGS-B',
                   bounds=[(None, None), (0, 2), (0.01, 20), (0.01, 20)])
    if res.fun < best_sse:
        best_sse = res.fun
        best_params = res.x

mu_hat, phi_hat, theta1_hat, theta2_hat = best_params
fitted_log_rv = mu_hat + phi_hat * (log_d_sq @ beta_weights(K, theta1_hat, theta2_hat))
in_sample_rmse = np.sqrt(np.mean((log_rv - fitted_log_rv)**2))

print(f"MIDAS-RV estimates:")
print(f"  μ = {mu_hat:.4f}")
print(f"  φ = {phi_hat:.4f}")
print(f"  θ₁ = {theta1_hat:.4f}")
print(f"  θ₂ = {theta2_hat:.4f}")
print(f"  In-sample RMSE (log-RV): {in_sample_rmse:.4f}")
print()

# VERIFY: economic plausibility
assert 0 < phi_hat < 2, f"phi should be in (0,2), got {phi_hat:.4f}"
assert theta1_hat > 0 and theta2_hat > 0, f"theta params must be positive"
assert in_sample_rmse < 5.0, f"RMSE too high: {in_sample_rmse:.4f}"
assert best_params is not None, "Optimisation failed"
print("TASK 2 PASSED: MIDAS-RV estimated successfully with plausible parameters.")
print()

# ──────────────────────────────────────────────────────────────────────────────
# TASK 3: Kupiec VaR backtest
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TASK 3: Kupiec (1995) VaR backtest")
print("=" * 60)

# Generate daily VaR from MIDAS-RV
# Scale monthly variance to daily, apply normal quantile
alpha_var = 0.01
z_alpha = stats.norm.ppf(alpha_var)  # ≈ -2.326

# Monthly MIDAS-RV → daily VaR
monthly_var_forecast = np.exp(fitted_log_rv)
daily_var_forecast = monthly_var_forecast / DAYS_PER_MONTH
var_daily = -z_alpha * np.sqrt(daily_var_forecast)

# Map monthly VaR to daily: each day in a month gets the same VaR
var_daily_expanded = np.repeat(var_daily, DAYS_PER_MONTH)

# Kupiec test
violations = daily_ret < -var_daily_expanded  # Actual loss exceeds VaR
T_d = len(daily_ret)
V = violations.sum()
p_hat = V / T_d

def kupiec_lr_stat(V, T, alpha):
    """Kupiec LR statistic for H0: violation rate = alpha."""
    if V == 0 or V == T:
        return np.nan, np.nan
    p = V / T
    lr = 2 * (V * np.log(p / alpha) + (T - V) * np.log((1 - p) / (1 - alpha)))
    pval = stats.chi2.sf(lr, df=1)
    return lr, pval

lr_stat, p_value = kupiec_lr_stat(V, T_d, alpha_var)

print(f"VaR level: {(1 - alpha_var)*100:.0f}%  (expected {alpha_var*100:.1f}% violations)")
print(f"Actual violation rate: {p_hat:.4f} ({V}/{T_d} days)")
print(f"Kupiec LR statistic: {lr_stat:.4f}")
print(f"p-value: {p_value:.4f}")
if p_value >= 0.05:
    print(f"Result: PASS — fail to reject H0 (model well-calibrated)")
else:
    print(f"Result: REJECT H0 — violation rate significantly different from {alpha_var*100:.1f}%")
print()

# VERIFY
assert isinstance(lr_stat, float), "LR statistic should be a float"
assert 0 <= p_hat <= 1, f"Violation rate must be in [0,1], got {p_hat}"
assert not np.isnan(lr_stat) or V == 0 or V == T_d, "NaN only allowed for edge cases"
print("TASK 3 PASSED: Kupiec test computed correctly.")
print()

# ──────────────────────────────────────────────────────────────────────────────
# TASK 4: QLIKE vs MSE — do they give the same model ranking?
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TASK 4: QLIKE vs MSE model ranking")
print("=" * 60)


def qlike(actual, forecast):
    """QLIKE loss function (Patton 2011, proxy-robust)."""
    u = actual / (forecast + 1e-10)
    return np.mean(u - np.log(u + 1e-10) - 1)


def mse(actual, forecast):
    return np.mean((actual - forecast)**2)


# Two models: MIDAS-RV (estimated) and a naive forecast (historical mean)
rv_levels = np.exp(log_rv)  # Monthly RV in levels

midas_rv_levels = np.exp(fitted_log_rv)
naive_rv_levels = np.full_like(rv_levels, rv_levels[:60].mean())  # Mean of first half

print(f"Comparing MIDAS-RV vs Naive (historical mean) on last 60 months:")
eval_sl = slice(60, None)
y_eval = rv_levels[eval_sl]
midas_eval = midas_rv_levels[eval_sl]
naive_eval = naive_rv_levels[eval_sl]

for name, preds in [('MIDAS-RV', midas_eval), ('Naive (mean)', naive_eval)]:
    ql = qlike(y_eval, preds)
    ms = mse(y_eval, preds)
    rmse_log = np.sqrt(np.mean((np.log(y_eval + 1e-10) - np.log(preds + 1e-10))**2))
    print(f"  {name:<20}: QLIKE={ql:.6f}, MSE={ms:.8f}, RMSE(log)={rmse_log:.4f}")

midas_ql = qlike(y_eval, midas_eval)
naive_ql = qlike(y_eval, naive_eval)
midas_mse = mse(y_eval, midas_eval)
naive_mse = mse(y_eval, naive_eval)

qlike_winner = 'MIDAS-RV' if midas_ql < naive_ql else 'Naive'
mse_winner = 'MIDAS-RV' if midas_mse < naive_mse else 'Naive'

print(f"\n  QLIKE winner: {qlike_winner}")
print(f"  MSE winner:   {mse_winner}")
print(f"  Consistent ranking: {'YES' if qlike_winner == mse_winner else 'NO — rankings differ!'}")
print()

# VERIFY
assert midas_ql >= 0, f"QLIKE must be non-negative, got {midas_ql}"
assert midas_mse >= 0, f"MSE must be non-negative, got {midas_mse}"
print("TASK 4 PASSED: QLIKE and MSE both computed correctly.")
print()

# ──────────────────────────────────────────────────────────────────────────────
# Final summary
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("ALL TASKS COMPLETED")
print("=" * 60)
print()
print(f"Summary:")
print(f"  MIDAS-RV: θ₁={theta1_hat:.3f}, θ₂={theta2_hat:.3f}, φ={phi_hat:.4f}")
print(f"  Weight pattern: {'geometric decay' if theta1_hat < 1.5 else 'hump-shaped'}")
print()
print(f"  VaR backtest: {V}/{T_d} violations ({p_hat:.2%})")
print(f"  Kupiec p-value: {p_value:.4f} → {'PASS' if p_value >= 0.05 else 'FAIL'}")
print()
print(f"  QLIKE: MIDAS-RV {midas_ql:.6f} vs Naive {naive_ql:.6f}")
print(f"  MIDAS-RV QLIKE improvement: {(1 - midas_ql/naive_ql)*100:.1f}%")
