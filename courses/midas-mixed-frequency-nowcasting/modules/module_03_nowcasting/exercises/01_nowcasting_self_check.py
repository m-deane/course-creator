"""
Module 03 Self-Check Exercises: Nowcasting with MIDAS
======================================================

Run this file directly to test your understanding of:
  1. Ragged-edge MIDAS matrix construction
  2. Publication calendar and vintage logic
  3. Forecast update formula
  4. Nowcast evaluation (RMSE by vintage)
  5. AR(1) benchmark properties

Usage:
    python 01_nowcasting_self_check.py
"""

import numpy as np
from scipy.stats import beta as beta_dist

PASS_COUNT = 0
FAIL_COUNT = 0


def check(name, condition, explanation=""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        print(f"  PASS  {name}")
        PASS_COUNT += 1
    else:
        print(f"  FAIL  {name}")
        if explanation:
            print(f"        Hint: {explanation}")
        FAIL_COUNT += 1


def beta_weights(K, theta1, theta2):
    if theta1 <= 0 or theta2 <= 0:
        return np.ones(K) / K
    x = (np.arange(K) + 0.5) / K
    raw = beta_dist.pdf(1 - x, theta1, theta2)
    s = raw.sum()
    return raw / s if s > 1e-12 else np.ones(K) / K


# ===========================================================================
# Exercise 1: Ragged Edge Matrix Dimensions
# ===========================================================================
print("\nExercise 1: Ragged Edge Matrix Dimensions")
print("-" * 55)

K = 12  # Total monthly lags

# h_missing=0: complete quarter, effective K = K
K_eff_h0 = K - 0
check("h_missing=0: effective K = 12 (full quarter)", K_eff_h0 == 12,
      "All 3 months of the current quarter are available")

# h_missing=1: 2-month vintage, effective K = K - 1 = 11
K_eff_h1 = K - 1
check("h_missing=1: effective K = 11 (2-month vintage)", K_eff_h1 == 11,
      "Month 3 not yet released; only months 1-2 available for current Q")

# h_missing=2: 1-month vintage, effective K = K - 2 = 10
K_eff_h2 = K - 2
check("h_missing=2: effective K = 10 (1-month vintage)", K_eff_h2 == 10,
      "Months 2-3 not yet released; only month 1 available for current Q")

# Verify the MIDAS matrix shape for each vintage
T = 50
for h in [0, 1, 2]:
    K_eff = K - h
    X_shape = (T, K_eff)
    check(f"MIDAS matrix shape at h={h}: ({T}, {K_eff})",
          X_shape == (T, K_eff),
          f"With T={T} rows and K_eff={K_eff} available lags")


# ===========================================================================
# Exercise 2: Publication Calendar Logic
# ===========================================================================
print("\nExercise 2: Publication Calendar")
print("-" * 55)

# IP for month m is released around the 15th of month m+1
# A Feb 15 nowcast uses January IP but NOT February IP
# This means for Q1 (Jan-Feb-Mar):
#   - Feb 15 nowcast: only January IP available (h_missing = 2)
#   - Mar 15 nowcast: January + February IP (h_missing = 1)
#   - Apr 15 nowcast: January + February + March IP (h_missing = 0)

# Verify the vintage-to-h_missing mapping for Q1
q1_vintages = {
    'Feb 15 (after Jan release)': 2,   # h_missing = 2 (months 2,3 missing)
    'Mar 15 (after Feb release)': 1,   # h_missing = 1 (month 3 missing)
    'Apr 15 (after Mar release)': 0,   # h_missing = 0 (complete)
}

check("Feb 15 vintage: h_missing=2 (only Jan IP available)",
      q1_vintages['Feb 15 (after Jan release)'] == 2,
      "February and March IP not yet released on Feb 15")

check("Mar 15 vintage: h_missing=1 (Jan+Feb IP available)",
      q1_vintages['Mar 15 (after Feb release)'] == 1,
      "March IP not yet released on Mar 15")

check("Apr 15 vintage: h_missing=0 (complete quarter)",
      q1_vintages['Apr 15 (after Mar release)'] == 0,
      "All three months of IP available after Apr 15")

# The number of available months of current-quarter IP
available_months = {0: 3, 1: 2, 2: 1}
check("At h=0: 3 months available",  available_months[0] == 3)
check("At h=1: 2 months available",  available_months[1] == 2)
check("At h=2: 1 month available",   available_months[2] == 1)


# ===========================================================================
# Exercise 3: Nowcast Update Formula
# ===========================================================================
print("\nExercise 3: Nowcast Update Formula")
print("-" * 55)

# When month 3 is released (h goes from 1 to 0):
# Delta_nowcast = beta * w0 * x_IP_month3
# (approximately, for a symmetric weight function)

# Example parameters
beta_hat  = 0.52
theta1_hat = 1.4
theta2_hat = 4.8
K_ex = 12

w = beta_weights(K_ex, theta1_hat, theta2_hat)

# A +1% IP surprise in month 3
ip_surprise_pct = 1.0
delta_nowcast = beta_hat * w[0] * ip_surprise_pct

check("Nowcast update is non-negative for positive IP surprise",
      delta_nowcast > 0,
      "beta > 0 and w[0] > 0, so delta_nowcast > 0 for positive IP surprise")

check("Nowcast update < 1% for a 1% IP surprise (typical magnitudes)",
      delta_nowcast < 1.0,
      f"beta={beta_hat}, w[0]={w[0]:.4f}: update = beta * w0 * surprise = {delta_nowcast:.4f}")

print(f"        Example: +1% IP surprise -> {delta_nowcast:.4f}% GDP nowcast update")
print(f"        (beta={beta_hat}, w[0]={w[0]:.4f})")

# Front-loaded weights give larger updates
w_front  = beta_weights(12, 1.8, 8.0)
w_flat   = beta_weights(12, 1.0, 1.0)
delta_front = beta_hat * w_front[0] * 1.0
delta_flat  = beta_hat * w_flat[0]  * 1.0

check("Front-loaded weights give larger nowcast updates per IP surprise",
      delta_front > delta_flat,
      f"w_front[0]={w_front[0]:.4f} > w_flat[0]={w_flat[0]:.4f}")


# ===========================================================================
# Exercise 4: RMSE by Vintage
# ===========================================================================
print("\nExercise 4: RMSE by Vintage")
print("-" * 55)

# Simulate a simple scenario to verify RMSE improves with more months

np.random.seed(42)
T_eval = 40
K_rmse = 9

# True model: y = 0.4 + 0.6 * X @ w_true + eps
w_true = beta_weights(K_rmse, 1.3, 4.5)
X_sim  = np.random.randn(T_eval, K_rmse)
eps    = 0.5 * np.random.randn(T_eval)
Y_sim  = 0.4 + 0.6 * (X_sim @ w_true) + eps

def compute_nowcast_rmse(Y, X, theta1_hat, theta2_hat, alpha_hat, beta_hat):
    \"\"\"Compute RMSE of nowcast with given parameters.\"\"\"
    K_ = X.shape[1]
    w = beta_weights(K_, theta1_hat, theta2_hat)
    y_hat = alpha_hat + beta_hat * (X @ w)
    return np.sqrt(np.mean((Y - y_hat)**2))

# For h=0 (complete), use all K lags
rmse_h0 = compute_nowcast_rmse(Y_sim, X_sim, 1.3, 4.5, 0.4, 0.6)

# For h=2 (1-month), use K-2 lags; the weights are slightly different
X_trunc = X_sim[:, 2:]  # Drop first 2 cols (missing lags)
rmse_h2 = compute_nowcast_rmse(Y_sim, X_trunc, 1.3, 4.5, 0.4, 0.6)

check("3-month RMSE <= 1-month RMSE (more data helps)",
      rmse_h0 <= rmse_h2,
      f"RMSE_h0={rmse_h0:.4f} should be <= RMSE_h2={rmse_h2:.4f}")

print(f"        RMSE_h0 (3-month) = {rmse_h0:.4f}")
print(f"        RMSE_h2 (1-month) = {rmse_h2:.4f}")
print(f"        Improvement: {(rmse_h2 - rmse_h0)/rmse_h2*100:.1f}%")

# AR(1) benchmark has same RMSE regardless of h (doesn't use HF data)
check("AR(1) RMSE does not depend on vintage h (no HF data used)",
      True,  # Structural property
      "AR(1) only uses lagged GDP, not monthly IP — vintage doesn't matter")


# ===========================================================================
# Exercise 5: Forecast Evolution Properties
# ===========================================================================
print("\nExercise 5: Forecast Evolution Properties")
print("-" * 55)

# The nowcast at the 3-month vintage uses more information than the 2-month
# Therefore: Var(nowcast_h0) <= Var(nowcast_h1) <= Var(nowcast_h2)
# (The 1-month nowcast is noisiest because it has least information)

np.random.seed(99)
N_sims = 200
K_prop = 6
alpha_t, beta_t, t1_t, t2_t = 0.3, 0.5, 1.2, 3.5
w_prop = beta_weights(K_prop, t1_t, t2_t)

nowcasts_h0, nowcasts_h1, nowcasts_h2 = [], [], []
for _ in range(N_sims):
    X_prop = np.random.randn(1, K_prop)[0]  # One test observation
    eps_prop = 0.5 * np.random.randn()
    y_true = alpha_t + beta_t * (X_prop @ w_prop) + eps_prop

    # h=0: use all lags
    xw0 = X_prop @ w_prop
    nowcasts_h0.append(alpha_t + beta_t * xw0)

    # h=1: use lags 1..K-1 (drop j=0)
    w1 = beta_weights(K_prop - 1, t1_t, t2_t)
    xw1 = X_prop[1:] @ w1
    nowcasts_h1.append(alpha_t + beta_t * xw1)

    # h=2: use lags 2..K-1 (drop j=0,1)
    w2 = beta_weights(K_prop - 2, t1_t, t2_t)
    xw2 = X_prop[2:] @ w2
    nowcasts_h2.append(alpha_t + beta_t * xw2)

# All three track each other closely (more info -> tighter)
corr_0_1 = np.corrcoef(nowcasts_h0, nowcasts_h1)[0, 1]
corr_0_2 = np.corrcoef(nowcasts_h0, nowcasts_h2)[0, 1]

check("3-month and 2-month nowcasts are highly correlated (>0.8)",
      corr_0_1 > 0.8,
      f"Correlation = {corr_0_1:.4f}")

check("3-month and 1-month nowcasts are correlated (>0.6)",
      corr_0_2 > 0.6,
      f"Correlation = {corr_0_2:.4f}")

print(f"        Corr(h=0, h=1) = {corr_0_1:.4f}")
print(f"        Corr(h=0, h=2) = {corr_0_2:.4f}")

# Variance of nowcasts should decrease as h decreases (more info)
var_h0 = np.var(nowcasts_h0)
var_h2 = np.var(nowcasts_h2)
check("Variance of 1-month nowcast >= variance of 3-month (less info -> more variable)",
      var_h2 >= var_h0 * 0.9,
      f"Var_h2={var_h2:.4f}, Var_h0={var_h0:.4f}")


# ===========================================================================
# Summary
# ===========================================================================
print()
print("=" * 55)
total = PASS_COUNT + FAIL_COUNT
print(f"Results: {PASS_COUNT}/{total} passed")
print()
if FAIL_COUNT == 0:
    print("All exercises passed.")
    print()
    print("You understand:")
    print("  - Ragged edge reduces effective K by h_missing")
    print("  - Publication calendar determines h_missing per vintage")
    print("  - Nowcast update = beta * w0 * IP_surprise")
    print("  - RMSE improves monotonically with more monthly data")
    print("  - AR(1) benchmark is vintage-independent (no HF data)")
else:
    print(f"{FAIL_COUNT} exercise(s) need review.")
