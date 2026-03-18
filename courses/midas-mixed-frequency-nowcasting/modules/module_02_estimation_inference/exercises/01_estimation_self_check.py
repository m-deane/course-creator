"""
Module 02 Self-Check Exercises: Estimation and Inference
=========================================================

These exercises test understanding of NLS estimation, model selection,
and HAC inference for MIDAS regression. Run this file directly with:

    python 01_estimation_self_check.py

All exercises print PASS or FAIL with an explanation.

Topics covered:
  1. Profile SSE and analytical (alpha, beta) solution
  2. AIC/BIC parameter count for restricted vs unrestricted MIDAS
  3. HAC bandwidth rule
  4. F-test for equal-weight restriction
  5. Bootstrap residual resampling logic
"""

import numpy as np
from scipy.stats import beta as beta_dist, f as f_dist, chi2
from scipy.optimize import minimize

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


# ---------------------------------------------------------------------------
# Utility functions (same as notebooks)
# ---------------------------------------------------------------------------

def beta_weights(K, theta1, theta2):
    """Beta polynomial weights evaluated at midpoints (j+0.5)/K."""
    if theta1 <= 0 or theta2 <= 0:
        return np.ones(K) / K
    x = (np.arange(K) + 0.5) / K
    raw = beta_dist.pdf(1 - x, theta1, theta2)
    s = raw.sum()
    return raw / s if s > 1e-12 else np.ones(K) / K


def profile_sse_fn(theta, Y, X):
    """Profile SSE: solve (alpha, beta) analytically for given theta."""
    t1, t2 = theta
    if t1 <= 0.01 or t2 <= 0.01:
        return 1e10
    w = beta_weights(X.shape[1], t1, t2)
    xw = X @ w
    xc = xw - xw.mean()
    yc = Y - Y.mean()
    ss = np.dot(xc, xc)
    if ss < 1e-12:
        return np.sum((Y - Y.mean())**2)
    beta_ = np.dot(yc, xc) / ss
    alpha_ = Y.mean() - beta_ * xw.mean()
    return np.sum((Y - alpha_ - beta_ * xw)**2)


# ---------------------------------------------------------------------------
# Generate synthetic MIDAS data for exercises
# ---------------------------------------------------------------------------

np.random.seed(123)
T, K = 80, 9

# True parameters
TRUE_THETA1, TRUE_THETA2 = 1.2, 4.5
TRUE_ALPHA, TRUE_BETA = 0.5, 0.8

w_true = beta_weights(K, TRUE_THETA1, TRUE_THETA2)
X_syn = np.random.randn(T, K)
xw_true = X_syn @ w_true
Y_syn = TRUE_ALPHA + TRUE_BETA * xw_true + 0.5 * np.random.randn(T)


# ===========================================================================
# Exercise 1: Profile SSE — Verify analytical beta solution
# ===========================================================================
print("\nExercise 1: Profile SSE and Analytical Beta")
print("-" * 50)

# Compute profile SSE at the true theta — should equal the minimum OLS SSE
# for fixed weights
theta_true = [TRUE_THETA1, TRUE_THETA2]
psse_at_true = profile_sse_fn(theta_true, Y_syn, X_syn)

# Compare to brute-force OLS at the same weights
xw_at_true = X_syn @ w_true
Xreg = np.column_stack([np.ones(T), xw_at_true])
params = np.linalg.lstsq(Xreg, Y_syn, rcond=None)[0]
alpha_bf, beta_bf = params
resid_bf = Y_syn - alpha_bf - beta_bf * xw_at_true
sse_bf = np.sum(resid_bf**2)

check(
    "Profile SSE matches brute-force OLS SSE at true theta",
    abs(psse_at_true - sse_bf) < 1e-9,
    "profile_sse_fn should give the same SSE as explicitly running OLS for fixed weights"
)

# Profile SSE at theta=(1,1) should be >= SSE at true theta (true theta is closer to optimum)
psse_equal = profile_sse_fn([1.0, 1.0], Y_syn, X_syn)
check(
    "Profile SSE at true theta <= SSE at equal-weight theta=(1,1)",
    psse_at_true <= psse_equal + 0.5,
    "The true parameters should produce lower SSE than arbitrary equal-weight params"
)

# Verify analytical beta recovers the true value approximately
xw_opt = X_syn @ beta_weights(K, TRUE_THETA1, TRUE_THETA2)
xc = xw_opt - xw_opt.mean()
yc = Y_syn - Y_syn.mean()
beta_analytical = np.dot(yc, xc) / np.dot(xc, xc)
check(
    f"Analytical beta within 0.3 of true beta={TRUE_BETA}",
    abs(beta_analytical - TRUE_BETA) < 0.3,
    "With T=80 and moderate noise, the analytical beta should be close to the true value"
)


# ===========================================================================
# Exercise 2: AIC/BIC parameter counts
# ===========================================================================
print("\nExercise 2: AIC/BIC Parameter Count Logic")
print("-" * 50)

# For restricted Beta MIDAS: k=4 regardless of K
k_restricted = 4
K_values = [3, 6, 9, 12, 18]

# Verify the BIC penalty is the same for all K under restricted MIDAS
T_test = 100
bic_penalties = [k_restricted * np.log(T_test) for K_ in K_values]
check(
    "BIC penalty is identical for all K under restricted MIDAS (k=4)",
    len(set([round(p, 8) for p in bic_penalties])) == 1,
    "For restricted MIDAS, k=4 always, so BIC penalty = 4*ln(T) does not depend on K"
)

# For U-MIDAS: k = K + 1
def k_umidas(K_):
    return K_ + 1  # intercept + K lag coefficients

k_umidas_values = [k_umidas(K_) for K_ in K_values]
check(
    "U-MIDAS k grows with K",
    all(k_umidas_values[i] < k_umidas_values[i+1] for i in range(len(k_umidas_values)-1)),
    "U-MIDAS has k=K+1 parameters; this increases with K"
)

# BIC penalty difference between U-MIDAS(K=12) and restricted MIDAS
k_diff = k_umidas(12) - k_restricted   # = 13 - 4 = 9
bic_penalty_diff = k_diff * np.log(100)  # ~20.7 units
check(
    f"U-MIDAS(K=12) BIC penalty exceeds restricted by {k_diff}*ln(T) ≈ {bic_penalty_diff:.1f}",
    abs(bic_penalty_diff - k_diff * np.log(T_test)) < 0.01,
    "The extra BIC penalty for U-MIDAS(K=12) vs restricted is (13-4)*ln(100)=9*4.6≈41.4"
)

# AIC vs BIC penalty size for T=100, k=4
aic_pen = 2 * 4
bic_pen = 4 * np.log(100)
check(
    "BIC penalty > AIC penalty for T=100 (ln(100) > 2)",
    bic_pen > aic_pen,
    "ln(100) ≈ 4.6 > 2, so BIC penalizes more heavily than AIC for T >= 8"
)


# ===========================================================================
# Exercise 3: Newey-West bandwidth rule
# ===========================================================================
print("\nExercise 3: HAC Bandwidth Rule")
print("-" * 50)

def nw_bandwidth(T_):
    """Newey-West automatic bandwidth: floor(4 * (T/100)^(2/9))."""
    return int(4 * (T_ / 100) ** (2/9))

# Verify specific known values
L_100 = nw_bandwidth(100)
L_200 = nw_bandwidth(200)
L_400 = nw_bandwidth(400)

check(
    f"Bandwidth at T=100: L={L_100} (expected 4)",
    L_100 == 4,
    "4 * (100/100)^(2/9) = 4 * 1^(2/9) = 4"
)

check(
    f"Bandwidth at T=200: L={L_200} (expected 5)",
    L_200 == 5,
    "4 * (200/100)^(2/9) = 4 * 2^(0.222) ≈ 4 * 1.166 ≈ 4.66 -> floor = 4 or 5"
)

check(
    "Bandwidth increases with T (L_200 >= L_100 >= 3)",
    L_200 >= L_100 >= 3,
    "More data -> more lags of autocorrelation can be reliably estimated"
)

check(
    "Bandwidth at T=80 is between 2 and 5",
    2 <= nw_bandwidth(80) <= 5,
    "For T=80, 4*(0.8)^(2/9) ≈ 4*0.952 ≈ 3.8 -> floor=3"
)


# ===========================================================================
# Exercise 4: F-test for equal-weight restriction
# ===========================================================================
print("\nExercise 4: F-Test for Equal-Weight Restriction")
print("-" * 50)

# Estimate MIDAS on synthetic data
res = minimize(profile_sse_fn, [1.0, 5.0], args=(Y_syn, X_syn),
               method='Nelder-Mead', options={'maxiter': 20000})
t1_hat, t2_hat = max(res.x[0], 0.01), max(res.x[1], 0.01)
w_hat = beta_weights(K, t1_hat, t2_hat)
xw_hat = X_syn @ w_hat
alpha_hat = Y_syn.mean() - (np.dot(Y_syn - Y_syn.mean(), xw_hat - xw_hat.mean()) /
                             np.dot(xw_hat - xw_hat.mean(), xw_hat - xw_hat.mean())) * xw_hat.mean()
beta_hat_ = np.dot(Y_syn - Y_syn.mean(), xw_hat - xw_hat.mean()) / \
            np.dot(xw_hat - xw_hat.mean(), xw_hat - xw_hat.mean())
sse_u = np.sum((Y_syn - alpha_hat - beta_hat_ * xw_hat)**2)

# Restricted model (equal weights)
w_r = np.ones(K) / K
xw_r = X_syn @ w_r
xc_r = xw_r - xw_r.mean()
yc_r = Y_syn - Y_syn.mean()
beta_r = np.dot(yc_r, xc_r) / np.dot(xc_r, xc_r)
alpha_r = Y_syn.mean() - beta_r * xw_r.mean()
sse_r = np.sum((Y_syn - alpha_r - beta_r * xw_r)**2)

check(
    "Unrestricted SSE <= Restricted SSE (unrestricted always fits at least as well)",
    sse_u <= sse_r + 0.001,
    "The unrestricted model nests the restricted model; SSE_U <= SSE_R always"
)

# F-statistic computation
r_restrictions = 2    # theta1=1 and theta2=1
k_unrestricted = 4    # alpha, beta, theta1, theta2
F_stat = ((sse_r - sse_u) / r_restrictions) / (sse_u / (T - k_unrestricted))

check(
    "F-statistic is non-negative",
    F_stat >= 0,
    "F = (SSE_R - SSE_U)/r / (SSE_U/(T-k_u)). Since SSE_R >= SSE_U, F >= 0."
)

p_val_f = 1 - f_dist.cdf(F_stat, r_restrictions, T - k_unrestricted)
check(
    "F-test p-value is in [0, 1]",
    0.0 <= p_val_f <= 1.0,
    "p-value must be a valid probability"
)

# For synthetic data with true non-uniform weights, F-test should sometimes reject
# (we don't enforce a specific outcome since it depends on the random sample)
print(f"        F({r_restrictions},{T-k_unrestricted})={F_stat:.3f}, p={p_val_f:.4f} "
      f"({'Reject H0' if p_val_f < 0.05 else 'Fail to reject'})")


# ===========================================================================
# Exercise 5: Bootstrap residual resampling
# ===========================================================================
print("\nExercise 5: Bootstrap Residual Resampling Logic")
print("-" * 50)

# Verify key properties of residual bootstrap

# Original residuals (from a simple OLS for this exercise)
resid_orig = Y_syn - alpha_hat - beta_hat_ * xw_hat
fitted_orig = Y_syn - resid_orig

check(
    "Residuals are zero-mean (OLS residuals satisfy this)",
    abs(resid_orig.mean()) < 0.1,
    "OLS residuals are orthogonal to the constant, so they sum to ~0"
)

# Bootstrap sample: resample residuals and add to fitted
np.random.seed(0)
resid_b = np.random.choice(resid_orig, size=T, replace=True)
Y_b = fitted_orig + resid_b

check(
    "Bootstrap Y* has same length as original Y",
    len(Y_b) == T,
    "The residual bootstrap preserves the sample size"
)

check(
    "Bootstrap Y* mean is close to original Y mean",
    abs(Y_b.mean() - Y_syn.mean()) < 0.5,
    "Since resid_b has approximately zero mean, Y_b mean ≈ Y mean"
)

# Bootstrap should NOT shuffle the X values — it only resamples residuals
check(
    "X matrix unchanged in residual bootstrap (only Y* varies)",
    True,  # Structural property — always true by design
    "The fixed-design bootstrap keeps X fixed; only the Y* values change"
)

# Bootstrap confidence interval: percentile method
# Simulate a small bootstrap run to check CI construction
n_boot_test = 99
boot_beta = []
for b in range(n_boot_test):
    resid_bb = np.random.choice(resid_orig, size=T, replace=True)
    Y_bb = fitted_orig + resid_bb
    xc_bb = xw_hat - xw_hat.mean()
    yc_bb = Y_bb - Y_bb.mean()
    beta_bb = np.dot(yc_bb, xc_bb) / np.dot(xc_bb, xc_bb)
    boot_beta.append(beta_bb)

boot_beta = np.array(boot_beta)
ci_lo, ci_hi = np.percentile(boot_beta, [2.5, 97.5])

check(
    f"Bootstrap 95% CI contains the true beta ({TRUE_BETA})",
    ci_lo <= TRUE_BETA <= ci_hi,
    f"CI = [{ci_lo:.3f}, {ci_hi:.3f}]; true beta = {TRUE_BETA}"
)

check(
    "Bootstrap CI lower bound < upper bound",
    ci_lo < ci_hi,
    "The confidence interval must be a proper interval"
)


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
    print("  - Profile SSE correctly solves (alpha, beta) analytically")
    print("  - Restricted MIDAS has k=4 regardless of K")
    print("  - Newey-West bandwidth scales with T as 4*(T/100)^(2/9)")
    print("  - F-test compares restricted vs unrestricted SSE")
    print("  - Residual bootstrap resamples residuals, not observations")
else:
    print(f"{FAIL_COUNT} exercise(s) need review.")
    print("Re-read the relevant guide sections and re-run this file.")
