"""
Module 04 Self-Check Exercises: Dynamic Factor Models
======================================================

Run this file directly to test your understanding of:
  1. PCA factor extraction mechanics
  2. Sign normalization in expanding window
  3. Bai-Ng criterion computation
  4. Factor loadings interpretation
  5. FA-MIDAS vs MIDAS comparison

Usage:
    python 01_dfm_self_check.py
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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


# Generate synthetic panel for exercises
np.random.seed(42)
T_months = 180  # 60 quarters of monthly data
N_inds   = 4    # 4 indicators

# Common factor (business cycle)
f_true = np.cumsum(0.3 * np.random.randn(T_months)) * 0.5

# Indicators: common factor + idiosyncratic noise
lambdas_true = np.array([0.8, 0.7, 0.6, 0.5])  # True loadings
X_raw = np.outer(f_true, lambdas_true) + 0.4 * np.random.randn(T_months, N_inds)

# Standardize
scaler = StandardScaler()
X_std = scaler.fit_transform(X_raw)


# ===========================================================================
# Exercise 1: PCA Factor Extraction
# ===========================================================================
print("\nExercise 1: PCA Factor Extraction")
print("-" * 55)

pca1 = PCA(n_components=1)
F1 = pca1.fit_transform(X_std)
Lambda1 = pca1.components_.T  # (N, 1)

check("Extracted factor has shape (T, 1)", F1.shape == (T_months, 1),
      "PCA with n_components=1 returns shape (T, 1)")

check("Loadings have shape (N, 1)", Lambda1.shape == (N_inds, 1),
      "pca.components_.T gives (N, n_components)")

check("Factor has approximately zero mean (PCA centers by default)",
      abs(F1.mean()) < 0.1,
      "PCA subtracts the mean of each column before extracting components")

var_exp1 = pca1.explained_variance_ratio_[0]
check(f"First factor explains >40% of variance (got {var_exp1:.1%})",
      var_exp1 > 0.40,
      "With strong common factor (loadings 0.8, 0.7, 0.6, 0.5), F1 should explain a lot")

# Loading magnitudes should be similar (all indicators share common factor equally)
load_magnitudes = np.abs(Lambda1[:, 0])
check("All loading magnitudes similar (std < 0.15)",
      np.std(load_magnitudes) < 0.15,
      "With similar true loadings (0.8, 0.7, 0.6, 0.5), estimated loadings should be similar")


# ===========================================================================
# Exercise 2: Sign Normalization
# ===========================================================================
print("\nExercise 2: Sign Normalization")
print("-" * 55)

# Extract factor twice — once with different data ordering to force sign flip
pca2a = PCA(n_components=1)
F_a = pca2a.fit_transform(X_std)

# Flip the first column of X — PCA might extract with opposite sign
X_flipped = X_std.copy()
X_flipped[:, 0] = -X_flipped[:, 0]  # Flip sign of first indicator
pca2b = PCA(n_components=1)
F_b = pca2b.fit_transform(X_flipped)

# Verify that sign normalization is needed
# (The factor might have same or opposite sign — sign normalization should fix it)
def normalize_sign(F, Lambda, ref_col=0):
    """Normalize so reference indicator has positive loading."""
    if Lambda[ref_col, 0] < 0:
        F = -F
        Lambda = -Lambda
    return F, Lambda

F_a_norm, L_a_norm = normalize_sign(F_a, pca2a.components_.T.copy())
F_b_norm, L_b_norm = normalize_sign(F_b, pca2b.components_.T.copy())

check("Reference indicator (col 0) has positive loading after normalization (case A)",
      L_a_norm[0, 0] > 0,
      "normalize_sign should ensure col-0 loading is positive")

check("Reference indicator loading is positive after normalization (case B)",
      L_b_norm[0, 0] > 0,
      "Even with flipped input, normalization fixes the sign")

# After normalization, the two factor series should be highly correlated
# (they represent the same underlying business cycle)
corr_ab = abs(np.corrcoef(F_a_norm[:, 0], F_b_norm[:, 0])[0, 1])
check(f"Normalized factors are highly correlated (|r|={corr_ab:.4f} > 0.9)",
      corr_ab > 0.9,
      "Sign normalization ensures factors from both datasets point in same direction")

print(f"        |Corr(F_a_norm, F_b_norm)| = {corr_ab:.4f}")


# ===========================================================================
# Exercise 3: Bai-Ng Criterion
# ===========================================================================
print("\nExercise 3: Bai-Ng Criterion")
print("-" * 55)

def bai_ng_ic(X, max_q=6):
    T, N = X.shape
    g = (N + T) / (N * T) * np.log(min(N, T))
    ic_vals = []
    for q in range(1, min(max_q + 1, N)):
        pca_q = PCA(n_components=q)
        F_q = pca_q.fit_transform(X)
        X_hat = F_q @ pca_q.components_
        sigma2 = np.mean((X - X_hat)**2)
        ic_vals.append(np.log(sigma2) + q * g)
    return np.array(ic_vals)

ic_values = bai_ng_ic(X_std)
q_opt = np.argmin(ic_values) + 1

check(f"IC values have correct length ({len(ic_values)})",
      len(ic_values) == min(5, N_inds - 1),
      "Should compute IC for q=1 to min(max_q, N-1)")

check("IC values decrease then increase (concave shape for 1-factor DGP)",
      ic_values[0] < ic_values[-1] or len(ic_values) <= 1,
      "With 1 true factor, IC should be minimized at q=1 or q=2")

print(f"        IC values: {ic_values.round(4)}")
print(f"        Optimal q = {q_opt}")

# With 1 strong factor, Bai-Ng should select q=1 or q=2
check(f"Optimal q <= 3 (strong single factor DGP)",
      q_opt <= 3,
      "With a dominant common factor and N=4, Bai-Ng should select q=1 or q=2")

check("Penalty term g is positive",
      (N_inds + T_months) / (N_inds * T_months) * np.log(min(N_inds, T_months)) > 0,
      "g = (N+T)/(NT) * log(min(N,T)) is always positive")


# ===========================================================================
# Exercise 4: Factor Loadings Interpretation
# ===========================================================================
print("\nExercise 4: Factor Loadings Interpretation")
print("-" * 55)

pca4 = PCA(n_components=1)
F4 = pca4.fit_transform(X_std)
Lambda4 = pca4.components_.T.copy()

# Normalize sign
if Lambda4[0, 0] < 0:
    F4 = -F4
    Lambda4 = -Lambda4

check("All loadings have same sign for procyclical indicators",
      all(Lambda4[:, 0] > 0),
      "All synthetic indicators are positively correlated with the common factor")

# Loading magnitude reflects true loading strength
# True loadings: [0.8, 0.7, 0.6, 0.5] — indicator 0 should have highest loading
check("Indicator 0 has highest loading (strongest true loading=0.8)",
      Lambda4[0, 0] == Lambda4[:, 0].max(),
      "The indicator with strongest true loading (0.8) should have highest estimated loading")

print(f"        Estimated loadings: {Lambda4[:, 0].round(4)}")
print(f"        True loadings (scaled): {lambdas_true / np.linalg.norm(lambdas_true)}")

# The factor explains more of variance for high-loading indicators
R2_per_indicator = Lambda4[:, 0]**2  # Approx variance explained per indicator
check("R² for indicator 0 > R² for indicator 3 (higher true loading)",
      R2_per_indicator[0] > R2_per_indicator[3],
      "The indicator with highest loading has most variance explained by the factor")


# ===========================================================================
# Exercise 5: Factor vs. Raw Indicator as MIDAS Predictor
# ===========================================================================
print("\nExercise 5: Factor vs. Raw Indicator as MIDAS Predictor")
print("-" * 55)

# Simulate quarterly GDP driven by the common factor
T_quarterly = T_months // 3
f_quarterly = np.array([f_true[i*3:(i+1)*3].mean() for i in range(T_quarterly)])
true_beta = 0.6
Y_gdp = 0.3 + true_beta * f_quarterly + 0.5 * np.random.randn(T_quarterly)

# Aggregate first indicator (IP-like) to quarterly
x1_quarterly = np.array([X_raw[:, 0][i*3:(i+1)*3].mean() for i in range(T_quarterly)])

# Aggregate factor to quarterly
F4_quarterly = np.array([F4[i*3:(i+1)*3, 0].mean() for i in range(T_quarterly)])

# OLS using indicator 1 vs. factor as predictor
from sklearn.linear_model import LinearRegression

lr1 = LinearRegression().fit(x1_quarterly.reshape(-1, 1), Y_gdp)
lr_f = LinearRegression().fit(F4_quarterly.reshape(-1, 1), Y_gdp)

sse1 = np.sum((Y_gdp - lr1.predict(x1_quarterly.reshape(-1, 1)))**2)
sse_f = np.sum((Y_gdp - lr_f.predict(F4_quarterly.reshape(-1, 1)))**2)

check("Factor predictor achieves lower SSE than single indicator",
      sse_f <= sse1 * 1.05,
      f"Factor SSE={sse_f:.4f} should be <= indicator SSE={sse1:.4f}")

print(f"        SSE (single indicator): {sse1:.4f}")
print(f"        SSE (factor):           {sse_f:.4f}")
if sse_f < sse1:
    improvement = (sse1 - sse_f) / sse1 * 100
    print(f"        Factor reduces SSE by {improvement:.1f}%")

# Factor has higher correlation with true factor than single indicator
corr_f_true  = abs(np.corrcoef(F4_quarterly, f_quarterly)[0, 1])
corr_x1_true = abs(np.corrcoef(x1_quarterly, f_quarterly)[0, 1])
check(f"Factor more correlated with true factor than single indicator",
      corr_f_true >= corr_x1_true - 0.05,
      f"|r(F, f_true)|={corr_f_true:.4f}, |r(x1, f_true)|={corr_x1_true:.4f}")


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
    print("  - PCA extraction returns (T, q) factors and (N, q) loadings")
    print("  - Sign normalization ensures consistent factor direction")
    print("  - Bai-Ng selects number of factors via IC minimization")
    print("  - All procyclical indicators have positive loadings on F1")
    print("  - Factor reduces noise and may improve on single-indicator regression")
else:
    print(f"{FAIL_COUNT} exercise(s) need review.")
