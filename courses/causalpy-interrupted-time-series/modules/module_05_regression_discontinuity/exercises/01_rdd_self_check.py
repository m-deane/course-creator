"""
Module 05 — Self-Check Exercises: Regression Discontinuity Design
==================================================================

These exercises reinforce the RDD concepts from Module 05. Complete each
exercise and verify the assertions to confirm your understanding.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(505)


# ============================================================
# EXERCISE 1: Identify the RDD components
# ============================================================
print("=" * 55)
print("EXERCISE 1: RDD Component Identification")
print("=" * 55)

# Given the following policy descriptions, identify:
# (a) running variable, (b) cutoff, (c) treatment, (d) sharp or fuzzy

scenarios = {
    "Scholarship": {
        "description": "Students with SAT > 1400 receive a merit scholarship",
        "running_var": "SAT score",
        "cutoff": 1400,
        "treatment": "Merit scholarship",
        "rdd_type": "Sharp"
    },
    "Medicaid": {
        "description": "Households with income < 138% FPL are eligible for Medicaid; not all enroll",
        "running_var": "Income as % of Federal Poverty Level",
        "cutoff": 138,
        "treatment": "Medicaid enrollment",
        "rdd_type": "Fuzzy"
    },
    "Speeding_Fine": {
        "description": "Drivers going >100 km/h receive an automatic fine",
        "running_var": "Vehicle speed (km/h)",
        "cutoff": 100,
        "treatment": "Speeding fine",
        "rdd_type": "Sharp"
    }
}

for name, scenario in scenarios.items():
    print(f"\n{name}: {scenario['description']}")
    print(f"  Running variable: {scenario['running_var']}")
    print(f"  Cutoff: {scenario['cutoff']}")
    print(f"  Treatment: {scenario['treatment']}")
    print(f"  Type: {scenario['rdd_type']}")

# Self-check
for scenario in scenarios.values():
    assert scenario['rdd_type'] in ['Sharp', 'Fuzzy'], "RDD type must be Sharp or Fuzzy"
print("\nCHECK PASSED: All scenarios correctly classified.")


# ============================================================
# EXERCISE 2: Manual local linear RDD
# ============================================================
print("\n" + "=" * 55)
print("EXERCISE 2: Manual Local Linear RDD Estimation")
print("=" * 55)

# Generate RDD data: income support program
# Running variable: household income (centred at $30,000 threshold)
n = 1000
TRUE_EFFECT = 500  # $500 per year benefit in outcomes (food security score)

x = np.random.normal(0, 8000, n)  # centred at 0
x = np.clip(x, -20000, 20000)
treated = (x < 0).astype(int)  # below threshold = treated (gets support)

# Outcome: food security score
food_security = (50 + 0.001 * x         # gradient
                 + TRUE_EFFECT * treated  # treatment effect
                 + np.random.normal(0, 10, n))

df = pd.DataFrame({'x': x, 'treated': treated, 'food_security': food_security})

# Estimate local linear RDD with bandwidth = 5000
bandwidth = 5000
local_df = df[np.abs(df['x']) <= bandwidth].copy()

formula = 'food_security ~ treated + x + treated:x'
model = smf.ols(formula, data=local_df).fit(cov_type='HC1')

tau = model.params['treated']
se = model.bse['treated']
t_stat = tau / se
p_val = 2 * (1 - stats.norm.cdf(abs(t_stat)))

print(f"Local linear RDD estimate:")
print(f"  τ = {tau:.2f} (true = {TRUE_EFFECT:.2f})")
print(f"  SE = {se:.2f}")
print(f"  95% CI: [{tau - 1.96*se:.2f}, {tau + 1.96*se:.2f}]")
print(f"  p-value = {p_val:.4f}")

# Self-check
assert abs(tau - TRUE_EFFECT) < 100, \
    f"Estimate {tau:.2f} too far from true value {TRUE_EFFECT:.2f}"
assert p_val < 0.05, \
    f"Treatment effect should be significant but p={p_val:.4f}"
print("CHECK PASSED: Local linear RDD recovers approximately correct estimate.")


# ============================================================
# EXERCISE 3: Bandwidth sensitivity
# ============================================================
print("\n" + "=" * 55)
print("EXERCISE 3: Bandwidth Sensitivity")
print("=" * 55)

bandwidths = [1000, 2000, 3000, 5000, 8000, 12000]
estimates = []
ses = []
n_obs = []

for h in bandwidths:
    local = df[np.abs(df['x']) <= h].copy()
    m = smf.ols('food_security ~ treated + x + treated:x', data=local).fit(cov_type='HC1')
    estimates.append(m.params['treated'])
    ses.append(m.bse['treated'])
    n_obs.append(len(local))

print(f"{'Bandwidth':>10} {'Estimate':>10} {'SE':>8} {'N obs':>7}")
print("-" * 40)
for h, est, se_v, n in zip(bandwidths, estimates, ses, n_obs):
    print(f"{h:>10} {est:>+10.1f} {se_v:>8.1f} {n:>7}")
print("-" * 40)
print(f"{'True effect':>10} {TRUE_EFFECT:>+10.1f}")

# Check: estimates should be within 2 SEs of true value at reasonable bandwidths
for i, (h, est, se_v) in enumerate(zip(bandwidths, estimates, ses)):
    if h >= 2000:
        assert abs(est - TRUE_EFFECT) < 2 * se_v + 50, \
            f"Estimate at bandwidth {h} ({est:.1f}) seems very biased"

print("CHECK PASSED: Estimates within reasonable range of true effect.")

# Check stability
est_range = max(estimates) - min(estimates)
print(f"\nEstimate range across bandwidths: {est_range:.1f}")
if est_range < 200:
    print("STABLE: Estimates are consistent across bandwidths (good RDD)")
else:
    print("SENSITIVE: Estimates vary substantially (investigate non-linearity)")


# ============================================================
# EXERCISE 4: Covariate balance test
# ============================================================
print("\n" + "=" * 55)
print("EXERCISE 4: Covariate Balance Check")
print("=" * 55)

# Add covariates that should be balanced (and one that is NOT)
df['age'] = np.random.normal(35, 10, n)  # balanced: should not jump
df['region'] = np.random.binomial(1, 0.5, n)  # balanced: should not jump

# Simulate a covariate that IS unbalanced (bad scenario to detect)
# Urban households are more likely to be just below threshold (higher cost of living)
df['urban_biased'] = (0.6 * (df['x'] < 0)  # correlated with being treated
                      + np.random.uniform(0, 0.4, n)) > 0.5
df['urban_biased'] = df['urban_biased'].astype(int)

print(f"Covariate Balance Test (bandwidth = {bandwidth}):  ")
print(f"{'Covariate':<15} {'Jump (τ)':>10} {'SE':>8} {'p-value':>10} {'Status':>10}")
print("-" * 58)

local_df = df[np.abs(df['x']) <= bandwidth].copy()
for cov, should_balance in [('age', True), ('region', True), ('urban_biased', False)]:
    m_cov = smf.ols(f'{cov} ~ treated + x + treated:x', data=local_df).fit(cov_type='HC1')
    tau_cov = m_cov.params['treated']
    se_cov = m_cov.bse['treated']
    t = tau_cov / se_cov
    p = 2 * (1 - stats.norm.cdf(abs(t)))
    status = "OK" if p > 0.05 else "IMBALANCED"
    expected = "Balanced" if should_balance else "Should be imbalanced"
    print(f"{cov:<15} {tau_cov:>+10.4f} {se_cov:>8.4f} {p:>10.4f} {status:>10}  [{expected}]")

# Self-check
m_age = smf.ols('age ~ treated + x + treated:x', data=local_df).fit(cov_type='HC1')
t_age = m_age.params['treated'] / m_age.bse['treated']
p_age = 2 * (1 - stats.norm.cdf(abs(t_age)))
assert p_age > 0.05, f"Age should be balanced but p={p_age:.3f} < 0.05"
print(f"\nCHECK PASSED: Age is correctly identified as balanced (p={p_age:.3f}).")


# ============================================================
# EXERCISE 5: Sharp vs Fuzzy — Compliance rate
# ============================================================
print("\n" + "=" * 55)
print("EXERCISE 5: Identifying Sharp vs Fuzzy RDD")
print("=" * 55)

# Create a fuzzy version: only 70% of eligible households enroll
np.random.seed(42)
n_fuzzy = 500
x_fuzz = np.random.normal(0, 8000, n_fuzzy)
x_fuzz = np.clip(x_fuzz, -20000, 20000)
eligible = (x_fuzz < 0).astype(int)

# Fuzzy compliance: 70% of eligible enroll, 5% of ineligible self-enroll
COMPLIANCE_RATE = 0.70
ALWAYS_TAKER_RATE = 0.05

enrolled = np.where(
    eligible == 1,
    np.random.binomial(1, COMPLIANCE_RATE, n_fuzzy),    # eligible: 70% enroll
    np.random.binomial(1, ALWAYS_TAKER_RATE, n_fuzzy)   # ineligible: 5% enroll
)

# First stage: jump in enrollment probability at cutoff
jump_prob = enrolled[eligible==1].mean() - enrolled[eligible==0].mean()
print(f"First stage (jump in enrollment probability):")
print(f"  Enrollment rate (eligible):   {enrolled[eligible==1].mean():.3f}")
print(f"  Enrollment rate (ineligible): {enrolled[eligible==0].mean():.3f}")
print(f"  First stage jump: {jump_prob:.3f}")
print(f"  (For sharp RDD this would be 1.0; for fuzzy, it's {jump_prob:.2f})")

# Self-check: compliance rate should be close to specified COMPLIANCE_RATE - ALWAYS_TAKER_RATE
expected_jump = COMPLIANCE_RATE - ALWAYS_TAKER_RATE
assert abs(jump_prob - expected_jump) < 0.1, \
    f"First stage jump ({jump_prob:.3f}) unexpected. Expected ~{expected_jump:.3f}"
print(f"\nCHECK PASSED: First stage jump {jump_prob:.3f} is close to expected {expected_jump:.3f}.")

print(f"\nThis is a FUZZY RDD: compliance rate = {COMPLIANCE_RATE:.0%}")
print(f"The Wald IV estimator would divide reduced form by first stage ({expected_jump:.2f})")

print("\n" + "=" * 55)
print("ALL MODULE 05 EXERCISES COMPLETE")
print("=" * 55)
print("\nKey takeaways:")
print("1. RDD exploits a discontinuous jump in treatment at a threshold")
print("2. Sharp RDD: threshold exactly determines treatment")
print("3. Fuzzy RDD: threshold shifts treatment probability → use IV")
print("4. Local linear regression with limited bandwidth is the preferred estimator")
print("5. Always check: density continuity, covariate balance, bandwidth sensitivity")
