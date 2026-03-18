"""
Module 04 — Self-Check Exercises: Difference-in-Differences
============================================================

These exercises reinforce the DiD concepts from Module 04. Run each
exercise, check the assertions, and modify parameters to explore
how estimates change.

No submission required — these are for your own understanding.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

np.random.seed(404)


# ============================================================
# EXERCISE 1: Compute DiD manually from group means
# ============================================================
print("=" * 55)
print("EXERCISE 1: Manual DiD from Group Means")
print("=" * 55)

# A retail chain introduces a loyalty card program in one region
# (treated) but not another (control).
# Pre-period: quarterly sales before the program
# Post-period: quarterly sales after the program

treated_pre = np.array([120, 115, 125, 118, 122, 119, 121, 117])
treated_post = np.array([135, 140, 138, 133, 141, 136, 139, 142])
control_pre = np.array([98, 102, 100, 97, 103, 99, 101, 100])
control_post = np.array([101, 98, 103, 99, 104, 100, 102, 101])

# TODO: Complete the DiD calculation
# Step 1: Compute mean for each group-period cell
mean_treated_pre = np.mean(treated_pre)
mean_treated_post = np.mean(treated_post)
mean_control_pre = np.mean(control_pre)
mean_control_post = np.mean(control_post)

# Step 2: Compute within-group changes
delta_treated = mean_treated_post - mean_treated_pre
delta_control = mean_control_post - mean_control_pre

# Step 3: Compute DiD
did_estimate = delta_treated - delta_control

print(f"Treated group: pre={mean_treated_pre:.1f}, post={mean_treated_post:.1f}, Δ={delta_treated:.1f}")
print(f"Control group: pre={mean_control_pre:.1f}, post={mean_control_post:.1f}, Δ={delta_control:.1f}")
print(f"DiD estimate: {did_estimate:.2f}")

# Self-check
assert did_estimate > 15, "DiD estimate should be around 17-18. Check your calculation."
assert did_estimate < 22, "DiD estimate seems too large. Check your calculation."
print("CHECK PASSED: DiD estimate is in the expected range.")


# ============================================================
# EXERCISE 2: Regression DiD
# ============================================================
print("\n" + "=" * 55)
print("EXERCISE 2: Regression-Based DiD")
print("=" * 55)

# Build a panel DataFrame from the arrays above
data = pd.DataFrame({
    'sales': np.concatenate([treated_pre, treated_post, control_pre, control_post]),
    'treated': np.concatenate([np.ones(8), np.ones(8), np.zeros(8), np.zeros(8)]),
    'post': np.concatenate([np.zeros(8), np.ones(8), np.zeros(8), np.ones(8)]),
})

# Fit the DiD regression
model = smf.ols('sales ~ post + treated + post:treated', data=data).fit()
ols_did = model.params['post:treated']

print(f"OLS DiD (regression): {ols_did:.3f}")
print(f"Manual DiD:           {did_estimate:.3f}")

# Self-check: regression should match manual calculation
assert abs(ols_did - did_estimate) < 0.01, \
    f"OLS DiD ({ols_did:.3f}) should match manual DiD ({did_estimate:.3f})"
print("CHECK PASSED: OLS and manual DiD estimates match.")

# Question: interpret each coefficient
print("\nAll regression coefficients:")
for name, val in model.params.items():
    print(f"  {name}: {val:.3f}")
print("\nInterpretation:")
print(f"  Intercept ({model.params['Intercept']:.1f}): Control group mean in pre-period")
print(f"  post ({model.params['post']:.1f}): Control group change over time")
print(f"  treated ({model.params['treated']:.1f}): Pre-period level difference")
print(f"  post:treated ({model.params['post:treated']:.1f}): DiD treatment effect (ATT)")


# ============================================================
# EXERCISE 3: Parallel Trends Assessment
# ============================================================
print("\n" + "=" * 55)
print("EXERCISE 3: Parallel Trends Assessment")
print("=" * 55)

# Simulate 4-period panel to check pre-trends
n_units = 60
n_treated = 30

# True treatment happens at period 3 (post-period is period 3 and 4)
true_effect = 10.0

rows = []
for u in range(n_units):
    unit_fe = np.random.normal(0, 3)
    treated = 1 if u < n_treated else 0
    for t in [1, 2, 3, 4]:
        # Apply treatment only at t >= 3 for treated units
        tau = true_effect if (treated and t >= 3) else 0
        y = 50 + 2 * t + unit_fe + tau + np.random.normal(0, 2)
        rows.append({'unit': u, 'period': t, 'treated': treated, 'outcome': y})

panel = pd.DataFrame(rows)

# Check parallel trends: compare pre-period trends (t=1 and t=2)
pre_period = panel[panel['period'].isin([1, 2])]
trend_treated = pre_period[pre_period['treated'] == 1].groupby('period')['outcome'].mean()
trend_control = pre_period[pre_period['treated'] == 0].groupby('period')['outcome'].mean()

print("Pre-period trends (periods 1 and 2):")
print(f"  Treated: period 1 = {trend_treated[1]:.1f}, period 2 = {trend_treated[2]:.1f}")
print(f"  Control: period 1 = {trend_control[1]:.1f}, period 2 = {trend_control[2]:.1f}")
print(f"  Treated slope:  {trend_treated[2] - trend_treated[1]:.2f}")
print(f"  Control slope:  {trend_control[2] - trend_control[1]:.2f}")
print(f"  Difference in pre-trends: {abs((trend_treated[2]-trend_treated[1]) - (trend_control[2]-trend_control[1])):.2f}")

# Self-check: pre-trends should be approximately parallel
pre_trend_diff = abs((trend_treated[2]-trend_treated[1]) - (trend_control[2]-trend_control[1]))
assert pre_trend_diff < 2.0, \
    f"Pre-trend difference too large ({pre_trend_diff:.2f}). Parallel trends may be violated."
print("CHECK PASSED: Pre-period trends are approximately parallel.")


# ============================================================
# EXERCISE 4: TWFE with multiple periods
# ============================================================
print("\n" + "=" * 55)
print("EXERCISE 4: TWFE with Multiple Periods")
print("=" * 55)

# Run TWFE on the 4-period panel
model_twfe = smf.ols('outcome ~ treated:C(period) + C(unit) + C(period)',
                      data=panel).fit()

# The treatment effect at each post period
post_effects = {}
for t in [3, 4]:
    param_name = f'treated:C(period)[T.{t}]'
    if param_name in model_twfe.params:
        post_effects[t] = model_twfe.params[param_name]

print("TWFE treatment effects by period:")
for t, eff in post_effects.items():
    print(f"  Period {t}: {eff:.2f} (true = {true_effect:.1f})")

# Simpler: binary treatment DiD
panel['post'] = (panel['period'] >= 3).astype(int)
model_binary = smf.ols('outcome ~ post + treated + post:treated + C(unit)',
                        data=panel).fit()
binary_did = model_binary.params['post:treated']
print(f"\nBinary post DiD estimate: {binary_did:.2f} (true = {true_effect:.1f})")

# Self-check
assert abs(binary_did - true_effect) < 3.0, \
    f"DiD estimate ({binary_did:.2f}) too far from true effect ({true_effect:.1f})"
print("CHECK PASSED: TWFE recovers approximately correct treatment effect.")


# ============================================================
# EXERCISE 5: Sensitivity to Parallel Trends Violation
# ============================================================
print("\n" + "=" * 55)
print("EXERCISE 5: Sensitivity to Parallel Trends Violation")
print("=" * 55)

results = []
for violation_size in [0.0, 0.5, 1.0, 2.0, 3.0]:
    rows_v = []
    for u in range(n_units):
        unit_fe = np.random.normal(0, 3)
        treated = 1 if u < n_treated else 0
        for t in [1, 2, 3, 4]:
            # Add a pre-existing trend to treated units (parallel trends violation)
            pre_trend_bias = treated * violation_size * t
            tau = true_effect if (treated and t >= 3) else 0
            y = 50 + 2 * t + unit_fe + tau + pre_trend_bias + np.random.normal(0, 2)
            rows_v.append({'unit': u, 'period': t, 'treated': treated,
                           'outcome': y, 'post': int(t >= 3)})
    panel_v = pd.DataFrame(rows_v)
    m = smf.ols('outcome ~ post + treated + post:treated + C(unit)', data=panel_v).fit()
    did_v = m.params['post:treated']
    bias = did_v - true_effect
    results.append({'violation': violation_size, 'estimate': did_v, 'bias': bias})

results_df = pd.DataFrame(results)
print(f"{'Violation Size':>16} {'DiD Estimate':>13} {'Bias':>8}")
print("-" * 40)
for _, row in results_df.iterrows():
    print(f"{row['violation']:>16.1f} {row['estimate']:>13.2f} {row['bias']:>+8.2f}")

print(f"\nThe DiD estimate is upward biased when treated units have a pre-existing")
print(f"positive trend. Bias grows proportionally with violation size.")

# Self-check: bias should grow with violation size
biases = results_df['bias'].values
assert biases[-1] > biases[0] + 2, \
    "Bias should be noticeably larger with a large pre-trend violation"
print("\nCHECK PASSED: Larger parallel trends violations produce larger bias.")

print("\n" + "=" * 55)
print("ALL EXERCISES COMPLETE")
print("=" * 55)
print("\nKey takeaways:")
print("1. DiD = (treated change) - (control change) = ATT under parallel trends")
print("2. OLS with post:treated interaction recovers the same estimate")
print("3. Parallel trends requires equal TRENDS, not equal LEVELS")
print("4. Pre-period trend differences signal parallel trends violations")
print("5. Violations cause bias proportional to the pre-trend difference")
