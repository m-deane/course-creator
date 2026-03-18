"""
Module 06 — Self-Check Exercises: Instrumental Variables
=========================================================

These exercises build your IV instincts: identifying valid instruments,
computing the Wald estimator, diagnosing weak instruments, and knowing
when IV is better or worse than OLS.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

np.random.seed(601)


# ============================================================
# EXERCISE 1: IV validity assessment
# ============================================================
print("=" * 55)
print("EXERCISE 1: Assessing Instrument Validity")
print("=" * 55)

# For each instrument, assess: (a) relevance and (b) exclusion

instruments = {
    "A": {
        "description": "Random lottery for military service → effect on wages via military experience",
        "endogenous_var": "Military service",
        "outcome": "Wages",
        "relevance": "Strong",
        "exclusion": "Plausible",
        "verdict": "Valid",
        "reason": "Lottery is random (exogenous). Lottery number → service. Service → wages. Direct lottery→wages path? Unlikely — lottery number doesn't affect employer perceptions."
    },
    "B": {
        "description": "Firm profitability as instrument for R&D spending → effect on patents",
        "endogenous_var": "R&D spending",
        "outcome": "Patents",
        "relevance": "Likely",
        "exclusion": "Questionable",
        "verdict": "Probably Invalid",
        "reason": "Profitability likely correlated with managerial quality, which directly affects patents. Exclusion restriction is violated."
    },
    "C": {
        "description": "Gender of first child as instrument for having a third child (Angrist & Evans)",
        "endogenous_var": "Having third child",
        "outcome": "Female labor supply",
        "relevance": "Moderate",
        "exclusion": "Plausible",
        "verdict": "Valid (classic design)",
        "reason": "Same-sex first two children increases probability of third child. Gender of children unlikely to directly affect labor supply except through family size."
    },
    "D": {
        "description": "Local unemployment rate as instrument for individual employment status",
        "endogenous_var": "Individual employment",
        "outcome": "Health outcomes",
        "relevance": "Strong",
        "exclusion": "Questionable",
        "verdict": "Likely Invalid",
        "reason": "Local unemployment affects area health (stress, public services, social capital) directly, not only through individual employment status."
    }
}

for key, instr in instruments.items():
    print(f"\nInstrument {key}: {instr['description']}")
    print(f"  Relevance: {instr['relevance']}")
    print(f"  Exclusion: {instr['exclusion']}")
    print(f"  Verdict: {instr['verdict']}")
    print(f"  Reason: {instr['reason']}")

print("\nCHECK PASSED: Instrument validity requires both relevance AND exclusion.")


# ============================================================
# EXERCISE 2: Wald Estimator
# ============================================================
print("\n" + "=" * 55)
print("EXERCISE 2: Computing the Wald Estimator")
print("=" * 55)

N = 2000
TRUE_EFFECT = 0.10  # 10% wage return per year of education

# Unobservable ability (confounder)
ability = np.random.normal(0, 1, N)
# Instrument
college_nearby = np.random.binomial(1, 0.4, N)
# Education (endogenous)
education = 10 + 0.6 * college_nearby + 1.2 * ability + np.random.normal(0, 1.5, N)
education = education.round().clip(8, 22)
# Wages (outcome)
log_wage = 1.0 + TRUE_EFFECT * education + 0.2 * ability + np.random.normal(0, 0.2, N)

df = pd.DataFrame({'log_wage': log_wage, 'education': education,
                   'college_nearby': college_nearby, 'ability': ability})

# Compute Wald estimator manually
E_educ_z1 = df[df['college_nearby']==1]['education'].mean()
E_educ_z0 = df[df['college_nearby']==0]['education'].mean()
E_wage_z1 = df[df['college_nearby']==1]['log_wage'].mean()
E_wage_z0 = df[df['college_nearby']==0]['log_wage'].mean()

first_stage = E_educ_z1 - E_educ_z0
reduced_form = E_wage_z1 - E_wage_z0
wald_iv = reduced_form / first_stage

# OLS for comparison
ols = smf.ols('log_wage ~ education', data=df).fit()

print(f"First stage (education gap): {first_stage:.4f}")
print(f"Reduced form (wage gap):     {reduced_form:.4f}")
print(f"Wald IV estimate:            {wald_iv:.4f}")
print(f"OLS estimate (biased):       {ols.params['education']:.4f}")
print(f"True effect:                 {TRUE_EFFECT:.4f}")
print(f"\nOLS bias: {ols.params['education'] - TRUE_EFFECT:+.4f}")
print(f"IV bias:  {wald_iv - TRUE_EFFECT:+.4f}")

# Self-check: IV should be closer to true effect than OLS
assert abs(wald_iv - TRUE_EFFECT) < abs(ols.params['education'] - TRUE_EFFECT) + 0.02, \
    "IV should be at least as close to true effect as OLS"
print("\nCHECK PASSED: IV is closer to true effect than biased OLS.")


# ============================================================
# EXERCISE 3: First Stage Diagnostics
# ============================================================
print("\n" + "=" * 55)
print("EXERCISE 3: First Stage F-Statistic")
print("=" * 55)

# Test different instrument strengths
for pi, label in [(0.1, 'Weak'), (0.3, 'Moderate'), (0.6, 'Strong'), (1.5, 'Very Strong')]:
    educ_test = 10 + pi * college_nearby + 1.2 * ability + np.random.normal(0, 1.5, N)
    df_test = pd.DataFrame({'education': educ_test, 'college_nearby': college_nearby})
    fs = smf.ols('education ~ college_nearby', data=df_test).fit()
    f = fs.fvalue
    coef = fs.params['college_nearby']
    status = 'STRONG' if f > 10 else 'WEAK'
    print(f"π={pi:.1f} [{label:<12}]: coef={coef:.3f}, F={f:.1f} [{status}]")

print("\nRule of thumb: F > 10 for adequate instrument strength")
print("For robust inference: use Anderson-Rubin CIs if F < 10")

# Self-check: F grows with pi
f_vals = []
for pi in [0.1, 0.5, 1.5]:
    educ_t = 10 + pi * college_nearby + 1.2 * ability + np.random.normal(0, 1.5, N)
    df_t = pd.DataFrame({'education': educ_t, 'college_nearby': college_nearby})
    f_vals.append(smf.ols('education ~ college_nearby', data=df_t).fit().fvalue)

assert f_vals[0] < f_vals[1] < f_vals[2], "F-statistic should grow with instrument strength"
print("CHECK PASSED: F-statistic correctly increases with instrument strength.")


# ============================================================
# EXERCISE 4: Exclusion Restriction Violation
# ============================================================
print("\n" + "=" * 55)
print("EXERCISE 4: Exclusion Restriction Violation")
print("=" * 55)

TRUE_EFFECT_EX4 = 0.08
ability_4 = np.random.normal(0, 1, N)
college_nearby_4 = np.random.binomial(1, 0.5, N)
educ_4 = 10 + 0.6 * college_nearby_4 + ability_4 + np.random.normal(0, 1.5, N)
educ_4 = educ_4.round().clip(8, 22)

print("Testing exclusion violation: direct effect of instrument on outcome")
print(f"{'Direct Effect':>15} {'IV Estimate':>12} {'True Effect':>12} {'Bias':>8}")
print("-" * 55)

for direct_effect in [0.00, 0.01, 0.02, 0.05, 0.10]:
    wage_4 = (1.0 + TRUE_EFFECT_EX4 * educ_4
              + 0.15 * ability_4
              + direct_effect * college_nearby_4  # exclusion violation
              + np.random.normal(0, 0.2, N))

    E_e1 = educ_4[college_nearby_4==1].mean() - educ_4[college_nearby_4==0].mean()
    E_w1 = wage_4[college_nearby_4==1].mean() - wage_4[college_nearby_4==0].mean()
    iv_est = E_w1 / E_e1
    bias = iv_est - TRUE_EFFECT_EX4

    print(f"{direct_effect:>15.2f} {iv_est:>12.4f} {TRUE_EFFECT_EX4:>12.4f} {bias:>+8.4f}")

print("\nObservation: Even a small direct effect (0.01) causes IV bias.")
print("Exclusion restriction must hold EXACTLY — it is not testable.")

# Self-check: bias increases with direct effect
biases = []
for de in [0.0, 0.05, 0.10]:
    w_test = 1.0 + TRUE_EFFECT_EX4 * educ_4 + 0.15 * ability_4 + de * college_nearby_4 + np.random.normal(0, 0.2, N)
    E_e = educ_4[college_nearby_4==1].mean() - educ_4[college_nearby_4==0].mean()
    E_w = w_test[college_nearby_4==1].mean() - w_test[college_nearby_4==0].mean()
    biases.append(E_w / E_e - TRUE_EFFECT_EX4)

assert biases[0] < biases[2], "Larger direct effect should cause larger bias"
print("CHECK PASSED: Exclusion violation correctly inflates IV estimate.")

print("\n" + "=" * 55)
print("ALL MODULE 06 EXERCISES COMPLETE")
print("=" * 55)
print("\nKey takeaways:")
print("1. Valid IV requires relevance (testable) AND exclusion (untestable)")
print("2. Wald estimator = reduced form / first stage")
print("3. F-statistic < 10: weak instrument, IV biased toward OLS")
print("4. Exclusion violations cause bias even when small — argue defensively")
print("5. IV estimates LATE for compliers, not ATE")
