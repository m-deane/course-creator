# Project 01 — Guided Causal Study: Minimum Wage and Employment

**Course:** Causal Inference with CausalPy
**Type:** Guided portfolio project
**Estimated time:** 4–6 hours
**Output:** A complete causal analysis notebook with report

## Overview

You will replicate and extend the Card & Krueger (1994) minimum wage study using the skills from Modules 04–07. This guided project walks you through each stage of a complete causal analysis, from design selection to production-ready reporting.

The project uses a simulated dataset calibrated to match the original study's statistics. You will produce a structured causal report with effect sizes, robustness checks, and sensitivity analysis — the same outputs expected in a professional policy evaluation.

## Learning Objectives

1. Apply the design selection framework (Module 07) to select the appropriate causal design for a given research setting
2. Implement a complete DiD analysis with pre-trend testing, fixed effects, and clustered standard errors
3. Report results with interpretable effect sizes, robustness tables, and sensitivity analysis
4. Extend the primary analysis with a staggered DiD design for a multi-state minimum wage policy

## Dataset

The dataset simulates a Card & Krueger-style minimum wage study with extensions:

- **Primary study:** New Jersey vs Pennsylvania, 400 fast-food stores, two periods (1992)
- **Extension:** 50 states, 20 quarters, staggered minimum wage adoption (2010–2018)

You will generate these datasets using the provided simulation functions — no external data download required.

## Stage 1: Design Selection (1 hour)

### 1.1 Research Question

The research question: **Did the 1992 New Jersey minimum wage increase affect fast-food employment?**

Before estimating anything, document your design choice using the framework from Module 07.

Write a design selection memo answering these questions:

1. Do I have a clear intervention with a known time of change? *(Yes — April 1992)*
2. Does treatment assignment follow a threshold rule? *(No — state-level policy)*
3. Is there a valid comparison group? *(Yes — Pennsylvania fast-food stores)*
4. Are parallel trends plausible? *(Assess this in Stage 2)*

**Your deliverable:** A 1-paragraph design choice justification, including the estimand (ATT vs ATE), the comparison group, and the main identification assumption.

### 1.2 Data Generation

```python
import numpy as np
import pandas as pd
from typing import Tuple

N_STORES     = 200       # stores per state
TRUE_ATT     = 2.76      # Card & Krueger (1994) estimate
BASELINE_NJ  = 20.44
BASELINE_PA  = 23.33
NOISE_SD     = 5.0

def generate_card_krueger(n_stores=N_STORES, true_att=TRUE_ATT, seed=42):
    """Generate the primary two-period Card & Krueger simulation."""
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n_stores):
        store_fe = rng.normal(0, 2)
        for state, treated, base in [('NJ', 1, BASELINE_NJ), ('PA', 0, BASELINE_PA)]:
            for t in [0, 1]:
                y = (
                    base
                    + store_fe
                    + (true_att if (t == 1 and treated == 1) else 0)
                    + rng.normal(0, NOISE_SD)
                )
                records.append({
                    'store_id': f'{state}_{i:03d}',
                    'state': state,
                    'treated': treated,
                    'post': t,
                    'fte': y,
                })
    df = pd.DataFrame(records)
    df['post_treated'] = df['post'] * df['treated']
    return df

df = generate_card_krueger()
print(f'Dataset: {len(df)} rows, {df["store_id"].nunique()} stores')
```

## Stage 2: Pre-Analysis and Diagnostics (1 hour)

### 2.1 Exploratory Data Analysis

Compute and report:

1. Mean FTE by state and period (a 2x2 table)
2. The Wald DiD estimate: `(NJ_post - NJ_pre) - (PA_post - PA_pre)`
3. A visual DiD plot showing group means over time

### 2.2 Balance Check

Test whether NJ and PA stores are comparable in the pre-period:

1. T-test of mean FTE in the pre-period (NJ vs PA)
2. Comment: a significant difference does **not** invalidate parallel trends — it only means groups started at different levels

### 2.3 Pre-Trend Assessment

With only two periods, you cannot formally test parallel trends. Write a 2-sentence discussion of what additional data you would need, and what the risk is if pre-trends were violated.

## Stage 3: Primary Estimation (1 hour)

### 3.1 OLS DiD with Fixed Effects

Fit the primary specification:

```python
import statsmodels.formula.api as smf

model = smf.ols(
    'fte ~ post_treated + C(store_id) + C(post)',
    data=df
).fit(cov_type='cluster', cov_kwds={'groups': df['store_id']})

print(f"ATT: {model.params['post_treated']:.3f}")
print(f"SE:  {model.bse['post_treated']:.3f}")
print(f"95% CI: {model.conf_int().loc['post_treated'].values}")
```

### 3.2 Effect Sizes

Report all three effect size metrics:
- Cohen's d
- Percentage of pre-treatment baseline
- Percentile shift (using `scipy.stats.norm.cdf`)

Use the `compute_effect_sizes()` function from Module 07 Notebook 02.

## Stage 4: Robustness (1 hour)

### 4.1 Robustness Table

Run and report the following alternative specifications:

| Specification | Description |
|---------------|-------------|
| Primary (store FE, clustered SE) | The main estimate |
| No store FE | Drop unit fixed effects |
| HC1 robust SE | Heteroskedasticity-robust SEs (no clustering) |
| State-level clustering | Cluster at state rather than store level |
| Restricted to chain restaurants | Subset if data available |

Produce a forest plot of all specifications.

### 4.2 Sensitivity Analysis

Using the `sensitivity_to_pretrend()` function from Module 07 Notebook 02, show how the estimate changes under pre-trend violations of 0, 0.5, 1.0, 1.5, 2.0 FTE units.

Report: "Our conclusion holds as long as the parallel trends violation is less than X FTE."

## Stage 5: Staggered Extension (1 hour)

### 5.1 Staggered Data Generation

Extend to a 50-state, 20-quarter panel where states adopt minimum wage increases at different times (three cohorts):

```python
def generate_staggered_mw(n_states=50, n_periods=20, seed=42):
    """
    Three-cohort staggered DiD.
    Group 1 (15 states): treatment at t=7
    Group 2 (15 states): treatment at t=11
    Group 3 (20 states): never treated (control)
    """
    rng = np.random.default_rng(seed)
    records = []
    for state in range(n_states):
        if state < 15:
            treatment_time = 7
        elif state < 30:
            treatment_time = 11
        else:
            treatment_time = None   # never treated

        state_fe = rng.normal(0, 2)
        for t in range(1, n_periods + 1):
            treated = int(treatment_time is not None)
            post    = int(treatment_time is not None and t >= treatment_time)
            tau     = (2.0 + 0.3 * (t - (treatment_time or 0))) if post else 0.0
            y = (
                25 + 0.4 * t + state_fe
                + 3 * treated        # level difference
                + tau                # dynamic ATT
                + rng.normal(0, 2)
            )
            records.append({
                'state_id': state,
                'period':   t,
                'treated':  treated,
                'cohort':   treatment_time,
                'outcome':  y,
            })
    return pd.DataFrame(records)

stag_df = generate_staggered_mw()
```

### 5.2 TWFE vs Cohort-Specific ATT

1. Estimate TWFE DiD on the staggered dataset
2. Estimate cohort-specific ATT(g,t) for each cohort manually (using the clean control group approach from Module 04)
3. Compare: does TWFE produce an estimate that is a weighted average of the cohort ATTs? Is the weighting reasonable?

### 5.3 Event Study

Produce an event study plot (Module 04, Notebook 03 pattern) showing:
- Pre-treatment leads (should be near zero)
- Post-treatment dynamic effects (growing, as expected from the DGP)
- The Wald joint test for pre-treatment parallel trends

## Stage 6: Report (30 min)

Generate a complete report using `generate_causal_report()` from Module 07 Notebook 02.

The report must include:
- [ ] Design choice justification (Stage 1)
- [ ] Point estimate with SE and 95% CI
- [ ] Effect sizes (Cohen's d, % of baseline, percentile shift)
- [ ] Robustness table (Stage 4)
- [ ] Sensitivity analysis (Stage 4)
- [ ] Pre-trend assessment discussion (Stage 2)
- [ ] One-paragraph limitation section

## Deliverable

A single Jupyter notebook with:

1. **Section 1:** Design choice memo (markdown cell)
2. **Section 2:** EDA and balance check
3. **Section 3:** Primary DiD estimate with effect sizes
4. **Section 4:** Robustness table and forest plot
5. **Section 5:** Sensitivity analysis and plot
6. **Section 6:** Staggered DiD extension
7. **Section 7:** Automated report output

The notebook should run end-to-end (`Kernel → Restart & Run All`) without errors.

## Success Criteria

Your project is complete when:

1. The primary estimate is within 1 SE of the true ATT (2.76 FTE)
2. All five robustness specifications are reported
3. The sensitivity analysis identifies the violation threshold
4. The event study shows near-zero pre-treatment coefficients
5. The automated report outputs a complete text with all required elements

## Extension Challenge

If you want to go further:

- Implement the Callaway-Sant'Anna estimator from scratch using the formula in Module 04 Guide 02
- Compare its estimate to TWFE across the three cohorts
- Show a Goodman-Bacon decomposition (which cohort pairs drive the TWFE estimate?)

---

**Next Project:** [02 — Open-Ended Causal Study](../02_open_ended_causal_study/)

**Return to Course:** [Module 07 Production Pipelines](../../modules/module_07_production_pipelines/)
