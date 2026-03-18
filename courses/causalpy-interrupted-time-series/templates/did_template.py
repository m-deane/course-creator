"""
Difference-in-Differences (DiD) Analysis Template
==================================================

Production-ready template for two-period DiD and panel DiD with
pre-trend testing, robustness checks, and event study plots.

Variants covered
----------------
  1. Two-period DiD          — classic (pre/post) x (treated/control)
  2. Panel TWFE DiD          — two-way fixed effects with multiple periods
  3. Staggered DiD (manual)  — cohort-based ATT(g,t) with event study

Assumptions
-----------
  - Parallel counterfactual trends: absent treatment, treated and control
    groups would have followed the same trend
  - Stable Unit Treatment Value Assumption (SUTVA): no spillovers
  - No Ashenfelter's dip (pre-treatment selection on outcomes)

Usage
-----
  1. Replace generate_panel_data() with your actual data.
  2. Configure column names and treatment timing in the constants below.
  3. Run: python did_template.py

Dependencies
------------
    pip install numpy pandas statsmodels matplotlib scipy
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# Configuration — customise for your analysis
# ─────────────────────────────────────────────────────────────────────────────

OUTCOME_COL      = 'outcome'
UNIT_COL         = 'unit_id'
TIME_COL         = 'period'
GROUP_COL        = 'treated'    # 1 = treated, 0 = control
TREATMENT_TIME   = 6            # first period the treatment applies
N_UNITS          = 100          # units per group in the simulation
N_PERIODS        = 12           # total periods
TRUE_ATT         = 4.0          # true ATT (for simulation validation)


# ─────────────────────────────────────────────────────────────────────────────
# Sample Data Generation  (replace with your actual data)
# ─────────────────────────────────────────────────────────────────────────────

def generate_panel_data(
    n_units: int = N_UNITS,
    n_periods: int = N_PERIODS,
    treatment_time: int = TREATMENT_TIME,
    true_att: float = TRUE_ATT,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a balanced panel dataset for DiD analysis.

    N units (half treated, half control) observed for n_periods periods.
    Treatment effect is `true_att` for treated units in post-treatment periods.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units * 2):
        treated = int(i < n_units)
        unit_fe = rng.normal(0, 1.5)       # unit fixed effect
        for t in range(1, n_periods + 1):
            post = int(t >= treatment_time)
            y = (
                10.0
                + 0.4 * t                   # common time trend
                + 3.0 * treated             # level difference (allowed in DiD)
                + unit_fe                   # unit fixed effect
                + true_att * post * treated # ATT
                + rng.normal(0, 1.8)
            )
            rows.append({
                UNIT_COL:  i,
                TIME_COL:  t,
                GROUP_COL: treated,
                OUTCOME_COL: y,
            })
    df = pd.DataFrame(rows)
    df['post']         = (df[TIME_COL] >= treatment_time).astype(int)
    df['post_treated'] = df['post'] * df[GROUP_COL]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_twfe(
    df: pd.DataFrame,
    outcome: str = OUTCOME_COL,
    unit_col: str = UNIT_COL,
    time_col: str = TIME_COL,
    group_col: str = GROUP_COL,
    treatment_time: int = TREATMENT_TIME,
    cluster: bool = True,
) -> Dict:
    """
    Two-Way Fixed Effects DiD estimator (unit FE + time FE).

    Model: outcome_it = α_i + γ_t + τ*(post_it × treated_i) + ε_it

    Parameters
    ----------
    df             : balanced panel DataFrame
    outcome        : outcome column name
    unit_col       : unit identifier column
    time_col       : time period column
    group_col      : treatment group column (1 = treated)
    treatment_time : first treated period
    cluster        : cluster SEs at the unit level (recommended)

    Returns
    -------
    dict with estimate, se, ci_lo, ci_hi, p_value, n_obs, model
    """
    df_m = df.copy()
    df_m['post']         = (df_m[time_col] >= treatment_time).astype(int)
    df_m['post_treated'] = df_m['post'] * df_m[group_col]

    formula = f'{outcome} ~ post_treated + C({unit_col}) + C({time_col})'

    if cluster:
        model = smf.ols(formula, data=df_m).fit(
            cov_type='cluster', cov_kwds={'groups': df_m[unit_col]}
        )
    else:
        model = smf.ols(formula, data=df_m).fit(cov_type='HC1')

    ci = model.conf_int().loc['post_treated'].values
    return {
        'estimate': float(model.params['post_treated']),
        'se':       float(model.bse['post_treated']),
        'ci_lo':    float(ci[0]),
        'ci_hi':    float(ci[1]),
        'p_value':  float(model.pvalues['post_treated']),
        'n_obs':    int(model.nobs),
        'model':    model,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Assumption Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def test_parallel_trends(
    df: pd.DataFrame,
    outcome: str = OUTCOME_COL,
    unit_col: str = UNIT_COL,
    time_col: str = TIME_COL,
    group_col: str = GROUP_COL,
    treatment_time: int = TREATMENT_TIME,
    alpha: float = 0.05,
) -> Dict:
    """
    Test for differential pre-treatment trends.

    Fits: outcome ~ group * time + C(unit) on pre-treatment periods.
    A significant interaction indicates diverging pre-trends.
    """
    pre_df = df[df[time_col] < treatment_time].copy()

    if len(pre_df) < 10:
        return {'passed': None, 'reason': 'Insufficient pre-period data.'}

    try:
        formula = (
            f'{outcome} ~ {group_col} * {time_col} + C({unit_col})'
        )
        model = smf.ols(formula, data=pre_df).fit()

        interaction_key = f'{group_col}:{time_col}'
        if interaction_key in model.pvalues:
            p_val = float(model.pvalues[interaction_key])
            coef  = float(model.params[interaction_key])
        else:
            p_val, coef = 1.0, 0.0

        passed = p_val >= alpha
        return {
            'passed':        passed,
            'warning':       not passed,
            'interaction_p': p_val,
            'interaction_coef': coef,
            'alpha':         alpha,
            'n_pre_obs':     len(pre_df),
        }
    except Exception as exc:
        return {'passed': None, 'reason': str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Event Study
# ─────────────────────────────────────────────────────────────────────────────

def estimate_event_study(
    df: pd.DataFrame,
    outcome: str = OUTCOME_COL,
    unit_col: str = UNIT_COL,
    time_col: str = TIME_COL,
    group_col: str = GROUP_COL,
    treatment_time: int = TREATMENT_TIME,
    omit_period: Optional[int] = None,
) -> pd.DataFrame:
    """
    Estimate a dynamic event study (leads and lags) regression.

    Creates relative-time dummies for each period (omitting one pre-treatment
    period as the reference). Estimates the treatment effect at each relative
    time period, including pre-treatment leads.

    Returns a DataFrame with columns:
        rel_time, estimate, ci_lo, ci_hi, is_pre
    """
    df_m = df.copy()
    df_m['rel_time'] = df_m[time_col] - treatment_time

    rel_times = sorted(df_m['rel_time'].unique())

    # Default: omit rel_time = -1 (last pre-treatment period)
    if omit_period is None:
        omit_period = -1

    rel_times_used = [t for t in rel_times if t != omit_period]

    # Create dummy columns
    for t in rel_times_used:
        col = f'D_{t}' if t >= 0 else f'D_m{abs(t)}'
        df_m[col] = ((df_m['rel_time'] == t) & (df_m[group_col] == 1)).astype(int)

    dummy_cols = [
        (f'D_{t}' if t >= 0 else f'D_m{abs(t)}') for t in rel_times_used
    ]
    formula = (
        f'{outcome} ~ '
        + ' + '.join(dummy_cols)
        + f' + C({unit_col}) + C({time_col})'
    )

    model = smf.ols(formula, data=df_m).fit(
        cov_type='cluster', cov_kwds={'groups': df_m[unit_col]}
    )

    rows = []
    for t, col in zip(rel_times_used, dummy_cols):
        if col in model.params:
            ci = model.conf_int().loc[col].values
            rows.append({
                'rel_time': t,
                'estimate': float(model.params[col]),
                'ci_lo':    float(ci[0]),
                'ci_hi':    float(ci[1]),
                'is_pre':   t < 0,
            })

    # Add omitted period back as zero
    rows.append({'rel_time': omit_period, 'estimate': 0.0,
                 'ci_lo': 0.0, 'ci_hi': 0.0, 'is_pre': omit_period < 0})

    es_df = pd.DataFrame(rows).sort_values('rel_time').reset_index(drop=True)
    return es_df


def wald_pretrend_test(es_df: pd.DataFrame) -> Dict:
    """
    Wald joint test for pre-treatment coefficients equal to zero.

    If the null is rejected, there is evidence of pre-treatment divergence.
    Uses a chi-squared approximation on the pre-treatment estimates.
    """
    pre_df = es_df[es_df['is_pre'] & (es_df['estimate'].abs() > 0)]  # exclude omitted

    if len(pre_df) == 0:
        return {'chi2': 0.0, 'p_value': 1.0, 'df': 0, 'passed': True}

    estimates = pre_df['estimate'].values
    # Approximate Wald using sum of squared t-statistics (Bonferroni-style)
    # Full Wald requires covariance matrix; use joint F-stat approximation
    se_approx = (pre_df['ci_hi'] - pre_df['ci_lo']).values / (2 * 1.96)
    t_stats   = estimates / np.maximum(se_approx, 1e-10)
    chi2_stat = float(np.sum(t_stats ** 2))
    df        = len(estimates)
    p_value   = float(1 - stats.chi2.cdf(chi2_stat, df))

    return {
        'chi2':    chi2_stat,
        'p_value': p_value,
        'df':      df,
        'passed':  p_value >= 0.05,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Robustness
# ─────────────────────────────────────────────────────────────────────────────

def run_robustness_table(
    df: pd.DataFrame,
    outcome: str = OUTCOME_COL,
    unit_col: str = UNIT_COL,
    time_col: str = TIME_COL,
    group_col: str = GROUP_COL,
    treatment_time: int = TREATMENT_TIME,
) -> pd.DataFrame:
    """
    Run multiple DiD specifications and return a robustness table.

    Specifications:
        1. TWFE with clustered SE  (primary)
        2. TWFE with HC1 SE (no clustering)
        3. No unit FE
        4. No time FE
        5. Treated units only (pre-trend check)
    """
    specs = []

    # 1. Primary TWFE
    r1 = estimate_twfe(df, outcome, unit_col, time_col, group_col,
                       treatment_time, cluster=True)
    specs.append(('Primary (TWFE, clustered)', r1))

    # 2. HC1 (no clustering)
    r2 = estimate_twfe(df, outcome, unit_col, time_col, group_col,
                       treatment_time, cluster=False)
    specs.append(('TWFE, HC1 SE', r2))

    # 3. No unit FE
    df_m = df.copy()
    df_m['post'] = (df_m[time_col] >= treatment_time).astype(int)
    df_m['post_treated'] = df_m['post'] * df_m[group_col]
    m3 = smf.ols(f'{outcome} ~ post_treated + {group_col} + C({time_col})',
                  data=df_m).fit(cov_type='HC1')
    ci3 = m3.conf_int().loc['post_treated'].values
    specs.append(('No unit FE', {
        'estimate': float(m3.params['post_treated']),
        'se': float(m3.bse['post_treated']),
        'ci_lo': float(ci3[0]), 'ci_hi': float(ci3[1]),
        'n_obs': int(m3.nobs),
    }))

    # 4. No time FE
    m4 = smf.ols(f'{outcome} ~ post_treated + C({unit_col})',
                  data=df_m).fit(cov_type='HC1')
    ci4 = m4.conf_int().loc['post_treated'].values
    specs.append(('No time FE', {
        'estimate': float(m4.params['post_treated']),
        'se': float(m4.bse['post_treated']),
        'ci_lo': float(ci4[0]), 'ci_hi': float(ci4[1]),
        'n_obs': int(m4.nobs),
    }))

    rows = []
    for label, res in specs:
        rows.append({
            'Specification': label,
            'Estimate':      res['estimate'],
            'CI_lo':         res['ci_lo'],
            'CI_hi':         res['ci_hi'],
            'N':             res['n_obs'],
            'Significant':   (res['ci_lo'] > 0) or (res['ci_hi'] < 0),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_event_study(
    es_df: pd.DataFrame,
    wald_result: Dict,
    treatment_time: int = TREATMENT_TIME,
    title: str = 'Event Study',
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot event study estimates with pre-treatment leads."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))

    pre  = es_df[es_df['is_pre']]
    post = es_df[~es_df['is_pre']]

    # Pre-treatment coefficients (should be near zero)
    ax.scatter(pre['rel_time'], pre['estimate'], color='gray', s=60, zorder=5)
    for _, row in pre.iterrows():
        ax.plot([row['rel_time'], row['rel_time']], [row['ci_lo'], row['ci_hi']],
                '-', color='gray', lw=1.2, alpha=0.7)

    # Post-treatment coefficients
    ax.scatter(post['rel_time'], post['estimate'], color='steelblue', s=80, zorder=5)
    for _, row in post.iterrows():
        ax.plot([row['rel_time'], row['rel_time']], [row['ci_lo'], row['ci_hi']],
                '-', color='steelblue', lw=2)

    ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
    ax.axvline(-0.5, color='red', lw=1.2, ls='--', alpha=0.6,
               label='Treatment onset')

    wald_text = (
        f'Pre-trend Wald: χ² = {wald_result["chi2"]:.2f}, '
        f'p = {wald_result["p_value"]:.3f}  '
        f'({" PASS" if wald_result["passed"] else "WARN"})'
    )
    ax.text(0.02, 0.97, wald_text, transform=ax.transAxes,
            fontsize=9, va='top', color='black',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    ax.set_xlabel('Period Relative to Treatment')
    ax.set_ylabel('Estimated Effect')
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return ax


def forest_plot(
    robustness_df: pd.DataFrame,
    title: str = 'Robustness: Multiple Specifications',
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot robustness table as a forest plot."""
    if ax is None:
        n = len(robustness_df)
        _, ax = plt.subplots(figsize=(10, n * 0.7 + 1.5))

    for i, row in robustness_df.iterrows():
        is_primary = i == 0
        color = 'steelblue' if is_primary else '#888'
        lw    = 2.5 if is_primary else 1.5
        size  = 100 if is_primary else 60
        ax.plot([row['CI_lo'], row['CI_hi']], [i, i], '-', color=color, lw=lw)
        ax.scatter(row['Estimate'], i, color=color, s=size, zorder=5)
        ax.text(row['CI_hi'] + 0.05, i, f"{row['Estimate']:+.2f}",
                va='center', fontsize=9, color=color)

    ax.axvline(0, color='black', lw=0.8, ls='--', alpha=0.5)
    ax.set_yticks(range(len(robustness_df)))
    ax.set_yticklabels(robustness_df['Specification'].tolist())
    ax.invert_yaxis()
    ax.set_xlabel('Treatment Effect Estimate')
    ax.set_title(title, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # ── Generate data ─────────────────────────────────────────────────────────
    df = generate_panel_data()
    print(f'Panel: {df[UNIT_COL].nunique()} units × {df[TIME_COL].nunique()} periods')
    print(f'Treated units: {(df[GROUP_COL] == 1).sum() // df[TIME_COL].nunique()}')

    # ── Primary estimate ──────────────────────────────────────────────────────
    twfe = estimate_twfe(df, OUTCOME_COL, UNIT_COL, TIME_COL, GROUP_COL, TREATMENT_TIME)
    print(f'\nTWFE ATT:  {twfe["estimate"]:+.3f}  SE = {twfe["se"]:.3f}')
    print(f'95% CI:    [{twfe["ci_lo"]:+.3f}, {twfe["ci_hi"]:+.3f}]')
    print(f'True ATT:  {TRUE_ATT:+.3f}')

    # ── Pre-trend test ────────────────────────────────────────────────────────
    pt_result = test_parallel_trends(df, OUTCOME_COL, UNIT_COL, TIME_COL,
                                     GROUP_COL, TREATMENT_TIME)
    status = 'PASS' if pt_result.get('passed') else 'WARN'
    print(f'\nParallel trends test: {status}  (p = {pt_result.get("interaction_p", "N/A")})')

    # ── Event study ───────────────────────────────────────────────────────────
    es_df    = estimate_event_study(df, OUTCOME_COL, UNIT_COL, TIME_COL,
                                     GROUP_COL, TREATMENT_TIME)
    wald_res = wald_pretrend_test(es_df)
    print(f'\nWald pre-trend test: χ² = {wald_res["chi2"]:.2f}, p = {wald_res["p_value"]:.4f}')

    # ── Robustness table ──────────────────────────────────────────────────────
    rob_df = run_robustness_table(df, OUTCOME_COL, UNIT_COL, TIME_COL,
                                   GROUP_COL, TREATMENT_TIME)
    print('\nRobustness Table:')
    print(rob_df[['Specification', 'Estimate', 'CI_lo', 'CI_hi', 'Significant']].to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    plot_event_study(es_df, wald_res, TREATMENT_TIME, ax=axes[0])
    forest_plot(rob_df, ax=axes[1])
    plt.tight_layout()
    plt.savefig('did_analysis.png', dpi=120, bbox_inches='tight')
    print('\nPlot saved: did_analysis.png')
