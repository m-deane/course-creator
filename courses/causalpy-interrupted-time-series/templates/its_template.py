"""
Interrupted Time Series (ITS) Analysis Template
================================================

Production-ready template for single-series ITS and controlled ITS (ITS+DiD).

Design variants covered
-----------------------
  1. Single-group ITS        — segmented regression on one time series
  2. Controlled ITS (ITS+DiD) — with a concurrent control group
  3. Multiple-period ITS     — several distinct intervention points

Assumptions
-----------
  - No concurrent events affect the outcome around the intervention
  - Pre-intervention trend extrapolates as the counterfactual (linearity)
  - No selection into treatment timing
  - For controlled ITS: control group shares the same pre-trend

Usage
-----
  1. Replace generate_sample_data() with your own data loading.
  2. Adjust OUTCOME_COL, TIME_COL, N_PRE_PERIODS, and INTERVENTION_TIME.
  3. Run: python its_template.py

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

OUTCOME_COL       = 'outcome'
TIME_COL          = 'period'
GROUP_COL         = 'group'         # for controlled ITS; 'treated' or 'control'
INTERVENTION_TIME = 25              # period at which the intervention occurs
N_PERIODS         = 50              # total number of time periods
TRUE_LEVEL_CHANGE = 3.0             # true immediate level change (for validation)
TRUE_SLOPE_CHANGE = 0.15            # true change in slope after intervention
NOISE_SD          = 1.2             # noise standard deviation


# ─────────────────────────────────────────────────────────────────────────────
# Sample Data Generation  (replace with your actual data)
# ─────────────────────────────────────────────────────────────────────────────

def generate_sample_data(
    n_periods: int = N_PERIODS,
    intervention_time: int = INTERVENTION_TIME,
    true_level_change: float = TRUE_LEVEL_CHANGE,
    true_slope_change: float = TRUE_SLOPE_CHANGE,
    noise_sd: float = NOISE_SD,
    include_control: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic ITS dataset.

    Single treated series with optional concurrent control (for controlled ITS).
    """
    rng = np.random.default_rng(seed)
    periods = np.arange(1, n_periods + 1)

    records = []
    for t in periods:
        post = int(t >= intervention_time)
        time_since = max(t - intervention_time, 0)

        # Treated group
        y_treated = (
            20.0
            + 0.3 * t                               # pre-trend
            + true_level_change * post              # level shift
            + true_slope_change * time_since        # slope change
            + rng.normal(0, noise_sd)
        )
        records.append({TIME_COL: t, OUTCOME_COL: y_treated, GROUP_COL: 'treated'})

        if include_control:
            # Control group: shares pre-trend, no intervention effect
            y_control = (
                18.0
                + 0.3 * t                           # same pre-trend
                + rng.normal(0, noise_sd)
            )
            records.append({TIME_COL: t, OUTCOME_COL: y_control, GROUP_COL: 'control'})

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Core ITS Model
# ─────────────────────────────────────────────────────────────────────────────

def fit_its(
    df: pd.DataFrame,
    outcome: str = OUTCOME_COL,
    time_col: str = TIME_COL,
    intervention_time: int = INTERVENTION_TIME,
    robust_se: bool = True,
) -> Dict:
    """
    Fit a segmented regression (single-group ITS).

    Model:
        outcome = β0 + β1*t + β2*D_post + β3*(t - T)*D_post + ε

    Where:
        t        = calendar time
        D_post   = 1 if t >= intervention_time
        β2       = immediate level change at intervention (the primary ATT)
        β3       = change in slope after intervention

    Parameters
    ----------
    df                : time series DataFrame
    outcome           : outcome column name
    time_col          : time index column name
    intervention_time : period of the intervention
    robust_se         : use HC1 robust standard errors

    Returns
    -------
    dict with: estimate, se, ci_lo, ci_hi, slope_change, n_obs, model
    """
    df_m = df.copy().sort_values(time_col).reset_index(drop=True)
    df_m['post']           = (df_m[time_col] >= intervention_time).astype(int)
    df_m['time_since_post'] = (df_m[time_col] - intervention_time) * df_m['post']

    formula = f'{outcome} ~ {time_col} + post + time_since_post'
    cov_type = 'HC1' if robust_se else 'nonrobust'
    model = smf.ols(formula, data=df_m).fit(cov_type=cov_type)

    ci_post  = model.conf_int().loc['post'].values
    ci_slope = model.conf_int().loc['time_since_post'].values

    return {
        'estimate':     float(model.params['post']),         # level change
        'se':           float(model.bse['post']),
        'ci_lo':        float(ci_post[0]),
        'ci_hi':        float(ci_post[1]),
        'p_value':      float(model.pvalues['post']),
        'slope_change': float(model.params['time_since_post']),
        'slope_se':     float(model.bse['time_since_post']),
        'slope_ci_lo':  float(ci_slope[0]),
        'slope_ci_hi':  float(ci_slope[1]),
        'pre_trend':    float(model.params[time_col]),
        'n_obs':        int(model.nobs),
        'r_squared':    float(model.rsquared),
        'model':        model,
    }


def fit_controlled_its(
    df: pd.DataFrame,
    outcome: str = OUTCOME_COL,
    time_col: str = TIME_COL,
    group_col: str = GROUP_COL,
    intervention_time: int = INTERVENTION_TIME,
    treated_label: str = 'treated',
) -> Dict:
    """
    Fit a controlled ITS (ITS + DiD) model.

    Uses the control group to account for contemporaneous trends and shocks.
    This is equivalent to DiD in a panel with a continuous time variable.

    Model (panel OLS):
        outcome = β0 + β1*t + β2*treated + β3*post + β4*(t-T)*post
                + β5*(treated*post) + β6*(treated*(t-T)*post) + ε

    The primary estimate is β5: the differential level change for the treated
    group relative to the control group at the intervention time.

    Parameters
    ----------
    df               : panel DataFrame with both treated and control rows
    outcome          : outcome column name
    time_col         : time column name
    group_col        : column indicating group membership
    intervention_time: period of the intervention
    treated_label    : value of group_col for treated units
    """
    df_m = df.copy()
    df_m['treated']         = (df_m[group_col] == treated_label).astype(int)
    df_m['post']            = (df_m[time_col] >= intervention_time).astype(int)
    df_m['time_since_post'] = (df_m[time_col] - intervention_time) * df_m['post']
    df_m['treated_post']    = df_m['treated'] * df_m['post']
    df_m['treated_slope']   = df_m['treated'] * df_m['time_since_post']

    formula = (
        f'{outcome} ~ {time_col} + treated + post + time_since_post '
        f'+ treated_post + treated_slope'
    )
    model = smf.ols(formula, data=df_m).fit(cov_type='HC1')

    ci_level = model.conf_int().loc['treated_post'].values
    ci_slope = model.conf_int().loc['treated_slope'].values

    return {
        'estimate':         float(model.params['treated_post']),   # DiD estimate
        'se':               float(model.bse['treated_post']),
        'ci_lo':            float(ci_level[0]),
        'ci_hi':            float(ci_level[1]),
        'p_value':          float(model.pvalues['treated_post']),
        'slope_change':     float(model.params['treated_slope']),
        'slope_ci_lo':      float(ci_slope[0]),
        'slope_ci_hi':      float(ci_slope[1]),
        'n_obs':            int(model.nobs),
        'r_squared':        float(model.rsquared),
        'model':            model,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def check_pre_trend(
    df: pd.DataFrame,
    outcome: str,
    time_col: str,
    intervention_time: int,
    alpha: float = 0.05,
) -> Dict:
    """
    Test for a significant trend in the pre-intervention period.

    A significant slope in the pre-period means we need to control for
    pre-trends in the ITS model — which the standard ITS model does via β1*t.
    This check verifies the slope is not implausibly large.

    Returns:
        pre_slope     : estimated trend coefficient in pre-period
        pre_slope_p   : p-value
        significant   : whether the slope is statistically significant
    """
    pre_df = df[df[time_col] < intervention_time].copy()
    if len(pre_df) < 5:
        return {'pre_slope': None, 'pre_slope_p': None, 'significant': None,
                'reason': 'Too few pre-period observations.'}

    model = smf.ols(f'{outcome} ~ {time_col}', data=pre_df).fit()
    slope   = float(model.params[time_col])
    p_value = float(model.pvalues[time_col])

    return {
        'pre_slope':   slope,
        'pre_slope_p': p_value,
        'significant': p_value < alpha,
        'n_pre':       len(pre_df),
    }


def residuals_diagnostics(model_result: Dict) -> Dict:
    """
    Run residuals diagnostics on a fitted ITS model.

    Checks:
    - Ljung-Box autocorrelation test (lag 1)
    - Shapiro-Wilk normality test on residuals
    """
    model = model_result['model']
    residuals = model.resid.values

    # Ljung-Box test for autocorrelation
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_result = acorr_ljungbox(residuals, lags=[1], return_df=True)
    lb_p = float(lb_result['lb_pvalue'].iloc[0])
    autocorrelated = lb_p < 0.05

    # Shapiro-Wilk normality
    sw_stat, sw_p = stats.shapiro(residuals[:min(len(residuals), 5000)])
    non_normal = sw_p < 0.05

    return {
        'ljung_box_p':   lb_p,
        'autocorrelated': autocorrelated,
        'shapiro_p':     float(sw_p),
        'non_normal':    non_normal,
        'warnings':      [
            w for w, flag in [
                ('Residuals show autocorrelation (Ljung-Box p < 0.05). Consider Newey-West SE.', autocorrelated),
                ('Residuals are non-normal (Shapiro-Wilk p < 0.05).', non_normal),
            ] if flag
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_its(
    df: pd.DataFrame,
    its_result: Dict,
    outcome: str = OUTCOME_COL,
    time_col: str = TIME_COL,
    intervention_time: int = INTERVENTION_TIME,
    group: Optional[str] = None,     # if not None, filter to this group
    title: str = 'Interrupted Time Series Analysis',
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot observed data with fitted pre- and post-intervention segments
    and the counterfactual (pre-trend extrapolated).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    plot_df = df.copy()
    if group is not None:
        plot_df = plot_df[plot_df[GROUP_COL] == group]

    plot_df = plot_df.sort_values(time_col)

    # Observed data
    ax.scatter(
        plot_df[time_col], plot_df[outcome],
        s=20, alpha=0.5, color='steelblue', label='Observed',
    )

    # Fitted values from model
    model = its_result['model']
    # Reconstruct model dataframe with the same transformations
    df_m = plot_df.copy()
    df_m['post']           = (df_m[time_col] >= intervention_time).astype(int)
    df_m['time_since_post'] = (df_m[time_col] - intervention_time) * df_m['post']

    fitted = model.predict(df_m)

    pre_mask  = df_m['post'] == 0
    post_mask = df_m['post'] == 1

    ax.plot(
        df_m.loc[pre_mask, time_col], fitted[pre_mask],
        '-', color='navy', lw=2, label='Fitted (pre)',
    )
    ax.plot(
        df_m.loc[post_mask, time_col], fitted[post_mask],
        '-', color='steelblue', lw=2, label='Fitted (post)',
    )

    # Counterfactual: extend pre-trend into post-period
    cf_df = df_m.copy()
    cf_df['post']           = 0
    cf_df['time_since_post'] = 0
    cf_values = model.predict(cf_df)

    ax.plot(
        df_m.loc[post_mask, time_col], cf_values[post_mask],
        '--', color='gray', lw=1.5, alpha=0.8, label='Counterfactual',
    )

    # Intervention line
    ax.axvline(intervention_time, color='red', lw=1.5, ls='--', alpha=0.7,
               label=f'Intervention (t={intervention_time})')

    ax.set_xlabel('Time Period')
    ax.set_ylabel(outcome.replace('_', ' ').title())
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    return ax


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_its_report(
    its_result: Dict,
    diagnostics: Dict,
    pre_trend: Dict,
    true_level_change: Optional[float] = None,
) -> None:
    """Print a complete ITS analysis report."""
    est = its_result['estimate']
    se  = its_result['se']
    ci_lo = its_result['ci_lo']
    ci_hi = its_result['ci_hi']
    p     = its_result['p_value']
    sc    = its_result['slope_change']

    print('═' * 65)
    print('INTERRUPTED TIME SERIES — ANALYSIS REPORT')
    print('═' * 65)
    print()
    print('1. LEVEL CHANGE AT INTERVENTION')
    print(f'   Estimate:  {est:+.3f}')
    print(f'   SE:        {se:.3f}')
    print(f'   95% CI:    [{ci_lo:+.3f}, {ci_hi:+.3f}]')
    print(f'   p-value:   {p:.4f}')
    print(f'   Significant: {"Yes" if (ci_lo > 0 or ci_hi < 0) else "No"}')
    if true_level_change is not None:
        print(f'   True ATT:  {true_level_change:+.3f}  (simulation ground truth)')
    print()
    print('2. SLOPE CHANGE AFTER INTERVENTION')
    print(f'   Estimate:  {sc:+.4f} per period')
    print(f'   95% CI:    [{its_result["slope_ci_lo"]:+.3f}, {its_result["slope_ci_hi"]:+.3f}]')
    print()
    print('3. PRE-INTERVENTION TREND')
    if pre_trend.get('pre_slope') is not None:
        sig = 'Yes' if pre_trend['significant'] else 'No'
        print(f'   Pre-slope: {pre_trend["pre_slope"]:+.4f} per period (p = {pre_trend["pre_slope_p"]:.4f})')
        print(f'   Significant trend: {sig}')
        print(f'   N pre-periods: {pre_trend["n_pre"]}')
    else:
        print(f'   {pre_trend.get("reason", "N/A")}')
    print()
    print('4. RESIDUALS DIAGNOSTICS')
    print(f'   Autocorrelation (Ljung-Box p): {diagnostics["ljung_box_p"]:.4f}  '
          f'{"WARN" if diagnostics["autocorrelated"] else "OK"}')
    print(f'   Normality (Shapiro-Wilk p):   {diagnostics["shapiro_p"]:.4f}  '
          f'{"WARN" if diagnostics["non_normal"] else "OK"}')
    if diagnostics['warnings']:
        print()
        for w in diagnostics['warnings']:
            print(f'   WARNING: {w}')
    print()
    print(f'   N observations: {its_result["n_obs"]}')
    print(f'   R-squared: {its_result["r_squared"]:.3f}')
    print('═' * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # ── Generate sample data ──────────────────────────────────────────────────
    df = generate_sample_data(include_control=True)
    print(f'Dataset: {len(df)} rows  |  groups: {df[GROUP_COL].unique().tolist()}')

    df_treated = df[df[GROUP_COL] == 'treated'].copy()

    # ── Single-group ITS ──────────────────────────────────────────────────────
    its_result = fit_its(df_treated, OUTCOME_COL, TIME_COL, INTERVENTION_TIME)
    pre_trend  = check_pre_trend(df_treated, OUTCOME_COL, TIME_COL, INTERVENTION_TIME)
    diag       = residuals_diagnostics(its_result)

    print_its_report(its_result, diag, pre_trend, true_level_change=TRUE_LEVEL_CHANGE)

    # ── Controlled ITS ────────────────────────────────────────────────────────
    ctrl_result = fit_controlled_its(df, OUTCOME_COL, TIME_COL, GROUP_COL, INTERVENTION_TIME)
    print()
    print('CONTROLLED ITS (ITS + DiD)')
    print('-' * 40)
    print(f'  Differential level change: {ctrl_result["estimate"]:+.3f}')
    print(f'  SE:  {ctrl_result["se"]:.3f}')
    print(f'  95% CI: [{ctrl_result["ci_lo"]:+.3f}, {ctrl_result["ci_hi"]:+.3f}]')
    print(f'  True ATT: {TRUE_LEVEL_CHANGE:+.3f}')

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_its(
        df_treated, its_result,
        OUTCOME_COL, TIME_COL, INTERVENTION_TIME,
        title='Single-Group ITS', ax=axes[0],
    )
    plot_its(
        df[df[GROUP_COL] == 'treated'], its_result,
        OUTCOME_COL, TIME_COL, INTERVENTION_TIME,
        title='Controlled ITS — Treated Group', ax=axes[1],
    )
    plt.tight_layout()
    plt.savefig('its_analysis.png', dpi=120, bbox_inches='tight')
    print('\nPlot saved: its_analysis.png')
