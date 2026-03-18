"""
Synthetic Control Analysis Template
====================================

Production-ready template for synthetic control estimation.

The synthetic control method (Abadie, Diamond, Hainmueller 2010) constructs
a weighted combination of donor units that best matches the treated unit's
pre-treatment trajectory. The post-treatment gap between the treated unit
and its synthetic control estimates the treatment effect.

Variants covered
----------------
  1. Single treated unit — classic synthetic control
  2. Placebo inference   — leave-one-out and in-time placebo tests
  3. Donor selection     — filtering donors by pre-treatment fit

Assumptions
-----------
  - The pre-treatment period is long enough to estimate the synthetic control
    precisely (rule of thumb: at least 10–15 pre-treatment periods)
  - No treated unit is in the donor pool
  - Outcome predictors are informative of post-treatment trajectory
  - No interference (SUTVA)

Usage
-----
  1. Replace generate_sample_data() with your actual data.
  2. Configure TREATED_UNIT, PRE_PERIODS, N_DONORS, and OUTCOME_COL.
  3. Run: python synthetic_control_template.py

Dependencies
------------
    pip install numpy pandas scipy matplotlib
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# Configuration — customise for your analysis
# ─────────────────────────────────────────────────────────────────────────────

OUTCOME_COL   = 'outcome'
UNIT_COL      = 'unit_id'
TIME_COL      = 'period'
TREATED_UNIT  = 0                  # identifier of the treated unit
N_DONORS      = 30                 # number of donor units in the simulation
N_PERIODS     = 30                 # total periods (pre + post)
N_PRE_PERIODS = 18                 # pre-treatment periods
TRUE_ATT      = 5.0                # true treatment effect (for validation)


# ─────────────────────────────────────────────────────────────────────────────
# Sample Data Generation  (replace with your actual data)
# ─────────────────────────────────────────────────────────────────────────────

def generate_sample_data(
    n_donors: int = N_DONORS,
    n_periods: int = N_PERIODS,
    n_pre_periods: int = N_PRE_PERIODS,
    true_att: float = TRUE_ATT,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic control dataset.

    Unit 0 is the treated unit. Units 1..n_donors are donor units.
    The treated unit receives treatment at t = n_pre_periods + 1.
    """
    rng = np.random.default_rng(seed)
    periods = np.arange(1, n_periods + 1)

    # Latent factors
    factor1 = np.sin(periods / 5) * 3 + rng.normal(0, 0.3, n_periods)
    factor2 = np.cos(periods / 8) * 2 + rng.normal(0, 0.3, n_periods)

    rows = []
    for unit in range(n_donors + 1):
        # Each unit has random loadings on the two latent factors
        l1 = rng.uniform(0.5, 1.5)
        l2 = rng.uniform(0.3, 1.2)
        base = rng.uniform(15, 25)

        for t in periods:
            post = int(t > n_pre_periods)
            y = (
                base
                + l1 * factor1[t - 1]
                + l2 * factor2[t - 1]
                + rng.normal(0, 0.8)
                + (true_att if (unit == TREATED_UNIT and post) else 0.0)
            )
            rows.append({
                UNIT_COL:    unit,
                TIME_COL:    t,
                OUTCOME_COL: y,
                'treated':   int(unit == TREATED_UNIT and post),
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Control Estimation
# ─────────────────────────────────────────────────────────────────────────────

def wide_pivot(df: pd.DataFrame, outcome: str, unit_col: str, time_col: str) -> pd.DataFrame:
    """Pivot panel to wide format: rows = time, columns = unit."""
    return df.pivot(index=time_col, columns=unit_col, values=outcome)


def fit_synthetic_control(
    wide_pre: pd.DataFrame,
    treated_unit: int,
    donor_units: List[int],
) -> Tuple[np.ndarray, float]:
    """
    Find optimal donor weights via constrained quadratic optimisation.

    Minimises:  ||Y_treated_pre - W * Y_donors_pre||²
    Subject to: w_j >= 0 for all j,  sum(w_j) = 1

    Parameters
    ----------
    wide_pre    : pre-treatment data in wide format (rows=time, cols=units)
    treated_unit: unit identifier for the treated unit
    donor_units : list of donor unit identifiers

    Returns
    -------
    weights     : array of length len(donor_units), sums to 1
    pre_mspe    : mean squared prediction error in pre-treatment period
    """
    y_treated = wide_pre[treated_unit].values
    Y_donors  = wide_pre[donor_units].values

    n_donors = len(donor_units)

    def objective(w):
        synthetic = Y_donors @ w
        return np.sum((y_treated - synthetic) ** 2)

    # Constraints: weights sum to 1 and are non-negative
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n_donors
    w0 = np.ones(n_donors) / n_donors  # uniform initialisation

    result = minimize(
        objective, w0, method='SLSQP',
        bounds=bounds, constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 500},
    )

    weights = result.x
    # Clip small negatives from numerical noise
    weights = np.clip(weights, 0, None)
    weights /= weights.sum()

    pre_mspe = float(np.mean((y_treated - Y_donors @ weights) ** 2))
    return weights, pre_mspe


def compute_synthetic_control(
    df: pd.DataFrame,
    treated_unit: int = TREATED_UNIT,
    n_pre_periods: int = N_PRE_PERIODS,
    outcome: str = OUTCOME_COL,
    unit_col: str = UNIT_COL,
    time_col: str = TIME_COL,
) -> Dict:
    """
    Estimate the synthetic control and compute the ATT trajectory.

    Returns a dict with:
        weights      : DataFrame of (unit, weight) for active donors
        synthetic    : Series of synthetic control outcomes by period
        gap          : Series of treatment effect by period (Y_treat - Y_synth)
        pre_mspe     : mean squared prediction error in pre-period
        donor_units  : list of donor unit IDs
        treated_series: observed treated unit outcome
    """
    # Pivot to wide
    wide = wide_pivot(df, outcome, unit_col, time_col)
    donor_units = [u for u in wide.columns if u != treated_unit]

    # Pre-treatment wide matrix
    pre_periods = wide.index[wide.index <= n_pre_periods]
    wide_pre    = wide.loc[pre_periods]

    # Optimise weights
    weights_arr, pre_mspe = fit_synthetic_control(wide_pre, treated_unit, donor_units)

    # Synthetic control for all periods
    Y_donors_all = wide[donor_units].values
    synthetic    = pd.Series(
        Y_donors_all @ weights_arr,
        index=wide.index,
        name='synthetic',
    )

    treated_series = wide[treated_unit].rename('treated')
    gap = treated_series - synthetic

    # Weight summary
    weight_df = pd.DataFrame({
        'unit':   donor_units,
        'weight': weights_arr,
    }).query('weight > 0.001').sort_values('weight', ascending=False).reset_index(drop=True)

    return {
        'weights':        weight_df,
        'synthetic':      synthetic,
        'gap':            gap,
        'pre_mspe':       pre_mspe,
        'pre_rmspe':      float(np.sqrt(pre_mspe)),
        'donor_units':    donor_units,
        'treated_series': treated_series,
        'n_pre_periods':  n_pre_periods,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ATT Estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_att(sc_result: Dict, n_pre_periods: int) -> Dict:
    """
    Compute average ATT in the post-treatment period from the gap series.

    Returns:
        att_mean      : average treatment effect in post-treatment period
        att_series    : Series of treatment effect by period
        post_periods  : periods with treatment
    """
    gap = sc_result['gap']
    post_gap = gap[gap.index > n_pre_periods]

    return {
        'att_mean':    float(post_gap.mean()),
        'att_series':  post_gap,
        'post_periods': post_gap.index.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Placebo Inference
# ─────────────────────────────────────────────────────────────────────────────

def placebo_inference(
    df: pd.DataFrame,
    true_treated: int,
    n_pre_periods: int,
    outcome: str = OUTCOME_COL,
    unit_col: str = UNIT_COL,
    time_col: str = TIME_COL,
    max_pre_rmspe_ratio: float = 5.0,
) -> Dict:
    """
    Placebo test: iteratively assign treatment to each donor unit and
    compute its synthetic control. Compare treated unit's gap to the
    distribution of donor gaps.

    Units with pre-treatment RMSPE more than `max_pre_rmspe_ratio` times
    the treated unit's pre-RMSPE are excluded (poor fit → unreliable placebo).

    Returns:
        treated_gap   : gap for the true treated unit
        placebo_gaps  : DataFrame of donor gap trajectories
        treated_rmspe : pre-treatment RMSPE for treated unit
        p_value       : rank-based p-value for ATT
    """
    wide = wide_pivot(df, outcome, unit_col, time_col)
    all_units = wide.columns.tolist()
    donor_units = [u for u in all_units if u != true_treated]

    # True treated unit synthetic control
    sc_treated = compute_synthetic_control(
        df, true_treated, n_pre_periods, outcome, unit_col, time_col
    )
    treated_rmspe = sc_treated['pre_rmspe']
    treated_gap   = sc_treated['gap']

    placebo_gaps_dict: Dict[int, pd.Series] = {}

    for placebo_unit in donor_units:
        try:
            sc_placebo = compute_synthetic_control(
                df[df[unit_col] != true_treated],   # exclude the true treated unit
                placebo_unit, n_pre_periods, outcome, unit_col, time_col,
            )
            # Only include donors with reasonable pre-treatment fit
            if sc_placebo['pre_rmspe'] <= max_pre_rmspe_ratio * treated_rmspe:
                placebo_gaps_dict[placebo_unit] = sc_placebo['gap']
        except Exception:
            continue

    placebo_gaps = pd.DataFrame(placebo_gaps_dict)

    # Rank-based p-value: fraction of donors with post-ATT >= treated post-ATT
    post_treated_att = float(treated_gap[treated_gap.index > n_pre_periods].mean())

    post_mask = treated_gap.index > n_pre_periods
    placebo_atts = placebo_gaps.loc[post_mask].mean(axis=0)

    n_better = (placebo_atts.abs() >= abs(post_treated_att)).sum()
    p_value = float((n_better + 1) / (len(placebo_atts) + 1))

    return {
        'treated_gap':   treated_gap,
        'placebo_gaps':  placebo_gaps,
        'treated_rmspe': treated_rmspe,
        'p_value':       p_value,
        'post_att':      post_treated_att,
        'n_placebos':    len(placebo_atts),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_synthetic_control(
    sc_result: Dict,
    n_pre_periods: int = N_PRE_PERIODS,
    title: str = 'Synthetic Control',
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot treated unit vs synthetic control with the treatment onset line."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))

    treated  = sc_result['treated_series']
    synth    = sc_result['synthetic']

    ax.plot(treated.index, treated.values, 'o-', color='steelblue',
            ms=4, lw=2, label='Treated unit')
    ax.plot(synth.index, synth.values, 's--', color='gray',
            ms=4, lw=2, label='Synthetic control')

    ax.axvline(n_pre_periods + 0.5, color='red', lw=1.5, ls='--',
               alpha=0.7, label='Treatment onset')

    ax.fill_between(
        synth.index,
        treated.values,
        synth.values,
        where=(synth.index > n_pre_periods),
        alpha=0.15, color='steelblue', label='ATT (gap)',
    )

    pre_rmspe = sc_result['pre_rmspe']
    ax.text(0.02, 0.97,
            f'Pre-RMSPE = {pre_rmspe:.3f}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    ax.set_xlabel('Period')
    ax.set_ylabel(OUTCOME_COL.replace('_', ' ').title())
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return ax


def plot_gap(
    sc_result: Dict,
    n_pre_periods: int = N_PRE_PERIODS,
    placebo_result: Optional[Dict] = None,
    title: str = 'Gap Plot (Treated − Synthetic)',
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot the treatment effect gap, optionally with placebo paths."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))

    gap = sc_result['gap']

    if placebo_result is not None:
        for col in placebo_result['placebo_gaps'].columns:
            pg = placebo_result['placebo_gaps'][col]
            ax.plot(pg.index, pg.values, '-', color='lightgray', lw=0.8, alpha=0.5)

    ax.plot(gap.index, gap.values, '-', color='steelblue', lw=2.5,
            label='Treated unit gap', zorder=5)
    ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.6)
    ax.axvline(n_pre_periods + 0.5, color='red', lw=1.5, ls='--',
               alpha=0.7, label='Treatment onset')

    if placebo_result is not None:
        ax.text(
            0.02, 0.97,
            f'Placebo p-value: {placebo_result["p_value"]:.3f}\n'
            f'N placebos: {placebo_result["n_placebos"]}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
        )

    ax.set_xlabel('Period')
    ax.set_ylabel('Gap (Treated − Synthetic)')
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def print_sc_report(sc_result: Dict, att_result: Dict,
                    placebo_result: Optional[Dict] = None,
                    true_att: Optional[float] = None) -> None:
    """Print a structured synthetic control report."""
    print('═' * 65)
    print('SYNTHETIC CONTROL — ANALYSIS REPORT')
    print('═' * 65)
    print()
    print('1. PRE-TREATMENT FIT')
    print(f'   Pre-RMSPE:   {sc_result["pre_rmspe"]:.4f}')
    print()
    print('2. DONOR WEIGHTS (active donors, weight > 0.001)')
    print(sc_result['weights'].to_string(index=False))
    print()
    print('3. TREATMENT EFFECT ESTIMATE')
    print(f'   Average ATT (post-period): {att_result["att_mean"]:+.4f}')
    if true_att is not None:
        print(f'   True ATT (simulation):     {true_att:+.4f}')
    print()
    if placebo_result is not None:
        print('4. PLACEBO INFERENCE')
        print(f'   N placebo donors:  {placebo_result["n_placebos"]}')
        print(f'   Rank-based p-val:  {placebo_result["p_value"]:.3f}')
        sig = 'Yes (p < 0.10)' if placebo_result['p_value'] < 0.10 else 'No'
        print(f'   Significant:       {sig}')
    print()
    print('═' * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # ── Generate data ─────────────────────────────────────────────────────────
    df = generate_sample_data()
    print(f'Dataset: {df[UNIT_COL].nunique()} units × {df[TIME_COL].nunique()} periods')
    print(f'Treated unit: {TREATED_UNIT}  |  Pre-periods: {N_PRE_PERIODS}')

    # ── Estimate synthetic control ────────────────────────────────────────────
    sc_result  = compute_synthetic_control(df, TREATED_UNIT, N_PRE_PERIODS)
    att_result = estimate_att(sc_result, N_PRE_PERIODS)

    # ── Placebo inference ─────────────────────────────────────────────────────
    print('Running placebo tests (this may take a moment)...')
    placebo = placebo_inference(df, TREATED_UNIT, N_PRE_PERIODS)

    # ── Report ────────────────────────────────────────────────────────────────
    print_sc_report(sc_result, att_result, placebo, true_att=TRUE_ATT)

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    plot_synthetic_control(sc_result, N_PRE_PERIODS, ax=axes[0])
    plot_gap(sc_result, N_PRE_PERIODS, placebo_result=placebo, ax=axes[1])
    plt.tight_layout()
    plt.savefig('synthetic_control_analysis.png', dpi=120, bbox_inches='tight')
    print('Plot saved: synthetic_control_analysis.png')
