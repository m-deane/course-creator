"""
Recipe: Synthetic Control Donor Unit Selection
===============================================

Selecting the right donor pool is critical for synthetic control validity.
A poor donor pool produces a synthetic control that fits the pre-treatment
period by construction but fails as a counterfactual.

Patterns covered
----------------
  1. Pre-treatment RMSPE filter — exclude donors that can't track the treated unit
  2. Economic/domain criteria filter — exclude units that received treatment
  3. Covariate similarity filter — exclude outliers on key characteristics
  4. Stepwise donor addition — build donor pool incrementally
  5. Leave-one-out sensitivity — check how each donor affects the estimate

When to use each
----------------
  - RMSPE filter:    always — a donor that can't fit the pre-period is uninformative
  - Domain filter:   when you know which units received the same treatment
  - Covariate filter: when donor units differ substantially on structural characteristics
  - Stepwise:        when you have many candidate donors and want to understand their role
  - Leave-one-out:   to assess whether a single donor is driving the estimate

Dependencies
------------
    pip install numpy pandas scipy matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 1. Generate Sample Data
# ─────────────────────────────────────────────────────────────────────────────

def make_sc_data(n_donors: int = 40, n_periods: int = 30, n_pre: int = 20,
                 true_att: float = 5.0, seed: int = 42) -> pd.DataFrame:
    """
    Generate panel data for synthetic control examples.
    Unit 0 is the treated unit. Some donors are systematically different.
    """
    rng = np.random.default_rng(seed)
    periods = np.arange(1, n_periods + 1)

    factor1 = np.sin(periods / 4) * 4 + rng.normal(0, 0.2, n_periods)
    factor2 = np.cos(periods / 6) * 3 + rng.normal(0, 0.2, n_periods)

    rows = []
    for unit in range(n_donors + 1):
        base = rng.uniform(18, 30)
        l1   = rng.uniform(0.5, 1.5)

        # Make some donors clearly different (outliers in their level)
        if unit in range(35, 41):
            base = rng.uniform(50, 70)  # outlier units

        l2 = rng.uniform(0.3, 1.2)
        covariate_size = rng.normal(100, 20)

        for t in periods:
            post = int(t > n_pre)
            y = (
                base
                + l1 * factor1[t - 1]
                + l2 * factor2[t - 1]
                + rng.normal(0, 0.8)
                + (true_att if (unit == 0 and post) else 0.0)
            )
            rows.append({
                'unit': unit,
                'period': t,
                'outcome': y,
                'covariate_size': covariate_size,
                'treated': int(unit == 0 and post),
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Core SC Solver (shared utility)
# ─────────────────────────────────────────────────────────────────────────────

def fit_sc_weights(Y_treated_pre: np.ndarray, Y_donors_pre: np.ndarray) -> Tuple[np.ndarray, float]:
    """Find optimal synthetic control weights via constrained QP."""
    n_donors = Y_donors_pre.shape[1]

    def objective(w):
        return np.sum((Y_treated_pre - Y_donors_pre @ w) ** 2)

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n_donors
    w0 = np.ones(n_donors) / n_donors

    result = minimize(objective, w0, method='SLSQP',
                      bounds=bounds, constraints=constraints,
                      options={'ftol': 1e-12, 'maxiter': 500})

    weights = np.clip(result.x, 0, None)
    weights /= weights.sum()
    rmspe = float(np.sqrt(np.mean((Y_treated_pre - Y_donors_pre @ weights) ** 2)))
    return weights, rmspe


def compute_donor_rmspe(
    wide_pre: pd.DataFrame,
    treated_unit: int,
    donor_list: List[int],
) -> Dict[int, float]:
    """
    Compute individual donor RMSPE: how well can each single donor
    replicate the treated unit in the pre-period?

    A donor with very high RMSPE when used alone is likely a poor match.
    """
    Y_treated = wide_pre[treated_unit].values
    donor_rmspe = {}

    for donor in donor_list:
        Y_donor = wide_pre[donor].values.reshape(-1, 1)
        # Single-donor synthetic control (weight = 1 for this donor)
        rmspe = float(np.sqrt(np.mean((Y_treated - Y_donor.flatten()) ** 2)))
        donor_rmspe[donor] = rmspe

    return donor_rmspe


# ─────────────────────────────────────────────────────────────────────────────
# 3. Pattern A: Pre-Treatment RMSPE Filter
# ─────────────────────────────────────────────────────────────────────────────

def filter_donors_by_rmspe(
    df: pd.DataFrame,
    treated_unit: int,
    n_pre: int,
    unit_col: str = 'unit',
    time_col: str = 'period',
    outcome_col: str = 'outcome',
    max_rmspe_multiple: float = 3.0,
) -> List[int]:
    """
    Exclude donor units with pre-treatment RMSPE more than
    `max_rmspe_multiple` times the median donor RMSPE.

    Donors that are structurally very different from the treated unit
    cannot form a plausible counterfactual.

    Parameters
    ----------
    max_rmspe_multiple : exclude donors whose individual RMSPE > median * multiple

    Returns
    -------
    List of retained donor unit IDs.
    """
    wide = df[df[time_col] <= n_pre].pivot(
        index=time_col, columns=unit_col, values=outcome_col
    )
    all_donors = [u for u in wide.columns if u != treated_unit]

    donor_rmspe = compute_donor_rmspe(wide, treated_unit, all_donors)

    rmspe_values = list(donor_rmspe.values())
    median_rmspe = np.median(rmspe_values)
    threshold = max_rmspe_multiple * median_rmspe

    retained = [u for u, r in donor_rmspe.items() if r <= threshold]

    n_excluded = len(all_donors) - len(retained)
    print(f'RMSPE filter: retained {len(retained)}/{len(all_donors)} donors '
          f'(excluded {n_excluded} with RMSPE > {threshold:.2f})')

    return retained


# ─────────────────────────────────────────────────────────────────────────────
# 4. Pattern B: Domain/Exclusion Criteria Filter
# ─────────────────────────────────────────────────────────────────────────────

def filter_donors_by_exclusion(
    all_donors: List[int],
    excluded_units: List[int],
    reason: str = 'received same treatment',
) -> List[int]:
    """
    Remove specific units from the donor pool based on domain knowledge.

    Common exclusion reasons:
    - Unit received the same treatment (contamination)
    - Unit is in the same region and may have spillovers
    - Unit has data quality issues in a specific period

    Parameters
    ----------
    all_donors     : full list of potential donor IDs
    excluded_units : list of unit IDs to remove
    reason         : description of why these units are excluded (for logging)

    Returns
    -------
    Filtered list of donor unit IDs.
    """
    retained = [u for u in all_donors if u not in excluded_units]
    n_excluded = len(all_donors) - len(retained)
    print(f'Domain filter: removed {n_excluded} units ({reason}). '
          f'Retained {len(retained)} donors.')
    return retained


# ─────────────────────────────────────────────────────────────────────────────
# 5. Pattern C: Covariate Similarity Filter
# ─────────────────────────────────────────────────────────────────────────────

def filter_donors_by_covariate(
    df: pd.DataFrame,
    treated_unit: int,
    covariate_col: str,
    unit_col: str = 'unit',
    n_sd_threshold: float = 2.0,
) -> List[int]:
    """
    Exclude donor units whose covariate value is more than `n_sd_threshold`
    standard deviations from the treated unit's value.

    This prevents the algorithm from constructing a synthetic control
    using fundamentally different types of units.

    Parameters
    ----------
    covariate_col    : column name of the pre-treatment characteristic
    n_sd_threshold   : exclude donors beyond this many SDs from treated value
    """
    unit_covars = df.drop_duplicates(unit_col)[[unit_col, covariate_col]].set_index(unit_col)

    treated_val = float(unit_covars.loc[treated_unit, covariate_col])
    all_donors  = [u for u in unit_covars.index if u != treated_unit]

    donor_vals = unit_covars.loc[all_donors, covariate_col].values
    covar_sd   = float(np.std(donor_vals))
    threshold  = n_sd_threshold * covar_sd

    retained = [
        u for u in all_donors
        if abs(float(unit_covars.loc[u, covariate_col]) - treated_val) <= threshold
    ]

    n_excluded = len(all_donors) - len(retained)
    print(f'Covariate filter ({covariate_col}): treated={treated_val:.1f}, '
          f'excluded {n_excluded} outliers beyond {n_sd_threshold} SD. '
          f'Retained {len(retained)} donors.')

    return retained


# ─────────────────────────────────────────────────────────────────────────────
# 6. Pattern D: Leave-One-Out Sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def leave_one_out_sensitivity(
    df: pd.DataFrame,
    treated_unit: int,
    donor_units: List[int],
    n_pre: int,
    unit_col: str = 'unit',
    time_col: str = 'period',
    outcome_col: str = 'outcome',
) -> pd.DataFrame:
    """
    Estimate synthetic control leaving out one donor at a time.

    If a single donor drives the estimate (large change when removed),
    the result is fragile. Robust results show stable estimates
    regardless of which donor is omitted.

    Returns a DataFrame with:
        excluded_unit : which donor was left out
        att_estimate  : average post-treatment gap with that donor excluded
        pre_rmspe     : pre-treatment fit with that donor excluded
    """
    wide = df.pivot(index=time_col, columns=unit_col, values=outcome_col)
    pre_mask  = wide.index <= n_pre
    post_mask = wide.index > n_pre

    Y_treated_pre  = wide.loc[pre_mask, treated_unit].values
    Y_treated_post = wide.loc[post_mask, treated_unit].values

    rows = []
    for excl in donor_units:
        remaining = [u for u in donor_units if u != excl]
        if len(remaining) == 0:
            continue

        Y_donors_pre  = wide.loc[pre_mask, remaining].values
        Y_donors_post = wide.loc[post_mask, remaining].values

        weights, rmspe = fit_sc_weights(Y_treated_pre, Y_donors_pre)
        synth_post = Y_donors_post @ weights
        att_mean = float(np.mean(Y_treated_post - synth_post))

        rows.append({
            'excluded_unit': excl,
            'att_estimate':  att_mean,
            'pre_rmspe':     rmspe,
            'n_donors':      len(remaining),
        })

    return pd.DataFrame(rows).sort_values('att_estimate').reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Demonstration
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    TRUE_ATT = 5.0
    N_PRE    = 20

    df = make_sc_data(n_donors=40, n_pre=N_PRE, true_att=TRUE_ATT)
    all_donors = [u for u in df['unit'].unique() if u != 0]

    print(f'Total candidate donors: {len(all_donors)}')
    print()

    # Step 1: Domain exclusion (units 38-40 received same policy)
    donors_domain = filter_donors_by_exclusion(
        all_donors, excluded_units=[38, 39, 40],
        reason='received same minimum wage increase'
    )

    # Step 2: Covariate filter
    donors_covar = filter_donors_by_covariate(
        df, treated_unit=0, covariate_col='covariate_size', n_sd_threshold=2.0
    )

    # Step 3: RMSPE filter
    donors_rmspe = filter_donors_by_rmspe(df, treated_unit=0, n_pre=N_PRE)

    # Final donor pool: intersection of all filters
    final_donors = list(
        set(donors_domain) & set(donors_covar) & set(donors_rmspe)
    )
    print(f'\nFinal donor pool: {len(final_donors)} units')

    # Leave-one-out on final pool (subset for speed)
    loo_sample = final_donors[:15]
    print(f'\nRunning leave-one-out sensitivity on {len(loo_sample)} donors...')
    loo_df = leave_one_out_sensitivity(df, 0, loo_sample, N_PRE)

    print('\nLeave-One-Out ATT estimates:')
    print(f'  Mean: {loo_df["att_estimate"].mean():.3f}')
    print(f'  Std:  {loo_df["att_estimate"].std():.3f}')
    print(f'  Min:  {loo_df["att_estimate"].min():.3f}')
    print(f'  Max:  {loo_df["att_estimate"].max():.3f}')
    print(f'  True ATT: {TRUE_ATT}')

    # Plot ATT stability
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(loo_df)), loo_df['att_estimate'], color='steelblue', alpha=0.7)
    ax.axhline(TRUE_ATT, color='red', ls='--', lw=1.5, label=f'True ATT = {TRUE_ATT}')
    ax.axhline(loo_df['att_estimate'].mean(), color='navy', ls='-', lw=1.5,
               label=f'Mean LOO ATT = {loo_df["att_estimate"].mean():.2f}')
    ax.set_xlabel('Excluded Donor (index)')
    ax.set_ylabel('ATT Estimate')
    ax.set_title('Leave-One-Out Sensitivity: How Stable is the ATT?', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('sc_donor_loo.png', dpi=120, bbox_inches='tight')
    print('\nPlot saved: sc_donor_loo.png')
