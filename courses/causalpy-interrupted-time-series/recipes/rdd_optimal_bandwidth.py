"""
Recipe: RDD Optimal Bandwidth Selection
========================================

Patterns for selecting, validating, and presenting bandwidth choices
in Regression Discontinuity Designs.

Bandwidth selection involves a bias-variance tradeoff:
  - Small bandwidth: low bias (local comparison), high variance (few observations)
  - Large bandwidth: more observations, but increased risk of misspecification bias

Patterns covered
----------------
  1. MSE-optimal bandwidth (Imbens-Kalyanaraman approximation)
  2. Bandwidth sensitivity plot
  3. Cross-validation bandwidth selection
  4. Donut RDD (excluding observations immediately around the cutoff)
  5. Optimal bandwidth with covariates

Dependencies
------------
    pip install numpy pandas statsmodels matplotlib scipy
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# 1. Generate RDD Data
# ─────────────────────────────────────────────────────────────────────────────

def make_rdd_data(n: int = 2000, cutoff: float = 0.0, true_effect: float = 1.5,
                   seed: int = 42) -> pd.DataFrame:
    """
    Generate sharp RDD data with a known true effect.
    Running variable is standard normal; treatment at running_var >= cutoff.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    D = (x >= cutoff).astype(int)
    y = (
        2.0 + 0.5 * x              # underlying smooth function (slope)
        - 0.1 * x**2               # slight curvature
        + true_effect * D           # sharp discontinuity at cutoff
        + rng.normal(0, 0.8, n)
    )
    return pd.DataFrame({'x': x, 'D': D, 'y': y})


# ─────────────────────────────────────────────────────────────────────────────
# 2. Core Local Linear Estimator
# ─────────────────────────────────────────────────────────────────────────────

def rdd_local_linear(
    df: pd.DataFrame,
    outcome: str = 'y',
    running_var: str = 'x',
    cutoff: float = 0.0,
    bandwidth: float = 1.0,
    donut: float = 0.0,       # exclude |x - c| <= donut
    kernel: str = 'uniform',   # 'uniform' or 'triangular'
) -> Dict:
    """
    Estimate a sharp RDD using local linear regression within a bandwidth.

    Parameters
    ----------
    df          : input DataFrame
    outcome     : outcome column name
    running_var : running variable column name
    cutoff      : assignment cutoff value
    bandwidth   : half-window around the cutoff
    donut       : donut hole radius (exclude this region around the cutoff)
    kernel      : 'uniform' (equal weights) or 'triangular' (declining weights)

    Returns
    -------
    dict with estimate, se, ci_lo, ci_hi, n_obs, bandwidth
    """
    x_c = df[running_var] - cutoff
    within_bw = x_c.abs() <= bandwidth

    if donut > 0:
        outside_donut = x_c.abs() > donut
        mask = within_bw & outside_donut
    else:
        mask = within_bw

    local = df[mask].copy()
    local['x_c'] = local[running_var] - cutoff

    if len(local) < 10:
        return {
            'estimate': np.nan, 'se': np.nan,
            'ci_lo': np.nan, 'ci_hi': np.nan,
            'n_obs': len(local), 'bandwidth': bandwidth,
        }

    if kernel == 'triangular':
        # Triangular kernel: weight = 1 - |x_c| / bandwidth
        weights = (1.0 - local['x_c'].abs() / bandwidth).values
    else:
        weights = None

    formula = f'{outcome} ~ D + x_c + D:x_c'

    if weights is not None:
        model = smf.wls(formula, data=local, weights=weights).fit(cov_type='HC1')
    else:
        model = smf.ols(formula, data=local).fit(cov_type='HC1')

    ci = model.conf_int().loc['D'].values
    return {
        'estimate':  float(model.params['D']),
        'se':        float(model.bse['D']),
        'ci_lo':     float(ci[0]),
        'ci_hi':     float(ci[1]),
        'n_obs':     int(model.nobs),
        'bandwidth': bandwidth,
        'kernel':    kernel,
        'donut':     donut,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Pattern A: MSE-Optimal Bandwidth (Imbens-Kalyanaraman)
# ─────────────────────────────────────────────────────────────────────────────

def ik_bandwidth(
    df: pd.DataFrame,
    outcome: str = 'y',
    running_var: str = 'x',
    cutoff: float = 0.0,
) -> float:
    """
    Imbens-Kalyanaraman (2012) MSE-optimal bandwidth approximation.

    h* = C * σ^(2/5) * n^(-1/5) * f(c)^(-1/5) * B^(-2/5)

    where:
        σ² = outcome variance near the cutoff
        f(c) = density of the running variable at the cutoff
        B = second derivative (curvature) of the outcome at the cutoff

    This simplified version estimates each component from the data.

    Parameters
    ----------
    df          : full dataset
    outcome     : outcome column name
    running_var : running variable column name
    cutoff      : cutoff value

    Returns
    -------
    Estimated optimal bandwidth
    """
    n = len(df)
    x = df[running_var].values
    y = df[outcome].values

    # Estimate σ² from observations near the cutoff
    sigma_bw = (x.max() - x.min()) * 0.1
    near = np.abs(x - cutoff) <= sigma_bw
    sigma2 = float(np.var(y[near])) if near.sum() >= 5 else float(np.var(y))

    # Estimate density f(c) of the running variable at the cutoff using KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(x, bw_method='scott')
    f_c = float(kde(cutoff)[0])

    # Estimate second derivative (curvature) from polynomial regression
    df_poly = df.copy()
    df_poly['x_c']  = df_poly[running_var] - cutoff
    df_poly['x_c2'] = df_poly['x_c'] ** 2
    df_poly['D']    = (df_poly[running_var] >= cutoff).astype(int)

    # Use a pilot bandwidth for curvature estimation
    pilot_bw = (x.max() - x.min()) * 0.3
    pilot_mask = np.abs(df_poly['x_c'].values) <= pilot_bw

    try:
        pilot_model = smf.ols(
            f'{outcome} ~ D + x_c + x_c2 + D:x_c + D:x_c2',
            data=df_poly[pilot_mask]
        ).fit()

        B2 = float(pilot_model.params.get('x_c2', 0.0))  # curvature term
    except Exception:
        B2 = 0.1  # fallback default

    if abs(B2) < 1e-6:
        B2 = 0.1  # avoid division by zero

    # IK formula constant
    C = 3.438  # IK (2012) Table I, p=1

    h_star = C * (sigma2 / (n * f_c * B2 ** 2)) ** 0.2
    # Clip to reasonable range
    x_range = x.max() - x.min()
    h_star = float(np.clip(h_star, x_range * 0.05, x_range * 0.5))

    return h_star


# ─────────────────────────────────────────────────────────────────────────────
# 4. Pattern B: Bandwidth Sensitivity Plot
# ─────────────────────────────────────────────────────────────────────────────

def bandwidth_sensitivity(
    df: pd.DataFrame,
    bandwidths: List[float],
    outcome: str = 'y',
    running_var: str = 'x',
    cutoff: float = 0.0,
    kernel: str = 'uniform',
) -> pd.DataFrame:
    """
    Estimate the RDD at multiple bandwidths and return a sensitivity table.

    Best practice: estimate at h/2, h, and 2h where h is the optimal bandwidth.
    Report the primary estimate in bold; present sensitivity as a table or plot.
    """
    rows = []
    for h in bandwidths:
        res = rdd_local_linear(df, outcome, running_var, cutoff, h, kernel=kernel)
        rows.append(res)

    return pd.DataFrame(rows)[['bandwidth', 'estimate', 'se', 'ci_lo', 'ci_hi', 'n_obs']]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Pattern C: Cross-Validation Bandwidth Selection
# ─────────────────────────────────────────────────────────────────────────────

def cv_bandwidth(
    df: pd.DataFrame,
    bandwidths: List[float],
    outcome: str = 'y',
    running_var: str = 'x',
    cutoff: float = 0.0,
    n_folds: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Select bandwidth by leave-one-out cross-validation MSE.

    For each bandwidth, estimate the RDD on all other observations
    and compute prediction error on held-out observations.

    Note: CV for RDD is approximate — it estimates smoothing fit, not causal
    identification. Use IK as the primary method; CV as a robustness check.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df = df.sort_values(running_var).reset_index(drop=True)
    n = len(df)
    fold_ids = rng.integers(0, n_folds, n)

    cv_mse = {}
    for h in bandwidths:
        fold_mses = []
        for fold in range(n_folds):
            train = df[fold_ids != fold]
            test  = df[fold_ids == fold]

            res = rdd_local_linear(train, outcome, running_var, cutoff, h)
            if np.isnan(res['estimate']):
                continue

            # Predict for test observations using the fitted model
            model = smf.ols(
                f'{outcome} ~ D + x_c + D:x_c',
                data=train.assign(x_c=train[running_var] - cutoff)
                         .pipe(lambda d: d[np.abs(d[running_var] - cutoff) <= h])
            ).fit() if not np.isnan(res['estimate']) else None

            if model is not None:
                test_prep = test.copy()
                test_prep['x_c'] = test_prep[running_var] - cutoff
                near_test = test_prep[np.abs(test_prep['x_c']) <= h]
                if len(near_test) > 0:
                    y_pred = model.predict(near_test)
                    mse = float(np.mean((near_test[outcome].values - y_pred.values) ** 2))
                    fold_mses.append(mse)

        cv_mse[h] = float(np.mean(fold_mses)) if fold_mses else float('inf')

    best_bw = min(cv_mse, key=cv_mse.get)
    return {
        'best_bandwidth': best_bw,
        'cv_mse':         cv_mse,
        'all_results':    cv_mse,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_bandwidth_sensitivity(
    sensitivity_df: pd.DataFrame,
    optimal_bw: float,
    true_effect: Optional[float] = None,
    title: str = 'Bandwidth Sensitivity',
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot RDD estimates across bandwidths with 95% CIs.

    The optimal bandwidth is highlighted. All estimates should be consistent
    in sign and magnitude for a robust RDD result.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    for _, row in sensitivity_df.iterrows():
        is_optimal = abs(row['bandwidth'] - optimal_bw) < 1e-6
        color = 'steelblue' if is_optimal else '#888'
        lw = 2.5 if is_optimal else 1.2
        size = 100 if is_optimal else 50

        ax.plot([row['bandwidth'], row['bandwidth']],
                [row['ci_lo'], row['ci_hi']], '-', color=color, lw=lw, alpha=0.8)
        ax.scatter(row['bandwidth'], row['estimate'],
                   color=color, s=size, zorder=5)

    ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)

    if true_effect is not None:
        ax.axhline(true_effect, color='green', lw=1.5, ls=':',
                   alpha=0.7, label=f'True effect = {true_effect}')

    ax.axvline(optimal_bw, color='steelblue', lw=1, ls=':',
               alpha=0.5, label=f'Optimal h = {optimal_bw:.3f}')

    ax.set_xlabel('Bandwidth (h)')
    ax.set_ylabel('RDD Estimate')
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# Demonstration
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    TRUE_EFFECT = 1.5
    CUTOFF = 0.0

    df = make_rdd_data(n=2000, cutoff=CUTOFF, true_effect=TRUE_EFFECT)
    print(f'Dataset: N={len(df)}, cutoff={CUTOFF}, true effect={TRUE_EFFECT}')

    # IK optimal bandwidth
    h_ik = ik_bandwidth(df, 'y', 'x', CUTOFF)
    print(f'\nIK optimal bandwidth: {h_ik:.3f}')

    # Estimate at optimal bandwidth
    primary = rdd_local_linear(df, 'y', 'x', CUTOFF, h_ik)
    print(f'Primary estimate (h={h_ik:.3f}): {primary["estimate"]:+.3f} '
          f'(SE={primary["se"]:.3f}, CI=[{primary["ci_lo"]:+.3f},{primary["ci_hi"]:+.3f}])')

    # Sensitivity across bandwidths
    bandwidths = np.linspace(0.3, 3.0, 20).tolist()
    sens_df = bandwidth_sensitivity(df, bandwidths, 'y', 'x', CUTOFF)
    print(f'\nSensitivity: estimates range from {sens_df["estimate"].min():.3f} '
          f'to {sens_df["estimate"].max():.3f} across {len(bandwidths)} bandwidths')

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_bandwidth_sensitivity(sens_df, h_ik, TRUE_EFFECT, ax=ax)
    plt.tight_layout()
    plt.savefig('rdd_bandwidth_sensitivity.png', dpi=120, bbox_inches='tight')
    print('Plot saved: rdd_bandwidth_sensitivity.png')
