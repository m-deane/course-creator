"""
Recipe: Publication-Quality Counterfactual Plots
==================================================

Copy-paste patterns for producing clean, informative counterfactual plots
for each causal design: ITS, DiD, RDD, and Synthetic Control.

Standards applied
-----------------
  - Colour scheme: steelblue (treated), gray (control/synthetic/counterfactual)
  - Consistent annotation: intervention line, ATT arrow, HDI shading
  - Figure sizes: 10–12 inches wide for presentations; 7–8 for papers
  - All labels: remove underscores, title-case

Patterns covered
----------------
  1. ITS: observed vs counterfactual with uncertainty band
  2. DiD: pre/post group means with parallel trends counterfactual
  3. RDD: scatter with local linear fits and gap annotation
  4. SC: treated vs synthetic with gap plot
  5. Multi-panel dashboard (all four in one figure)

Dependencies
------------
    pip install numpy pandas statsmodels matplotlib scipy
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, Tuple, Dict, List

plt.rcParams.update({
    'figure.dpi': 120,
    'font.size':  11,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

BLUE    = 'steelblue'
GRAY    = '#888888'
RED     = '#CC4444'
GREEN   = '#339966'
NAVY    = '#1F3E6A'
LTYEL   = '#FFFACD'  # light yellow for annotation boxes


# ─────────────────────────────────────────────────────────────────────────────
# 1. ITS Counterfactual Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_its_counterfactual(
    df: pd.DataFrame,
    outcome: str,
    time_col: str,
    intervention_time: int,
    model,              # fitted statsmodels OLS model
    title: str = 'Interrupted Time Series',
    ylabel: str = None,
    show_uncertainty: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    ITS plot: observed data, fitted pre/post segments, and counterfactual.

    If the model has a prediction interval (predict() with alpha argument),
    the uncertainty band is shown. Works with any statsmodels OLS result.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))

    df_plot = df.sort_values(time_col).copy()
    df_plot['post']           = (df_plot[time_col] >= intervention_time).astype(int)
    df_plot['time_since_post'] = np.maximum(df_plot[time_col] - intervention_time, 0)

    pre_mask  = df_plot['post'] == 0
    post_mask = df_plot['post'] == 1

    # Observed data
    ax.scatter(df_plot[time_col], df_plot[outcome],
               s=18, alpha=0.55, color=BLUE, zorder=3, label='Observed')

    # Fitted values
    fitted = model.fittedvalues.reindex(df_plot.index)
    ax.plot(df_plot.loc[pre_mask, time_col], fitted[pre_mask],
            '-', color=NAVY, lw=2.5, label='Fitted (pre-intervention)', zorder=4)
    ax.plot(df_plot.loc[post_mask, time_col], fitted[post_mask],
            '-', color=BLUE, lw=2.5, label='Fitted (post-intervention)', zorder=4)

    # Counterfactual (zero out post and time_since_post)
    df_cf = df_plot.copy()
    df_cf['post']           = 0
    df_cf['time_since_post'] = 0
    cf_vals = model.predict(df_cf)
    ax.plot(df_plot.loc[post_mask, time_col], cf_vals[post_mask],
            '--', color=GRAY, lw=1.8, alpha=0.85, label='Counterfactual', zorder=4)

    if show_uncertainty:
        # Prediction interval from the model (95%)
        try:
            pred_df = model.get_prediction(df_cf)
            pred_summary = pred_df.summary_frame(alpha=0.05)
            ax.fill_between(
                df_plot.loc[post_mask, time_col],
                pred_summary.loc[post_mask, 'obs_ci_lower'],
                pred_summary.loc[post_mask, 'obs_ci_upper'],
                color=GRAY, alpha=0.15, label='95% prediction interval',
            )
        except Exception:
            pass

    # Intervention line
    ax.axvline(intervention_time, color=RED, lw=1.5, ls='--', alpha=0.7,
               label=f'Intervention (t={intervention_time})')

    ax.set_xlabel(time_col.replace('_', ' ').title())
    ax.set_ylabel((ylabel or outcome).replace('_', ' ').title())
    ax.set_title(title, fontweight='bold', pad=10)
    ax.legend(fontsize=8.5, loc='upper left')
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# 2. DiD Counterfactual Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_did_counterfactual(
    df: pd.DataFrame,
    outcome: str,
    post_col: str,
    treated_col: str,
    treated_label: str = '1',
    control_label: str = '0',
    title: str = 'Difference-in-Differences',
    pre_label: str = 'Pre',
    post_label: str = 'Post',
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    DiD plot: group means before and after, with parallel counterfactual.

    Annotates the ATT as the gap at the post-period between the treated
    group's observed outcome and the counterfactual trajectory.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    # Compute group means by period
    means = df.groupby([post_col, treated_col])[outcome].mean()

    try:
        t_pre  = float(means[(0, int(treated_label))])
        t_post = float(means[(1, int(treated_label))])
        c_pre  = float(means[(0, int(control_label))])
        c_post = float(means[(1, int(control_label))])
    except Exception:
        t_pre  = float(df[(df[post_col] == 0) & (df[treated_col] == int(treated_label))][outcome].mean())
        t_post = float(df[(df[post_col] == 1) & (df[treated_col] == int(treated_label))][outcome].mean())
        c_pre  = float(df[(df[post_col] == 0) & (df[treated_col] == int(control_label))][outcome].mean())
        c_post = float(df[(df[post_col] == 1) & (df[treated_col] == int(control_label))][outcome].mean())

    times = [0, 1]
    labels = [pre_label, post_label]

    # Treated group
    ax.plot(times, [t_pre, t_post], 'o-', color=BLUE, lw=2.5, ms=10,
            label='Treated group')
    # Control group
    ax.plot(times, [c_pre, c_post], 's--', color=GRAY, lw=2.5, ms=10,
            label='Control group')
    # Counterfactual
    cf_post = t_pre + (c_post - c_pre)
    ax.plot([0, 1], [t_pre, cf_post], 'o:', color=BLUE, lw=1.8, ms=10, alpha=0.55,
            label='Treated counterfactual')

    # ATT annotation arrow
    att = t_post - cf_post
    ax.annotate('', xy=(1, t_post), xytext=(1, cf_post),
                arrowprops=dict(arrowstyle='<->', color=GREEN, lw=2.0))
    ax.text(1.04, (t_post + cf_post) / 2,
            f'ATT = {att:+.2f}',
            va='center', color=GREEN, fontsize=10, fontweight='bold')

    ax.set_xticks(times)
    ax.set_xticklabels(labels)
    ax.set_ylabel(outcome.replace('_', ' ').title())
    ax.set_title(title, fontweight='bold', pad=10)
    ax.legend(fontsize=9, loc='upper left')
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# 3. RDD Discontinuity Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_rdd_discontinuity(
    df: pd.DataFrame,
    outcome: str,
    running_var: str,
    cutoff: float,
    bandwidth: float,
    n_bins: int = 30,
    title: str = 'Regression Discontinuity',
    ylabel: str = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    RDD scatter plot with bin-averaged means and local linear fits.

    Shows the discontinuity at the cutoff clearly.
    Bins are constructed separately on each side of the cutoff.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))

    local = df[np.abs(df[running_var] - cutoff) <= bandwidth].copy()
    local['x_c'] = local[running_var] - cutoff
    local['D']   = (local[running_var] >= cutoff).astype(int)

    # Bin scatter
    left  = local[local[running_var] < cutoff].copy()
    right = local[local[running_var] >= cutoff].copy()

    def bin_means(side_df, n_bins):
        bins = pd.cut(side_df[running_var], bins=n_bins)
        gp   = side_df.groupby(bins, observed=True)[outcome].mean()
        midpoints = gp.index.map(lambda iv: iv.mid)
        return midpoints.values, gp.values

    lx, ly = bin_means(left, n_bins // 2)
    rx, ry = bin_means(right, n_bins // 2)

    ax.scatter(lx, ly, s=40, color=GRAY, alpha=0.8, zorder=4)
    ax.scatter(rx, ry, s=40, color=BLUE, alpha=0.8, zorder=4)

    # Local linear fits
    dense_left  = np.linspace(local[running_var].min(), cutoff, 100)
    dense_right = np.linspace(cutoff, local[running_var].max(), 100)

    def llr_predict(x_new, side_df, side='left'):
        m = smf.ols(f'{outcome} ~ x_c + D:x_c + D', data=local).fit()
        x_c_new = x_new - cutoff
        D_new   = np.ones_like(x_new) if side == 'right' else np.zeros_like(x_new)
        pred_df = pd.DataFrame({'x_c': x_c_new, 'D': D_new})
        return m.predict(pred_df)

    try:
        ax.plot(dense_left,  llr_predict(dense_left, left, 'left'),
                '-', color=GRAY, lw=2)
        ax.plot(dense_right, llr_predict(dense_right, right, 'right'),
                '-', color=BLUE, lw=2)
    except Exception:
        pass

    # Cutoff line
    ax.axvline(cutoff, color=RED, lw=1.5, ls='--', alpha=0.7,
               label=f'Cutoff = {cutoff}')

    # Annotate jump
    try:
        m = smf.ols(f'{outcome} ~ x_c + D:x_c + D', data=local).fit()
        jump = float(m.params['D'])
        ax.text(cutoff + bandwidth * 0.03, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.08,
                f'LATE = {jump:+.2f}', color=GREEN, fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=LTYEL, alpha=0.7))
    except Exception:
        pass

    ax.set_xlabel(running_var.replace('_', ' ').title())
    ax.set_ylabel((ylabel or outcome).replace('_', ' ').title())
    ax.set_title(title, fontweight='bold', pad=10)
    ax.legend(fontsize=9)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# 4. Synthetic Control Gap Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_sc_comparison(
    treated_series: pd.Series,
    synthetic_series: pd.Series,
    n_pre: int,
    title: str = 'Synthetic Control',
    ylabel: str = 'Outcome',
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Treated unit vs synthetic control with ATT gap shaded.

    Parameters
    ----------
    treated_series  : pd.Series indexed by period
    synthetic_series: pd.Series indexed by period
    n_pre           : number of pre-treatment periods
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))

    ax.plot(treated_series.index, treated_series.values,
            'o-', color=BLUE, ms=4, lw=2, label='Treated unit')
    ax.plot(synthetic_series.index, synthetic_series.values,
            's--', color=GRAY, ms=4, lw=2, label='Synthetic control')

    post_mask = treated_series.index > n_pre
    ax.fill_between(
        treated_series.index[post_mask],
        treated_series.values[post_mask],
        synthetic_series.values[post_mask],
        alpha=0.20, color=BLUE, label='ATT gap',
    )

    att_mean = float((treated_series[post_mask] - synthetic_series[post_mask]).mean())
    ax.axhline(np.nan, color='white', label=f'Mean ATT = {att_mean:+.2f}')

    ax.axvline(n_pre + 0.5, color=RED, lw=1.5, ls='--', alpha=0.7,
               label='Treatment onset')

    ax.set_xlabel('Period')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold', pad=10)
    ax.legend(fontsize=9)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# 5. Multi-Panel Dashboard
# ─────────────────────────────────────────────────────────────────────────────

def plot_causal_dashboard(
    its_data:  Optional[Dict] = None,
    did_data:  Optional[Dict] = None,
    rdd_data:  Optional[Dict] = None,
    sc_data:   Optional[Dict] = None,
    suptitle:  str = 'Causal Inference: Method Comparison Dashboard',
) -> plt.Figure:
    """
    Four-panel dashboard showing all four causal designs.

    Each panel accepts a dict with keys:
        'df'        : DataFrame
        'model'     : fitted model (for ITS)
        'title'     : panel title
        'args'      : additional kwargs passed to the plot function

    Pass None for any panel you don't need — it will show a blank placeholder.
    """
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]

    panels = [
        ('ITS',  its_data),
        ('DiD',  did_data),
        ('RDD',  rdd_data),
        ('SC',   sc_data),
    ]

    for ax, (design, data) in zip(axes, panels):
        if data is None:
            ax.axis('off')
            ax.text(0.5, 0.5, f'{design}\n(not configured)',
                    ha='center', va='center', transform=ax.transAxes, color='gray')
            continue

        try:
            if design == 'ITS':
                plot_its_counterfactual(
                    data['df'], data.get('outcome', 'outcome'),
                    data.get('time_col', 'period'),
                    data.get('intervention_time', 30),
                    data['model'],
                    title=data.get('title', 'ITS'),
                    ax=ax,
                )
            elif design == 'DiD':
                plot_did_counterfactual(
                    data['df'], data.get('outcome', 'outcome'),
                    data.get('post_col', 'post'),
                    data.get('treated_col', 'treated'),
                    title=data.get('title', 'DiD'),
                    ax=ax,
                )
            elif design == 'RDD':
                plot_rdd_discontinuity(
                    data['df'], data.get('outcome', 'outcome'),
                    data.get('running_var', 'x'),
                    data.get('cutoff', 0.0),
                    data.get('bandwidth', 1.0),
                    title=data.get('title', 'RDD'),
                    ax=ax,
                )
            elif design == 'SC':
                plot_sc_comparison(
                    data['treated'],
                    data['synthetic'],
                    data.get('n_pre', 15),
                    title=data.get('title', 'Synthetic Control'),
                    ax=ax,
                )
        except Exception as exc:
            ax.axis('off')
            ax.text(0.5, 0.5, f'{design} error:\n{exc}',
                    ha='center', va='center', transform=ax.transAxes,
                    color='red', fontsize=9)

    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.01)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Demonstration: Minimal usage of each plot
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    np.random.seed(42)
    rng = np.random.default_rng(42)

    # ── ITS data ──────────────────────────────────────────────────────────────
    n_periods = 50
    intervention = 30
    periods = np.arange(1, n_periods + 1)
    df_its = pd.DataFrame({
        'period':         periods,
        'outcome':        (60 + 0.4*periods
                          + 12*(periods >= intervention)
                          + rng.normal(0, 3, n_periods)),
        'post':           (periods >= intervention).astype(int),
        'time_since_post': np.maximum(periods - intervention, 0),
    })
    its_model = smf.ols('outcome ~ period + post + time_since_post', data=df_its).fit()

    # ── DiD data ──────────────────────────────────────────────────────────────
    n = 200
    df_did = pd.DataFrame({
        'treated':  np.repeat([0, 1, 0, 1], n // 4),
        'post':     np.tile([0, 0, 1, 1], n // 4),
        'outcome':  (20 + 3*np.repeat([0, 1, 0, 1], n//4)
                    + 2*np.tile([0, 0, 1, 1], n//4)
                    + 4*np.repeat([0, 1, 0, 1], n//4)*np.tile([0, 0, 1, 1], n//4)
                    + rng.normal(0, 2, n)),
    })
    df_did['post_treated'] = df_did['post'] * df_did['treated']

    # ── RDD data ──────────────────────────────────────────────────────────────
    x = rng.normal(0, 1, 2000)
    df_rdd = pd.DataFrame({
        'x':  x,
        'D':  (x >= 0).astype(int),
        'outcome': 2 + 0.5*x + 1.5*(x >= 0) + rng.normal(0, 0.8, 2000),
    })

    # ── SC data (synthetic) ───────────────────────────────────────────────────
    t_idx = np.arange(1, 31)
    treated_sc  = pd.Series(50 + 0.5*t_idx + rng.normal(0, 1, 30)
                            + 5*(t_idx > 20), index=t_idx)
    synthetic_sc = pd.Series(50 + 0.5*t_idx + rng.normal(0, 1, 30), index=t_idx)

    # ── Four-panel dashboard ──────────────────────────────────────────────────
    fig = plot_causal_dashboard(
        its_data={'df': df_its, 'model': its_model, 'title': 'A. ITS — Level Change',
                  'intervention_time': intervention},
        did_data={'df': df_did, 'title': 'B. DiD — Group Means'},
        rdd_data={'df': df_rdd, 'running_var': 'x', 'cutoff': 0.0, 'bandwidth': 2.0,
                  'title': 'C. RDD — Discontinuity at Cutoff'},
        sc_data={'treated': treated_sc, 'synthetic': synthetic_sc, 'n_pre': 20,
                 'title': 'D. Synthetic Control — Gap'},
        suptitle='Publication-Quality Counterfactual Plots — All Four Designs',
    )
    plt.tight_layout()
    plt.savefig('counterfactual_plots_dashboard.png', dpi=120, bbox_inches='tight')
    print('Dashboard saved: counterfactual_plots_dashboard.png')
