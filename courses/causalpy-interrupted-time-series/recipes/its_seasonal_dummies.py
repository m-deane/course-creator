"""
Recipe: ITS with Seasonal Dummy Variables
==========================================

Interrupted Time Series models with monthly, quarterly, or weekly
seasonal dummies to prevent seasonal variation from biasing the
level-change and slope-change estimates.

When to use
-----------
  Your time series has regular seasonal cycles (monthly flu rates, quarterly
  retail sales, weekly website traffic) and the intervention occurs at a
  specific point in the cycle. Without seasonal adjustment, the level-change
  estimate will absorb seasonal effects.

Copy-paste patterns
-------------------
  1. Monthly seasonal dummies (12-month cycle)
  2. Quarterly seasonal dummies (4-quarter cycle)
  3. Fourier terms (continuous seasonal adjustment — no dummy multicollinearity)
  4. Seasonal DiD (interacting seasonal dummies with post)

Dependencies
------------
    pip install numpy pandas statsmodels matplotlib
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. Generate Seasonal ITS Data
# ─────────────────────────────────────────────────────────────────────────────

def make_seasonal_its(
    n_months: int = 60,
    intervention_month: int = 36,
    true_level_change: float = 8.0,
    seasonal_amplitude: float = 10.0,
    noise_sd: float = 2.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a monthly time series with a strong 12-month seasonal cycle
    and an ITS intervention at `intervention_month`.
    """
    rng = np.random.default_rng(seed)
    months = np.arange(1, n_months + 1)
    month_of_year = ((months - 1) % 12) + 1     # 1..12

    seasonal = seasonal_amplitude * np.sin(2 * np.pi * month_of_year / 12)

    y = (
        50.0
        + 0.2 * months                          # slow underlying trend
        + seasonal                               # 12-month seasonality
        + true_level_change * (months >= intervention_month)  # intervention
        + rng.normal(0, noise_sd, n_months)
    )

    return pd.DataFrame({
        'month':         months,
        'month_of_year': month_of_year,
        'quarter':       ((month_of_year - 1) // 3) + 1,
        'outcome':       y,
        'post':          (months >= intervention_month).astype(int),
        'time_since_post': np.maximum(months - intervention_month, 0),
    })


# ─────────────────────────────────────────────────────────────────────────────
# 2. Pattern A: Monthly Seasonal Dummies
# ─────────────────────────────────────────────────────────────────────────────

def fit_its_monthly_dummies(
    df: pd.DataFrame,
    outcome: str = 'outcome',
    time_col: str = 'month',
    month_of_year_col: str = 'month_of_year',
) -> dict:
    """
    ITS with 11 monthly dummies (month 12 = reference).

    Model:
        outcome = β0 + β1*t + β2*D_post + β3*time_since_post
                + Σ_j γ_j * Month_j + ε

    The monthly dummies absorb calendar-month variation, so β2 and β3
    estimate the intervention effect net of seasonal patterns.
    """
    # C(month_of_year) automatically creates 11 dummies (one omitted)
    formula = (
        f'{outcome} ~ {time_col} + post + time_since_post '
        f'+ C({month_of_year_col})'
    )
    model = smf.ols(formula, data=df).fit(cov_type='HC1')

    ci_level = model.conf_int().loc['post'].values
    ci_slope = model.conf_int().loc['time_since_post'].values

    return {
        'estimate':     float(model.params['post']),
        'se':           float(model.bse['post']),
        'ci_lo':        float(ci_level[0]),
        'ci_hi':        float(ci_level[1]),
        'slope_change': float(model.params['time_since_post']),
        'slope_ci_lo':  float(ci_slope[0]),
        'slope_ci_hi':  float(ci_slope[1]),
        'aic':          float(model.aic),
        'model':        model,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Pattern B: Quarterly Seasonal Dummies
# ─────────────────────────────────────────────────────────────────────────────

def fit_its_quarterly_dummies(
    df: pd.DataFrame,
    outcome: str = 'outcome',
    time_col: str = 'month',
    quarter_col: str = 'quarter',
) -> dict:
    """
    ITS with 3 quarterly dummies (Q4 = reference).

    Appropriate when variation is primarily seasonal at the quarterly level
    (e.g., business output, consumer spending).
    """
    formula = (
        f'{outcome} ~ {time_col} + post + time_since_post '
        f'+ C({quarter_col})'
    )
    model = smf.ols(formula, data=df).fit(cov_type='HC1')

    ci = model.conf_int().loc['post'].values
    return {
        'estimate': float(model.params['post']),
        'se':       float(model.bse['post']),
        'ci_lo':    float(ci[0]),
        'ci_hi':    float(ci[1]),
        'aic':      float(model.aic),
        'model':    model,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Pattern C: Fourier Terms (no dummy multicollinearity)
# ─────────────────────────────────────────────────────────────────────────────

def add_fourier_terms(
    df: pd.DataFrame,
    period: int,           # seasonality period (e.g., 12 for monthly annual)
    n_harmonics: int,      # number of sin/cos pairs to include (1–4)
    time_col: str = 'month',
) -> pd.DataFrame:
    """
    Add Fourier sin/cos terms for flexible seasonal modelling.

    More harmonics → more flexible seasonal pattern (but more parameters).
    Start with n_harmonics=2; add more if residuals show seasonal pattern.

    Parameters
    ----------
    df          : DataFrame with a time column
    period      : seasonal period (12 = annual for monthly data)
    n_harmonics : number of sin/cos pairs (K)
    time_col    : name of the time column
    """
    df = df.copy()
    t = df[time_col].values
    for k in range(1, n_harmonics + 1):
        df[f'sin_{k}'] = np.sin(2 * np.pi * k * t / period)
        df[f'cos_{k}'] = np.cos(2 * np.pi * k * t / period)
    return df


def fit_its_fourier(
    df: pd.DataFrame,
    period: int = 12,
    n_harmonics: int = 2,
    outcome: str = 'outcome',
    time_col: str = 'month',
) -> dict:
    """
    ITS with Fourier seasonal terms.

    Advantage over dummy variables: no dummy trap, fewer parameters when
    the seasonal pattern is smooth.
    """
    df_f = add_fourier_terms(df, period, n_harmonics, time_col)

    fourier_terms = ' + '.join(
        [f'sin_{k} + cos_{k}' for k in range(1, n_harmonics + 1)]
    )
    formula = (
        f'{outcome} ~ {time_col} + post + time_since_post + {fourier_terms}'
    )
    model = smf.ols(formula, data=df_f).fit(cov_type='HC1')

    ci = model.conf_int().loc['post'].values
    return {
        'estimate': float(model.params['post']),
        'se':       float(model.bse['post']),
        'ci_lo':    float(ci[0]),
        'ci_hi':    float(ci[1]),
        'aic':      float(model.aic),
        'model':    model,
        'df_fourier': df_f,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Model Comparison Utility
# ─────────────────────────────────────────────────────────────────────────────

def compare_seasonal_models(df: pd.DataFrame, outcome: str = 'outcome') -> pd.DataFrame:
    """
    Fit all three seasonal ITS specifications and compare by AIC.

    Returns a summary table with estimate, SE, CI, and AIC for each model.
    Lower AIC = better fit.
    """
    no_seasonal = smf.ols(
        f'{outcome} ~ month + post + time_since_post', data=df
    ).fit(cov_type='HC1')
    ci_ns = no_seasonal.conf_int().loc['post'].values

    monthly = fit_its_monthly_dummies(df, outcome)
    quarterly = fit_its_quarterly_dummies(df, outcome)
    fourier = fit_its_fourier(df, period=12, n_harmonics=2, outcome=outcome)

    rows = [
        ('No seasonal adjustment', float(no_seasonal.params['post']),
         float(no_seasonal.bse['post']), float(ci_ns[0]), float(ci_ns[1]),
         float(no_seasonal.aic)),
        ('Monthly dummies',       monthly['estimate'],   monthly['se'],
         monthly['ci_lo'],   monthly['ci_hi'],   monthly['aic']),
        ('Quarterly dummies',     quarterly['estimate'], quarterly['se'],
         quarterly['ci_lo'], quarterly['ci_hi'], quarterly['aic']),
        ('Fourier (K=2)',         fourier['estimate'],   fourier['se'],
         fourier['ci_lo'],   fourier['ci_hi'],   fourier['aic']),
    ]

    return pd.DataFrame(rows, columns=['Model', 'Estimate', 'SE', 'CI_lo', 'CI_hi', 'AIC'])


# ─────────────────────────────────────────────────────────────────────────────
# 6. Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_seasonal_its(
    df: pd.DataFrame,
    model_result: dict,
    intervention_month: int = 36,
    title: str = 'ITS with Seasonal Adjustment',
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot observed data and model-fitted values with seasonal adjustment."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    ax.scatter(df['month'], df['outcome'], s=20, alpha=0.5,
               color='steelblue', label='Observed')

    fitted = model_result['model'].fittedvalues
    pre_mask  = df['post'] == 0
    post_mask = df['post'] == 1

    ax.plot(df.loc[pre_mask, 'month'], fitted[pre_mask], '-',
            color='navy', lw=2, label='Fitted (pre)')
    ax.plot(df.loc[post_mask, 'month'], fitted[post_mask], '-',
            color='steelblue', lw=2, label='Fitted (post)')

    # Counterfactual
    df_cf = df.copy()
    df_cf['post'] = 0
    df_cf['time_since_post'] = 0
    cf = model_result['model'].predict(df_cf)
    ax.plot(df.loc[post_mask, 'month'], cf[post_mask], '--',
            color='gray', lw=1.5, alpha=0.8, label='Counterfactual')

    ax.axvline(intervention_month, color='red', lw=1.5, ls='--', alpha=0.7,
               label=f'Intervention (t={intervention_month})')

    ax.set_xlabel('Month')
    ax.set_ylabel('Outcome')
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# Demonstration
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    TRUE_LEVEL = 8.0
    INTERVENTION = 36

    df = make_seasonal_its(true_level_change=TRUE_LEVEL, intervention_month=INTERVENTION)

    comparison = compare_seasonal_models(df)
    print('Model Comparison (lower AIC = better fit):')
    print(comparison.to_string(index=False))
    print(f'\nTrue level change: {TRUE_LEVEL}')

    best_model = fit_its_monthly_dummies(df)
    print(f'\nBest model estimate: {best_model["estimate"]:+.3f}  '
          f'95% CI: [{best_model["ci_lo"]:+.3f}, {best_model["ci_hi"]:+.3f}]')

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    no_seasonal_model = {
        'model': smf.ols('outcome ~ month + post + time_since_post', data=df).fit()
    }
    plot_seasonal_its(df, no_seasonal_model, INTERVENTION,
                      'ITS — No Seasonal Adjustment', axes[0])
    plot_seasonal_its(df, best_model, INTERVENTION,
                      'ITS — Monthly Seasonal Dummies', axes[1])
    plt.tight_layout()
    plt.savefig('its_seasonal.png', dpi=120, bbox_inches='tight')
    print('Plot saved: its_seasonal.png')
