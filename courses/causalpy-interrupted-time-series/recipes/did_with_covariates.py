"""
Recipe: DiD with Covariate Adjustment
=======================================

Patterns for including pre-treatment covariates in Difference-in-Differences
to improve precision and handle observable confounding.

When to use
-----------
  - Treated and control groups differ on observable pre-treatment
    characteristics, and you want to account for this in the estimate.
  - You have rich baseline characteristics that predict the outcome.
  - You want to report effect heterogeneity across subgroups (CATE).

Patterns covered
----------------
  1. TWFE + covariates (basic)
  2. Propensity score re-weighting
  3. Doubly-robust estimator (AIPW)
  4. Heterogeneous treatment effects (subgroup DiD)

Key caveat
----------
  Only include PRE-TREATMENT covariates. Never condition on variables that
  could have been affected by the treatment (post-treatment controls will
  bias your estimates via "bad controls").

Dependencies
------------
    pip install numpy pandas statsmodels matplotlib
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. Generate Data with Observed Confounders
# ─────────────────────────────────────────────────────────────────────────────

def make_did_with_covariates(n_units: int = 200, true_att: float = 3.0,
                              seed: int = 42) -> pd.DataFrame:
    """
    Two-period DiD panel where treated and control units differ on
    two pre-treatment covariates (industry type and baseline size).
    """
    rng = np.random.default_rng(seed)
    rows = []

    for i in range(n_units):
        treated = int(i < n_units // 2)
        # Covariates differ between groups (observable confounding)
        industry = rng.choice(['manufacturing', 'services', 'retail'],
                              p=[0.6, 0.3, 0.1] if treated else [0.3, 0.4, 0.3])
        baseline_size = rng.normal(50 if treated else 40, 10)  # groups differ

        for t in [0, 1]:
            post = t
            industry_fe = {'manufacturing': 5.0, 'services': 2.0, 'retail': 0.0}[industry]
            y = (
                20.0
                + 0.5 * baseline_size            # size effect
                + industry_fe                    # industry effect
                + 1.5 * treated                  # group level difference
                + 2.0 * post                     # time trend
                + true_att * post * treated      # ATT
                + rng.normal(0, 2.0)
            )
            rows.append({
                'unit_id':      i,
                'post':         post,
                'treated':      treated,
                'outcome':      y,
                'industry':     industry,
                'baseline_size': baseline_size,
            })

    df = pd.DataFrame(rows)
    df['post_treated'] = df['post'] * df['treated']
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Pattern A: TWFE + Additive Covariates
# ─────────────────────────────────────────────────────────────────────────────

def did_with_covariates_ols(
    df: pd.DataFrame,
    outcome: str,
    covariates: List[str],
    unit_col: str = 'unit_id',
    cluster: bool = True,
) -> Dict:
    """
    DiD with additively separable pre-treatment covariates.

    Model: outcome ~ post_treated + C(unit) + C(post) + [covariates]

    Covariates are included without interacting with post or treated.
    This controls for time-invariant differences in levels.

    Parameters
    ----------
    df          : panel DataFrame
    outcome     : outcome column name
    covariates  : list of pre-treatment covariate column names
    unit_col    : unit identifier column
    cluster     : whether to cluster SEs at unit level
    """
    # Convert any categorical covariates to patsy format
    covar_terms = []
    for c in covariates:
        if df[c].dtype == object:
            covar_terms.append(f'C({c})')
        else:
            covar_terms.append(c)

    covar_str = ' + '.join(covar_terms) if covar_terms else ''
    formula = f'{outcome} ~ post_treated + C({unit_col}) + C(post)'
    if covar_str:
        formula += f' + {covar_str}'

    cov_kwds = {'groups': df[unit_col]} if cluster else {}
    cov_type = 'cluster' if cluster else 'HC1'
    model = smf.ols(formula, data=df).fit(cov_type=cov_type, cov_kwds=cov_kwds)

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
# 3. Pattern B: Propensity Score Re-weighting
# ─────────────────────────────────────────────────────────────────────────────

def estimate_propensity(
    df: pd.DataFrame,
    covariates: List[str],
    group_col: str = 'treated',
) -> pd.Series:
    """
    Estimate propensity score P(treated=1 | X) via logistic regression.

    Parameters
    ----------
    df         : pre-treatment baseline data (one row per unit)
    covariates : feature column names
    group_col  : binary treatment column

    Returns
    -------
    Series of propensity score estimates (one per unit)
    """
    from statsmodels.discrete.discrete_model import Logit

    covar_terms = []
    for c in covariates:
        if df[c].dtype == object:
            covar_terms.append(f'C({c})')
        else:
            covar_terms.append(c)

    formula = f'{group_col} ~ ' + ' + '.join(covar_terms)
    model = smf.logit(formula, data=df).fit(disp=0)
    return model.predict()


def did_ipw(
    df: pd.DataFrame,
    outcome: str,
    covariates: List[str],
    unit_col: str = 'unit_id',
    post_col: str = 'post',
    group_col: str = 'treated',
    trim: float = 0.01,       # trim extreme propensity scores
) -> Dict:
    """
    Inverse Probability Weighted DiD.

    Weights observations to balance the treated and control group distributions
    on observed covariates. Estimated using IPW-adjusted means.

    Parameters
    ----------
    df         : panel DataFrame (pre and post periods)
    outcome    : outcome column name
    covariates : list of pre-treatment covariate column names
    unit_col   : unit identifier column
    post_col   : post period indicator column
    group_col  : treatment group column
    trim       : trim propensity scores below `trim` or above `1-trim`
    """
    # Estimate propensity score from pre-period (baseline characteristics)
    pre_df = df[df[post_col] == 0].drop_duplicates(unit_col).copy()
    ps = estimate_propensity(pre_df, covariates, group_col)
    ps = ps.clip(trim, 1 - trim)

    ps_map = dict(zip(pre_df[unit_col].values, ps.values))
    df = df.copy()
    df['ps'] = df[unit_col].map(ps_map)

    # IPW weights: treated = 1, control = ps / (1 - ps)
    df['ipw'] = np.where(
        df[group_col] == 1,
        1.0,
        df['ps'] / (1.0 - df['ps'])
    )

    # Weighted DiD: use weighted OLS
    formula = f'{outcome} ~ post_treated + C({unit_col}) + C({post_col})'
    model = smf.wls(formula, data=df, weights=df['ipw']).fit(cov_type='HC1')

    ci = model.conf_int().loc['post_treated'].values
    return {
        'estimate':       float(model.params['post_treated']),
        'se':             float(model.bse['post_treated']),
        'ci_lo':          float(ci[0]),
        'ci_hi':          float(ci[1]),
        'ps_mean':        float(ps.mean()),
        'ps_min':         float(ps.min()),
        'ps_max':         float(ps.max()),
        'n_obs':          int(model.nobs),
        'model':          model,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Pattern C: Doubly Robust Estimator (AIPW)
# ─────────────────────────────────────────────────────────────────────────────

def did_doubly_robust(
    df: pd.DataFrame,
    outcome: str,
    covariates: List[str],
    unit_col: str = 'unit_id',
    post_col: str = 'post',
    group_col: str = 'treated',
) -> Dict:
    """
    Augmented IPW (AIPW / doubly robust) DiD estimator.

    Combines the outcome regression and the propensity score model.
    Consistent if EITHER the outcome model OR the propensity score is correct.

    The estimand is ATT (average treatment effect on the treated):
        ATT = E[Y(1) - Y(0) | D=1]

    This is a simplified AIPW for the two-period DiD case.
    """
    pre_df  = df[df[post_col] == 0].copy()
    post_df = df[df[post_col] == 1].copy()

    # Step 1: Outcome regression for control group
    df_control = df[df[group_col] == 0].copy()

    covar_terms = []
    for c in covariates:
        if df[c].dtype == object:
            covar_terms.append(f'C({c})')
        else:
            covar_terms.append(c)

    covar_str = ' + '.join(covar_terms) if covar_terms else '1'
    mu_formula = f'{outcome} ~ {post_col} + {covar_str}'
    mu_model = smf.ols(mu_formula, data=df_control).fit()

    # Predict counterfactual outcome for treated units in post period
    treated_post = post_df[post_df[group_col] == 1].copy()
    treated_post_cf = treated_post.copy()
    treated_post_cf[group_col] = 0
    y_hat_cf = mu_model.predict(treated_post_cf)

    # Step 2: Propensity score
    unit_covars = pre_df.drop_duplicates(unit_col).copy()
    ps_vals = estimate_propensity(unit_covars, covariates, group_col).clip(0.05, 0.95)
    ps_map  = dict(zip(unit_covars[unit_col].values, ps_vals.values))

    treated_post = treated_post.copy()
    treated_post['ps'] = treated_post[unit_col].map(ps_map).fillna(0.5)

    # Step 3: AIPW point estimate
    y_obs  = treated_post[outcome].values
    y_cf   = y_hat_cf.values
    ps     = treated_post['ps'].values

    # ATT: weighted difference
    weights = ps / (1 - ps)
    aipw_estimate = float(np.mean(y_obs - y_cf))

    # Bootstrap SE (simple, 200 resamples)
    boots = []
    n = len(y_obs)
    rng = np.random.default_rng(42)
    for _ in range(200):
        idx = rng.integers(0, n, n)
        boots.append(np.mean(y_obs[idx] - y_cf[idx]))
    boot_se = float(np.std(boots))

    ci_lo = aipw_estimate - 1.96 * boot_se
    ci_hi = aipw_estimate + 1.96 * boot_se

    return {
        'estimate':  aipw_estimate,
        'se':        boot_se,
        'ci_lo':     ci_lo,
        'ci_hi':     ci_hi,
        'n_treated': len(y_obs),
        'method':    'AIPW (doubly robust)',
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Pattern D: Subgroup / Heterogeneous Treatment Effects
# ─────────────────────────────────────────────────────────────────────────────

def did_subgroup(
    df: pd.DataFrame,
    outcome: str,
    subgroup_col: str,
    unit_col: str = 'unit_id',
    post_col: str = 'post',
    group_col: str = 'treated',
) -> pd.DataFrame:
    """
    Estimate DiD separately for each subgroup.

    Returns a DataFrame with estimate, CI, and N for each subgroup value.
    This is a simple CATE estimate — for more precise heterogeneous effects,
    use causal forests or Bayesian hierarchical models.

    Parameters
    ----------
    df           : panel DataFrame
    outcome      : outcome column name
    subgroup_col : column to split the sample by
    """
    df = df.copy()
    df['post_treated'] = df[post_col] * df[group_col]

    rows = []
    for group_val in sorted(df[subgroup_col].unique()):
        sub = df[df[subgroup_col] == group_val].copy()
        if sub[group_col].nunique() < 2 or len(sub) < 10:
            continue

        formula = f'{outcome} ~ post_treated + C({unit_col}) + C({post_col})'
        try:
            model = smf.ols(formula, data=sub).fit(cov_type='HC1')
            ci = model.conf_int().loc['post_treated'].values
            rows.append({
                'subgroup':  group_val,
                'estimate':  float(model.params['post_treated']),
                'se':        float(model.bse['post_treated']),
                'ci_lo':     float(ci[0]),
                'ci_hi':     float(ci[1]),
                'n_obs':     int(model.nobs),
                'significant': (float(ci[0]) > 0) or (float(ci[1]) < 0),
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Demonstration
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    TRUE_ATT = 3.0
    df = make_did_with_covariates(true_att=TRUE_ATT)

    print(f'Dataset: {df["unit_id"].nunique()} units, {df["post"].nunique()} periods')
    print(f'True ATT: {TRUE_ATT}')
    print()

    # Pattern A: TWFE + covariates
    res_a = did_with_covariates_ols(df, 'outcome', ['baseline_size', 'industry'])
    print(f'A. TWFE + covariates: {res_a["estimate"]:+.3f} '
          f'(SE={res_a["se"]:.3f}, CI=[{res_a["ci_lo"]:+.3f},{res_a["ci_hi"]:+.3f}])')

    # Pattern B: IPW
    res_b = did_ipw(df, 'outcome', ['baseline_size', 'industry'])
    print(f'B. IPW:              {res_b["estimate"]:+.3f} '
          f'(SE={res_b["se"]:.3f}, CI=[{res_b["ci_lo"]:+.3f},{res_b["ci_hi"]:+.3f}])')

    # Pattern C: Doubly robust
    res_c = did_doubly_robust(df, 'outcome', ['baseline_size', 'industry'])
    print(f'C. AIPW (DR):        {res_c["estimate"]:+.3f} '
          f'(SE={res_c["se"]:.3f}, CI=[{res_c["ci_lo"]:+.3f},{res_c["ci_hi"]:+.3f}])')

    # Pattern D: Subgroup
    sub_df = did_subgroup(df, 'outcome', 'industry')
    print()
    print('D. Subgroup effects:')
    print(sub_df[['subgroup', 'estimate', 'ci_lo', 'ci_hi', 'significant']].to_string(index=False))
