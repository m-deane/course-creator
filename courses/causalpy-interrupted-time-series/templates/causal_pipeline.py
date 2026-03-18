"""
Causal Inference Production Pipeline Template
==============================================

End-to-end template for automated causal analysis:
  data loading → design selection → estimation → diagnostics → report

Usage
-----
1. Copy this file to your project.
2. Replace the load_data() and configure() functions with your specifics.
3. Run: python causal_pipeline.py

Dependencies
------------
    pip install causalpy pandas numpy statsmodels scipy

Design coverage
---------------
    DiD   — Difference-in-Differences (two-period or panel)
    RDD   — Regression Discontinuity Design (sharp, local linear)
    ITS   — Interrupted Time Series
    IV    — Instrumental Variables (2SLS)
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

# Optional: install causalpy for Bayesian backend
try:
    import causalpy as cp
    HAS_CAUSALPY = True
except ImportError:
    HAS_CAUSALPY = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """All configuration for a causal pipeline run.

    Customise these fields for your analysis.
    """

    # --- Data ---
    data_path: str = 'data/panel.parquet'     # path to input data file
    results_dir: str = 'results'              # directory for output JSON files

    # --- Design (one of: 'did', 'rdd', 'its', 'iv') ---
    design: str = 'did'

    # --- Column names ---
    outcome: str = 'outcome'
    unit_col: str = 'unit_id'
    time_col: str = 'period'
    group_col: str = 'treated'               # 1 = treated, 0 = control (DiD)
    running_var: str = 'score'               # running variable for RDD
    instrument: str = 'z'                    # instrument column for IV
    treatment_col: str = 'treatment'         # endogenous treatment for IV

    # --- DiD parameters ---
    treatment_time: int = 5                  # first treated period
    pre_trend_alpha: float = 0.05            # significance level for pre-trend test

    # --- RDD parameters ---
    cutoff: float = 0.0                      # assignment cutoff
    bandwidth: float = 0.5                   # bandwidth (half-window around cutoff)

    # --- ITS parameters ---
    n_pre_periods: int = 12                  # periods before intervention

    # --- IV parameters ---
    first_stage_f_min: float = 10.0         # minimum acceptable first-stage F

    # --- Monitoring ---
    drift_z_threshold: float = 2.0          # z-score to trigger drift alert
    baseline_estimate: Optional[float] = None
    baseline_se: Optional[float] = None

    # --- Report ---
    code_version: str = '1.0.0'


def configure() -> PipelineConfig:
    """Return pipeline configuration.

    Modify this function to set your parameters, or load from a YAML/JSON file.
    """
    return PipelineConfig(
        data_path='data/panel.parquet',
        design='did',
        outcome='outcome',
        unit_col='unit_id',
        time_col='period',
        group_col='treated',
        treatment_time=5,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_data(config: PipelineConfig) -> pd.DataFrame:
    """Load the analysis dataset.

    Replace this function with your actual data loading logic.
    Supported formats: parquet, csv, feather.
    """
    path = Path(config.data_path)
    suffix = path.suffix.lower()

    if suffix == '.parquet':
        return pd.read_parquet(path)
    elif suffix == '.csv':
        return pd.read_csv(path)
    elif suffix == '.feather':
        return pd.read_feather(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .parquet, .csv, or .feather.")


def hash_dataframe(df: pd.DataFrame) -> str:
    """Compute a deterministic MD5 hash of a DataFrame for reproducibility."""
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Results Container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Container for all outputs of a causal pipeline run."""

    # Core estimate
    estimate: Optional[float] = None
    se: Optional[float] = None
    ci_lo: Optional[float] = None
    ci_hi: Optional[float] = None
    n_obs: Optional[int] = None

    # Assumption status
    assumptions_passed: bool = False
    assumption_warnings: List[str] = field(default_factory=list)

    # Diagnostics
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def significant(self) -> bool:
        if self.ci_lo is None or self.ci_hi is None:
            return False
        return (self.ci_lo > 0) or (self.ci_hi < 0)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['significant'] = self.significant
        d['timestamp'] = datetime.utcnow().isoformat()
        return d

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        log.info('Result saved to %s', path)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Data Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_data(df: pd.DataFrame, config: PipelineConfig) -> Tuple[bool, List[str]]:
    """Validate input data before estimation.

    Checks:
    - Required columns present
    - No missing values in outcome
    - Sufficient group/time variation for the chosen design
    """
    errors: List[str] = []

    # Determine required columns by design
    required: List[str] = [config.outcome]
    if config.design == 'did':
        required += [config.unit_col, config.time_col, config.group_col]
    elif config.design == 'rdd':
        required += [config.running_var]
    elif config.design == 'its':
        required += [config.time_col]
    elif config.design == 'iv':
        required += [config.instrument, config.treatment_col]

    for col in required:
        if col not in df.columns:
            errors.append(f"Missing required column: '{col}'")

    if config.outcome in df.columns:
        n_missing = df[config.outcome].isna().sum()
        if n_missing > 0:
            errors.append(f"Missing values in outcome column: {n_missing} rows")

    if config.design == 'did' and config.group_col in df.columns:
        n_groups = df[config.group_col].nunique()
        if n_groups < 2:
            errors.append(f"DiD requires at least 2 groups; found {n_groups}.")

    if config.design in ('did', 'its') and config.time_col in df.columns:
        n_periods = df[config.time_col].nunique()
        if n_periods < 2:
            errors.append(f"Need at least 2 time periods; found {n_periods}.")

    if config.design == 'rdd' and config.running_var in df.columns:
        n_obs = len(df)
        if n_obs < 20:
            errors.append(f"RDD requires at least 20 observations; found {n_obs}.")

    return len(errors) == 0, errors


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Assumption Checks
# ─────────────────────────────────────────────────────────────────────────────

def check_parallel_trends(df: pd.DataFrame, config: PipelineConfig) -> Dict[str, Any]:
    """Test for parallel pre-treatment trends (DiD).

    Fits: outcome ~ treated * period + C(unit) on pre-period data only.
    Returns dict with passed, interaction_p, and warning.
    """
    pre_df = df[df[config.time_col] < config.treatment_time].copy()
    n_pre = len(pre_df)

    if n_pre < 10:
        return {'passed': None, 'reason': 'Insufficient pre-period data', 'n_pre_obs': n_pre}

    try:
        formula = (
            f'{config.outcome} ~ {config.group_col} * {config.time_col} '
            f'+ C({config.unit_col})'
        )
        model = smf.ols(formula, data=pre_df).fit()

        interaction_key = f'{config.group_col}:{config.time_col}'
        if interaction_key in model.pvalues:
            p_val = float(model.pvalues[interaction_key])
            coef  = float(model.params[interaction_key])
        else:
            p_val = 1.0
            coef  = 0.0

        passed = p_val >= config.pre_trend_alpha
        return {
            'passed':        passed,
            'warning':       not passed,
            'interaction_p': p_val,
            'interaction_coef': coef,
            'alpha':         config.pre_trend_alpha,
            'n_pre_obs':     n_pre,
        }
    except Exception as exc:
        return {'passed': None, 'reason': str(exc), 'n_pre_obs': n_pre}


def check_rdd_density(df: pd.DataFrame, config: PipelineConfig) -> Dict[str, Any]:
    """McCrary-style density test for RDD (simplified).

    Splits observations on each side of the cutoff and tests whether the
    density (count per bin) is discontinuous at the cutoff. A statistically
    significant discontinuity suggests manipulation.
    """
    rv = df[config.running_var].dropna()
    left  = rv[(rv >= config.cutoff - config.bandwidth) & (rv < config.cutoff)]
    right = rv[(rv >= config.cutoff) & (rv <= config.cutoff + config.bandwidth)]

    if len(left) < 5 or len(right) < 5:
        return {'passed': None, 'reason': 'Too few observations near cutoff.'}

    # Density ratio test: left-side vs right-side counts
    n_left, n_right = len(left), len(right)
    ratio = n_right / max(n_left, 1)

    # Binomial test: under H0, expect 50/50 split
    binom_result = stats.binomtest(n_right, n_left + n_right, 0.5)
    p_val = binom_result.pvalue

    passed = p_val >= 0.05
    return {
        'passed':        passed,
        'warning':       not passed,
        'n_left':        n_left,
        'n_right':       n_right,
        'density_ratio': ratio,
        'binomial_p':    p_val,
    }


def check_iv_first_stage(df: pd.DataFrame, config: PipelineConfig) -> Dict[str, Any]:
    """Check IV first-stage strength via F-statistic.

    Fits: treatment ~ instrument + controls
    Reports first-stage F and whether it exceeds the minimum threshold.
    """
    try:
        formula = f'{config.treatment_col} ~ {config.instrument}'
        first_stage = smf.ols(formula, data=df).fit()
        # F-stat for the restricted model (instrument only)
        f_stat = float(first_stage.fvalue)
        passed = f_stat >= config.first_stage_f_min
        return {
            'passed':          passed,
            'warning':         not passed,
            'first_stage_f':   f_stat,
            'f_threshold':     config.first_stage_f_min,
            'n_obs':           int(first_stage.nobs),
        }
    except Exception as exc:
        return {'passed': None, 'reason': str(exc)}


def run_assumption_checks(df: pd.DataFrame, config: PipelineConfig) -> Dict[str, Any]:
    """Run the appropriate assumption checks for the configured design."""
    checks: Dict[str, Any] = {}

    if config.design == 'did':
        checks['parallel_trends'] = check_parallel_trends(df, config)

    elif config.design == 'rdd':
        checks['density_test'] = check_rdd_density(df, config)

    elif config.design == 'iv':
        checks['first_stage'] = check_iv_first_stage(df, config)

    return checks


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_did(df: pd.DataFrame, config: PipelineConfig) -> PipelineResult:
    """Estimate treatment effect using Difference-in-Differences."""
    result = PipelineResult()
    df_model = df.copy()
    df_model['post'] = (df_model[config.time_col] >= config.treatment_time).astype(int)
    df_model['post_treated'] = df_model['post'] * df_model[config.group_col]

    formula = (
        f'{config.outcome} ~ post_treated '
        f'+ C({config.unit_col}) + C({config.time_col})'
    )
    model = smf.ols(formula, data=df_model).fit(
        cov_type='cluster', cov_kwds={'groups': df_model[config.unit_col]}
    )

    result.estimate = float(model.params['post_treated'])
    result.se       = float(model.bse['post_treated'])
    ci = model.conf_int().loc['post_treated'].values
    result.ci_lo = float(ci[0])
    result.ci_hi = float(ci[1])
    result.n_obs = int(model.nobs)
    return result


def estimate_rdd(df: pd.DataFrame, config: PipelineConfig) -> PipelineResult:
    """Estimate treatment effect using sharp RDD (local linear)."""
    result = PipelineResult()

    mask = (df[config.running_var] - config.cutoff).abs() <= config.bandwidth
    local = df[mask].copy()

    if len(local) < 10:
        result.assumption_warnings.append(
            f'Fewer than 10 observations within bandwidth {config.bandwidth}.'
        )
        return result

    local['x_c'] = local[config.running_var] - config.cutoff
    local['D']   = (local[config.running_var] >= config.cutoff).astype(int)

    formula = f'{config.outcome} ~ D + x_c + D:x_c'
    model = smf.ols(formula, data=local).fit(cov_type='HC1')

    result.estimate = float(model.params['D'])
    result.se       = float(model.bse['D'])
    ci = model.conf_int().loc['D'].values
    result.ci_lo = float(ci[0])
    result.ci_hi = float(ci[1])
    result.n_obs = int(model.nobs)
    return result


def estimate_its(df: pd.DataFrame, config: PipelineConfig) -> PipelineResult:
    """Estimate treatment effect using Interrupted Time Series (segmented regression)."""
    result = PipelineResult()

    df_m = df.copy().sort_values(config.time_col).reset_index(drop=True)
    df_m['post'] = (df_m[config.time_col] >= config.n_pre_periods).astype(int)
    df_m['time_since'] = df_m[config.time_col] - config.n_pre_periods
    df_m['time_since_post'] = df_m['time_since'] * df_m['post']

    formula = f'{config.outcome} ~ {config.time_col} + post + time_since_post'
    model = smf.ols(formula, data=df_m).fit(cov_type='HC1')

    result.estimate = float(model.params['post'])
    result.se       = float(model.bse['post'])
    ci = model.conf_int().loc['post'].values
    result.ci_lo = float(ci[0])
    result.ci_hi = float(ci[1])
    result.n_obs = int(model.nobs)
    return result


def estimate_iv(df: pd.DataFrame, config: PipelineConfig) -> PipelineResult:
    """Estimate treatment effect using 2SLS Instrumental Variables."""
    result = PipelineResult()

    try:
        from statsmodels.sandbox.regression.gmm import IV2SLS

        X_first = smf.add_constant(df[[config.instrument]])
        first_stage = smf.OLS(df[config.treatment_col], X_first).fit()
        df = df.copy()
        df['d_hat'] = first_stage.fittedvalues

        X_second = smf.add_constant(df[['d_hat']])
        second_stage = smf.OLS(df[config.outcome], X_second).fit()

        result.estimate = float(second_stage.params['d_hat'])
        result.se       = float(second_stage.bse['d_hat'])
        ci = second_stage.conf_int().loc['d_hat'].values
        result.ci_lo = float(ci[0])
        result.ci_hi = float(ci[1])
        result.n_obs = int(second_stage.nobs)

        # Note: 2SLS SEs computed here are not asymptotically correct for
        # standard inference. Use linearmodels.IV2SLS in production.
        result.assumption_warnings.append(
            'SE computed via manual 2SLS. Use linearmodels.IV2SLS for correct SEs.'
        )
    except Exception as exc:
        result.assumption_warnings.append(f'IV estimation failed: {exc}')

    return result


def run_estimation(df: pd.DataFrame, config: PipelineConfig) -> PipelineResult:
    """Dispatch to the appropriate estimator based on config.design."""
    dispatch = {
        'did': estimate_did,
        'rdd': estimate_rdd,
        'its': estimate_its,
        'iv':  estimate_iv,
    }
    if config.design not in dispatch:
        raise ValueError(f"Unknown design: '{config.design}'. Choose from {list(dispatch)}.")
    return dispatch[config.design](df, config)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Monitoring
# ─────────────────────────────────────────────────────────────────────────────

class DriftMonitor:
    """Monitor a causal estimate for drift relative to a stored baseline."""

    def __init__(self, baseline_estimate: float, baseline_se: float,
                 z_threshold: float = 2.0):
        self.baseline_estimate = baseline_estimate
        self.baseline_se = baseline_se
        self.z_threshold = z_threshold

    def check(self, new_estimate: float) -> Dict[str, Any]:
        """Return drift status for a new estimate."""
        z_score = abs(new_estimate - self.baseline_estimate) / self.baseline_se
        stable = z_score < self.z_threshold
        return {
            'stable':   stable,
            'alert':    not stable,
            'z_score':  z_score,
            'delta':    new_estimate - self.baseline_estimate,
            'baseline': self.baseline_estimate,
            'new':      new_estimate,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: Reproducibility Manifest
# ─────────────────────────────────────────────────────────────────────────────

def create_manifest(
    df: pd.DataFrame,
    config: PipelineConfig,
    result: PipelineResult,
    assumption_checks: Dict,
    drift_status: Optional[Dict] = None,
) -> Dict:
    """Create a full reproducibility manifest for this pipeline run."""
    return {
        'run_id':           datetime.utcnow().isoformat(),
        'code_version':     config.code_version,
        'data_hash':        hash_dataframe(df),
        'n_rows':           len(df),
        'columns':          list(df.columns),
        'config':           asdict(config),
        'assumption_checks': assumption_checks,
        'result':           result.to_dict(),
        'drift_status':     drift_status,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(config: Optional[PipelineConfig] = None) -> PipelineResult:
    """
    Execute the full causal pipeline.

    Stages:
        1. Load data
        2. Validate data
        3. Run assumption checks
        4. Estimate treatment effect
        5. Check for drift (if baseline available)
        6. Save reproducibility manifest
        7. Return result

    Parameters
    ----------
    config : PipelineConfig, optional
        If None, uses configure() to build the configuration.

    Returns
    -------
    PipelineResult
    """
    if config is None:
        config = configure()

    log.info('Starting causal pipeline  design=%s', config.design)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    log.info('Loading data from %s', config.data_path)
    df = load_data(config)
    log.info('Loaded %d rows, %d columns.', len(df), len(df.columns))

    # ── 2. Validate data ──────────────────────────────────────────────────────
    valid, errors = validate_data(df, config)
    if not valid:
        log.error('Data validation FAILED: %s', errors)
        result = PipelineResult(assumption_warnings=errors)
        return result
    log.info('Data validation passed.')

    # ── 3. Assumption checks ──────────────────────────────────────────────────
    assumption_checks = run_assumption_checks(df, config)
    warnings: List[str] = []
    for check_name, check_result in assumption_checks.items():
        if check_result.get('warning'):
            msg = f'{check_name}: assumption may be violated.'
            warnings.append(msg)
            log.warning(msg)
        else:
            log.info('%s: passed.', check_name)

    # ── 4. Estimation ─────────────────────────────────────────────────────────
    log.info('Running %s estimation...', config.design.upper())
    result = run_estimation(df, config)
    result.assumption_warnings.extend(warnings)
    result.assumptions_passed = len(result.assumption_warnings) == 0
    result.diagnostics['assumption_checks'] = assumption_checks
    result.metadata = {
        'design':       config.design,
        'outcome':      config.outcome,
        'n_rows_input': len(df),
        'code_version': config.code_version,
    }

    log.info(
        'Estimate: τ = %.4f  SE = %.4f  CI = [%.4f, %.4f]  N = %s  sig = %s',
        result.estimate or float('nan'),
        result.se or float('nan'),
        result.ci_lo or float('nan'),
        result.ci_hi or float('nan'),
        result.n_obs,
        result.significant,
    )

    # ── 5. Drift monitoring ───────────────────────────────────────────────────
    drift_status = None
    if config.baseline_estimate is not None and config.baseline_se is not None:
        monitor = DriftMonitor(config.baseline_estimate, config.baseline_se,
                               config.drift_z_threshold)
        drift_status = monitor.check(result.estimate or 0.0)
        if drift_status['alert']:
            log.warning(
                'DRIFT ALERT: new estimate deviates %.2f baseline SEs from baseline.',
                drift_status['z_score'],
            )

    # ── 6. Save manifest ──────────────────────────────────────────────────────
    results_dir = Path(config.results_dir)
    run_id = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    manifest = create_manifest(df, config, result, assumption_checks, drift_status)
    manifest_path = results_dir / f'{config.design}_result_{run_id}.json'
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    log.info('Manifest saved: %s', manifest_path)

    result_path = results_dir / f'{config.design}_estimate_{run_id}.json'
    result.save(result_path)

    if result.assumption_warnings:
        log.warning('Assumption warnings: %s', result.assumption_warnings)

    log.info('Pipeline complete.')
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cfg = configure()
    result = run_pipeline(cfg)

    print()
    print('═' * 60)
    print('CAUSAL PIPELINE RESULT')
    print('═' * 60)
    print(f'  Design:           {cfg.design.upper()}')
    print(f'  Outcome:          {cfg.outcome}')
    print(f'  Estimate (τ):     {result.estimate}')
    print(f'  SE:               {result.se}')
    print(f'  95% CI:           [{result.ci_lo}, {result.ci_hi}]')
    print(f'  N observations:   {result.n_obs}')
    print(f'  Significant:      {result.significant}')
    print(f'  Assumptions OK:   {result.assumptions_passed}')
    if result.assumption_warnings:
        print(f'  Warnings:')
        for w in result.assumption_warnings:
            print(f'    - {w}')
    print('═' * 60)
