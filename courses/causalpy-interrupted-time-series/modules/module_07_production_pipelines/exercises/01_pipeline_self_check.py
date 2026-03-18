"""
Module 07 — Production Pipelines: Self-Check Exercises
=======================================================

These exercises test your ability to build, validate, and report causal
inference pipelines. Run this file directly; assertions will fail loudly
if your implementation is incorrect.

Estimated time: 20 minutes
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS  (do not modify)
# ─────────────────────────────────────────────────────────────────────────────

def _make_did_panel(n_units=80, n_periods=8, treatment_time=5, true_att=3.0, seed=0):
    """Generate a balanced panel dataset for DiD exercises."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        treated = int(i < n_units // 2)
        fe = rng.normal(0, 1)
        for t in range(1, n_periods + 1):
            post = int(t >= treatment_time)
            y = (
                10
                + 0.5 * t              # common time trend
                + 2 * treated          # time-invariant group difference
                + fe                   # unit fixed effect
                + true_att * post * treated   # treatment effect
                + rng.normal(0, 1.5)
            )
            rows.append({
                'unit_id': i,
                'period':  t,
                'treated': treated,
                'post':    post,
                'outcome': y,
            })
    df = pd.DataFrame(rows)
    df['post_treated'] = df['post'] * df['treated']
    return df


def _make_two_period(n=200, true_att=5.0, seed=42):
    """Generate a simple two-period, two-group DiD dataset."""
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        treated = int(i < n // 2)
        for t in [0, 1]:
            y = (
                20
                + 3 * treated
                + 1.5 * t                        # time trend
                + true_att * t * treated         # ATT
                + rng.normal(0, 3)
            )
            records.append({'unit_id': i, 'treated': treated, 'post': t, 'outcome': y})
    df = pd.DataFrame(records)
    df['post_treated'] = df['post'] * df['treated']
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Exercise 1: Data Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_did_data(df: pd.DataFrame, outcome: str, group_col: str,
                      time_col: str, unit_col: str) -> Tuple[bool, List[str]]:
    """
    Validate a DiD dataset before estimation.

    Check for:
    1. All required columns present
    2. No missing values in the outcome column
    3. At least 2 distinct values in the group column (treated + control)
    4. At least 2 distinct values in the time column
    5. No negative values in the outcome (if all values are expected positive)

    Parameters
    ----------
    df         : input DataFrame
    outcome    : name of the outcome column
    group_col  : name of the treatment group column
    time_col   : name of the period column
    unit_col   : name of the unit identifier column

    Returns
    -------
    (is_valid: bool, errors: list of str)
    """
    # ── YOUR CODE ─────────────────────────────────────────────────────────────
    # Collect errors in a list. Return (True, []) if no errors, else (False, errors).
    # Do NOT raise exceptions — just collect error strings.
    raise NotImplementedError("Implement validate_did_data()")
    # ── END YOUR CODE ─────────────────────────────────────────────────────────


def _test_validate_did_data():
    """Exercise 1 tests."""
    df = _make_two_period()

    # ── Test 1a: valid dataset passes ────────────────────────────────────────
    is_valid, errors = validate_did_data(
        df, 'outcome', 'treated', 'post', 'unit_id'
    )
    assert is_valid, f"Valid dataset should pass. Errors: {errors}"
    assert errors == [], f"No errors expected for valid dataset. Got: {errors}"

    # ── Test 1b: missing column detected ─────────────────────────────────────
    df_bad_cols = df.drop(columns=['outcome'])
    is_valid2, errors2 = validate_did_data(
        df_bad_cols, 'outcome', 'treated', 'post', 'unit_id'
    )
    assert not is_valid2, "Dataset with missing column should fail validation."
    assert any('outcome' in e.lower() or 'missing' in e.lower() for e in errors2), \
        f"Error message should mention the missing column. Got: {errors2}"

    # ── Test 1c: missing values in outcome detected ───────────────────────────
    df_missing = df.copy()
    df_missing.loc[0, 'outcome'] = np.nan
    is_valid3, errors3 = validate_did_data(
        df_missing, 'outcome', 'treated', 'post', 'unit_id'
    )
    assert not is_valid3, "Dataset with NaN outcome should fail."
    assert len(errors3) >= 1, "At least one error expected for NaN outcome."

    # ── Test 1d: single group detected ───────────────────────────────────────
    df_one_group = df[df['treated'] == 1].copy()
    is_valid4, errors4 = validate_did_data(
        df_one_group, 'outcome', 'treated', 'post', 'unit_id'
    )
    assert not is_valid4, "Dataset with only one group should fail."

    print("  Exercise 1 PASSED — validate_did_data() works correctly.")


# ─────────────────────────────────────────────────────────────────────────────
# Exercise 2: Automated Pre-Trend Test
# ─────────────────────────────────────────────────────────────────────────────

def test_parallel_trends(
    df: pd.DataFrame,
    outcome: str,
    treated_col: str,
    time_col: str,
    unit_col: str,
    treatment_time: int,
    alpha: float = 0.05,
) -> Dict:
    """
    Test for parallel pre-treatment trends.

    Using only the pre-treatment rows (period < treatment_time), fit an OLS
    regression of the form:

        outcome ~ treated * period + C(unit_id)

    Extract the p-value of the treated:period interaction.
    - If p < alpha: warn (possible pre-trend violation)
    - If p >= alpha: pass (no strong evidence against parallel trends)

    Return a dictionary with keys:
        'passed'         (bool)
        'interaction_p'  (float: p-value of the interaction)
        'alpha'          (float: significance level used)
        'warning'        (bool: True if interaction_p < alpha)
        'n_pre_obs'      (int: number of pre-period observations used)

    Parameters
    ----------
    df             : full panel dataset
    outcome        : outcome column name
    treated_col    : binary treatment group column (0/1)
    time_col       : period column name
    unit_col       : unit identifier column
    treatment_time : first treatment period (exclude from pre-period)
    alpha          : significance level for the trend test
    """
    # ── YOUR CODE ─────────────────────────────────────────────────────────────
    raise NotImplementedError("Implement test_parallel_trends()")
    # ── END YOUR CODE ─────────────────────────────────────────────────────────


def _test_parallel_trends():
    """Exercise 2 tests."""
    # ── Dataset with parallel trends ─────────────────────────────────────────
    df_parallel = _make_did_panel(true_att=3.0)

    result = test_parallel_trends(
        df_parallel, 'outcome', 'treated', 'period', 'unit_id',
        treatment_time=5, alpha=0.05,
    )

    assert 'passed' in result, "Result must have 'passed' key."
    assert 'interaction_p' in result, "Result must have 'interaction_p' key."
    assert 'warning' in result, "Result must have 'warning' key."
    assert 'n_pre_obs' in result, "Result must have 'n_pre_obs' key."
    assert isinstance(result['passed'], bool), "'passed' must be bool."
    assert 0 <= result['interaction_p'] <= 1, "p-value must be in [0, 1]."

    # With parallel trends, we should fail to reject at alpha=0.05 most of the time
    # (this is a probabilistic check — use a lenient criterion)
    # We just verify the function runs and returns reasonable values.
    assert result['n_pre_obs'] > 0, "Must use some pre-period observations."

    # ── Dataset with strong pre-trend violation ───────────────────────────────
    rng = np.random.default_rng(99)
    df_violation = _make_did_panel(true_att=0.0)
    # Add a differential pre-trend: treated group trends faster pre-treatment
    df_violation['outcome'] += (
        df_violation['treated']
        * (df_violation['period'] < 5)
        * df_violation['period']
        * 3.0    # strong pre-trend
    )

    result_v = test_parallel_trends(
        df_violation, 'outcome', 'treated', 'period', 'unit_id',
        treatment_time=5, alpha=0.05,
    )
    # Strong pre-trend should trigger a warning
    assert result_v['warning'], \
        f"Strong pre-trend should trigger a warning. Got p = {result_v['interaction_p']:.4f}"

    print("  Exercise 2 PASSED — test_parallel_trends() works correctly.")


# ─────────────────────────────────────────────────────────────────────────────
# Exercise 3: Effect Size Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_effect_sizes(
    estimate: float,
    outcome_values: np.ndarray,
    pre_outcome_values: np.ndarray,
) -> Dict[str, float]:
    """
    Compute standardised effect size metrics for a treatment effect estimate.

    Parameters
    ----------
    estimate           : point estimate of the treatment effect
    outcome_values     : all outcome values in the dataset (for SD)
    pre_outcome_values : outcome values in the pre-treatment period (for baseline mean)

    Returns a dictionary with keys:
        'cohens_d'        (float): estimate / SD of outcome
        'pct_change'      (float): 100 * estimate / pre-treatment mean
        'magnitude_label' (str):  'small', 'medium', or 'large' based on Cohen's d
        'baseline_mean'   (float): mean of pre_outcome_values
        'outcome_sd'      (float): SD of outcome_values
    """
    # ── YOUR CODE ─────────────────────────────────────────────────────────────
    raise NotImplementedError("Implement compute_effect_sizes()")
    # ── END YOUR CODE ─────────────────────────────────────────────────────────


def _test_compute_effect_sizes():
    """Exercise 3 tests."""
    # Known values for verification
    outcome_values     = np.array([10.0] * 100 + [12.0] * 100)  # SD ≈ 1.0
    pre_outcome_values = np.array([10.0] * 50)                   # mean = 10.0
    estimate = 2.0

    result = compute_effect_sizes(estimate, outcome_values, pre_outcome_values)

    assert 'cohens_d' in result, "Missing 'cohens_d' key."
    assert 'pct_change' in result, "Missing 'pct_change' key."
    assert 'magnitude_label' in result, "Missing 'magnitude_label' key."
    assert 'baseline_mean' in result, "Missing 'baseline_mean' key."
    assert 'outcome_sd' in result, "Missing 'outcome_sd' key."

    assert abs(result['baseline_mean'] - 10.0) < 1e-6, \
        f"Baseline mean should be 10.0. Got {result['baseline_mean']}"

    assert abs(result['pct_change'] - 20.0) < 1e-3, \
        f"% change should be 20.0% (2.0 / 10.0). Got {result['pct_change']}"

    # Cohen's d: d = 2.0 / SD; SD of [10]*100 + [12]*100 ~ 1.0
    # Just check it's positive and in a reasonable range
    assert result['cohens_d'] > 0, "Cohen's d should be positive for positive estimate."

    # Magnitude labels
    small_result = compute_effect_sizes(0.1, outcome_values, pre_outcome_values)
    assert small_result['magnitude_label'] == 'small', \
        f"d < 0.5 should be 'small'. Got '{small_result['magnitude_label']}'"

    large_result = compute_effect_sizes(10.0, outcome_values, pre_outcome_values)
    assert large_result['magnitude_label'] == 'large', \
        f"Large d should be 'large'. Got '{large_result['magnitude_label']}'"

    print("  Exercise 3 PASSED — compute_effect_sizes() works correctly.")


# ─────────────────────────────────────────────────────────────────────────────
# Exercise 4: Drift Monitor
# ─────────────────────────────────────────────────────────────────────────────

class CausalDriftMonitor:
    """
    Monitor a causal estimate for drift relative to a baseline.

    Usage:
        monitor = CausalDriftMonitor(baseline_estimate=3.0, baseline_se=0.5)
        status  = monitor.check(new_estimate=3.1)
        # status['stable'] == True

    A new estimate is considered stable if it is within `z_threshold` baseline
    standard errors of the baseline estimate.
    """

    def __init__(self, baseline_estimate: float, baseline_se: float, z_threshold: float = 2.0):
        """
        Parameters
        ----------
        baseline_estimate : the reference point estimate (from the original run)
        baseline_se       : the standard error of the baseline estimate
        z_threshold       : alert if |z_score| >= z_threshold (default 2.0)
        """
        # ── YOUR CODE ─────────────────────────────────────────────────────────
        raise NotImplementedError("Implement CausalDriftMonitor.__init__()")
        # ── END YOUR CODE ─────────────────────────────────────────────────────

    def check(self, new_estimate: float) -> Dict:
        """
        Check whether a new estimate has drifted from baseline.

        Returns a dict with keys:
            'stable'     (bool): True if z_score < z_threshold
            'z_score'    (float): |new - baseline| / baseline_se
            'delta'      (float): new_estimate - baseline_estimate
            'alert'      (bool): True if drift is detected (not stable)
            'baseline'   (float): stored baseline
            'new'        (float): new_estimate
        """
        # ── YOUR CODE ─────────────────────────────────────────────────────────
        raise NotImplementedError("Implement CausalDriftMonitor.check()")
        # ── END YOUR CODE ─────────────────────────────────────────────────────


def _test_causal_drift_monitor():
    """Exercise 4 tests."""
    monitor = CausalDriftMonitor(baseline_estimate=3.0, baseline_se=0.5, z_threshold=2.0)

    # ── Test 4a: stable estimate ──────────────────────────────────────────────
    status_stable = monitor.check(new_estimate=3.2)
    assert 'stable' in status_stable, "Result must have 'stable' key."
    assert 'z_score' in status_stable, "Result must have 'z_score' key."
    assert 'delta' in status_stable, "Result must have 'delta' key."
    assert 'alert' in status_stable, "Result must have 'alert' key."

    # z = |3.2 - 3.0| / 0.5 = 0.4 < 2.0 → stable
    assert status_stable['stable'], \
        f"3.2 should be stable vs baseline 3.0 (SE=0.5). z = {status_stable['z_score']:.2f}"
    assert not status_stable['alert'], "No alert expected for stable estimate."
    assert abs(status_stable['z_score'] - 0.4) < 1e-6, \
        f"z_score should be 0.4. Got {status_stable['z_score']}"
    assert abs(status_stable['delta'] - 0.2) < 1e-6, \
        f"delta should be 0.2. Got {status_stable['delta']}"

    # ── Test 4b: drifting estimate ────────────────────────────────────────────
    status_drift = monitor.check(new_estimate=5.0)
    # z = |5.0 - 3.0| / 0.5 = 4.0 >= 2.0 → alert
    assert not status_drift['stable'], \
        f"5.0 should be drifting vs baseline 3.0 (SE=0.5). z = {status_drift['z_score']:.2f}"
    assert status_drift['alert'], "Alert expected for drifting estimate."
    assert abs(status_drift['z_score'] - 4.0) < 1e-6, \
        f"z_score should be 4.0. Got {status_drift['z_score']}"

    # ── Test 4c: custom threshold ─────────────────────────────────────────────
    strict_monitor = CausalDriftMonitor(baseline_estimate=3.0, baseline_se=0.5, z_threshold=1.0)
    status_strict = strict_monitor.check(new_estimate=3.6)
    # z = |3.6 - 3.0| / 0.5 = 1.2 >= 1.0 → alert
    assert status_strict['alert'], \
        f"3.6 should trigger alert with z_threshold=1.0. z = {status_strict['z_score']:.2f}"

    print("  Exercise 4 PASSED — CausalDriftMonitor works correctly.")


# ─────────────────────────────────────────────────────────────────────────────
# Exercise 5: Retraining Decision Logic
# ─────────────────────────────────────────────────────────────────────────────

def decide_retraining_action(
    drift_status: Dict,
    pre_trend_result: Dict,
    new_data_available: bool,
) -> Dict[str, object]:
    """
    Decide what action a production causal pipeline should take based on
    monitoring results.

    Rules (apply in order of priority):
    1. If pre_trend_result['warning'] is True → action = 'FLAG_FOR_REVIEW',
       reason = 'Pre-trend assumption may be violated.'
    2. If drift_status['alert'] is True → action = 'HUMAN_INVESTIGATION',
       reason = 'Estimate has drifted more than threshold from baseline.'
    3. If new_data_available is True and no warnings → action = 'AUTO_REESTIMATE',
       reason = 'New data available; assumptions pass.'
    4. Otherwise → action = 'NO_ACTION', reason = 'No trigger conditions met.'

    Returns a dict with keys:
        'action'        (str): one of the four action strings above
        'reason'        (str): human-readable explanation
        'auto_safe'     (bool): True only when action is 'AUTO_REESTIMATE'

    Parameters
    ----------
    drift_status        : output of CausalDriftMonitor.check()
    pre_trend_result    : output of test_parallel_trends()
    new_data_available  : whether a new period of data has arrived
    """
    # ── YOUR CODE ─────────────────────────────────────────────────────────────
    raise NotImplementedError("Implement decide_retraining_action()")
    # ── END YOUR CODE ─────────────────────────────────────────────────────────


def _test_decide_retraining_action():
    """Exercise 5 tests."""
    # Helpers to construct minimal status dicts
    stable_drift   = {'alert': False, 'stable': True, 'z_score': 0.3}
    drifting_drift = {'alert': True,  'stable': False, 'z_score': 3.1}
    pass_pt        = {'warning': False, 'passed': True, 'interaction_p': 0.3}
    fail_pt        = {'warning': True,  'passed': False, 'interaction_p': 0.01}

    # ── Test 5a: pre-trend failure → FLAG_FOR_REVIEW (highest priority) ──────
    result_a = decide_retraining_action(stable_drift, fail_pt, new_data_available=True)
    assert result_a['action'] == 'FLAG_FOR_REVIEW', \
        f"Pre-trend failure should trigger FLAG_FOR_REVIEW. Got: {result_a['action']}"
    assert result_a['auto_safe'] is False, \
        "auto_safe must be False for FLAG_FOR_REVIEW."

    # ── Test 5b: drift → HUMAN_INVESTIGATION ─────────────────────────────────
    result_b = decide_retraining_action(drifting_drift, pass_pt, new_data_available=True)
    assert result_b['action'] == 'HUMAN_INVESTIGATION', \
        f"Drift should trigger HUMAN_INVESTIGATION. Got: {result_b['action']}"
    assert result_b['auto_safe'] is False, \
        "auto_safe must be False for HUMAN_INVESTIGATION."

    # ── Test 5c: new data, no issues → AUTO_REESTIMATE ───────────────────────
    result_c = decide_retraining_action(stable_drift, pass_pt, new_data_available=True)
    assert result_c['action'] == 'AUTO_REESTIMATE', \
        f"New data + no issues should trigger AUTO_REESTIMATE. Got: {result_c['action']}"
    assert result_c['auto_safe'] is True, \
        "auto_safe must be True for AUTO_REESTIMATE."

    # ── Test 5d: no new data, no issues → NO_ACTION ──────────────────────────
    result_d = decide_retraining_action(stable_drift, pass_pt, new_data_available=False)
    assert result_d['action'] == 'NO_ACTION', \
        f"No issues + no new data should be NO_ACTION. Got: {result_d['action']}"
    assert result_d['auto_safe'] is False, \
        "auto_safe must be False for NO_ACTION."

    # ── Test 5e: both pre-trend failure AND drift → pre-trend takes priority ──
    result_e = decide_retraining_action(drifting_drift, fail_pt, new_data_available=True)
    assert result_e['action'] == 'FLAG_FOR_REVIEW', \
        (f"When both pre-trend and drift fail, FLAG_FOR_REVIEW should take priority. "
         f"Got: {result_e['action']}")

    print("  Exercise 5 PASSED — decide_retraining_action() works correctly.")


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Module 07 — Production Pipelines: Self-Check')
    print('=' * 60)

    exercises = [
        ('Exercise 1: Data Validation',      _test_validate_did_data),
        ('Exercise 2: Pre-Trend Test',        _test_parallel_trends),
        ('Exercise 3: Effect Sizes',          _test_compute_effect_sizes),
        ('Exercise 4: Drift Monitor',         _test_causal_drift_monitor),
        ('Exercise 5: Retraining Decision',   _test_decide_retraining_action),
    ]

    passed = 0
    for name, test_fn in exercises:
        print(f'\n{name}')
        print('-' * 40)
        try:
            test_fn()
            passed += 1
        except NotImplementedError as e:
            print(f'  NOT STARTED — {e}')
        except AssertionError as e:
            print(f'  FAILED — {e}')
        except Exception as e:
            print(f'  ERROR — {type(e).__name__}: {e}')

    print()
    print('=' * 60)
    print(f'Results: {passed}/{len(exercises)} exercises passed.')
    if passed == len(exercises):
        print('All exercises complete. You are ready for the production pipeline.')
    else:
        remaining = len(exercises) - passed
        print(f'{remaining} exercise(s) remaining.')
