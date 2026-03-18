"""
Module 08 Self-Check: Production Nowcasting Systems

Four tasks covering:
  Task 1 — Vintage database as-of query
  Task 2 — Ragged-edge fill comparison
  Task 3 — CUSUM break detection
  Task 4 — Re-estimation trigger logic

Run with:
    python exercises/01_production_self_check.py

All tasks use synthetic data. No internet access required.
"""

import numpy as np
import pandas as pd
import sqlite3
import datetime
from typing import Dict, List, Optional

np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ──────────────────────────────────────────────────────────────────────────────

def make_monthly_series(n: int = 36, seed: int = 0) -> pd.Series:
    """AR1 monthly series, 2020-01 through 2022-12."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2020-01-01', periods=n, freq='MS')
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.7 * y[t-1] + rng.normal(0, 0.5)
    return pd.Series(y, index=dates, name='TEST')


# ──────────────────────────────────────────────────────────────────────────────
# TASK 1 — Vintage database as-of query
# ──────────────────────────────────────────────────────────────────────────────
# Goal: implement a vintage database with:
#   - store_snapshot(series_id, data, vintage_date): insert a snapshot
#   - get_as_of(series_id, as_of_date): return the latest vintage available
#     as of as_of_date for each observation date
#
# Rules:
#   - Rows should never be deleted or modified (INSERT OR REPLACE is fine)
#   - get_as_of must filter: vintage_date <= as_of_date
#   - get_as_of must deduplicate: for each obs_date, keep the row with the
#     most recent vintage_date that is still <= as_of_date

class VintageDB:
    def __init__(self):
        self.conn = sqlite3.connect(':memory:')
        self.conn.execute("""
            CREATE TABLE vintages (
                series_id    TEXT,
                obs_date     TEXT,
                vintage_date TEXT,
                value        REAL,
                PRIMARY KEY (series_id, obs_date, vintage_date)
            )
        """)
        self.conn.commit()

    def store_snapshot(self, series_id: str, data: pd.Series, vintage_date: str) -> None:
        """
        TODO: Insert all (series_id, obs_date, vintage_date, value) rows
              from `data` into the vintages table.
              Use the vintage_date string as-is.
        """
        raise NotImplementedError("Implement store_snapshot")

    def get_as_of(self, series_id: str, as_of_date: str) -> pd.Series:
        """
        TODO: Return a pd.Series with DatetimeIndex of obs_date values,
              containing the most recent value for each obs_date
              where vintage_date <= as_of_date.
        """
        raise NotImplementedError("Implement get_as_of")


def test_task1():
    print("\n=== Task 1: Vintage Database ===")

    series = make_monthly_series(n=24, seed=1)
    # Revision: the first 12 obs get slightly different values in the second vintage
    rng = np.random.default_rng(99)
    series_revised = series.copy()
    series_revised.iloc[:12] += rng.normal(0, 0.1, 12)

    db = VintageDB()
    db.store_snapshot('TEST', series, vintage_date='2022-01-15')
    db.store_snapshot('TEST', series_revised, vintage_date='2022-02-15')

    # As of 2022-01-20: should see first vintage (pre-revision)
    as_of_early = db.get_as_of('TEST', '2022-01-20')
    assert isinstance(as_of_early, pd.Series), "get_as_of must return a pd.Series"
    assert len(as_of_early) > 0, "as_of_early must be non-empty"

    # Check a specific obs_date that exists in both vintages
    first_obs = series.index[0]
    val_early = as_of_early.get(first_obs, None)
    assert val_early is not None, f"obs_date {first_obs} not in as_of_early"
    expected_early = float(series.iloc[0])
    assert abs(val_early - expected_early) < 1e-9, (
        f"As of 2022-01-20, obs 0 should be {expected_early:.6f} "
        f"(first vintage), got {val_early:.6f}"
    )

    # As of 2022-02-20: should see second vintage for obs 0
    as_of_late = db.get_as_of('TEST', '2022-02-20')
    val_late = as_of_late.get(first_obs, None)
    assert val_late is not None, f"obs_date {first_obs} not in as_of_late"
    expected_late = float(series_revised.iloc[0])
    assert abs(val_late - expected_late) < 1e-9, (
        f"As of 2022-02-20, obs 0 should be {expected_late:.6f} "
        f"(revised vintage), got {val_late:.6f}"
    )

    # The two values should be different (revision exists)
    assert abs(val_late - val_early) > 1e-6, (
        "Revision noise was not applied; preliminary and revised values are identical"
    )

    # Count: as_of_early and as_of_late should have the same number of obs
    assert len(as_of_early) == len(as_of_late), (
        "Both as-of queries should return the same number of observations"
    )

    print("PASS — VintageDB correctly tracks preliminary vs revised vintages")
    print(f"  First obs preliminary: {val_early:.4f}")
    print(f"  First obs revised:     {val_late:.4f}")
    print(f"  Revision:              {val_late - val_early:+.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# TASK 2 — Ragged-edge fill comparison
# ──────────────────────────────────────────────────────────────────────────────
# Goal: implement fill_ragged_edge(series, target_end, method) that extends
#       a monthly series to target_end using one of three methods:
#         'carry_forward' : ffill
#         'zero'          : fill with 0.0
#         'ar1'           : OLS AR1 extrapolation
#
# Then verify that the three methods give different results and that all
# produce a series that ends at (or after) target_end.

def fill_ragged_edge(
    series: pd.Series,
    target_end: pd.Timestamp,
    method: str = 'carry_forward',
) -> pd.Series:
    """
    TODO: Extend series to target_end using the specified fill method.
          Use pd.date_range(series.index[0], target_end, freq='MS') to build
          the full index, then fill missing values by the chosen method.

    ar1 method:
      - Fit OLS AR1 on the last min(24, len(observed)) observations
      - Extrapolate forward for each missing period
      - Use np.linalg.lstsq to fit [1, y_{t-1}] → y_t
    """
    raise NotImplementedError("Implement fill_ragged_edge")


def test_task2():
    print("\n=== Task 2: Ragged-Edge Fill Methods ===")

    series = make_monthly_series(n=20, seed=2)
    # Truncate to 18 obs; need to fill 2 months to 2021-08
    truncated = series.iloc[:18].copy()
    target_end = pd.Timestamp('2021-08-01')

    cf = fill_ragged_edge(truncated, target_end, method='carry_forward')
    zf = fill_ragged_edge(truncated, target_end, method='zero')
    ar = fill_ragged_edge(truncated, target_end, method='ar1')

    # All methods should end at target_end or later
    for name, filled in [('carry_forward', cf), ('zero', zf), ('ar1', ar)]:
        assert filled.index[-1] >= target_end, (
            f"{name}: last index {filled.index[-1]} is before target_end {target_end}"
        )
        assert not filled.isna().any(), (
            f"{name}: contains NaN after fill"
        )
        # Historical values (first 18) should be unchanged
        np.testing.assert_array_almost_equal(
            filled.iloc[:18].values,
            truncated.values,
            decimal=10,
            err_msg=f"{name}: historical values were modified"
        )

    # Methods should differ on the two filled periods
    filled_cf = cf.iloc[18:].values
    filled_zf = zf.iloc[18:].values
    filled_ar = ar.iloc[18:].values

    # Zero fill should be exactly 0
    np.testing.assert_array_almost_equal(
        filled_zf, np.zeros(2), decimal=10,
        err_msg="zero fill should produce exactly 0.0"
    )

    # Carry-forward should equal the last observed value
    last_obs = float(truncated.iloc[-1])
    np.testing.assert_array_almost_equal(
        filled_cf, np.full(2, last_obs), decimal=10,
        err_msg=f"carry-forward should repeat last obs = {last_obs:.4f}"
    )

    # AR1 should differ from both carry-forward and zero
    assert not np.allclose(filled_ar, filled_cf, atol=1e-6), (
        "AR1 fill should produce different values than carry-forward "
        "(unless the AR1 coefficient is exactly 1.0)"
    )
    assert not np.allclose(filled_ar, filled_zf, atol=1e-6), (
        "AR1 fill should produce different values than zero fill "
        "(unless the AR1 mean is exactly 0)"
    )

    print("PASS — All three fill methods produce correct results")
    print(f"  Carry-forward: {filled_cf}")
    print(f"  Zero fill:     {filled_zf}")
    print(f"  AR1 projection:{filled_ar}")


# ──────────────────────────────────────────────────────────────────────────────
# TASK 3 — CUSUM break detection
# ──────────────────────────────────────────────────────────────────────────────
# Goal: implement cusum_detect(errors) that:
#   1. Standardises errors by (e - mean) / std
#   2. Computes the cumulative sum
#   3. Applies 5% boundary: ±0.948 * sqrt(n) * (1 + 2*t/n)
#   4. Returns dict with: cusum, any_crossing (bool), break_index (int or None)
#
# Verify on synthetic data where a mean shift is injected at a known point.

def cusum_detect(errors: np.ndarray) -> Dict:
    """
    TODO: Implement the simplified CUSUM test.
    """
    raise NotImplementedError("Implement cusum_detect")


def test_task3():
    print("\n=== Task 3: CUSUM Structural Break Detection ===")

    rng = np.random.default_rng(7)
    n = 24
    break_at = 14

    # Errors: zero mean before break, +1.2 mean after break
    errors_pre = rng.normal(0.0, 0.4, break_at)
    errors_post = rng.normal(1.2, 0.4, n - break_at)
    errors = np.concatenate([errors_pre, errors_post])

    result = cusum_detect(errors)

    assert isinstance(result, dict), "cusum_detect must return a dict"
    assert 'cusum' in result, "result must have 'cusum' key"
    assert 'any_crossing' in result, "result must have 'any_crossing' key"
    assert 'break_index' in result, "result must have 'break_index' key"

    cusum = np.asarray(result['cusum'])
    assert len(cusum) == n, (
        f"cusum must have length {n}, got {len(cusum)}"
    )

    assert result['any_crossing'] is True, (
        "CUSUM should detect a break with a mean shift of +1.2 sigma over 10 periods"
    )

    assert result['break_index'] is not None, (
        "break_index must be non-None when any_crossing is True"
    )

    # Break should be detected reasonably close to the true break
    # Allow 4 periods of lag (typical for CUSUM)
    detected = result['break_index']
    assert detected >= break_at - 2, (
        f"Break detected too early: {detected} vs true break {break_at}"
    )
    assert detected <= break_at + 6, (
        f"Break detected too late: {detected} vs true break {break_at}. "
        f"CUSUM should flag within 6 quarters of the true break."
    )

    # No-break case: should not flag
    errors_stable = rng.normal(0.0, 0.4, n)
    result_stable = cusum_detect(errors_stable)
    # We cannot guarantee no crossing for a short stable series,
    # but the probability is 5% under H0; just check the return format
    assert 'any_crossing' in result_stable
    assert 'break_index' in result_stable

    print("PASS — CUSUM correctly detects mean shift")
    print(f"  True break at index:     {break_at}")
    print(f"  CUSUM detected at index: {detected}")
    print(f"  Detection lag: {detected - break_at} quarters")


# ──────────────────────────────────────────────────────────────────────────────
# TASK 4 — Re-estimation trigger
# ──────────────────────────────────────────────────────────────────────────────
# Goal: implement a ReEstimationTrigger class with method:
#   check(current_rmse, cusum_break) -> dict with 'fired' (bool) and 'reason' (str)
#
# Trigger fires when ANY of:
#   a) calendar_quarters have elapsed since last re-estimation
#   b) current_rmse > backtest_rmse * (1 + rmse_threshold) for TWO consecutive calls
#   c) cusum_break is True
#
# After a trigger fires, the calendar counter resets to 0.
# The consecutive-exceedance counter resets to 0 whenever current_rmse
# falls back below the threshold.

class ReEstimationTrigger:
    """
    TODO: Implement the three-trigger re-estimation monitor.

    Parameters
    ----------
    backtest_rmse : float
    rmse_threshold : float  (e.g. 0.20 = 20%)
    calendar_quarters : int  (re-estimate at least every N quarters)
    """

    def __init__(
        self,
        backtest_rmse: float,
        rmse_threshold: float = 0.20,
        calendar_quarters: int = 4,
    ):
        raise NotImplementedError("Implement __init__")

    def check(self, current_rmse: float, cusum_break: bool = False) -> Dict:
        """
        TODO: Implement the trigger check.
        Returns dict with 'fired' (bool) and 'reason' (str).
        """
        raise NotImplementedError("Implement check")


def test_task4():
    print("\n=== Task 4: Re-Estimation Trigger ===")

    backtest_rmse = 0.40
    trigger = ReEstimationTrigger(
        backtest_rmse=backtest_rmse,
        rmse_threshold=0.20,   # threshold = 0.48
        calendar_quarters=4,
    )

    # Calls 1-3: below threshold, no CUSUM break
    for i in range(3):
        r = trigger.check(current_rmse=0.42, cusum_break=False)
        assert r['fired'] is False, (
            f"Call {i+1}: trigger should not fire (RMSE 0.42 < threshold 0.48, "
            f"and {i+1} quarters < calendar threshold 4)"
        )

    # Call 4: calendar trigger should fire (4 quarters elapsed)
    r4 = trigger.check(current_rmse=0.42, cusum_break=False)
    assert r4['fired'] is True, (
        "Calendar trigger should fire after 4 quarters"
    )
    assert 'calendar' in r4['reason'].lower(), (
        f"Reason should mention 'calendar', got: '{r4['reason']}'"
    )

    # After firing, counter resets. Call 1-2 with high RMSE (performance trigger)
    r_p1 = trigger.check(current_rmse=0.55, cusum_break=False)  # 1st exceedance
    assert r_p1['fired'] is False, (
        "Performance trigger requires 2 consecutive exceedances; should not fire on first"
    )

    r_p2 = trigger.check(current_rmse=0.55, cusum_break=False)  # 2nd exceedance
    assert r_p2['fired'] is True, (
        "Performance trigger should fire on 2nd consecutive RMSE exceedance "
        f"(0.55 > threshold {backtest_rmse * 1.20:.3f})"
    )
    assert 'performance' in r_p2['reason'].lower(), (
        f"Reason should mention 'performance', got: '{r_p2['reason']}'"
    )

    # CUSUM break trigger: fires immediately
    trigger2 = ReEstimationTrigger(backtest_rmse=0.40, calendar_quarters=4)
    r_cusum = trigger2.check(current_rmse=0.42, cusum_break=True)
    assert r_cusum['fired'] is True, (
        "CUSUM break trigger should fire immediately when cusum_break=True"
    )
    assert 'cusum' in r_cusum['reason'].lower() or 'break' in r_cusum['reason'].lower(), (
        f"Reason should mention 'cusum' or 'break', got: '{r_cusum['reason']}'"
    )

    # Consecutive exceedance resets when RMSE drops back below threshold
    trigger3 = ReEstimationTrigger(backtest_rmse=0.40, calendar_quarters=10)
    r_h1 = trigger3.check(current_rmse=0.55)   # 1st exceedance
    assert r_h1['fired'] is False

    r_drop = trigger3.check(current_rmse=0.41)  # drops below threshold → reset
    assert r_drop['fired'] is False

    r_h2 = trigger3.check(current_rmse=0.55)   # 1st exceedance again (counter reset)
    assert r_h2['fired'] is False, (
        "After RMSE drops below threshold, the consecutive-exceedance counter should "
        "reset to 0. The next exceedance should be the first of a new consecutive run."
    )

    r_h3 = trigger3.check(current_rmse=0.55)   # 2nd exceedance
    assert r_h3['fired'] is True, (
        "Should fire on 2nd exceedance after reset"
    )

    print("PASS — ReEstimationTrigger fires correctly for all three trigger types")
    print(f"  Calendar trigger fires after 4 quarters")
    print(f"  Performance trigger fires after 2 consecutive RMSE > {backtest_rmse * 1.20:.3f}")
    print(f"  CUSUM trigger fires immediately on structural break")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    passed = 0
    failed = 0
    errors = []

    for task_fn in [test_task1, test_task2, test_task3, test_task4]:
        try:
            task_fn()
            passed += 1
        except NotImplementedError as e:
            failed += 1
            errors.append(f"{task_fn.__name__}: not implemented — {e}")
        except AssertionError as e:
            failed += 1
            errors.append(f"{task_fn.__name__}: FAIL — {e}")
        except Exception as e:
            failed += 1
            errors.append(f"{task_fn.__name__}: ERROR — {type(e).__name__}: {e}")

    print(f"\n{'='*50}")
    print(f"Results: {passed}/4 tasks passed")
    if errors:
        print("\nFailures:")
        for err in errors:
            print(f"  {err}")
    else:
        print("All tasks complete. Module 08 production systems verified.")
    print('='*50)
