"""
Module 01: Point Forecasting — Self-Check Exercises

These exercises reinforce the key concepts from Guide 02 (Hyperparameter Tuning)
and Notebooks 01–02. Work through each exercise in order.

Prerequisites:
  pip install neuralforecast utilsforecast matplotlib pandas

Run the file to check your answers:
  python 01_point_forecasting_exercises.py

Each exercise has an assert-based check. When an assert passes, you see a
success message. When it fails, the message explains what to fix.
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Data loading helper — shared by all exercises
# ---------------------------------------------------------------------------

def load_bakery_data() -> pd.DataFrame:
    """Load French Bakery data in Nixtla long format."""
    url = (
        "https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series"
        "/main/datasets/french_bakery.csv"
    )
    df = pd.read_csv(url, parse_dates=["ds"])
    return df


# ===========================================================================
# EXERCISE 1: Impact of input_size on Forecast Quality
# ===========================================================================
# Goal: Confirm that input_size=28 (4x horizon) outperforms input_size=7 (1x)
#       on the French Bakery dataset.
#
# Instructions:
#   1. Train NHITS with input_size=7 and input_size=28 (both with h=7)
#   2. Use cross_validation(n_windows=2) for each configuration
#   3. Compute overall MAE for each
#   4. Store results in mae_input7 and mae_input28

def exercise_1_input_size_comparison():
    """
    Train NHITS with two different input_size values and compare MAE.

    Returns
    -------
    mae_input7 : float
        Mean absolute error for NHITS with input_size=7
    mae_input28 : float
        Mean absolute error for NHITS with input_size=28
    """
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS
    from utilsforecast.losses import mae

    df = load_bakery_data()

    # -----------------------------------------------------------------------
    # YOUR CODE HERE
    # -----------------------------------------------------------------------
    # Hint: NeuralForecast(models=[NHITS(h=7, input_size=<value>, max_steps=300)], freq="D")
    # Hint: nf.cross_validation(df, n_windows=2)
    # Hint: mae(cv_df["y"], cv_df["NHITS"]).mean()

    mae_input7 = None   # replace with your computed value
    mae_input28 = None  # replace with your computed value

    return mae_input7, mae_input28


def check_exercise_1():
    print("Exercise 1: input_size comparison")
    print("-" * 50)

    mae_input7, mae_input28 = exercise_1_input_size_comparison()

    assert mae_input7 is not None, (
        "mae_input7 is None. "
        "Train NHITS with input_size=7 and assign the cross-validation MAE."
    )
    assert mae_input28 is not None, (
        "mae_input28 is None. "
        "Train NHITS with input_size=28 and assign the cross-validation MAE."
    )
    assert isinstance(mae_input7, (float, np.floating)), (
        f"mae_input7 should be a float, got {type(mae_input7)}. "
        "Use mae(cv_df['y'], cv_df['NHITS']).mean() to compute it."
    )
    assert isinstance(mae_input28, (float, np.floating)), (
        f"mae_input28 should be a float, got {type(mae_input28)}. "
        "Use mae(cv_df['y'], cv_df['NHITS']).mean() to compute it."
    )
    assert mae_input7 > 0 and mae_input28 > 0, (
        "MAE values must be positive. Check that you are comparing 'y' vs 'NHITS'."
    )
    assert mae_input28 < mae_input7, (
        f"Expected input_size=28 (MAE={mae_input28:.2f}) to outperform "
        f"input_size=7 (MAE={mae_input7:.2f}). "
        "A larger lookback window gives NHITS more context to learn the weekly pattern. "
        "If this fails, increase max_steps to 500 or check that you are using the same n_windows."
    )

    improvement = (mae_input7 - mae_input28) / mae_input7 * 100
    print(f"  input_size=7:  MAE = {mae_input7:.2f}")
    print(f"  input_size=28: MAE = {mae_input28:.2f}")
    print(f"  Improvement: {improvement:.1f}%")
    print("  PASSED")
    print()


# ===========================================================================
# EXERCISE 2: Comparing scaler_type Options
# ===========================================================================
# Goal: Compare "standard", "robust", and no scaler (None) on bakery data.
#
# Instructions:
#   1. Train three NHITS models, one for each scaler_type
#   2. Use cross_validation(n_windows=2) for each
#   3. Store results in a dict: {scaler_name: mae_value}
#   4. Identify which scaler produces the lowest MAE

def exercise_2_scaler_comparison():
    """
    Compare NHITS with three different scaler_type values.

    Returns
    -------
    scaler_results : dict
        Keys are scaler names: "standard", "robust", "none"
        Values are the cross-validation MAE for each
    best_scaler : str
        The scaler name that produced the lowest MAE
    """
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS
    from utilsforecast.losses import mae

    df = load_bakery_data()

    # -----------------------------------------------------------------------
    # YOUR CODE HERE
    # -----------------------------------------------------------------------
    # Hint: scaler_type accepts "standard", "robust", or None (no scaling)
    # Hint: Store results like: scaler_results = {"standard": 18.3, "robust": 16.1, "none": 21.4}
    # Hint: best_scaler = min(scaler_results, key=scaler_results.get)

    scaler_results = {}  # replace with your results
    best_scaler = None   # replace with the name of the best scaler

    return scaler_results, best_scaler


def check_exercise_2():
    print("Exercise 2: scaler_type comparison")
    print("-" * 50)

    scaler_results, best_scaler = exercise_2_scaler_comparison()

    assert isinstance(scaler_results, dict), (
        "scaler_results must be a dict mapping scaler name -> MAE float."
    )
    assert len(scaler_results) == 3, (
        f"Expected 3 entries in scaler_results (standard, robust, none), "
        f"got {len(scaler_results)}. "
        "Include all three scaler options."
    )
    required_keys = {"standard", "robust", "none"}
    assert set(scaler_results.keys()) == required_keys, (
        f"Expected keys {required_keys}, got {set(scaler_results.keys())}. "
        "Use 'none' (string) for the None scaler case."
    )
    for name, score in scaler_results.items():
        assert isinstance(score, (float, np.floating)) and score > 0, (
            f"scaler_results['{name}'] = {score!r}. Must be a positive float."
        )
    assert best_scaler in required_keys, (
        f"best_scaler must be one of {required_keys}, got {best_scaler!r}."
    )
    assert scaler_results[best_scaler] == min(scaler_results.values()), (
        f"best_scaler='{best_scaler}' (MAE={scaler_results[best_scaler]:.2f}) "
        f"is not the minimum. Check your min() logic."
    )

    for name, score in sorted(scaler_results.items(), key=lambda x: x[1]):
        marker = " <-- best" if name == best_scaler else ""
        print(f"  scaler_type={name!r:12s}  MAE = {score:.2f}{marker}")
    print(f"  Best scaler: {best_scaler!r}")
    print("  PASSED")
    print()


# ===========================================================================
# EXERCISE 3: NHITS vs DLinear Comparison
# ===========================================================================
# Goal: Determine whether NHITS outperforms the DLinear baseline on this dataset.
#       Report the % improvement.
#
# Instructions:
#   1. Train DLinear and NHITS (with your best scaler from Exercise 2, or "robust")
#   2. Use cross_validation(n_windows=3) for both
#   3. Compute MAE for each
#   4. Compute nhits_improvement_pct = (mae_dlinear - mae_nhits) / mae_dlinear * 100

def exercise_3_nhits_vs_baseline():
    """
    Compare NHITS against the DLinear baseline with cross-validation.

    Returns
    -------
    mae_dlinear : float
        Cross-validation MAE for DLinear
    mae_nhits : float
        Cross-validation MAE for NHITS
    nhits_improvement_pct : float
        Improvement of NHITS over DLinear, as a percentage.
        Positive means NHITS is better. Negative means DLinear is better.
    """
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS, DLinear
    from utilsforecast.losses import mae

    df = load_bakery_data()

    # -----------------------------------------------------------------------
    # YOUR CODE HERE
    # -----------------------------------------------------------------------
    # Hint: NeuralForecast(models=[DLinear(...), NHITS(...)], freq="D")
    # Hint: Both models train in a single cross_validation() call
    # Hint: cv_df will have columns 'DLinear' and 'NHITS'
    # Hint: nhits_improvement_pct = (mae_dlinear - mae_nhits) / mae_dlinear * 100

    mae_dlinear = None
    mae_nhits = None
    nhits_improvement_pct = None

    return mae_dlinear, mae_nhits, nhits_improvement_pct


def check_exercise_3():
    print("Exercise 3: NHITS vs DLinear")
    print("-" * 50)

    mae_dlinear, mae_nhits, nhits_improvement_pct = exercise_3_nhits_vs_baseline()

    assert mae_dlinear is not None, (
        "mae_dlinear is None. Train DLinear and compute its cross-validation MAE."
    )
    assert mae_nhits is not None, (
        "mae_nhits is None. Train NHITS and compute its cross-validation MAE."
    )
    assert nhits_improvement_pct is not None, (
        "nhits_improvement_pct is None. "
        "Compute: (mae_dlinear - mae_nhits) / mae_dlinear * 100"
    )
    assert isinstance(mae_dlinear, (float, np.floating)) and mae_dlinear > 0, (
        f"mae_dlinear must be a positive float, got {mae_dlinear!r}."
    )
    assert isinstance(mae_nhits, (float, np.floating)) and mae_nhits > 0, (
        f"mae_nhits must be a positive float, got {mae_nhits!r}."
    )
    assert isinstance(nhits_improvement_pct, (float, np.floating)), (
        f"nhits_improvement_pct must be a float, got {type(nhits_improvement_pct)}."
    )

    # Verify the improvement formula is correct
    expected_pct = (mae_dlinear - mae_nhits) / mae_dlinear * 100
    assert abs(nhits_improvement_pct - expected_pct) < 0.1, (
        f"nhits_improvement_pct={nhits_improvement_pct:.2f} does not match "
        f"the expected formula result {expected_pct:.2f}. "
        "Use: (mae_dlinear - mae_nhits) / mae_dlinear * 100"
    )

    print(f"  DLinear MAE:  {mae_dlinear:.2f}")
    print(f"  NHITS MAE:    {mae_nhits:.2f}")
    print(f"  Improvement:  {nhits_improvement_pct:+.1f}%")
    if nhits_improvement_pct > 5:
        print("  NHITS clearly beats the linear baseline on this dataset.")
    elif nhits_improvement_pct > 0:
        print("  NHITS slightly better. Try tuning input_size or increasing max_steps.")
    else:
        print("  DLinear is competitive here. The signal may be largely linear.")
    print("  PASSED")
    print()


# ===========================================================================
# EXERCISE 4 (Extension): Best input_size via Cross-Validation
# ===========================================================================
# Goal: Systematically find the best input_size from [14, 28, 56, 112].
#
# Instructions:
#   1. For each candidate input_size, train NHITS and compute cross-validation MAE
#   2. Store results in a dict: {input_size: mae_value}
#   3. Identify the best_input_size (the one with lowest MAE)
#   4. Verify best_input_size is at least 2x the horizon (h=7)

def exercise_4_best_input_size():
    """
    Find the best input_size from [14, 28, 56, 112] using cross-validation.

    Returns
    -------
    input_size_results : dict
        Keys are int input_size values; values are cross-validation MAE
    best_input_size : int
        The input_size with the lowest cross-validation MAE
    """
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS
    from utilsforecast.losses import mae

    df = load_bakery_data()
    candidates = [14, 28, 56, 112]

    # -----------------------------------------------------------------------
    # YOUR CODE HERE
    # -----------------------------------------------------------------------
    # Hint: Loop over candidates, train NHITS for each, compute CV MAE
    # Hint: Use n_windows=2 to keep training time manageable
    # Hint: best_input_size = min(input_size_results, key=input_size_results.get)

    input_size_results = {}  # replace
    best_input_size = None   # replace

    return input_size_results, best_input_size


def check_exercise_4():
    print("Exercise 4 (Extension): Best input_size")
    print("-" * 50)

    input_size_results, best_input_size = exercise_4_best_input_size()

    if not input_size_results:
        print("  Skipped (no results returned). Implement exercise_4_best_input_size().")
        return

    candidates = [14, 28, 56, 112]
    assert isinstance(input_size_results, dict), (
        "input_size_results must be a dict mapping int -> float."
    )
    assert set(input_size_results.keys()) == set(candidates), (
        f"Expected keys {candidates}, got {list(input_size_results.keys())}."
    )
    assert best_input_size in candidates, (
        f"best_input_size must be one of {candidates}, got {best_input_size}."
    )
    assert input_size_results[best_input_size] == min(input_size_results.values()), (
        f"best_input_size={best_input_size} (MAE={input_size_results[best_input_size]:.2f}) "
        f"is not the minimum. Check your min() logic."
    )
    assert best_input_size >= 14, (
        f"best_input_size={best_input_size} is less than 2x the horizon (h=7=14). "
        "Ensure input_size is at least 2x the forecast horizon."
    )

    print(f"  Results:")
    for size in sorted(input_size_results.keys()):
        marker = " <-- best" if size == best_input_size else ""
        print(f"    input_size={size:4d}  MAE = {input_size_results[size]:.2f}{marker}")
    print(f"  Best input_size: {best_input_size}")
    rule_of_thumb = 4 * 7  # 4x horizon
    if best_input_size == rule_of_thumb:
        print(f"  Best matches the 4x rule of thumb ({rule_of_thumb}).")
    elif best_input_size > rule_of_thumb:
        print(f"  Best ({best_input_size}) > 4x rule of thumb ({rule_of_thumb}). Strong annual patterns in the data.")
    else:
        print(f"  Best ({best_input_size}) < 4x rule of thumb ({rule_of_thumb}). Weekly pattern dominates.")
    print("  PASSED")
    print()


# ===========================================================================
# Main: run all checks
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Module 01 — Point Forecasting Self-Check Exercises")
    print("=" * 60)
    print()
    print("Loading bakery data to verify connectivity...")
    df = load_bakery_data()
    print(f"Data loaded: {len(df)} rows, {df['unique_id'].nunique()} products")
    print()

    # Run checks in order
    # Each check calls the corresponding exercise function.
    # Implement the exercise functions above, then re-run this file.

    try:
        check_exercise_1()
    except AssertionError as e:
        print(f"  FAILED: {e}\n")
    except Exception as e:
        print(f"  ERROR in exercise 1: {e}\n")

    try:
        check_exercise_2()
    except AssertionError as e:
        print(f"  FAILED: {e}\n")
    except Exception as e:
        print(f"  ERROR in exercise 2: {e}\n")

    try:
        check_exercise_3()
    except AssertionError as e:
        print(f"  FAILED: {e}\n")
    except Exception as e:
        print(f"  ERROR in exercise 3: {e}\n")

    try:
        check_exercise_4()
    except AssertionError as e:
        print(f"  FAILED: {e}\n")
    except Exception as e:
        print(f"  ERROR in exercise 4: {e}\n")

    print("=" * 60)
    print("Done. Fix any FAILED exercises and re-run.")
    print("=" * 60)
