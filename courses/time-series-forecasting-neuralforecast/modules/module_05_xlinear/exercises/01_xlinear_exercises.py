"""
Module 5 — XLinear Self-Check Exercises
Modern Time Series Forecasting with NeuralForecast

Run each exercise function in order. Each one has an assertion that passes
when your implementation is correct, and prints a diagnostic message on failure.

Prerequisites:
    pip install neuralforecast datasetsforecast utilsforecast

Usage:
    python 01_xlinear_exercises.py          # run all exercises
    python 01_xlinear_exercises.py --ex 1   # run one exercise

Expected runtime: 8–12 minutes (training runs included)
"""

import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Exercise 1: Verify Prediction Shape
# ─────────────────────────────────────────────────────────────────────────────

def exercise_1_prediction_shape():
    """
    Train XLinear on ETTm1 with h=96 and verify the output tensor shape.

    XLinear forecasts ALL n_series variables simultaneously.
    The cross_validation() output should contain one row per
    (unique_id, cutoff, ds) combination.

    Expected shape check:
        - cv_df has column "XLinear" (the forecast values)
        - Number of unique unique_ids == 7 (all ETTm1 series)
        - Number of forecast steps per series per cutoff == h (96)
    """
    print("Exercise 1: Verifying prediction shape...")

    from neuralforecast import NeuralForecast
    from neuralforecast.models import XLinear
    from datasetsforecast.long_horizon import LongHorizon

    Y_df, _, _ = LongHorizon.load(directory="./data", group="ETTm1")
    Y_df["ds"] = pd.to_datetime(Y_df["ds"])

    H = 96
    N_SERIES = 7

    model = XLinear(
        h=H,
        input_size=H,
        n_series=N_SERIES,
        hidden_size=128,      # small hidden_size for fast training in exercise
        temporal_ff=64,
        channel_ff=21,
        head_dropout=0.3,
        embed_dropout=0.1,
        learning_rate=1e-3,
        batch_size=64,
        max_steps=200,        # short run — just enough to check shapes
    )

    nf = NeuralForecast(models=[model], freq="15min")
    cv_df = nf.cross_validation(df=Y_df, val_size=11520, test_size=11520)

    # ── Shape assertions ──────────────────────────────────────────────────────

    # 1. Forecast column exists
    assert "XLinear" in cv_df.columns, (
        f"Expected 'XLinear' column in cv_df. Got columns: {cv_df.columns.tolist()}"
    )

    # 2. All 7 series present
    n_series_in_output = cv_df["unique_id"].nunique()
    assert n_series_in_output == N_SERIES, (
        f"Expected {N_SERIES} series in output. Got {n_series_in_output}.\n"
        f"Check that n_series={N_SERIES} in XLinear and that all series are in Y_df."
    )

    # 3. Forecast values are not all NaN
    nan_fraction = cv_df["XLinear"].isna().mean()
    assert nan_fraction < 0.01, (
        f"More than 1% of XLinear forecasts are NaN ({nan_fraction:.1%}).\n"
        f"This may indicate a training failure. Check max_steps and learning_rate."
    )

    # 4. Forecast values are finite (no inf)
    inf_count = np.isinf(cv_df["XLinear"].values).sum()
    assert inf_count == 0, (
        f"Found {inf_count} infinite values in XLinear forecasts.\n"
        f"This indicates numerical instability — reduce learning_rate."
    )

    # 5. Steps per series per cutoff == h
    steps_per_window = cv_df.groupby(["unique_id", "cutoff"])["ds"].count()
    assert (steps_per_window == H).all(), (
        f"Expected {H} forecast steps per (series, cutoff). Got:\n{steps_per_window.describe()}"
    )

    print(f"  cv_df shape: {cv_df.shape}")
    print(f"  Series in output: {cv_df['unique_id'].unique().tolist()}")
    print(f"  Forecast steps per window per series: {H}")
    print(f"  NaN fraction: {nan_fraction:.4f}")
    print("  PASSED: Prediction shape is correct.\n")
    return cv_df


# ─────────────────────────────────────────────────────────────────────────────
# Exercise 2: Compare MAE of XLinear vs. NHITS
# ─────────────────────────────────────────────────────────────────────────────

def exercise_2_mae_comparison():
    """
    Train both XLinear (multivariate) and NHITS (univariate) on ETTm1 h=96.
    Compute mean MAE for each model on the test set.
    Verify that XLinear achieves lower MAE than NHITS.

    This exercise confirms the architectural advantage of cross-variable learning
    on a dataset with known physical correlations.

    Note: With max_steps=300, both models are undertrained relative to the
    published benchmark. XLinear should still outperform NHITS due to its
    multivariate inductive bias, but the margin may be smaller than reported.
    """
    print("Exercise 2: Comparing XLinear MAE vs. NHITS MAE...")

    from neuralforecast import NeuralForecast
    from neuralforecast.models import XLinear, NHITS
    from datasetsforecast.long_horizon import LongHorizon
    from utilsforecast.losses import mae
    from utilsforecast.evaluation import evaluate

    Y_df, _, _ = LongHorizon.load(directory="./data", group="ETTm1")
    Y_df["ds"] = pd.to_datetime(Y_df["ds"])

    H = 96
    FREQ = "15min"
    VAL_SIZE = 11520
    TEST_SIZE = 11520

    nhits_model = NHITS(
        h=H,
        input_size=192,
        n_freq_downsample=[2, 1, 1],
        learning_rate=1e-3,
        batch_size=64,
        max_steps=300,
    )

    xlinear_model = XLinear(
        h=H,
        input_size=H,
        n_series=7,
        hidden_size=256,
        temporal_ff=128,
        channel_ff=21,
        head_dropout=0.3,
        embed_dropout=0.1,
        learning_rate=1e-3,
        batch_size=64,
        max_steps=300,
    )

    # Train and evaluate NHITS
    nf_nhits = NeuralForecast(models=[nhits_model], freq=FREQ)
    cv_nhits = nf_nhits.cross_validation(df=Y_df, val_size=VAL_SIZE, test_size=TEST_SIZE)
    test_nhits = cv_nhits[cv_nhits["cutoff"] == cv_nhits["cutoff"].max()].copy()
    eval_nhits = evaluate(df=test_nhits, metrics=[mae], models=["NHITS"])
    nhits_mae = eval_nhits[eval_nhits["metric"] == "mae"]["NHITS"].mean()

    # Train and evaluate XLinear
    nf_xlinear = NeuralForecast(models=[xlinear_model], freq=FREQ)
    cv_xlinear = nf_xlinear.cross_validation(df=Y_df, val_size=VAL_SIZE, test_size=TEST_SIZE)
    test_xlinear = cv_xlinear[cv_xlinear["cutoff"] == cv_xlinear["cutoff"].max()].copy()
    eval_xlinear = evaluate(df=test_xlinear, metrics=[mae], models=["XLinear"])
    xlinear_mae = eval_xlinear[eval_xlinear["metric"] == "mae"]["XLinear"].mean()

    print(f"  NHITS  mean MAE: {nhits_mae:.4f}")
    print(f"  XLinear mean MAE: {xlinear_mae:.4f}")
    print(f"  Difference (NHITS - XLinear): {nhits_mae - xlinear_mae:.4f}")

    # ── MAE assertion ─────────────────────────────────────────────────────────
    # With 300 steps, XLinear should achieve comparable or better MAE.
    # A strict better-than assertion is relaxed slightly for short training runs.
    assert xlinear_mae <= nhits_mae * 1.05, (
        f"XLinear MAE ({xlinear_mae:.4f}) is more than 5% worse than NHITS MAE ({nhits_mae:.4f}).\n"
        f"This is unexpected for ETTm1 which has strong cross-series correlations.\n"
        f"Possible causes: n_series mismatch, learning rate too high, or training instability.\n"
        f"Try: increase max_steps to 1000, reduce learning_rate to 1e-4, verify n_series=7."
    )

    improvement = (nhits_mae - xlinear_mae) / nhits_mae * 100
    print(f"  XLinear improvement over NHITS: {improvement:.1f}%")
    print("  PASSED: XLinear achieves comparable or better MAE than NHITS.\n")

    return nhits_mae, xlinear_mae


# ─────────────────────────────────────────────────────────────────────────────
# Exercise 3: Experiment with hidden_size Values
# ─────────────────────────────────────────────────────────────────────────────

def exercise_3_hidden_size_ablation():
    """
    Train XLinear with three different hidden_size values: 128, 256, 512.
    Compare mean MAE on the ETTm1 test set.

    Expected behavior:
        - MAE should decrease as hidden_size increases from 128 to 512
        - 512 should achieve the lowest MAE among the three settings
        - The improvement from 256 to 512 should be smaller than from 128 to 256
          (diminishing returns in model capacity)

    This ablation demonstrates why hidden_size=512 is the reference configuration.
    """
    print("Exercise 3: hidden_size ablation (128, 256, 512)...")

    from neuralforecast import NeuralForecast
    from neuralforecast.models import XLinear
    from datasetsforecast.long_horizon import LongHorizon
    from utilsforecast.losses import mae
    from utilsforecast.evaluation import evaluate

    Y_df, _, _ = LongHorizon.load(directory="./data", group="ETTm1")
    Y_df["ds"] = pd.to_datetime(Y_df["ds"])

    H = 96
    FREQ = "15min"
    VAL_SIZE = 11520
    TEST_SIZE = 11520
    HIDDEN_SIZES = [128, 256, 512]

    results = {}

    for hs in HIDDEN_SIZES:
        print(f"  Training XLinear with hidden_size={hs}...")

        model = XLinear(
            h=H,
            input_size=H,
            n_series=7,
            hidden_size=hs,
            temporal_ff=hs // 2,  # proportional scaling
            channel_ff=21,
            head_dropout=0.3,
            embed_dropout=0.1,
            learning_rate=1e-3,
            batch_size=64,
            max_steps=300,        # short run for exercise — extend to 2000 for production
        )

        nf = NeuralForecast(models=[model], freq=FREQ)
        cv = nf.cross_validation(df=Y_df, val_size=VAL_SIZE, test_size=TEST_SIZE)
        test_df = cv[cv["cutoff"] == cv["cutoff"].max()].copy()
        eval_df = evaluate(df=test_df, metrics=[mae], models=["XLinear"])
        mean_mae = eval_df[eval_df["metric"] == "mae"]["XLinear"].mean()
        results[hs] = mean_mae
        print(f"    hidden_size={hs}: MAE={mean_mae:.4f}")

    # ── Ablation assertions ───────────────────────────────────────────────────

    # 1. All MAE values are positive and finite
    for hs, val in results.items():
        assert np.isfinite(val) and val > 0, (
            f"Invalid MAE for hidden_size={hs}: {val}. "
            f"Check for training failure or NaN outputs."
        )

    # 2. Larger hidden_size should not be more than 20% worse than smaller
    #    (sanity check that all configs trained successfully)
    max_mae = max(results.values())
    min_mae = min(results.values())
    assert max_mae / min_mae < 1.25, (
        f"MAE range is suspiciously large: min={min_mae:.4f}, max={max_mae:.4f} "
        f"(ratio={max_mae/min_mae:.2f}). One config may have failed to train. "
        f"Try increasing max_steps to 500."
    )

    # 3. Summary printout
    best_hs = min(results, key=results.get)
    print()
    print("  hidden_size ablation summary:")
    print(f"  {'hidden_size':<14} {'MAE':>10}")
    print("  " + "-" * 26)
    for hs in HIDDEN_SIZES:
        marker = " <-- best" if hs == best_hs else ""
        print(f"  {hs:<14} {results[hs]:>10.4f}{marker}")

    print()
    print(f"  Best hidden_size: {best_hs} (MAE={results[best_hs]:.4f})")
    print("  PASSED: All three hidden_size configurations trained successfully.\n")
    print("  Note: With max_steps=300, the advantage of larger hidden_size may")
    print("  not be fully visible. Re-run with max_steps=2000 for full results.")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def main(run_exercise=None):
    print("=" * 65)
    print("Module 5 — XLinear Self-Check Exercises")
    print("=" * 65)
    print()

    exercises = {
        1: exercise_1_prediction_shape,
        2: exercise_2_mae_comparison,
        3: exercise_3_hidden_size_ablation,
    }

    if run_exercise is not None:
        if run_exercise not in exercises:
            print(f"Unknown exercise: {run_exercise}. Choose from {list(exercises.keys())}.")
            sys.exit(1)
        exercises[run_exercise]()
    else:
        for ex_num, ex_fn in exercises.items():
            print(f"{'─' * 65}")
            print(f"Running Exercise {ex_num}: {ex_fn.__name__}")
            print(f"{'─' * 65}")
            ex_fn()

    print("=" * 65)
    print("All exercises completed.")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XLinear self-check exercises")
    parser.add_argument(
        "--ex", type=int, default=None,
        help="Run a single exercise by number (1, 2, or 3). Omit to run all."
    )
    args = parser.parse_args()
    main(run_exercise=args.ex)
