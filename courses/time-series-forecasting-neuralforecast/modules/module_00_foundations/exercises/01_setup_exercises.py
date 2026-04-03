"""
Module 0 Setup Exercises
========================
Self-check exercises to verify your environment is correctly configured for
the Modern Time Series Forecasting with NeuralForecast course.

These are not graded. Run each section and read the output. If an assertion
fails, the message tells you exactly what to fix.

Usage:
    python 01_setup_exercises.py

Expected output when everything passes:
    [PASS] Exercise 1: neuralforecast imports correctly
    [PASS] Exercise 2: datasetsforecast imports correctly
    [PASS] Exercise 3: utilsforecast imports correctly
    [PASS] Exercise 4: nixtla data format validation
    [PASS] Exercise 5: load French Bakery dataset
    [PASS] Exercise 6: train/test split integrity
    [PASS] Exercise 7: instantiate NeuralForecast
    [PASS] Exercise 8: fit and predict workflow
    All exercises passed. Your environment is ready for the course.
"""

import sys

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check(label: str, condition: bool, fix: str = "") -> None:
    """Print a pass/fail result for a single check."""
    if condition:
        print(f"  [OK] {label}")
    else:
        hint = f"\n       How to fix: {fix}" if fix else ""
        print(f"  [FAIL] {label}{hint}")
        raise AssertionError(f"Check failed: {label}")


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Exercise 1: neuralforecast imports
# ---------------------------------------------------------------------------

section("Exercise 1: neuralforecast imports correctly")

try:
    import neuralforecast  # noqa: F401
    check(
        "import neuralforecast",
        True,
    )
    check(
        "neuralforecast has __version__",
        hasattr(neuralforecast, "__version__"),
        fix="Reinstall: pip install neuralforecast",
    )
    # Require version 1.x or 2.x (not 0.x legacy)
    major = int(neuralforecast.__version__.split(".")[0])
    check(
        f"neuralforecast version is 1.x or 2.x (found {neuralforecast.__version__})",
        major >= 1,
        fix="Upgrade: pip install --upgrade neuralforecast",
    )
    print(f"  neuralforecast {neuralforecast.__version__} — OK")
    print("[PASS] Exercise 1: neuralforecast imports correctly")
except ImportError as exc:
    print(f"[FAIL] Exercise 1: {exc}")
    print("       Fix: pip install neuralforecast")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Exercise 2: datasetsforecast imports
# ---------------------------------------------------------------------------

section("Exercise 2: datasetsforecast imports correctly")

try:
    import datasetsforecast  # noqa: F401
    from datasetsforecast.m4 import M4  # noqa: F401
    check("import datasetsforecast", True)
    check("from datasetsforecast.m4 import M4", True)
    print(f"  datasetsforecast {datasetsforecast.__version__} — OK")
    print("[PASS] Exercise 2: datasetsforecast imports correctly")
except ImportError as exc:
    print(f"[FAIL] Exercise 2: {exc}")
    print("       Fix: pip install datasetsforecast")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Exercise 3: utilsforecast imports
# ---------------------------------------------------------------------------

section("Exercise 3: utilsforecast imports correctly")

try:
    import utilsforecast  # noqa: F401
    from utilsforecast.losses import mqloss  # noqa: F401
    from utilsforecast.evaluation import evaluate  # noqa: F401
    check("import utilsforecast", True)
    check("from utilsforecast.losses import mqloss", True)
    check("from utilsforecast.evaluation import evaluate", True)
    print(f"  utilsforecast {utilsforecast.__version__} — OK")
    print("[PASS] Exercise 3: utilsforecast imports correctly")
except ImportError as exc:
    print(f"[FAIL] Exercise 3: {exc}")
    print("       Fix: pip install utilsforecast")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Exercise 4: nixtla data format validation
# ---------------------------------------------------------------------------

section("Exercise 4: nixtla data format validation")

import pandas as pd
import numpy as np

# Build a minimal valid nixtla DataFrame
valid_df = pd.DataFrame({
    "unique_id": ["series_A"] * 10 + ["series_B"] * 10,
    "ds": pd.date_range("2024-01-01", periods=10).tolist() * 2,
    "y": np.random.rand(20) * 100,
})

# Build an intentionally broken DataFrame (missing ds dtype)
broken_df = pd.DataFrame({
    "unique_id": ["series_A"] * 5,
    "ds": ["2024-01-01"] * 5,  # string, not datetime
    "y": [1.0, 2.0, 3.0, 4.0, 5.0],
})

check(
    "valid_df has required columns (unique_id, ds, y)",
    {"unique_id", "ds", "y"}.issubset(valid_df.columns),
)
check(
    "valid_df.ds is datetime dtype",
    pd.api.types.is_datetime64_any_dtype(valid_df["ds"]),
)
check(
    "valid_df.y is numeric",
    pd.api.types.is_numeric_dtype(valid_df["y"]),
)
check(
    "valid_df has no duplicate (unique_id, ds) pairs",
    not valid_df.duplicated(subset=["unique_id", "ds"]).any(),
)
check(
    "valid_df has no missing y values",
    valid_df["y"].notna().all(),
)

# Verify that broken_df.ds is NOT datetime before fixing it
check(
    "broken_df.ds is string (not datetime) before conversion",
    not pd.api.types.is_datetime64_any_dtype(broken_df["ds"]),
)

# Fix the broken DataFrame
broken_df["ds"] = pd.to_datetime(broken_df["ds"])
check(
    "broken_df.ds is datetime after pd.to_datetime()",
    pd.api.types.is_datetime64_any_dtype(broken_df["ds"]),
)

print("[PASS] Exercise 4: nixtla data format validation")


# ---------------------------------------------------------------------------
# Exercise 5: load French Bakery dataset
# ---------------------------------------------------------------------------

section("Exercise 5: load French Bakery dataset")

url = (
    "https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/"
    "main/datasets/french_bakery_daily.csv"
)

try:
    bakery = pd.read_csv(url, parse_dates=["ds"])

    check(
        "French Bakery loaded successfully",
        len(bakery) > 0,
        fix="Check internet connection or URL",
    )
    check(
        "French Bakery has required columns",
        {"unique_id", "ds", "y"}.issubset(bakery.columns),
    )
    check(
        "French Bakery ds is datetime",
        pd.api.types.is_datetime64_any_dtype(bakery["ds"]),
    )
    check(
        "French Bakery has at least 3 series",
        bakery["unique_id"].nunique() >= 3,
    )
    check(
        "French Bakery has no missing y values",
        bakery["y"].notna().all(),
    )
    check(
        "French Bakery has more than 100 rows",
        len(bakery) > 100,
    )
    print(f"  Loaded {len(bakery):,} rows, {bakery['unique_id'].nunique()} series")
    print("[PASS] Exercise 5: load French Bakery dataset")
except Exception as exc:
    print(f"[FAIL] Exercise 5: {exc}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Exercise 6: train/test split integrity
# ---------------------------------------------------------------------------

section("Exercise 6: train/test split integrity")

HORIZON = 7

train = (
    bakery.groupby("unique_id", group_keys=False)
    .apply(lambda x: x.sort_values("ds").iloc[:-HORIZON])
    .reset_index(drop=True)
)

test = (
    bakery.groupby("unique_id", group_keys=False)
    .apply(lambda x: x.sort_values("ds").iloc[-HORIZON:])
    .reset_index(drop=True)
)

# Test set must have exactly HORIZON rows per series
test_sizes = test.groupby("unique_id").size()
check(
    f"Test set has exactly {HORIZON} rows per series",
    (test_sizes == HORIZON).all(),
    fix=f"Check the iloc slice: .iloc[-{HORIZON}:]",
)

# No date leakage: max train date must be strictly before min test date for each series
for series_id in bakery["unique_id"].unique():
    train_max = train[train["unique_id"] == series_id]["ds"].max()
    test_min = test[test["unique_id"] == series_id]["ds"].min()
    check(
        f"No data leakage in series '{series_id}' (train max < test min)",
        train_max < test_min,
        fix="The split is not per-series — use groupby + apply",
    )

# Train and test together must cover the full dataset
total_rows = len(bakery)
split_rows = len(train) + len(test)
check(
    f"Train + test = full dataset ({total_rows} rows)",
    split_rows == total_rows,
)

print(f"  Train: {len(train):,} rows | Test: {len(test):,} rows")
print("[PASS] Exercise 6: train/test split integrity")


# ---------------------------------------------------------------------------
# Exercise 7: instantiate NeuralForecast
# ---------------------------------------------------------------------------

section("Exercise 7: instantiate NeuralForecast")

try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS, XLinear
    from neuralforecast.losses.pytorch import MAE, MQLoss

    # Point forecast model
    point_model = NHITS(h=HORIZON, input_size=2 * HORIZON, loss=MAE(), max_steps=1)

    check(
        "NHITS instantiated with MAE loss",
        hasattr(point_model, "fit"),
        fix="Check neuralforecast version — NHITS should have a fit method",
    )

    # Probabilistic forecast model
    prob_model = NHITS(
        h=HORIZON,
        input_size=2 * HORIZON,
        loss=MQLoss(quantiles=[0.1, 0.5, 0.9]),
        max_steps=1,
    )
    check(
        "NHITS instantiated with MQLoss",
        hasattr(prob_model, "fit"),
    )

    # Linear baseline
    linear_model = XLinear(h=HORIZON, input_size=2 * HORIZON)
    check(
        "XLinear instantiated",
        hasattr(linear_model, "fit"),
    )

    # NeuralForecast wrapper
    nf = NeuralForecast(models=[linear_model, prob_model], freq="D")
    check(
        "NeuralForecast wrapper instantiated with 2 models",
        len(nf.models) == 2,
    )

    print("[PASS] Exercise 7: instantiate NeuralForecast")
except Exception as exc:
    print(f"[FAIL] Exercise 7: {exc}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Exercise 8: fit and predict workflow (fast smoke test)
# ---------------------------------------------------------------------------

section("Exercise 8: fit and predict workflow")

print("  Training for 5 steps on a small subset — this takes ~10 seconds...")

try:
    # Use only 2 series and 60 rows for speed
    small_train = (
        train[train["unique_id"].isin(train["unique_id"].unique()[:2])]
        .groupby("unique_id", group_keys=False)
        .apply(lambda x: x.tail(30))
        .reset_index(drop=True)
    )

    nf_quick = NeuralForecast(
        models=[
            NHITS(
                h=HORIZON,
                input_size=14,
                loss=MQLoss(quantiles=[0.1, 0.5, 0.9]),
                max_steps=5,   # 5 steps only — just verifying the API works
            )
        ],
        freq="D",
    )

    nf_quick.fit(df=small_train)
    check("nf.fit() completed without error", True)

    forecasts = nf_quick.predict()
    check("nf.predict() returned a DataFrame", isinstance(forecasts, pd.DataFrame))

    # Output must have unique_id, ds, and at least one forecast column
    check(
        "forecasts has unique_id column",
        "unique_id" in forecasts.columns,
    )
    check(
        "forecasts has ds column",
        "ds" in forecasts.columns,
    )

    forecast_cols = [c for c in forecasts.columns if c not in ("unique_id", "ds")]
    check(
        f"forecasts has at least 1 model output column (found: {forecast_cols})",
        len(forecast_cols) >= 1,
    )

    # Forecast horizon matches h
    n_rows_per_series = forecasts.groupby("unique_id").size()
    check(
        f"Each series has exactly {HORIZON} forecast rows",
        (n_rows_per_series == HORIZON).all(),
    )

    print(f"  Forecast shape: {forecasts.shape}")
    print(f"  Forecast columns: {forecasts.columns.tolist()}")
    print("[PASS] Exercise 8: fit and predict workflow")

except Exception as exc:
    print(f"[FAIL] Exercise 8: {exc}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("  All exercises passed.")
print("  Your environment is ready for the course.")
print("=" * 60)
print()
print("Next steps:")
print("  1. Open notebooks/01_quickstart_neuralforecast.ipynb")
print("  2. Read guides/01_forecasting_landscape.md")
print("  3. Proceed to Module 01: Point Forecasting")
