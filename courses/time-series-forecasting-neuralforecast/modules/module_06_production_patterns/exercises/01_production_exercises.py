"""
Module 6 Production Patterns — Self-Check Exercises

Goal: Build a ForecastPipeline class from scratch and verify that each stage
produces the correct output shape and semantics.

Run this file directly:
    python exercises/01_production_exercises.py

All checks print PASS or FAIL with a description of what went wrong.
No internet connection is required after the first run (data is cached).
"""

import warnings
warnings.filterwarnings('ignore')

import os
import tempfile
import pandas as pd
import numpy as np

# ── Helpers ───────────────────────────────────────────────────────────────────

_PASS = "\033[92mPASS\033[0m"
_FAIL = "\033[91mFAIL\033[0m"


def check(condition: bool, name: str, detail: str = "") -> bool:
    """Print a labeled pass/fail result. Returns True on pass."""
    if condition:
        print(f"  {_PASS}  {name}")
    else:
        msg = f"  {_FAIL}  {name}"
        if detail:
            msg += f"\n         Hint: {detail}"
        print(msg)
    return condition


def section(title: str) -> None:
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


# ── Data Fixture ──────────────────────────────────────────────────────────────

def load_bakery_data() -> pd.DataFrame:
    """
    Load the French Bakery dataset and convert to nixtla long format.

    Returns columns: unique_id | ds | y
    """
    cache_path = os.path.join(tempfile.gettempdir(), "bakery_cache.csv")

    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, parse_dates=["ds"])

    url = (
        "https://raw.githubusercontent.com/nixtla/transfer-learning-time-series"
        "/main/datasets/french_bakery/bakery.csv"
    )
    raw = pd.read_csv(url, parse_dates=["date"])
    df = (
        raw.rename(columns={"date": "ds", "Quantity": "y", "article": "unique_id"})
        [["unique_id", "ds", "y"]]
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )
    df.to_csv(cache_path, index=False)
    return df


# ── Exercise 1: Build ForecastPipeline ────────────────────────────────────────

section("Exercise 1: ForecastPipeline Class")

print("""
  Build a ForecastPipeline class with the following interface:

    pipeline = ForecastPipeline(horizon=7, freq='D')
    pipeline.ingest(df, id_col='unique_id', ds_col='ds', y_col='y')
    pipeline.train()
    forecast = pipeline.predict()
    decision = pipeline.service_level_order(forecast, service_level=0.8)

  Requirements:
    - .ingest() stores the data and returns self (for chaining)
    - .train() fits an NHITS or equivalent model and returns self
    - .predict() returns a pd.DataFrame with columns: unique_id | ds | <quantile cols>
    - .service_level_order() returns a dict with keys:
        service_level, total_order, peak_day, daily_breakdown

  Write your implementation below, then run the checks.
""")

# ── YOUR IMPLEMENTATION ───────────────────────────────────────────────────────
# Replace the NotImplemented stubs below with your implementation.
# The class must pass all checks in Exercise 2 to be considered complete.

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MQLoss


class ForecastPipeline:
    """
    Your production forecasting pipeline.

    Stages: ingest → train → predict → service_level_order
    """

    def __init__(
        self,
        horizon: int = 7,
        freq: str = "D",
        quantiles: list | None = None,
        max_steps: int = 200,
        random_seed: int = 42,
    ):
        self.horizon = horizon
        self.freq = freq
        self.quantiles = quantiles or [0.1, 0.5, 0.8, 0.9]
        self.max_steps = max_steps
        self.random_seed = random_seed
        self._nf = None
        self._train_df = None

    def ingest(
        self,
        df: pd.DataFrame,
        id_col: str = "unique_id",
        ds_col: str = "ds",
        y_col: str = "y",
    ) -> "ForecastPipeline":
        """
        Convert df to nixtla format (unique_id | ds | y) and store.

        Raise ValueError if:
          - any of id_col, ds_col, y_col are missing from df
          - there are duplicate (unique_id, ds) pairs
          - any series has fewer than 2 * horizon observations
        """
        raise NotImplementedError("Implement .ingest()")

    def train(self) -> "ForecastPipeline":
        """
        Fit an NHITS model with MQLoss on the stored data.

        Raise RuntimeError if .ingest() has not been called.
        """
        raise NotImplementedError("Implement .train()")

    def predict(self) -> pd.DataFrame:
        """
        Return quantile forecasts for the next `horizon` time steps.

        Raise RuntimeError if .train() has not been called.
        """
        raise NotImplementedError("Implement .predict()")

    def service_level_order(
        self,
        forecast: pd.DataFrame,
        service_level: float = 0.8,
        unique_id: str | None = None,
    ) -> dict:
        """
        Compute the order quantity at the given service level.

        Parameters
        ----------
        forecast : pd.DataFrame
            Output of .predict().
        service_level : float
            Target quantile, e.g. 0.8.
        unique_id : str | None
            Filter to a specific series. If None, sum across all.

        Returns
        -------
        dict with keys: service_level, total_order, peak_day, daily_breakdown
        """
        raise NotImplementedError("Implement .service_level_order()")


# ── Exercise 2: Shape and Semantic Checks ─────────────────────────────────────

section("Exercise 2: Shape and Semantic Checks")

print("  Loading data...")
try:
    df = load_bakery_data()
    baguette = df[df["unique_id"] == "BAGUETTE"].copy()
    print(f"  Data loaded: {len(baguette)} rows for BAGUETTE.")
except Exception as e:
    print(f"  Could not load data: {e}. Skipping checks.")
    raise SystemExit(1)

H = 7
QUANTILES = [0.1, 0.5, 0.8, 0.9]

# ── 2a. Ingest checks ─────────────────────────────────────────────────────────
print("\n  2a. Ingest checks")

try:
    pipeline = ForecastPipeline(horizon=H, quantiles=QUANTILES, max_steps=150)
    result = pipeline.ingest(baguette)

    check(
        result is pipeline,
        ".ingest() returns self (enables method chaining)",
        "return 'self' at the end of .ingest()",
    )
    check(
        pipeline._train_df is not None,
        ".ingest() stores data in _train_df",
        "assign the processed DataFrame to self._train_df",
    )
    check(
        set(pipeline._train_df.columns) >= {"unique_id", "ds", "y"},
        "_train_df has required columns: unique_id, ds, y",
        "rename columns to unique_id, ds, y in .ingest()",
    )
    check(
        pd.api.types.is_datetime64_any_dtype(pipeline._train_df["ds"]),
        "_train_df['ds'] is datetime dtype",
        "use pd.to_datetime() on the ds column",
    )
except NotImplementedError:
    print("  [skipped — .ingest() not yet implemented]")
except Exception as e:
    print(f"  [error in .ingest() checks]: {e}")

# ── 2b. Validation checks ─────────────────────────────────────────────────────
print("\n  2b. Ingest validation checks")

try:
    # Test: missing column raises ValueError
    try:
        ForecastPipeline(horizon=H).ingest(
            baguette.rename(columns={"y": "demand"}),  # wrong column name
            id_col="unique_id", ds_col="ds", y_col="y",
        )
        check(False, ".ingest() raises ValueError on missing column",
              "check that all of id_col, ds_col, y_col exist in df")
    except ValueError:
        check(True, ".ingest() raises ValueError on missing column")
    except NotImplementedError:
        print("  [skipped — .ingest() not yet implemented]")

    # Test: duplicate timestamps raise ValueError
    dup_df = pd.concat([baguette.head(5), baguette.head(5)], ignore_index=True)
    try:
        ForecastPipeline(horizon=H).ingest(dup_df)
        check(False, ".ingest() raises ValueError on duplicate (unique_id, ds) pairs",
              "use .duplicated(['unique_id', 'ds']).sum() to count duplicates")
    except ValueError:
        check(True, ".ingest() raises ValueError on duplicate timestamps")
    except NotImplementedError:
        print("  [skipped — .ingest() not yet implemented]")

    # Test: series too short raises ValueError
    short_df = baguette.head(H - 1).copy()  # shorter than minimum
    try:
        ForecastPipeline(horizon=H).ingest(short_df)
        check(False, ".ingest() raises ValueError on series shorter than 2*horizon",
              "check min series length >= 2 * self.horizon")
    except ValueError:
        check(True, ".ingest() raises ValueError on series shorter than 2*horizon")
    except NotImplementedError:
        print("  [skipped — .ingest() not yet implemented]")

except Exception as e:
    print(f"  [error in validation checks]: {e}")

# ── 2c. Train checks ──────────────────────────────────────────────────────────
print("\n  2c. Train checks")

try:
    pipeline2 = ForecastPipeline(horizon=H, quantiles=QUANTILES, max_steps=150)

    # Test: calling train() before ingest() raises RuntimeError
    try:
        pipeline2.train()
        check(False, ".train() raises RuntimeError before .ingest()",
              "check that self._train_df is not None at the start of .train()")
    except RuntimeError:
        check(True, ".train() raises RuntimeError before .ingest()")
    except NotImplementedError:
        print("  [skipped — .train() not yet implemented]")

    # Now properly ingest then train
    pipeline2.ingest(baguette)
    print("  Training model (this may take 30–60 seconds)...")

    result = pipeline2.train()

    check(
        result is pipeline2,
        ".train() returns self (enables method chaining)",
        "return 'self' at the end of .train()",
    )
    check(
        pipeline2._nf is not None,
        ".train() stores the NeuralForecast object in _nf",
        "assign the fitted NeuralForecast to self._nf",
    )

except NotImplementedError:
    print("  [skipped — .train() not yet implemented]")
except Exception as e:
    print(f"  [error in .train() checks]: {e}")

# ── 2d. Predict checks ────────────────────────────────────────────────────────
print("\n  2d. Predict checks")

try:
    # Test: calling predict() before train() raises RuntimeError
    pipeline3 = ForecastPipeline(horizon=H, quantiles=QUANTILES, max_steps=150)
    pipeline3.ingest(baguette)
    try:
        pipeline3.predict()
        check(False, ".predict() raises RuntimeError before .train()",
              "check that self._nf is not None at the start of .predict()")
    except RuntimeError:
        check(True, ".predict() raises RuntimeError before .train()")
    except NotImplementedError:
        print("  [skipped — .predict() not yet implemented]")

    # Test: predict() returns correct shape (only if pipeline2 was trained)
    if pipeline2._nf is not None:
        forecast = pipeline2.predict()

        check(
            isinstance(forecast, pd.DataFrame),
            ".predict() returns a pd.DataFrame",
            "return the output of self._nf.predict()",
        )
        check(
            "unique_id" in forecast.columns and "ds" in forecast.columns,
            ".predict() result has 'unique_id' and 'ds' columns",
            "NeuralForecast.predict() returns these automatically",
        )
        q_cols = [c for c in forecast.columns if "q-" in c]
        check(
            len(q_cols) == len(QUANTILES),
            f".predict() result has {len(QUANTILES)} quantile columns",
            f"expected one column per quantile in {QUANTILES}",
        )
        check(
            len(forecast) == H,
            f".predict() result has {H} rows (one per forecast step)",
            f"expected {H} rows for horizon={H}, got {len(forecast)}",
        )
        check(
            (forecast[q_cols] >= 0).all().all(),
            "All quantile forecasts are non-negative",
            "Bakery demand cannot be negative; check that the model is trained correctly",
        )

        # Monotonicity: q10 <= q50 <= q80 <= q90
        q_sorted = sorted(q_cols, key=lambda c: float(c.split("q-")[1]))
        if len(q_sorted) >= 2:
            monotone = all(
                (forecast[q_sorted[i]] <= forecast[q_sorted[i + 1]] + 0.01).all()
                for i in range(len(q_sorted) - 1)
            )
            check(
                monotone,
                "Quantile forecasts are monotonically ordered (q10 ≤ q50 ≤ q80 ≤ q90)",
                "Check that NHITS is trained correctly — quantile crossing is uncommon but possible",
            )

except NotImplementedError:
    print("  [skipped — .predict() not yet implemented]")
except Exception as e:
    print(f"  [error in .predict() checks]: {e}")

# ── 2e. service_level_order checks ───────────────────────────────────────────
print("\n  2e. service_level_order checks")

try:
    if pipeline2._nf is not None:
        forecast = pipeline2.predict()
        decision = pipeline2.service_level_order(forecast, service_level=0.8)

        check(
            isinstance(decision, dict),
            ".service_level_order() returns a dict",
            "return a dict, not a DataFrame or list",
        )
        required_keys = {"service_level", "total_order", "peak_day", "daily_breakdown"}
        check(
            required_keys.issubset(decision.keys()),
            f".service_level_order() dict has required keys: {required_keys}",
            f"missing keys: {required_keys - set(decision.keys())}",
        )
        check(
            decision["service_level"] == 0.8,
            "decision['service_level'] equals the requested 0.8",
            "store the service_level parameter in the output dict",
        )
        check(
            isinstance(decision["total_order"], int),
            "decision['total_order'] is an int",
            "use int(np.ceil(...)) to ensure integer output",
        )
        check(
            isinstance(decision["daily_breakdown"], list),
            "decision['daily_breakdown'] is a list",
            "convert to a Python list, not a numpy array",
        )
        check(
            len(decision["daily_breakdown"]) == H,
            f"decision['daily_breakdown'] has {H} elements (one per forecast day)",
            f"expected {H} elements, got {len(decision['daily_breakdown'])}",
        )
        check(
            decision["total_order"] == sum(decision["daily_breakdown"]),
            "total_order equals sum(daily_breakdown)",
            "total_order should be the sum of all daily orders",
        )
        check(
            decision["peak_day"] == max(decision["daily_breakdown"]),
            "peak_day equals max(daily_breakdown)",
            "peak_day should be the maximum single-day order",
        )

        # Test: invalid service level raises ValueError
        try:
            pipeline2.service_level_order(forecast, service_level=0.42)
            check(False, ".service_level_order() raises ValueError for unavailable quantile",
                  "check that the requested quantile exists in the forecast columns")
        except ValueError:
            check(True, ".service_level_order() raises ValueError for unavailable quantile")

except NotImplementedError:
    print("  [skipped — .service_level_order() not yet implemented]")
except Exception as e:
    print(f"  [error in .service_level_order() checks]: {e}")

# ── Exercise 3: Method Chaining ───────────────────────────────────────────────

section("Exercise 3: Method Chaining")

print("""
  Verify that the pipeline supports method chaining:

    pipeline = (
        ForecastPipeline(horizon=7)
        .ingest(df)
        .train()
    )
    forecast = pipeline.predict()
""")

try:
    pipeline_chain = (
        ForecastPipeline(horizon=H, quantiles=QUANTILES, max_steps=150)
        .ingest(baguette)
        .train()
    )
    forecast_chain = pipeline_chain.predict()

    check(
        isinstance(forecast_chain, pd.DataFrame),
        "Method chaining works: .ingest().train().predict() succeeds",
        "Each of .ingest() and .train() must return self",
    )
    check(
        len(forecast_chain) == H,
        f"Chained pipeline produces {H}-row forecast",
        f"expected {H} rows, got {len(forecast_chain)}",
    )
except NotImplementedError:
    print("  [skipped — pipeline not yet implemented]")
except Exception as e:
    print(f"  [error in method chaining check]: {e}")

# ── Final Summary ─────────────────────────────────────────────────────────────

section("Done")
print("""
  Fix any FAIL items above by implementing the stub methods in ForecastPipeline.

  When all checks pass:
    - Your pipeline correctly validates input data
    - Training produces a fitted NeuralForecast model
    - Predictions have the right shape and quantile ordering
    - The order decision includes total_order, peak_day, and daily_breakdown

  Next: open notebooks/01_end_to_end_pipeline.ipynb to see the complete pipeline
  applied to the full French Bakery dataset with visualizations.
""")
