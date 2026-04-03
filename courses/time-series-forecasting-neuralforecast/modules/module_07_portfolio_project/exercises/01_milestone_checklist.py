"""
Milestone Checklist — Module 7 Portfolio Project
=================================================

Self-check script that validates each project milestone programmatically.
Run this after completing each milestone to confirm structural correctness
before moving to the next one.

Usage
-----
    python 01_milestone_checklist.py

All checks print PASS or FAIL with an explanation. A project is structurally
complete when every check prints PASS.

Requirements
------------
The script expects the following variables to be defined in your project
notebook or loaded from saved artifacts. The simplest workflow is to run your
notebook to completion, then run this script in the same environment.

    df          : pd.DataFrame — full dataset in nixtla format
    df_train    : pd.DataFrame — training split
    df_test     : pd.DataFrame — test split
    nf          : NeuralForecast — fitted NeuralForecast instance
    forecasts   : pd.DataFrame — output of nf.predict()
    paths_df    : pd.DataFrame — output of nf.models[0].simulate()
    explanations: dict — output of nf.models[0].explain()
    business_answer : float — numeric answer to your business question
    business_units  : str   — units for the answer, e.g. "units/week"

The script imports these from a sidecar file `project_artifacts.py` which you
create by saving results from your notebook. A template for that file is
printed if it does not exist.
"""

import sys
import importlib
import numpy as np
import pandas as pd

# ── Color helpers ─────────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"
BOLD  = "\033[1m"


def _pass(check_name: str, detail: str = "") -> None:
    suffix = f" — {detail}" if detail else ""
    print(f"  {GREEN}PASS{RESET}  {check_name}{suffix}")


def _fail(check_name: str, reason: str) -> None:
    print(f"  {RED}FAIL{RESET}  {check_name}")
    print(f"        {reason}")


def _section(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}")
    print("-" * 60)


# ── Artifact loader ───────────────────────────────────────────────────────────

ARTIFACT_TEMPLATE = '''\
"""
project_artifacts.py — save your notebook outputs here for milestone checking.

Replace each None with the actual object from your notebook.
Run: python 01_milestone_checklist.py
"""
import pandas as pd
import numpy as np

# -- Milestone 1 artifacts ---------------------------------------------------
df       = None  # full dataset
df_train = None  # training split
df_test  = None  # test split

# -- Milestone 2 artifacts ---------------------------------------------------
# nf        = None  # NeuralForecast instance (cannot be imported directly)
# Provide forecast DataFrame instead:
forecasts = None  # output of nf.predict()

# -- Milestone 3 artifacts ---------------------------------------------------
paths_df = None  # output of nf.models[0].simulate()

business_answer = None  # float: numeric answer (e.g. 148.0)
business_units  = None  # str: units of the answer (e.g. "units/week")

# -- Milestone 4 artifacts ---------------------------------------------------
explanations = None  # dict from nf.models[0].explain()
'''


def _load_artifacts():
    """Load artifacts from project_artifacts.py if it exists."""
    try:
        spec = importlib.util.spec_from_file_location(
            "project_artifacts", "project_artifacts.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except FileNotFoundError:
        print(
            f"{RED}project_artifacts.py not found.{RESET}\n"
            "Create it with the template below and populate each variable from your notebook.\n"
        )
        print(ARTIFACT_TEMPLATE)
        sys.exit(1)
    except Exception as exc:
        print(f"{RED}Error loading project_artifacts.py: {exc}{RESET}")
        sys.exit(1)


# ── Milestone checks ──────────────────────────────────────────────────────────

def check_milestone_1(art) -> int:
    """Data loaded in nixtla format with series, date range, and business question."""
    _section("Milestone 1: Data Selection and EDA")
    failures = 0

    # Check 1: df exists and is a DataFrame
    df = getattr(art, "df", None)
    if df is None or not isinstance(df, pd.DataFrame):
        _fail("Dataset loaded", "art.df must be a non-None pd.DataFrame")
        failures += 1
    else:
        _pass("Dataset loaded", f"{len(df):,} rows")

    if df is not None and isinstance(df, pd.DataFrame):
        # Check 2: required columns present
        required_cols = {"unique_id", "ds", "y"}
        missing = required_cols - set(df.columns)
        if missing:
            _fail("Nixtla format (columns)", f"Missing columns: {missing}")
            failures += 1
        else:
            _pass("Nixtla format (columns)", "unique_id, ds, y present")

        # Check 3: ds is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["ds"]):
            _fail("Nixtla format (ds type)", "Column 'ds' is not datetime — use pd.to_datetime()")
            failures += 1
        else:
            _pass("Nixtla format (ds type)", "datetime64")

        # Check 4: y is numeric
        if not pd.api.types.is_numeric_dtype(df["y"]):
            _fail("Nixtla format (y type)", "Column 'y' must be numeric")
            failures += 1
        else:
            _pass("Nixtla format (y type)", "numeric")

        # Check 5: multiple series (unique_id variety)
        n_series = df["unique_id"].nunique()
        if n_series < 1:
            _fail("Series count", "Dataset has no series")
            failures += 1
        else:
            _pass("Series count", f"{n_series} series")

        # Check 6: minimum length — at least 30 observations per series on average
        mean_len = df.groupby("unique_id").size().mean()
        if mean_len < 30:
            _fail(
                "Series length",
                f"Mean series length is {mean_len:.0f} — need at least 30 observations per series"
            )
            failures += 1
        else:
            _pass("Series length", f"Mean {mean_len:.0f} observations per series")

    # Check 7: train/test split exists
    df_train = getattr(art, "df_train", None)
    df_test  = getattr(art, "df_test", None)
    if df_train is None or df_test is None:
        _fail("Train/test split", "art.df_train and art.df_test must both be defined")
        failures += 1
    else:
        _pass("Train/test split", f"train={len(df_train):,} rows, test={len(df_test):,} rows")

    return failures


def check_milestone_2(art) -> int:
    """Model trains without errors; forecasts in expected format."""
    _section("Milestone 2: Model Training and Evaluation")
    failures = 0

    forecasts = getattr(art, "forecasts", None)

    # Check 1: forecasts exists
    if forecasts is None or not isinstance(forecasts, pd.DataFrame):
        _fail("Forecasts produced", "art.forecasts must be a non-None pd.DataFrame")
        failures += 1
        return failures
    else:
        _pass("Forecasts produced", f"{len(forecasts):,} rows")

    # Check 2: forecasts has unique_id and ds
    for col in ("unique_id", "ds"):
        if col not in forecasts.columns:
            _fail(f"Forecast column '{col}'", f"Missing from forecasts DataFrame")
            failures += 1
        else:
            _pass(f"Forecast column '{col}'", "present")

    # Check 3: at least one model column with quantile output
    quantile_cols = [c for c in forecasts.columns if "-q-" in c or "q0." in c.lower() or "lo-" in c.lower()]
    if not quantile_cols:
        _fail(
            "Quantile output columns",
            "No quantile columns found (expected columns like 'NHITS-q-0.5' or 'NHITS-lo-80'). "
            "Ensure model was trained with MQLoss."
        )
        failures += 1
    else:
        _pass("Quantile output columns", f"{len(quantile_cols)} quantile columns found")

    # Check 4: no NaN in forecasts
    nan_count = forecasts[quantile_cols].isna().sum().sum() if quantile_cols else 0
    if nan_count > 0:
        _fail("No NaN in forecasts", f"{nan_count} NaN values — check model training")
        failures += 1
    else:
        _pass("No NaN in forecasts", "clean")

    return failures


def check_milestone_3(art) -> int:
    """Sample paths generated with correct shape; business answer computed."""
    _section("Milestone 3: Sample Paths and Business Decision")
    failures = 0

    paths_df = getattr(art, "paths_df", None)

    # Check 1: paths_df exists
    if paths_df is None or not isinstance(paths_df, pd.DataFrame):
        _fail("paths_df produced", "art.paths_df must be a non-None pd.DataFrame from .simulate()")
        failures += 1
    else:
        _pass("paths_df produced", f"{paths_df.shape}")

        # Check 2: sample columns present
        sample_cols = [c for c in paths_df.columns if c.startswith("sample_")]
        if len(sample_cols) < 100:
            _fail(
                "Minimum 100 sample paths",
                f"Only {len(sample_cols)} sample_ columns found — use n_paths >= 100 "
                "(200 recommended for stable estimates)"
            )
            failures += 1
        else:
            _pass("Minimum 100 sample paths", f"{len(sample_cols)} paths")

        # Check 3: unique_id and ds columns present in paths_df
        for col in ("unique_id", "ds"):
            if col not in paths_df.columns:
                _fail(f"paths_df column '{col}'", "Missing — paths_df should include unique_id and ds")
                failures += 1
            else:
                _pass(f"paths_df column '{col}'", "present")

        # Check 4: no NaN in path values
        nan_count = paths_df[sample_cols].isna().sum().sum() if sample_cols else 0
        if nan_count > 0:
            _fail("No NaN in paths", f"{nan_count} NaN values in sample path matrix")
            failures += 1
        else:
            _pass("No NaN in paths", "clean")

    # Check 5: business answer is a numeric scalar
    business_answer = getattr(art, "business_answer", None)
    business_units  = getattr(art, "business_units", None)

    if business_answer is None:
        _fail(
            "Business answer computed",
            "art.business_answer must be a float — the numeric result of your business question"
        )
        failures += 1
    elif not np.isscalar(business_answer) or np.isnan(float(business_answer)):
        _fail(
            "Business answer is scalar",
            f"Got {business_answer!r} — must be a single finite number"
        )
        failures += 1
    else:
        units_str = f" {business_units}" if business_units else ""
        _pass("Business answer computed", f"{float(business_answer):.2f}{units_str}")

    # Check 6: business units string provided
    if not isinstance(business_units, str) or not business_units.strip():
        _fail(
            "Business units specified",
            "art.business_units must be a non-empty string (e.g. 'units/week', 'MWh', 'USD')"
        )
        failures += 1
    else:
        _pass("Business units specified", f"'{business_units}'")

    return failures


def check_milestone_4(art) -> int:
    """Explainability report with attribution dict and top-5 features."""
    _section("Milestone 4: Explainability Report")
    failures = 0

    explanations = getattr(art, "explanations", None)

    # Check 1: explanations exists
    if explanations is None:
        _fail(
            "explanations produced",
            "art.explanations must be the dict returned by nf.models[0].explain()"
        )
        failures += 1
        return failures
    elif not isinstance(explanations, dict):
        _fail("explanations is dict", f"Got type {type(explanations).__name__}")
        failures += 1
        return failures
    else:
        _pass("explanations produced", "dict")

    # Check 2: required keys present
    for key in ("attributions", "feature_names"):
        if key not in explanations:
            _fail(f"explanations['{key}'] key", f"Key '{key}' missing from explanations dict")
            failures += 1
        else:
            _pass(f"explanations['{key}'] key", "present")

    if "attributions" in explanations and "feature_names" in explanations:
        attr = explanations["attributions"]
        feat = explanations["feature_names"]

        # Check 3: attributions is array-like with numeric values
        try:
            attr_arr = np.array(attr, dtype=float)
            if attr_arr.ndim not in (1, 2):
                raise ValueError(f"Expected 1D or 2D, got shape {attr_arr.shape}")
            _pass("attributions is numeric array", f"shape {attr_arr.shape}")
        except Exception as exc:
            _fail("attributions is numeric array", str(exc))
            failures += 1
            attr_arr = None

        # Check 4: feature_names is a list of strings
        if not isinstance(feat, (list, np.ndarray)) or len(feat) == 0:
            _fail("feature_names is non-empty list", f"Got {type(feat).__name__} of length {len(feat) if hasattr(feat, '__len__') else '?'}")
            failures += 1
        else:
            _pass("feature_names is non-empty list", f"{len(feat)} features")

        # Check 5: can extract top-5 features
        if attr_arr is not None and len(feat) > 0:
            try:
                if attr_arr.ndim == 2:
                    mean_abs = np.abs(attr_arr).mean(axis=0)
                else:
                    mean_abs = np.abs(attr_arr)
                top5_idx = np.argsort(mean_abs)[-5:][::-1]
                top5 = [(feat[i], float(mean_abs[i])) for i in top5_idx]
                summary = ", ".join(f"{name} ({val:.3f})" for name, val in top5[:3])
                _pass("Top-5 features extractable", f"Top 3: {summary}")
            except Exception as exc:
                _fail("Top-5 features extractable", str(exc))
                failures += 1

    return failures


# ── Main runner ───────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{BOLD}Portfolio Project — Milestone Checklist{RESET}")
    print("=" * 60)
    print("Loading artifacts from project_artifacts.py ...")

    art = _load_artifacts()

    total_failures = 0
    total_failures += check_milestone_1(art)
    total_failures += check_milestone_2(art)
    total_failures += check_milestone_3(art)
    total_failures += check_milestone_4(art)

    print()
    print("=" * 60)
    if total_failures == 0:
        print(f"{GREEN}{BOLD}All checks passed. Project is structurally complete.{RESET}")
        print(
            "Review the stakeholder summary section of your notebook\n"
            "to confirm the executive paragraph reads clearly to a\n"
            "non-technical audience."
        )
    else:
        print(f"{RED}{BOLD}{total_failures} check(s) failed.{RESET}")
        print(
            "Fix the FAIL items above and re-run this script.\n"
            "Each failure message explains what is missing and how to fix it."
        )
    print()


if __name__ == "__main__":
    main()
