# Pipeline Architecture for Real-Time Nowcasting

> **Reading time:** ~20 min | **Module:** 08 — Production Systems | **Prerequisites:** Module 7


## Learning Objectives

<div class="flow">
<div class="flow-step mint">1. Load Ragged Data</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">2. Estimate MIDAS</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">3. Generate Nowcast</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. Update as Data Arrives</div>
</div>


<div class="callout-key">

**Key Concept Summary:** A research nowcasting script answers: "What is the best model?" A production pipeline answers: "What is the best estimate *right now*, given exactly what data exist *right now*, and how do we deliv...

</div>

After reading this guide you will be able to:

1. Design the component architecture of a production nowcasting pipeline
2. Build a publication-calendar-aware data acquisition layer
3. Implement revision tracking and vintage management
4. Handle ragged-edge data deterministically in production
5. Schedule automated re-estimation and forecast publication

---

## 1. Why Production Pipelines Differ from Research Code

A research nowcasting script answers: "What is the best model?" A production pipeline answers: "What is the best estimate *right now*, given exactly what data exist *right now*, and how do we deliver that estimate reliably?"

The gap between the two is large. Research code can assume data are always available, cleanly formatted, and never revised. Production code must handle:

<div class="callout-insight">

**Insight:** Real-time nowcasting is fundamentally different from pseudo out-of-sample backtesting. The ragged-edge data structure means your model sees different information at different points within a quarter.

</div>


| Challenge | Research assumption | Production reality |
|-----------|--------------------|--------------------|
| Data availability | All series exist | Ragged edge: some series lag by weeks |
| Revisions | Final vintage only | Preliminary data revised 2–5 times |
| Timing | Run once | Automated: runs on each release event |
| Failures | Reraise exception | Retry, fallback, alert, and continue |
| Audit trail | None needed | Full reproducibility required |
| Latency | Minutes acceptable | <60 s from data pull to published forecast |

Production pipelines at the NY Fed, ECB, and Bank of England address all of these. This guide describes how to build a scaled-down version suitable for a team of analysts.

---

## 2. Architecture Overview

<div class="callout-warning">

**Warning:** Pseudo out-of-sample exercises that do not properly account for the real-time data vintage will overstate nowcast accuracy. Always use the ragged-edge structure that would have been available at each historical nowcast date.

</div>


A nowcasting pipeline has five layers. Each layer has a single responsibility and communicates with adjacent layers through well-defined interfaces.

```
┌────────────────────────────────────────────────────────────┐
│  Layer 1 — Scheduler                                       │
│  Triggers pipeline on publication calendar events          │
└───────────────────────────┬────────────────────────────────┘
                            │ trigger(release_id, timestamp)
┌───────────────────────────▼────────────────────────────────┐
│  Layer 2 — Data Acquisition                                │
│  Fetches raw series, stores versioned vintage snapshots    │
└───────────────────────────┬────────────────────────────────┘
                            │ vintage_df(series_id, as_of_date)
┌───────────────────────────▼────────────────────────────────┐
│  Layer 3 — Feature Engineering                             │
│  Ragged-edge filling, MIDAS lag construction, scaling      │
└───────────────────────────┬────────────────────────────────┘
                            │ X_midas, y, forecast_date
┌───────────────────────────▼────────────────────────────────┐
│  Layer 4 — Estimation & Forecasting                        │
│  Fits MIDAS model, generates point + interval forecast     │
└───────────────────────────┬────────────────────────────────┘
                            │ forecast_record
┌───────────────────────────▼────────────────────────────────┐
│  Layer 5 — Publication & Monitoring                        │
│  Writes to database, generates report, triggers alerts     │
└────────────────────────────────────────────────────────────┘
```

Each layer is implemented as a Python class. The pipeline orchestrator calls them in sequence and handles exceptions at each boundary.

---

## 3. Layer 1 — The Scheduler

### Publication Calendar

Every economic release has a deterministic publication schedule known in advance. The BLS releases CPI on a fixed calendar; FRED provides API metadata for each series' next release date.

A production system maintains a **publication calendar** — a table of `(series_id, release_date, pub_lag_days)` triples that drives all scheduling.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import datetime
from dataclasses import dataclass, field
from typing import List, Optional
import json
import os

@dataclass
class Release:
    """A single scheduled data release."""
    series_id: str          # e.g. "PAYEMS", "INDPRO"
    release_date: datetime.date
    pub_lag_days: int       # days after reference period end
    frequency: str          # "monthly", "weekly", "daily"
    source: str             # "FRED", "BLS", "BEA"
    priority: int = 1       # higher = run pipeline immediately

@dataclass
class PublicationCalendar:
    """Manages the schedule of economic data releases."""
    releases: List[Release] = field(default_factory=list)

    def add_release(self, release: Release) -> None:
        self.releases.append(release)

    def releases_on(self, date: datetime.date) -> List[Release]:
        return [r for r in self.releases if r.release_date == date]

    def releases_between(
        self,
        start: datetime.date,
        end: datetime.date
    ) -> List[Release]:
        return [
            r for r in self.releases
            if start <= r.release_date <= end
        ]

    def next_release_after(self, date: datetime.date) -> Optional[Release]:
        future = [r for r in self.releases if r.release_date > date]
        if not future:
            return None
        return min(future, key=lambda r: r.release_date)

    def to_json(self, path: str) -> None:
        data = [
            {
                "series_id": r.series_id,
                "release_date": r.release_date.isoformat(),
                "pub_lag_days": r.pub_lag_days,
                "frequency": r.frequency,
                "source": r.source,
                "priority": r.priority,
            }
            for r in self.releases
        ]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
```

</div>

### Standard Publication Lags

The table below gives typical lags from the end of the reference period to the first data release. Use these to build your calendar.

| Series | Mnemonic | Lag (days) | Frequency |
|--------|----------|------------|-----------|
| ISM Manufacturing PMI | NAPM | 1 | Monthly |
| Nonfarm Payrolls | PAYEMS | 4 | Monthly |
| Initial Claims | ICSA | 5 | Weekly |
| Retail Sales | RETAILSL | 14 | Monthly |
| Industrial Production | INDPRO | 16 | Monthly |
| CPI All Items | CPIAUCSL | 16 | Monthly |
| Advance GDP | GDP | 28 | Quarterly |

### Scheduler Implementation

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import time
import logging
from typing import Callable

logger = logging.getLogger(__name__)

class PipelineScheduler:
    """Polls the publication calendar and triggers the pipeline."""

    def __init__(
        self,
        calendar: PublicationCalendar,
        pipeline_fn: Callable,
        poll_interval_seconds: int = 3600,
    ):
        self.calendar = calendar
        self.pipeline_fn = pipeline_fn
        self.poll_interval = poll_interval_seconds
        self._processed: set = set()

    def run_forever(self) -> None:
        """Poll calendar and trigger pipeline on release days."""
        logger.info("Scheduler started")
        while True:
            today = datetime.date.today()
            releases = self.calendar.releases_on(today)
            for release in releases:
                key = (release.series_id, release.release_date.isoformat())
                if key not in self._processed:
                    logger.info(
                        f"Triggering pipeline for {release.series_id} "
                        f"released {release.release_date}"
                    )
                    try:
                        self.pipeline_fn(release)
                        self._processed.add(key)
                    except Exception as exc:
                        logger.error(
                            f"Pipeline failed for {release.series_id}: {exc}"
                        )
            time.sleep(self.poll_interval)
```

</div>

---

## 4. Layer 2 — Data Acquisition and Vintage Storage

### Why Store Vintages?

When the BLS publishes payrolls in early October, it simultaneously revises September's preliminary figure. If you only store the latest value, you cannot reconstruct what your model saw in October — you lose the ability to evaluate it honestly.

**Vintage storage** means saving a complete snapshot of every series at every point in time. The ALFRED (Archival FRED) API provides historical vintages for FRED series.

### Vintage Database Design

The minimal viable schema stores one row per `(series_id, observation_date, vintage_date, value)` tuple.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import sqlite3
import pandas as pd
from datetime import date

class VintageDatabase:
    """
    Stores versioned time series data.

    Schema:
        vintages(series_id TEXT, obs_date TEXT, vintage_date TEXT, value REAL)
        PRIMARY KEY (series_id, obs_date, vintage_date)
    """

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS vintages (
                series_id    TEXT NOT NULL,
                obs_date     TEXT NOT NULL,
                vintage_date TEXT NOT NULL,
                value        REAL,
                PRIMARY KEY (series_id, obs_date, vintage_date)
            )
        """)
        self.conn.commit()

    def insert_vintage(
        self,
        series_id: str,
        obs_date: date,
        vintage_date: date,
        value: float,
    ) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO vintages
                (series_id, obs_date, vintage_date, value)
            VALUES (?, ?, ?, ?)
            """,
            (series_id, obs_date.isoformat(), vintage_date.isoformat(), value),
        )
        self.conn.commit()

    def get_as_of(self, series_id: str, as_of_date: date) -> pd.Series:
        """
        Retrieve the latest available vintage for each observation
        as of `as_of_date`. This is the pseudo-real-time query.
        """
        df = pd.read_sql_query(
            """
            SELECT obs_date, value
            FROM vintages
            WHERE series_id = ?
              AND vintage_date <= ?
            ORDER BY obs_date, vintage_date DESC
            """,
            self.conn,
            params=(series_id, as_of_date.isoformat()),
        )
        # Keep the latest vintage for each obs_date
        df = df.drop_duplicates(subset="obs_date", keep="first")
        df["obs_date"] = pd.to_datetime(df["obs_date"])
        return df.set_index("obs_date")["value"].sort_index()

    def store_bulk(
        self,
        series_id: str,
        data: pd.Series,
        vintage_date: date,
    ) -> None:
        """Store a full series snapshot as a single vintage."""
        rows = [
            (series_id, str(idx.date()), vintage_date.isoformat(), float(val))
            for idx, val in data.items()
            if pd.notna(val)
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO vintages VALUES (?, ?, ?, ?)", rows
        )
        self.conn.commit()
```

</div>

### Data Acquisition Layer

```python
import requests
import time
from typing import Dict, Optional

class FREDClient:
    """Fetches series data from the FRED API with retry logic."""

    BASE_URL = "https://api.stlouisfed.org/fred"
    RETRY_DELAYS = [1, 2, 4, 8]  # exponential back-off seconds

    def __init__(self, api_key: str, cache_dir: str = ".cache"):
        self.api_key = api_key
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def fetch_series(
        self,
        series_id: str,
        observation_start: Optional[str] = None,
    ) -> pd.Series:
        """Fetch latest vintage of a series."""
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }
        if observation_start:
            params["observation_start"] = observation_start

        for delay in [0] + self.RETRY_DELAYS:
            if delay:
                time.sleep(delay)
            try:
                resp = requests.get(
                    f"{self.BASE_URL}/series/observations",
                    params=params,
                    timeout=30,
                )
                resp.raise_for_status()
                obs = resp.json()["observations"]
                values = {
                    o["date"]: float(o["value"])
                    for o in obs
                    if o["value"] != "."
                }
                return pd.Series(values, name=series_id, dtype=float)
            except (requests.RequestException, KeyError, ValueError) as exc:
                logger.warning(f"FRED fetch attempt failed: {exc}")

        raise RuntimeError(f"Failed to fetch {series_id} after retries")

    def fetch_vintage_dates(self, series_id: str) -> List[str]:
        """Return list of all vintage dates for a series (ALFRED)."""
        resp = requests.get(
            f"{self.BASE_URL}/series/vintagedates",
            params={
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["vintage_dates"]
```

---

## 5. Layer 3 — Feature Engineering

### Ragged-Edge Handling in Production

In production the ragged edge arises on every run because different series have different publication lags. The feature engineering layer must compute, for each series, whether its latest observation is available for the current reference period — and if not, fill the missing values deterministically.

```python
import numpy as np
from typing import Literal

FillMethod = Literal["carry_forward", "zero", "ar1"]

def fill_ragged_edge(
    series: pd.Series,
    target_end: pd.Timestamp,
    method: FillMethod = "carry_forward",
) -> pd.Series:
    """
    Extend series to target_end using the specified fill method.

    Parameters
    ----------
    series : pd.Series
        Monthly series with DatetimeIndex (month-end or month-start).
    target_end : pd.Timestamp
        The last period that should be present after filling.
    method : str
        'carry_forward' : repeat last observed value
        'zero'          : fill with zero (appropriate for growth rates
                          with zero mean under the null)
        'ar1'           : extrapolate using OLS AR(1) on last 24 obs

    Returns
    -------
    pd.Series
        Extended series, never shorter than input.
    """
    freq = pd.infer_freq(series.index)
    if freq is None:
        freq = "MS"

    full_index = pd.date_range(series.index[0], target_end, freq=freq)
    extended = series.reindex(full_index)

    missing_mask = extended.isna()
    if not missing_mask.any():
        return extended

    if method == "carry_forward":
        extended = extended.ffill()

    elif method == "zero":
        extended = extended.fillna(0.0)

    elif method == "ar1":
        observed = extended.dropna()
        if len(observed) < 4:
            extended = extended.ffill()
        else:
            y = observed.values
            X = np.column_stack([np.ones(len(y) - 1), y[:-1]])
            b = np.linalg.lstsq(X, y[1:], rcond=None)[0]
            last_val = y[-1]
            n_missing = missing_mask.sum()
            projections = []
            for _ in range(n_missing):
                next_val = b[0] + b[1] * last_val
                projections.append(next_val)
                last_val = next_val
            extended[missing_mask] = projections

    return extended


def build_midas_feature_matrix(
    high_freq: pd.Series,
    low_freq_dates: pd.DatetimeIndex,
    n_lags: int,
    fill_method: FillMethod = "carry_forward",
) -> np.ndarray:
    """
    Construct the K×T MIDAS lag matrix aligned to low-frequency dates.

    Each row t corresponds to the most recent n_lags observations of
    high_freq available as of low_freq_dates[t].
    """
    n_periods = len(low_freq_dates)
    X = np.full((n_periods, n_lags), np.nan)

    for t, lf_date in enumerate(low_freq_dates):
        available = high_freq[high_freq.index <= lf_date]
        if len(available) >= n_lags:
            X[t, :] = available.values[-n_lags:][::-1]

    return X
```

### Scaling and Normalisation

MIDAS features built from multiple indicators with different units must be standardised before entering a regularised model. The scaler is fit on the training window only and applied to the forecast window — never fit on the full sample.

```python
from sklearn.preprocessing import StandardScaler

class FeaturePipeline:
    """Stateful feature pipeline: fit on train, transform on all."""

    def __init__(self, n_lags: int, fill_method: FillMethod = "carry_forward"):
        self.n_lags = n_lags
        self.fill_method = fill_method
        self.scaler = StandardScaler()
        self._fitted = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.fit_transform(X)
        self._fitted = True
        return X_scaled

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit_transform on training data first")
        return self.scaler.transform(X)
```

---

## 6. Layer 4 — Estimation and Forecasting

### Model Registry

Production systems must support multiple model specifications. A model registry maps string identifiers to estimator classes, enabling configuration-driven model selection without changing code.

```python
from sklearn.linear_model import ElasticNetCV, RidgeCV
from typing import Type, Any

MODEL_REGISTRY: Dict[str, Any] = {
    "elasticnet": ElasticNetCV,
    "ridge": RidgeCV,
}

def get_estimator(name: str, **kwargs) -> Any:
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](**kwargs)
```

### Forecast Record

Every model run produces a structured forecast record — a dictionary with everything needed to reproduce or audit the result.

```python
from dataclasses import dataclass, asdict
from typing import Tuple

@dataclass
class ForecastRecord:
    """Immutable record of a single model run."""
    forecast_date: str         # ISO date of the run
    target_period: str         # Reference quarter/month being nowcast
    model_name: str
    point_forecast: float
    lower_80: float
    upper_80: float
    lower_95: float
    upper_95: float
    n_train: int
    indicators_used: List[str]
    news_decomposition: Dict[str, float]
    run_timestamp: str

    def to_dict(self) -> Dict:
        return asdict(self)
```

### Prediction Intervals

MIDAS models produce point forecasts. Prediction intervals can be derived via:

1. **Residual bootstrap**: resample residuals from the training window, re-forecast, take quantiles
2. **Conformal prediction**: split-conformal intervals calibrated on a holdout set
3. **Parametric**: assume Gaussian residuals, use `t_{n-p}` critical values

```python
def bootstrap_prediction_interval(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_forecast: np.ndarray,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """
    Returns (point, lower, upper) at confidence level 1-alpha.
    Uses residual bootstrap: resample fitted residuals, re-fit, re-forecast.
    """
    model.fit(X_train, y_train)
    point = float(model.predict(X_forecast)[0])
    residuals = y_train - model.predict(X_train)

    boot_forecasts = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        y_boot = model.predict(X_train) + rng.choice(residuals, size=len(y_train))
        model.fit(X_train, y_boot)
        boot_forecasts.append(float(model.predict(X_forecast)[0]))

    lo = float(np.percentile(boot_forecasts, 100 * alpha / 2))
    hi = float(np.percentile(boot_forecasts, 100 * (1 - alpha / 2)))
    return point, lo, hi
```

---

## 7. Layer 5 — Publication and Storage

### Forecast Database

```python
class ForecastDatabase:
    """Stores and retrieves forecast records."""

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS forecasts (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                forecast_date    TEXT NOT NULL,
                target_period    TEXT NOT NULL,
                model_name       TEXT NOT NULL,
                point_forecast   REAL NOT NULL,
                lower_80         REAL,
                upper_80         REAL,
                lower_95         REAL,
                upper_95         REAL,
                n_train          INTEGER,
                run_timestamp    TEXT NOT NULL,
                metadata_json    TEXT
            )
        """)
        self.conn.commit()

    def insert(self, record: ForecastRecord) -> int:
        import json
        cur = self.conn.execute(
            """
            INSERT INTO forecasts (
                forecast_date, target_period, model_name,
                point_forecast, lower_80, upper_80, lower_95, upper_95,
                n_train, run_timestamp, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.forecast_date,
                record.target_period,
                record.model_name,
                record.point_forecast,
                record.lower_80,
                record.upper_80,
                record.lower_95,
                record.upper_95,
                record.n_train,
                record.run_timestamp,
                json.dumps(record.news_decomposition),
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_history(
        self,
        model_name: str,
        target_period: str,
    ) -> pd.DataFrame:
        return pd.read_sql_query(
            """
            SELECT * FROM forecasts
            WHERE model_name = ? AND target_period = ?
            ORDER BY forecast_date
            """,
            self.conn,
            params=(model_name, target_period),
        )
```

---

## 8. Orchestrator: Wiring the Layers

The orchestrator is the entry point for each pipeline run. It calls each layer in sequence, catches layer-specific exceptions, and writes the final forecast record to the database.

```python
import datetime
import logging
import traceback

class NowcastingPipeline:
    """
    End-to-end orchestrator for a MIDAS nowcasting pipeline.

    Usage
    -----
    pipeline = NowcastingPipeline(config)
    pipeline.run(as_of_date=datetime.date.today())
    """

    def __init__(self, config: Dict):
        self.config = config
        self.vintage_db = VintageDatabase(config["vintage_db_path"])
        self.forecast_db = ForecastDatabase(config["forecast_db_path"])
        self.fred_client = FREDClient(config.get("fred_api_key", ""))
        self.feature_pipeline = FeaturePipeline(
            n_lags=config.get("n_lags", 12),
            fill_method=config.get("fill_method", "carry_forward"),
        )
        self.model = get_estimator(
            config.get("model", "elasticnet"),
            cv=5,
            max_iter=10000,
        )
        logger.info("Pipeline initialised")

    def run(self, as_of_date: datetime.date) -> ForecastRecord:
        logger.info(f"Pipeline run: as_of={as_of_date}")

        # Layer 2 — acquire data
        vintage_data = self._acquire_data(as_of_date)

        # Layer 3 — engineer features
        X, y, lf_dates = self._build_features(vintage_data, as_of_date)

        # Layer 4 — fit and forecast
        record = self._estimate_and_forecast(X, y, lf_dates, as_of_date)

        # Layer 5 — store
        self.forecast_db.insert(record)
        logger.info(f"Forecast stored: {record.point_forecast:.4f}")

        return record

    def _acquire_data(self, as_of_date: datetime.date) -> Dict[str, pd.Series]:
        data = {}
        for sid in self.config["series_ids"]:
            try:
                data[sid] = self.vintage_db.get_as_of(sid, as_of_date)
            except Exception:
                logger.warning(f"No vintage data for {sid}; skipping")
        return data

    def _build_features(
        self,
        vintage_data: Dict[str, pd.Series],
        as_of_date: datetime.date,
    ):
        target_series = vintage_data.pop(self.config["target_series_id"])
        lf_dates = target_series.index[:-1]  # exclude last (unknown target)
        y = target_series.values[:-1]

        blocks = []
        for sid, series in vintage_data.items():
            filled = fill_ragged_edge(
                series,
                pd.Timestamp(as_of_date),
                method=self.fill_method if hasattr(self, "fill_method")
                       else "carry_forward",
            )
            block = build_midas_feature_matrix(
                filled, lf_dates, self.config.get("n_lags", 12)
            )
            blocks.append(block)

        X = np.hstack(blocks)
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        return X[mask], y[mask], lf_dates[mask]

    def _estimate_and_forecast(self, X, y, lf_dates, as_of_date):
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)

        X_train, y_train = X[:-1], y[:-1]
        X_scaled = self.feature_pipeline.fit_transform(X_train)

        self.model.fit(X_scaled, y_train)
        point = float(self.model.predict(X_scaled[-1:])[0])
        residuals = y_train - self.model.predict(X_scaled)
        sigma = residuals.std()

        from scipy import stats
        t_crit_80 = stats.t.ppf(0.9, df=len(y_train) - 2)
        t_crit_95 = stats.t.ppf(0.975, df=len(y_train) - 2)

        return ForecastRecord(
            forecast_date=as_of_date.isoformat(),
            target_period=str(lf_dates[-1].date()),
            model_name=self.config.get("model", "elasticnet"),
            point_forecast=point,
            lower_80=point - t_crit_80 * sigma,
            upper_80=point + t_crit_80 * sigma,
            lower_95=point - t_crit_95 * sigma,
            upper_95=point + t_crit_95 * sigma,
            n_train=len(y_train),
            indicators_used=self.config["series_ids"],
            news_decomposition={},
            run_timestamp=datetime.datetime.utcnow().isoformat(),
        )
```

---

## 9. Configuration Management

All pipeline parameters live in a single YAML/JSON configuration file. This makes the pipeline auditable and reproducible: two runs with identical configs on identical data produce identical results.

```yaml
# nowcast_config.yaml
pipeline:
  target_series_id: "GDPC1"
  series_ids:
    - "PAYEMS"
    - "INDPRO"
    - "RETAILSL"
    - "NAPM"
    - "CPIAUCSL"
  n_lags: 12
  fill_method: "carry_forward"
  model: "elasticnet"
  vintage_db_path: "data/vintages.db"
  forecast_db_path: "data/forecasts.db"
  fred_api_key: "${FRED_API_KEY}"

scheduler:
  poll_interval_seconds: 3600
  timezone: "America/New_York"

publication:
  output_dir: "reports/"
  alert_email: "team@example.com"
```

Load with:

```python
import yaml
import os

def load_config(path: str) -> Dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    # Expand environment variables
    def expand(obj):
        if isinstance(obj, str) and obj.startswith("${"):
            key = obj[2:-1]
            return os.environ.get(key, obj)
        if isinstance(obj, dict):
            return {k: expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [expand(i) for i in obj]
        return obj
    return expand(raw)
```

---

## 10. Revision Handling

When preliminary data are revised, existing vintage rows remain unchanged. New vintage rows are inserted with the updated `vintage_date`. The `get_as_of` query returns whichever vintage existed at the query date — no rows are ever deleted or modified.

This gives a complete audit trail: to reconstruct the forecast from any past date, query all series with `as_of_date = past_date`. The reconstruction will be exact.

**Revision surprise** — the difference between the revised and preliminary figure for a given observation — feeds into news decomposition:

```python
def compute_revision_surprise(
    db: VintageDatabase,
    series_id: str,
    obs_date: datetime.date,
    preliminary_vintage: datetime.date,
    revised_vintage: datetime.date,
) -> float:
    """
    Compute the data revision for a given observation.
    Returns revised_value - preliminary_value.
    """
    prelim = db.get_as_of(series_id, preliminary_vintage)
    revised = db.get_as_of(series_id, revised_vintage)
    obs_ts = pd.Timestamp(obs_date)
    prelim_val = prelim.get(obs_ts, float("nan"))
    revised_val = revised.get(obs_ts, float("nan"))
    return float(revised_val - prelim_val)
```

---

## 11. Deployment Considerations

### Environment Isolation

Use a `requirements.txt` pinned to exact versions. The nowcasting environment must be reproducible across deployments.

```
numpy==1.26.4
pandas==2.2.1
scikit-learn==1.4.1
scipy==1.12.0
requests==2.31.0
pyyaml==6.0.1
```

### Secrets Management

Never hardcode API keys. Use environment variables or a secrets manager (AWS Secrets Manager, HashiCorp Vault). The config loader above shows the `${ENV_VAR}` pattern.

### Logging Strategy

Use Python's `logging` module. Write structured JSON logs in production so they can be ingested by log aggregation tools (Splunk, CloudWatch, Datadog).

```python
import json
import logging

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_dict = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_dict["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_dict)
```

### Health Checks

Expose a `/health` endpoint if the pipeline runs as a service. Return the timestamp of the last successful run and the current forecast value so monitoring systems can detect stale forecasts.

---

## 12. Summary

<div class="callout-danger">

**Danger:** Never use future information when constructing the high-frequency regressor matrix. In a real-time nowcasting context, you only have data up to the current date -- using the full quarter of monthly data when nowcasting mid-quarter is a look-ahead bias that invalidates your results.

</div>


A production nowcasting pipeline has five distinct layers, each with a single responsibility:

1. **Scheduler** — triggers runs on the publication calendar
2. **Data Acquisition** — fetches series, stores versioned vintages in SQLite
3. **Feature Engineering** — fills ragged edges, builds MIDAS lag matrices, scales
4. **Estimation** — fits the MIDAS model, generates point and interval forecasts
5. **Publication** — writes forecast records, logs, and alerts

The key invariant is **immutability of vintage data**: once stored, a vintage row is never modified. This makes the pipeline fully auditable and reproducible.

In Module 08 Notebook 01 you will build a simplified version of this pipeline end-to-end on synthetic macro data.

---

## References

- Andreou, E., Ghysels, E., & Kourtellos, A. (2013). Should macroeconomic forecasters use daily financial data and how? *Journal of Business & Economic Statistics*, 31(2), 240–251.
- McCracken, M., & Ng, S. (2016). FRED-MD: A monthly database for macroeconomic research. *Journal of Business & Economic Statistics*, 34(4), 574–589.
- NY Fed Staff Reports on the FRBNY Nowcast (2016–present).
- ALFRED (Archival FRED): https://alfred.stlouisfed.org/


---

## Conceptual Practice Questions

**Practice Question 1:** How does the ragged-edge problem affect the reliability of real-time nowcasts compared to pseudo out-of-sample exercises?

**Practice Question 2:** What is the key difference between direct and iterated multi-step forecasts in a MIDAS context?


---

## Cross-References

<a class="link-card" href="./02_monitoring_reporting_guide.md">
  <div class="link-card-title">02 Monitoring Reporting</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_monitoring_reporting_slides.md">
  <div class="link-card-title">02 Monitoring Reporting — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./03_decision_flowchart_guide.md">
  <div class="link-card-title">03 Decision Flowchart</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_decision_flowchart_slides.md">
  <div class="link-card-title">03 Decision Flowchart — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

