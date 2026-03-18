"""
nowcasting_pipeline.py — Production MIDAS Nowcasting Pipeline Template

End-to-end pipeline:
  ingest → ragged-edge fill → MIDAS features → estimate → diagnostics → publish

Usage
-----
    from nowcasting_pipeline import NowcastingPipeline, PipelineConfig

    config = PipelineConfig(
        target_series_id="GDPC1",
        indicator_ids=["PAYEMS", "INDPRO", "RETAILSL", "NAPM"],
        n_lags=3,
        model="elasticnet",
        vintage_db_path="data/vintages.db",
        forecast_db_path="data/forecasts.db",
    )
    pipeline = NowcastingPipeline(config)
    record = pipeline.run(as_of_date=datetime.date.today())
    print(record)

Adapt by:
  - Replacing FREDClient with your data source
  - Adjusting PUB_LAGS for your indicator set
  - Swapping ElasticNetCV for any sklearn-compatible estimator
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, RidgeCV, LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from scipy import stats

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("nowcasting_pipeline")

# ──────────────────────────────────────────────────────────────────────────────
# Publication lags (calendar days after reference period end)
# Adjust these to match your actual data sources.
# ──────────────────────────────────────────────────────────────────────────────

PUB_LAGS: Dict[str, int] = {
    "NAPM":     1,   # ISM Manufacturing PMI
    "PAYEMS":   4,   # Nonfarm Payrolls
    "ICSA":     5,   # Initial Claims
    "RETAILSL": 14,  # Retail Sales
    "INDPRO":   16,  # Industrial Production
    "CPIAUCSL": 16,  # CPI All Items
    "GDPC1":    28,  # Advance GDP (quarterly)
}

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    target_series_id: str
    indicator_ids: List[str]
    n_lags: int = 3
    fill_method: str = "carry_forward"   # "carry_forward" | "zero" | "ar1"
    model: str = "elasticnet"            # "elasticnet" | "ridge" | "lasso"
    vintage_db_path: str = "data/vintages.db"
    forecast_db_path: str = "data/forecasts.db"
    fred_api_key: str = ""
    cv_splits: int = 5
    bootstrap_samples: int = 200
    alpha_pi: float = 0.05               # prediction interval coverage = 1-alpha


# ──────────────────────────────────────────────────────────────────────────────
# Layer 1 — Forecast Record (output contract)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ForecastRecord:
    """Immutable record of a single pipeline run."""
    forecast_date: str
    target_period: str
    model_name: str
    point_forecast: float
    lower_80: float
    upper_80: float
    lower_95: float
    upper_95: float
    n_train: int
    indicators_used: Tuple[str, ...]
    news_decomposition: Dict[str, float]
    run_timestamp: str

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["indicators_used"] = list(self.indicators_used)
        return d

    def __str__(self) -> str:
        return (
            f"Forecast for {self.target_period} "
            f"(as of {self.forecast_date}): "
            f"{self.point_forecast:+.3f} "
            f"[{self.lower_95:+.3f}, {self.upper_95:+.3f}]"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Layer 2 — Vintage Database
# ──────────────────────────────────────────────────────────────────────────────

class VintageDatabase:
    """
    SQLite-backed immutable vintage store.

    Schema
    ------
    vintages(series_id TEXT, obs_date TEXT, vintage_date TEXT, value REAL)
    PRIMARY KEY (series_id, obs_date, vintage_date)

    Rows are never modified. Revisions create new rows with a later vintage_date.
    """

    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
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

    def store_bulk(
        self,
        series_id: str,
        data: pd.Series,
        vintage_date: datetime.date,
    ) -> None:
        """Insert a complete series snapshot (one vintage)."""
        rows = [
            (series_id, str(idx.date()), vintage_date.isoformat(), float(val))
            for idx, val in data.items()
            if pd.notna(val)
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO vintages VALUES (?, ?, ?, ?)", rows
        )
        self.conn.commit()
        logger.debug(f"Stored {len(rows)} rows for {series_id} @ {vintage_date}")

    def get_as_of(self, series_id: str, as_of_date: datetime.date) -> pd.Series:
        """Return series using the latest vintage available as of as_of_date."""
        df = pd.read_sql_query(
            """
            SELECT obs_date, value FROM vintages
            WHERE series_id = ?
              AND vintage_date <= ?
            ORDER BY obs_date, vintage_date DESC
            """,
            self.conn,
            params=(series_id, as_of_date.isoformat()),
        )
        if df.empty:
            return pd.Series(dtype=float, name=series_id)
        df = df.drop_duplicates("obs_date", keep="first")
        df["obs_date"] = pd.to_datetime(df["obs_date"])
        return df.set_index("obs_date")["value"].sort_index()


# ──────────────────────────────────────────────────────────────────────────────
# Layer 2b — FRED Data Client
# ──────────────────────────────────────────────────────────────────────────────

class FREDClient:
    """Fetch FRED series with exponential back-off retry."""

    BASE = "https://api.stlouisfed.org/fred"
    DELAYS = [0, 1, 2, 4, 8]

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch(self, series_id: str, start: Optional[str] = None) -> pd.Series:
        """Return the latest-vintage FRED series as a pd.Series."""
        import requests

        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }
        if start:
            params["observation_start"] = start

        last_exc: Optional[Exception] = None
        for delay in self.DELAYS:
            if delay:
                time.sleep(delay)
            try:
                resp = requests.get(
                    f"{self.BASE}/series/observations",
                    params=params,
                    timeout=30,
                )
                resp.raise_for_status()
                obs = resp.json().get("observations", [])
                values = {
                    o["date"]: float(o["value"])
                    for o in obs
                    if o["value"] != "."
                }
                logger.info(f"Fetched {len(values)} obs for {series_id}")
                return pd.Series(values, name=series_id, dtype=float)
            except Exception as exc:
                last_exc = exc
                logger.warning(f"Fetch attempt failed for {series_id}: {exc}")

        raise RuntimeError(
            f"Failed to fetch {series_id} after retries: {last_exc}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Layer 3 — Feature Engineering
# ──────────────────────────────────────────────────────────────────────────────

def fill_ragged_edge(
    series: pd.Series,
    target_end: pd.Timestamp,
    method: str = "carry_forward",
) -> pd.Series:
    """
    Extend series to target_end using the specified fill method.

    Parameters
    ----------
    series : pd.Series (monthly, DatetimeIndex)
    target_end : pd.Timestamp
    method : "carry_forward" | "zero" | "ar1"

    Returns
    -------
    pd.Series extended to target_end, no NaN values.
    """
    if series.empty:
        return series

    full_idx = pd.date_range(series.index[0], target_end, freq="MS")
    extended = series.reindex(full_idx)
    missing = extended.isna()

    if not missing.any():
        return extended

    if method == "carry_forward":
        return extended.ffill()

    elif method == "zero":
        return extended.fillna(0.0)

    elif method == "ar1":
        observed = extended.dropna()
        if len(observed) < 4:
            return extended.ffill()
        y = observed.values
        X = np.column_stack([np.ones(len(y) - 1), y[:-1]])
        b = np.linalg.lstsq(X, y[1:], rcond=None)[0]
        last_val = float(y[-1])
        n_missing = int(missing.sum())
        projections = []
        for _ in range(n_missing):
            next_val = float(b[0] + b[1] * last_val)
            projections.append(next_val)
            last_val = next_val
        extended[missing] = projections
        return extended

    else:
        raise ValueError(f"Unknown fill method: '{method}'")


def build_midas_matrix(
    indicator_data: Dict[str, pd.Series],
    low_freq_dates: pd.DatetimeIndex,
    n_lags: int,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build the (T × K*N) MIDAS feature matrix.

    For each low-frequency date t and each indicator, we take the
    n_lags most recent high-frequency observations available as of t.

    Returns
    -------
    X : np.ndarray, shape (T, n_lags * N)
    feature_names : List[str]
    """
    T = len(low_freq_dates)
    N = len(indicator_data)
    X = np.full((T, n_lags * N), np.nan)
    feature_names: List[str] = []

    col = 0
    for series_id, series in indicator_data.items():
        for lag in range(n_lags):
            feature_names.append(f"{series_id}_lag{lag + 1}")
            for t, lf_date in enumerate(low_freq_dates):
                available = series[series.index <= lf_date]
                if len(available) > lag:
                    X[t, col] = float(available.iloc[-(lag + 1)])
            col += 1

    return X, feature_names


# ──────────────────────────────────────────────────────────────────────────────
# Layer 4 — Estimation
# ──────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "elasticnet": lambda cv: ElasticNetCV(
        l1_ratio=[0.5, 0.7, 1.0], cv=cv, max_iter=10000, random_state=0
    ),
    "ridge": lambda cv: RidgeCV(cv=cv),
    "lasso": lambda cv: LassoCV(cv=cv, max_iter=10000, random_state=0),
}


def parametric_intervals(
    point: float,
    residuals: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, float, float, float]:
    """
    Return (lo_80, hi_80, lo_95, hi_95) using t-distribution.
    """
    n = len(residuals)
    sigma = float(residuals.std(ddof=1))
    t80 = stats.t.ppf(0.90, df=max(n - 2, 1))
    t95 = stats.t.ppf(0.975, df=max(n - 2, 1))
    return (
        point - t80 * sigma,
        point + t80 * sigma,
        point - t95 * sigma,
        point + t95 * sigma,
    )


def compute_news_decomposition(
    coef: np.ndarray,
    feature_names: List[str],
    x_prev: np.ndarray,
    x_curr: np.ndarray,
    n_lags: int,
    indicator_ids: List[str],
) -> Dict[str, float]:
    """
    Compute indicator-level news contributions to the nowcast revision.

    delta_forecast = sum_i beta_i * (x_curr_i - x_prev_i)

    Groups lag-level contributions by indicator.
    """
    delta = x_curr - x_prev
    contributions = coef * delta
    news: Dict[str, float] = {}
    for j, sid in enumerate(indicator_ids):
        start = j * n_lags
        end = (j + 1) * n_lags
        news[sid] = float(np.nansum(contributions[start:end]))
    return news


# ──────────────────────────────────────────────────────────────────────────────
# Layer 5 — Forecast Database
# ──────────────────────────────────────────────────────────────────────────────

class ForecastDatabase:
    """Append-only SQLite store for ForecastRecord objects."""

    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
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
        return cur.lastrowid  # type: ignore[return-value]

    def get_history(self, model_name: str, target_period: str) -> pd.DataFrame:
        """Return all forecasts for a given model and target period."""
        return pd.read_sql_query(
            """
            SELECT * FROM forecasts
            WHERE model_name = ? AND target_period = ?
            ORDER BY forecast_date
            """,
            self.conn,
            params=(model_name, target_period),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

class NowcastingPipeline:
    """
    End-to-end nowcasting pipeline orchestrator.

    Usage
    -----
    pipeline = NowcastingPipeline(config)
    record = pipeline.run(
        as_of_date=datetime.date.today(),
        target_period=pd.Timestamp("2024-10-01"),
        train_dates=pd.date_range("2010-01-01", "2024-07-01", freq="QS"),
    )
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.vintage_db = VintageDatabase(config.vintage_db_path)
        self.forecast_db = ForecastDatabase(config.forecast_db_path)
        if config.fred_api_key:
            self.fred_client: Optional[FREDClient] = FREDClient(config.fred_api_key)
        else:
            self.fred_client = None
        logger.info(
            f"Pipeline initialised: target={config.target_series_id}, "
            f"model={config.model}, lags={config.n_lags}"
        )

    def ingest(self, as_of_date: datetime.date) -> Dict[str, pd.Series]:
        """
        Layer 2: Retrieve all series from the vintage database.
        If FRED client is available and a series has no vintage data, fetch it.
        """
        all_ids = [self.config.target_series_id] + self.config.indicator_ids
        data: Dict[str, pd.Series] = {}

        for sid in all_ids:
            series = self.vintage_db.get_as_of(sid, as_of_date)
            if series.empty and self.fred_client is not None:
                logger.info(f"Fetching {sid} from FRED (no vintage data found)")
                series = self.fred_client.fetch(sid)
                self.vintage_db.store_bulk(sid, series, as_of_date)
                series = self.vintage_db.get_as_of(sid, as_of_date)
            data[sid] = series

        return data

    def engineer_features(
        self,
        raw_data: Dict[str, pd.Series],
        as_of_date: datetime.date,
        low_freq_dates: pd.DatetimeIndex,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Layer 3: Fill ragged edges and build MIDAS lag matrix.
        """
        current_month = pd.Timestamp(as_of_date).to_period("M").to_timestamp()
        indicator_data: Dict[str, pd.Series] = {}

        for sid in self.config.indicator_ids:
            series = raw_data.get(sid, pd.Series(dtype=float))
            if series.empty:
                logger.warning(f"{sid}: no data available; using zero-filled series")
                idx = pd.date_range(
                    low_freq_dates[0],
                    current_month,
                    freq="MS",
                )
                series = pd.Series(0.0, index=idx, name=sid)
            else:
                series = fill_ragged_edge(
                    series, current_month, method=self.config.fill_method
                )
            indicator_data[sid] = series

        return build_midas_matrix(indicator_data, low_freq_dates, self.config.n_lags)

    def estimate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_forecast: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Layer 4: Fit model and generate point + interval forecast.

        Returns
        -------
        point : float
        coef : np.ndarray of model coefficients
        X_train_scaled : np.ndarray (for news decomp)
        X_forecast_scaled : np.ndarray
        """
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_fc_s = scaler.transform(X_forecast)

        cv = TimeSeriesSplit(n_splits=self.config.cv_splits)
        builder = MODEL_REGISTRY.get(self.config.model)
        if builder is None:
            raise ValueError(
                f"Unknown model '{self.config.model}'. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )
        model = builder(cv)
        model.fit(X_tr_s, y_train)
        point = float(model.predict(X_fc_s)[0])

        return point, model.coef_, X_tr_s, X_fc_s

    def run(
        self,
        as_of_date: datetime.date,
        target_period: pd.Timestamp,
        train_dates: pd.DatetimeIndex,
        x_prev_scaled: Optional[np.ndarray] = None,
    ) -> ForecastRecord:
        """
        Execute the full pipeline for one (as_of_date, target_period) pair.

        Parameters
        ----------
        as_of_date : datetime.date
            The data vintage date (only data available on this date is used).
        target_period : pd.Timestamp
            The period being nowcast (e.g. pd.Timestamp("2024-10-01") for Q4 2024).
        train_dates : pd.DatetimeIndex
            All historical low-frequency periods to use as training targets.
        x_prev_scaled : np.ndarray, optional
            Scaled feature vector from the previous run (for news decomposition).
        """
        logger.info(f"Pipeline run: as_of={as_of_date}, target={target_period.date()}")

        # Layer 2
        raw_data = self.ingest(as_of_date)

        # Layer 3
        all_dates = pd.DatetimeIndex(list(train_dates) + [target_period])
        X, feature_names = self.engineer_features(raw_data, as_of_date, all_dates)

        X_train = X[:-1]
        X_forecast = X[-1:]
        y_train = raw_data[self.config.target_series_id].reindex(train_dates).values

        # Drop NaN rows
        mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
        X_train = X_train[mask]
        y_train = y_train[mask]
        n_train = len(y_train)

        if n_train < 16 or np.isnan(X_forecast).any():
            raise ValueError(
                f"Insufficient data: n_train={n_train}, "
                f"NaN in forecast features: {np.isnan(X_forecast).any()}"
            )

        # Layer 4
        point, coef, X_tr_s, X_fc_s = self.estimate(X_train, y_train, X_forecast)

        residuals = y_train - np.dot(X_tr_s, coef)
        lo_80, hi_80, lo_95, hi_95 = parametric_intervals(point, residuals)

        # News decomposition (if previous feature vector provided)
        news: Dict[str, float] = {}
        if x_prev_scaled is not None:
            news = compute_news_decomposition(
                coef, feature_names,
                x_prev_scaled.flatten(), X_fc_s.flatten(),
                self.config.n_lags, self.config.indicator_ids,
            )

        record = ForecastRecord(
            forecast_date=as_of_date.isoformat(),
            target_period=str(target_period.date()),
            model_name=self.config.model,
            point_forecast=point,
            lower_80=lo_80,
            upper_80=hi_80,
            lower_95=lo_95,
            upper_95=hi_95,
            n_train=n_train,
            indicators_used=tuple(self.config.indicator_ids),
            news_decomposition=news,
            run_timestamp=datetime.datetime.utcnow().isoformat(),
        )

        # Layer 5
        self.forecast_db.insert(record)
        logger.info(f"Forecast stored: {record}")
        return record


# ──────────────────────────────────────────────────────────────────────────────
# Convenience function: expanding-window backtest
# ──────────────────────────────────────────────────────────────────────────────

def run_expanding_backtest(
    pipeline: NowcastingPipeline,
    all_low_freq_dates: pd.DatetimeIndex,
    min_train: int = 20,
) -> pd.DataFrame:
    """
    Run an expanding-window pseudo-real-time backtest.

    For each date in all_low_freq_dates[min_train:], treat it as the
    target period and all prior dates as training. Uses the last-day-of-quarter
    as the 'as_of_date' (conservative: assumes all within-quarter indicators
    are available by end of quarter).

    Returns pd.DataFrame with columns: quarter, forecast, actual, error
    """
    results = []
    n = len(all_low_freq_dates)

    for t in range(min_train, n - 1):
        target = all_low_freq_dates[t]
        train_dates = all_low_freq_dates[:t]

        # Use end of quarter as as_of_date
        quarter_end = target + pd.offsets.QuarterEnd(0)
        as_of = quarter_end.date()

        try:
            rec = pipeline.run(
                as_of_date=as_of,
                target_period=target,
                train_dates=train_dates,
            )
            raw_data = pipeline.ingest(as_of)
            actual_series = raw_data.get(pipeline.config.target_series_id, pd.Series())
            actual = float(actual_series.get(target, np.nan))

            results.append({
                "quarter": target,
                "forecast": rec.point_forecast,
                "actual": actual,
                "error": rec.point_forecast - actual,
                "lower_95": rec.lower_95,
                "upper_95": rec.upper_95,
            })
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Backtest failed for {target.date()}: {e}")

    df = pd.DataFrame(results)
    if len(df) > 0:
        rmse = float(np.sqrt((df["error"] ** 2).mean()))
        logger.info(f"Backtest RMSE: {rmse:.4f} ({len(df)} quarters)")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Example usage (run as script)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("NowcastingPipeline template loaded successfully.")
    print()
    print("To use with FRED data:")
    print("  config = PipelineConfig(")
    print('      target_series_id="GDPC1",')
    print('      indicator_ids=["PAYEMS", "INDPRO", "RETAILSL"],')
    print('      fred_api_key=os.environ["FRED_API_KEY"],')
    print("  )")
    print("  pipeline = NowcastingPipeline(config)")
    print()
    print("To pre-populate the vintage database from local CSVs:")
    print("  data = pd.read_csv('data/PAYEMS.csv', index_col=0, parse_dates=True)")
    print("  pipeline.vintage_db.store_bulk('PAYEMS', data['value'],")
    print("      vintage_date=datetime.date.today())")
