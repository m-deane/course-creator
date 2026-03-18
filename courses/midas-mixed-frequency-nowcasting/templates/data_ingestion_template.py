"""
data_ingestion_template.py — FRED / Yahoo Finance Data Ingestion with Caching

Provides:
  - FREDFetcher: fetch and cache FRED series (requires fred_api_key)
  - YahooFetcher: fetch and cache Yahoo Finance price data (requires yfinance)
  - CSVFallback: load from local CSV when network unavailable
  - DataRegistry: central registry mapping series_id → fetcher + fallback path

Usage
-----
    from data_ingestion_template import DataRegistry

    registry = DataRegistry(
        cache_dir=".data_cache",
        fred_api_key=os.environ.get("FRED_API_KEY", ""),
    )
    registry.register_fred("PAYEMS", fallback="resources/payems.csv")
    registry.register_yahoo("^GSPC", fallback="resources/sp500_returns.csv")

    # Fetch (uses cache if available, falls back to CSV if network fails)
    payems = registry.get("PAYEMS")
    sp500 = registry.get("^GSPC")
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# Cache utilities
# ──────────────────────────────────────────────────────────────────────────────

class DiskCache:
    """
    Simple file-based cache for pd.Series and pd.DataFrame objects.
    Stores as Parquet if pyarrow is available, else CSV.
    Cache entries expire after `ttl_days` days.
    """

    def __init__(self, cache_dir: str = ".data_cache", ttl_days: int = 1):
        self.cache_dir = Path(cache_dir)
        self.ttl_days = ttl_days
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str, fmt: str = "csv") -> Path:
        safe = hashlib.md5(key.encode()).hexdigest()[:12]
        return self.cache_dir / f"{safe}.{fmt}"

    def get(self, key: str) -> Optional[pd.Series]:
        """Return cached series or None if expired/missing."""
        for fmt in ("parquet", "csv"):
            path = self._key_to_path(key, fmt)
            if path.exists():
                age_days = (
                    datetime.datetime.now()
                    - datetime.datetime.fromtimestamp(path.stat().st_mtime)
                ).total_seconds() / 86400
                if age_days > self.ttl_days:
                    return None
                try:
                    if fmt == "parquet":
                        df = pd.read_parquet(path)
                    else:
                        df = pd.read_csv(path, index_col=0, parse_dates=True)
                    return df.iloc[:, 0]
                except Exception:
                    return None
        return None

    def set(self, key: str, data: pd.Series) -> None:
        """Cache a series to disk."""
        df = data.to_frame("value")
        try:
            df.to_parquet(self._key_to_path(key, "parquet"))
        except Exception:
            df.to_csv(self._key_to_path(key, "csv"))

    def invalidate(self, key: str) -> None:
        """Remove a cache entry."""
        for fmt in ("parquet", "csv"):
            path = self._key_to_path(key, fmt)
            if path.exists():
                path.unlink()


# ──────────────────────────────────────────────────────────────────────────────
# FRED Fetcher
# ──────────────────────────────────────────────────────────────────────────────

class FREDFetcher:
    """
    Fetch FRED series via the FRED API.

    Parameters
    ----------
    api_key : str
        FRED API key. Obtain from https://fred.stlouisfed.org/docs/api/api_key.html
    cache : DiskCache, optional
    """

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
    RETRY_DELAYS = [0, 1, 2, 4]

    def __init__(self, api_key: str, cache: Optional[DiskCache] = None):
        self.api_key = api_key
        self.cache = cache

    def fetch(
        self,
        series_id: str,
        observation_start: Optional[str] = None,
        observation_end: Optional[str] = None,
        frequency: Optional[str] = None,
        units: str = "lin",
    ) -> pd.Series:
        """
        Fetch a FRED series.

        Parameters
        ----------
        series_id : str
            FRED series identifier (e.g. "PAYEMS", "GDPC1").
        observation_start : str, optional
            Start date in "YYYY-MM-DD" format.
        observation_end : str, optional
            End date in "YYYY-MM-DD" format.
        frequency : str, optional
            Frequency aggregation: "a" annual, "q" quarterly, "m" monthly,
            "w" weekly, "d" daily.
        units : str
            Data transformation: "lin" (levels), "chg" (change),
            "pch" (percent change), "pc1" (percent change from year ago).

        Returns
        -------
        pd.Series with DatetimeIndex.
        """
        cache_key = f"fred_{series_id}_{observation_start}_{units}"
        if self.cache is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        import requests

        params: Dict = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "units": units,
        }
        if observation_start:
            params["observation_start"] = observation_start
        if observation_end:
            params["observation_end"] = observation_end
        if frequency:
            params["frequency"] = frequency

        last_exc: Optional[Exception] = None
        for delay in self.RETRY_DELAYS:
            if delay:
                time.sleep(delay)
            try:
                resp = requests.get(self.BASE_URL, params=params, timeout=30)
                resp.raise_for_status()
                obs = resp.json().get("observations", [])
                values = {
                    o["date"]: float(o["value"])
                    for o in obs
                    if o["value"] != "."
                }
                series = pd.Series(values, name=series_id, dtype=float)
                series.index = pd.to_datetime(series.index)

                if self.cache is not None:
                    self.cache.set(cache_key, series)

                return series
            except Exception as exc:
                last_exc = exc

        raise RuntimeError(
            f"FRED fetch failed for {series_id} after {len(self.RETRY_DELAYS)} attempts. "
            f"Last error: {last_exc}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Yahoo Finance Fetcher
# ──────────────────────────────────────────────────────────────────────────────

class YahooFetcher:
    """
    Fetch price data from Yahoo Finance via yfinance.

    Returns daily log returns by default.

    Parameters
    ----------
    cache : DiskCache, optional
    """

    def __init__(self, cache: Optional[DiskCache] = None):
        self.cache = cache

    def fetch(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        return_type: str = "log_return",  # "log_return" | "price" | "adj_close"
    ) -> pd.Series:
        """
        Fetch Yahoo Finance data.

        Parameters
        ----------
        ticker : str
            Yahoo Finance ticker (e.g. "^GSPC", "CL=F", "GC=F").
        start : str, optional "YYYY-MM-DD"
        end : str, optional "YYYY-MM-DD"
        return_type : str
            "log_return" : daily log return
            "price"      : adjusted close price
            "adj_close"  : same as "price"

        Returns
        -------
        pd.Series with DatetimeIndex.
        """
        cache_key = f"yahoo_{ticker}_{start}_{return_type}"
        if self.cache is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance is not installed. Install with: pip install yfinance"
            )

        data = yf.download(
            ticker,
            start=start or "2000-01-01",
            end=end,
            progress=False,
            auto_adjust=True,
        )

        if data.empty:
            raise ValueError(f"No data returned for {ticker}")

        if return_type == "log_return":
            series = np.log(data["Close"] / data["Close"].shift(1)).dropna()
            series.name = f"{ticker}_log_ret"
        else:
            series = data["Close"]
            series.name = ticker

        series = series.squeeze()
        if self.cache is not None:
            self.cache.set(cache_key, series)

        return series


# ──────────────────────────────────────────────────────────────────────────────
# CSV Fallback loader
# ──────────────────────────────────────────────────────────────────────────────

class CSVFallback:
    """
    Load a time series from a local CSV file.

    Expected CSV format:
        date,value
        2020-01-01,1.23
        2020-02-01,1.45
        ...

    Or multi-column:
        date,col1,col2,...
        — in this case, `col` specifies which column to use.
    """

    def __init__(self, path: str, date_col: str = "date", value_col: str = "value"):
        self.path = path
        self.date_col = date_col
        self.value_col = value_col

    def load(self) -> pd.Series:
        df = pd.read_csv(self.path, parse_dates=[self.date_col])
        df = df.set_index(self.date_col)
        series = df[self.value_col] if self.value_col in df.columns else df.iloc[:, 0]
        series.index = pd.to_datetime(series.index)
        return series.sort_index().dropna()


# ──────────────────────────────────────────────────────────────────────────────
# Data Registry
# ──────────────────────────────────────────────────────────────────────────────

class DataRegistry:
    """
    Central registry mapping series identifiers to fetchers and CSV fallbacks.

    Fetch logic:
      1. Check disk cache (if available)
      2. Try live data source (FRED or Yahoo)
      3. Fall back to local CSV if network fails or key not set

    Usage
    -----
        registry = DataRegistry(
            cache_dir=".data_cache",
            fred_api_key=os.environ.get("FRED_API_KEY", ""),
        )
        registry.register_fred("PAYEMS", fallback="resources/payems.csv")
        registry.register_yahoo("^GSPC", fallback="resources/sp500_returns.csv")
        payems = registry.get("PAYEMS", start="2010-01-01")
    """

    def __init__(
        self,
        cache_dir: str = ".data_cache",
        fred_api_key: str = "",
        cache_ttl_days: int = 1,
    ):
        self.cache = DiskCache(cache_dir, ttl_days=cache_ttl_days)
        self.fred = FREDFetcher(fred_api_key, cache=self.cache) if fred_api_key else None
        self.yahoo = YahooFetcher(cache=self.cache)
        self._entries: Dict[str, Dict] = {}

    def register_fred(
        self,
        series_id: str,
        fallback: Optional[str] = None,
        units: str = "lin",
    ) -> "DataRegistry":
        """Register a FRED series with optional CSV fallback path."""
        self._entries[series_id] = {
            "source": "fred",
            "fallback": fallback,
            "units": units,
        }
        return self

    def register_yahoo(
        self,
        ticker: str,
        fallback: Optional[str] = None,
        return_type: str = "log_return",
    ) -> "DataRegistry":
        """Register a Yahoo Finance ticker with optional CSV fallback."""
        self._entries[ticker] = {
            "source": "yahoo",
            "fallback": fallback,
            "return_type": return_type,
        }
        return self

    def register_csv(self, series_id: str, path: str) -> "DataRegistry":
        """Register a CSV-only series (no live source)."""
        self._entries[series_id] = {"source": "csv", "fallback": path}
        return self

    def get(
        self,
        series_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.Series:
        """
        Retrieve a series by ID.

        Tries live source first, falls back to CSV on any failure.
        Raises ValueError if neither succeeds.
        """
        if series_id not in self._entries:
            raise KeyError(
                f"Series '{series_id}' not registered. "
                f"Call register_fred(), register_yahoo(), or register_csv() first."
            )

        entry = self._entries[series_id]
        source = entry["source"]
        fallback_path = entry.get("fallback")

        # Try live source
        try:
            if source == "fred":
                if self.fred is None:
                    raise RuntimeError("FRED API key not provided")
                return self.fred.fetch(
                    series_id, observation_start=start, observation_end=end,
                    units=entry.get("units", "lin"),
                )
            elif source == "yahoo":
                return self.yahoo.fetch(
                    series_id, start=start, end=end,
                    return_type=entry.get("return_type", "log_return"),
                )
            elif source == "csv" and fallback_path:
                return CSVFallback(fallback_path).load()
        except Exception as exc:
            warnings.warn(
                f"Live fetch failed for '{series_id}': {exc}. "
                f"Trying CSV fallback.",
                stacklevel=2,
            )

        # Fallback to CSV
        if fallback_path and os.path.exists(fallback_path):
            series = CSVFallback(fallback_path).load()
            warnings.warn(
                f"Using CSV fallback for '{series_id}': {fallback_path}",
                stacklevel=2,
            )
            return series

        raise ValueError(
            f"Could not retrieve series '{series_id}': "
            f"live fetch failed and no valid CSV fallback found "
            f"(path='{fallback_path}')"
        )

    def get_all(
        self, start: Optional[str] = None, end: Optional[str] = None
    ) -> Dict[str, pd.Series]:
        """Fetch all registered series and return as a dict."""
        return {sid: self.get(sid, start=start, end=end) for sid in self._entries}


# ──────────────────────────────────────────────────────────────────────────────
# Frequency conversion utilities
# ──────────────────────────────────────────────────────────────────────────────

def daily_to_monthly(
    daily: pd.Series,
    method: str = "last",
) -> pd.Series:
    """
    Aggregate daily series to monthly.

    Parameters
    ----------
    daily : pd.Series (DatetimeIndex, daily frequency)
    method : "last" | "mean" | "sum" | "first"

    Returns
    -------
    pd.Series with month-start DatetimeIndex.
    """
    agg_fn = {"last": "last", "mean": "mean", "sum": "sum", "first": "first"}
    if method not in agg_fn:
        raise ValueError(f"Unknown method '{method}'")
    return daily.resample("MS").agg(agg_fn[method])


def monthly_to_quarterly(
    monthly: pd.Series,
    method: str = "mean",
) -> pd.Series:
    """
    Aggregate monthly series to quarterly.

    Parameters
    ----------
    monthly : pd.Series (DatetimeIndex, monthly frequency)
    method : "mean" | "last" | "sum"

    Returns
    -------
    pd.Series with quarter-start DatetimeIndex.
    """
    agg_fn = {"mean": "mean", "last": "last", "sum": "sum"}
    if method not in agg_fn:
        raise ValueError(f"Unknown method '{method}'")
    return monthly.resample("QS").agg(agg_fn[method])


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns from price series."""
    returns = np.log(prices / prices.shift(1)).dropna()
    returns.name = f"{prices.name}_log_ret" if prices.name else "log_ret"
    return returns


def realised_volatility(
    daily_returns: pd.Series,
    freq: str = "MS",
) -> pd.Series:
    """
    Compute realised volatility by aggregating squared daily returns.

    RV_m = sum_{d in m} r_d^2

    Returns annualised standard deviation (multiply by sqrt(252) / sqrt(avg_days_per_period)).
    """
    rv = (daily_returns ** 2).resample(freq).sum()
    rv.name = "RV"
    return rv


# ──────────────────────────────────────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("data_ingestion_template.py loaded successfully.")
    print()
    print("Example (with FRED API key):")
    print("  registry = DataRegistry(")
    print('      cache_dir=".data_cache",')
    print('      fred_api_key=os.environ["FRED_API_KEY"],')
    print("  )")
    print('  registry.register_fred("PAYEMS", fallback="resources/payems.csv")')
    print('  payems = registry.get("PAYEMS", start="2010-01-01")')
    print()
    print("Example (CSV fallback only, no API key required):")
    print("  registry = DataRegistry()")
    print('  registry.register_csv("SP500", "resources/sp500_returns.csv")')
    print('  sp500 = registry.get("SP500")')
    print()

    # Demonstrate CSVFallback on the module 06 resource file
    fallback_path = os.path.join(
        os.path.dirname(__file__),
        "..", "modules", "module_06_financial_applications",
        "resources", "sp500_returns.csv"
    )
    if os.path.exists(fallback_path):
        series = CSVFallback(fallback_path, value_col="return").load()
        print(f"CSV fallback loaded: {len(series)} rows, {series.index[0].date()} to {series.index[-1].date()}")
    else:
        print(f"CSV fallback not found at: {fallback_path}")
        print("(Run from the course root or adjust the path)")
