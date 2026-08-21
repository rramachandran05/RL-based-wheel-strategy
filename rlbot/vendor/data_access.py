# VENDORED from ../wheel-strategy/data_access.py on 2026-08-21 (source sha256: 7fdb000e60b0a9fe86ad0ef4eaa03e58b3d762bb913aa278e6bf8b633c4624c4)
# Do not edit — see SPEC-002 REQ-2.2. Changes belong in rlbot/, not here.
"""
data_access.py
==============
Data layer. Downloads adjusted OHLCV from Tiingo and loads it back.

Security note
-------------
The API key is read ONLY from the environment (TIINGO_API_KEY). There is no
hardcoded fallback. The previous notebook embedded a literal key -- that key
should be treated as compromised and rotated.

Portability note
----------------
No Google Colab imports. Paths default to the working directory and can be
overridden via environment variables.

Dependencies: numpy, pandas, requests.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests


def get_api_key() -> str:
    """Read the Tiingo key from the environment. Empty string if unset."""
    return os.getenv("TIINGO_API_KEY", "")


def default_data_path() -> str:
    base = os.getenv("WHEEL_BASE_PATH", os.getcwd())
    return os.path.join(base, "stockData")


def default_output_path() -> str:
    base = os.getenv("WHEEL_BASE_PATH", os.getcwd())
    return os.path.join(base, "wheel_outputs")


def download_stock_data(ticker: str, api_key: str, data_path: str,
                        years: int = 5) -> Optional[pd.DataFrame]:
    """Download adjusted OHLCV from Tiingo and cache it to CSV."""
    if not api_key:
        print(f"  skip {ticker}: TIINGO_API_KEY not set")
        return None

    start = (datetime.today() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    url = (f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
           f"?startDate={start}&token={api_key}")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"  download failed {ticker}: {exc}")
        return None

    if not isinstance(data, list) or not data:
        print(f"  no data for {ticker}")
        return None

    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={
        "adjOpen": "Open", "adjHigh": "High", "adjLow": "Low",
        "adjClose": "Close", "adjVolume": "Volume",
    }).set_index("Date")
    ohlcv = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    os.makedirs(data_path, exist_ok=True)
    ohlcv.to_csv(os.path.join(data_path, f"{ticker}.csv"))
    print(f"  saved {ticker} ({len(ohlcv)} rows)")
    return ohlcv


def load_stock_data(ticker: str, data_path: str) -> Optional[pd.DataFrame]:
    """Load and standardise cached OHLCV for one ticker."""
    path = os.path.join(data_path, f"{ticker}.csv")
    if not os.path.exists(path):
        print(f"  missing data file for {ticker}")
        return None

    for index_col in ("date", "Date"):
        try:
            df = pd.read_csv(path, index_col=index_col, parse_dates=True)
            break
        except (ValueError, KeyError):
            df = None
    if df is None:
        print(f"  could not parse {ticker}")
        return None

    df.columns = [c.title() for c in df.columns]
    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in df.columns:
            print(f"  {ticker} missing column {col}")
            return None
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=required).sort_index()


def is_data_stale(df: pd.DataFrame, max_age_days: int = 1) -> bool:
    """True if the most recent bar is older than max_age_days."""
    if df is None or df.empty:
        return True
    last = df.index.max()
    if last.tzinfo is not None:
        last = last.tz_localize(None)
    return (pd.Timestamp.today() - last).days > max_age_days


def download_all(tickers, api_key: str, data_path: str) -> None:
    print(f"Downloading {len(tickers)} tickers from Tiingo...")
    for t in tickers:
        download_stock_data(t, api_key, data_path)


# ----------------------------------------------------------------------
# Tests (offline -- no network)
# ----------------------------------------------------------------------

def _run_tests() -> None:
    import numpy as np
    import tempfile

    # staleness check
    fresh = pd.DataFrame(
        {"Open": [1], "High": [1], "Low": [1], "Close": [1], "Volume": [1]},
        index=[pd.Timestamp.today()],
    )
    assert is_data_stale(fresh) is False
    old = pd.DataFrame(
        {"Open": [1], "High": [1], "Low": [1], "Close": [1], "Volume": [1]},
        index=[pd.Timestamp.today() - pd.Timedelta(days=30)],
    )
    assert is_data_stale(old) is True
    assert is_data_stale(pd.DataFrame()) is True

    # load round-trip
    with tempfile.TemporaryDirectory() as d:
        idx = pd.bdate_range("2024-01-01", periods=10, name="date")
        sample = pd.DataFrame({
            "Open": np.arange(10.0), "High": np.arange(10.0) + 1,
            "Low": np.arange(10.0) - 1, "Close": np.arange(10.0),
            "Volume": np.arange(10) + 100,
        }, index=idx)
        sample.to_csv(os.path.join(d, "TEST.csv"))
        loaded = load_stock_data("TEST", d)
        assert loaded is not None and len(loaded) == 10
        assert list(loaded.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert load_stock_data("NOPE", d) is None

    # key reader
    assert isinstance(get_api_key(), str)

    print("data_access: all tests passed")


if __name__ == "__main__":
    _run_tests()
