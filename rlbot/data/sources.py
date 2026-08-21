"""External data pulls and loaders (SPEC-002 §3).

Sources, per the 2026-08-21 constraint (Tiingo + Alpha Vantage APIs only,
plus free no-key public files):
  - Daily bars: Tiingo, via the vendored data_access module (TIINGO_API_KEY).
  - VIX history: FRED VIXCLS public CSV (no key).
  - CBOE PUT index: manual CSV download into data_local/external/put_index.csv.

REQ-2.4: failures raise DataUnavailable — never silent NaN propagation.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

from rlbot.data.env import get_key
from rlbot.vendor.data_access import download_stock_data, load_stock_data

FRED_VIXCLS_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"


class DataUnavailable(RuntimeError):
    """A required data source is missing or unreadable."""


def download_bars(tickers: list, dest: Path, years: int) -> dict:
    """Tiingo daily-adjusted OHLCV → CSV cache. Returns {ticker: DataFrame}."""
    api_key = get_key("TIINGO_API_KEY")
    if not api_key:
        raise DataUnavailable("TIINGO_API_KEY not found in env or .env files")
    dest.mkdir(parents=True, exist_ok=True)
    out = {}
    for ticker in tickers:
        df = download_stock_data(ticker, api_key, str(dest), years=years)
        if df is None:
            raise DataUnavailable(f"Tiingo download failed for {ticker}")
        out[ticker] = df
    return out


def load_bars(ticker: str, bars_path: Path) -> pd.DataFrame:
    df = load_stock_data(ticker, str(bars_path))
    if df is None or df.empty:
        raise DataUnavailable(
            f"No cached bars for {ticker} in {bars_path}; run build with --download"
        )
    return df


def fetch_vix_fred(dest: Path, timeout: int = 30) -> pd.Series:
    """Download the full VIXCLS history from FRED and cache it."""
    resp = requests.get(FRED_VIXCLS_URL, timeout=timeout)
    if resp.status_code != 200:
        raise DataUnavailable(f"FRED VIXCLS fetch failed: HTTP {resp.status_code}")
    series = _parse_vixcls(resp.text)
    dest.parent.mkdir(parents=True, exist_ok=True)
    series.to_frame("vix_close").to_csv(dest, index_label="date")
    return series


def _parse_vixcls(text: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = "observation_date" if "observation_date" in df.columns else "date"
    if date_col not in df.columns or "vixcls" not in df.columns:
        raise DataUnavailable(f"Unexpected VIXCLS format: columns={list(df.columns)}")
    df[date_col] = pd.to_datetime(df[date_col])
    series = pd.to_numeric(df["vixcls"], errors="coerce")
    series.index = df[date_col]
    series = series.dropna().sort_index()
    series.name = "vix_close"
    if series.empty:
        raise DataUnavailable("VIXCLS series parsed empty")
    return series


def load_vix(external_path: Path) -> pd.Series:
    path = external_path / "vixcls.csv"
    if not path.exists():
        raise DataUnavailable(f"{path} missing; run build with --download")
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return df["vix_close"].dropna().sort_index()


def load_put_index(external_path: Path) -> pd.Series:
    """CBOE PUT index closes — manual download (SPEC-002 §3.8).

    Expected file: data_local/external/put_index.csv with columns date,close.
    """
    path = external_path / "put_index.csv"
    if not path.exists():
        raise DataUnavailable(
            f"{path} missing. Download the PUT index history CSV from cboe.com "
            "and save it there with columns: date,close"
        )
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns or "close" not in df.columns:
        raise DataUnavailable("put_index.csv must have columns: date,close")
    series = pd.Series(
        pd.to_numeric(df["close"], errors="coerce").values,
        index=pd.to_datetime(df["date"]),
        name="put_index_close",
    )
    return series.dropna().sort_index()
