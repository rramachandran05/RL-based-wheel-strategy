"""Daily bars from Alpha Vantage premium (2026-08-30): primary bar source,
replacing Tiingo (which 429s on repeated full pulls; the AV premium tier has
per-minute pacing but no practical daily cap for ~25 tickers/day).

One TIME_SERIES_DAILY_ADJUSTED call per ticker yields BOTH caches:
  bars/<t>.csv        Date,Open,High,Low,Close,Volume  (split+div adjusted,
                      same schema the vendored Tiingo loader reads)
  bars_unadj/<t>.csv  date,close_unadj,close_adj,adj_ratio  (chain-strike
                      filtering + the historical source's adjustment ratio)

Adjustment math: AV gives raw OHLC + adjusted close + split coefficients.
  ratio(t)   = adjusted_close / raw_close          (splits + dividends)
  adj OHLC   = raw OHLC x ratio(t)                 (per-day, exact)
  adj volume = raw volume x prod(split coeffs strictly after t)  (splits only)
Tiingo remains the fallback when AV fails (sources.download_bars).
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from rlbot.data.env import get_key
from rlbot.data.sources import DataUnavailable

AV_URL = "https://www.alphavantage.co/query"
PACE_SECONDS = 0.6          # evenly spaced: AV's burst detector dislikes clusters
_session = requests.Session()


def fetch_daily_adjusted(ticker: str, api_key: str, timeout: int = 60) -> pd.DataFrame:
    """-> DataFrame indexed by date: raw o/h/l/c, adj_close, volume, split."""
    r = _session.get(AV_URL, params={
        "function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": ticker,
        "outputsize": "full", "apikey": api_key}, timeout=timeout)
    j = r.json()
    series = j.get("Time Series (Daily)")
    if not series:
        msg = str(j.get("Information") or j.get("Note") or j.get("Error Message")
                  or "empty payload")[:160]
        raise DataUnavailable(f"AV daily-adjusted failed for {ticker}: {msg}")
    return parse_daily_adjusted(series)


def parse_daily_adjusted(series: dict) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(series, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.rename(columns={
        "1. open": "open", "2. high": "high", "3. low": "low",
        "4. close": "close", "5. adjusted close": "adj_close",
        "6. volume": "volume", "8. split coefficient": "split",
    })[["open", "high", "low", "close", "adj_close", "volume", "split"]]
    return df.astype(float).sort_index()


def to_caches(df: pd.DataFrame, years: int) -> tuple:
    """(adjusted_bars, unadj) frames, truncated to the trailing `years`."""
    ratio = df["adj_close"] / df["close"]
    # split-only factor for volume: product of split coefficients AFTER t
    rev_cum = df["split"][::-1].cumprod()[::-1]
    vol_factor = rev_cum / df["split"]           # excludes t's own coefficient
    bars = pd.DataFrame({
        "Open": df["open"] * ratio,
        "High": df["high"] * ratio,
        "Low": df["low"] * ratio,
        "Close": df["adj_close"],
        "Volume": (df["volume"] * vol_factor).round().astype("int64"),
    })
    bars.index.name = "Date"
    unadj = pd.DataFrame({
        "close_unadj": df["close"],
        "close_adj": df["adj_close"],
        "adj_ratio": ratio,
    })
    unadj.index.name = "date"
    start = pd.Timestamp.now().normalize() - pd.DateOffset(years=years)
    return bars.loc[bars.index >= start], unadj.loc[unadj.index >= start]


def download_stock_data_av(ticker: str, bars_dir: Path, unadj_dir: Path,
                           years: int) -> pd.DataFrame:
    api_key = get_key("ALPHA_API_KEY")
    if not api_key:
        raise DataUnavailable("ALPHA_API_KEY not found")
    df = fetch_daily_adjusted(ticker, api_key)
    bars, unadj = to_caches(df, years)
    if bars.empty:
        raise DataUnavailable(f"AV returned no rows in window for {ticker}")
    bars_dir.mkdir(parents=True, exist_ok=True)
    unadj_dir.mkdir(parents=True, exist_ok=True)
    bars.to_csv(bars_dir / f"{ticker}.csv")
    unadj.to_csv(unadj_dir / f"{ticker}.csv")
    print(f"  saved {ticker} ({len(bars)} rows, AV)")
    time.sleep(PACE_SECONDS)
    return bars
