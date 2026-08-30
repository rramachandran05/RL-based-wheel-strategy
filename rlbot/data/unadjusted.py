"""Unadjusted closes + adjustment ratios (SPEC-002 §3.7 amendment 2026-08-22).

Historical option chains quote RAW (unadjusted) strikes and premiums, while
every bar/frame in this project is split+dividend adjusted (Tiingo adj*).
This module caches, per ticker: the raw close and adj_ratio = adjClose/close.

Uses: (a) the chain ingester filters strikes against the RAW close;
(b) HistoricalChainPremiumSource multiplies raw strikes/premiums by the
snapshot day's adj_ratio so the simulator sees everything in adjusted space
(per-day moneyness is preserved exactly; cross-day drift from dividend
adjustments is <1% per quarter and handled by nearest-strike matching).

Run:  python -m rlbot.data.unadjusted
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from rlbot.config import RlbotConfig
from rlbot.data.env import get_key
from rlbot.data.sources import DataUnavailable


def fetch_unadjusted(ticker: str, dest: Path, years: int = 15,
                     timeout: int = 30) -> pd.DataFrame:
    """AV premium primary (2026-08-30): one daily-adjusted call refreshes both
    caches; falls through to the original Tiingo implementation on failure."""
    try:
        from rlbot.data.av_bars import download_stock_data_av
        download_stock_data_av(ticker, dest.parent / "bars", dest, years)
        return load_unadjusted(ticker, dest)
    except DataUnavailable:
        pass
    return _fetch_unadjusted_tiingo(ticker, dest, years, timeout)


def _fetch_unadjusted_tiingo(ticker: str, dest: Path, years: int = 15,
                             timeout: int = 30) -> pd.DataFrame:
    api_key = get_key("TIINGO_API_KEY")
    if not api_key:
        raise DataUnavailable("TIINGO_API_KEY not found")
    start = (pd.Timestamp.now() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    r = requests.get(url, params={"startDate": start, "token": api_key,
                                  "columns": "close,adjClose"}, timeout=timeout)
    if r.status_code != 200:
        raise DataUnavailable(f"Tiingo unadjusted fetch failed for {ticker}: {r.status_code}")
    df = pd.DataFrame(r.json())
    if df.empty or "close" not in df.columns:
        raise DataUnavailable(f"Tiingo unadjusted fetch empty for {ticker}")
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    out = df.set_index("date")[["close", "adjClose"]].rename(
        columns={"close": "close_unadj", "adjClose": "close_adj"})
    out["adj_ratio"] = out["close_adj"] / out["close_unadj"]
    dest.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest / f"{ticker}.csv", index_label="date")
    return out


def load_unadjusted(ticker: str, src: Path) -> pd.DataFrame:
    path = src / f"{ticker}.csv"
    if not path.exists():
        raise DataUnavailable(f"{path} missing; run python -m rlbot.data.unadjusted")
    return pd.read_csv(path, parse_dates=["date"], index_col="date")


def load_adj_ratio(ticker: str, cfg: RlbotConfig) -> pd.Series:
    return load_unadjusted(ticker, cfg.data.base_path / "bars_unadj")["adj_ratio"]


if __name__ == "__main__":
    cfg = RlbotConfig()
    dest = cfg.data.base_path / "bars_unadj"
    for t in cfg.tickers + ["TQQQ"]:
        df = fetch_unadjusted(t, dest)
        splits = (df["adj_ratio"].round(4).diff().abs() > 0.01).sum()
        print(f"  {t}: {len(df)} days, ratio {df['adj_ratio'].iloc[0]:.4f} -> "
              f"{df['adj_ratio'].iloc[-1]:.4f} ({splits} jumps)")
