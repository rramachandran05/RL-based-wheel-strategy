"""Backfill historical option chains from Alpha Vantage HISTORICAL_OPTIONS
(premium endpoint) into per-(ticker, year) parquet files (SPEC-002 §3.7,
DATA-GAP-1 closure, 2026-08-22).

- One request per (ticker, trading day); days come from the cached bars index.
- Rows filtered on ingest: 1 <= DTE <= 75 and strike within +/-50% of that
  day's close (everything the selector/marker can ever use).
- Adaptive AIMD rate control: backs off on throttle messages, creeps faster
  after sustained success — works on any premium tier without configuration.
- Resumable: completed (ticker, year) parquet files are skipped on restart.

Run:  python -m rlbot.data.options_ingest [--start 2012-01-01] [--tickers ...]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

from rlbot.config import RlbotConfig
from rlbot.data.env import get_key
from rlbot.data.sources import DataUnavailable
from rlbot.data.unadjusted import load_unadjusted

AV_URL = "https://www.alphavantage.co/query"
DTE_MAX = 75
STRIKE_SPAN = 0.50
# SINGLE-THREADED, evenly paced (2026-08-22 lesson): AV runs a burst detector
# that flags sub-second clusters even far below the tier's per-minute cap, and
# requests.Session is not thread-safe — 8 workers produced timeout storms
# (250 requests / 3.6h). Sequential at ~0.45s ≈ 130 rpm is invisible to the
# burst detector and completes the whole backfill in ~5h.
BASE_INTERVAL = 0.45
MIN_INTERVAL = 0.30
MAX_INTERVAL = 5.0
SPEEDUP_EVERY = 500       # successes between gentle speedups

_session = requests.Session()


class RateLimiter:
    """Sequential even pacing + gentle AIMD adaptation."""

    def __init__(self, interval: float):
        self.interval = interval
        self.next_ok = time.monotonic()
        self.ok = 0
        self.retries = 0

    def acquire(self):
        delay = self.next_ok - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self.next_ok = time.monotonic() + self.interval

    def success(self):
        self.ok += 1
        if self.ok % SPEEDUP_EVERY == 0:
            self.interval = max(MIN_INTERVAL, self.interval * 0.95)

    def throttle(self):
        self.retries += 1
        self.interval = min(MAX_INTERVAL, self.interval * 1.20)
        time.sleep(2.0)

NUMERIC = ["strike", "last", "mark", "bid", "ask", "volume", "open_interest",
           "implied_volatility", "delta", "gamma", "theta", "vega", "rho"]


def fetch_day(ticker: str, date: str, api_key: str, timeout: int = 60) -> tuple:
    """(rows, throttled). rows=None on transport error."""
    try:
        r = _session.get(AV_URL, params={"function": "HISTORICAL_OPTIONS",
                                         "symbol": ticker, "date": date,
                                         "apikey": api_key}, timeout=timeout)
        j = r.json()
    except Exception:
        return None, False
    if "data" in j:
        return j["data"], False
    msg = str(j.get("message") or j.get("Information") or j.get("Note") or "").lower()
    throttled = any(k in msg for k in
                    ("frequency", "per minute", "premium", "sparingly",
                     "burst", "spread", "evenly"))
    return None, throttled


def filter_rows(rows: list, date: str, close: float) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["expiration"] = pd.to_datetime(df["expiration"], errors="coerce")
    snap = pd.Timestamp(date)
    df["dte"] = (df["expiration"] - snap).dt.days
    df = df[(df["dte"] >= 1) & (df["dte"] <= DTE_MAX)
            & (df["strike"] >= close * (1 - STRIKE_SPAN))
            & (df["strike"] <= close * (1 + STRIKE_SPAN))]
    if df.empty:
        return df
    out = pd.DataFrame({
        "snapshot_date": snap,
        "expiration": df["expiration"],
        "dte": df["dte"].astype("int32"),
        "cp": df["type"].str.upper().str[0],           # 'C' / 'P'
        "strike": df["strike"],
        "bid": df["bid"], "ask": df["ask"], "mark": df["mark"],
        "iv": df["implied_volatility"], "delta": df["delta"],
        "gamma": df["gamma"], "theta": df["theta"], "vega": df["vega"],
        "volume": df["volume"], "open_interest": df["open_interest"],
    })
    return out


def _fetch_one(ticker: str, d, close: float, api_key: str, limiter: RateLimiter):
    ds = d.strftime("%Y-%m-%d")
    for attempt in range(60):
        limiter.acquire()
        rows, throttled = fetch_day(ticker, ds, api_key)
        if rows is not None:
            limiter.success()
            return filter_rows(rows, ds, close)
        limiter.throttle() if throttled else time.sleep(5)
    raise RuntimeError(f"{ticker} {ds}: gave up after 60 attempts")


def ingest_ticker_year(ticker: str, year: int, dates: list, closes: dict,
                       api_key: str, out_path: Path, limiter: RateLimiter) -> bool:
    chunks = [_fetch_one(ticker, d, closes[d], api_key, limiter) for d in dates]
    frames = [c for c in chunks if not c.empty]
    empties = len(chunks) - len(frames)
    result = pd.concat(frames).sort_values(["snapshot_date", "cp", "expiration", "strike"]) \
        if frames else pd.DataFrame()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path)
    print(f"  {ticker} {year}: {len(result)} rows from {len(dates)} days "
          f"({empties} empty), interval {limiter.interval:.2f}s", flush=True)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2012-01-01")
    parser.add_argument("--tickers", nargs="*", default=None)
    args = parser.parse_args(argv)

    cfg = RlbotConfig()
    api_key = get_key("ALPHA_API_KEY")
    if not api_key:
        raise DataUnavailable("ALPHA_API_KEY not found")
    tickers = args.tickers or (cfg.tickers + ["TQQQ"])
    chains_dir = cfg.data.base_path / "chains"
    limiter = RateLimiter(BASE_INTERVAL)
    t0 = time.time()

    for ticker in tickers:
        # CRITICAL: chains quote RAW strikes; the ±50% strike filter must use
        # the UNADJUSTED close (adjusted closes are up to 200x off pre-split).
        unadj = load_unadjusted(ticker, cfg.data.base_path / "bars_unadj")
        closes = dict(zip(unadj.index, unadj["close_unadj"].values))
        dates = [d for d in unadj.index if d >= pd.Timestamp(args.start)]
        by_year: dict = {}
        for d in dates:
            by_year.setdefault(d.year, []).append(d)
        for year in sorted(by_year):
            out_path = chains_dir / ticker / f"{year}.parquet"
            if out_path.exists():
                continue
            ingest_ticker_year(ticker, year, by_year[year], closes,
                               api_key, out_path, limiter)
            elapsed = time.time() - t0
            print(f"[progress] {limiter.ok} requests, "
                  f"{elapsed/3600:.2f}h elapsed", flush=True)
    print("BACKFILL COMPLETE", flush=True)


if __name__ == "__main__":
    main()
