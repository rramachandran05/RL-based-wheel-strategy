"""Track B — valuation proxy from Alpha Vantage quarterly EPS (SPEC-007 §3).

Causal by construction: eps_ttm(t) sums the last 4 quarters whose
``reportedDate`` is on or before t (never fiscal period end). pe_pct is the
right-inclusive rolling 5-year percentile of price / eps_ttm. Mapping:
pe_pct < 0.20 → ATTRACTIVE, > 0.80 → EXPENSIVE, else FAIR; negative or
missing eps_ttm → NaN pe (→ FAIR downstream).

Run:  python -m rlbot.data.eps_proxy          # fetch snapshots + build table
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from rlbot.config import RlbotConfig
from rlbot.data.env import get_key
from rlbot.data.sources import DataUnavailable, load_bars
from rlbot.features.regime import rolling_percentile

AV_URL = "https://www.alphavantage.co/query"
REQUEST_DELAY_S = 13.0          # free tier: 5 req/min
PE_LOW, PE_HIGH = 0.20, 0.80
PCT_WINDOW, PCT_MIN_OBS = 1260, 252


def fetch_eps_snapshots(tickers: list, dest: Path) -> None:
    api_key = get_key("ALPHA_API_KEY")
    if not api_key:
        raise DataUnavailable("ALPHA_API_KEY not found")
    dest.mkdir(parents=True, exist_ok=True)
    for i, ticker in enumerate(tickers):
        out = dest / f"{ticker}.csv"
        if out.exists():
            print(f"  {ticker}: snapshot exists, skipping")
            continue
        if i:
            time.sleep(REQUEST_DELAY_S)
        resp = requests.get(AV_URL, params={"function": "EARNINGS", "symbol": ticker,
                                            "apikey": api_key}, timeout=30)
        rows = resp.json().get("quarterlyEarnings", [])
        if not rows:
            raise DataUnavailable(f"AV EARNINGS empty for {ticker}")
        df = pd.DataFrame(rows)[["fiscalDateEnding", "reportedDate", "reportedEPS"]]
        df.to_csv(out, index=False)
        print(f"  {ticker}: {len(df)} quarters")


def load_eps(ticker: str, snap_dir: Path) -> pd.DataFrame:
    path = snap_dir / f"{ticker}.csv"
    if not path.exists():
        raise DataUnavailable(f"{path} missing; run python -m rlbot.data.eps_proxy")
    df = pd.read_csv(path)
    df["reportedDate"] = pd.to_datetime(df["reportedDate"])
    df["reportedEPS"] = pd.to_numeric(df["reportedEPS"], errors="coerce")
    return df.dropna(subset=["reportedDate", "reportedEPS"]) \
             .sort_values("reportedDate").reset_index(drop=True)


def eps_ttm_series(eps: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """Trailing-4-reported-quarters EPS, available as of each date (REQ-7.3)."""
    ttm = eps["reportedEPS"].rolling(4).sum()
    by_report = pd.Series(ttm.values, index=eps["reportedDate"].values).dropna()
    by_report = by_report[~by_report.index.duplicated(keep="last")].sort_index()
    return by_report.reindex(dates, method="ffill")


def build_proxy_for_ticker(ticker: str, close: pd.Series, snap_dir: Path) -> pd.DataFrame:
    eps = load_eps(ticker, snap_dir)
    idx = close.index
    ttm = eps_ttm_series(eps, idx)
    pe = close / ttm
    pe[ttm <= 0] = np.nan
    pe_pct = rolling_percentile(pe, PCT_WINDOW, PCT_MIN_OBS)
    val = pd.Series(np.select([pe_pct < PE_LOW, pe_pct > PE_HIGH], [0, 2], default=1),
                    index=idx, dtype="Int8")
    val[pe_pct.isna()] = pd.NA
    return pd.DataFrame({"ticker": ticker, "pe": pe, "pe_pct": pe_pct,
                         "valuation_state_proxy": val}, index=idx)


def build_proxy_table(cfg: RlbotConfig) -> Path:
    snap_dir = cfg.data.external_path / "eps"
    frames = []
    for t in cfg.tickers:
        bars = load_bars(t, cfg.data.bars_path)
        close = bars["Close"]
        close.index = close.index.tz_localize(None)
        frames.append(build_proxy_for_ticker(t, close, snap_dir))
        cov = frames[-1]["valuation_state_proxy"].notna().mean()
        print(f"  {t}: proxy coverage {cov:.0%}")
    out = pd.concat(frames)
    out.index.name = "date"
    path = cfg.data.canonical_path / "valuation_proxy.parquet"
    out.to_parquet(path)
    print(f"wrote {path}  rows={len(out)}")
    return path


if __name__ == "__main__":
    cfg = RlbotConfig()
    print("--- fetching EPS snapshots (Alpha Vantage) ---")
    fetch_eps_snapshots(cfg.tickers, cfg.data.external_path / "eps")
    print("--- building proxy table ---")
    build_proxy_table(cfg)
