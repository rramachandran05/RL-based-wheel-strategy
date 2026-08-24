"""Keep the chain store current for the daily brief (SPEC-008 real-quote
upgrade, 2026-08-23): fetch the latest trading day's chain for every
assistant-universe ticker and merge it into that ticker's year parquet.
~16 requests/day, evenly paced. Also backfills bars_unadj for any universe
ticker that lacks it (adj_ratio is required by the historical source).
"""
from __future__ import annotations

import pandas as pd

from rlbot.config import RlbotConfig
from rlbot.data.env import get_key
from rlbot.data.options_ingest import RateLimiter, fetch_day, filter_rows
from rlbot.data.sources import DataUnavailable
from rlbot.data.unadjusted import fetch_unadjusted, load_unadjusted


def update_daily_chains(cfg: RlbotConfig, tickers: list | None = None) -> dict:
    """Returns {ticker: 'updated'|'current'|'no_data'|'error: ...'}."""
    api_key = get_key("ALPHA_API_KEY")
    if not api_key:
        raise DataUnavailable("ALPHA_API_KEY not found")
    tickers = tickers or cfg.assistant_universe
    unadj_dir = cfg.data.base_path / "bars_unadj"
    chains_dir = cfg.data.base_path / "chains"
    limiter = RateLimiter(0.6)
    status = {}
    for t in tickers:
        try:
            try:
                unadj = load_unadjusted(t, unadj_dir)
            except DataUnavailable:
                unadj = fetch_unadjusted(t, unadj_dir,
                                         years=cfg.data.ticker_years)
            # refresh if the cached unadj series is stale
            if (pd.Timestamp.now().normalize() - unadj.index.max()).days > 4:
                unadj = fetch_unadjusted(t, unadj_dir, years=cfg.data.ticker_years)
            last = unadj.index.max()
            year_file = chains_dir / t / f"{last.year}.parquet"
            existing = pd.read_parquet(year_file) if year_file.exists() else pd.DataFrame()
            if len(existing) and existing["snapshot_date"].max() >= last:
                status[t] = "current"
                continue
            limiter.acquire()
            rows, throttled = fetch_day(t, last.strftime("%Y-%m-%d"), api_key)
            if rows is None:
                status[t] = "error: fetch failed/throttled"
                continue
            chunk = filter_rows(rows, last.strftime("%Y-%m-%d"),
                                float(unadj.loc[last, "close_unadj"]))
            if chunk.empty:
                status[t] = "no_data"
                continue
            merged = pd.concat([existing, chunk]).drop_duplicates(
                subset=["snapshot_date", "cp", "expiration", "strike"])
            year_file.parent.mkdir(parents=True, exist_ok=True)
            merged.sort_values(["snapshot_date", "cp", "expiration", "strike"]) \
                  .to_parquet(year_file)
            status[t] = "updated"
        except Exception as e:                          # never kill the brief
            status[t] = f"error: {e}"
    return status


if __name__ == "__main__":
    for t, s in update_daily_chains(RlbotConfig()).items():
        print(f"  {t}: {s}")
