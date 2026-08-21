"""Canonical table builders (SPEC-002 §2) + build_all CLI.

Every builder is causal: each row uses only data at or before its date
(REQ-2.1, verified by truncation-equivalence tests).

Usage:
    python -m rlbot.data.build --download   # pull bars + VIX, then build
    python -m rlbot.data.build              # build from cached snapshots
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from rlbot.config import RlbotConfig
from rlbot.data import sources
from rlbot.features.regime import (
    classify_regime_series,
    expanding_drawdown,
    realized_vol,
    rolling_percentile,
)
from rlbot.features.technicals_series import build_feature_frame
from rlbot.features.valuation import classify_valuation_series
from rlbot.features.vol_comp import classify_vol_comp_series
from rlbot.vendor.technicals import sma

_FV_FILE_RE = re.compile(r"fair_value_(\d{4}-\d{2}-\d{2})\.csv$")


def build_market(spy_df: pd.DataFrame, vix: pd.Series, cfg: RlbotConfig) -> pd.DataFrame:
    """SPEC-002 §2 `market` table from SPY bars + VIX closes."""
    close = spy_df["Close"]
    idx = close.index.tz_localize(None) if close.index.tz is not None else close.index
    close = pd.Series(close.values, index=idx, name="spy_close")

    out = pd.DataFrame(index=close.index)
    out["spy_close"] = close
    out["spy_ret_1d"] = close.pct_change(1)
    out["spy_sma50"] = sma(close, 50)
    out["spy_sma100"] = sma(close, 100)
    out["spy_sma200"] = sma(close, 200)
    out["spy_sma50_slope"] = out["spy_sma50"] - out["spy_sma50"].shift(10)
    out["spy_drawdown"] = expanding_drawdown(close)
    out["spy_realized_vol_20"] = realized_vol(close, cfg.data.realized_vol_market_window)

    vix = vix.copy()
    vix.index = vix.index.tz_localize(None) if vix.index.tz is not None else vix.index
    out["vix_close"] = vix.reindex(out.index).ffill(limit=3)
    out["vix_pct_5y"] = rolling_percentile(
        out["vix_close"], cfg.data.vix_percentile_window, cfg.data.vix_percentile_min_obs
    )
    out["vrp"] = out["vix_close"] - 100.0 * out["spy_realized_vol_20"]

    out["market_regime"] = classify_regime_series(
        out["spy_close"], out["spy_sma200"], out["spy_drawdown"], out["vix_pct_5y"], cfg.regime
    )
    out["vol_compensation"] = classify_vol_comp_series(out["vrp"], out["vix_pct_5y"], cfg.vol_comp)
    out.index.name = "date"
    return out


def build_underlying(bars: dict, cfg: RlbotConfig) -> pd.DataFrame:
    """SPEC-002 §2 `underlying` table: per-day features for every ticker."""
    frames = []
    for ticker, df in sorted(bars.items()):
        feat = build_feature_frame(df, rv_window=cfg.data.realized_vol_ticker_window)
        feat.index = feat.index.tz_localize(None) if feat.index.tz is not None else feat.index
        feat.insert(0, "ticker", ticker)
        frames.append(feat)
    out = pd.concat(frames)
    out.index.name = "date"
    return out


def build_valuation(snapshot_dir: Path, cfg: RlbotConfig) -> pd.DataFrame:
    """SPEC-002 §3.4: ingest the sibling's dated fair_value_*.csv snapshots.

    Historical era has no FV series (DATA-GAP-3): consumers treat missing
    (date, ticker) rows as FAIR via classify_valuation_series(NaN).
    """
    rows = []
    if snapshot_dir.exists():
        for path in sorted(snapshot_dir.iterdir()):
            m = _FV_FILE_RE.search(path.name)
            if not m:
                continue
            snap = pd.read_csv(path)
            snap["date"] = pd.to_datetime(m.group(1))
            rows.append(snap)
    if not rows:
        return pd.DataFrame(
            columns=["date", "ticker", "fv_buy", "fv_sell", "fmp_median",
                     "source", "confidence", "fv_dist", "valuation_state"]
        ).set_index(["date", "ticker"])

    df = pd.concat(rows, ignore_index=True)
    out = pd.DataFrame({
        "date": df["date"],
        "ticker": df["Ticker"],
        "fv_buy": pd.to_numeric(df["FV_Buy"], errors="coerce"),
        "fv_sell": pd.to_numeric(df["FV_Sell"], errors="coerce"),
        "fmp_median": pd.to_numeric(df.get("FMP_Median"), errors="coerce"),
        "source": df.get("Source"),
        "confidence": df.get("Confidence"),
    })
    price = pd.to_numeric(df["Price"], errors="coerce")
    out["fv_dist"] = (price - out["fv_buy"]) / out["fv_buy"]
    out["valuation_state"] = classify_valuation_series(out["fv_dist"], cfg.valuation)
    return out.set_index(["date", "ticker"]).sort_index()


def build_all(cfg: RlbotConfig, download: bool = False) -> dict:
    """Materialize every canonical table to parquet (REQ-2.5). Returns paths."""
    data = cfg.data
    if download:
        sources.download_bars([cfg.market_ticker], data.bars_path, years=data.spy_years)
        sources.download_bars(cfg.tickers, data.bars_path, years=data.ticker_years)
        sources.fetch_vix_fred(data.external_path / "vixcls.csv")

    spy_df = sources.load_bars(cfg.market_ticker, data.bars_path)
    vix = sources.load_vix(data.external_path)
    bars = {t: sources.load_bars(t, data.bars_path) for t in cfg.tickers}

    market = build_market(spy_df, vix, cfg)
    underlying = build_underlying(bars, cfg)
    valuation = build_valuation(data.fair_value_snapshot_dir, cfg)

    data.canonical_path.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, frame in [("market", market), ("underlying", underlying), ("valuation", valuation)]:
        path = data.canonical_path / f"{name}.parquet"
        frame.to_parquet(path)
        paths[name] = path
        print(f"wrote {path}  rows={len(frame)}")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="pull bars + VIX first")
    args = parser.parse_args()
    build_all(RlbotConfig(), download=args.download)
