"""Per-ticker IV features from the real chain store (SPEC-002 §3.5's
anticipated `vol_comp_source="ticker_iv"` — enabled 2026-08-23).

Daily ATM IV = median IV of 25-45 DTE puts with |delta| in [0.35, 0.65]
(near-the-money, the strikes the wheel actually trades around). From it:
  iv_pct_5y : rolling 5y right-inclusive percentile (causal)
  vrp_t     : 100·(atm_iv − realized_vol_30)  — vol points, same units as
              the market proxy so the SAME VolCompThresholds apply
  vol_comp  : the standard 3-level enum

Run:  python -m rlbot.features.ticker_iv     # build canonical ticker_iv.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rlbot.config import RlbotConfig
from rlbot.features.regime import rolling_percentile
from rlbot.features.vol_comp import classify_vol_comp_series

DTE_MIN, DTE_MAX = 25, 45
DELTA_LO, DELTA_HI = 0.35, 0.65
FFILL_LIMIT = 5


def atm_iv_series(chain_dir: Path) -> pd.Series:
    """Daily near-ATM put IV from a ticker's chain parquets."""
    parts = []
    for f in sorted(chain_dir.glob("*.parquet")):
        df = pd.read_parquet(f, columns=["snapshot_date", "cp", "dte", "delta", "iv"])
        df = df[(df.cp == "P") & df.dte.between(DTE_MIN, DTE_MAX)
                & df.delta.abs().between(DELTA_LO, DELTA_HI)
                & df.iv.notna() & (df.iv > 0.03)]   # excludes placeholder greeks
        if len(df):
            parts.append(df.groupby("snapshot_date")["iv"].median())
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts).sort_index()


def build_ticker_iv(ticker: str, cfg: RlbotConfig,
                    realized_vol_30: pd.Series) -> pd.DataFrame:
    iv = atm_iv_series(cfg.data.base_path / "chains" / ticker)
    idx = realized_vol_30.index
    out = pd.DataFrame(index=idx)
    out["atm_iv"] = iv.reindex(idx).ffill(limit=FFILL_LIMIT)
    out["iv_pct_5y"] = rolling_percentile(
        out["atm_iv"], cfg.data.vix_percentile_window, cfg.data.vix_percentile_min_obs)
    out["vrp_t"] = 100.0 * (out["atm_iv"] - realized_vol_30)
    out["vol_comp_ticker"] = classify_vol_comp_series(
        out["vrp_t"], out["iv_pct_5y"], cfg.vol_comp)
    out["ticker"] = ticker
    out.index.name = "date"
    return out


def build_all(cfg: RlbotConfig | None = None) -> Path:
    cfg = cfg or RlbotConfig()
    underlying = pd.read_parquet(cfg.data.canonical_path / "underlying.parquet")
    rv_col = f"realized_vol_{cfg.data.realized_vol_ticker_window}"
    frames = []
    for t in cfg.tickers + ["TQQQ"]:
        if not (cfg.data.base_path / "chains" / t).exists():
            continue
        rv = underlying[underlying.ticker == t][rv_col]
        tiv = build_ticker_iv(t, cfg, rv)
        cov = tiv["vol_comp_ticker"].notna().mean()
        print(f"  {t}: IV coverage {cov:.0%}, median ATM IV "
              f"{tiv['atm_iv'].median():.3f}")
        frames.append(tiv)
    out = pd.concat(frames)
    path = cfg.data.canonical_path / "ticker_iv.parquet"
    out.to_parquet(path)
    print(f"wrote {path}  rows={len(out)}")
    return path


if __name__ == "__main__":
    build_all()
