"""Ticker decision frame + Q-state encoding (SPEC-001 §3, REQ-1.2).

``build_ticker_frame`` joins the canonical tables into one per-ticker,
per-day frame holding everything the simulator, policies, and trajectory
logger need. ``encode_q_state`` is the single pure encoding function.
"""
from __future__ import annotations

import pandas as pd

from rlbot.config import RlbotConfig
from rlbot.features.valuation import classify_valuation_series
from rlbot.state.enums import ValuationState

FV_FFILL_LIMIT = 21  # business days a valuation snapshot stays fresh


def encode_q_state(regime, valuation_state, vol_comp) -> tuple | None:
    """(market_regime, valuation_state, vol_compensation) or None during warmup."""
    if pd.isna(regime) or pd.isna(vol_comp):
        return None
    val = int(ValuationState.FAIR) if pd.isna(valuation_state) else int(valuation_state)
    return (int(regime), val, int(vol_comp))


def build_ticker_frame(
    ticker: str,
    underlying: pd.DataFrame,   # canonical, all tickers
    market: pd.DataFrame,       # canonical
    valuation: pd.DataFrame,    # canonical (may be empty)
    cfg: RlbotConfig,
    valuation_proxy: pd.DataFrame | None = None,   # SPEC-007 Track B
) -> pd.DataFrame:
    u = underlying[underlying["ticker"] == ticker].drop(columns=["ticker"]).copy()
    rv_col = f"realized_vol_{cfg.data.realized_vol_ticker_window}"
    u = u.rename(columns={rv_col: "vol_proxy"})

    m = market[["market_regime", "vol_compensation", "vix_close", "vix_pct_5y", "vrp",
                "spy_close", "spy_drawdown"]]
    frame = u.join(m, how="left")

    # Valuation: ffill each snapshot up to FV_FFILL_LIMIT days, then recompute
    # fv_dist daily against the close; missing → NaN → FAIR at encode time.
    frame["fv_buy"] = float("nan")
    if not valuation.empty and ticker in valuation.index.get_level_values("ticker"):
        v = valuation.xs(ticker, level="ticker")["fv_buy"]
        frame["fv_buy"] = v.reindex(frame.index).ffill(limit=FV_FFILL_LIMIT)
    frame["fv_dist"] = (frame["close"] - frame["fv_buy"]) / frame["fv_buy"]
    frame["valuation_state"] = classify_valuation_series(frame["fv_dist"], cfg.valuation)
    frame.loc[frame["fv_dist"].isna(), "valuation_state"] = pd.NA

    # Track B: sheet-based FV wins where present; EPS-percentile proxy fills
    # the rest (SPEC-007 §3.1). Absent both -> NA -> FAIR at encode time.
    if valuation_proxy is not None and not valuation_proxy.empty:
        prox = valuation_proxy[valuation_proxy["ticker"] == ticker]
        if not prox.empty:
            p = prox["valuation_state_proxy"].reindex(frame.index)
            fill = frame["valuation_state"].isna() & p.notna()
            frame.loc[fill, "valuation_state"] = p[fill]
    return frame
