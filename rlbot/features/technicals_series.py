"""Per-day technical feature series (SPEC-002 §3.2).

The vendored ``technicals`` module computes latest-snapshot context only; RL
training needs full per-day columns. Indicator functions are reused directly
(they already return Series). The structure/momentum classifiers are
re-expressed vectorized here, with golden tests (REQ-2.3) asserting exact
agreement with the vendored scalar versions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rlbot.features.regime import expanding_drawdown, realized_vol
from rlbot.state.enums import MOMENTUM_TO_BUCKET, STRUCTURE_TO_TREND
from rlbot.vendor.technicals import adx, atr, rsi, sma

STRUCTURE_LABELS = ["Bull Trend", "Recovery", "Pullback in Uptrend", "Base", "Breakdown"]
MOMENTUM_LABELS = ["Overextended", "Extended", "Building", "Weakening", "Neutral", "Mixed"]


def classify_structure_vec(
    price: pd.Series, sma50: pd.Series, sma200: pd.Series, atr20: pd.Series
) -> pd.Series:
    """Vectorized twin of vendored ``classify_structure`` (same rule order)."""
    conditions = [
        (price > sma50) & (sma50 > sma200),
        (price > sma50) & (sma50 <= sma200),
        (price < sma50) & (price >= sma200),
        (price - sma50).abs() <= atr20,
    ]
    labels = STRUCTURE_LABELS[:4]
    out = np.select(conditions, labels, default="Breakdown")
    return pd.Series(out, index=price.index, dtype="object")


def classify_momentum_vec(
    rsi14: pd.Series, adx14: pd.Series, di_plus: pd.Series, di_minus: pd.Series
) -> pd.Series:
    """Vectorized twin of vendored ``classify_momentum`` (same rule order)."""
    conditions = [
        rsi14 > 75,
        (rsi14 >= 68) & (adx14 >= 25),
        (rsi14 >= 60) & (adx14 >= 20) & (di_plus > di_minus),
        (rsi14 < 45) & (di_minus > di_plus),
        (rsi14 >= 45) & (rsi14 < 55) & (adx14 < 20),
    ]
    labels = MOMENTUM_LABELS[:5]
    out = np.select(conditions, labels, default="Mixed")
    return pd.Series(out, index=rsi14.index, dtype="object")


def build_feature_frame(df: pd.DataFrame, rv_window: int = 30) -> pd.DataFrame:
    """OHLCV frame (Open/High/Low/Close/Volume, DatetimeIndex) → per-day features.

    Columns match the ``underlying`` canonical table (SPEC-002 §2), minus
    the ticker key which the table builder adds.
    """
    close = df["Close"]
    out = pd.DataFrame(index=df.index)
    out["open"], out["high"], out["low"], out["close"], out["volume"] = (
        df["Open"], df["High"], df["Low"], close, df["Volume"],
    )
    out["ret_1d"] = close.pct_change(1)
    out["ret_5d"] = close.pct_change(5)
    out["ret_20d"] = close.pct_change(20)
    out["drawdown"] = expanding_drawdown(close)
    out["sma50"] = sma(close, 50)
    out["sma100"] = sma(close, 100)
    out["sma200"] = sma(close, 200)
    out["rsi14"] = rsi(close, 14)
    adx14, di_plus, di_minus = adx(df, 14)
    out["adx14"], out["di_plus"], out["di_minus"] = adx14, di_plus, di_minus
    out["atr20"] = atr(df, 20)
    out[f"realized_vol_{rv_window}"] = realized_vol(close, rv_window)

    out["structure"] = classify_structure_vec(close, out["sma50"], out["sma200"], out["atr20"])
    out["momentum"] = classify_momentum_vec(out["rsi14"], out["adx14"], out["di_plus"], out["di_minus"])
    out["trend_bucket"] = out["structure"].map(lambda s: int(STRUCTURE_TO_TREND[s]))
    out["momentum_bucket"] = out["momentum"].map(lambda s: int(MOMENTUM_TO_BUCKET[s]))

    # Classifier outputs are meaningless during indicator warmup; blank them.
    warm = out["sma200"].notna() & out["atr20"].notna() & out["rsi14"].notna() & out["adx14"].notna()
    for col in ("structure", "momentum"):
        out.loc[~warm, col] = None
    for col in ("trend_bucket", "momentum_bucket"):
        out[col] = out[col].astype("Int8")
        out.loc[~warm, col] = pd.NA
    return out
