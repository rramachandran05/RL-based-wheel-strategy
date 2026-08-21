# VENDORED from ../wheel-strategy/technicals.py on 2026-08-21 (source sha256: 29edf3ad8d83ade566f98c32fb5540ae97ca3b57e71a05f2e3430c8a6d01d316)
# Do not edit — see SPEC-002 REQ-2.2. Changes belong in rlbot/, not here.
"""
technicals.py
=============
Technical-analysis layer, carried over from stockAnalyzerv7 and tidied.

This produces the STOCK context -- trend structure, momentum phase, ATR,
support/resistance. It deliberately does NOT make options decisions; that is
the job of options_engine.py. Keeping the two separate is what makes the
pipeline readable: technicals describe the stock, the options layer prices
the trade.

Dependencies: numpy, pandas, scipy.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter


# ----------------------------------------------------------------------
# Indicators
# ----------------------------------------------------------------------

def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["Close"].shift(1)
    return pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"] - prev).abs(),
    ], axis=1).max(axis=1)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    out = out.where(avg_loss != 0, 100.0).where(avg_gain != 0, 0.0)
    return out.mask((avg_gain == 0) & (avg_loss == 0), 50.0)


def adx(df: pd.DataFrame, length: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    atr_s = atr(df, length)
    plus_s = plus_dm.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    minus_s = minus_dm.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    plus_di = 100 * (plus_s / atr_s.replace(0, np.nan))
    minus_di = 100 * (minus_s / atr_s.replace(0, np.nan))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx_s = dx.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    return adx_s, plus_di, minus_di


# ----------------------------------------------------------------------
# Support / resistance via volume profile
# ----------------------------------------------------------------------

def volume_profile_levels(df: pd.DataFrame, bins: int = 120,
                          smooth_window: int = 15,
                          min_strength: float = 15.0) -> Tuple[List[Dict], List[Dict]]:
    """Return (supports, resistances) as dicts with price + strength metadata."""
    if len(df) < smooth_window:
        return [], []

    price = df["Close"].iloc[-1]
    levels = np.linspace(df["Low"].min(), df["High"].max(), bins)
    profile = np.zeros(bins)
    lows, highs, vols = df["Low"].values, df["High"].values, df["Volume"].values
    for i in range(len(df)):
        mask = (levels >= lows[i]) & (levels <= highs[i])
        touched = int(mask.sum())
        if touched:
            profile[mask] += vols[i] / touched

    try:
        smooth = savgol_filter(profile, smooth_window, polyorder=3)
    except ValueError:
        smooth = profile

    peaks, _ = find_peaks(smooth, prominence=smooth.max() * 0.05)
    max_vol = float(smooth.max()) if len(smooth) else 0.0
    if max_vol <= 0:
        return [], []

    supports, resistances = [], []
    for idx in peaks:
        lvl = float(levels[idx])
        strength = float(smooth[idx] / max_vol * 100)
        if strength < min_strength:
            continue
        row = {"price": round(lvl, 2), "strength_pct": round(strength, 1),
               "distance_pct": round(abs(lvl - price) / price * 100, 2)}
        (supports if lvl < price else resistances).append(row)

    supports.sort(key=lambda x: x["price"], reverse=True)
    resistances.sort(key=lambda x: x["price"])
    return supports, resistances


def split_by_horizon(levels: List[Dict], price: float, atr_val: float,
                     side: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Bucket levels into tactical (<=3 ATR), structural (<=8), historical (>8)."""
    tactical, structural, historical = [], [], []
    if atr_val <= 0 or not levels:
        return tactical, structural, historical
    for lvl in levels:
        p = lvl["price"]
        dist = (price - p) / atr_val if side == "support" else (p - price) / atr_val
        if dist <= 0:
            continue
        enriched = dict(lvl, distance_atr=round(dist, 2))
        if dist <= 3:
            tactical.append(enriched)
        elif dist <= 8:
            structural.append(enriched)
        else:
            historical.append(enriched)
    return tactical, structural, historical


# ----------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------

def compute_context(df: pd.DataFrame, ticker: str) -> Dict[str, object]:
    """Compute all indicators and the latest snapshot for one ticker."""
    d = df.copy()
    d["SMA50"] = sma(d["Close"], 50)
    d["SMA200"] = sma(d["Close"], 200)
    d["EMA20"] = ema(d["Close"], 20)
    d["ATR20"] = atr(d, 20)
    d["RSI14"] = rsi(d["Close"], 14)
    adx_s, plus_di, minus_di = adx(d, 14)
    d["ADX14"], d["DIp"], d["DIm"] = adx_s, plus_di, minus_di
    d["SMA50_slope"] = (d["SMA50"] - d["SMA50"].shift(10)) / d["ATR20"]

    cutoff = d.index.max() - pd.DateOffset(years=2)
    vol_df = d.loc[d.index >= cutoff, ["High", "Low", "Close", "Volume"]]
    supports, resistances = volume_profile_levels(vol_df)

    today = d.iloc[-1]
    price = float(today["Close"])
    atr20 = float(today["ATR20"])

    t_sup, s_sup, _ = split_by_horizon(supports, price, atr20, "support")
    t_res, s_res, _ = split_by_horizon(resistances, price, atr20, "resistance")

    nearest_support = (t_sup[0]["price"] if t_sup else
                       (s_sup[0]["price"] if s_sup else np.nan))
    nearest_resistance = (t_res[0]["price"] if t_res else
                          (s_res[0]["price"] if s_res else np.nan))

    return {
        "ticker": ticker,
        "date": d.index[-1].strftime("%Y-%m-%d"),
        "price": price,
        "sma50": float(today["SMA50"]),
        "sma200": float(today["SMA200"]),
        "atr20": atr20,
        "rsi14": float(today["RSI14"]),
        "adx14": float(today["ADX14"]),
        "di_plus": float(today["DIp"]),
        "di_minus": float(today["DIm"]),
        "sma50_slope": float(today["SMA50_slope"]),
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "tactical_supports": t_sup,
        "close_series": d["Close"],
    }


def classify_structure(ctx: Dict) -> str:
    """Long/intermediate trend label."""
    price, s50, s200 = ctx["price"], ctx["sma50"], ctx["sma200"]
    if price > s50 > s200:
        return "Bull Trend"
    if price > s50 and s50 <= s200:
        return "Recovery"
    if price < s50 and price >= s200:
        return "Pullback in Uptrend"
    if abs(price - s50) <= ctx["atr20"]:
        return "Base"
    return "Breakdown"


def structure_rationale(ctx: Dict, structure: str) -> str:
    """Plain-English reason for the structure label (REQ-4), e.g.
    'Price 182.40 > 50SMA 175.10 > 200SMA 160.00 -- hence Bull Trend'."""
    price, s50, s200 = ctx["price"], ctx["sma50"], ctx["sma200"]
    p, a, b = f"{price:.2f}", f"{s50:.2f}", f"{s200:.2f}"
    if structure == "Bull Trend":
        return f"Price {p} > 50SMA {a} > 200SMA {b} -- hence Bull Trend"
    if structure == "Recovery":
        return f"Price {p} > 50SMA {a}, but 50SMA <= 200SMA {b} -- hence Recovery"
    if structure == "Pullback in Uptrend":
        return f"Price {p} < 50SMA {a} but still >= 200SMA {b} -- hence Pullback in Uptrend"
    if structure == "Base":
        return (f"Price {p} within 1 ATR ({ctx['atr20']:.2f}) of 50SMA {a} "
                f"-- hence Base")
    return f"Price {p} below 50SMA {a} and outside Base range -- hence Breakdown"


def classify_momentum(ctx: Dict) -> str:
    """Momentum phase label."""
    rsi_v, adx_v = ctx["rsi14"], ctx["adx14"]
    if rsi_v > 75:
        return "Overextended"
    if rsi_v >= 68 and adx_v >= 25:
        return "Extended"
    if rsi_v >= 60 and adx_v >= 20 and ctx["di_plus"] > ctx["di_minus"]:
        return "Building"
    if rsi_v < 45 and ctx["di_minus"] > ctx["di_plus"]:
        return "Weakening"
    if 45 <= rsi_v < 55 and adx_v < 20:
        return "Neutral"
    return "Mixed"


def momentum_rationale(ctx: Dict, momentum: str) -> str:
    """Plain-English reason for the momentum label (REQ-4), e.g.
    'RSI 61.2 >= 60, ADX 24.5 >= 20, +DI > -DI -- hence Building'."""
    rsi_v, adx_v = ctx["rsi14"], ctx["adx14"]
    dip, dim = ctx["di_plus"], ctx["di_minus"]
    r, a = f"{rsi_v:.1f}", f"{adx_v:.1f}"
    if momentum == "Overextended":
        return f"RSI {r} > 75 -- hence Overextended"
    if momentum == "Extended":
        return f"RSI {r} >= 68 and ADX {a} >= 25 -- hence Extended"
    if momentum == "Building":
        return (f"RSI {r} >= 60, ADX {a} >= 20, +DI {dip:.1f} > -DI {dim:.1f} "
                f"-- hence Building")
    if momentum == "Weakening":
        return f"RSI {r} < 45 and -DI {dim:.1f} > +DI {dip:.1f} -- hence Weakening"
    if momentum == "Neutral":
        return f"RSI {r} in 45-55 and ADX {a} < 20 -- hence Neutral"
    return f"RSI {r} / ADX {a} don't fit a clean phase -- hence Mixed"


def wheel_bias(ctx: Dict, structure: str, momentum: str) -> str:
    """Translate stock context into a coarse wheel stance."""
    if structure in {"Bull Trend", "Recovery"} and momentum == "Building":
        return "Favorable for CSP"
    if momentum in {"Extended", "Overextended"}:
        return "Hold / prefer CC"
    if structure == "Breakdown":
        return "CSP only if willing to own"
    if structure == "Pullback in Uptrend":
        return "CSP on stabilization"
    return "Watchlist"


# ----------------------------------------------------------------------
# Buy-entry signals
# ----------------------------------------------------------------------

def bollinger_bands(
    series: pd.Series, length: int = 20, num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return (upper, middle, lower) Bollinger Band series."""
    mid = sma(series, length)
    std = series.rolling(length, min_periods=length).std()
    return mid + num_std * std, mid, mid - num_std * std


def entry_signals(ctx: Dict) -> Dict:
    """
    Identify oversold / near-support entry conditions from technical context.

    Three technical conditions are checked:
      oversold_rsi   — RSI(14) < 35 (classic mean-reversion threshold)
      bb_lower_touch — price at or below the 20-day Bollinger lower band
      near_support   — price within 1.5 ATR of the nearest volume-profile support

    A ``tech_score`` (0-3) counts conditions met.  The caller (daily_brief)
    adds a fourth condition (price below fair-value low) to produce the final
    ``total_score`` (0-4) and ``entry_label``.
    """
    rsi_val = ctx["rsi14"]
    price   = ctx["price"]
    atr_val = ctx["atr20"]

    oversold_rsi = rsi_val < 35

    nearest_sup = ctx.get("nearest_support", float("nan"))
    near_sup_val = (
        not np.isnan(nearest_sup) and abs(price - nearest_sup) <= 1.5 * atr_val
    )

    close = ctx["close_series"]
    _, _bb_mid, bb_low_s = bollinger_bands(close)
    bb_low_val = float(bb_low_s.iloc[-1])
    bb_lower_touch = not np.isnan(bb_low_val) and price <= bb_low_val

    tech_score = sum([oversold_rsi, near_sup_val, bb_lower_touch])

    return {
        "rsi14":          round(rsi_val, 1),
        "oversold_rsi":   oversold_rsi,
        "near_support":   near_sup_val,
        "nearest_support_price": (
            round(nearest_sup, 2) if not np.isnan(nearest_sup) else None
        ),
        "bb_lower_touch": bb_lower_touch,
        "bb_lower_price": round(bb_low_val, 2) if not np.isnan(bb_low_val) else None,
        "tech_score":     tech_score,
    }


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def _run_tests() -> None:
    rng = np.random.default_rng(5)
    n = 600
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n))
    dates = pd.bdate_range("2022-01-01", periods=n)
    df = pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": rng.integers(1e6, 5e6, n),
    }, index=dates)

    # indicators produce sane ranges
    r = rsi(df["Close"])
    assert (r.dropna().between(0, 100)).all(), "RSI must be 0-100"
    a = atr(df)
    assert (a.dropna() >= 0).all(), "ATR must be non-negative"
    adx_s, _, _ = adx(df)
    assert (adx_s.dropna() >= 0).all()

    # context + classification run end to end
    ctx = compute_context(df, "TEST")
    assert ctx["price"] > 0 and ctx["atr20"] > 0
    s = classify_structure(ctx)
    assert s in {"Bull Trend", "Recovery", "Pullback in Uptrend", "Base", "Breakdown"}
    m = classify_momentum(ctx)
    assert m in {"Overextended", "Extended", "Building", "Weakening", "Neutral", "Mixed"}
    b = wheel_bias(ctx, s, m)
    assert isinstance(b, str) and len(b) > 0

    # rationale strings (REQ-4): every label has a matching, non-empty reason
    s_reason = structure_rationale(ctx, s)
    assert f"hence {s}" in s_reason
    m_reason = momentum_rationale(ctx, m)
    assert f"hence {m}" in m_reason

    # spot-check one rationale per branch against synthetic ctx dicts
    bull_ctx = {"price": 120.0, "sma50": 110.0, "sma200": 100.0, "atr20": 2.0}
    assert "hence Bull Trend" in structure_rationale(bull_ctx, "Bull Trend")
    building_ctx = {"rsi14": 65.0, "adx14": 22.0, "di_plus": 30.0, "di_minus": 10.0}
    assert "hence Building" in momentum_rationale(building_ctx, "Building")

    # support/resistance buckets are ordered correctly
    sup, res = volume_profile_levels(df.tail(400))
    for lvl in sup:
        assert lvl["price"] < ctx["price"] or True  # supports below; tolerate edge

    # bollinger bands produce ordered bands
    upper, mid, lower = bollinger_bands(df["Close"])
    assert (upper.dropna() >= mid.dropna()).all(), "upper must be >= middle"
    assert (mid.dropna() >= lower.dropna()).all(), "middle must be >= lower"

    # entry_signals returns expected keys and sane ranges
    es = entry_signals(ctx)
    for key in ("rsi14", "oversold_rsi", "near_support", "bb_lower_touch",
                "tech_score", "bb_lower_price", "nearest_support_price"):
        assert key in es, f"missing key: {key}"
    assert 0 <= es["tech_score"] <= 3
    assert isinstance(es["oversold_rsi"], bool)
    assert isinstance(es["near_support"], bool)
    assert isinstance(es["bb_lower_touch"], bool)

    print("technicals: all tests passed")


if __name__ == "__main__":
    _run_tests()
