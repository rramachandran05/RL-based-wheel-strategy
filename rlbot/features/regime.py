"""Market-regime classification (SPEC-001 §3.1, SPEC-002 §3.3).

Rule-based, computed from SPY bars + VIX percentile. Precedence:
BEAR_STRESS > BULL_HIGH_VOL > BULL_LOW_VOL > SIDEWAYS (default bucket).
Rows with insufficient history (NaN inputs) get pandas NA.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rlbot.config import RegimeThresholds
from rlbot.state.enums import MarketRegime


def classify_regime_series(
    close: pd.Series,
    sma200: pd.Series,
    drawdown: pd.Series,
    vix_pct: pd.Series,
    thresholds: RegimeThresholds = RegimeThresholds(),
) -> pd.Series:
    t = thresholds
    valid = close.notna() & sma200.notna() & drawdown.notna() & vix_pct.notna()

    bear = (
        ((close < sma200) & (drawdown <= t.dd_bull))
        | (drawdown <= t.dd_stress)
        | (vix_pct >= t.vix_pct_stress)
    )
    bull = (close > sma200) & (drawdown > t.dd_bull) & ~bear
    bull_high = bull & (vix_pct >= t.vix_pct_high)
    bull_low = bull & (vix_pct < t.vix_pct_high)

    out = np.select(
        [bear, bull_high, bull_low],
        [MarketRegime.BEAR_STRESS, MarketRegime.BULL_HIGH_VOL, MarketRegime.BULL_LOW_VOL],
        default=MarketRegime.SIDEWAYS,
    )
    result = pd.Series(out, index=close.index, dtype="Int8")
    result[~valid] = pd.NA
    return result


def rolling_percentile(series: pd.Series, window: int, min_obs: int) -> pd.Series:
    """Right-inclusive rolling percentile rank of the latest value (SPEC-002 §3.3)."""
    return series.rolling(window, min_periods=min_obs).apply(
        lambda x: float(np.mean(x <= x[-1])), raw=True
    )


def expanding_drawdown(close: pd.Series) -> pd.Series:
    """Distance from the running all-time high (causal)."""
    return close / close.cummax() - 1.0


def realized_vol(close: pd.Series, window: int) -> pd.Series:
    """Annualized close-to-close realized volatility (matches vendored proxy)."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std(ddof=1) * np.sqrt(252)
