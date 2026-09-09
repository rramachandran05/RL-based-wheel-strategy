"""Valuation state (SPEC-001 §3.1): distance from fv_buy, FAIR when unknown."""
from __future__ import annotations

import numpy as np
import pandas as pd

from rlbot.config import ValuationThresholds
from rlbot.state.enums import ValuationState


def classify_valuation_series(
    fv_dist: pd.Series,
    thresholds: ValuationThresholds = ValuationThresholds(),
) -> pd.Series:
    """fv_dist = (price − fv_buy) / fv_buy. NaN → FAIR (declared degradation,
    SPEC-002 DATA-GAP-3)."""
    band = thresholds.band
    out = np.select(
        [fv_dist < -band, fv_dist > band],
        [ValuationState.ATTRACTIVE, ValuationState.EXPENSIVE],
        default=ValuationState.FAIR,
    )
    result = pd.Series(out, index=fv_dist.index, dtype="Int8")
    result[fv_dist.isna()] = int(ValuationState.FAIR)
    return result


# SPEC-007 §3C (Track C, gate G9): price-position proxy with zero historical
# gap. Percentile of the ticker's OWN drawdown, not a fixed % band — a stock
# that routinely swings ±30% needs different bands than one that rarely
# moves. Mirrors the EPS proxy's percentile mapping direction exactly.
DD_PCT_WINDOW, DD_PCT_MIN_OBS = 1260, 252
DD_LOW, DD_HIGH = 0.20, 0.80


def classify_drawdown_series(
    drawdown: pd.Series,
    window: int = DD_PCT_WINDOW,
    min_obs: int = DD_PCT_MIN_OBS,
    low: float = DD_LOW,
    high: float = DD_HIGH,
) -> pd.Series:
    """drawdown = close / cummax(close) − 1 (≤ 0, causal). Rolling
    right-inclusive percentile of the series against itself: the worst fifth
    of the ticker's own history (deep correction) → ATTRACTIVE; the top fifth
    (at/near highs) → EXPENSIVE; else FAIR. Warmup (< min_obs) → NA so the
    encoder treats the state as undefined, exactly as it does today."""
    from rlbot.features.regime import rolling_percentile
    dd_pct = rolling_percentile(drawdown.astype(float), window, min_obs)
    out = np.select(
        [dd_pct < low, dd_pct > high],
        [ValuationState.ATTRACTIVE, ValuationState.EXPENSIVE],
        default=ValuationState.FAIR,
    )
    result = pd.Series(out, index=drawdown.index, dtype="Int8")
    result[dd_pct.isna()] = pd.NA
    return result
