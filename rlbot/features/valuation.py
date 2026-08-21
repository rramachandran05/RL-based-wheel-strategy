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
