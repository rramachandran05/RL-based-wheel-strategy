"""Vol-compensation state (SPEC-001 §3.1) — market-level VRP proxy.

vrp = VIX − 100·realized_vol_20(SPY), both in vol points.
POOR if vrp < 0 or vix_pct < poor-threshold;
ATTRACTIVE if vrp >= attractive-threshold and vix_pct >= attractive-percentile;
NORMAL otherwise. NA where inputs are NA.

When per-ticker IV exists (historical chains — not in v1 data plan), the same
enum is fed from iv_pctile / (iv − rv30) behind a config switch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rlbot.config import VolCompThresholds
from rlbot.state.enums import VolCompensation


def classify_vol_comp_series(
    vrp: pd.Series,
    vix_pct: pd.Series,
    thresholds: VolCompThresholds = VolCompThresholds(),
) -> pd.Series:
    t = thresholds
    valid = vrp.notna() & vix_pct.notna()

    poor = (vrp < 0) | (vix_pct < t.vix_pct_poor)
    attractive = ~poor & (vrp >= t.vrp_attractive) & (vix_pct >= t.vix_pct_attractive)

    out = np.select(
        [poor, attractive],
        [VolCompensation.POOR, VolCompensation.ATTRACTIVE],
        default=VolCompensation.NORMAL,
    )
    result = pd.Series(out, index=vrp.index, dtype="Int8")
    result[~valid] = pd.NA
    return result
