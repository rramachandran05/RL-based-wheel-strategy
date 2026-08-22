"""Management-state encoding (SPEC-001A §2) — MgmtStateV1 buckets."""
from __future__ import annotations

import pandas as pd

NEAR_BAND = 0.05
DTE_EXPIRY_WEEK = 7
DTE_MID_MAX = 21
CHALLENGE_DELTA = 0.40   # M3 trigger (SPEC-001A §4)


def moneyness(cp: str, spot: float, strike: float) -> float:
    """Positive = safe side for both legs (SPEC-001A §2)."""
    return (spot - strike) / strike if cp == "P" else (strike - spot) / spot


def moneyness_bucket(cp: str, spot: float, strike: float) -> int:
    m = moneyness(cp, spot, strike)
    if m < 0:
        return 0   # BREACHED
    if m <= NEAR_BAND:
        return 1   # NEAR
    return 2       # SAFE


def dte_bucket(dte: int) -> int:
    if dte <= DTE_EXPIRY_WEEK:
        return 0   # EXPIRY_WEEK
    if dte <= DTE_MID_MAX:
        return 1   # MID
    return 2       # EARLY


def premium_captured(mark: float, premium_fill: float) -> float:
    if premium_fill <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - mark / premium_fill))


def premium_captured_bucket(mark: float, premium_fill: float) -> int:
    pc = premium_captured(mark, premium_fill)
    if pc < 0.50:
        return 0
    if pc <= 0.85:
        return 1
    return 2


def encode_mgmt_state(regime, cp: str, spot: float, strike: float, dte: int) -> tuple | None:
    """(market_regime, moneyness_bucket, dte_bucket) — None during regime warmup."""
    if pd.isna(regime):
        return None
    return (int(regime), moneyness_bucket(cp, spot, strike), dte_bucket(dte))
