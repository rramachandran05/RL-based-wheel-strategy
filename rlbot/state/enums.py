"""All enum <-> int mappings for the frozen state/action contract (SPEC-001).

REQ-1.5: this is the single home for these mappings. Nothing else may define
its own regime/valuation/action integers.
"""
from __future__ import annotations

from enum import Enum, IntEnum

SCHEMA_VERSION = "trajectory_v1"
REWARD_VERSION = "diff_v1"


class PositionState(str, Enum):
    CASH = "CASH"
    SHORT_PUT = "SHORT_PUT"
    LONG_STOCK = "LONG_STOCK"
    COVERED_CALL = "COVERED_CALL"


class MarketRegime(IntEnum):
    BULL_LOW_VOL = 0
    BULL_HIGH_VOL = 1
    SIDEWAYS = 2
    BEAR_STRESS = 3


class ValuationState(IntEnum):
    ATTRACTIVE = 0
    FAIR = 1          # also the FV-unknown degradation value (SPEC-001 §3.1)
    EXPENSIVE = 2


class VolCompensation(IntEnum):
    POOR = 0
    NORMAL = 1
    ATTRACTIVE = 2


class TrendBucket(IntEnum):
    """Logged-only feature (SPEC-002 §3.2): vendored structure labels -> ints."""
    BREAKDOWN = 0
    PULLBACK_IN_UPTREND = 1
    BASE = 2
    RECOVERY = 3
    BULL_TREND = 4


class MomentumBucket(IntEnum):
    """Logged-only feature: vendored momentum labels -> ints (0 reserved)."""
    WEAKENING = 1
    NEUTRAL = 2
    BUILDING = 3
    OVEREXTENDED = 4


STRUCTURE_TO_TREND = {
    "Breakdown": TrendBucket.BREAKDOWN,
    "Pullback in Uptrend": TrendBucket.PULLBACK_IN_UPTREND,
    "Base": TrendBucket.BASE,
    "Recovery": TrendBucket.RECOVERY,
    "Bull Trend": TrendBucket.BULL_TREND,
}

MOMENTUM_TO_BUCKET = {
    "Weakening": MomentumBucket.WEAKENING,
    "Neutral": MomentumBucket.NEUTRAL,
    "Mixed": MomentumBucket.NEUTRAL,
    "Building": MomentumBucket.BUILDING,
    "Extended": MomentumBucket.BUILDING,
    "Overextended": MomentumBucket.OVEREXTENDED,
}


class CashAction(IntEnum):
    WAIT = 0
    PUT_DEFENSIVE = 1
    PUT_CONSERVATIVE = 2
    PUT_BALANCED = 3
    PUT_AGGRESSIVE = 4
    PUT_VERY_AGGRESSIVE = 5


class StockAction(IntEnum):
    WAIT = 0
    CALL_DEFENSIVE = 1
    CALL_CONSERVATIVE = 2
    CALL_BALANCED = 3
    CALL_AGGRESSIVE = 4


class PutMgmtAction(IntEnum):
    HOLD = 0
    CLOSE = 1
    ROLL_SAME_RISK = 2
    ROLL_LOWER_RISK = 3
    ROLL_HIGHER_RISK = 4
    ACCEPT_ASSIGNMENT = 5


class CallMgmtAction(IntEnum):
    HOLD = 0
    CLOSE = 1
    ROLL_OUT = 2
    ROLL_UP_AND_OUT = 3
    ALLOW_CALL_AWAY = 4


# Delta bands per opening action (config-tunable engineering parameters;
# the frozen part is action identity/ordering, not these numbers).
PUT_DELTA_BANDS = {
    CashAction.PUT_DEFENSIVE: (0.05, 0.10),
    CashAction.PUT_CONSERVATIVE: (0.10, 0.18),
    CashAction.PUT_BALANCED: (0.18, 0.25),
    CashAction.PUT_AGGRESSIVE: (0.25, 0.35),
    CashAction.PUT_VERY_AGGRESSIVE: (0.35, 0.45),
}

CALL_DELTA_BANDS = {
    StockAction.CALL_DEFENSIVE: (0.05, 0.10),
    StockAction.CALL_CONSERVATIVE: (0.10, 0.18),
    StockAction.CALL_BALANCED: (0.18, 0.25),
    StockAction.CALL_AGGRESSIVE: (0.25, 0.35),
}

TARGET_DTE = (25, 45, 30)  # (min, max, preferred)


def legal_actions(position_state: PositionState) -> list:
    """REQ-1.1: the single authority on which actions are legal per state."""
    return {
        PositionState.CASH: list(CashAction),
        PositionState.SHORT_PUT: list(PutMgmtAction),
        PositionState.LONG_STOCK: list(StockAction),
        PositionState.COVERED_CALL: list(CallMgmtAction),
    }[position_state]
