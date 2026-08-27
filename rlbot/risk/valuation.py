"""Valuation gates (SPEC-009): Wheel-FV economic boundaries and action masks.

Fed by ../fair-value-discount's ensemble CSV (SPEC-002). Everything here is
a pure function of (WheelValuation, live spot, cost basis, config) so
intraday spot moves never stale a boundary. Valuation is used as a
CONSTRAINT layer only — never a return predictor (VMI post-mortem).

Asymmetry:  PUT  = max acceptable acquisition price (net of premium)
            CALL = min acceptable exit price (net of premium)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from rlbot.config import ValuationGateConfig
from rlbot.state.enums import CashAction, StockAction


@dataclass(frozen=True)
class WheelValuation:
    """One ticker's row from fair_value_ensemble_<date>.csv (SPEC-002)."""
    ticker: str
    date: str                      # CSV as-of date
    wheel_fv: float
    put_required_mos: float
    reliability: float | None = None
    reliability_tier: str = "med"
    sentiment: float | None = None
    coverage: int = 0


class WheelRegime(IntEnum):
    """SPEC-009 five-band regime on spot/wheel_fv. Distinct from the frozen
    3-state ValuationState in the Q-state contract (SPEC-001) — this is a
    risk-layer construct, not an MDP dimension."""
    DEEP_UNDERVALUED = 0
    UNDERVALUED = 1
    FAIR_VALUED = 2
    EXPENSIVE = 3
    VERY_EXPENSIVE = 4


_REGIME_BANDS = (0.80, 0.95, 1.05, 1.20)

# VREQ-3 action masks: legal tiers per regime. WAIT is always legal.
ALLOWED_CASH = {
    WheelRegime.DEEP_UNDERVALUED: set(CashAction),
    WheelRegime.UNDERVALUED: {CashAction.WAIT, CashAction.PUT_DEFENSIVE,
                              CashAction.PUT_CONSERVATIVE,
                              CashAction.PUT_BALANCED,
                              CashAction.PUT_AGGRESSIVE},
    WheelRegime.FAIR_VALUED: {CashAction.WAIT, CashAction.PUT_DEFENSIVE,
                              CashAction.PUT_CONSERVATIVE,
                              CashAction.PUT_BALANCED},
    WheelRegime.EXPENSIVE: {CashAction.WAIT, CashAction.PUT_DEFENSIVE,
                            CashAction.PUT_CONSERVATIVE},
    WheelRegime.VERY_EXPENSIVE: {CashAction.WAIT},       # NO_PUT
}
ALLOWED_STOCK = {
    WheelRegime.DEEP_UNDERVALUED: {StockAction.WAIT,
                                   StockAction.CALL_DEFENSIVE},  # ~NO_CALL
    WheelRegime.UNDERVALUED: {StockAction.WAIT, StockAction.CALL_DEFENSIVE,
                              StockAction.CALL_CONSERVATIVE},
    WheelRegime.FAIR_VALUED: {StockAction.WAIT, StockAction.CALL_DEFENSIVE,
                              StockAction.CALL_CONSERVATIVE,
                              StockAction.CALL_BALANCED},
    WheelRegime.EXPENSIVE: set(StockAction),
    WheelRegime.VERY_EXPENSIVE: set(StockAction),
}


def wheel_regime(spot: float, wheel_fv: float) -> WheelRegime | None:
    if not spot or not wheel_fv or wheel_fv <= 0:
        return None
    ratio = spot / wheel_fv
    if ratio < _REGIME_BANDS[0]:
        return WheelRegime.DEEP_UNDERVALUED
    if ratio < _REGIME_BANDS[1]:
        return WheelRegime.UNDERVALUED
    if ratio <= _REGIME_BANDS[2]:
        return WheelRegime.FAIR_VALUED
    if ratio <= _REGIME_BANDS[3]:
        return WheelRegime.EXPENSIVE
    return WheelRegime.VERY_EXPENSIVE


def put_ceiling(val: WheelValuation, spot: float,
                cfg: ValuationGateConfig = ValuationGateConfig()) -> float:
    """Max acceptable net assignment basis: the FV margin-of-safety side AND
    the spot-discount side (deep undervaluation must never authorize a
    near-the-money put)."""
    fv_side = val.wheel_fv * (1 - val.put_required_mos)
    if spot and spot > 0:
        return min(fv_side, spot * (1 - cfg.spot_discount))
    return fv_side


def exit_floor(val: WheelValuation, cost_basis: float | None,
               cfg: ValuationGateConfig = ValuationGateConfig()) -> float:
    """Min acceptable effective exit (strike + premium): don't sell an
    undervalued company too cheaply AND clear the economic return bar."""
    floor = cfg.exit_floor_factor * val.wheel_fv
    if cost_basis and cost_basis > 0:
        floor = max(floor, cost_basis * (1 + cfg.min_call_gain))
    return floor


def premium_required(strike: float, ceiling: float) -> float:
    """Minimum premium that makes a put strike acceptable (verify the live
    quote clears this — model premiums are synthetic)."""
    return max(0.0, strike - ceiling)


def allowed_actions(action_cls, regime: WheelRegime | None) -> set:
    """Legal opening tiers for a regime; unknown regime -> everything
    (declared degradation, VREQ-7)."""
    table = ALLOWED_CASH if action_cls is CashAction else ALLOWED_STOCK
    if regime is None:
        return set(action_cls)
    return table[regime]


def clamp_action(action, regime: WheelRegime | None):
    """Map a blocked tier to the highest allowed tier at/below it (tiers are
    ordered IntEnums; WAIT=0 is always allowed)."""
    cls = type(action)
    allowed = allowed_actions(cls, regime)
    if action in allowed:
        return action
    for v in range(int(action) - 1, -1, -1):
        if cls(v) in allowed:
            return cls(v)
    return cls(0)


def put_gate_flags(strike: float, premium: float, val: WheelValuation,
                   spot: float,
                   cfg: ValuationGateConfig = ValuationGateConfig()) -> list:
    flags = []
    regime = wheel_regime(spot, val.wheel_fv)
    if regime == WheelRegime.VERY_EXPENSIVE:
        flags.append("VAL-2:very_expensive_no_put")
    if strike - premium > put_ceiling(val, spot, cfg) + 1e-9:
        flags.append("VAL-1:net_basis_above_ceiling")
    return flags


def call_gate_flags(strike: float, premium: float, val: WheelValuation,
                    cost_basis: float | None,
                    cfg: ValuationGateConfig = ValuationGateConfig()) -> list:
    if strike + premium < exit_floor(val, cost_basis, cfg) - 1e-9:
        return ["VAL-3:below_exit_floor"]
    return []
