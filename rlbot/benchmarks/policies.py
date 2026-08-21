"""Baseline policies B1-B3 (SPEC-006 §2). All implement the same interface as
the learned policy: decide(position_state, q_state, row) -> action.

B3's full 36-state rule table doubles as the Q-table prior (REQ-6.2: single
source — the RULE_TABLE built here is the one the learner imports).
"""
from __future__ import annotations

import itertools

from rlbot.state.enums import (
    CashAction,
    MarketRegime,
    PositionState,
    StockAction,
    ValuationState,
    VolCompensation,
)


class FixedWheelPolicy:
    """B1: fixed ~20-delta wheel (PUT_BALANCED / CALL_BALANCED tiers).
    Also the CASH-state reward reference."""

    def decide(self, position_state, q_state, row):
        if position_state == PositionState.CASH:
            return CashAction.PUT_BALANCED
        return StockAction.CALL_BALANCED


class ConservativeWheelPolicy:
    """B2: fixed 10-15 delta wheel."""

    def decide(self, position_state, q_state, row):
        if position_state == PositionState.CASH:
            return CashAction.PUT_CONSERVATIVE
        return StockAction.CALL_CONSERVATIVE


def _rule_cash(regime, val, vc) -> CashAction:
    """SPEC-006 B3 cash rules."""
    if regime == MarketRegime.BEAR_STRESS:
        if vc == VolCompensation.ATTRACTIVE:
            return CashAction.PUT_DEFENSIVE
        return CashAction.WAIT
    if regime == MarketRegime.SIDEWAYS:
        return CashAction.PUT_CONSERVATIVE
    # bull regimes
    if val == ValuationState.ATTRACTIVE:
        return CashAction.PUT_AGGRESSIVE
    if val == ValuationState.EXPENSIVE:
        return CashAction.PUT_CONSERVATIVE
    return CashAction.PUT_BALANCED


def _rule_stock(regime, val, vc) -> StockAction:
    """SPEC-006 B3 stock rules (mirrored: bearish/expensive -> closer strikes)."""
    if regime == MarketRegime.BEAR_STRESS or val == ValuationState.EXPENSIVE:
        return StockAction.CALL_AGGRESSIVE
    if regime in (MarketRegime.BULL_LOW_VOL, MarketRegime.BULL_HIGH_VOL) \
            and val == ValuationState.ATTRACTIVE:
        return StockAction.CALL_DEFENSIVE
    return StockAction.CALL_BALANCED


def build_rule_table() -> dict:
    """{(pos_state_value, q_state): action_int} over all 2x36 policy states."""
    table = {}
    for r, v, c in itertools.product(MarketRegime, ValuationState, VolCompensation):
        q = (int(r), int(v), int(c))
        table[(PositionState.CASH.value, q)] = int(_rule_cash(r, v, c))
        table[(PositionState.LONG_STOCK.value, q)] = int(_rule_stock(r, v, c))
    return table


RULE_TABLE = build_rule_table()


class AdaptiveRulePolicy:
    """B3: regime/valuation/vol-comp keyed rules — the yardstick and Q prior."""

    def decide(self, position_state, q_state, row):
        a = RULE_TABLE[(position_state.value, q_state)]
        return CashAction(a) if position_state == PositionState.CASH else StockAction(a)
