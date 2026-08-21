"""REQ-1.5 / SPEC-001 AC-2: enum round-trips and legal-action authority."""
import pytest

from rlbot.state.enums import (
    CALL_DELTA_BANDS,
    MOMENTUM_TO_BUCKET,
    PUT_DELTA_BANDS,
    STRUCTURE_TO_TREND,
    CallMgmtAction,
    CashAction,
    MarketRegime,
    MomentumBucket,
    PositionState,
    PutMgmtAction,
    StockAction,
    TrendBucket,
    ValuationState,
    VolCompensation,
    legal_actions,
)


@pytest.mark.parametrize("enum_cls", [
    MarketRegime, ValuationState, VolCompensation, TrendBucket, MomentumBucket,
    CashAction, StockAction, PutMgmtAction, CallMgmtAction,
])
def test_int_round_trip(enum_cls):
    for member in enum_cls:
        assert enum_cls(int(member)) is member


def test_legal_actions_every_state_nonempty_with_wait_or_hold():
    for state in PositionState:
        actions = legal_actions(state)
        assert actions, state
        names = {a.name for a in actions}
        assert "WAIT" in names or "HOLD" in names


def test_structure_and_momentum_maps_are_total():
    assert set(STRUCTURE_TO_TREND) == {
        "Bull Trend", "Recovery", "Pullback in Uptrend", "Base", "Breakdown"
    }
    assert set(MOMENTUM_TO_BUCKET) == {
        "Overextended", "Extended", "Building", "Weakening", "Neutral", "Mixed"
    }


def test_delta_bands_cover_all_opening_actions_and_are_ordered():
    assert set(PUT_DELTA_BANDS) == set(CashAction) - {CashAction.WAIT}
    assert set(CALL_DELTA_BANDS) == set(StockAction) - {StockAction.WAIT}
    for lo, hi in list(PUT_DELTA_BANDS.values()) + list(CALL_DELTA_BANDS.values()):
        assert 0 < lo < hi < 0.5
