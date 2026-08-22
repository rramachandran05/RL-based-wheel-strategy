"""SPEC-008 §4 leveraged-ETF rule tests."""
import itertools

import pytest

from rlbot.benchmarks.policies import (
    LeveragedETFPolicy,
    leveraged_cash_action,
    leveraged_stock_action,
)
from rlbot.config import RlbotConfig
from rlbot.state.enums import (
    CashAction,
    MarketRegime,
    PositionState,
    StockAction,
    ValuationState,
    VolCompensation,
)


def test_leveraged_waits_in_stress_regardless_of_vol_comp():
    """No attractive-vol-comp exception for 3x funds (unlike B3)."""
    for val, vc in itertools.product(ValuationState, VolCompensation):
        q = (int(MarketRegime.BEAR_STRESS), int(val), int(vc))
        assert leveraged_cash_action(q) == CashAction.WAIT


def test_leveraged_never_above_balanced():
    for r, v, c in itertools.product(MarketRegime, ValuationState, VolCompensation):
        a = leveraged_cash_action((int(r), int(v), int(c)))
        assert int(a) <= int(CashAction.PUT_BALANCED)


@pytest.mark.parametrize("regime,expected", [
    (MarketRegime.BULL_LOW_VOL, CashAction.PUT_BALANCED),
    (MarketRegime.BULL_HIGH_VOL, CashAction.PUT_CONSERVATIVE),
    (MarketRegime.SIDEWAYS, CashAction.PUT_CONSERVATIVE),
    (MarketRegime.BEAR_STRESS, CashAction.WAIT),
])
def test_leveraged_cash_table(regime, expected):
    assert leveraged_cash_action((int(regime), 1, 1)) == expected


def test_leveraged_stock_bias_reduces_exposure():
    assert leveraged_stock_action((0, 1, 1)) == StockAction.CALL_BALANCED
    for r in (1, 2, 3):
        assert leveraged_stock_action((r, 1, 1)) == StockAction.CALL_AGGRESSIVE


def test_policy_interface():
    pol = LeveragedETFPolicy()
    assert pol.decide(PositionState.CASH, (3, 1, 2), None) == CashAction.WAIT
    assert pol.decide(PositionState.LONG_STOCK, (0, 1, 1), None) == StockAction.CALL_BALANCED


def test_config_universe_and_flags():
    cfg = RlbotConfig()
    assert set(["TQQQ", "SPXL", "CHPS", "SPYI"]).issubset(cfg.assistant_universe)
    assert "TQQQ" not in cfg.tickers          # training universe untouched
    assert cfg.is_leveraged("tqqq") and cfg.is_leveraged("SPXL")
    assert not cfg.is_leveraged("SPYI") and not cfg.is_leveraged("CHPS")


def test_leveraged_recommendation_uses_capped_policy():
    from rlbot.assistant.daily import recommend_opening
    from rlbot.benchmarks.policies import LeveragedETFPolicy
    from rlbot.options.premium_source import SyntheticBSPremiumSource
    from tests.test_environment import make_frame

    ps = SyntheticBSPremiumSource(iv_uplift=0.1)
    # BULL_HIGH_VOL: B3 (valuation FAIR) would pick BALANCED; leveraged must pick CONSERVATIVE
    frame = make_frame(n=300, seed=5, regime=1, vol_comp=1)
    rec = recommend_opening("TQQQ", frame, ps, 100_000,
                            policy=LeveragedETFPolicy(), leveraged=True)
    assert rec["leveraged"] is True
    assert rec["policy_action"] == "PUT_CONSERVATIVE"
    assert abs(rec["contract"]["delta"]) <= 0.20
    # stress -> WAIT
    frame_s = make_frame(n=300, seed=5, regime=3, vol_comp=2)
    rec_s = recommend_opening("TQQQ", frame_s, ps, 100_000,
                              policy=LeveragedETFPolicy(), leveraged=True)
    assert rec_s["action"] == "WAIT"
