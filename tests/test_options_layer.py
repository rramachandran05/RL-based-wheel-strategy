"""SPEC-004 tests: premium source, selector (golden/property/monotonicity), risk engine."""
import numpy as np
import pandas as pd
import pytest

from rlbot.options.premium_source import SyntheticBSPremiumSource, bs_delta
from rlbot.options.selector import SelectorConfig, select_contract
from rlbot.risk.engine import RiskConfig, validate_open
from rlbot.state.enums import CashAction, StockAction, ValuationState

DATE = pd.Timestamp("2024-03-11")  # a Monday
SPOT, VOL = 100.0, 0.30
PS = SyntheticBSPremiumSource()


@pytest.fixture(scope="module")
def put_chain():
    return PS.chain(DATE, SPOT, VOL, "P")


@pytest.fixture(scope="module")
def call_chain():
    return PS.chain(DATE, SPOT, VOL, "C")


def test_chain_dte_window_and_delta_signs(put_chain):
    assert put_chain
    for q in put_chain:
        assert 25 <= q.dte <= 45
        assert -1.0 <= q.delta < 0.0
        assert q.expiration.dayofweek == 4


def test_put_delta_monotone_in_strike(put_chain):
    exp = put_chain[0].expiration
    same_exp = sorted([q for q in put_chain if q.expiration == exp], key=lambda q: q.strike)
    deltas = [q.delta for q in same_exp]
    assert all(a >= b - 1e-12 for a, b in zip(deltas, deltas[1:])), \
        "put delta must grow more negative as strike rises"


def test_reprice_at_expiry_is_intrinsic():
    exp = DATE + pd.Timedelta(days=30)
    assert PS.reprice("P", 100, exp, exp, spot=90.0, vol_proxy=VOL) == pytest.approx(10.0)
    assert PS.reprice("P", 100, exp, exp, spot=110.0, vol_proxy=VOL) == 0.0
    assert PS.reprice("C", 100, exp, exp, spot=110.0, vol_proxy=VOL) == pytest.approx(10.0)


def test_iv_uplift_raises_premiums():
    base = SyntheticBSPremiumSource(iv_uplift=0.0).price("P", SPOT, 95.0, 30 / 365, VOL)
    up = SyntheticBSPremiumSource(iv_uplift=0.2).price("P", SPOT, 95.0, 30 / 365, VOL)
    assert up > base


@pytest.mark.parametrize("action", [a for a in CashAction if a != CashAction.WAIT])
def test_selected_put_within_band(put_chain, action):
    from rlbot.state.enums import PUT_DELTA_BANDS
    q, n = select_contract(action, put_chain, SPOT, VOL, ValuationState.FAIR)
    assert q is not None and n > 0
    lo, hi = PUT_DELTA_BANDS[action]
    cfg = SelectorConfig()
    assert lo - cfg.band_widen <= abs(q.delta) <= hi + cfg.band_widen


def test_wait_returns_none(put_chain):
    q, n = select_contract(CashAction.WAIT, put_chain, SPOT, VOL, ValuationState.FAIR)
    assert q is None and n == 0


def test_valuation_monotonicity(put_chain):
    """REQ-4.3: EXPENSIVE must never pick a strike nearer the money than ATTRACTIVE."""
    qa, _ = select_contract(CashAction.PUT_BALANCED, put_chain, SPOT, VOL, ValuationState.ATTRACTIVE)
    qe, _ = select_contract(CashAction.PUT_BALANCED, put_chain, SPOT, VOL, ValuationState.EXPENSIVE)
    assert qe.strike <= qa.strike


def test_call_respects_cost_basis(call_chain):
    q, _ = select_contract(StockAction.CALL_AGGRESSIVE, call_chain, SPOT, VOL,
                           ValuationState.FAIR, cost_basis=99.0)
    assert q is None or q.strike >= 99.0


def test_selector_deterministic(put_chain):
    picks = {select_contract(CashAction.PUT_BALANCED, put_chain, SPOT, VOL,
                             ValuationState.FAIR)[0].strike for _ in range(5)}
    assert len(picks) == 1


# ---------------- risk engine ----------------
def _quote(cp="P", strike=95.0):
    from rlbot.options.premium_source import Quote
    return Quote(cp=cp, strike=strike, expiration=DATE + pd.Timedelta(days=32),
                 dte=32, mid=2.0, delta=-0.2 if cp == "P" else 0.2, vol_used=VOL)


def test_risk1_cash_secured():
    bad = validate_open(_quote(), 1, cash=5_000, shares=0, nav=100_000,
                        open_put_escrow=0, event_in_window=False,
                        cfg=RiskConfig.single_ticker())
    assert not bad.passed and "RISK-1:cash_secured" in bad.flags
    ok = validate_open(_quote(), 1, cash=10_000, shares=0, nav=100_000,
                       open_put_escrow=0, event_in_window=False,
                       cfg=RiskConfig.single_ticker())
    assert ok.passed


def test_risk2_naked_call():
    bad = validate_open(_quote("C"), 1, cash=100_000, shares=50, nav=100_000,
                        open_put_escrow=0, event_in_window=False,
                        cfg=RiskConfig.single_ticker())
    assert not bad.passed and "RISK-2:naked_call" in bad.flags


def test_risk8_assignment_at_once_portfolio_mode():
    cfg = RiskConfig()  # portfolio defaults: 40% cap
    bad = validate_open(_quote(strike=350.0), 1, cash=100_000, shares=0, nav=100_000,
                        open_put_escrow=10_000, event_in_window=False, cfg=cfg)
    assert "RISK-8:assignment_at_once" in bad.flags


def test_risk7_earnings_blackout():
    bad = validate_open(_quote(), 1, cash=100_000, shares=0, nav=1_000_000,
                        open_put_escrow=0, event_in_window=True,
                        cfg=RiskConfig.single_ticker())
    assert "RISK-7:earnings_blackout" in bad.flags


def test_wait_always_passes():
    assert validate_open(None, 1, 0, 0, 0, 0, True, RiskConfig()).passed
