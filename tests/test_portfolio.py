"""SPEC-003 AC-1: hand-computed 3-cycle path through the state machine."""
import pandas as pd
import pytest

from rlbot.options.premium_source import Quote
from rlbot.simulator.portfolio import (
    ExecutionConfig,
    Portfolio,
    nav,
    open_short_option,
    settle_expiration,
)
from rlbot.state.enums import PositionState

CFG = ExecutionConfig()  # 3% slippage, $0.65/contract, 1 contract
EXP = pd.Timestamp("2024-04-19")


def _put(strike=95.0, mid=2.00):
    return Quote("P", strike, EXP, 32, mid, -0.22, 0.30)


def _call(strike=105.0, mid=1.50):
    return Quote("C", strike, EXP + pd.Timedelta(days=35), 35, mid, 0.20, 0.30)


def test_three_cycle_hand_computed_path():
    port = Portfolio(cash=100_000.0)

    # --- Cycle 1: sell 95P for mid 2.00 -> fill 1.94, proceeds 194 - 0.65
    port = open_short_option(port, _put(), CFG)
    assert port.position_state == PositionState.SHORT_PUT
    assert port.cash == pytest.approx(100_000 + 2.00 * 0.97 * 100 - 0.65)
    cash_after_open = port.cash

    # expires OTM (close 97 >= 95): premium kept, back to CASH
    port = settle_expiration(port, close=97.0)
    assert port.position_state == PositionState.CASH
    assert port.cash == pytest.approx(cash_after_open)

    # --- Cycle 2: sell again, assigned at 95 (close 90 < 95)
    port = open_short_option(port, _put(), CFG)
    port = settle_expiration(port, close=90.0)
    assert port.position_state == PositionState.LONG_STOCK
    assert port.shares == 100
    assert port.cost_basis == pytest.approx(95.0 - 1.94)  # strike - premium fill
    cash_after_assign = cash_after_open + (2.00 * 0.97 * 100 - 0.65) - 95.0 * 100
    assert port.cash == pytest.approx(cash_after_assign)

    # --- Cycle 3: covered call 105C, called away (close 110 > 105)
    port = open_short_option(port, _call(), CFG)
    assert port.position_state == PositionState.COVERED_CALL
    port = settle_expiration(port, close=110.0)
    assert port.position_state == PositionState.CASH
    assert port.shares == 0 and port.cost_basis is None
    expected = cash_after_assign + (1.50 * 0.97 * 100 - 0.65) + 105.0 * 100
    assert port.cash == pytest.approx(expected)


def test_boundary_close_equals_strike_no_assignment():
    port = open_short_option(Portfolio(cash=100_000.0), _put(strike=95.0), CFG)
    port = settle_expiration(port, close=95.0)
    assert port.position_state == PositionState.CASH

    port2 = Portfolio(cash=0.0, shares=100, cost_basis=90.0)
    port2 = open_short_option(port2, _call(strike=105.0), CFG)
    port2 = settle_expiration(port2, close=105.0)
    assert port2.position_state == PositionState.LONG_STOCK


def test_nav_includes_short_option_liability():
    port = open_short_option(Portfolio(cash=100_000.0), _put(mid=2.00), CFG)
    # option mark moved against us to 5.00/share
    n = nav(port, spot=92.0, option_mark=5.0)
    assert n == pytest.approx(port.cash - 500.0)
