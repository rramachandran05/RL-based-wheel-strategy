"""SPEC-003 environment tests: state-machine flow, determinism, reward-vs-self,
trajectory schema round-trip."""
import numpy as np
import pandas as pd
import pytest

from rlbot.benchmarks.policies import AdaptiveRulePolicy, FixedWheelPolicy
from rlbot.learning.trajectories import decision_to_record, validate_record
from rlbot.options.premium_source import SyntheticBSPremiumSource
from rlbot.simulator.environment import WheelEnv, attach_rewards
from rlbot.state.enums import PositionState


def make_frame(n=300, seed=5, regime=0, vol_comp=1, trend_drift=0.0004):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n)
    close = 100.0 * np.exp(np.cumsum(rng.normal(trend_drift, 0.012, n)))
    f = pd.DataFrame(index=dates)
    f["close"] = close
    f["high"] = close * 1.005
    f["low"] = close * 0.995
    f["vol_proxy"] = 0.28
    f["market_regime"] = regime
    f["vol_compensation"] = vol_comp
    f["valuation_state"] = pd.array([pd.NA] * n, dtype="Int8")
    f["fv_dist"] = np.nan
    for col in ("vix_close", "vix_pct_5y", "vrp", "rsi14", "sma50", "sma200",
                "atr20", "drawdown"):
        f[col] = 1.0
    return f


@pytest.fixture(scope="module")
def env():
    return WheelEnv("TEST", make_frame(), SyntheticBSPremiumSource())


def test_episode_runs_and_wheels(env):
    res = env.run(FixedWheelPolicy(), "2021-01-04", "2022-02-28")
    assert len(res.nav) > 200
    assert res.cycles, "no option cycles completed"
    states = {d.position_state for d in res.decisions}
    assert PositionState.CASH.value in states
    # every decision record was finalized
    assert all(d.next_epoch_date is not None for d in res.decisions)
    assert res.decisions[-1].terminal


def test_determinism(env):
    r1 = env.run(FixedWheelPolicy(), "2021-01-04", "2022-02-28")
    r2 = env.run(FixedWheelPolicy(), "2021-01-04", "2022-02-28")
    pd.testing.assert_series_equal(r1.nav, r2.nav)
    assert [d.chosen_action for d in r1.decisions] == [d.chosen_action for d in r2.decisions]


def test_reward_zero_against_self(env):
    """REQ-3.4 / AC-3: policy identical to its reference -> zero reward on
    cash-state decisions when its own NAV is the cash reference."""
    res = env.run(FixedWheelPolicy(), "2021-01-04", "2022-02-28")
    attach_rewards(res, cash_ref_nav=res.nav, frame=env.frame)
    cash_rewards = [d.reward for d in res.decisions
                    if d.position_state == PositionState.CASH.value]
    assert cash_rewards
    assert all(abs(r) < 1e-12 for r in cash_rewards)


def test_stock_state_reference_is_buy_and_hold(env):
    res = env.run(FixedWheelPolicy(), "2021-01-04", "2022-06-30")
    attach_rewards(res, cash_ref_nav=res.nav, frame=env.frame)
    stock_ds = [d for d in res.decisions
                if d.position_state == PositionState.LONG_STOCK.value
                and d.reward is not None]
    for d in stock_ds:
        w = env.frame["close"].loc[d.date:d.next_epoch_date]
        assert d.reference_return == pytest.approx(w.iloc[-1] / w.iloc[0] - 1.0)


def test_rule_policy_waits_in_stress():
    frame = make_frame(regime=3, vol_comp=1)  # BEAR_STRESS, vol comp NORMAL
    env = WheelEnv("TEST", frame, SyntheticBSPremiumSource())
    res = env.run(AdaptiveRulePolicy(), "2021-01-04", "2021-12-31")
    cash_ds = [d for d in res.decisions if d.position_state == PositionState.CASH.value]
    assert cash_ds
    assert all(d.chosen_action == 0 for d in cash_ds), "B3 must WAIT in stress+normal vol"
    assert not res.cycles


def test_trajectory_records_validate(env):
    res = env.run(FixedWheelPolicy(), "2021-01-04", "2021-12-31")
    attach_rewards(res, cash_ref_nav=res.nav, frame=env.frame)
    for i, d in enumerate(res.decisions):
        rec = decision_to_record(d, "TEST", "run0", "ep0", i)
        validate_record(rec)


def test_schema_rejects_missing_field(env):
    res = env.run(FixedWheelPolicy(), "2021-01-04", "2021-06-30")
    attach_rewards(res, cash_ref_nav=res.nav, frame=env.frame)
    rec = decision_to_record(res.decisions[0], "TEST", "run0", "ep0", 0)
    del rec["reward"]
    import jsonschema
    with pytest.raises(jsonschema.ValidationError):
        validate_record(rec)


def test_nav_accounting_identity(env):
    """REQ-3.2 (simplified): with no open option, NAV == cash + shares*close."""
    res = env.run(FixedWheelPolicy(), "2021-01-04", "2022-02-28")
    port = res.final_portfolio
    last_date = res.nav.index[-1]
    spot = env.frame.loc[last_date, "close"]
    if port.option is None:
        assert res.nav.iloc[-1] == pytest.approx(port.cash + port.shares * spot)
