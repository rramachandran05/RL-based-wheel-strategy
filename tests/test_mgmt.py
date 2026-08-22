"""SPEC-001A / SPEC-007 tests: regression guard (REQ-7.1), bucket goldens
(AC-2), roll mechanics (AC-3), commit suppression (AC-5), schema v2 (AC-1)."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from rlbot.benchmarks.policies import (
    AdaptiveRulePolicy,
    FixedWheelPolicy,
    HoldMgmtPolicy,
    MosRollMgmtPolicy,
)
from rlbot.learning.trajectories import decision_to_record, validate_record
from rlbot.options.premium_source import SyntheticBSPremiumSource
from rlbot.simulator.environment import WheelEnv
from rlbot.simulator.portfolio import ExecutionConfig, Portfolio, buy_to_close, open_short_option
from rlbot.state.enums import PutMgmtAction, PositionState
from rlbot.state.mgmt import dte_bucket, moneyness_bucket, premium_captured_bucket
from tests.test_environment import make_frame

GOLDEN = json.loads((Path(__file__).parent / "golden_env_regression.json").read_text())


# ---------------- REQ-7.1: inert when management disabled ----------------
@pytest.mark.parametrize("name,policy", [("b1", FixedWheelPolicy()), ("b3", AdaptiveRulePolicy())])
def test_regression_golden_with_mgmt_disabled(name, policy):
    env = WheelEnv("TEST", make_frame(n=400, seed=5), SyntheticBSPremiumSource(iv_uplift=0.1))
    res = env.run(policy, "2021-01-04", "2022-06-30", mgmt_policy=None)
    sha = hashlib.sha256(
        json.dumps([round(v, 10) for v in res.nav.tolist()]).encode()).hexdigest()
    g = GOLDEN[name]
    assert sha == g["nav_sha"]
    assert len(res.decisions) == g["n_decisions"]
    assert [d.chosen_action for d in res.decisions] == g["actions"]


def test_hold_mgmt_policy_identical_to_disabled():
    """HOLD-always management must not change NAV vs no management at all."""
    env = WheelEnv("TEST", make_frame(n=400, seed=5), SyntheticBSPremiumSource(iv_uplift=0.1))
    off = env.run(FixedWheelPolicy(), "2021-01-04", "2022-06-30")
    on = env.run(FixedWheelPolicy(), "2021-01-04", "2022-06-30", mgmt_policy=HoldMgmtPolicy())
    pd.testing.assert_series_equal(off.nav, on.nav)
    # but management decisions were recorded
    assert any(d.mgmt_state is not None for d in on.decisions)


# ---------------- AC-2: bucket boundary goldens ----------------
@pytest.mark.parametrize("cp,spot,strike,expected", [
    ("P", 94.9, 95.0, 0),    # put ITM -> BREACHED
    ("P", 95.0, 95.0, 1),    # m == 0 -> NEAR (not breached)
    ("P", 99.75, 95.0, 1),   # m = 0.05 -> NEAR (inclusive)
    ("P", 99.8, 95.0, 2),    # m > 0.05 -> SAFE
    ("C", 105.1, 105.0, 0),  # call ITM -> BREACHED
    ("C", 100.0, 105.0, 1),  # m = 0.05 -> NEAR
    ("C", 99.0, 105.0, 2),   # SAFE
])
def test_moneyness_buckets(cp, spot, strike, expected):
    assert moneyness_bucket(cp, spot, strike) == expected


@pytest.mark.parametrize("dte,expected", [(0, 0), (7, 0), (8, 1), (21, 1), (22, 2), (40, 2)])
def test_dte_buckets(dte, expected):
    assert dte_bucket(dte) == expected


@pytest.mark.parametrize("mark,fill,expected", [
    (1.5, 2.0, 0),    # 25% captured
    (1.0, 2.0, 1),    # 50% captured (inclusive lower)
    (0.3, 2.0, 1),    # 85% captured
    (0.2, 2.0, 2),    # 90% captured
])
def test_premium_captured_buckets(mark, fill, expected):
    assert premium_captured_bucket(mark, fill) == expected


# ---------------- AC-3: roll & close mechanics ----------------
def test_buy_to_close_cash_to_the_cent():
    from rlbot.options.premium_source import Quote
    cfg = ExecutionConfig()
    q = Quote("P", 95.0, pd.Timestamp("2024-04-19"), 32, 2.00, -0.22, 0.30)
    port = open_short_option(Portfolio(cash=100_000.0), q, cfg, tier=2)
    cash_after_open = port.cash
    port = buy_to_close(port, mark=1.20, cfg=cfg)
    assert port.option is None
    assert port.cash == pytest.approx(cash_after_open - (1.20 * 1.03 * 100 + 0.65))


class _ScriptedMgmt:
    """Roll lower at the first management epoch, then hold."""
    def __init__(self):
        self.fired = False
        self.decisions = []

    def decide_mgmt(self, position_state, m_state, ctx, row):
        from rlbot.state.enums import CallMgmtAction
        if position_state != PositionState.SHORT_PUT:
            return CallMgmtAction.HOLD
        if not self.fired:
            self.fired = True
            self.decisions.append(ctx)
            return PutMgmtAction.ROLL_LOWER_RISK
        return PutMgmtAction.HOLD


def test_roll_lower_produces_lower_tier_contract():
    env = WheelEnv("TEST", make_frame(n=300, seed=5), SyntheticBSPremiumSource(iv_uplift=0.1))
    scripted = _ScriptedMgmt()
    res = env.run(FixedWheelPolicy(), "2021-01-04", "2021-12-31", mgmt_policy=scripted)
    rolls = [d for d in res.decisions
             if d.mgmt_state is not None and d.chosen_action == int(PutMgmtAction.ROLL_LOWER_RISK)]
    assert rolls, "scripted roll never fired"
    r = rolls[0]
    assert r.contract is not None and r.contract["type"] == "PUT"
    # opening tier is PUT_BALANCED (3, band .18-.25); rolled tier 2 => |delta| <= .20
    assert abs(r.contract["delta"]) <= 0.20
    assert any(c.get("early_close") for c in res.cycles)


class _CommitMgmt:
    def __init__(self):
        self.calls = 0

    def decide_mgmt(self, position_state, m_state, ctx, row):
        from rlbot.state.enums import CallMgmtAction
        if position_state != PositionState.SHORT_PUT:
            return CallMgmtAction.HOLD
        self.calls += 1
        return PutMgmtAction.ACCEPT_ASSIGNMENT


def test_commit_suppresses_further_mgmt_epochs():
    """AC-5: after ACCEPT_ASSIGNMENT, no further management calls this cycle."""
    env = WheelEnv("TEST", make_frame(n=300, seed=5), SyntheticBSPremiumSource(iv_uplift=0.1))
    commit = _CommitMgmt()
    res = env.run(FixedWheelPolicy(), "2021-01-04", "2021-06-30", mgmt_policy=commit)
    put_cycles = [c for c in res.cycles if c["leg"] == "CSP" and not c.get("early_close")]
    assert commit.calls == len([d for d in res.decisions
                                if d.mgmt_state is not None
                                and d.position_state == PositionState.SHORT_PUT.value])
    assert commit.calls <= len(put_cycles), \
        "at most one management decision per cycle after commit"


# ---------------- AC-1: schema v2 discrimination ----------------
def test_v2_schema_round_trip_and_v1_rejection():
    env = WheelEnv("TEST", make_frame(n=300, seed=5), SyntheticBSPremiumSource(iv_uplift=0.1))
    res = env.run(FixedWheelPolicy(), "2021-01-04", "2021-12-31",
                  mgmt_policy=MosRollMgmtPolicy())
    mgmt_ds = [d for d in res.decisions if d.mgmt_state is not None]
    assert mgmt_ds
    rec = decision_to_record(mgmt_ds[0], "TEST", "r", "e", 0,
                             schema_version="trajectory_v2")
    validate_record(rec)                      # v2 validates
    assert rec["reward_version"] == "diff_v2"
    with pytest.raises(ValueError):
        decision_to_record(mgmt_ds[0], "TEST", "r", "e", 0)   # v1 refuses mgmt

    rec_bad = dict(rec, schema_version="trajectory_v1")
    import jsonschema
    with pytest.raises(jsonschema.ValidationError):
        validate_record(rec_bad)              # v1 schema rejects the v2 record


def test_opening_records_still_v1_valid():
    env = WheelEnv("TEST", make_frame(n=300, seed=5), SyntheticBSPremiumSource(iv_uplift=0.1))
    res = env.run(FixedWheelPolicy(), "2021-01-04", "2021-12-31",
                  mgmt_policy=HoldMgmtPolicy())
    opening = [d for d in res.decisions if d.mgmt_state is None]
    rec = decision_to_record(opening[0], "TEST", "r", "e", 0)
    validate_record(rec)
    assert rec["schema_version"] == "trajectory_v1"
