"""SPEC-007 Track A learning tests: sweep HOLD≡0 (REQ-7.2), roll branch
mechanics, MgmtQTable pessimism, z→∞ limit."""
import pandas as pd
import pytest

from rlbot.learning.mgmt_qtable import MgmtQTable, LearnedMgmtPolicy
from rlbot.learning.mgmt_sweep import (
    MgmtSweepTarget,
    simulate_mgmt_branch,
    sweep_mgmt_ticker,
)
from rlbot.options.premium_source import SyntheticBSPremiumSource
from rlbot.state.enums import PositionState, PutMgmtAction
from tests.test_environment import make_frame

PS = SyntheticBSPremiumSource(iv_uplift=0.1)


@pytest.fixture(scope="module")
def sweep_targets():
    frame = make_frame(n=420, seed=5)
    return sweep_mgmt_ticker("TEST", frame, PS, "2021-01-04", "2022-06-30")


def test_sweep_produces_targets_and_hold_is_zero(sweep_targets):
    assert sweep_targets, "no management epochs swept"
    holds = [t for t in sweep_targets if t.action == int(PutMgmtAction.HOLD)
             and t.pos_state == PositionState.SHORT_PUT.value]
    assert holds
    assert all(t.target == 0.0 for t in holds)          # REQ-7.2
    others = [t for t in sweep_targets if t.action != 0]
    assert others, "no counterfactual branches simulated"


def test_close_branch_on_flat_path_costs_roughly_remaining_premium(sweep_targets):
    """On paths where the put expires OTM, closing early forfeits remaining
    premium + friction, so CLOSE targets should be mostly negative."""
    closes = [t.target for t in sweep_targets
              if t.action == int(PutMgmtAction.CLOSE)
              and t.pos_state == PositionState.SHORT_PUT.value]
    assert closes
    assert sum(1 for c in closes if c < 0) / len(closes) > 0.5


def test_branch_common_horizon(sweep_targets):
    frame = make_frame(n=420, seed=5)
    t0 = sweep_targets[0].date
    # rebuild ctx via a direct branch call for HOLD: must return a finite float
    # (already exercised inside the sweep); here just check determinism
    holds = [t for t in sweep_targets if t.action == 0]
    again = sweep_mgmt_ticker("TEST", frame, PS, "2021-01-04", "2022-06-30")
    assert [t.target for t in again] == [t.target for t in sweep_targets]


def test_qtable_deploys_hold_without_evidence():
    table = MgmtQTable().fit([])
    a = table.deployed_action(PositionState.SHORT_PUT.value, (0, 1, 1))
    assert a == PutMgmtAction.HOLD


def _mk(action, target, cluster):
    return MgmtSweepTarget("T", pd.Timestamp("2020-01-01"),
                           PositionState.SHORT_PUT.value, (0, 0, 1),
                           int(action), target, cluster)


def test_qtable_deploys_challenger_only_with_strong_evidence():
    # 8 independent clusters of consistently positive CLOSE advantage
    targets = [_mk(PutMgmtAction.HOLD, 0.0, f"c{i}") for i in range(8)]
    targets += [_mk(PutMgmtAction.CLOSE, 0.02 + 0.001 * i, f"c{i}") for i in range(8)]
    table = MgmtQTable().fit(targets)
    assert table.deployed_action(PositionState.SHORT_PUT.value, (0, 0, 1)) \
        == PutMgmtAction.CLOSE
    # same mean but only 2 clusters -> stays HOLD (MIN_N_EFF)
    t2 = [_mk(PutMgmtAction.HOLD, 0.0, f"d{i}") for i in range(2)]
    t2 += [_mk(PutMgmtAction.CLOSE, 0.02, f"d{i}") for i in range(2)]
    table2 = MgmtQTable().fit(t2)
    assert table2.deployed_action(PositionState.SHORT_PUT.value, (0, 0, 1)) \
        == PutMgmtAction.HOLD


def test_qtable_z_infinity_converges_to_hold():
    targets = [_mk(PutMgmtAction.CLOSE, 0.05, f"c{i}") for i in range(20)]
    table = MgmtQTable().fit(targets)
    assert table.deployed_action(PositionState.SHORT_PUT.value, (0, 0, 1), z=1e9) \
        == PutMgmtAction.HOLD


def test_noisy_challenger_rejected():
    """High-variance positive-mean challenger must not clear the LCB bar."""
    import itertools
    vals = itertools.cycle([0.30, -0.28])
    targets = [_mk(PutMgmtAction.ROLL_LOWER_RISK, next(vals), f"c{i}") for i in range(10)]
    table = MgmtQTable().fit(targets)
    assert table.deployed_action(PositionState.SHORT_PUT.value, (0, 0, 1)) \
        == PutMgmtAction.HOLD


def test_learned_policy_interface(sweep_targets):
    table = MgmtQTable().fit(sweep_targets)
    pol = LearnedMgmtPolicy(table)
    a = pol.decide_mgmt(PositionState.SHORT_PUT, (0, 1, 1), {}, None)
    assert isinstance(a, PutMgmtAction)
