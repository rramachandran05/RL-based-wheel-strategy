"""SPEC-005 tests: sweep engine, Q-table estimator math, pessimism limits."""
import numpy as np
import pandas as pd
import pytest

from rlbot.benchmarks.policies import RULE_TABLE
from rlbot.learning.qtable import MIN_N_EFF, CashQTable, LearnedCashPolicy
from rlbot.learning.sweep import SweepTarget, simulate_branch, sweep_ticker
from rlbot.options.premium_source import SyntheticBSPremiumSource
from rlbot.state.enums import CashAction, PositionState
from tests.test_environment import make_frame

PS = SyntheticBSPremiumSource(iv_uplift=0.1)
Q = (0, 1, 1)  # BULL_LOW_VOL / FAIR / NORMAL — rule action = PUT_BALANCED


def _targets(action, values, cluster_prefix="c"):
    return [SweepTarget("T", pd.Timestamp("2021-01-04"), Q, int(action), v, v, 0.0,
                        "CASH", f"{cluster_prefix}{i}") for i, v in enumerate(values)]


def test_estimator_math():
    vals = [0.01, 0.02, 0.03, 0.04]
    table = CashQTable().fit(_targets(CashAction.PUT_BALANCED, vals))
    assert table.q_mean(Q, CashAction.PUT_BALANCED) == pytest.approx(np.mean(vals))
    assert table.q_var(Q, CashAction.PUT_BALANCED) == pytest.approx(np.var(vals))
    assert table.n_eff(Q, CashAction.PUT_BALANCED) == 4  # distinct clusters


def test_n_eff_counts_clusters_not_rows():
    same_cluster = [SweepTarget("T", pd.Timestamp("2021-01-04"), Q, 3, v, v, 0.0,
                                "CASH", "one-cluster") for v in [0.01, 0.02, 0.05]]
    table = CashQTable().fit(same_cluster)
    assert table.n[(Q, 3)] == 3
    assert table.n_eff(Q, 3) == 1  # REQ-5.4


def test_z_infinity_converges_to_rule():
    """REQ-5.6: with huge z the deployed policy IS the rule table."""
    rng = np.random.default_rng(0)
    targets = []
    for a in CashAction:
        targets += _targets(a, list(rng.normal(0.01 * int(a), 0.01, 8)), f"a{int(a)}-")
    table = CashQTable().fit(targets)
    rule_a = RULE_TABLE[(PositionState.CASH.value, Q)]
    assert int(table.deployed_action(Q, z=1e6)) == rule_a


def test_untrusted_cell_falls_back_to_rule():
    table = CashQTable().fit(_targets(CashAction.PUT_BALANCED, [0.01, 0.02]))
    # challenger has huge mean but only appears in MIN_N_EFF-1 clusters
    few = _targets(CashAction.PUT_VERY_AGGRESSIVE, [0.5] * (MIN_N_EFF - 1), "x")
    table = CashQTable().fit(_targets(CashAction.PUT_BALANCED, [0.01, 0.02, 0.03]) + few)
    rule_a = RULE_TABLE[(PositionState.CASH.value, Q)]
    assert int(table.deployed_action(Q)) == rule_a


def test_strong_challenger_deploys():
    targets = _targets(CashAction.PUT_BALANCED, [0.001] * 10, "r")
    targets += _targets(CashAction.PUT_CONSERVATIVE, [0.05] * 10, "ch")
    table = CashQTable().fit(targets)
    assert table.deployed_action(Q, z=1.0) == CashAction.PUT_CONSERVATIVE


def test_ab_halves_populated():
    rng = np.random.default_rng(1)
    targets = _targets(CashAction.PUT_BALANCED, list(rng.normal(0.01, 0.005, 30)))
    table = CashQTable().fit(targets)
    a = table.half_mean(Q, CashAction.PUT_BALANCED, "A")
    b = table.half_mean(Q, CashAction.PUT_BALANCED, "B")
    assert a is not None and b is not None


# ---------------- sweep on a synthetic frame ----------------
def test_wait_branch_is_exactly_zero():
    """SPEC-005 REQ-5.2 / SPEC-001 AC-5."""
    frame = make_frame(n=200)
    end, state, ret = simulate_branch(frame, PS, frame.index[10], CashAction.WAIT, 1, 10)
    assert ret == 0.0 and state == "CASH"


def test_sweep_produces_all_action_targets():
    frame = make_frame(n=420, seed=9)
    targets = sweep_ticker("T", frame, PS, frame.index[0], frame.index[-1], cadence=20)
    assert targets
    by_epoch = {}
    for t in targets:
        by_epoch.setdefault(t.date, set()).add(t.action)
    # every swept epoch covers WAIT plus most put tiers (some tiers may be
    # unimplementable on a given chain, but never fewer than 4 actions)
    for date, actions in by_epoch.items():
        assert 0 in actions, f"WAIT missing at {date}"
        assert len(actions) >= 4
    # targets are finite and reasonably bounded
    vals = [t.target for t in targets]
    assert all(np.isfinite(v) for v in vals)
    assert max(abs(v) for v in vals) < 1.0


def test_learned_policy_interface():
    frame = make_frame(n=420, seed=9)
    targets = sweep_ticker("T", frame, PS, frame.index[0], frame.index[-1], cadence=20)
    policy = LearnedCashPolicy(CashQTable().fit(targets))
    a = policy.decide(PositionState.CASH, Q, frame.iloc[-1])
    assert isinstance(a, CashAction)
