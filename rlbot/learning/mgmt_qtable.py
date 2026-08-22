"""Management Q-table (SPEC-007 §2.3): same estimator machinery as the cash
table, with HOLD as the prior/incumbent. All diff_v2 targets are relative to
HOLD, so the prior value of HOLD is exactly 0 and a challenger deploys only
when its LCB clears 0 with enough independent evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from rlbot.state.enums import CallMgmtAction, PositionState, PutMgmtAction

N0_PRIOR = 5.0
PRIOR_MARGIN = 0.0005
Z_LCB = 1.0
MIN_N_EFF = 3
TABLE_VERSION = "qmgmt_v1"


def _half_of(cluster_id: str) -> str:
    return "A" if int(hashlib.sha256(cluster_id.encode()).hexdigest(), 16) % 2 == 0 else "B"


def _hold_for(pos_value: str):
    return PutMgmtAction.HOLD if pos_value == PositionState.SHORT_PUT.value \
        else CallMgmtAction.HOLD


def _action_cls(pos_value: str):
    return PutMgmtAction if pos_value == PositionState.SHORT_PUT.value else CallMgmtAction


class MgmtQTable:
    def __init__(self):
        # key = ((pos_value, m_state tuple), action int)
        self.sum = defaultdict(float)
        self.sumsq = defaultdict(float)
        self.n = defaultdict(int)
        self.clusters = defaultdict(set)
        self.half_sum = defaultdict(lambda: {"A": 0.0, "B": 0.0})
        self.half_n = defaultdict(lambda: {"A": 0, "B": 0})
        self.states: set = set()

    def fit(self, targets: list) -> "MgmtQTable":
        for t in targets:
            state = (t.pos_state, tuple(t.m_state))
            key = (state, int(t.action))
            self.states.add(state)
            self.sum[key] += t.target
            self.sumsq[key] += t.target ** 2
            self.n[key] += 1
            self.clusters[key].add(t.cluster_id)
            h = _half_of(t.cluster_id)
            self.half_sum[key][h] += t.target
            self.half_n[key][h] += 1
        return self

    def n_eff(self, state, a) -> int:
        return len(self.clusters[(state, int(a))])

    def q_mean(self, state, a) -> float:
        key = (state, int(a))
        return self.sum[key] / self.n[key] if self.n[key] else 0.0

    def q_var(self, state, a) -> float:
        key = (state, int(a))
        if self.n[key] < 2:
            return 0.0
        m = self.q_mean(state, a)
        return max(self.sumsq[key] / self.n[key] - m * m, 0.0)

    def _prior_value(self, state, a) -> float:
        hold = _hold_for(state[0])
        return 0.0 if int(a) == int(hold) else -PRIOR_MARGIN

    def q_shrunk(self, state, a) -> float:
        ne = self.n_eff(state, a)
        return (ne * self.q_mean(state, a) + N0_PRIOR * self._prior_value(state, a)) \
            / (ne + N0_PRIOR)

    def lcb(self, state, a, z: float = Z_LCB) -> float:
        ne = self.n_eff(state, a)
        if ne == 0:
            return self._prior_value(state, a)
        se = math.sqrt(self.q_var(state, a)) / math.sqrt(ne)
        return self.q_shrunk(state, a) - z * se

    def half_mean(self, state, a, half: str) -> float | None:
        key = (state, int(a))
        n = self.half_n[key][half]
        return self.half_sum[key][half] / n if n else None

    def deployed_action(self, pos_value: str, m_state, z: float = Z_LCB):
        """HOLD unless a challenger's LCB clears HOLD's shrunk value (= ~0)
        with MIN_N_EFF independent clusters. z → ∞ converges to HOLD-always."""
        state = (pos_value, tuple(m_state))
        cls = _action_cls(pos_value)
        hold = _hold_for(pos_value)
        bar = self.q_shrunk(state, hold)      # ≈ 0 by construction
        challengers = [a for a in cls
                       if a != hold and self.n_eff(state, a) >= MIN_N_EFF]
        if not challengers:
            return hold
        best = max(challengers, key=lambda a: (self.lcb(state, a, z), -int(a)))
        return best if self.lcb(state, best, z) > bar else hold

    def to_json(self, path: Path, manifest: dict) -> None:
        cells = []
        for (state, a), n in sorted(self.n.items(), key=lambda kv: (str(kv[0][0][0]), kv[0][0][1], kv[0][1])):
            cells.append({
                "pos_state": state[0], "m_state": list(state[1]), "action": a,
                "n": n, "n_eff": len(self.clusters[(state, a)]),
                "mean": self.q_mean(state, a), "var": self.q_var(state, a),
                "shrunk": self.q_shrunk(state, a), "lcb": self.lcb(state, a),
                "half_A": self.half_mean(state, a, "A"),
                "half_B": self.half_mean(state, a, "B"),
            })
        payload = {"table_version": TABLE_VERSION, "manifest": manifest, "cells": cells}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=1))


class LearnedMgmtPolicy:
    def __init__(self, table: MgmtQTable, z: float = Z_LCB):
        self.table = table
        self.z = z

    def decide_mgmt(self, position_state, m_state, ctx, row):
        if m_state is None:
            return _hold_for(position_state.value)
        return self.table.deployed_action(position_state.value, m_state, self.z)
