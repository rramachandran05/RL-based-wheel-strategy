"""Cash-policy Q-table (SPEC-005 §3-5): per-state regression over sweep
targets, cluster-based effective N, A/B double estimation, rule-baseline
prior with shrinkage, LCB action selection, promotion-ready serialization.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from rlbot.benchmarks.policies import RULE_TABLE
from rlbot.state.enums import CashAction, PositionState, StockAction

N0_PRIOR = 5.0          # prior pseudo-clusters
PRIOR_MARGIN = 0.0005   # rule action prior edge over other tiers
Z_LCB = 1.0
MIN_N_EFF = 3           # below this a cell is UNTRUSTED (falls back to prior)
TABLE_VERSION = "qcash_v1"


def _half_of(cluster_id: str) -> str:
    return "A" if int(hashlib.sha256(cluster_id.encode()).hexdigest(), 16) % 2 == 0 else "B"


class CashQTable:
    def __init__(self):
        # key = (q_state tuple, action int)
        self.sum = defaultdict(float)
        self.sumsq = defaultdict(float)
        self.n = defaultdict(int)
        self.clusters = defaultdict(set)
        self.half_sum = defaultdict(lambda: {"A": 0.0, "B": 0.0})
        self.half_n = defaultdict(lambda: {"A": 0, "B": 0})
        self.prior: dict = {}    # q_state -> prior mean (rule action's mean target)

    # ---------------- training ----------------
    def fit(self, targets: list) -> "CashQTable":
        for t in targets:
            key = (tuple(t.q_state), int(t.action))
            self.sum[key] += t.target
            self.sumsq[key] += t.target ** 2
            self.n[key] += 1
            self.clusters[key].add(t.cluster_id)
            h = _half_of(t.cluster_id)
            self.half_sum[key][h] += t.target
            self.half_n[key][h] += 1
        # prior per state = mean target of the rule policy's tier (SPEC-005 §4)
        for q in {tuple(t.q_state) for t in targets}:
            rule_a = RULE_TABLE[(PositionState.CASH.value, q)]
            key = (q, rule_a)
            self.prior[q] = self.sum[key] / self.n[key] if self.n[key] else 0.0
        return self

    # ---------------- estimates ----------------
    def n_eff(self, q, a) -> int:
        return len(self.clusters[(tuple(q), int(a))])

    def q_mean(self, q, a) -> float:
        key = (tuple(q), int(a))
        return self.sum[key] / self.n[key] if self.n[key] else 0.0

    def q_var(self, q, a) -> float:
        key = (tuple(q), int(a))
        if self.n[key] < 2:
            return 0.0
        m = self.q_mean(q, a)
        return max(self.sumsq[key] / self.n[key] - m * m, 0.0)

    def _prior_value(self, q, a) -> float:
        base = self.prior.get(tuple(q), 0.0)
        rule_a = RULE_TABLE[(PositionState.CASH.value, tuple(q))]
        return base if int(a) == rule_a else base - PRIOR_MARGIN

    def q_shrunk(self, q, a) -> float:
        ne = self.n_eff(q, a)
        return (ne * self.q_mean(q, a) + N0_PRIOR * self._prior_value(q, a)) / (ne + N0_PRIOR)

    def lcb(self, q, a, z: float = Z_LCB) -> float:
        ne = self.n_eff(q, a)
        if ne == 0:
            return self._prior_value(q, a)
        se = math.sqrt(self.q_var(q, a)) / math.sqrt(ne)
        return self.q_shrunk(q, a) - z * se

    def half_mean(self, q, a, half: str) -> float | None:
        key = (tuple(q), int(a))
        n = self.half_n[key][half]
        return self.half_sum[key][half] / n if n else None

    # ---------------- policy ----------------
    def deployed_action(self, q, z: float = Z_LCB) -> CashAction:
        """Pessimistic deployment (SPEC-005 §4, REQ-5.6): the rule action is
        the default; a challenger deploys only when its LCB beats the rule
        action's shrunk estimate. As z → ∞ this converges to the rule table."""
        q = tuple(q)
        rule_a = CashAction(RULE_TABLE[(PositionState.CASH.value, q)])
        bar = self.q_shrunk(q, rule_a)
        challengers = [a for a in CashAction
                       if a != rule_a and self.n_eff(q, a) >= MIN_N_EFF]
        if not challengers:
            return rule_a  # UNTRUSTED state: rule prior acts (SPEC-005 §6)
        best = max(challengers, key=lambda a: (self.lcb(q, a, z), -int(a)))
        return CashAction(best) if self.lcb(q, best, z) > bar else rule_a

    # ---------------- persistence ----------------
    def to_json(self, path: Path, manifest: dict) -> None:
        cells = []
        for (q, a), n in sorted(self.n.items()):
            cells.append({
                "q_state": list(q), "action": a, "n": n,
                "n_eff": len(self.clusters[(q, a)]),
                "mean": self.q_mean(q, a), "var": self.q_var(q, a),
                "shrunk": self.q_shrunk(q, a), "lcb": self.lcb(q, a),
                "half_A": self.half_mean(q, a, "A"),
                "half_B": self.half_mean(q, a, "B"),
            })
        payload = {"table_version": TABLE_VERSION, "manifest": manifest,
                   "prior": {str(k): v for k, v in self.prior.items()},
                   "cells": cells}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=1))


class LearnedCashPolicy:
    """MVP-2 deployed policy: LCB-greedy learned cash actions; stock side
    stays on the B3 rules (management learning is MVP-3)."""

    def __init__(self, table: CashQTable, z: float = Z_LCB):
        self.table = table
        self.z = z

    def decide(self, position_state, q_state, row):
        if position_state == PositionState.CASH:
            return self.table.deployed_action(q_state, self.z)
        return StockAction(RULE_TABLE[(PositionState.LONG_STOCK.value, tuple(q_state))])
