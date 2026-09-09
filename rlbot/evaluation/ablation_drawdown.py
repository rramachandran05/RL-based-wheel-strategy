"""Ablation G9 (SPEC-007 §3C): drawdown-percentile replaces valuation in
the Q-state.

Question: does a cash policy conditioned on regime × DRAWDOWN-PERCENTILE ×
vol-comp beat its own rule baseline where the FV axis (G2), the EPS proxy
(G2-rerun) and the trend axis (ablation) could not?

Mechanics: identical walk-forward pipeline; the frame's `valuation_state`
column is overwritten with `classify_drawdown_series(frame["drawdown"])` —
the rolling 5-year percentile of the ticker's OWN drawdown (worst fifth →
ATTRACTIVE, top fifth → EXPENSIVE, else FAIR), so the B3 rule table's
slot-1 semantics stay coherent (0 → aggressive … 2 → conservative: buy the
deep correction, stand back near highs). Zero historical gap by
construction. Everything else — sweep, estimator, LCB deployment, folds,
criteria — is byte-identical to the G2 run and the trend ablation.

Pre-registered failure mode (SPEC-007 §3C.3): value traps — deep-drawdown
ATTRACTIVE states buying into names that keep falling. The verdict JSON
carries per-fold loss segmentation by entry valuation_state for that check.

Run:  python -m rlbot.evaluation.ablation_drawdown
"""
from __future__ import annotations

import json

from rlbot.config import RlbotConfig
from rlbot.data.loaders import FrameStore
from rlbot.evaluation.put_gate import require_gate
from rlbot.evaluation.walkforward import FOLDS, g2_verdict, run_fold
from rlbot.features.valuation import classify_drawdown_series
from rlbot.options.premium_source import SyntheticBSPremiumSource


class DrawdownFrameStore:
    """FrameStore wrapper: valuation slot carries the drawdown percentile."""

    def __init__(self, cfg: RlbotConfig):
        self.inner = FrameStore(cfg)
        self.tables = self.inner.tables
        self._cache: dict = {}

    def frame(self, ticker: str):
        if ticker not in self._cache:
            f = self.inner.frame(ticker).copy()
            f["valuation_state"] = classify_drawdown_series(f["drawdown"])
            self._cache[ticker] = f
        return self._cache[ticker]


VAL_NAMES = {0: "ATTRACTIVE", 1: "FAIR", 2: "EXPENSIVE"}


def value_trap_segmentation(traj_dir, folds=("F1", "F2")) -> dict:
    """SPEC-007 §3C.3 pre-registered check: test-period CASH-decision outcomes
    by ENTRY valuation_state under the drawdown axis. The failure mode being
    hunted is deep-drawdown ATTRACTIVE states that keep losing (buying into
    names that keep falling) — 'cheaper than it was' is not 'cheaper than it
    is worth'."""
    out = {}
    for fold in folds:
        path = traj_dir / f"{fold}_test.jsonl"
        if not path.exists():
            continue
        agg = {s: {"n": 0, "sum": 0.0, "neg": 0, "sold": 0} for s in VAL_NAMES.values()}
        for line in open(path):
            r = json.loads(line)
            if r.get("position_state") != "CASH" or r.get("reward") is None:
                continue
            a = agg[VAL_NAMES[r["q_state"][1]]]
            a["n"] += 1; a["sum"] += r["reward"]
            a["neg"] += r["reward"] < 0; a["sold"] += r["chosen_action"] != 0
        out[fold] = {s: {"n": a["n"],
                         "mean_diff_reward": a["sum"] / a["n"] if a["n"] else None,
                         "loss_share": a["neg"] / a["n"] if a["n"] else None,
                         "sold_share": a["sold"] / a["n"] if a["n"] else None}
                     for s, a in agg.items()}
    return out


def coverage_summary(fold_results) -> dict:
    """Cells populated of 36 per fold (coverage maps state -> {action: n})."""
    return {f["fold"]: sum(1 for c in f["coverage"].values() if sum(c.values()) > 0)
            for f in fold_results}


def main():
    cfg = RlbotConfig(use_valuation_proxy=False)   # axis is overwritten anyway
    gate = require_gate(cfg)
    ps = SyntheticBSPremiumSource(iv_uplift=gate["iv_uplift"])
    store = DrawdownFrameStore(cfg)
    traj_dir = cfg.data.base_path / "trajectories_ablation_drawdown"
    fold_results = [run_fold(f, store, ps, cfg, traj_dir) for f in FOLDS]
    verdict = g2_verdict(fold_results)
    out = {"ablation": "drawdown_percentile_replaces_valuation",
           "mapping": {"dd_pct < 0.20": "ATTRACTIVE", "dd_pct > 0.80": "EXPENSIVE",
                       "else": "FAIR", "window": 1260, "min_obs": 252},
           "g1": {"iv_uplift": gate["iv_uplift"]},
           "verdict": verdict, "folds": fold_results,
           "cells_populated_of_36": coverage_summary(fold_results),
           "value_trap_segmentation": value_trap_segmentation(traj_dir),
           "disclaimer": ("Survivorship scope (DATA-GAP-5) applies. "
                          "'Cheaper than it was' is not 'cheaper than it is "
                          "worth' — inspect the value-trap segmentation before "
                          "reading a pass as adoptable.")}
    report_dir = cfg.data.base_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "ablation_drawdown_verdict.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(verdict, indent=2))
    return out


if __name__ == "__main__":
    main()
