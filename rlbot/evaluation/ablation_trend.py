"""Ablation (SPEC-001 §3.2 gate): trend replaces valuation in the Q-state.

Question: does a cash policy conditioned on regime × TREND × vol-comp beat its
own rule baseline where the valuation axis could not (G2, G2-rerun)?

Mechanics: identical walk-forward pipeline; the frame's `valuation_state`
column is overwritten with a 3-level collapse of `trend_bucket`
(Bull Trend→0, Recovery/Base→1, Pullback/Breakdown→2), chosen so the B3 rule
table's slot-1 semantics stay coherent (0 → aggressive … 2 → conservative,
i.e. a momentum-following rule prior). Everything else — sweep, estimator,
LCB deployment, folds, criteria — is byte-identical to the G2 run.

Run:  python -m rlbot.evaluation.ablation_trend
"""
from __future__ import annotations

import json

from rlbot.config import RlbotConfig
from rlbot.data.loaders import FrameStore
from rlbot.evaluation.put_gate import require_gate
from rlbot.evaluation.walkforward import FOLDS, g2_verdict, run_fold
from rlbot.options.premium_source import SyntheticBSPremiumSource

# trend_bucket (0-4) -> slot values matching B3's aggressive..conservative order
TREND_TO_SLOT = {4: 0, 3: 1, 2: 1, 1: 2, 0: 2}


class TrendFrameStore:
    """FrameStore wrapper: valuation slot carries the trend bucket."""

    def __init__(self, cfg: RlbotConfig):
        self.inner = FrameStore(cfg)
        self.tables = self.inner.tables
        self._cache: dict = {}

    def frame(self, ticker: str):
        if ticker not in self._cache:
            f = self.inner.frame(ticker).copy()
            f["valuation_state"] = f["trend_bucket"].map(TREND_TO_SLOT).astype("Int8")
            self._cache[ticker] = f
        return self._cache[ticker]


def main():
    cfg = RlbotConfig(use_valuation_proxy=False)
    gate = require_gate(cfg)
    ps = SyntheticBSPremiumSource(iv_uplift=gate["iv_uplift"])
    store = TrendFrameStore(cfg)
    traj_dir = cfg.data.base_path / "trajectories_ablation_trend"
    fold_results = [run_fold(f, store, ps, cfg, traj_dir) for f in FOLDS]
    verdict = g2_verdict(fold_results)
    out = {"ablation": "trend_replaces_valuation",
           "trend_to_slot": {str(k): v for k, v in TREND_TO_SLOT.items()},
           "g1": {"iv_uplift": gate["iv_uplift"]},
           "verdict": verdict, "folds": fold_results,
           "disclaimer": "Survivorship scope (DATA-GAP-5) applies."}
    report_dir = cfg.data.base_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "ablation_trend_verdict.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"ablation_trend": verdict}, indent=2))
    return out


if __name__ == "__main__":
    main()
