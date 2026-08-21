"""Walk-forward training + evaluation and the GATE G2 verdict (SPEC-006 §4-5).

Run:  python -m rlbot.evaluation.walkforward

Fold dates are adapted to the available snapshot (ticker bars begin 2011-08,
indicator warmup ~1y): two expanding folds instead of the spec's three —
the 2-of-3-folds criterion becomes "pooled > 0, at least one fold > 0, and
no fold below −0.5pt annualized" (documented adaptation).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rlbot.benchmarks.policies import AdaptiveRulePolicy, FixedWheelPolicy
from rlbot.config import RlbotConfig
from rlbot.data.loaders import FrameStore
from rlbot.evaluation.metrics import nav_metrics
from rlbot.evaluation.put_gate import require_gate
from rlbot.learning.qtable import CashQTable, LearnedCashPolicy
from rlbot.learning.sweep import sweep_ticker
from rlbot.learning.trajectories import decision_to_record, write_jsonl
from rlbot.options.premium_source import SyntheticBSPremiumSource
from rlbot.simulator.environment import WheelEnv, attach_rewards
from rlbot.state.enums import CashAction, PositionState

FOLDS = [
    {"name": "F1", "train": ("2013-01-01", "2019-12-31"),
     "val": ("2020-01-01", "2021-12-31"), "test": ("2022-01-01", "2023-12-31")},
    {"name": "F2", "train": ("2013-01-01", "2021-12-31"),
     "val": ("2022-01-01", "2023-12-31"), "test": ("2024-01-01", "2026-06-30")},
]
EPISODE_TDAYS = 252
FOLD_MIN_ANN = -0.005     # no fold below −0.5pt annualized
REGIME_SEG_MAX_LAG = -0.02
DD_RATIO_MAX = 1.10


@dataclass
class EpisodeEval:
    ticker: str
    start: str
    pol: dict
    ref: dict
    diff_ann: float
    pol_nav: pd.Series
    ref_nav: pd.Series


def _episode_starts(frame: pd.DataFrame, win_start, win_end) -> list:
    idx = frame.loc[pd.Timestamp(win_start):pd.Timestamp(win_end)].index
    if len(idx) < 60:
        return []
    starts, seen = [], set()
    for d in idx:
        key = (d.year, d.quarter)
        if key not in seen:
            seen.add(key)
            starts.append(d)
    return starts


def _run_pair(env: WheelEnv, policy, ref_policy, start, frame) -> EpisodeEval | None:
    idx = frame.index
    i0 = idx.get_loc(start)
    i1 = min(i0 + EPISODE_TDAYS, len(idx) - 1)
    if i1 - i0 < 100:
        return None
    end = idx[i1]
    pol = env.run(policy, start, end)
    ref = env.run(ref_policy, start, end)
    years = (end - start).days / 365.25
    pol_ret = pol.nav.iloc[-1] / pol.nav.iloc[0]
    ref_ret = ref.nav.iloc[-1] / ref.nav.iloc[0]
    diff_ann = pol_ret ** (1 / years) - ref_ret ** (1 / years)
    return EpisodeEval(env.ticker, str(start.date()), nav_metrics(pol.nav),
                       nav_metrics(ref.nav), float(diff_ann), pol.nav, ref.nav), pol


def _regime_segments(evals: list, store: FrameStore) -> dict:
    market = store.tables["market"]["market_regime"]
    daily = []
    for e in evals:
        d = (e.pol_nav.pct_change() - e.ref_nav.pct_change()).dropna()
        daily.append(d)
    if not daily:
        return {}
    diff = pd.concat(daily).groupby(level=0).mean()
    regs = market.reindex(diff.index)
    out = {}
    for r in [0, 1, 2, 3]:
        seg = diff[regs == r]
        if len(seg) >= 20:
            out[str(r)] = float(seg.mean() * 252)
    return out


def _ab_check(table: CashQTable) -> dict:
    """SPEC-005: deployed-vs-rule advantage must be non-negative on both halves."""
    from rlbot.benchmarks.policies import RULE_TABLE
    diffs = {"A": [], "B": []}
    for q in table.prior:
        deployed = int(table.deployed_action(q))
        rule = RULE_TABLE[(PositionState.CASH.value, q)]
        if deployed == rule:
            continue
        for h in ("A", "B"):
            dm, rm = table.half_mean(q, deployed, h), table.half_mean(q, rule, h)
            if dm is not None and rm is not None:
                diffs[h].append(dm - rm)
    return {h: (float(np.mean(v)) if v else None) for h, v in diffs.items()}


def run_fold(fold: dict, store: FrameStore, ps, cfg: RlbotConfig, traj_dir) -> dict:
    print(f"--- fold {fold['name']}: training sweep ---")
    targets = []
    for t in cfg.tickers:
        tt = sweep_ticker(t, store.frame(t), ps, *fold["train"])
        targets.extend(tt)
        print(f"  {t}: {len(tt)} targets")
    table = CashQTable().fit(targets)

    coverage = {str(q): {int(a): table.n_eff(q, a) for a in CashAction}
                for q in sorted(table.prior)}
    ab = _ab_check(table)

    results = {}
    for split in ("val", "test"):
        evals, records = [], []
        for t in cfg.tickers:
            frame = store.frame(t)
            env = WheelEnv(t, frame, ps)
            for start in _episode_starts(frame, *fold[split]):
                pair = _run_pair(env, LearnedCashPolicy(table), AdaptiveRulePolicy(),
                                 start, frame)
                if pair is None:
                    continue
                ev, pol_result = pair
                evals.append(ev)
                b1 = env.run(FixedWheelPolicy(), start, ev.pol_nav.index[-1])
                attach_rewards(pol_result, b1.nav, frame)
                ep_id = f"{fold['name']}-{split}-{t}-{ev.start}"
                for i, d in enumerate(pol_result.decisions):
                    records.append(decision_to_record(d, t, f"wf-{fold['name']}", ep_id, i))
        write_jsonl(records, traj_dir / f"{fold['name']}_{split}.jsonl")
        diffs = [e.diff_ann for e in evals]
        dd_ratios = [e.pol["max_drawdown"] / e.ref["max_drawdown"]
                     for e in evals if e.ref["max_drawdown"] < -1e-9]
        results[split] = {
            "n_episodes": len(evals),
            "mean_diff_ann": float(np.mean(diffs)) if diffs else None,
            "median_diff_ann": float(np.median(diffs)) if diffs else None,
            "pct_episodes_positive": float(np.mean([d > 0 for d in diffs])) if diffs else None,
            "dd_ratio_mean": float(np.mean(dd_ratios)) if dd_ratios else None,
            "regime_segments_ann_diff": _regime_segments(evals, store),
        }
        print(f"  {split}: {results[split]['n_episodes']} episodes, "
              f"mean diff {results[split]['mean_diff_ann']:+.4f}/yr")
    return {"fold": fold["name"], "coverage": coverage, "ab_check": ab, **results}


def g2_verdict(folds: list) -> dict:
    tests = [f["test"]["mean_diff_ann"] for f in folds]
    pooled = float(np.mean(tests))
    dd_ok = all((f["test"]["dd_ratio_mean"] or 0) <= DD_RATIO_MAX for f in folds)
    seg_ok = all(v >= REGIME_SEG_MAX_LAG
                 for f in folds for v in f["test"]["regime_segments_ann_diff"].values())
    ab_ok = all((v is None or v >= 0)
                for f in folds for v in f["ab_check"].values())
    criteria = {
        "pooled_diff_ann_positive": pooled > 0,
        "folds_positive": sum(t > 0 for t in tests) >= 1 and min(tests) >= FOLD_MIN_ANN,
        "drawdown_within_1p1x": dd_ok,
        "no_regime_segment_lagging_2pts": seg_ok,
        "ab_halves_non_negative": ab_ok,
    }
    return {"pass": all(criteria.values()), "pooled_test_diff_ann": pooled,
            "per_fold_test_diff_ann": tests, "criteria": criteria}


def main():
    cfg = RlbotConfig()
    gate = require_gate(cfg)                       # GATE G1 enforced (AC-5)
    ps = SyntheticBSPremiumSource(iv_uplift=gate["iv_uplift"])
    store = FrameStore(cfg)
    traj_dir = cfg.data.base_path / "trajectories"
    fold_results = [run_fold(f, store, ps, cfg, traj_dir) for f in FOLDS]
    verdict = g2_verdict(fold_results)
    out = {"g1": {"iv_uplift": gate["iv_uplift"], "pass": gate["pass"]},
           "g2": verdict, "folds": fold_results,
           "disclaimer": ("Survivorship scope (DATA-GAP-5): universe is the "
                           "'willing to own' screen's survivors; results are "
                           "conditional on that screen.")}
    report_dir = cfg.data.base_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "g2_verdict.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"g2": verdict}, indent=2))
    return out


if __name__ == "__main__":
    main()
