"""GATE G3 walk-forward: learned management vs hold-to-expiry (SPEC-007 §2.4).

Both arms use Baseline-3 openings; only management differs. M-B2 (the MOS
±3% mechanical roll rule) runs on test windows for context. Criteria:
pooled diff > 0 with fold floor, drawdown ratio ≤ 1.0, pooled monthly CVaR
no worse, A/B halves non-negative.

Run:  python -m rlbot.evaluation.walkforward_g3
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from rlbot.benchmarks.policies import AdaptiveRulePolicy, MosRollMgmtPolicy
from rlbot.config import RlbotConfig
from rlbot.data.loaders import FrameStore
from rlbot.evaluation.put_gate import require_gate
from rlbot.evaluation.walkforward import FOLDS, EPISODE_TDAYS, _episode_starts
from rlbot.learning.mgmt_qtable import MgmtQTable, LearnedMgmtPolicy
from rlbot.learning.mgmt_sweep import sweep_mgmt_ticker
from rlbot.learning.trajectories import decision_to_record, write_jsonl
from rlbot.options.premium_source import SyntheticBSPremiumSource
from rlbot.simulator.environment import WheelEnv

FOLD_MIN_ANN = -0.005
DD_RATIO_MAX = 1.00          # SPEC-007 §2.4: management must not add drawdown
CVAR_TOL = 0.001


def _episode_pair(env, frame, start, mgmt_pol):
    idx = frame.index
    i0 = idx.get_loc(start)
    i1 = min(i0 + EPISODE_TDAYS, len(idx) - 1)
    if i1 - i0 < 100:
        return None
    end = idx[i1]
    pol = env.run(AdaptiveRulePolicy(), start, end, mgmt_policy=mgmt_pol)
    ref = env.run(AdaptiveRulePolicy(), start, end, mgmt_policy=None)
    years = (end - start).days / 365.25
    diff_ann = (pol.nav.iloc[-1] / pol.nav.iloc[0]) ** (1 / years) \
        - (ref.nav.iloc[-1] / ref.nav.iloc[0]) ** (1 / years)
    return pol, ref, float(diff_ann)


def _dd(nav: pd.Series) -> float:
    return float((nav / nav.cummax() - 1).min())


def _pooled_cvar(navs: list) -> float | None:
    monthly = pd.concat([n.resample("ME").last().pct_change().dropna() for n in navs])
    if len(monthly) < 20:
        return None
    return float(monthly[monthly <= monthly.quantile(0.05)].mean())


def _ab_check(table: MgmtQTable) -> dict:
    from rlbot.learning.mgmt_qtable import _hold_for
    diffs = {"A": [], "B": []}
    for state in table.states:
        deployed = table.deployed_action(state[0], state[1])
        hold = _hold_for(state[0])
        if int(deployed) == int(hold):
            continue
        for h in ("A", "B"):
            dm = table.half_mean(state, deployed, h)
            if dm is not None:
                diffs[h].append(dm)      # targets are already vs-HOLD
    return {h: (float(np.mean(v)) if v else None) for h, v in diffs.items()}


def run_fold(fold, store, ps, cfg, traj_dir):
    print(f"--- G3 fold {fold['name']}: management sweep ---")
    targets = []
    for t in cfg.tickers:
        tt = sweep_mgmt_ticker(t, store.frame(t), ps, *fold["train"])
        targets.extend(tt)
        print(f"  {t}: {len(tt)} mgmt targets")
    table = MgmtQTable().fit(targets)
    deviating = sum(
        1 for s in table.states
        if int(table.deployed_action(s[0], s[1])) != 0
    )
    print(f"  states: {len(table.states)}, deviating from HOLD: {deviating}")
    ab = _ab_check(table)

    results = {}
    for split in ("val", "test"):
        diffs, dd_ratios, records = [], [], []
        pol_navs, ref_navs, mb2_diffs = [], [], []
        for t in cfg.tickers:
            frame = store.frame(t)
            env = WheelEnv(t, frame, ps)
            for start in _episode_starts(frame, *fold[split]):
                pair = _episode_pair(env, frame, start, LearnedMgmtPolicy(table))
                if pair is None:
                    continue
                pol, ref, diff_ann = pair
                diffs.append(diff_ann)
                pol_navs.append(pol.nav)
                ref_navs.append(ref.nav)
                if _dd(ref.nav) < -1e-9:
                    dd_ratios.append(_dd(pol.nav) / _dd(ref.nav))
                ep_id = f"g3-{fold['name']}-{split}-{t}-{start.date()}"
                for i, d in enumerate(pol.decisions):
                    records.append(decision_to_record(
                        d, t, f"g3-{fold['name']}", ep_id, i,
                        schema_version="trajectory_v2"))
                if split == "test":
                    mb2 = _episode_pair(env, frame, start, MosRollMgmtPolicy())
                    if mb2 is not None:
                        mb2_diffs.append(mb2[2])
        write_jsonl(records, traj_dir / f"g3_{fold['name']}_{split}.jsonl")
        results[split] = {
            "n_episodes": len(diffs),
            "mean_diff_ann": float(np.mean(diffs)) if diffs else None,
            "median_diff_ann": float(np.median(diffs)) if diffs else None,
            "pct_episodes_positive": float(np.mean([d > 0 for d in diffs])) if diffs else None,
            "dd_ratio_mean": float(np.mean(dd_ratios)) if dd_ratios else None,
            "cvar_pol": _pooled_cvar(pol_navs),
            "cvar_ref": _pooled_cvar(ref_navs),
            "mb2_mean_diff_ann": float(np.mean(mb2_diffs)) if mb2_diffs else None,
        }
        print(f"  {split}: {results[split]['n_episodes']} episodes, "
              f"mean diff {results[split]['mean_diff_ann']:+.4f}/yr")
    return {"fold": fold["name"], "n_states": len(table.states),
            "n_deviating": deviating, "ab_check": ab, **results}, table


def g3_verdict(folds: list) -> dict:
    tests = [f["test"]["mean_diff_ann"] for f in folds]
    pooled = float(np.mean(tests))
    dd_ok = all((f["test"]["dd_ratio_mean"] or 0) <= DD_RATIO_MAX for f in folds)
    cvar_ok = all(
        f["test"]["cvar_pol"] is None or f["test"]["cvar_ref"] is None
        or f["test"]["cvar_pol"] >= f["test"]["cvar_ref"] - CVAR_TOL
        for f in folds
    )
    ab_ok = all((v is None or v >= 0) for f in folds for v in f["ab_check"].values())
    criteria = {
        "pooled_diff_ann_positive": pooled > 0,
        "folds_positive": sum(x > 0 for x in tests) >= 1 and min(tests) >= FOLD_MIN_ANN,
        "drawdown_within_1p0x": dd_ok,
        "cvar_no_worse": cvar_ok,
        "ab_halves_non_negative": ab_ok,
    }
    return {"pass": all(criteria.values()), "pooled_test_diff_ann": pooled,
            "per_fold_test_diff_ann": tests, "criteria": criteria}


def main():
    cfg = RlbotConfig()
    gate = require_gate(cfg)                                 # REQ-7.5
    ps = SyntheticBSPremiumSource(iv_uplift=gate["iv_uplift"])
    store = FrameStore(cfg)
    traj_dir = cfg.data.base_path / "trajectories"
    fold_results = []
    for f in FOLDS:
        result, table = run_fold(f, store, ps, cfg, traj_dir)
        table.to_json(cfg.data.base_path / "tables" / f"qmgmt_{f['name']}.json",
                      {"fold": f, "gate_iv_uplift": gate["iv_uplift"]})
        fold_results.append(result)
    verdict = g3_verdict(fold_results)
    out = {"g1": {"iv_uplift": gate["iv_uplift"], "pass": gate["pass"]},
           "g3": verdict, "folds": fold_results,
           "disclaimer": ("Survivorship scope (DATA-GAP-5) applies. Openings "
                           "fixed to Baseline 3 in both arms; management only.")}
    report_dir = cfg.data.base_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "g3_verdict.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"g3": verdict}, indent=2))
    return out


if __name__ == "__main__":
    main()
