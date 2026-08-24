"""G4 experiment: does per-ticker IV vol-comp improve B3's rules?

A/B on real premiums, paired episodes (same ticker, same start, same chains):
arm A = B3 with the validated market-proxy vol comp; arm B = B3 with
per-ticker ATM-IV percentile vol comp (market fallback where IV missing).
No fitting anywhere — both arms use identical fixed thresholds — so the
whole 2014→present span is a fair comparison, with the 2022 bear reported
separately (the window that motivated the experiment).

Adoption criteria (pass -> daily brief switches to ticker_iv):
  pooled diff (B - A) annualized > 0
  2022 bear-window diff > 0 (the mechanism must show where it should)
  pooled drawdown ratio <= 1.10

Run:  python -m rlbot.evaluation.vol_comp_ab
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from rlbot.benchmarks.policies import AdaptiveRulePolicy
from rlbot.config import RlbotConfig
from rlbot.data.loaders import FrameStore
from rlbot.evaluation.put_gate import require_gate
from rlbot.evaluation.walkforward import EPISODE_TDAYS, _episode_starts
from rlbot.options.historical_source import historical_source_for
from rlbot.simulator.environment import WheelEnv

BEAR_WINDOW = ("2022-01-01", "2022-12-31")


def run_ab():
    gate = require_gate(RlbotConfig())
    cfg_a = RlbotConfig(use_valuation_proxy=True, vol_comp_source="market")
    cfg_b = RlbotConfig(use_valuation_proxy=True, vol_comp_source="ticker_iv")
    store_a, store_b = FrameStore(cfg_a), FrameStore(cfg_b)

    episodes = []
    for t in cfg_a.tickers:
        frame_a, frame_b = store_a.frame(t), store_b.frame(t)
        ps_a = historical_source_for(t, cfg_a, gate["iv_uplift"])
        ps_b = historical_source_for(t, cfg_b, gate["iv_uplift"])
        env_a = WheelEnv(t, frame_a, ps_a, dynamic_sizing=True)
        env_b = WheelEnv(t, frame_b, ps_b, dynamic_sizing=True)
        for start in _episode_starts(frame_a, "2014-01-01", "2026-06-30"):
            idx = frame_a.index
            i0 = idx.get_loc(start)
            i1 = min(i0 + EPISODE_TDAYS, len(idx) - 1)
            if i1 - i0 < 100:
                continue
            end = idx[i1]
            a = env_a.run(AdaptiveRulePolicy(), start, end)
            b = env_b.run(AdaptiveRulePolicy(), start, end)
            years = (end - start).days / 365.25
            episodes.append({
                "ticker": t, "start": str(start.date()),
                "a_ret_ann": float((a.nav.iloc[-1] / a.nav.iloc[0]) ** (1 / years) - 1),
                "b_ret_ann": float((b.nav.iloc[-1] / b.nav.iloc[0]) ** (1 / years) - 1),
                "a_dd": float((a.nav / a.nav.cummax() - 1).min()),
                "b_dd": float((b.nav / b.nav.cummax() - 1).min()),
                "diverged": [d.chosen_action for d in a.decisions]
                             != [d.chosen_action for d in b.decisions],
            })
        print(f"  {t}: {len([e for e in episodes if e['ticker']==t])} episodes", flush=True)
    return episodes


def verdict(episodes):
    diffs = [e["b_ret_ann"] - e["a_ret_ann"] for e in episodes]
    bear = [e["b_ret_ann"] - e["a_ret_ann"] for e in episodes
            if BEAR_WINDOW[0] <= e["start"] <= BEAR_WINDOW[1]]
    dd_ratios = [e["b_dd"] / e["a_dd"] for e in episodes if e["a_dd"] < -1e-9]
    diverged = float(np.mean([e["diverged"] for e in episodes]))
    out = {
        "n_episodes": len(episodes),
        "pct_diverged": diverged,
        "pooled_diff_ann": float(np.mean(diffs)),
        "median_diff_ann": float(np.median(diffs)),
        "bear_2022_diff_ann": float(np.mean(bear)) if bear else None,
        "n_bear_episodes": len(bear),
        "dd_ratio_mean": float(np.mean(dd_ratios)),
        "pct_episodes_positive": float(np.mean([d > 0 for d in diffs])),
    }
    out["criteria"] = {
        "pooled_positive": out["pooled_diff_ann"] > 0,
        "bear_2022_positive": (out["bear_2022_diff_ann"] or 0) > 0,
        "dd_within_1p1x": out["dd_ratio_mean"] <= 1.10,
    }
    out["pass"] = all(out["criteria"].values())
    return out


def main():
    episodes = run_ab()
    v = verdict(episodes)
    cfg = RlbotConfig()
    out = cfg.data.base_path / "reports" / "g4_vol_comp_ab.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"g4": v, "episodes": episodes}, indent=1))
    print(json.dumps({"g4": v}, indent=2))
    return v


if __name__ == "__main__":
    main()
