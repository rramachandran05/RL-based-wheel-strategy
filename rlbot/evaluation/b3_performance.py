"""Absolute performance of the operational rule policy (B3 + hold-to-expiry).

Answers "what does B3 actually return?" in dollars: fresh $100K per single-
ticker sleeve, per window; pooled = equal-weight average of the 10 sleeve NAV
curves (a $100K account split across the sleeves). Context columns: B1 fixed
20-delta wheel and buy-and-hold of the same tickers.

Run:  python -m rlbot.evaluation.b3_performance
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from rlbot.benchmarks.policies import AdaptiveRulePolicy, FixedWheelPolicy
from rlbot.config import RlbotConfig
from rlbot.data.loaders import FrameStore
from rlbot.evaluation.metrics import nav_metrics
from rlbot.evaluation.put_gate import require_gate
from rlbot.options.premium_source import SyntheticBSPremiumSource
from rlbot.simulator.environment import WheelEnv

WINDOWS = {
    "Full 2013-2026": ("2013-01-01", "2026-08-21"),
    "Test-1 2022-2023": ("2022-01-01", "2023-12-31"),
    "Test-2 2024-2026": ("2024-01-01", "2026-08-21"),
}
START_CASH = 100_000.0


def _bh_nav(frame: pd.DataFrame, start, end) -> pd.Series:
    close = frame["close"].loc[pd.Timestamp(start):pd.Timestamp(end)]
    return close / close.iloc[0] * START_CASH


def run_window(store, ps, cfg, start, end) -> dict:
    navs = {"B3": [], "B1": [], "BH": []}
    per_ticker = {}
    for t in cfg.tickers:
        frame = store.frame(t)
        env = WheelEnv(t, frame, ps, dynamic_sizing=True)   # full deployment
        b3 = env.run(AdaptiveRulePolicy(), start, end, starting_cash=START_CASH)
        b1 = env.run(FixedWheelPolicy(), start, end, starting_cash=START_CASH)
        bh = _bh_nav(frame, start, end)
        navs["B3"].append(b3.nav)
        navs["B1"].append(b1.nav)
        navs["BH"].append(bh)
        m = nav_metrics(b3.nav)
        per_ticker[t] = {"cagr": m["cagr"], "max_dd": m["max_drawdown"],
                          "final": float(b3.nav.iloc[-1]),
                          "bh_cagr": nav_metrics(bh)["cagr"]}
    pooled = {}
    for name, series_list in navs.items():
        curve = pd.concat(series_list, axis=1).ffill().mean(axis=1)
        m = nav_metrics(curve)
        pooled[name] = {"cagr": m["cagr"], "ann_vol": m["ann_vol"],
                         "sharpe": m["sharpe"], "max_dd": m["max_drawdown"],
                         "final": float(curve.iloc[-1])}
    return {"pooled": pooled, "per_ticker": per_ticker}


def main():
    cfg = RlbotConfig(use_valuation_proxy=True)
    gate = require_gate(cfg)
    ps = SyntheticBSPremiumSource(iv_uplift=gate["iv_uplift"])
    store = FrameStore(cfg)
    results = {}
    for name, (start, end) in WINDOWS.items():
        print(f"--- {name} ---")
        results[name] = run_window(store, ps, cfg, start, end)
        p = results[name]["pooled"]
        for k in ("B3", "B1", "BH"):
            print(f"  {k}: CAGR {p[k]['cagr']:+.2%}  maxDD {p[k]['max_dd']:.2%}  "
                  f"$100K -> ${p[k]['final']:,.0f}")
    out = cfg.data.base_path / "reports" / "b3_performance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"windows": results, "start_cash": START_CASH,
         "notes": ["synthetic-BS premiums (G1-calibrated, iv_uplift={:.2f})".format(gate["iv_uplift"]),
                    "pooled = equal-weight average of 10 single-ticker sleeves",
                    "survivorship scope DATA-GAP-5 applies", "not investment advice"]},
        indent=2))
    print(f"wrote {out}")
    return results


if __name__ == "__main__":
    main()
