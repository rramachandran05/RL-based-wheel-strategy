"""B3 total-return decomposition: premium income vs stock P&L.

Answers "is the CAGR premium or capital gains?" for the real-chain B3
backtest (same windows, universe, sizing and premium source as
b3_performance.py --historical). Identity per sleeve:

    ΔNAV = (gross premium − commissions) + realized stock P&L (call-away vs
           put strike) + unrealized stock P&L (still-held shares at the final
           close vs put strike) + residual (open option liability at the
           window end, fill/mark rounding)

Premium is counted in full as income, so the stock leg is measured
strike-to-strike (or strike-to-final-mark) — "what happened to the shares
you were forced to buy", with no premium double-counting.

Run:  python -m rlbot.evaluation.b3_decompose            # writes reports/b3_decomposition_historical.json
"""
from __future__ import annotations

import json

import pandas as pd

from rlbot.benchmarks.policies import AdaptiveRulePolicy
from rlbot.config import RlbotConfig
from rlbot.data.loaders import FrameStore
from rlbot.evaluation.b3_performance import START_CASH, WINDOWS
from rlbot.evaluation.put_gate import require_gate
from rlbot.simulator.environment import WheelEnv
from rlbot.simulator.portfolio import ExecutionConfig

COMMISSION = ExecutionConfig().commission_per_contract


def decompose(result, final_close: float) -> dict:
    prem_gross = comm = realized = 0.0
    n_puts = n_calls = n_assign = n_called = 0
    held_strike = None
    held_n = 0
    for d in result.decisions:
        c = d.contract
        if c is None or d.chosen_action == 0:
            continue
        n = int(c.get("contracts", 1))
        prem_gross += c["premium_fill"] * 100 * n
        comm += COMMISSION * n
        if c["type"] == "PUT":
            n_puts += 1
            if d.next_position_state == "LONG_STOCK":       # assigned
                held_strike, held_n = c["strike"], 100 * n
                n_assign += 1
        else:
            n_calls += 1
            if d.next_position_state == "CASH" and held_strike is not None:  # called away
                realized += (c["strike"] - held_strike) * held_n
                n_called += 1
                held_strike, held_n = None, 0
    fp = result.final_portfolio
    unrealized = 0.0
    if fp.shares > 0:
        basis_strike = held_strike if held_strike is not None else (
            (fp.cost_basis or final_close))       # fallback; premium already counted
        unrealized = (final_close - basis_strike) * fp.shares
    d_nav = float(result.nav.iloc[-1] - result.nav.iloc[0])
    net_prem = prem_gross - comm
    residual = d_nav - net_prem - realized - unrealized
    return {"d_nav": d_nav, "premium_gross": prem_gross, "commissions": comm,
            "premium_net": net_prem, "stock_realized": realized,
            "stock_unrealized_end": unrealized, "residual_open_liability_etc": residual,
            "n_puts": n_puts, "n_calls": n_calls, "n_assigned": n_assign,
            "n_called_away": n_called, "end_shares": int(fp.shares)}


def main():
    cfg = RlbotConfig(use_valuation_proxy=True)
    gate = require_gate(cfg)
    from rlbot.options.historical_source import historical_source_for
    store = FrameStore(cfg)
    out = {"source": "historical", "start_cash_per_sleeve": START_CASH, "windows": {}}
    for name, (start, end) in WINDOWS.items():
        per, tot = {}, {}
        years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
        for t in cfg.tickers:
            frame = store.frame(t)
            ps = historical_source_for(t, cfg, gate["iv_uplift"])
            env = WheelEnv(t, frame, ps, dynamic_sizing=True)
            res = env.run(AdaptiveRulePolicy(), start, end, starting_cash=START_CASH)
            final_close = float(frame.loc[:pd.Timestamp(end), "close"].iloc[-1])
            per[t] = decompose(res, final_close)
            for k, v in per[t].items():
                tot[k] = tot.get(k, 0) + v
        cap = START_CASH * len(cfg.tickers)
        pct = {k: tot[k] / cap for k in ("d_nav", "premium_gross", "commissions",
                                         "premium_net", "stock_realized",
                                         "stock_unrealized_end",
                                         "residual_open_liability_etc")}
        out["windows"][name] = {
            "years": round(years, 2), "totals_usd": tot,
            "pct_of_start_capital": pct,
            "simple_annualized_pct": {k: v / years for k, v in pct.items()},
            "per_ticker": per}
        print(f"--- {name} ({years:.1f}y) ---")
        for k in ("premium_net", "stock_realized", "stock_unrealized_end",
                  "residual_open_liability_etc", "d_nav"):
            print(f"  {k:28} ${tot[k]:>12,.0f}  {pct[k]:+7.1%} of capital  "
                  f"{pct[k]/years:+6.2%}/yr")
        print(f"  puts sold {tot['n_puts']}, assigned {tot['n_assigned']}, "
              f"calls sold {tot['n_calls']}, called away {tot['n_called_away']}, "
              f"sleeves still holding stock at end: "
              f"{sum(1 for v in per.values() if v['end_shares'] > 0)}/10")
    p = cfg.data.base_path / "reports" / "b3_decomposition_historical.json"
    p.write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
