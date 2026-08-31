"""Metric suite (SPEC-006 §3) over a daily NAV series."""
from __future__ import annotations

import numpy as np
import pandas as pd


def nav_metrics(nav: pd.Series) -> dict:
    nav = nav.dropna()
    if len(nav) < 40:
        return {"error": "series too short"}
    rets = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    ann_vol = rets.std() * np.sqrt(252)
    downside = rets[rets < 0].std() * np.sqrt(252)
    dd = (nav / nav.cummax() - 1).min()
    monthly = nav.resample("ME").last().pct_change().dropna()
    cvar5 = monthly[monthly <= monthly.quantile(0.05)].mean() if len(monthly) >= 20 else np.nan
    return {
        "cagr": float(cagr),
        "total_return": float(nav.iloc[-1] / nav.iloc[0] - 1),
        "ann_vol": float(ann_vol),
        # standard arithmetic Sharpe/Sortino (rf=0), not the CAGR hybrid
        # (2026-08-30 review fix)
        "sharpe": float(rets.mean() * 252 / ann_vol) if ann_vol > 0 else np.nan,
        "sortino": float(rets.mean() * 252 / downside)
        if downside and downside > 0 else np.nan,
        "max_drawdown": float(dd),
        "cvar5_monthly": float(cvar5) if pd.notna(cvar5) else None,
        "days": int(len(nav)),
    }


def wheel_metrics(decisions: list, cycles: list) -> dict:
    opens = [d for d in decisions if d.contract is not None]
    waits = [d for d in decisions if d.contract is None]
    puts = [c for c in cycles if c["leg"] == "CSP"]
    calls = [c for c in cycles if c["leg"] == "CC"]
    return {
        "n_decisions": len(decisions),
        "n_opens": len(opens),
        "wait_rate": len(waits) / len(decisions) if decisions else None,
        "avg_abs_delta": float(np.mean([abs(d.contract["delta"]) for d in opens])) if opens else None,
        "avg_dte": float(np.mean([d.contract["dte"] for d in opens])) if opens else None,
        "premium_collected": float(sum(d.contract["premium_fill"] * 100 for d in opens)),
        "assignment_rate": float(np.mean([c["assigned"] for c in puts])) if puts else None,
        "call_away_rate": float(np.mean([c["assigned"] for c in calls])) if calls else None,
        "n_put_cycles": len(puts),
        "n_call_cycles": len(calls),
    }
