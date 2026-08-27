"""Daily assistant (SPEC-008): opening recommendations + position guidance
from the twice-validated rule policy, with live trajectory logging.

Run:  python -m rlbot.assistant.daily [--download] [--cash 100000]
                                       [--positions data_local/positions.csv]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rlbot.benchmarks.policies import AdaptiveRulePolicy, LeveragedETFPolicy
from rlbot.config import RlbotConfig
from rlbot.data.loaders import FrameStore
from rlbot.evaluation.put_gate import require_gate
from rlbot.learning.trajectories import validate_record
from rlbot.options.premium_source import SyntheticBSPremiumSource
from rlbot.options.selector import SelectorConfig, select_contract
from rlbot.risk.engine import RiskConfig, validate_open
from rlbot.risk.valuation import (clamp_action, exit_floor, premium_required,
                                  put_ceiling, wheel_regime)
from rlbot.simulator.portfolio import ExecutionConfig
from rlbot.state.encoder import encode_q_state
from rlbot.state.enums import CashAction, PositionState, legal_actions
from rlbot.state.mgmt import (
    CHALLENGE_DELTA,
    encode_mgmt_state,
    moneyness_bucket,
    premium_captured,
)

STALE_TRADING_DAYS = 3
REGIME_NAMES = {0: "BULL_LOW_VOL", 1: "BULL_HIGH_VOL", 2: "SIDEWAYS", 3: "BEAR_STRESS"}
VAL_NAMES = {0: "ATTRACTIVE", 1: "FAIR", 2: "EXPENSIVE"}
VC_NAMES = {0: "POOR", 1: "NORMAL", 2: "ATTRACTIVE"}
MGMT_NOTE = ("Validated guidance is HOLD to expiration. G3 evidence: mechanical "
             "MOS-based rolling cost 1.2-2.3%/yr vs holding. Flags below are "
             "attention signals, not roll instructions.")
MODEL_NOTE = ("Quotes marked historical_chain are REAL previous-close chain "
              "mids/deltas (AV); synthetic_bs rows are model values. Either "
              "way, confirm against your broker's live quote before acting. "
              "Recommendations only - not investment advice.")
VAL_GATE_NOTE = ("Valuation gates (SPEC-009) use the fair-value-discount "
                 "ensemble's Wheel FV: puts must land a net basis "
                 "(strike - premium) at/below the acquisition ceiling and "
                 "are masked by regime (no puts on VERY_EXPENSIVE names); "
                 "'Prem req' is the minimum LIVE premium that makes the "
                 "strike acceptable - check it against your broker quote.")
LEVERAGED_NOTE = ("3x rows use the capped leveraged rule table (max BALANCED, "
                  "WAIT in any stress regime). Assignment means 3x market "
                  "exposure; model premiums are least reliable on these names "
                  "(vol clustering) — consider reduced contract size.")

LEGEND = """## Legend

**State column** = `market regime / valuation / vol-compensation` — the three inputs the rule policy conditions on.

| Market regime | Definition (from SPY + VIX) |
|---|---|
| BULL_LOW_VOL | SPY above its 200-day SMA, drawdown > −10%, VIX below its 5-yr 60th percentile |
| BULL_HIGH_VOL | Same bull structure, but VIX at/above its 60th percentile |
| SIDEWAYS | Neither bull nor stress conditions met |
| BEAR_STRESS | SPY below 200-SMA with ≥10% drawdown, or drawdown ≤ −15%, or VIX ≥ 85th percentile |

| Valuation | Definition (vs the FV-buy anchor = min of FMP median, TipRanks mean, Stock Oracle) |
|---|---|
| ATTRACTIVE | Price more than 5% below the anchor |
| FAIR | Within ±5% of the anchor, or no valuation data |
| EXPENSIVE | Price more than 5% above the anchor |

| Vol-compensation | Definition (per-ticker ATM-IV where chains exist; VIX proxy otherwise) |
|---|---|
| POOR | IV below realized vol, or IV percentile < 20th — thin premium for the risk |
| NORMAL | In between |
| ATTRACTIVE | IV ≥ 2 pts over realized vol AND IV percentile ≥ 60th — rich premium |

**Actions.** The policy picks a risk tier; the selector converts it to a contract:

| Tier | Target put delta | When the rules choose it |
|---|---|---|
| WAIT | — | Bear/stress (unless vol-comp ATTRACTIVE), or no liquid contract in band |
| PUT_DEFENSIVE | 0.05–0.10 | Stress + attractive vol-comp only |
| PUT_CONSERVATIVE | 0.10–0.18 | Sideways, or bull + EXPENSIVE valuation |
| PUT_BALANCED | 0.18–0.25 | Bull + FAIR valuation (also the 3x cap) |
| PUT_AGGRESSIVE | 0.25–0.35 | Bull + ATTRACTIVE valuation |

The Δ column is the selected contract's actual delta (≈ assignment probability); DTE targets 25–45 days. Leveraged ETFs (3x) cap at BALANCED and always WAIT in stress.

**Valuation gates (SPEC-009).** Wheel FV comes from the fair-value-discount ensemble (reliability-weighted intrinsic + haircut analyst target). Regime = spot/Wheel FV: <0.80 DEEP_UNDERVALUED, <0.95 UNDERVALUED, ≤1.05 FAIR_VALUED, ≤1.20 EXPENSIVE, >1.20 VERY_EXPENSIVE. Puts are masked by regime (VERY_EXPENSIVE → no puts; EXPENSIVE caps at CONSERVATIVE) and every candidate must land `strike − premium ≤ Ceiling = min(FV·(1−MOS), spot·0.95)`. `Prem req` = the minimum live premium making the shown strike acceptable. Open CSPs/CCs are flagged when their filled net basis / effective exit violates the boundary. Calls on DEEP_UNDERVALUED names are capped at DEFENSIVE (don't sell away large fundamental upside for a few dollars of premium). Stale or missing valuation data disables all of this (warning shown).

**Position guidance.** HOLD to expiration is the validated default (rolling on margin-of-safety triggers tested 1.2–2.3%/yr worse). Flags are attention signals: `BREACHED` = option in the money; `challenged` = |delta| ≥ 0.40; `expiry week` = ≤ 7 days left. A breached covered call at/above your cost basis is the wheel's intended profit-taking exit, not a failure.
"""


def recommend_opening(ticker: str, frame: pd.DataFrame, ps, cash: float,
                      policy=None, leveraged: bool = False,
                      valuation=None, val_cfg=None) -> dict:
    policy = policy or AdaptiveRulePolicy()
    row = frame.iloc[-1]
    date = frame.index[-1]
    q = encode_q_state(row["market_regime"], row["valuation_state"], row["vol_compensation"])
    out = {"ticker": ticker, "date": str(date.date()), "spot": float(row["close"]),
           "leveraged": leveraged}
    if q is None or pd.isna(row["vol_proxy"]) or row["vol_proxy"] <= 0:
        out["action"] = "SKIP"
        out["reason"] = "state undefined (indicator warmup or missing data)"
        return out
    out["q_state"] = list(q)
    out["state_names"] = [REGIME_NAMES[q[0]], VAL_NAMES[q[1]], VC_NAMES[q[2]]]
    action = policy.decide(PositionState.CASH, q, row)
    out["policy_action"] = action.name
    regime = None
    if valuation is not None:
        regime = wheel_regime(out["spot"], valuation.wheel_fv)
        out["valuation"] = {
            "wheel_fv": round(valuation.wheel_fv, 2),
            "regime": regime.name if regime is not None else None,
            "reliability_tier": valuation.reliability_tier,
            "put_ceiling": round(put_ceiling(valuation, out["spot"],
                                             val_cfg), 2),
            "as_of": valuation.date,
        }
        gated = clamp_action(action, regime)
        if gated != action:
            out["policy_action_raw"] = action.name
            out["policy_action"] = gated.name
            action = gated
            if action == CashAction.WAIT:
                out["action"] = "WAIT"
                out["reason"] = (f"valuation gate: regime {regime.name} "
                                 "blocks new puts")
                return out
    if action == CashAction.WAIT:
        out["action"] = "WAIT"
        out["reason"] = "rule policy: conditions do not pay enough for assignment risk"
        return out
    vol = float(row["vol_proxy"])
    chain = ps.chain(date, out["spot"], vol, "P")
    quote, n_cands = select_contract(action, chain, out["spot"], vol, q[1],
                                     cfg=SelectorConfig(),
                                     valuation=valuation, val_cfg=val_cfg)
    risk = validate_open(quote, 1, cash, 0, cash, 0.0, False,
                         RiskConfig.single_ticker(),
                         valuation=valuation, spot=out["spot"],
                         val_cfg=val_cfg)
    if quote is None:
        out["action"] = "WAIT"
        out["reason"] = ("no strike clears the valuation ceiling in the "
                         "chain window" if valuation is not None else
                         "tier unimplementable in current chain window")
        return out
    if not risk.passed:
        out["action"] = "WAIT"
        out["reason"] = f"risk engine: {risk.flags}"
        return out
    out["action"] = "SELL_PUT"
    if valuation is not None:
        out["valuation"]["premium_required"] = round(
            premium_required(quote.strike,
                             put_ceiling(valuation, out["spot"], val_cfg)), 2)
    out["contract"] = {
        "type": "PUT", "strike": quote.strike,
        "expiration": str(quote.expiration.date()), "dte": quote.dte,
        "delta": round(quote.delta, 4), "model_premium": round(quote.mid, 2),
        "premium_source": getattr(ps, "source_name", "synthetic_bs"),
        "candidates_considered": n_cands,
    }
    return out


def guide_position(pos: dict, frame: pd.DataFrame, ps,
                   valuation=None, val_cfg=None) -> dict:
    row = frame.iloc[-1]
    date = frame.index[-1]
    spot = float(row["close"])
    vol = float(row["vol_proxy"]) if pd.notna(row["vol_proxy"]) else 0.25
    cp = "P" if pos["type"].upper() == "CSP" else "C"
    exp = pd.Timestamp(pos["expiration"])
    dte = max((exp - date.normalize()).days, 0)
    mark = ps.reprice(cp, pos["strike"], exp, date, spot, vol)
    delta = ps.delta_now(cp, pos["strike"], exp, date, spot, vol)
    m_state = encode_mgmt_state(row["market_regime"], cp, spot, pos["strike"], dte)
    flags = []
    if moneyness_bucket(cp, spot, pos["strike"]) == 0:
        flags.append("BREACHED (in the money)")
    if abs(delta) >= CHALLENGE_DELTA:
        flags.append(f"delta {abs(delta):.2f} >= 0.40 (challenged)")
    if dte <= 7:
        flags.append("expiry week")
    if valuation is not None:
        prem0 = float(pos.get("premium_fill", 0) or 0)
        if cp == "P":
            ceil = put_ceiling(valuation, spot, val_cfg)
            if pos["strike"] - prem0 > ceil + 1e-9:
                flags.append(f"net basis {pos['strike'] - prem0:.2f} above "
                             f"acquisition ceiling {ceil:.2f}")
        else:
            floor = exit_floor(valuation, None, val_cfg)
            if pos["strike"] + prem0 < floor - 1e-9:
                flags.append(f"effective exit {pos['strike'] + prem0:.2f} "
                             f"below fundamental exit floor {floor:.2f} "
                             "(selling upside too cheaply)")
    pc = premium_captured(mark, float(pos.get("premium_fill", 0) or 0))
    return {
        "ticker": pos["ticker"], "type": pos["type"].upper(),
        "strike": pos["strike"], "expiration": str(exp.date()), "dte": dte,
        "spot": spot, "model_mark": round(mark, 2),
        "model_delta": round(delta, 4),
        "premium_captured": round(pc, 3),
        "mgmt_state": list(m_state) if m_state is not None else None,
        "guidance": "HOLD", "attention_flags": flags,
    }


def load_positions(path: Path) -> tuple:
    if not path.exists():
        return [], f"positions file not found ({path}); openings-only brief"
    try:
        df = pd.read_csv(path, comment="#")
        need = {"ticker", "type", "strike", "expiration", "premium_fill"}
        if not need.issubset(df.columns):
            return [], f"positions file missing columns {need - set(df.columns)}"
        return df.to_dict("records"), None
    except Exception as e:                                   # REQ-8.3
        return [], f"positions file unreadable: {e}"


def decision_record(rec: dict, cash: float, run_id: str, seq: int) -> dict | None:
    if "q_state" not in rec:
        return None
    contract = None
    if rec.get("contract"):
        c = rec["contract"]
        contract = {"type": c["type"], "strike": c["strike"],
                    "expiration": c["expiration"], "dte": c["dte"],
                    "delta": c["delta"], "premium_fill": c["model_premium"],
                    "premium_source": c.get("premium_source", "synthetic_bs"),
                    "candidates_considered": c["candidates_considered"]}
    action = {"WAIT": 0, "SELL_PUT": None}.get(rec["action"], 0)
    if action is None:
        action = int(CashAction[rec["policy_action"]])
    return {
        "schema_version": "trajectory_v2", "reward_version": "diff_v1",
        "run_id": run_id, "episode_id": f"live-{rec['ticker']}",
        "decision_id": f"live-{rec['ticker']}:{rec['date']}",
        "date": rec["date"], "ticker": rec["ticker"], "position_state": "CASH",
        "q_state": rec["q_state"],
        "features": {"market_regime": rec["q_state"][0],
                      "valuation_state": rec["q_state"][1],
                      "vol_compensation": rec["q_state"][2],
                      "raw": {"spot": rec["spot"]}},
        "available_actions": [int(a) for a in legal_actions(PositionState.CASH)],
        "chosen_action": action, "action_source": "policy",
        "policy_meta": {"policy": "B3-rules", "validated": "G2/G2-rerun/G3"},
        "contract": contract,
        "risk_checks": {"passed": True, "flags": []},
        "portfolio_before": {"cash": cash, "shares": 0, "cost_basis": None, "nav": cash},
        "next_epoch_date": None, "portfolio_after": None, "reward": None,
        "reference_return": None, "next_q_state": None,
        "next_position_state": None, "terminal": False, "counterfactuals": None,
        "mgmt_state": None,
    }


def render_brief(date: str, recs: list, guides: list, warnings: list) -> str:
    lines = [f"# Wheel Daily Brief — {date}", "",
             f"_{MODEL_NOTE}_", ""]
    for w in warnings:
        lines.append(f"> ⚠️ {w}")
    lines += ["", "## Opening recommendations (cash sleeve)", "",
              f"_{VAL_GATE_NOTE}_", "",
              "| Ticker | State | Action | Strike | DTE | Δ | Model prem "
              "| Wheel FV | Regime | Ceiling | Prem req |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        state = "/".join(r.get("state_names", ["—"]))
        c = r.get("contract") or {}
        v = r.get("valuation") or {}
        name = f"{r['ticker']} (3x)" if r.get("leveraged") else r["ticker"]
        lines.append(f"| {name} | {state} | {r['action']} "
                     f"| {c.get('strike', '—')} | {c.get('dte', '—')} "
                     f"| {c.get('delta', '—')} | {c.get('model_premium', '—')} "
                     f"| {v.get('wheel_fv', '—')} | {v.get('regime', '—')} "
                     f"| {v.get('put_ceiling', '—')} "
                     f"| {v.get('premium_required', '—')} |")
    if any(r.get("leveraged") for r in recs):
        lines += ["", f"_{LEVERAGED_NOTE}_"]
    lines += ["", "## Open positions", "", f"_{MGMT_NOTE}_", ""]
    if guides:
        lines += ["| Ticker | Type | Strike | DTE | Δ now | Prem captured | Guidance | Flags |",
                  "|---|---|---|---|---|---|---|---|"]
        for g in guides:
            lines.append(f"| {g['ticker']} | {g['type']} | {g['strike']} | {g['dte']} "
                         f"| {g['model_delta']} | {g['premium_captured']:.0%} "
                         f"| **{g['guidance']}** | {'; '.join(g['attention_flags']) or '—'} |")
    else:
        lines.append("_No open positions on file._")
    lines += ["", LEGEND]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--cash", type=float, default=100_000.0)
    parser.add_argument("--positions", type=Path, default=None)
    parser.add_argument("--no-sync-positions", action="store_true",
                        help="skip the Google-Sheet monitor-tab sync")
    parser.add_argument("--no-real-quotes", action="store_true",
                        help="skip the daily chain fetch; synthetic quotes only")
    parser.add_argument("--market-volcomp", action="store_true",
                        help="revert to the market-proxy vol comp (per-ticker "
                             "IV is the default since G4 passed, 2026-08-23)")
    args = parser.parse_args(argv)

    cfg = RlbotConfig(
        use_valuation_proxy=True,
        vol_comp_source="market" if args.market_volcomp else "ticker_iv",
    )
    warnings = []
    if args.download:
        from rlbot.data.build import build_all
        build_all(RlbotConfig(), download=True)
        try:                                  # FV snapshot (ported from sibling)
            from rlbot.data.fv_snapshot import refresh_valuation, snapshot_fair_value
            snapshot_fair_value(cfg)
            refresh_valuation(cfg)
        except Exception as e:
            warnings.append(f"FV snapshot skipped: {e}")
        if not args.no_real_quotes:
            from rlbot.data.daily_chain_update import update_daily_chains
            chain_status = update_daily_chains(cfg)
            bad = {t: s for t, s in chain_status.items() if s.startswith("error")}
            if bad:
                warnings.append(f"chain refresh issues: {bad}")
            if not args.market_volcomp:
                from rlbot.features.ticker_iv import build_all as build_tiv
                build_tiv(cfg)
    from rlbot.data.fv_ensemble import load_wheel_valuations
    valuations, val_warns = load_wheel_valuations(cfg)
    warnings.extend(val_warns)
    if valuations:
        warnings.append(f"valuation gates active for {len(valuations)} "
                        "tickers (SPEC-009)")
    gate = require_gate(cfg)
    synth = SyntheticBSPremiumSource(iv_uplift=gate["iv_uplift"])

    def ps_for(ticker, last_date):
        """Real prev-close chains when today's snapshot exists; synthetic else."""
        if args.no_real_quotes:
            return synth
        try:
            from rlbot.options.historical_source import historical_source_for
            src = historical_source_for(ticker, cfg, gate["iv_uplift"])
            day = src._day(last_date)
            if day is not None and not day.empty:
                return src
        except Exception:
            pass
        return synth

    store = FrameStore(cfg)
    recs, guides = [], []
    latest = None
    for t in cfg.assistant_universe:
        try:
            frame = store.frame(t)
        except Exception:
            warnings.append(f"{t}: no data in canonical tables; run --download")
            continue
        if frame.empty:
            warnings.append(f"{t}: no data in canonical tables; run --download")
            continue
        latest = max(latest or frame.index[-1], frame.index[-1])
        lev = cfg.is_leveraged(t)
        policy = LeveragedETFPolicy() if lev else AdaptiveRulePolicy()
        recs.append(recommend_opening(t, frame, ps_for(t, frame.index[-1]),
                                      args.cash, policy=policy, leveraged=lev,
                                      valuation=valuations.get(t),
                                      val_cfg=cfg.val_gates))
    if (pd.Timestamp.now().normalize() - latest).days > STALE_TRADING_DAYS + 2:
        warnings.append(f"latest bar is {latest.date()} — data is stale; "
                        "re-run with --download")             # REQ-8.4

    pos_path = args.positions or cfg.data.base_path / "positions.csv"
    if not args.no_sync_positions:
        from rlbot.data.positions_sheet import fetch_active_positions, write_positions_csv
        synced, sync_warns = fetch_active_positions(pd.Timestamp.now().normalize())
        warnings.extend(sync_warns)
        if synced or not sync_warns:      # reachable sheet (even if empty) wins
            write_positions_csv(synced, pos_path)
            warnings.append(f"positions.csv synced from monitor sheet: "
                            f"{len(synced)} active position(s)")
    positions, pos_warn = load_positions(pos_path)
    if pos_warn:
        warnings.append(pos_warn)
    for p in positions:
        tkr = str(p["ticker"]).upper()
        if tkr in cfg.assistant_universe:
            frame_t = store.frame(tkr)
            guides.append(guide_position(p, frame_t,
                                         ps_for(tkr, frame_t.index[-1]),
                                         valuation=valuations.get(tkr),
                                         val_cfg=cfg.val_gates))
        else:
            warnings.append(f"position ticker {tkr} not in universe; skipped")

    date = str(latest.date())
    live = cfg.data.base_path / "live"
    live.mkdir(parents=True, exist_ok=True)
    payload = {"date": date, "iv_uplift": gate["iv_uplift"],
               "openings": recs, "positions": guides, "warnings": warnings,
               "notes": [MODEL_NOTE, MGMT_NOTE, LEVERAGED_NOTE]}
    (live / f"recommendations_{date}.json").write_text(json.dumps(payload, indent=2))
    brief = render_brief(date, recs, guides, warnings)
    (live / f"brief_{date}.md").write_text(brief)

    run_id = f"live-{date}"
    with open(live / "decisions.jsonl", "a") as f:
        for i, r in enumerate(recs):
            record = decision_record(r, args.cash, run_id, i)
            if record is not None:
                validate_record(record)                       # REQ-8.2
                f.write(json.dumps(record) + "\n")

    print(brief)
    print(f"wrote {live}/brief_{date}.md, recommendations_{date}.json, decisions.jsonl")
    return payload


if __name__ == "__main__":
    main()
