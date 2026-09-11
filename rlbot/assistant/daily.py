"""Daily assistant (SPEC-008): opening recommendations + position guidance
from the twice-validated rule policy, with live trajectory logging.

Run:  python -m rlbot.assistant.daily [--download] [--cash 100000]
                                       [--positions data_local/positions.csv]
"""
from __future__ import annotations

import argparse
import datetime as _dt
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
from rlbot.risk.mcb_gates import (LOW_YIELD_ROC, cca_call_cap,
                                  classify_covered_call, is_downtrend,
                                  mcb_binding, mcb_valuation_state,
                                  net_basis_flag, opportunity_scan,
                                  premium_required, reachability_advice,
                                  tradeable)
from rlbot.simulator.portfolio import ExecutionConfig
from rlbot.state.encoder import encode_q_state
from rlbot.state.enums import CashAction, PositionState, StockAction, legal_actions
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
VAL_GATE_NOTE = ("Valuation gates use mcb-wheel's Maximum Comfortable Basis "
                 "report as an aggressiveness pivot, not a uniform rejection "
                 "boundary (v2, 2026-09-09): the ceiling only blocks a trade "
                 "when the producer's per-ticker Posture calls for it — "
                 "CONSERVATIVE never blocks, HIGHER always does, MODERATE "
                 "depends on which risk tier is being filled — or when this "
                 "repo's own downtrend override applies. 'Prem req' is the "
                 "minimum LIVE premium that makes the strike acceptable - "
                 "check it against your broker quote.")
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

**Column definitions (openings).**

| Column | Meaning |
|---|---|
| Price | Last close used for every calculation in the row |
| Decision / Reason | One status per row — Candidate; Candidate — review required; Blocked — position limit / MCB ceiling / layer A; Reference only; No contract — and the one reason behind it. Full machine reasons live in the JSON |
| Strike / DTE / Δ / Model prem | The recommended contract; on Blocked and Reference-only rows, the contract the selector would have chosen |
| Posture | mcb-wheel's per-ticker `delta_posture` (CONSERVATIVE / MODERATE / HIGHER); `(hard)` = the ceiling actually filtered this row |
| **Ceiling** | **The MCB value** — mcb-wheel's `wheel_entry`: the highest net cost basis (strike − premium) you would still be comfortable acquiring the stock at; the put's net basis must sit at/below it when the gate is hard |
| Prem req | `max(0, strike − Ceiling)`: the minimum live premium that makes the shown strike acceptable |
| Ref high (date) | mcb-wheel's trailing high (`ref_high`) the pullback percentiles are measured from, with the date our close series printed it |
| Typical / Larger historical pullback | mcb-wheel's median (`dd50`) and 75th-percentile (`dd75`) corrections from the trailing high, translated into a resulting price and dollar decline: e.g. from a 200.00 high an 18% pullback is a 36.00 decline to 164.00, shown as `164.00 (−36.00, 18.0%)`. When the report carries no trailing high the percentage is applied to today's price and the cell is marked *illustrative*. Reference for judging `drop_needed` (how far spot sits above the Ceiling): within the typical pullback → entry reachable in an ordinary correction |
| MCB flags | guardrail (CAUTION/SEVERE), reachability (PATIENCE/UNREACHABLE — informational), layer A (MONITOR_ONLY/HALT — never trade) |

**Column definitions (covered calls).**

| Column | Meaning |
|---|---|
| CC posture | mcb-wheel's `cc_posture`: HIGHER (spot at/above the call-away level — assignment welcome), STANDARD (level reachable within typical 45-day upside), CONSERVATIVE (assignment undesirable here — tier capped) |
| **CCA** | **Covered-Call Assignment level** — mcb-wheel's `selected_call_away_level`: the price at which being called away is acceptable, per your per-ticker `assignment_preference` (EXIT ≤ TRIM ≤ PROTECT), computed from upper-percentile valuation multiples, forward fundamental value and the FMP median target, floored at cost basis × (1 + minimum return). The call-side twin of the MCB ceiling |
| Exit check | `strike + premium` vs CCA: `OK (+x)` = ASSIGNMENT_ACCEPTABLE; `INCOME_WAIT (−x)` = effective exit below the level by x — Candidate — review required, never auto-rejected |
| Shares held / Committed / Additional contracts | Shares on the positions tab; shares already covering open CC contracts (100 × contracts in positions.csv); `(held − committed) ÷ 100` = contracts still coverable. Zero → Blocked — no uncovered shares |

**MCB gates (mcb-wheel v2, 2026-09-09) — an aggressiveness pivot, not a uniform rejection boundary.** `wheel_entry` is the net cost basis (strike − premium) comfortable to acquire at. The producer publishes a `Posture` per ticker: **CONSERVATIVE** (far above entry, rising/sideways) — the ceiling is advisory only, since a conservative-delta put here is a low-probability income trade, not an acquisition attempt; **HIGHER** (at/near entry) — the ceiling is hard, always, since assignment is welcome and sought; **MODERATE** (entry reachable via a typical correction) — hard only for BALANCED-and-up tiers, advisory for income tiers (WAIT/DEFENSIVE/CONSERVATIVE), the one case the producer leaves to us. `(hard)` next to the posture in the table means the ceiling actually filtered this trade. **A confirmed downtrend overrides all of this to conservative-or-wait** — "don't catch a falling knife" — since the producer computes no trend signal; this repo supplies it. `Prem req` = the minimum live premium making the shown strike acceptable. Layer-A `MONITOR_ONLY`/`HALT` names are never traded. Reachability remains a separate, informational signal (`UNREACHABLE`/`PATIENCE`), unrelated to Posture. Open CSPs are flagged only when the ceiling was/would be hard for them; the call side is governed by cost-basis discipline (calls never sold below basis), since MCB is acquisition-side only. **The selector's assignment penalty is also MCB-fed** (`score_valuation` in the JSON): ATTRACTIVE when entry is within the ticker's typical correction (`drop_needed ≤ dd50`), EXPENSIVE when it needs a beyond-90th-percentile correction, one notch of relief when a severe correction is already underway outside a stressed market — this replaces the Google-Sheet FV anchor in the score only; the State column's valuation (the policy's input) is unchanged. A report older than 5 trading sessions is expired and disables all of this (warning shown).

**Position guidance.** HOLD to expiration is the validated default (rolling on margin-of-safety triggers tested 1.2–2.3%/yr worse). Flags are attention signals: `BREACHED` = option in the money; `challenged` = |delta| ≥ 0.40; `expiry week` = ≤ 7 days left. A breached covered call at/above your cost basis is the wheel's intended profit-taking exit, not a failure.
"""


def _correlated_exposures(ticker, book, closes, spots, nav, rcfg_risk):
    """RISK-9 (SPEC-004 §2.7) support data: book underlyings whose trailing
    corr_lookback-day return correlation with `ticker` ≥ corr_threshold,
    each with its potential exposure as % NAV."""
    if not closes or ticker not in closes:
        return []
    mine = closes[ticker].pct_change().dropna().tail(rcfg_risk.corr_lookback)
    hits = []
    for other in sorted(book.underlyings):
        if other == ticker or other not in closes:
            continue
        theirs = closes[other].pct_change().dropna().tail(rcfg_risk.corr_lookback)
        joined = pd.concat([mine, theirs], axis=1, join="inner").dropna()
        if len(joined) < rcfg_risk.corr_lookback // 2:
            continue
        corr = joined.corr().iloc[0, 1]
        if pd.notna(corr) and corr >= rcfg_risk.corr_threshold:
            hits.append({"ticker": other, "corr": float(corr),
                         "exposure_pct": book.potential_exposure(
                             other, spots.get(other)) / nav if nav > 0 else 0})
    return hits


def week_label(year: int, week: int) -> str:
    """ISO (year, week) -> the expiration week's Mon–Fri calendar dates,
    e.g. 'October 5–9, 2026' or 'September 28–October 2, 2026'."""
    mon = _dt.date.fromisocalendar(year, week, 1)
    fri = _dt.date.fromisocalendar(year, week, 5)
    if mon.month == fri.month:
        return f"{mon:%B} {mon.day}–{fri.day}, {fri.year}"
    return f"{mon:%B} {mon.day}–{fri:%B} {fri.day}, {fri.year}"


def high_date(frame: pd.DataFrame, ref_high: float | None,
              tol: float = 0.001, years: int = 6):
    """Date on which the close series printed mcb-wheel's trailing high
    (within tol); None when the frame does not contain it."""
    if ref_high is None or frame is None or frame.empty or "close" not in frame:
        return None
    closes = frame["close"].tail(252 * years).dropna()
    if closes.empty:
        return None
    gap = (closes - ref_high).abs()
    i = gap.idxmin()                       # the close nearest the reported high
    return str(i.date()) if gap[i] / ref_high <= tol else None


def pullback_scenarios(ref_high, spot, dd50, dd75) -> dict | None:
    """DD percentiles as dollar moves and resulting prices. Measured from the
    trailing high when the producer supplied it; otherwise applied to spot
    and labelled illustrative (the percentiles describe corrections from a
    high, not from today's price)."""
    base = ref_high if ref_high else spot
    if not base:
        return None
    out = {"basis": "ref_high" if ref_high else "spot", "base": round(base, 2)}
    for name, dd in (("typical", dd50), ("larger", dd75)):
        out[name] = (None if dd is None else
                     {"pct": dd, "drop": round(base * dd, 2),
                      "price": round(base * (1 - dd), 2)})
    return out


def _short_warning(w: str) -> str:
    code = w.split(":")[0]
    return {"RISK-5-CUM": "week escrow cap if every candidate executes",
            "RISK-8-CUM": "stress reserve if every candidate executes",
            "RISK-7": "earnings inside the expiry window",
            "RISK-9": "correlated with existing book exposure",
            "CCA": "exit below the CCA level"}.get(code, code)


def decision_status(rec: dict) -> dict:
    """One status + one short reason per opening row (2026-09-11). The full
    machine reason stays in `reason`; the review block carries full texts."""
    a, reason = rec.get("action"), rec.get("reason") or ""
    t = rec.get("ticker", "")
    if a == "SELL_PUT":
        ws = rec.get("review_warnings") or []
        if not ws:
            return {"status": "Candidate",
                    "reason": "passes policy, MCB gate and risk engine"}
        return {"status": "Candidate — review required",
                "reason": "; ".join(dict.fromkeys(_short_warning(w) for w in ws))}
    if a == "SKIP":
        return {"status": "Skipped", "reason": reason}
    if reason.startswith("risk engine"):
        d = rec.get("risk_detail") or {}
        parts = []
        for f in d.get("flags", []):
            code = f.split(":")[0]
            if code == "RISK-3" and d.get("exposure_pct") is not None:
                parts.append(f"concentration limit: exposure to {t} (shares + puts "
                             f"incl. this one) would be {d['exposure_pct']:.1%} of NAV "
                             f"vs the {d['exposure_cap_pct']:.0%} cap")
            elif code == "RISK-5" and d.get("week_pct") is not None:
                parts.append(f"expiration-week escrow would be {d['week_pct']:.1%} "
                             f"of NAV vs the {d['week_cap_pct']:.0%} cap")
            elif code == "RISK-8":
                parts.append("assignment-stress reserve would fall below minimum")
            elif code == "RISK-4":
                parts.append("distinct-underlying count at cap")
            elif code == "RISK-1":
                parts.append("not cash-secured at this NAV")
            else:
                parts.append(f)
        return {"status": "Blocked — position limit", "reason": "; ".join(parts) or reason}
    if reason.startswith("every viable expiry"):
        return {"status": "Blocked — position limit",
                "reason": "every 25–45 DTE expiry falls in a week already at the escrow cap"}
    if reason.startswith("MCB gate: layer A"):
        return {"status": "Blocked — layer A", "reason": reason.split(": ", 1)[-1]}
    if reason.startswith("MCB unreachable") or reason.startswith("MCB gate"):
        v = rec.get("mcb") or {}
        return {"status": "Blocked — MCB ceiling",
                "reason": f"no delta-band strike nets below the ceiling "
                          f"{v.get('ceiling', '?')} (posture {v.get('posture') or '?'}"
                          f"{', downtrend override' if v.get('downtrend_override') else ''}"
                          "); see opportunity scan"}
    if reason.startswith("rule policy"):
        return {"status": "Reference only",
                "reason": "policy chose WAIT; contract shown is the CONSERVATIVE-tier reference"}
    return {"status": "No contract", "reason": reason}


def call_decision_status(rec: dict) -> dict:
    """Covered-call twin of decision_status."""
    a, reason = rec.get("action"), rec.get("reason") or ""
    k = rec.get("cca") or {}
    shares, committed = int(k.get("shares") or 0), int(k.get("committed_shares") or 0)
    avail = k.get("available_contracts")
    if a == "SKIP":
        return {"status": "Skipped", "reason": reason}
    if shares < 100:
        return {"status": "Reference only", "reason": "no shares held — hypothetical"}
    if avail == 0:
        return {"status": "Blocked — no uncovered shares",
                "reason": f"{shares:,} held, {committed:,} already committed to open calls"}
    if a == "SELL_CALL":
        if rec.get("review_warnings"):
            return {"status": "Candidate — review required",
                    "reason": f"exit {k.get('exit_value')} sits ${k.get('shortfall'):,.2f} "
                              f"below the CCA level {k.get('level'):.2f}"}
        tail = ("exit at/above CCA" if k.get("classification") == "ASSIGNMENT_ACCEPTABLE"
                else "no CCA level (constraint absent)")
        return {"status": "Candidate", "reason": f"{avail} contract(s) coverable; {tail}"}
    if "cost basis" in reason:
        spot, basis = rec.get("spot"), k.get("basis")
        if spot is not None and basis is not None and spot < basis:
            return {"status": "Blocked — cost basis",
                    "reason": f"price {spot:.2f} below basis {basis:.2f}; "
                              "no call at/above basis in the window"}
        return {"status": "No contract",
                "reason": f"no call strike at/above basis {basis:.2f} in the 25–45 DTE "
                          "window (thin chain); price is above basis"}
    if reason.startswith("stock policy"):
        return {"status": "Wait — policy",
                "reason": "stock rule: preserve upside; contract shown is the CONSERVATIVE reference"}
    if reason.startswith("risk engine"):
        return {"status": "Blocked — risk engine", "reason": reason}
    return {"status": "No contract", "reason": reason}


def capital_summary(book, nav: float, recs: list, stock_values: dict,
                    rk, available_cash: float | None = None) -> dict:
    """Total account value, cash reserved for puts, stock sleeve, cash
    available for new trades (verified only when the user supplies the
    broker figure), and per-expiration-week capacity with calendar dates."""
    escrow = float(book.put_escrow) if book is not None else 0.0
    stock = sum(v["value"] for v in stock_values.values())
    proposed = [(r["ticker"], r["contract"]["strike"] * 100,
                 pd.Timestamp(r["contract"]["expiration"]))
                for r in recs if r.get("action") == "SELL_PUT" and r.get("contract")]
    prop_total = sum(e for _, e, _ in proposed)
    est = nav - escrow - stock
    verified = available_cash is not None
    avail = available_cash if verified else est
    d = {"nav": nav, "nav_source": "assumed via --cash; not broker-verified",
         "put_escrow": escrow,
         "n_put_rows": len(book.put_positions) if book is not None else 0,
         "stock_value": stock, "stock_detail": stock_values,
         "available_cash": avail,
         "available_verified": verified,
         "available_source": ("verified — broker figure entered via --available-cash"
                              if verified else
                              "estimated = NAV − put escrow − stock sleeve; verify against broker"),
         "inconsistent": (not verified) and est < 0,
         "proposed_escrow": prop_total, "n_proposed": len(proposed),
         "after_all_candidates": avail - prop_total,
         "week_cap_pct": rk.max_week_assignment_pct, "weeks": []}
    cap = rk.max_week_assignment_pct * nav
    by_week: dict = {}
    for t, e, exp in proposed:
        iso = exp.isocalendar()
        by_week.setdefault((iso.year, iso.week), []).append((t, e))
    existing = dict(book.expiry_week_escrow) if book is not None else {}
    for wk in sorted(set(existing) | set(by_week)):
        ex = existing.get(wk, 0.0)
        items = by_week.get(wk, [])
        add = sum(e for _, e in items)
        if ex <= 0 and add <= 0:
            continue
        total = ex + add
        d["weeks"].append({"iso": f"{wk[0]}-W{wk[1]:02d}", "label": week_label(*wk),
                           "existing": ex, "cap": cap, "remaining": max(0.0, cap - ex),
                           "proposed": add, "proposed_tickers": [t for t, _ in items],
                           "if_all": total, "if_all_pct": total / nav if nav else 0.0,
                           "over_by": max(0.0, total - cap)})
    return d


def cumulative_review(recs: list, book, rk, nav: float) -> tuple:
    """SPEC-004 §2.10 (2026-09-11): the per-trade hard rules validate each
    recommendation against the EXISTING book, so a brief can propose a set
    that jointly breaches RISK-5 (week cap) or RISK-8 (stress reserve). This
    surfaces the joint effect as a human-review warning — never a block —
    with the room left and one example subset that fits. The user decides.

    Returns (per_ticker_warnings: {ticker: [text]}, summary_lines: [str]).
    """
    per, summary = {}, []
    if book is None or nav <= 0:
        return per, summary
    new = [(r["ticker"], r["contract"]["strike"] * 100,
            pd.Timestamp(r["contract"]["expiration"]))
           for r in recs if r.get("action") == "SELL_PUT" and r.get("contract")]
    if not new:
        return per, summary
    cap = rk.max_week_assignment_pct * nav
    by_week = {}
    for t, esc, exp in new:
        iso = exp.isocalendar()
        by_week.setdefault((iso.year, iso.week), []).append((t, esc))
    for wk, items in sorted(by_week.items()):
        existing = book.expiry_week_escrow.get(wk, 0.0)
        added = sum(e for _, e in items)
        total = existing + added
        if total <= cap + 1e-9:
            continue
        room = max(0.0, cap - existing)
        fit, used = [], 0.0
        for t, e in sorted(items, key=lambda x: -x[1]):   # largest-first example
            if used + e <= room + 1e-9:
                fit.append(t); used += e
        names = ", ".join(f"{t} ${e:,.0f}" for t, e in items)
        text = (f"RISK-5-CUM:week_cap_if_all_executed — Expiration week of "
                f"{week_label(*wk)}: existing commitment ${existing:,.0f}; cap "
                f"${cap:,.0f} ({rk.max_week_assignment_pct:.0%} of NAV); remaining "
                f"capacity ${room:,.0f}. Proposed this brief: {names} (total "
                f"${added:,.0f}) → ${total:,.0f} if all executed ({total / nav:.1%} "
                f"of NAV), over the cap by ${total - cap:,.0f}. Each trade passes "
                f"alone; together they do not. Capacity example — fits the limit, "
                f"not a ranking: {', '.join(fit) if fit else 'none'} (${used:,.0f}). "
                "Human decision.")
        summary.append(text)
        for t, _ in items:
            per.setdefault(t, []).append(text)
    # RISK-8 joint stress: all proposed puts added to the book at once
    extra = [{"ticker": t, "strike": esc / 100, "expiration": exp, "escrow": esc}
             for t, esc, exp in new]
    puts = list(book.put_positions) + extra
    tmp = type(book)(n_open_positions=book.n_open_positions, put_escrow=book.put_escrow,
                     expiry_week_counts=dict(book.expiry_week_counts),
                     underlyings=set(book.underlyings), put_positions=puts)
    stress = tmp.stressed_assignment({}, week1_pct=rk.stress_week1_pct,
                                     week2_pct=rk.stress_week2_pct, itm_pct=rk.stress_itm_pct)
    if rk.min_stress_reserve_pct > 0 and (nav - stress) / nav < rk.min_stress_reserve_pct:
        text = (f"RISK-8-CUM:stress_reserve_if_all_executed — with every proposed put "
                f"added, the assignment stress is ${stress:,.0f}, leaving "
                f"{(nav - stress) / nav:.1%} of NAV vs the {rk.min_stress_reserve_pct:.0%} "
                f"reserve. Each trade passes alone. Human decision.")
        summary.append(text)
        for t, _, _ in new:
            per.setdefault(t, []).append(text)
    return per, summary


def _contract_dict(quote, ps, n_cands: int, basis: str | None = None) -> dict:
    """Uniform contract record. `basis` is set only on candidate_contract
    (the contract a WAIT row *would* have traded) so the brief can show
    Strike/DTE/Δ/prem on every row while the decision record stays clean."""
    d = {"type": "PUT" if quote.cp == "P" else "CALL", "strike": quote.strike,
         "expiration": str(quote.expiration.date()), "dte": quote.dte,
         "delta": round(quote.delta, 4), "model_premium": round(quote.mid, 2),
         "premium_source": getattr(ps, "source_name", "synthetic_bs"),
         "candidates_considered": n_cands}
    if basis is not None:
        d["basis"] = basis
    return d


def recommend_opening(ticker: str, frame: pd.DataFrame, ps, cash: float,
                      policy=None, leveraged: bool = False,
                      mcb=None, book=None, rcfg=None,
                      risk_cfg=None, book_spots=None, closes=None,
                      low_yield_roc: float = LOW_YIELD_ROC) -> dict:
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
    ceiling, hard = None, False
    if mcb is not None:
        downtrend = is_downtrend(row.get("structure"))  # pd.Series.get: None if absent
        ceiling, hard = mcb_binding(mcb, action, downtrend)
        out["mcb"] = {
            "posture": mcb.delta_posture,
            "ceiling": round(ceiling, 2) if ceiling is not None else None,
            "hard": hard,
            "downtrend_override": downtrend,
            "guardrail": mcb.guardrail,
            "layer_a": mcb.layer_a,
            "reachability": mcb.reachability,
            "dd50": mcb.dd50, "dd75": mcb.dd75,      # correction context (ref.)
            "ref_high": mcb.ref_high,
            "ref_high_date": high_date(frame, mcb.ref_high),
            "pullbacks": pullback_scenarios(mcb.ref_high, out["spot"],
                                            mcb.dd50, mcb.dd75),
            "as_of": mcb.date,
        }
        ok, why = tradeable(mcb)
        if not ok:
            out["action"] = "WAIT"
            out["reason"] = why
            return out
    # Only a HARD binding filters candidates (SPEC-011 §2 rule 1, v2): a
    # CONSERVATIVE-posture ceiling, or a MODERATE-posture ceiling on an
    # income tier, is advisory and must not touch the selector at all.
    binding_ceiling = ceiling if hard else None
    # Score-side valuation (SPEC-011 §2 rule 6 / SPEC-004 §1.2): the
    # assignment-penalty multiplier is fed by MCB position + drawdown
    # severity, not the FV-sheet axis. The policy's Q-state q[1] is untouched.
    score_val = q[1]
    if mcb is not None:
        score_val = int(mcb_valuation_state(mcb, q[0]))
        out["mcb"]["score_valuation"] = VAL_NAMES[score_val]
    rk = risk_cfg or RiskConfig()
    # Book-aware RISK-5 (SPEC-004 §1.1, 2026-09-11): tell the selector how
    # much escrow room each ISO expiry week still has so it skips capped weeks
    # instead of picking a contract the risk engine will certainly reject.
    headroom = (book.expiry_week_headroom(rk.max_week_assignment_pct, cash)
                if book is not None else None)
    vol = float(row["vol_proxy"])
    chain = ps.chain(date, out["spot"], vol, "P")
    if action == CashAction.WAIT:
        out["action"] = "WAIT"
        out["reason"] = "rule policy: conditions do not pay enough for assignment risk"
        # Contract columns always populated (SPEC-008 §1.3, 2026-09-11): show
        # the CONSERVATIVE-tier reference so the reader sees what a put here
        # would pay today. Information, not a recommendation.
        ref, n_ref = select_contract(CashAction.PUT_CONSERVATIVE, chain,
                                     out["spot"], vol, score_val,
                                     cfg=SelectorConfig(),
                                     expiry_week_headroom=headroom)
        if ref is not None:
            out["candidate_contract"] = _contract_dict(
                ref, ps, n_ref, basis="reference_conservative_tier")
        return out
    # Reachability is informational (user choice 2026-08-31): the strike scan
    # proceeds and the advisory rides along in the JSON + MCB-flags column.
    advice = reachability_advice(mcb, q[2])
    if advice is not None:
        out["mcb"]["advisory"] = advice
    quote, n_cands = select_contract(action, chain, out["spot"], vol, score_val,
                                     cfg=SelectorConfig(),
                                     net_basis_ceiling=binding_ceiling,
                                     expiry_week_headroom=headroom)
    # Book-level enforcement (2026-08-30): with a book, the whole
    # position set feeds RISK-4/5/8 and the estimated-earnings blackout.
    if book is not None and quote is not None and rcfg is not None:
        from rlbot.risk.book import earnings_in_window, next_earnings_estimate
        spots = book_spots or {}
        est = next_earnings_estimate(ticker, rcfg)
        earnings_info = {
            "date": str(est.date()) if est is not None else "?",
            "source": "estimated (last AV reportedDate + ~91d, ±5d window)",
            "expiration": str(quote.expiration.date()),
        }
        exposure_before = book.potential_exposure(ticker, spots.get(ticker, out["spot"]))
        week_before = book.same_week_escrow(quote.expiration)
        notional = quote.strike * 100
        out["risk_detail"] = {                     # feeds decision_status()
            "exposure_pct": (exposure_before + notional) / cash if cash else None,
            "exposure_cap_pct": rk.max_pct_per_underlying,
            "week_pct": (week_before + notional) / cash if cash else None,
            "week_cap_pct": rk.max_week_assignment_pct}
        risk = validate_open(
            quote, 1, cash, 0, cash,
            open_put_escrow=book.put_escrow,
            event_in_window=earnings_in_window(ticker, quote.expiration, rcfg),
            cfg=rk,                                # portfolio-mode caps
            n_underlyings=book.n_underlyings,
            is_new_underlying=not book.has(ticker),
            underlying_exposure=exposure_before,
            same_week_escrow=week_before,
            stressed_assignment=book.stressed_assignment(
                spots, extra_put={"ticker": ticker, "strike": quote.strike,
                                  "expiration": quote.expiration,
                                  "escrow": quote.strike * 100},
                week1_pct=rk.stress_week1_pct, week2_pct=rk.stress_week2_pct,
                itm_pct=rk.stress_itm_pct),
            earnings_info=earnings_info,
            correlated=_correlated_exposures(ticker, book, closes, spots,
                                             cash, rk),
        )
    else:
        risk = validate_open(quote, 1, cash, 0, cash, 0.0, False,
                             RiskConfig.single_ticker())
    if quote is None:
        out["action"] = "WAIT"
        # Attribute the empty scan honestly, in order: (1) every viable
        # expiry sits in a week already at the RISK-5 cap; (2) the MCB
        # ceiling was HARD and binding; (3) chain/liquidity.
        if headroom is not None:
            unweeked, n_uw = select_contract(action, chain, out["spot"], vol,
                                             score_val, cfg=SelectorConfig(),
                                             net_basis_ceiling=binding_ceiling)
            if unweeked is not None:
                iso = unweeked.expiration.isocalendar()
                capped = sorted(f"{y}-W{w:02d}" for (y, w), room in headroom.items()
                                if room <= 0)
                out["candidate_contract"] = _contract_dict(
                    unweeked, ps, n_uw, basis="blocked_by_risk5_week_cap")
                out["reason"] = (
                    "every viable expiry in the 25–45 DTE window falls in an "
                    "ISO week already at the RISK-5 cap "
                    f"({', '.join(capped) or 'none listed'}); best contract "
                    f"absent the cap: {unweeked.strike:g} put "
                    f"{unweeked.expiration.date()} (week {iso.year}-W{iso.week:02d})")
                return out
        if binding_ceiling is not None:
            ungated, _ = select_contract(action, chain, out["spot"], vol, score_val,
                                         cfg=SelectorConfig())
            if ungated is not None:
                out["candidate_contract"] = _contract_dict(
                    ungated, ps, 0, basis="blocked_by_mcb_ceiling")
                head = (
                    f"MCB unreachable within normal delta bands "
                    f"(posture {mcb.delta_posture or '?'}"
                    f"{', downtrend override' if out['mcb']['downtrend_override'] else ''}"
                    f"): ceiling {binding_ceiling:.2f} (best band strike "
                    f"{ungated.strike:g} needs premium >= "
                    f"{premium_required(ungated.strike, binding_ceiling):.2f}, "
                    f"model {ungated.mid:.2f})")
                # SPEC-011 §6: below-band advisory scan — the economics are
                # rendered, the judgment is the user's (2026-09-01: the ROC
                # threshold is a LOW YIELD flag, never a blocker). Never
                # executable.
                opp = opportunity_scan(chain, binding_ceiling,
                                       low_yield_roc=low_yield_roc)
                if opp is None:
                    out["reason"] = (head + "; no MCB-compliant strike in "
                                     "the chain window — geometrically "
                                     "unreachable")
                else:
                    out["mcb"]["opportunity"] = opp
                    tags = f" [{', '.join(opp['flags'])}]" if opp["flags"]                         else ""
                    out["reason"] = (
                        head + "; below-band MCB-compliant candidate "
                        "(advisory — user judgment on opportunity cost): "
                        f"{opp['strike']:g} put, {opp['dte']} DTE, "
                        f"Δ {opp['delta']:.3f}, premium "
                        f"{opp['premium']:.2f}, net basis "
                        f"{opp['net_basis']:.2f} (headroom "
                        f"{opp['mcb_headroom']:.2f}), ROC on escrow "
                        f"{opp['roc_ann']:.1%}/yr{tags}")
                return out
        out["reason"] = "tier unimplementable in current chain window"
        return out
    mcb_viol = net_basis_flag(quote.strike, quote.mid, binding_ceiling)
    if mcb_viol is not None:               # belt-and-suspenders: selector
        out["action"] = "WAIT"             # already pre-filtered on this
        out["reason"] = f"MCB gate: {mcb_viol}"
        out["candidate_contract"] = _contract_dict(quote, ps, n_cands,
                                                   basis="blocked_by_mcb_ceiling")
        return out
    if not risk.passed:
        out["action"] = "WAIT"
        out["reason"] = f"risk engine: {risk.flags}"
        out.setdefault("risk_detail", {})["flags"] = list(risk.flags)
        out["candidate_contract"] = _contract_dict(quote, ps, n_cands,
                                                   basis="blocked_by_risk_engine")
        return out
    out.pop("risk_detail", None)
    out["action"] = "SELL_PUT"
    if risk.warnings:      # SPEC-004 §2.8: human-review, never blocking
        out["review_warnings"] = list(risk.warnings)
    if ceiling is not None:
        out["mcb"]["premium_required"] = round(
            premium_required(quote.strike, ceiling), 2)
    out["contract"] = _contract_dict(quote, ps, n_cands)
    return out


CC_NOTE = ("Covered calls for every universe name, gated by mcb-wheel's "
           "Covered-Call Assignment level (CCA) instead of MCB: CCA is the "
           "selected call-away price (EXIT/TRIM/PROTECT per your "
           "assignment_preference, floored at basis x (1 + min return)). It "
           "classifies, never rejects: strike + premium >= CCA -> "
           "ASSIGNMENT_ACCEPTABLE; below -> INCOME_WAIT with the shortfall "
           "(⚠ REVIEW). CC posture caps the tier (CONSERVATIVE -> "
           "CALL_CONSERVATIVE at most; a Bull Trend on top -> DEFENSIVE, "
           "protect upside). Holdings are split from a hypothetical watchlist: "
           "shares held come from the positions tab, shares committed from open "
           "CC contracts in positions.csv, and only the uncovered remainder "
           "(in 100-share lots) counts as coverable.")


def recommend_call(ticker: str, frame: pd.DataFrame, ps, cash: float,
                   policy=None, leveraged: bool = False, mcb=None,
                   book=None) -> dict:
    """Stock-side twin of recommend_opening for the brief's covered-call
    table (SPEC-008 §1 step 3b, SPEC-011 §2 rule 7). Not logged to the
    trajectory file: rows without shares are hypothetical."""
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
    shares = 0
    basis = None
    committed = int(book.cc_shares.get(ticker, 0)) if book is not None else 0
    if mcb is not None and mcb.position_shares:
        shares, basis = int(mcb.position_shares), mcb.position_basis
    else:
        shares = committed          # only what open CCs imply -> none free
    available = max(0, (shares - committed) // 100) if shares >= 100 else 0
    out["cca"] = {"level": mcb.cca if mcb else None,
                  "mode": mcb.cca_mode if mcb else None,
                  "cc_posture": mcb.cc_posture if mcb else None,
                  "rise_needed": mcb.rise_needed if mcb else None,
                  "up50": mcb.up50 if mcb else None, "up75": mcb.up75 if mcb else None,
                  "shares": shares, "basis": basis,
                  "committed_shares": committed,       # 100 x open CC contracts
                  "available_contracts": available,
                  "reference_only": shares < 100}
    action = policy.decide(PositionState.LONG_STOCK, q, row)
    out["policy_action"] = action.name
    capped, why = cca_call_cap(action, out["cca"]["cc_posture"], row.get("structure"))
    if capped != action:
        out["policy_action_raw"], action = action.name, capped
        out["policy_action"], out["cca"]["cap_reason"] = capped.name, why
    vol = float(row["vol_proxy"])
    chain = ps.chain(date, out["spot"], vol, "C")
    if action == StockAction.WAIT:
        out["action"] = "WAIT"
        out["reason"] = "stock policy: no call (preserve upside)"
        ref, n_ref = select_contract(StockAction.CALL_CONSERVATIVE, chain, out["spot"],
                                     vol, q[1], cost_basis=basis, cfg=SelectorConfig())
        if ref is not None:
            out["candidate_contract"] = _contract_dict(
                ref, ps, n_ref, basis="reference_conservative_tier")
        return out
    quote, n_cands = select_contract(action, chain, out["spot"], vol, q[1],
                                     cost_basis=basis, cfg=SelectorConfig())
    if quote is None:
        out["action"] = "WAIT"
        out["reason"] = ("tier unimplementable in current chain window"
                         + (" (no call strike at/above cost basis)" if basis else ""))
        if basis:
            # Show what the tier would pick without the basis floor, so the
            # row still carries Strike | DTE | Δ | prem (blocked, reference)
            ref, n_ref = select_contract(action, chain, out["spot"], vol, q[1],
                                         cost_basis=None, cfg=SelectorConfig())
            if ref is not None:
                out["candidate_contract"] = _contract_dict(
                    ref, ps, n_ref, basis="blocked_by_cost_basis")
                cls, shortfall = classify_covered_call(ref.strike, ref.mid,
                                                       out["cca"]["level"])
                out["cca"].update({"classification": cls, "shortfall": shortfall,
                                   "exit_value": round(ref.strike + ref.mid, 2)})
        return out
    cls, shortfall = classify_covered_call(quote.strike, quote.mid, out["cca"]["level"])
    out["cca"].update({"classification": cls, "shortfall": shortfall,
                       "exit_value": round(quote.strike + quote.mid, 2)})
    if shares >= 100:
        risk = validate_open(quote, 1, cash, shares, cash, 0.0, False,
                             RiskConfig.single_ticker())
        if not risk.passed:
            out["action"] = "WAIT"
            out["reason"] = f"risk engine: {risk.flags}"
            out["candidate_contract"] = _contract_dict(quote, ps, n_cands,
                                                       basis="blocked_by_risk_engine")
            return out
    out["action"] = "SELL_CALL"
    out["contract"] = _contract_dict(quote, ps, n_cands)
    if cls == "INCOME_WAIT":
        out.setdefault("review_warnings", []).append(
            f"CCA:INCOME_WAIT — effective call-away {quote.strike:g} + "
            f"{quote.mid:.2f} = {quote.strike + quote.mid:.2f} sits "
            f"${shortfall:,.2f} below the selected call-away level "
            f"{out['cca']['level']:.2f} ({out['cca']['mode']}): assignment is "
            f"not desired at this price. Permit only with conservative delta "
            f"and explicit intent. Human decision.")
    return out


def guide_position(pos: dict, frame: pd.DataFrame, ps,
                   mcb=None, market_regime: int | None = None) -> dict:
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
    # MCB is an acquisition-side construct: flag open CSPs whose filled net
    # basis sits above the required-tier ceiling. The call side has no MCB
    # analogue — cost-basis discipline (calls never below basis) governs it.
    if mcb is not None and cp == "P":
        # v2 (SPEC-011 §2 rule 1): flag only when the ceiling was/would be
        # HARD for this position — CONSERVATIVE-posture names above their
        # wheel_entry are a legitimate income position, not a violation.
        # No recorded entry tier for open positions, so the current model
        # delta stands in for MODERATE's tier split (same 0.18 boundary).
        prem0 = float(pos.get("premium_fill", 0) or 0)
        downtrend = is_downtrend(row.get("structure"))
        posture = mcb.delta_posture
        hard = bool(downtrend or posture == "HIGHER"
                    or (posture == "MODERATE" and abs(delta) >= 0.18))
        ceil = mcb.wheel_entry
        if ceil is not None and hard and pos["strike"] - prem0 > ceil + 1e-9:
            tag = " (downtrend override)" if downtrend else f" (posture {posture})"
            flags.append(
                f"net basis {pos['strike'] - prem0:.2f} above MCB "
                f"ceiling {ceil:.2f}{tag}")
        if mcb.layer_a in ("MONITOR_ONLY", "HALT"):
            flags.append(f"MCB layer A now {mcb.layer_a} — no new exposure")
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


def _pct(x) -> str:
    return "—" if x is None else f"{x:.1%}"


def _mcb_flags(v: dict) -> str:
    """Compact guardrail/reachability/layer-A cell for the brief tables."""
    bits = []
    if v.get("guardrail") and v["guardrail"] != "NORMAL":
        bits.append(v["guardrail"])
    if v.get("reachability") and v["reachability"] != "NORMAL":
        bits.append(v["reachability"])
    if v.get("layer_a") and v["layer_a"] != "OWN":
        bits.append(v["layer_a"])
    return ", ".join(bits) if bits else ("—" if not v else "ok")


def _pullback_cell(p: dict | None, key: str) -> str:
    if not p or not p.get(key):
        return "—"
    x = p[key]
    cell = f"{x['price']:.2f} (−{x['drop']:.2f}, {x['pct']:.1%})"
    return cell if p["basis"] == "ref_high" else cell + " ~illustrative from today's price"


def _ref_high_cell(v: dict) -> str:
    if v.get("ref_high") is None:
        return "—"
    return f"{v['ref_high']:.2f} ({v.get('ref_high_date') or 'date n/a'})"


def _fmt_price(x) -> str:
    return "—" if x is None else f"{x:,.2f}"


def _render_capital(cap: dict) -> list:
    lines = ["", "## Capital and limits", "",
             "| Item | Amount | Basis |", "|---|---|---|",
             f"| Total account value (NAV) | ${cap['nav']:,.0f} | {cap['nav_source']} |",
             f"| Cash reserved for open puts (escrow) | ${cap['put_escrow']:,.0f} "
             f"| {cap['n_put_rows']} open put rows; strike × 100 × contracts |"]
    sd = cap.get("stock_detail") or {}
    detail = (", ".join(f"{t} {d['shares']:,} × {d['spot']:,.2f}" for t, d in sd.items())
              or "none on the positions tab")
    lines.append(f"| Stock sleeve (shares × last close) | ${cap['stock_value']:,.0f} | {detail} |")
    tag = "Verified" if cap.get("available_verified") else "Estimated"
    lines.append(f"| {tag} cash available for new trades | ${cap['available_cash']:,.0f} "
                 f"| {cap['available_source']} |")
    if cap.get("inconsistent"):
        lines.append("| ⚠ Inconsistent inputs | — | put escrow + stock sleeve exceed the "
                     "assumed NAV: either the account is larger than `--cash`, or the "
                     "open puts are not fully cash-secured (margin). Every ratio above "
                     "and below inherits this. Pass `--cash` and `--available-cash` "
                     "from the broker before acting |")
    lines.append(f"| New put escrow proposed in this brief | ${cap['proposed_escrow']:,.0f} "
                 f"| {cap['n_proposed']} candidate row(s) |")
    lines.append(f"| Remaining if every candidate executes | ${cap['after_all_candidates']:,.0f} "
                 f"| {tag.lower()} available − proposed |")
    if cap.get("weeks"):
        pct = cap["week_cap_pct"]
        lines += ["", f"### Expiration-week capacity (cap {pct:.0%} of NAV = "
                      f"${pct * cap['nav']:,.0f} per expiration week)", "",
                  "| Expiration week | Existing commitment | Remaining capacity "
                  "| Proposed this brief | If all executed | Status |",
                  "|---|---|---|---|---|---|"]
        for w in cap["weeks"]:
            prop = (f"${w['proposed']:,.0f} ({', '.join(w['proposed_tickers'])})"
                    if w["proposed"] else "—")
            status = (f"⚠ over the cap by ${w['over_by']:,.0f}" if w["over_by"] > 0
                      else "within cap")
            lines.append(f"| {w['label']} | ${w['existing']:,.0f} | ${w['remaining']:,.0f} "
                         f"| {prop} | ${w['if_all']:,.0f} ({w['if_all_pct']:.1%}) | {status} |")
        lines += ["", "_Existing commitment = put escrow already open in that week. "
                      "Remaining capacity = cap − existing. A subset that fits the cap "
                      "is a capacity example, not a ranking of the trades._"]
    return lines


def _review_block(rows: list, title: str) -> list:
    """Identical warning texts (the cumulative notices are shared by every
    affected row) render once, naming all tickers; per-ticker ones as-is."""
    grouped: dict = {}
    for r in rows:
        for w in r.get("review_warnings", []):
            grouped.setdefault(w, []).append(r["ticker"])
    if not grouped:
        return []
    lines = ["", title, ""]
    for w, tkrs in grouped.items():
        lines.append(f"> **{', '.join(tkrs)}** — {w}")
        lines.append(">")
    lines.pop()
    return lines


def _call_row(r: dict, holdings: bool) -> str:
    state = "/".join(r.get("state_names", ["—"]))
    c = r.get("contract") or r.get("candidate_contract") or {}
    k = r.get("cca") or {}
    d = r.get("decision") or call_decision_status(r)
    name = f"{r['ticker']} (3x)" if r.get("leveraged") else r["ticker"]
    cls = k.get("classification")
    check = ("—" if cls is None else
             f"OK (+{k['exit_value'] - k['level']:.2f})" if cls == "ASSIGNMENT_ACCEPTABLE"
             else f"INCOME_WAIT (−{k['shortfall']:.2f})")
    level = "—" if k.get("level") is None else f"{k['level']:.2f} ({k.get('mode') or '?'})"
    row = (f"| {name} | {_fmt_price(r.get('spot'))} | {state} | **{d['status']}** | {d['reason']} "
           f"| {c.get('strike', '—')} | {c.get('dte', '—')} "
           f"| {c.get('delta', '—')} | {c.get('model_premium', '—')} "
           f"| {k.get('cc_posture') or '—'} | {level} | {check} |")
    if holdings:
        row += (f" {k.get('shares', 0):,} | {k.get('committed_shares', 0):,} "
                f"| {k.get('available_contracts', 0)} |")
    return row


def render_brief(date: str, recs: list, guides: list, warnings: list,
                 call_recs: list | None = None, capital: dict | None = None) -> str:
    lines = [f"# Wheel Daily Brief — {date}", "",
             f"_{MODEL_NOTE}_", ""]
    for w in warnings:
        lines.append(f"> ⚠️ {w}")
    if capital:
        lines += _render_capital(capital)
    lines += ["", "## Opening recommendations (cash sleeve)", "",
              f"_{VAL_GATE_NOTE}_", "",
              "| Ticker | Price | State | Decision | Reason | Strike | DTE | Δ | Model prem "
              "| Posture | Ceiling | Prem req | Ref high (date) | Typical historical pullback "
              "| Larger historical pullback | MCB flags |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        state = "/".join(r.get("state_names", ["—"]))
        # Contract columns always populated (2026-09-11): the recommendation
        # for candidates, the would-be/blocked/reference contract otherwise.
        c = r.get("contract") or r.get("candidate_contract") or {}
        v = r.get("mcb") or {}
        d = r.get("decision") or decision_status(r)
        name = f"{r['ticker']} (3x)" if r.get("leveraged") else r["ticker"]
        posture = v.get("posture", "—")
        if v.get("hard"):
            posture += " (hard)"
        lines.append(f"| {name} | {_fmt_price(r.get('spot'))} | {state} | **{d['status']}** "
                     f"| {d['reason']} "
                     f"| {c.get('strike', '—')} | {c.get('dte', '—')} "
                     f"| {c.get('delta', '—')} | {c.get('model_premium', '—')} "
                     f"| {posture} | {v.get('ceiling', '—')} "
                     f"| {v.get('premium_required', '—')} "
                     f"| {_ref_high_cell(v)} | {_pullback_cell(v.get('pullbacks'), 'typical')} "
                     f"| {_pullback_cell(v.get('pullbacks'), 'larger')} "
                     f"| {_mcb_flags(v)} |")
    lines += ["", "_Decision is the one status that matters per row: **Candidate** "
                  "(clear to trade), **Candidate — review required** (trade only after "
                  "reading its review item below), **Blocked — …** (the named limit "
                  "stops it today), **Reference only** (the policy chose WAIT; the "
                  "contract shown is what a CONSERVATIVE put would pay). On Blocked and "
                  "Reference rows the Strike/DTE/Δ/prem columns show the contract the "
                  "selector would have chosen (`candidate_contract` in the JSON) — "
                  "information, not a recommendation._"]
    # SPEC-011 §6.4: the scan's outcome must be visible in the brief, with
    # 'geometrically unreachable' and 'economically unattractive' distinct.
    opps = [(r["ticker"], r["reason"],
             (r.get("mcb") or {}).get("opportunity")) for r in recs
            if "MCB unreachable within normal delta bands" in r.get("reason", "")]
    if opps:
        lines += ["", "### MCB opportunity scan (advisory — SPEC-011 §6, "
                      "never executable; opportunity cost is your call)", ""]
        for tkr, reason, opp in opps:
            mark = "⚠ " if opp is not None and not opp.get("low_yield") else ""
            lines.append(f"- {mark}**{tkr}**: {reason}")
    lines += _review_block(recs, "### ⚠ Review items for candidates marked "
                                 "'review required' (SPEC-004 §2.8 — approve or "
                                 "reject before trading)")
    if any(r.get("leveraged") for r in recs):
        lines += ["", f"_{LEVERAGED_NOTE}_"]
    if call_recs:
        live = [r for r in call_recs if r.get("action") != "SKIP"]
        held = [r for r in live if (r.get("cca") or {}).get("shares", 0) >= 100]
        watch = [r for r in live if r not in held]
        lines += ["", "## Covered-call recommendations (stock sleeve)", "",
                  f"_{CC_NOTE}_", "",
                  "### Holdings — shares available to cover", ""]
        if held:
            lines += ["| Ticker | Price | State | Decision | Reason | Strike | DTE | Δ "
                      "| Model prem | CC posture | CCA | Exit check | Shares held "
                      "| Committed to open calls | Additional contracts coverable |",
                      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            lines += [_call_row(r, holdings=True) for r in held]
        else:
            lines.append("_No holdings with ≥100 shares on the positions tab._")
        if watch:
            lines += ["", "### Hypothetical watchlist — no shares held", "",
                      "_Reference only: a covered call needs 100 shares per contract; "
                      "these rows show what the stock-side rule would sell if the put "
                      "were assigned today._", "",
                      "| Ticker | Price | State | Decision | Reason | Strike | DTE | Δ "
                      "| Model prem | CC posture | CCA | Exit check |",
                      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
            lines += [_call_row(r, holdings=False) for r in watch]
        lines += _review_block(held, "### ⚠ Covered-call review items (CCA INCOME_WAIT — "
                                     "assignment not desired at this price)")

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


def sync_universe_from_sheet(cfg, positions_path: Path) -> tuple:
    """SPEC-010 fix (2026-08-29): the FV sheet tab is the live core list.
    Adds sheet tickers missing from the universe (via cfg.watch_extra) and
    returns (skip_set, notes): config stocks no longer on the sheet — and
    with no open position — are skipped from the brief. Sheet unreachable
    -> no changes (config fallback)."""
    from rlbot.data.fv_snapshot import FV_GID, FV_SHEET_ID, parse_fv_rows
    from rlbot.vendor.sheet_data import fetch_sheet_rows
    notes = []
    try:
        raw = fetch_sheet_rows(FV_SHEET_ID, FV_GID)
        sheet = set(parse_fv_rows(raw)) if raw else set()
    except Exception:
        sheet = set()
    if not sheet:
        return set(), ["FV sheet unreachable: universe from config fallback"]
    held = set()
    try:
        rows, _ = load_positions(positions_path)
        held = {str(r["ticker"]).upper() for r in rows}
    except Exception:
        pass
    added = sorted(sheet - set(cfg.assistant_universe))
    if added:
        cfg.watch_extra = list(cfg.watch_extra) + added
        notes.append(f"universe synced from FV sheet: added {added}")
    skip = {t for t in cfg.tickers
            if t not in sheet and t not in held}
    if skip:
        notes.append(f"not on FV sheet, no position — skipped: {sorted(skip)}")
    return skip, notes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--cash", type=float, default=100_000.0,
                        help="account NAV the risk ratios are computed against")
    parser.add_argument("--max-underlyings", type=int, default=None,
                        help="RISK-4 distinct-ticker cap (default: RiskConfig's "
                             "12; an attention cap — capital caps are the "
                             "escrow percentages)")
    parser.add_argument("--max-week-pct", type=float, default=None,
                        help="RISK-5 same-expiry-week put escrow / NAV ceiling "
                             "(default: RiskConfig's 0.15)")
    parser.add_argument("--max-exposure-pct", type=float, default=None,
                        help="RISK-3 potential exposure per underlying / NAV "
                             "(shares + puts; default: RiskConfig's 0.15)")
    parser.add_argument("--min-stress-reserve-pct", type=float, default=None,
                        help="RISK-8 unencumbered cash / NAV that must remain "
                             "after the assignment stress (default: "
                             "RiskConfig's 0.15)")
    parser.add_argument("--low-yield-roc", type=float,
                        default=LOW_YIELD_ROC,
                        help="SPEC-011 §6 opportunity scan: annualized ROC "
                             "below which a below-band candidate carries a "
                             "LOW YIELD flag (decision support, never a "
                             "blocker; default 0.07)")
    parser.add_argument("--available-cash", type=float, default=None,
                        help="broker-verified cash available for new trades; "
                             "when given, the brief labels it verified instead "
                             "of estimating NAV − put escrow − stock sleeve")
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
    skip_core, uni_notes = sync_universe_from_sheet(
        cfg, args.positions or cfg.data.base_path / "positions.csv")
    warnings.extend(uni_notes)
    if args.download:
        from rlbot.data.build import build_all
        build_all(cfg, download=True)  # cfg carries the sheet-synced universe
        try:                                  # FV snapshot (ported from sibling)
            from rlbot.data.fv_snapshot import refresh_valuation, snapshot_fair_value
            snapshot_fair_value(cfg)
            refresh_valuation(cfg)
        except Exception as e:
            warnings.append(f"FV snapshot skipped: {e}")
        for ct in [t for t in cfg.assistant_universe
                   if t not in skip_core]:                  # onboard new names
            from rlbot.data import sources as _src
            try:
                _src.load_bars(ct, cfg.data.bars_path)
            except Exception:
                try:
                    _src.download_bars([ct], cfg.data.bars_path,
                                       years=cfg.data.ticker_years)
                except Exception as e:
                    warnings.append(f"{ct}: bars unavailable ({e})")
        try:            # refresh the MCB report (sheet-driven universe)
            import subprocess, sys as _sys
            mcb_repo = Path(cfg.data.mcb_dir).parent
            if (mcb_repo / "run_mcb.py").exists():
                subprocess.run([_sys.executable, "run_mcb.py"],
                               cwd=mcb_repo, capture_output=True, timeout=1200)
        except Exception as e:
            warnings.append(f"MCB refresh skipped: {e}")
        if not args.no_real_quotes:
            from rlbot.data.daily_chain_update import update_daily_chains
            chain_status = update_daily_chains(cfg)
            bad = {t: s for t, s in chain_status.items() if s.startswith("error")}
            if bad:
                warnings.append(f"chain refresh issues: {bad}")
            if not args.market_volcomp:
                from rlbot.features.ticker_iv import build_all as build_tiv
                build_tiv(cfg)
    from rlbot.data.mcb_feed import load_mcb
    mcb_rows, mcb_warns = load_mcb(cfg)
    warnings.extend(mcb_warns)
    if mcb_rows:
        as_of = next(iter(mcb_rows.values())).date
        warnings.append(f"MCB gates active for {len(mcb_rows)} tickers "
                        f"(mcb-wheel report {as_of}; replaces Wheel-FV feed)")
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

    # positions sync moved BEFORE openings (2026-08-30) so the book feeds
    # book-level risk checks on every recommendation
    pos_path = args.positions or cfg.data.base_path / "positions.csv"
    if not args.no_sync_positions:
        from rlbot.data.positions_sheet import fetch_active_positions, write_positions_csv
        synced, sync_warns = fetch_active_positions(pd.Timestamp.now().normalize())
        warnings.extend(sync_warns)
        if synced or not sync_warns:
            write_positions_csv(synced, pos_path)
            warnings.append(f"positions.csv synced from monitor sheet: "
                            f"{len(synced)} active position(s)")
    positions, pos_warn = load_positions(pos_path)
    if pos_warn:
        warnings.append(pos_warn)
    from rlbot.risk.book import build_book
    book = build_book(positions)
    base_risk = RiskConfig()
    risk_cfg = RiskConfig(
        max_underlyings=args.max_underlyings
        if args.max_underlyings is not None else base_risk.max_underlyings,
        max_week_assignment_pct=args.max_week_pct
        if args.max_week_pct is not None else base_risk.max_week_assignment_pct,
        max_pct_per_underlying=args.max_exposure_pct
        if args.max_exposure_pct is not None else base_risk.max_pct_per_underlying,
        min_stress_reserve_pct=args.min_stress_reserve_pct
        if args.min_stress_reserve_pct is not None
        else base_risk.min_stress_reserve_pct,
    )
    warnings.append(
        f"book-level risk active (SPEC-004 §2 v2): {book.n_open_positions} "
        f"positions across {book.n_underlyings} names, "
        f"${book.put_escrow:,.0f} put escrow "
        f"({book.put_escrow / args.cash:.0%} of ${args.cash:,.0f} NAV) — "
        f"hard caps: {risk_cfg.max_underlyings} names, "
        f"week escrow <= {risk_cfg.max_week_assignment_pct:.0%}, "
        f"per-name exposure <= {risk_cfg.max_pct_per_underlying:.0%}, "
        f"stress reserve >= {risk_cfg.min_stress_reserve_pct:.0%}; "
        "earnings + correlation surface as ⚠ REVIEW warnings")
    # Cash-secured puts imply NAV >= escrow: floor the NAV assumption so
    # RISK-3/8 ratios are at least coherent, and say so loudly — pass
    # --cash <account NAV> for real checks.
    if book.put_escrow > args.cash:
        warnings.append(
            f"assumed NAV ${args.cash:,.0f} < book escrow "
            f"${book.put_escrow:,.0f}: flooring NAV at escrow for risk "
            "ratios — pass --cash <your account NAV> for meaningful "
            "RISK-3/5/8 enforcement")
        args.cash = book.put_escrow

    # RISK-3/8/9 support data: latest closes for every name the book or the
    # universe touches (spot marks share exposure and classifies ITM puts;
    # the close series feed the 120d correlation check).
    book_spots, closes = {}, {}
    for t in sorted(set(cfg.assistant_universe) | book.underlyings):
        try:
            f = store.frame(t)
            if not f.empty:
                book_spots[t] = float(f.iloc[-1]["close"])
                closes[t] = f["close"]
        except Exception:
            continue

    recs, guides = [], []
    latest = None
    for t in cfg.assistant_universe:
        if t in skip_core:
            continue
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
                                      mcb=mcb_rows.get(t),
                                      book=book, rcfg=cfg,
                                      risk_cfg=risk_cfg,
                                      book_spots=book_spots, closes=closes,
                                      low_yield_roc=args.low_yield_roc))
    # Covered-call table for every name (SPEC-008 §1 step 3b): same frames,
    # stock-side policy, CCA classification instead of the MCB ceiling.
    call_recs = []
    for t in cfg.assistant_universe:
        if t in skip_core:
            continue
        try:
            frame = store.frame(t)
        except Exception:
            continue
        if frame.empty:
            continue
        lev = cfg.is_leveraged(t)
        call_recs.append(recommend_call(
            t, frame, ps_for(t, frame.index[-1]), args.cash,
            policy=LeveragedETFPolicy() if lev else AdaptiveRulePolicy(),
            leveraged=lev, mcb=mcb_rows.get(t), book=book))

    # SPEC-004 §2.10: joint effect of this run's SELL_PUTs on RISK-5/RISK-8 —
    # a human-review warning on the affected rows, never a block.
    cum_per, cum_summary = cumulative_review(recs, book, risk_cfg, args.cash)
    for r in recs:
        if r["ticker"] in cum_per:
            r.setdefault("review_warnings", []).extend(cum_per[r["ticker"]])
    for line in cum_summary:
        warnings.append("REVIEW (cumulative): " + line.split(" — ", 1)[1])
    for r in recs:
        r["decision"] = decision_status(r)
    for r in call_recs:
        r["decision"] = call_decision_status(r)
    # Capital and limits (2026-09-11): stock sleeve from the MCB positions
    # tab marked at the last close; cash available is estimated unless the
    # broker figure is passed via --available-cash.
    stock_values = {}
    for t, m in mcb_rows.items():
        if m.position_shares and t in book_spots:
            stock_values[t] = {"shares": int(m.position_shares), "spot": book_spots[t],
                               "value": int(m.position_shares) * book_spots[t]}
    capital = capital_summary(book, args.cash, recs, stock_values, risk_cfg,
                              available_cash=args.available_cash)

    if (pd.Timestamp.now().normalize() - latest).days > STALE_TRADING_DAYS + 2:
        warnings.append(f"latest bar is {latest.date()} — data is stale; "
                        "re-run with --download")             # REQ-8.4

    for p in positions:
        tkr = str(p["ticker"]).upper()
        if tkr in cfg.assistant_universe:
            frame_t = store.frame(tkr)
            guides.append(guide_position(p, frame_t,
                                         ps_for(tkr, frame_t.index[-1]),
                                         mcb=mcb_rows.get(tkr)))
        else:
            warnings.append(f"position ticker {tkr} not in universe; skipped")

    date = str(latest.date())
    live = cfg.data.base_path / "live"
    live.mkdir(parents=True, exist_ok=True)
    payload = {"date": date, "iv_uplift": gate["iv_uplift"],
               "openings": recs, "covered_calls": call_recs,
               "capital": capital,
               "positions": guides, "warnings": warnings,
               "notes": [MODEL_NOTE, MGMT_NOTE, LEVERAGED_NOTE]}
    (live / f"recommendations_{date}.json").write_text(json.dumps(payload, indent=2))
    brief = render_brief(date, recs, guides, warnings, call_recs, capital)
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
