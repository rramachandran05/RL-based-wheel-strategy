"""Brief changes 2026-09-11 (SPEC-008 §1 step 3b, SPEC-011 §2 rule 7):
covered-call recommendations for every name, gated by mcb-wheel's CCA
(Covered-Call Assignment level) as a classification — never a rejection;
DD50/DD75 reference columns; legend column definitions."""
from __future__ import annotations

import pandas as pd
import pytest

from rlbot.data.mcb_feed import McbRow
from rlbot.options.premium_source import SyntheticBSPremiumSource
from rlbot.risk.mcb_gates import cca_call_cap, classify_covered_call
from rlbot.state.enums import StockAction

DATE = pd.Timestamp("2026-08-27")
VOL = 0.25
PS = SyntheticBSPremiumSource()


def _frame(structure=None, regime=0, val=1):
    idx = pd.DatetimeIndex([DATE])
    d = {"close": [100.0], "market_regime": [regime], "valuation_state": [val],
         "vol_compensation": [1], "vol_proxy": [VOL]}
    if structure is not None:
        d["structure"] = [structure]
    return pd.DataFrame(d, index=idx)


def _mcb(**kw):
    base = dict(ticker="T", date="2026-08-27", mcb={"FAIR": 90.0},
                min_eligible_tier="FAIR", guardrail=None, layer_a="OWN",
                reachability=None, confidence=0.8, wheel_entry=90.0,
                delta_posture="CONSERVATIVE", dd50=0.10, dd75=0.16, dd90=0.30,
                cc_posture="STANDARD", cca=108.0, cca_mode="TRIM",
                rise_needed=0.08, up50=0.10, up75=0.17)
    base.update(kw)
    return McbRow(**base)


# ------------------------------------------------------------- pure helpers

def test_classify_covered_call():
    assert classify_covered_call(110.0, 2.0, 108.0) == ("ASSIGNMENT_ACCEPTABLE", 0.0)
    assert classify_covered_call(105.0, 1.5, 108.0) == ("INCOME_WAIT", 1.5)
    assert classify_covered_call(105.0, 1.5, None) == (None, None)


def test_cca_call_cap_by_posture_and_uptrend():
    A = StockAction
    assert cca_call_cap(A.CALL_AGGRESSIVE, "HIGHER", "Base") == (A.CALL_AGGRESSIVE, None)
    assert cca_call_cap(A.CALL_AGGRESSIVE, "STANDARD", "Base")[0] == A.CALL_BALANCED
    assert cca_call_cap(A.CALL_AGGRESSIVE, "CONSERVATIVE", "Base")[0] == A.CALL_CONSERVATIVE
    # uptrend + assignment undesirable -> protect upside: DEFENSIVE at most
    capped, why = cca_call_cap(A.CALL_BALANCED, "CONSERVATIVE", "Bull Trend")
    assert capped == A.CALL_DEFENSIVE and "uptrend override" in why
    # never raises a tier; WAIT and unknown posture pass through
    assert cca_call_cap(A.CALL_DEFENSIVE, "HIGHER", "Base") == (A.CALL_DEFENSIVE, None)
    assert cca_call_cap(A.WAIT, "CONSERVATIVE", "Bull Trend") == (A.WAIT, None)
    assert cca_call_cap(A.CALL_AGGRESSIVE, None, None) == (A.CALL_AGGRESSIVE, None)


# --------------------------------------------------------------- recommend_call

def test_recommend_call_classifies_and_flags_income_wait():
    from rlbot.assistant.daily import recommend_call
    # level far above any in-band call -> INCOME_WAIT review, still SELL_CALL
    rec = recommend_call("T", _frame(), PS, 100_000.0,
                         mcb=_mcb(cca=140.0, cc_posture="STANDARD"))
    assert rec["action"] == "SELL_CALL"
    assert rec["contract"]["type"] == "CALL"
    assert rec["cca"]["classification"] == "INCOME_WAIT"
    assert rec["cca"]["shortfall"] > 0
    assert any("CCA:INCOME_WAIT" in w for w in rec["review_warnings"])
    assert rec["cca"]["reference_only"] is True          # no shares held
    # level at/below the chosen exit -> acceptable, no warning
    rec2 = recommend_call("T", _frame(), PS, 100_000.0,
                          mcb=_mcb(cca=90.0, cc_posture="HIGHER"))
    assert rec2["cca"]["classification"] == "ASSIGNMENT_ACCEPTABLE"
    assert "review_warnings" not in rec2


def test_recommend_call_tier_cap_and_basis_floor():
    from rlbot.assistant.daily import recommend_call
    # STANDARD posture caps the policy's tier at CALL_BALANCED
    rec = recommend_call("T", _frame(val=2), PS, 100_000.0,    # EXPENSIVE -> AGGRESSIVE rule
                         mcb=_mcb(cc_posture="STANDARD"))
    assert rec["policy_action"] in ("CALL_BALANCED", "CALL_CONSERVATIVE", "CALL_DEFENSIVE")
    if "policy_action_raw" in rec:
        assert "tier capped" in rec["cca"]["cap_reason"]
    # held shares + basis: calls never below basis (cost_basis filter)
    rec2 = recommend_call("T", _frame(), PS, 100_000.0,
                          mcb=_mcb(position_shares=300, position_basis=104.0,
                                   cc_posture="HIGHER", cca=104.0))
    assert rec2["cca"]["shares"] == 300 and rec2["cca"]["reference_only"] is False
    if rec2["action"] == "SELL_CALL":
        assert rec2["contract"]["strike"] >= 104.0


def test_recommend_call_below_basis_shows_blocked_reference():
    from rlbot.assistant.daily import recommend_call
    # spot 100, basis 150: no call at/above basis -> WAIT, but the row still
    # carries the tier's contract as a blocked reference (never executable)
    rec = recommend_call("T", _frame(), PS, 100_000.0,
                         mcb=_mcb(position_shares=200, position_basis=150.0,
                                  cc_posture="HIGHER", cca=150.0))
    assert rec["action"] == "WAIT" and "cost basis" in rec["reason"]
    assert rec["candidate_contract"]["basis"] == "blocked_by_cost_basis"
    assert rec["candidate_contract"]["strike"] < 150.0
    assert rec["cca"]["classification"] == "INCOME_WAIT"
    assert "review_warnings" not in rec           # not a trade; nothing to approve


def test_recommend_call_without_mcb_is_constraint_absent():
    from rlbot.assistant.daily import recommend_call
    rec = recommend_call("T", _frame(), PS, 100_000.0)
    assert rec["action"] in ("SELL_CALL", "WAIT")
    assert rec["cca"]["level"] is None and rec["cca"].get("classification") is None


# ------------------------------------------------------------------ rendering

def test_brief_renders_cc_section_dd_columns_and_legend():
    from rlbot.assistant.daily import recommend_call, recommend_opening, render_brief
    put = recommend_opening("PP", _frame(), PS, 100_000.0, mcb=_mcb(wheel_entry=60.0,
                                                                     mcb={"FAIR": 60.0}))
    call = recommend_call("PP", _frame(), PS, 100_000.0, mcb=_mcb(cca=140.0))
    text = render_brief("2026-08-27", [put], [], [], [call])
    # pullback columns: dollar move + resulting price, % secondary; no
    # ref_high in the fixture -> applied to spot and labelled illustrative
    assert "| Typical historical pullback | Larger historical pullback |" in text
    row = [ln for ln in text.splitlines() if ln.startswith("| PP |")][0]
    assert "90.00 (−10.00, 10.0%) ~illustrative from today's price" in row
    assert "84.00 (−16.00, 16.0%) ~illustrative" in row
    assert "| 100.00 |" in row                                   # Price column
    # CC section: no shares -> hypothetical watchlist, not the holdings table
    assert "## Covered-call recommendations (stock sleeve)" in text
    assert "### Hypothetical watchlist — no shares held" in text
    assert "_No holdings with ≥100 shares on the positions tab._" in text
    assert "140.00 (TRIM)" in text and "INCOME_WAIT (−" in text
    assert "**Reference only** | no shares held — hypothetical" in text
    # legend defines Ceiling as the MCB value and CCA
    assert "**Ceiling** | **The MCB value**" in text
    assert "**CCA** | **Covered-Call Assignment level**" in text
    assert "Typical / Larger historical pullback" in text


def test_render_brief_without_calls_unchanged():
    from rlbot.assistant.daily import render_brief
    text = render_brief("2026-08-27", [], [], [])
    assert "Covered-call recommendations" not in text


# ------------------------------------------------- 2026-09-11 report changes

def test_pullback_scenarios_from_reference_high_and_illustrative():
    from rlbot.assistant.daily import pullback_scenarios
    p = pullback_scenarios(200.0, 150.0, 0.18, 0.25)
    assert p["basis"] == "ref_high"
    assert p["typical"] == {"pct": 0.18, "drop": 36.0, "price": 164.0}
    assert p["larger"]["price"] == 150.0
    q = pullback_scenarios(None, 150.0, 0.18, None)
    assert q["basis"] == "spot" and q["typical"]["price"] == 123.0 and q["larger"] is None
    assert pullback_scenarios(None, None, 0.1, 0.2) is None


def test_high_date_found_in_close_series():
    from rlbot.assistant.daily import high_date
    idx = pd.bdate_range("2026-01-05", periods=5)
    f = pd.DataFrame({"close": [100, 120, 110, 119.9, 105]}, index=idx)
    assert high_date(f, 120.0) == "2026-01-06"
    assert high_date(f, 200.0) is None            # not in the series -> no guess
    assert high_date(f, None) is None


def test_week_label_calendar_dates():
    from rlbot.assistant.daily import week_label
    assert week_label(2026, 41) == "October 5–9, 2026"
    assert week_label(2026, 40) == "September 28–October 2, 2026"


def test_decision_status_put_rows():
    from rlbot.assistant.daily import decision_status
    assert decision_status({"action": "SELL_PUT"})["status"] == "Candidate"
    d = decision_status({"action": "SELL_PUT", "review_warnings": [
        "RISK-5-CUM:week_cap_if_all_executed — ...", "RISK-7:earnings_review — ..."]})
    assert d["status"] == "Candidate — review required"
    assert "week escrow cap" in d["reason"] and "earnings" in d["reason"]
    d = decision_status({"ticker": "MSFT", "action": "WAIT",
                         "reason": "risk engine: ['RISK-3:concentration']",
                         "risk_detail": {"flags": ["RISK-3:concentration"],
                                         "exposure_pct": 0.244, "exposure_cap_pct": 0.15}})
    assert d["status"] == "Blocked — position limit"
    assert "concentration limit" in d["reason"] and "24.4%" in d["reason"] and "15%" in d["reason"]
    assert decision_status({"action": "WAIT", "reason": "rule policy: ..."})["status"] == "Reference only"
    d = decision_status({"action": "WAIT", "reason": "MCB unreachable within normal delta bands ...",
                         "mcb": {"ceiling": 50.05, "posture": "CONSERVATIVE", "downtrend_override": True}})
    assert d["status"] == "Blocked — MCB ceiling" and "downtrend override" in d["reason"]
    assert decision_status({"action": "WAIT", "reason": "every viable expiry ..."})["status"] == "Blocked — position limit"


def test_call_decision_status_and_coverage_counts():
    from rlbot.assistant.daily import call_decision_status, recommend_call
    from rlbot.risk.book import BookState
    # 400 shares, 4 open CC contracts -> nothing left to cover
    book = BookState(cc_shares={"T": 400})
    rec = recommend_call("T", _frame(), PS, 100_000.0, book=book,
                         mcb=_mcb(position_shares=400, position_basis=90.0,
                                  cc_posture="HIGHER", cca=95.0))
    assert rec["cca"]["shares"] == 400 and rec["cca"]["committed_shares"] == 400
    assert rec["cca"]["available_contracts"] == 0
    d = call_decision_status(rec)
    assert d["status"] == "Blocked — no uncovered shares" and "400 already committed" in d["reason"]
    # 950 shares, no open calls -> 9 contracts coverable
    rec2 = recommend_call("T", _frame(), PS, 100_000.0, book=BookState(),
                          mcb=_mcb(position_shares=950, position_basis=90.0,
                                   cc_posture="HIGHER", cca=95.0))
    assert rec2["cca"]["available_contracts"] == 9
    if rec2["action"] == "SELL_CALL":
        assert call_decision_status(rec2)["status"].startswith("Candidate")
    # open CCs but no MCB share count -> shares inferred = committed, none free
    rec3 = recommend_call("T", _frame(), PS, 100_000.0, book=BookState(cc_shares={"T": 200}))
    assert rec3["cca"]["shares"] == 200 and rec3["cca"]["available_contracts"] == 0


def test_capital_summary_and_rendering():
    from rlbot.assistant.daily import capital_summary, render_brief
    from rlbot.risk.book import BookState
    from rlbot.risk.engine import RiskConfig
    book = BookState(put_escrow=60_500.0, expiry_week_escrow={(2026, 41): 60_500.0},
                     put_positions=[{"ticker": "Z", "strike": 605.0, "expiration": "2026-10-09",
                                     "escrow": 60_500.0}])
    recs = [{"ticker": "AAPL", "action": "SELL_PUT",
             "contract": {"strike": 300.0, "expiration": "2026-10-09"}},
            {"ticker": "AMD", "action": "SELL_PUT",
             "contract": {"strike": 465.0, "expiration": "2026-10-09"}},
            {"ticker": "BRK-B", "action": "SELL_PUT",
             "contract": {"strike": 485.0, "expiration": "2026-10-23"}}]
    stock = {"MSFT": {"shares": 400, "spot": 492.44, "value": 196_976.0}}
    cap = capital_summary(book, 1_000_000.0, recs, stock, RiskConfig())
    assert cap["available_verified"] is False
    assert cap["available_cash"] == 1_000_000.0 - 60_500.0 - 196_976.0
    assert cap["proposed_escrow"] == 125_000.0 and cap["n_proposed"] == 3
    w41 = [w for w in cap["weeks"] if w["iso"] == "2026-W41"][0]
    assert w41["label"] == "October 5–9, 2026"
    assert w41["existing"] == 60_500.0 and w41["remaining"] == 89_500.0
    assert w41["proposed"] == 76_500.0 and w41["over_by"] == 0.0
    w43 = [w for w in cap["weeks"] if w["iso"] == "2026-W43"][0]
    assert w43["existing"] == 0.0 and w43["proposed"] == 48_500.0
    # verified figure supplied -> labelled verified, used as-is
    capv = capital_summary(book, 1_000_000.0, recs, stock, RiskConfig(), available_cash=500_000.0)
    assert capv["available_verified"] and capv["available_cash"] == 500_000.0
    text = render_brief("2026-09-10", [], [], [], None, cap)
    assert "## Capital and limits" in text
    assert "| Total account value (NAV) | $1,000,000 |" in text
    assert "| Cash reserved for open puts (escrow) | $60,500 |" in text
    assert "| Stock sleeve (shares × last close) | $196,976 | MSFT 400 × 492.44 |" in text
    assert "| Estimated cash available for new trades | $742,524 |" in text
    assert "### Expiration-week capacity (cap 15% of NAV = $150,000 per expiration week)" in text
    assert "| October 5–9, 2026 | $60,500 | $89,500 | $76,500 (AAPL, AMD) | $137,000 (13.7%) | within cap |" in text
    assert "capacity example, not a ranking" in text
    textv = render_brief("2026-09-10", [], [], [], None, capv)
    assert "| Verified cash available for new trades | $500,000 |" in textv


def test_capital_summary_flags_negative_estimate():
    from rlbot.assistant.daily import capital_summary, render_brief
    from rlbot.risk.book import BookState
    from rlbot.risk.engine import RiskConfig
    book = BookState(put_escrow=630_500.0)
    stock = {"NOW": {"shares": 1500, "spot": 131.17, "value": 196_755.0}}
    cap = capital_summary(book, 800_000.0, [], stock, RiskConfig())
    assert cap["inconsistent"] is True and cap["available_cash"] < 0
    text = render_brief("2026-09-10", [], [], [], None, cap)
    assert "| ⚠ Inconsistent inputs |" in text and "not fully cash-secured" in text
    capv = capital_summary(book, 800_000.0, [], stock, RiskConfig(), available_cash=50_000.0)
    assert capv["inconsistent"] is False


def test_call_status_thin_chain_above_basis_is_not_cost_basis_block():
    from rlbot.assistant.daily import call_decision_status
    rec = {"action": "WAIT", "spot": 53.18,
           "reason": "tier unimplementable in current chain window (no call strike at/above cost basis)",
           "cca": {"shares": 950, "committed_shares": 0, "available_contracts": 9, "basis": 52.45}}
    d = call_decision_status(rec)
    assert d["status"] == "No contract" and "thin chain" in d["reason"]
    rec["spot"] = 40.0
    d = call_decision_status(rec)
    assert d["status"] == "Blocked — cost basis" and "40.00 below basis 52.45" in d["reason"]
