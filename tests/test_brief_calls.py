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
    # DD columns present and populated on the put row
    assert "| DD50 | DD75 |" in text
    row = [ln for ln in text.splitlines() if ln.startswith("| PP |")][0]
    assert "10.0%" in row and "16.0%" in row
    # CC section, CCA column with mode, INCOME_WAIT check, reference marker
    assert "## Covered-call recommendations (stock sleeve)" in text
    assert "| CC posture | CCA | Exit check | Shares |" in text
    assert "140.00 (TRIM)" in text and "INCOME_WAIT (−" in text and "none (ref)" in text
    assert "Covered-call review (CCA INCOME_WAIT" in text
    # legend defines Ceiling as the MCB value and CCA
    assert "**Ceiling** | **The MCB value**" in text
    assert "**CCA** | **Covered-Call Assignment level**" in text
    assert "DD50 / DD75" in text


def test_render_brief_without_calls_unchanged():
    from rlbot.assistant.daily import render_brief
    text = render_brief("2026-08-27", [], [], [])
    assert "Covered-call recommendations" not in text
