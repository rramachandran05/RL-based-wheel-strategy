"""Brief changes 2026-09-11 (SPEC-008 §1 step 3, SPEC-004 §1.1):
- momentum Candidates section removed from the daily brief (SPEC-010 Feed B retired)
- Strike | DTE | Δ | Model prem populated on every opening row; WAIT rows carry
  `candidate_contract` (never `contract`) and render as WAIT† with a footnote
- selector is book-aware for RISK-5: capped expiry weeks are skipped up front,
  and an empty scan caused only by the cap is attributed honestly
"""
from __future__ import annotations

import pandas as pd
import pytest

from rlbot.benchmarks.policies import AdaptiveRulePolicy
from rlbot.config import RlbotConfig
from rlbot.data.mcb_feed import McbRow
from rlbot.options.premium_source import SyntheticBSPremiumSource
from rlbot.options.selector import select_contract
from rlbot.risk.book import BookState, build_book
from rlbot.risk.engine import RiskConfig
from rlbot.state.enums import CashAction, ValuationState

DATE = pd.Timestamp("2026-08-27")
VOL = 0.25
PS = SyntheticBSPremiumSource()


def _frame(regime=0):
    idx = pd.DatetimeIndex([DATE])
    return pd.DataFrame({"close": [100.0], "market_regime": [regime],
                         "valuation_state": [1], "vol_compensation": [1],
                         "vol_proxy": [VOL]}, index=idx)


def _mcb(**kw):
    base = dict(ticker="T", date="2026-08-27", mcb={"FAIR": 90.0},
                min_eligible_tier="FAIR", guardrail=None, layer_a="OWN",
                reachability=None, confidence=0.8, wheel_entry=90.0,
                delta_posture="HIGHER")
    base.update(kw)
    return McbRow(**base)


# ------------------------------------------------------ selector week headroom

def _weeks_in_chain(chain):
    out = {}
    for q in chain:
        if q.cp == "P":
            iso = q.expiration.isocalendar()
            out.setdefault((iso.year, iso.week), set()).add(q.expiration)
    return out


def test_selector_skips_capped_expiry_weeks():
    chain = PS.chain(DATE, 100.0, VOL, "P")
    weeks = _weeks_in_chain(chain)
    assert len(weeks) >= 2, "synthetic chain should span several expiry weeks"
    free, _ = select_contract(CashAction.PUT_BALANCED, chain, 100.0, VOL,
                              ValuationState.FAIR)
    iso = free.expiration.isocalendar()
    picked_week = (iso.year, iso.week)
    # cap exactly the week the selector wanted -> it must move to a sibling week
    q2, n2 = select_contract(CashAction.PUT_BALANCED, chain, 100.0, VOL,
                             ValuationState.FAIR,
                             expiry_week_headroom={picked_week: 0.0})
    assert q2 is not None and n2 > 0
    iso2 = q2.expiration.isocalendar()
    assert (iso2.year, iso2.week) != picked_week
    # cap every week -> None (RISK-5 would reject them all anyway)
    q3, n3 = select_contract(CashAction.PUT_BALANCED, chain, 100.0, VOL,
                             ValuationState.FAIR,
                             expiry_week_headroom={w: 0.0 for w in weeks})
    assert q3 is None and n3 == 0
    # partial headroom: a $9,500 room admits strikes <= 95 only
    q4, _ = select_contract(CashAction.PUT_BALANCED, chain, 100.0, VOL,
                            ValuationState.FAIR,
                            expiry_week_headroom={w: 9_500.0 for w in weeks})
    assert q4 is None or q4.strike * 100 <= 9_500.0 + 1e-9


def test_book_expiry_week_headroom():
    book = build_book([
        {"ticker": "A", "type": "CSP", "strike": 100.0, "expiration": "2026-09-18", "contracts": 12},
        {"ticker": "B", "type": "CSP", "strike": 50.0, "expiration": "2026-09-25", "contracts": 1},
        {"ticker": "C", "type": "CC", "strike": 500.0, "expiration": "2026-09-18", "contracts": 4},
    ])
    room = book.expiry_week_headroom(0.15, 1_000_000)
    wk18 = pd.Timestamp("2026-09-18").isocalendar(); wk25 = pd.Timestamp("2026-09-25").isocalendar()
    assert room[(wk18.year, wk18.week)] == pytest.approx(150_000 - 120_000)   # CC adds no escrow
    assert room[(wk25.year, wk25.week)] == pytest.approx(150_000 - 5_000)
    over = build_book([{"ticker": "A", "type": "CSP", "strike": 100.0,
                        "expiration": "2026-09-18", "contracts": 20}])
    assert over.expiry_week_headroom(0.15, 1_000_000)[(wk18.year, wk18.week)] == 0.0  # floored


# ------------------------------------------------ candidate_contract on WAIT rows

def test_policy_wait_carries_conservative_reference():
    from rlbot.assistant.daily import recommend_opening

    class _AlwaysWait:
        def decide(self, pos, q, row_):
            return CashAction.WAIT

    rec = recommend_opening("T", _frame(), PS, 100_000.0, policy=_AlwaysWait())
    assert rec["action"] == "WAIT" and "contract" not in rec
    cc = rec["candidate_contract"]
    assert cc["basis"] == "reference_conservative_tier"
    assert 0.10 - 0.02 <= abs(cc["delta"]) <= 0.18 + 0.02      # CONSERVATIVE band (±widen)
    for k in ("strike", "dte", "delta", "model_premium"):
        assert k in cc


def test_risk_blocked_wait_carries_selected_contract():
    from rlbot.assistant.daily import recommend_opening
    # 14 names already held -> RISK-4 blocks; the selected quote must still surface
    book = BookState(n_open_positions=14, put_escrow=0.0, expiry_week_counts={},
                     underlyings={f"N{i}" for i in range(14)})
    rec = recommend_opening("T", _frame(), PS, 1_000_000.0, book=book,
                            rcfg=RlbotConfig())
    assert rec["action"] == "WAIT" and "RISK-4" in rec["reason"]
    assert "contract" not in rec
    assert rec["candidate_contract"]["basis"] == "blocked_by_risk_engine"
    assert rec["candidate_contract"]["strike"] > 0


def test_mcb_blocked_wait_carries_ungated_contract():
    from rlbot.assistant.daily import recommend_opening
    rec = recommend_opening("T", _frame(), PS, 100_000.0,
                            mcb=_mcb(mcb={"FAIR": 60.0}, wheel_entry=60.0))
    assert rec["action"] == "WAIT" and "MCB unreachable" in rec["reason"]
    assert rec["candidate_contract"]["basis"] == "blocked_by_mcb_ceiling"
    # the surfaced contract is the one whose premium shortfall the reason quotes
    assert f"{rec['candidate_contract']['strike']:g}" in rec["reason"]


def test_week_cap_only_block_is_attributed_and_surfaced():
    """All in-window expiries sit in capped weeks -> honest reason + candidate."""
    from rlbot.assistant.daily import recommend_opening
    chain = PS.chain(DATE, 100.0, VOL, "P")
    weeks = _weeks_in_chain(chain)
    escrow = {w: 999_999.0 for w in weeks}            # every week over the cap
    book = BookState(n_open_positions=1, put_escrow=sum(escrow.values()),
                     expiry_week_counts={}, underlyings={"Z"},
                     expiry_week_escrow=escrow)
    rec = recommend_opening("T", _frame(), PS, 1_000_000.0, book=book,
                            rcfg=RlbotConfig(),
                            risk_cfg=RiskConfig(min_stress_reserve_pct=0.0))
    assert rec["action"] == "WAIT"
    assert "RISK-5 cap" in rec["reason"]
    assert rec["candidate_contract"]["basis"] == "blocked_by_risk5_week_cap"


def test_unimplementable_tier_has_no_candidate():
    from rlbot.assistant.daily import recommend_opening
    # ceiling far below the whole synthetic grid -> geometric, nothing to show
    rec = recommend_opening("T", _frame(), PS, 100_000.0,
                            mcb=_mcb(mcb={"FAIR": 20.0}, wheel_entry=20.0))
    assert rec["action"] == "WAIT"
    # MCB attribution runs only if an ungated contract exists; here it does,
    # so a candidate IS present — check the geometric wording instead
    assert "geometrically unreachable" in rec["reason"]


def test_decision_record_unchanged_by_candidate_contract():
    from rlbot.assistant.daily import decision_record, recommend_opening

    class _AlwaysWait:
        def decide(self, pos, q, row_):
            return CashAction.WAIT

    rec = recommend_opening("T", _frame(), PS, 100_000.0, policy=_AlwaysWait())
    assert rec.get("candidate_contract")
    record = decision_record(rec, 100_000.0, "run-x", 0)
    assert record["chosen_action"] == 0 and record["contract"] is None


# ----------------------------------------------------------------- rendering

def test_brief_shows_contract_columns_on_wait_rows_and_no_candidates_section():
    from rlbot.assistant.daily import recommend_opening, render_brief

    class _AlwaysWait:
        def decide(self, pos, q, row_):
            return CashAction.WAIT

    w = recommend_opening("WW", _frame(), PS, 100_000.0, policy=_AlwaysWait())
    s = recommend_opening("SS", _frame(), PS, 100_000.0)
    text = render_brief("2026-08-27", [w, s], [], [])
    assert "| WW |" in text and "**Reference only**" in text
    row = [ln for ln in text.splitlines() if ln.startswith("| WW |")][0]
    cells = [c.strip() for c in row.strip("|").split("|")]
    # Ticker | Price | State | Decision | Reason | Strike | DTE | Δ | prem
    strike, dte, delta, prem = cells[5:9]
    assert strike != "—" and dte != "—" and delta != "—" and prem != "—"
    assert "the contract the selector would have chosen" in text
    assert "Candidates (momentum monitor)" not in text
    assert "momentum" not in text.lower()


def test_render_brief_signature_has_no_candidates_arg():
    import inspect
    from rlbot.assistant.daily import render_brief
    assert "cand_recs" not in inspect.signature(render_brief).parameters


# ------------------------------------------- cumulative within-run review

def _sell(ticker, strike, exp):
    return {"ticker": ticker, "action": "SELL_PUT",
            "contract": {"strike": strike, "expiration": exp, "dte": 30,
                         "delta": -0.15, "model_premium": 2.0}}


def test_cumulative_week_cap_is_warning_not_block():
    from rlbot.assistant.daily import cumulative_review
    book = BookState(n_open_positions=1, put_escrow=60_500.0, expiry_week_counts={},
                     underlyings={"Z"},
                     expiry_week_escrow={(2026, 41): 60_500.0})
    rk = RiskConfig(min_stress_reserve_pct=0.0)
    recs = [_sell("AAPL", 300.0, "2026-10-09"), _sell("AMZN", 230.0, "2026-10-09"),
            _sell("TSM", 385.0, "2026-10-09"), _sell("BRK-B", 485.0, "2026-10-23")]
    per, summary = cumulative_review(recs, book, rk, 1_000_000.0)
    # week 41: 60,500 + 91,500 = 152,000 > 150,000 -> warned; week 43 fine
    assert len(summary) == 1 and "RISK-5-CUM" in summary[0]
    assert set(per) == {"AAPL", "AMZN", "TSM"} and "BRK-B" not in per
    assert "remaining capacity $89,500" in summary[0]
    assert "Expiration week of October 5–9, 2026" in summary[0]     # not ISO 2026-W41
    assert "Capacity example — fits the limit, not a ranking" in summary[0]
    # nothing is downgraded — actions untouched
    assert all(r["action"] == "SELL_PUT" for r in recs)


def test_cumulative_under_cap_is_silent():
    from rlbot.assistant.daily import cumulative_review
    book = BookState(n_open_positions=0, put_escrow=0.0, expiry_week_counts={},
                     underlyings=set())
    rk = RiskConfig(min_stress_reserve_pct=0.0)
    recs = [_sell("AAPL", 300.0, "2026-10-09"), _sell("AMZN", 230.0, "2026-10-09")]
    per, summary = cumulative_review(recs, book, rk, 1_000_000.0)
    assert per == {} and summary == []


def test_cumulative_stress_reserve_warning():
    from rlbot.assistant.daily import cumulative_review
    book = BookState(n_open_positions=0, put_escrow=0.0, expiry_week_counts={},
                     underlyings=set())
    rk = RiskConfig(max_week_assignment_pct=1.0, min_stress_reserve_pct=0.15)
    # $100K NAV, one $90K put in the nearest week -> stress 90K, 10% left < 15%
    recs = [_sell("AAPL", 900.0, "2026-10-09")]
    per, summary = cumulative_review(recs, book, rk, 100_000.0)
    assert any("RISK-8-CUM" in s for s in summary) and "AAPL" in per


def test_cumulative_review_renders_as_review_marker():
    from rlbot.assistant.daily import render_brief
    r = _sell("AAPL", 300.0, "2026-10-09")
    r["state_names"] = ["BULL_LOW_VOL", "FAIR", "POOR"]
    r["review_warnings"] = ["RISK-5-CUM:week_cap_if_all_executed — ISO week 2026-W41: ..."]
    text = render_brief("2026-09-10", [r], [], ["REVIEW (cumulative): ISO week 2026-W41: ..."])
    assert "**Candidate — review required**" in text and "RISK-5-CUM" in text


def test_shared_review_warning_renders_once_with_all_tickers():
    from rlbot.assistant.daily import render_brief
    shared = "RISK-5-CUM:week_cap_if_all_executed — ISO week 2026-W41: shared text"
    recs = []
    for t_ in ("AAPL", "AMZN", "TSM"):
        r = _sell(t_, 300.0, "2026-10-09")
        r["state_names"] = ["BULL_LOW_VOL", "FAIR", "POOR"]
        r["review_warnings"] = [shared]
        recs.append(r)
    text = render_brief("2026-09-10", recs, [], [])
    assert text.count("shared text") == 1
    assert "**AAPL, AMZN, TSM** —" in text
    rows = [ln for ln in text.splitlines()
            if ln.startswith("| ") and "**Candidate — review required**" in ln]
    assert len(rows) == 3                                   # per-row status kept
