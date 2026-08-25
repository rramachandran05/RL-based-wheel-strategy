"""SPEC-008 tests: opening recommendation (AC-1), position guidance (AC-2),
record validation (REQ-8.2), positions degradation (REQ-8.3)."""
import pandas as pd
import pytest

from rlbot.assistant.daily import (
    decision_record,
    guide_position,
    load_positions,
    recommend_opening,
)
from rlbot.learning.trajectories import validate_record
from rlbot.options.premium_source import SyntheticBSPremiumSource
from rlbot.state.enums import PUT_DELTA_BANDS, CashAction
from tests.test_environment import make_frame

PS = SyntheticBSPremiumSource(iv_uplift=0.1)


def test_opening_recommendation_in_band():
    frame = make_frame(n=300, seed=5, regime=0, vol_comp=1)   # bull/fair -> BALANCED
    rec = recommend_opening("TEST", frame, PS, cash=100_000)
    assert rec["action"] == "SELL_PUT"
    assert rec["policy_action"] == "PUT_BALANCED"
    lo, hi = PUT_DELTA_BANDS[CashAction.PUT_BALANCED]
    assert lo - 0.02 <= abs(rec["contract"]["delta"]) <= hi + 0.02
    assert 25 <= rec["contract"]["dte"] <= 45


def test_opening_wait_in_stress():
    frame = make_frame(n=300, seed=5, regime=3, vol_comp=1)   # bear + normal vc
    rec = recommend_opening("TEST", frame, PS, cash=100_000)
    assert rec["action"] == "WAIT"


def test_position_guidance_flags():
    frame = make_frame(n=300, seed=5)
    spot = float(frame["close"].iloc[-1])
    exp = str((frame.index[-1] + pd.Timedelta(days=5)).date())
    safe_put = {"ticker": "TEST", "type": "CSP", "strike": spot * 0.8,
                "expiration": exp, "premium_fill": 2.0}
    g = guide_position(safe_put, frame, PS)
    assert g["guidance"] == "HOLD"
    assert "expiry week" in "; ".join(g["attention_flags"])
    breached_call = {"ticker": "TEST", "type": "CC", "strike": spot * 0.9,
                     "expiration": exp, "premium_fill": 1.5}
    g2 = guide_position(breached_call, frame, PS)
    assert g2["guidance"] == "HOLD"
    assert any("BREACHED" in f for f in g2["attention_flags"])


def test_decision_record_validates():
    frame = make_frame(n=300, seed=5, regime=0, vol_comp=1)
    rec = recommend_opening("TEST", frame, PS, cash=100_000)
    record = decision_record(rec, 100_000, "live-test", 0)
    validate_record(record)
    assert record["schema_version"] == "trajectory_v2"
    assert record["reward"] is None                # outcomes attach later


def test_positions_file_degradation(tmp_path):
    rows, warn = load_positions(tmp_path / "nope.csv")
    assert rows == [] and "not found" in warn
    bad = tmp_path / "bad.csv"
    bad.write_text("ticker,oops\nAAPL,1\n")
    rows, warn = load_positions(bad)
    assert rows == [] and "missing columns" in warn
    good = tmp_path / "good.csv"
    good.write_text("# comment line\nticker,type,strike,expiration,premium_fill\n"
                    "AAPL,CSP,180,2026-09-18,2.10\n")
    rows, warn = load_positions(good)
    assert warn is None and rows[0]["ticker"] == "AAPL"


def test_brief_includes_legend():
    from rlbot.assistant.daily import LEGEND, render_brief
    md = render_brief("2026-08-24", [], [], [])
    assert "## Legend" in md
    for term in ("BULL_LOW_VOL", "BEAR_STRESS", "PUT_BALANCED", "TipRanks",
                 "challenged", "0.25–0.35"):
        assert term in md, term
