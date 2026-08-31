"""MCB feed + gates (2026-08-30): mcb-wheel's report replaces the Wheel-FV
feed as the daily assistant's valuation input. Covers the consumer contract:
hard net-basis ceiling at the required tier (deeper of report tier and
regime posture), never trade MONITOR_ONLY/HALT, reachability advisory,
5-session staleness, NaN rows -> constraint absent.
"""
from __future__ import annotations

import pandas as pd
import pytest

from rlbot.data.mcb_feed import McbRow, load_mcb
from rlbot.options.premium_source import SyntheticBSPremiumSource
from rlbot.options.selector import select_contract
from rlbot.risk.mcb_gates import (mcb_ceiling, net_basis_flag,
                                  premium_required, reachability_advice,
                                  required_tier, tradeable)
from rlbot.state.enums import CashAction, MarketRegime, ValuationState

DATE = pd.Timestamp("2026-08-27")
VOL = 0.25
PS = SyntheticBSPremiumSource()

CSV_COLS = ("ticker,date,spot,archetype,char_flags,layer_a,conf,a1_fair,"
            "a2_fair,a3_fair,a4,a4_strength,current_pe,mcb_fair,"
            "mcb_attractive,mcb_excellent,iv_ratio,guardrail_status,"
            "min_eligible_tier,mcb_fair_low,mcb_fair_high,corr_typical,"
            "corr_25pct,corr_bear,reachability,ref_high,dd50,dd75,dd90,"
            "etf_subtype,flags,sheet_section")


def _row(t, fair="90", attr="85", exc="80", guard="NORMAL", tier="FAIR",
         layer="OWN", reach="NORMAL", conf="0.8"):
    return (f"{t},2026-08-27,100,COMPOUNDER,,{layer},{conf},,,,,,,"
            f"{fair},{attr},{exc},1.0,{guard},{tier},,,,,,{reach},,,,,,,CORE")


def _write_report(tmp_path, rows, date="2026-08-27"):
    d = tmp_path / "outputs"
    d.mkdir(exist_ok=True)
    (d / f"mcb_{date}.csv").write_text("\n".join([CSV_COLS] + rows) + "\n")

    class _Data:
        mcb_dir = d

    class _Cfg:
        data = _Data()

    return _Cfg()


def _mcb(**kw):
    base = dict(ticker="T", date="2026-08-27",
                mcb={"FAIR": 90.0, "ATTRACTIVE": 85.0, "EXCELLENT": 80.0},
                min_eligible_tier="FAIR", guardrail="NORMAL", layer_a="OWN",
                reachability="NORMAL", confidence=0.8)
    base.update(kw)
    return McbRow(**base)


# ---------------------------------------------------------------- feed

def test_load_mcb_parses_and_is_pit_safe(tmp_path):
    cfg = _write_report(tmp_path, [_row("AAPL"), _row("MSFT", guard="SEVERE",
                                                      tier="EXCELLENT")])
    rows, warns = load_mcb(cfg, as_of="2026-08-28")
    assert set(rows) == {"AAPL", "MSFT"}
    assert rows["AAPL"].mcb["FAIR"] == 90.0
    assert rows["MSFT"].min_eligible_tier == "EXCELLENT"
    # a report dated after as_of is invisible (PIT)
    rows2, _ = load_mcb(cfg, as_of="2026-08-26")
    assert rows2 == {}


def test_load_mcb_staleness_expires_report(tmp_path):
    cfg = _write_report(tmp_path, [_row("AAPL")], date="2026-08-10")
    rows, warns = load_mcb(cfg, as_of="2026-08-28")
    assert rows == {} and any("expired" in w for w in warns)


def test_load_mcb_nan_row_is_constraint_absent(tmp_path):
    # TSM-style row: conf 0, every zone NaN -> absent, with a warning
    cfg = _write_report(tmp_path, [_row("AAPL"),
                                   _row("TSM", fair="", attr="", exc="",
                                        conf="0")])
    rows, warns = load_mcb(cfg, as_of="2026-08-28")
    assert "TSM" not in rows and "AAPL" in rows
    assert any("TSM" in w for w in warns)


# ---------------------------------------------------------------- tiers

def test_required_tier_deeper_of_report_and_posture():
    row = _mcb(min_eligible_tier="ATTRACTIVE")   # guardrail CAUTION upstream
    assert required_tier(row, int(MarketRegime.BULL_LOW_VOL)) == "ATTRACTIVE"
    assert required_tier(_mcb(), int(MarketRegime.BULL_LOW_VOL)) == "FAIR"
    # defensive regimes force at least ATTRACTIVE
    assert required_tier(_mcb(), int(MarketRegime.BEAR_STRESS)) == "ATTRACTIVE"
    # SEVERE guardrail dominates a calm market
    row2 = _mcb(min_eligible_tier="EXCELLENT")
    assert required_tier(row2, int(MarketRegime.BULL_LOW_VOL)) == "EXCELLENT"
    assert mcb_ceiling(row2, int(MarketRegime.BULL_LOW_VOL)) == 80.0
    assert mcb_ceiling(None, 0) is None


def test_required_tier_missing_zone_falls_back_conservatively():
    row = _mcb(mcb={"FAIR": 90.0, "ATTRACTIVE": 85.0},
               min_eligible_tier="EXCELLENT")
    # EXCELLENT zone missing -> nearest available shallower tier
    assert required_tier(row, 0) == "ATTRACTIVE"


# ---------------------------------------------------------------- masks

def test_tradeable_blocks_monitor_only_and_halt():
    assert tradeable(_mcb())[0]
    assert tradeable(None)[0]
    for layer in ("MONITOR_ONLY", "HALT"):
        ok, why = tradeable(_mcb(layer_a=layer))
        assert not ok and layer in why


def test_reachability_advice():
    assert reachability_advice(_mcb(), 1) is None
    assert "UNREACHABLE" in reachability_advice(
        _mcb(reachability="UNREACHABLE"), 2)
    # PATIENCE passes only with vol-comp ATTRACTIVE (=2)
    assert reachability_advice(_mcb(reachability="PATIENCE"), 2) is None
    assert "PATIENCE" in reachability_advice(_mcb(reachability="PATIENCE"), 1)
    assert reachability_advice(_mcb(reachability="UNREACHABLE"), 2,
                               honor=False) is None


def test_net_basis_flag_and_premium_required():
    assert net_basis_flag(90.0, 2.0, 88.5) is None          # 88.0 <= 88.5
    assert net_basis_flag(92.0, 2.0, 88.5) == "MCB-1:net_basis_above_ceiling"
    assert net_basis_flag(92.0, 2.0, None) is None
    assert premium_required(92.0, 88.5) == pytest.approx(3.5)
    assert premium_required(80.0, 88.5) == 0.0


# ---------------------------------------------------------------- selector

def test_selector_net_basis_ceiling_filters_puts():
    spot = 100.0
    chain = PS.chain(DATE, spot, VOL, "P")
    free_q, _ = select_contract(CashAction.PUT_BALANCED, chain, spot, VOL,
                                ValuationState.FAIR)
    assert free_q is not None
    # harsh ceiling: no 18-25 delta put lands a net basis <= 60
    gated, n = select_contract(CashAction.PUT_BALANCED, chain, spot, VOL,
                               ValuationState.FAIR, net_basis_ceiling=60.0)
    assert gated is None and n == 0
    # permissive ceiling: survivors all honor the bound
    q2, _ = select_contract(CashAction.PUT_BALANCED, chain, spot, VOL,
                            ValuationState.FAIR, net_basis_ceiling=95.0)
    assert q2 is not None and q2.strike - q2.mid <= 95.0 + 1e-9


# ---------------------------------------------------------------- daily

def _fake_frame(regime=0):
    idx = pd.DatetimeIndex([DATE])
    return pd.DataFrame({"close": [100.0], "market_regime": [regime],
                         "valuation_state": [1], "vol_compensation": [1],
                         "vol_proxy": [VOL]}, index=idx)


def test_recommend_opening_mcb_blocks_and_annotates():
    from rlbot.assistant.daily import recommend_opening
    # layer A HALT -> WAIT regardless of policy action
    rec = recommend_opening("T", _fake_frame(), PS, 100_000.0,
                            mcb=_mcb(layer_a="HALT"))
    assert rec["action"] == "WAIT" and "HALT" in rec["reason"]
    # UNREACHABLE -> WAIT (advisory strike-scan skip)
    rec2 = recommend_opening("T", _fake_frame(), PS, 100_000.0,
                             mcb=_mcb(reachability="UNREACHABLE"))
    assert rec2["action"] == "WAIT" and "UNREACHABLE" in rec2["reason"]
    # harsh ceiling -> WAIT with the MCB reason
    rec3 = recommend_opening("T", _fake_frame(), PS, 100_000.0,
                             mcb=_mcb(mcb={"FAIR": 60.0}))
    assert rec3["action"] == "WAIT" and "MCB" in rec3["reason"]
    # permissive ceiling -> trade, annotated with tier + premium required
    rec4 = recommend_opening("T", _fake_frame(), PS, 100_000.0,
                             mcb=_mcb(mcb={"FAIR": 95.0}))
    if rec4["action"] == "SELL_PUT":
        c, v = rec4["contract"], rec4["mcb"]
        assert v["tier"] == "FAIR"
        assert c["strike"] - c["model_premium"] <= v["ceiling"] + 0.01
        assert v["premium_required"] == pytest.approx(
            max(0.0, c["strike"] - v["ceiling"]), abs=0.01)
    # no MCB row -> classic behavior, no mcb key
    rec5 = recommend_opening("T", _fake_frame(), PS, 100_000.0)
    assert "mcb" not in rec5


def test_render_brief_includes_mcb_columns():
    from rlbot.assistant.daily import recommend_opening, render_brief
    rec = recommend_opening("T", _fake_frame(), PS, 100_000.0,
                            mcb=_mcb(layer_a="HALT", guardrail="SEVERE"))
    text = render_brief("2026-08-27", [rec], [], [])
    assert "MCB tier" in text and "Prem req" in text
    assert "HALT" in text
    assert "Maximum Comfortable Basis" in text        # note + legend


def test_guide_position_mcb_flags():
    from rlbot.assistant.daily import guide_position
    pos = {"ticker": "T", "type": "CSP", "strike": 99.0,
           "expiration": str((DATE + pd.Timedelta(days=20)).date()),
           "premium_fill": 1.0}
    g = guide_position(pos, _fake_frame(), PS, mcb=_mcb())   # ceiling 90
    assert any("MCB" in f and "ceiling" in f for f in g["attention_flags"])
    g_ok = guide_position(pos, _fake_frame(), PS,
                          mcb=_mcb(mcb={"FAIR": 99.0}))
    assert not any("ceiling" in f for f in g_ok["attention_flags"])
    # covered calls carry no MCB flag (acquisition-side construct)
    cc = {"ticker": "T", "type": "CC", "strike": 101.0,
          "expiration": str((DATE + pd.Timedelta(days=20)).date()),
          "premium_fill": 1.0}
    g2 = guide_position(cc, _fake_frame(), PS, mcb=_mcb(mcb={"FAIR": 50.0}))
    assert not any("MCB" in f and "ceiling" in f for f in g2["attention_flags"])
    # layer-A downgrade is surfaced on open puts
    g3 = guide_position(pos, _fake_frame(), PS,
                        mcb=_mcb(mcb={"FAIR": 99.0}, layer_a="MONITOR_ONLY"))
    assert any("MONITOR_ONLY" in f for f in g3["attention_flags"])
