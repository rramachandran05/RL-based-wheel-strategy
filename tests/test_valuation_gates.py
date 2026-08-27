"""SPEC-009 tests: ensemble ingestion, boundaries, action masks, engine
flags, selector pre-filter, and daily-brief integration (VAC-1..6)."""
import pandas as pd
import pytest

from rlbot.config import RlbotConfig, ValuationGateConfig
from rlbot.data.fv_ensemble import load_wheel_valuations
from rlbot.options.premium_source import SyntheticBSPremiumSource
from rlbot.options.selector import select_contract
from rlbot.risk.engine import RiskConfig, validate_open
from rlbot.risk.valuation import (WheelRegime, WheelValuation, clamp_action,
                                  allowed_actions, call_gate_flags,
                                  exit_floor, premium_required, put_ceiling,
                                  put_gate_flags, wheel_regime)
from rlbot.state.enums import CashAction, StockAction, ValuationState

DATE = pd.Timestamp("2024-03-11")
VOL = 0.30
PS = SyntheticBSPremiumSource()
VCFG = ValuationGateConfig()


def _val(fv=115.0, mos=0.15, **kw):
    return WheelValuation(ticker="T", date="2026-08-27", wheel_fv=fv,
                          put_required_mos=mos, **kw)


# ----------------------------------------------------------------------
# VAC-1: ingestion
# ----------------------------------------------------------------------

def _write_csv(path, date, rows):
    df = pd.DataFrame(rows)
    df.to_csv(path / f"fair_value_ensemble_{date}.csv", index=False)


def _cfg_for(tmp_path):
    cfg = RlbotConfig()
    cfg.data.fv_ensemble_dir = tmp_path      # DataConfig is non-frozen
    return cfg


def test_loader_parses_and_skips_empty_wheel_fv(tmp_path):
    _write_csv(tmp_path, "2026-08-27", [
        {"ticker": "AAPL", "wheel_fv": 202.0, "put_required_mos": 0.10,
         "iv_reliability": 0.92, "iv_reliability_tier": "high",
         "analyst_sentiment": 0.30, "analyst_coverage": 46},
        {"ticker": "CHPS", "wheel_fv": "", "put_required_mos": ""},
    ])
    vals, warns = load_wheel_valuations(_cfg_for(tmp_path),
                                        as_of=pd.Timestamp("2026-08-27"))
    assert set(vals) == {"AAPL"}
    v = vals["AAPL"]
    assert v.wheel_fv == 202.0 and v.put_required_mos == 0.10
    assert v.reliability_tier == "high" and v.coverage == 46


def test_loader_staleness_and_pit(tmp_path):
    _write_csv(tmp_path, "2026-08-01", [
        {"ticker": "AAPL", "wheel_fv": 202.0, "put_required_mos": 0.10}])
    vals, warns = load_wheel_valuations(_cfg_for(tmp_path),
                                        as_of=pd.Timestamp("2026-08-27"))
    assert vals == {} and any("old" in w for w in warns)      # stale -> off
    # PIT: a file dated in the future is never read
    _write_csv(tmp_path, "2026-09-15", [
        {"ticker": "MSFT", "wheel_fv": 500.0, "put_required_mos": 0.10}])
    vals2, _ = load_wheel_valuations(_cfg_for(tmp_path),
                                     as_of=pd.Timestamp("2026-08-27"))
    assert "MSFT" not in vals2


def test_loader_missing_dir_degrades(tmp_path):
    cfg = _cfg_for(tmp_path / "nope")
    vals, warns = load_wheel_valuations(cfg)
    assert vals == {} and warns


# ----------------------------------------------------------------------
# VAC-2: boundaries
# ----------------------------------------------------------------------

def test_put_ceiling_both_sides():
    # FV side binds: min(115*0.85, 105*0.95) = 97.75
    assert put_ceiling(_val(115, 0.15), 105.0, VCFG) == pytest.approx(97.75)
    # spot side binds when FV >> spot: min(170, 142.5) = 142.5
    assert put_ceiling(_val(200, 0.15), 150.0, VCFG) == pytest.approx(142.5)
    assert premium_required(100.0, 97.75) == pytest.approx(2.25)
    assert premium_required(95.0, 97.75) == 0.0


def test_exit_floor_max_of_economic_and_fundamental():
    v = _val(200, 0.15)
    assert exit_floor(v, None, VCFG) == pytest.approx(190.0)      # 0.95 x FV
    assert exit_floor(v, 160.0, VCFG) == pytest.approx(190.0)     # fund binds
    assert exit_floor(v, 185.0, VCFG) == pytest.approx(203.5)     # econ binds


def test_regime_bands():
    assert wheel_regime(79.9, 100) == WheelRegime.DEEP_UNDERVALUED
    assert wheel_regime(94.9, 100) == WheelRegime.UNDERVALUED
    assert wheel_regime(100, 100) == WheelRegime.FAIR_VALUED
    assert wheel_regime(120, 100) == WheelRegime.EXPENSIVE
    assert wheel_regime(121, 100) == WheelRegime.VERY_EXPENSIVE
    assert wheel_regime(100, 0) is None


# ----------------------------------------------------------------------
# VAC-3: action masks
# ----------------------------------------------------------------------

def test_masks_no_put_when_very_expensive():
    assert allowed_actions(CashAction, WheelRegime.VERY_EXPENSIVE) == \
        {CashAction.WAIT}
    for a in CashAction:
        assert clamp_action(a, WheelRegime.VERY_EXPENSIVE) == CashAction.WAIT


def test_masks_calls_capped_when_deep_undervalued():
    assert allowed_actions(StockAction, WheelRegime.DEEP_UNDERVALUED) == \
        {StockAction.WAIT, StockAction.CALL_DEFENSIVE}
    assert clamp_action(StockAction.CALL_AGGRESSIVE,
                        WheelRegime.DEEP_UNDERVALUED) == \
        StockAction.CALL_DEFENSIVE


def test_clamp_maps_to_highest_allowed_tier():
    assert clamp_action(CashAction.PUT_AGGRESSIVE,
                        WheelRegime.FAIR_VALUED) == CashAction.PUT_BALANCED
    assert clamp_action(CashAction.PUT_BALANCED,
                        WheelRegime.UNDERVALUED) == CashAction.PUT_BALANCED
    # unknown regime -> no masking
    assert clamp_action(CashAction.PUT_VERY_AGGRESSIVE, None) == \
        CashAction.PUT_VERY_AGGRESSIVE


# ----------------------------------------------------------------------
# VAC-4: risk-engine flags
# ----------------------------------------------------------------------

def _quote(cp="P", strike=100.0, mid=2.0):
    chain = PS.chain(DATE, 100.0, VOL, cp)
    q = chain[0].__class__(cp=cp, strike=strike, expiration=chain[0].expiration,
                           dte=30, mid=mid, delta=-0.2 if cp == "P" else 0.2,
                           vol_used=VOL)
    return q


def test_engine_val_flags_put():
    v = _val(115, 0.15)                      # ceiling at spot 105 = 97.75
    ok = validate_open(_quote(strike=99.0, mid=2.0), 1, 100_000, 0, 100_000,
                       0.0, False, RiskConfig.single_ticker(),
                       valuation=v, spot=105.0)
    assert ok.passed                          # net basis 97 <= 97.75
    bad = validate_open(_quote(strike=101.0, mid=2.0), 1, 100_000, 0,
                        100_000, 0.0, False, RiskConfig.single_ticker(),
                        valuation=v, spot=105.0)
    assert "VAL-1:net_basis_above_ceiling" in bad.flags
    exp = validate_open(_quote(strike=90.0, mid=2.0), 1, 100_000, 0, 100_000,
                        0.0, False, RiskConfig.single_ticker(),
                        valuation=_val(80, 0.15), spot=105.0)  # spot/FV 1.31
    assert "VAL-2:very_expensive_no_put" in exp.flags


def test_engine_val_flags_call_and_none_is_silent():
    v = _val(200, 0.15)                      # fundamental floor 190
    bad = validate_open(_quote("C", strike=180.0, mid=3.0), 1, 0, 100,
                        100_000, 0.0, False, RiskConfig.single_ticker(),
                        valuation=v, spot=195.0)
    assert "VAL-3:below_exit_floor" in bad.flags
    ok = validate_open(_quote("C", strike=190.0, mid=3.0), 1, 0, 100,
                       100_000, 0.0, False, RiskConfig.single_ticker(),
                       valuation=v, spot=195.0)
    assert "VAL-3:below_exit_floor" not in ok.flags
    # valuation=None -> engine behaves exactly as before
    legacy = validate_open(_quote(strike=101.0, mid=2.0), 1, 100_000, 0,
                           100_000, 0.0, False, RiskConfig.single_ticker())
    assert legacy.passed


# ----------------------------------------------------------------------
# VAC-5: selector pre-filter (NO_PUT / NO_CALL emerge as None)
# ----------------------------------------------------------------------

def test_selector_put_ceiling_filters_strikes():
    spot = 100.0
    chain = PS.chain(DATE, spot, VOL, "P")
    free_q, _ = select_contract(CashAction.PUT_BALANCED, chain, spot, VOL,
                                ValuationState.FAIR)
    assert free_q is not None
    # a harsh ceiling: FV 70 => ceiling min(59.5, 95) = 59.5; no 18-25d put
    # can land a net basis that low -> None (NO_PUT)
    gated, n = select_contract(CashAction.PUT_BALANCED, chain, spot, VOL,
                               ValuationState.FAIR,
                               valuation=_val(70, 0.15))
    assert gated is None and n == 0
    # a permissive ceiling keeps the tier implementable and every surviving
    # candidate honors the boundary
    v2 = _val(105, 0.05)                     # ceiling min(99.75, 95) = 95
    q2, n2 = select_contract(CashAction.PUT_BALANCED, chain, spot, VOL,
                             ValuationState.FAIR, valuation=v2)
    assert q2 is not None and q2.strike - q2.mid <= 95 + 1e-9


def test_selector_call_exit_floor():
    spot = 100.0
    chain = PS.chain(DATE, spot, VOL, "C")
    # floor above any strike+premium in the band -> NO_CALL
    gated, n = select_contract(StockAction.CALL_BALANCED, chain, spot, VOL,
                               ValuationState.FAIR,
                               valuation=_val(150, 0.15))   # floor 142.5
    assert gated is None
    ok, _ = select_contract(StockAction.CALL_BALANCED, chain, spot, VOL,
                            ValuationState.FAIR,
                            valuation=_val(90, 0.15))       # floor 85.5
    assert ok is not None and ok.strike + ok.mid >= 85.5 - 1e-9


# ----------------------------------------------------------------------
# VAC-6: daily-brief integration
# ----------------------------------------------------------------------

def _fake_frame():
    idx = pd.DatetimeIndex([DATE])
    return pd.DataFrame({"close": [100.0], "market_regime": [0],
                         "valuation_state": [1], "vol_compensation": [1],
                         "vol_proxy": [VOL]}, index=idx)


def test_recommend_opening_valuation_gate_blocks_and_annotates():
    from rlbot.assistant.daily import recommend_opening
    # VERY_EXPENSIVE (spot 100 vs FV 70) -> WAIT with a valuation reason
    rec = recommend_opening("T", _fake_frame(), PS, 100_000.0,
                            valuation=_val(70, 0.15), val_cfg=VCFG)
    assert rec["action"] == "WAIT" and "valuation gate" in rec["reason"]
    assert rec["valuation"]["regime"] == "VERY_EXPENSIVE"
    # FAIR-ish valuation still sells, and reports the required premium
    rec2 = recommend_opening("T", _fake_frame(), PS, 100_000.0,
                             valuation=_val(105, 0.05), val_cfg=VCFG)
    if rec2["action"] == "SELL_PUT":
        c, v = rec2["contract"], rec2["valuation"]
        assert v["premium_required"] == pytest.approx(
            max(0.0, c["strike"] - v["put_ceiling"]), abs=0.01)
    # no valuation -> classic behavior, no valuation key
    rec3 = recommend_opening("T", _fake_frame(), PS, 100_000.0)
    assert "valuation" not in rec3


def test_render_brief_includes_gate_columns():
    from rlbot.assistant.daily import render_brief, recommend_opening
    rec = recommend_opening("T", _fake_frame(), PS, 100_000.0,
                            valuation=_val(70, 0.15), val_cfg=VCFG)
    text = render_brief("2026-08-27", [rec], [], [])
    assert "Wheel FV" in text and "Prem req" in text
    assert "VERY_EXPENSIVE" in text
    assert "Valuation gates" in text          # legend section


def test_guide_position_boundary_flags():
    from rlbot.assistant.daily import guide_position
    pos = {"ticker": "T", "type": "CSP", "strike": 99.0,
           "expiration": str((DATE + pd.Timedelta(days=20)).date()),
           "premium_fill": 1.0}
    g = guide_position(pos, _fake_frame(), PS,
                       valuation=_val(90, 0.15), val_cfg=VCFG)
    assert any("acquisition ceiling" in f for f in g["attention_flags"])
    cc = {"ticker": "T", "type": "CC", "strike": 101.0,
          "expiration": str((DATE + pd.Timedelta(days=20)).date()),
          "premium_fill": 1.0}
    g2 = guide_position(cc, _fake_frame(), PS,
                        valuation=_val(120, 0.15), val_cfg=VCFG)
    assert any("exit floor" in f for f in g2["attention_flags"])
