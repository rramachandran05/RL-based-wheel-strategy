"""SPEC-007 §3C (Track C, gate G9): drawdown-percentile valuation proxy and
SPEC-011 §2 rule 6: MCB-position + drawdown-severity score-side valuation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rlbot.data.mcb_feed import McbRow
from rlbot.features.valuation import classify_drawdown_series
from rlbot.risk.mcb_gates import mcb_valuation_state
from rlbot.state.enums import MarketRegime, ValuationState


def _dd(closes):
    s = pd.Series(closes, dtype=float,
                  index=pd.bdate_range("2015-01-01", periods=len(closes)))
    return s / s.cummax() - 1.0


def test_classify_drawdown_direction_and_warmup():
    # 300 days: first 250 flat-ish grind up (dd≈0), then a 40% crash.
    rng = np.random.default_rng(0)
    up = 100 * np.cumprod(1 + rng.normal(0.0005, 0.002, 260))
    crash = up[-1] * np.linspace(1.0, 0.6, 40)
    dd = _dd(np.concatenate([up, crash]))
    st = classify_drawdown_series(dd, window=1260, min_obs=252)
    assert st.iloc[:251].isna().all()                # warmup -> NA (undefined)
    assert st.iloc[-1] == int(ValuationState.ATTRACTIVE)   # deepest drawdown ever
    # at a fresh all-time high the drawdown is 0 = its own max -> EXPENSIVE
    at_high = st.iloc[255]
    assert at_high == int(ValuationState.EXPENSIVE)


def test_classify_drawdown_is_causal_and_per_ticker_scaled():
    # a low-vol name: 1% pullback is its worst fifth; a high-vol name needs
    # far more — the percentile adapts to the ticker's own distribution
    n = 400
    calm = 100 + np.cumsum(np.full(n, 0.05)); calm[-1] -= 1.0
    wild = 100 * np.cumprod(1 + np.random.default_rng(1).normal(0, 0.03, n))
    st_calm = classify_drawdown_series(_dd(calm), min_obs=252)
    st_wild = classify_drawdown_series(_dd(wild), min_obs=252)
    assert st_calm.iloc[-1] == int(ValuationState.ATTRACTIVE)   # tiny dip, but ITS worst
    assert st_wild.notna().sum() > 0
    # causality: truncating the future never changes past labels
    full = classify_drawdown_series(_dd(wild), min_obs=252)
    part = classify_drawdown_series(_dd(wild[:350]), min_obs=252)
    pd.testing.assert_series_equal(full.iloc[:350], part, check_names=False)


def _row(**kw):
    base = dict(ticker="T", date="2026-09-09", mcb={"FAIR": 90.0},
                min_eligible_tier="FAIR", guardrail=None, layer_a="OWN",
                reachability=None, confidence=0.8, wheel_entry=90.0,
                delta_posture="MODERATE", dd50=0.10, dd75=0.16, dd90=0.30)
    base.update(kw)
    return McbRow(**base)


BULL, STRESS = int(MarketRegime.BULL_LOW_VOL), int(MarketRegime.BEAR_STRESS)


def test_mcb_valuation_state_bands():
    assert mcb_valuation_state(_row(drop_needed=0.05, dd_now=0.01), BULL) == ValuationState.ATTRACTIVE
    assert mcb_valuation_state(_row(drop_needed=0.20, dd_now=0.01), BULL) == ValuationState.FAIR
    assert mcb_valuation_state(_row(drop_needed=0.35, dd_now=0.01), BULL) == ValuationState.EXPENSIVE
    assert mcb_valuation_state(_row(drop_needed=-0.03, dd_now=0.0), BULL) == ValuationState.ATTRACTIVE


def test_mcb_valuation_relief_notch_only_outside_stress():
    # correction already at/beyond the 75th pct -> one notch less penalizing
    r = _row(drop_needed=0.35, dd_now=0.20)
    assert mcb_valuation_state(r, BULL) == ValuationState.FAIR       # EXPENSIVE -> FAIR
    assert mcb_valuation_state(r, STRESS) == ValuationState.EXPENSIVE  # no relief in stress
    r2 = _row(drop_needed=0.20, dd_now=0.20)
    assert mcb_valuation_state(r2, BULL) == ValuationState.ATTRACTIVE  # FAIR -> ATTRACTIVE
    # already ATTRACTIVE: relief cannot go below the floor
    assert mcb_valuation_state(_row(drop_needed=0.05, dd_now=0.20), BULL) == ValuationState.ATTRACTIVE


def test_mcb_valuation_missing_inputs_are_neutral():
    assert mcb_valuation_state(None, BULL) == ValuationState.FAIR
    assert mcb_valuation_state(_row(drop_needed=None, dd_now=0.0), BULL) == ValuationState.FAIR
    assert mcb_valuation_state(_row(drop_needed=0.35, dd_now=0.0, dd90=None), BULL) == ValuationState.FAIR


def test_daily_feeds_mcb_state_to_score_not_to_policy():
    """The policy's Q-state axis must be untouched; only the score input moves."""
    from rlbot.assistant.daily import recommend_opening
    from rlbot.options.premium_source import SyntheticBSPremiumSource
    idx = pd.DatetimeIndex([pd.Timestamp("2026-08-27")])
    frame = pd.DataFrame({"close": [100.0], "market_regime": [0],
                          "valuation_state": [2],          # Q-state says EXPENSIVE
                          "vol_compensation": [1], "vol_proxy": [0.25]}, index=idx)
    row = _row(delta_posture="CONSERVATIVE", wheel_entry=60.0, mcb={"FAIR": 60.0},
               drop_needed=0.40, dd_now=0.0)            # MCB says EXPENSIVE too
    rec = recommend_opening("T", frame, SyntheticBSPremiumSource(), 100_000.0, mcb=row)
    assert rec["q_state"][1] == 2                          # policy input unchanged
    assert rec["mcb"]["score_valuation"] == "EXPENSIVE"
    row2 = _row(delta_posture="CONSERVATIVE", wheel_entry=60.0, mcb={"FAIR": 60.0},
                drop_needed=0.02, dd_now=0.0)           # MCB says ATTRACTIVE
    rec2 = recommend_opening("T", frame, SyntheticBSPremiumSource(), 100_000.0, mcb=row2)
    assert rec2["q_state"][1] == 2 and rec2["mcb"]["score_valuation"] == "ATTRACTIVE"


def test_drawdown_frame_store_overwrites_axis_only():
    from rlbot.config import RlbotConfig
    from rlbot.evaluation.ablation_drawdown import DrawdownFrameStore
    cfg = RlbotConfig(use_valuation_proxy=False)
    try:
        store = DrawdownFrameStore(cfg)
        f = store.frame("AAPL")
    except FileNotFoundError:
        pytest.skip("canonical tables not built in this environment")
    inner = store.inner.frame("AAPL")
    assert set(f["valuation_state"].dropna().unique()) <= {0, 1, 2}
    assert f["valuation_state"].notna().sum() > 1000     # zero historical gap
    pd.testing.assert_series_equal(f["market_regime"], inner["market_regime"])
