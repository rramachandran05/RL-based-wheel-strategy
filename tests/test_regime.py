"""SPEC-001 §3.1 market_regime golden tests, including precedence tie-breaks
(part of SPEC-001 AC-3)."""
import pandas as pd
import pytest

from rlbot.features.regime import classify_regime_series
from rlbot.state.enums import MarketRegime


def _classify_one(close, sma200, drawdown, vix_pct):
    idx = pd.Index([pd.Timestamp("2024-01-02")])
    result = classify_regime_series(
        pd.Series([close], index=idx),
        pd.Series([sma200], index=idx),
        pd.Series([drawdown], index=idx),
        pd.Series([vix_pct], index=idx),
    )
    return result.iloc[0]


GOLDEN = [
    # (close, sma200, drawdown, vix_pct, expected)
    (110, 100, -0.02, 0.30, MarketRegime.BULL_LOW_VOL),
    (110, 100, -0.02, 0.59, MarketRegime.BULL_LOW_VOL),   # boundary: pct < high
    (110, 100, -0.02, 0.60, MarketRegime.BULL_HIGH_VOL),  # boundary: pct >= high
    (110, 100, -0.09, 0.70, MarketRegime.BULL_HIGH_VOL),
    # above SMA200 but drawdown fails bull test -> SIDEWAYS
    (110, 100, -0.12, 0.30, MarketRegime.SIDEWAYS),
    # below SMA200 with shallow drawdown -> not bear, falls to SIDEWAYS
    (95, 100, -0.05, 0.30, MarketRegime.SIDEWAYS),
    # bear: below SMA200 AND drawdown <= -10%
    (95, 100, -0.10, 0.30, MarketRegime.BEAR_STRESS),
    # bear: deep drawdown alone, even above SMA200
    (110, 100, -0.15, 0.30, MarketRegime.BEAR_STRESS),
    # bear: VIX stress alone — precedence over an otherwise-bull row
    (110, 100, -0.02, 0.85, MarketRegime.BEAR_STRESS),
    # precedence: stress VIX + bull price structure is still BEAR_STRESS
    (150, 100, 0.0, 0.90, MarketRegime.BEAR_STRESS),
]


@pytest.mark.parametrize("close,sma200,dd,vix_pct,expected", GOLDEN)
def test_regime_golden(close, sma200, dd, vix_pct, expected):
    assert _classify_one(close, sma200, dd, vix_pct) == expected


def test_nan_inputs_give_na():
    assert pd.isna(_classify_one(float("nan"), 100, -0.02, 0.30))
    assert pd.isna(_classify_one(110, 100, -0.02, float("nan")))


def test_series_output_dtype_and_coverage(ohlcv, vix_series):
    from rlbot.config import RlbotConfig
    from rlbot.data.build import build_market

    market = build_market(ohlcv, vix_series, RlbotConfig())
    regime = market["market_regime"]
    assert str(regime.dtype) == "Int8"
    assert regime.notna().sum() > 0
    assert regime.isna().sum() > 0  # warmup rows must be NA, not guessed
