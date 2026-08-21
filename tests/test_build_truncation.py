"""REQ-2.1 / SPEC-002 AC-2: truncation equivalence — building with data
truncated at as_of equals the rows <= as_of of the full build. Proves every
builder is causal (no look-ahead)."""
import pandas as pd
import pytest

from rlbot.config import RlbotConfig
from rlbot.data.build import build_market, build_underlying, build_valuation
from rlbot.features.technicals_series import build_feature_frame


@pytest.fixture
def cfg(tmp_path):
    c = RlbotConfig()
    c.data.base_path = tmp_path
    # small percentile window so the synthetic fixture produces non-NA rows
    c.data.vix_percentile_window = 120
    c.data.vix_percentile_min_obs = 60
    return c


def _assert_frames_equal(full_head: pd.DataFrame, truncated: pd.DataFrame):
    pd.testing.assert_frame_equal(full_head, truncated, check_exact=False, rtol=1e-12)


def test_market_truncation_equivalence(ohlcv, vix_series, cfg):
    as_of = ohlcv.index[350]
    full = build_market(ohlcv, vix_series, cfg)
    truncated = build_market(ohlcv.loc[:as_of], vix_series.loc[:as_of], cfg)
    _assert_frames_equal(full.loc[:as_of], truncated)


def test_feature_frame_truncation_equivalence(ohlcv):
    as_of = ohlcv.index[400]
    full = build_feature_frame(ohlcv)
    truncated = build_feature_frame(ohlcv.loc[:as_of])
    _assert_frames_equal(full.loc[:as_of], truncated)


def test_underlying_builder_stacks_tickers(ohlcv, cfg):
    from tests.conftest import make_ohlcv

    bars = {"AAA": ohlcv, "BBB": make_ohlcv(seed=99)}
    table = build_underlying(bars, cfg)
    assert set(table["ticker"].unique()) == {"AAA", "BBB"}
    assert table.index.name == "date"
    assert len(table) == len(ohlcv) * 2


def test_valuation_builder_on_fixture_snapshots(tmp_path, cfg):
    snap = tmp_path / "snaps"
    snap.mkdir()
    (snap / "fair_value_2026-08-01.csv").write_text(
        "Ticker,Price,FV_Buy,FV_Sell,FMP_Median,Source,Confidence\n"
        "AAPL,100.0,120.0,140.0,125.0,sheet+fmp,high\n"   # dist -16.7% -> ATTRACTIVE
        "MSFT,110.0,100.0,130.0,,sheet,medium\n"           # dist +10% -> EXPENSIVE
    )
    table = build_valuation(snap, cfg)
    assert len(table) == 2
    aapl = table.loc[(pd.Timestamp("2026-08-01"), "AAPL")]
    msft = table.loc[(pd.Timestamp("2026-08-01"), "MSFT")]
    assert aapl["valuation_state"] == 0   # ATTRACTIVE
    assert msft["valuation_state"] == 2   # EXPENSIVE


def test_valuation_builder_empty_dir_gives_empty_table(tmp_path, cfg):
    table = build_valuation(tmp_path / "nonexistent", cfg)
    assert table.empty
