"""SPEC-007 Track B tests: reportedDate causality (REQ-7.3), truncation
equivalence, mapping boundaries, frame integration."""
import numpy as np
import pandas as pd
import pytest

from rlbot.data.eps_proxy import build_proxy_for_ticker, eps_ttm_series
from tests.conftest import make_ohlcv


@pytest.fixture
def eps_fixture(tmp_path):
    """8 quarters; note Q8 reported 45 days after fiscal end."""
    rows = []
    for i in range(8):
        fiscal = pd.Timestamp("2020-03-31") + pd.DateOffset(months=3 * i)
        rows.append({"fiscalDateEnding": fiscal.date(),
                     "reportedDate": (fiscal + pd.Timedelta(days=45)).date(),
                     "reportedEPS": 1.0 + 0.1 * i})
    df = pd.DataFrame(rows)
    d = tmp_path / "eps"
    d.mkdir()
    df.to_csv(d / "TST.csv", index=False)
    return d


def test_eps_ttm_uses_reported_date_not_fiscal_end(eps_fixture):
    from rlbot.data.eps_proxy import load_eps
    eps = load_eps("TST", eps_fixture)
    # Q4 fiscal ends 2020-12-31, reported 2021-02-14. On 2021-02-01 only
    # 3 quarters are reported -> ttm must still be NaN.
    dates = pd.DatetimeIndex(["2021-02-01", "2021-02-15", "2021-08-20"])
    ttm = eps_ttm_series(eps, dates)
    assert pd.isna(ttm.iloc[0])
    assert ttm.iloc[1] == pytest.approx(1.0 + 1.1 + 1.2 + 1.3)   # Q1-Q4
    assert ttm.iloc[2] == pytest.approx(1.2 + 1.3 + 1.4 + 1.5)   # Q3-Q6


def test_proxy_truncation_equivalence(eps_fixture):
    close = make_ohlcv(n=700, seed=3, start="2020-01-01")["Close"]
    full = build_proxy_for_ticker("TST", close, eps_fixture)
    as_of = close.index[500]
    trunc = build_proxy_for_ticker("TST", close.loc[:as_of], eps_fixture)
    pd.testing.assert_frame_equal(full.loc[:as_of], trunc, rtol=1e-12)


def test_negative_ttm_gives_nan_pe(eps_fixture, tmp_path):
    df = pd.read_csv(eps_fixture / "TST.csv")
    df["reportedEPS"] = -1.0
    df.to_csv(eps_fixture / "NEG.csv", index=False)
    close = make_ohlcv(n=400, seed=3, start="2020-01-01")["Close"]
    out = build_proxy_for_ticker("NEG", close, eps_fixture)
    assert out["pe"].isna().all()
    assert out["valuation_state_proxy"].isna().all()


def test_frame_integration_proxy_fills_only_gaps():
    from rlbot.config import RlbotConfig
    from rlbot.state.encoder import build_ticker_frame

    ohlcv = make_ohlcv(n=300, seed=3)
    idx = ohlcv.index
    underlying = ohlcv.rename(columns=str.lower).assign(
        ticker="TST", realized_vol_30=0.3)
    underlying.index.name = "date"
    market = pd.DataFrame({"market_regime": 0, "vol_compensation": 1,
                           "vix_close": 18.0, "vix_pct_5y": 0.5, "vrp": 2.0,
                           "spy_close": 400.0, "spy_drawdown": -0.02}, index=idx)
    valuation = pd.DataFrame(
        {"fv_buy": [ohlcv["Close"].iloc[250] * 2.0]},   # ATTRACTIVE via sheet
        index=pd.MultiIndex.from_tuples([(idx[250], "TST")], names=["date", "ticker"]),
    )
    proxy = pd.DataFrame({"ticker": "TST", "pe": 30.0, "pe_pct": 0.9,
                          "valuation_state_proxy": pd.array([2] * len(idx), dtype="Int8")},
                         index=idx)
    cfg = RlbotConfig()
    frame = build_ticker_frame("TST", underlying, market, valuation, cfg,
                               valuation_proxy=proxy)
    # day 250 onwards (ffill window): sheet FV says ATTRACTIVE (0), proxy says 2
    assert frame["valuation_state"].iloc[250] == 0        # sheet wins
    assert frame["valuation_state"].iloc[100] == 2        # proxy fills the gap
