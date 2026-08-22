"""HistoricalChainPremiumSource tests — fixture parquet, no network."""
import numpy as np
import pandas as pd
import pytest

from rlbot.options.historical_source import HistoricalChainPremiumSource
from rlbot.options.selector import SelectorConfig, select_contract
from rlbot.state.enums import CashAction, ValuationState

DATE = pd.Timestamp("2024-03-11")
EXP = pd.Timestamp("2024-04-12")   # 32 dte


def _row(cp, strike, mark, delta, iv=0.30, bid=None, ask=None, oi=500, vol=25,
         date=DATE, exp=EXP):
    bid = mark * 0.97 if bid is None else bid
    ask = mark * 1.03 if ask is None else ask
    return {"snapshot_date": date, "expiration": exp, "dte": (exp - date).days,
            "cp": cp, "strike": strike, "bid": bid, "ask": ask, "mark": mark,
            "iv": iv, "delta": delta, "gamma": 0.01, "theta": -0.02,
            "vega": 0.1, "volume": vol, "open_interest": oi}


@pytest.fixture
def source(tmp_path):
    rows = [
        _row("P", 90.0, 0.80, -0.12),
        _row("P", 95.0, 1.60, -0.22),
        _row("P", 100.0, 3.20, -0.42),
        _row("P", 85.0, 0.40, -0.07, oi=0),                    # illiquid: no OI
        _row("P", 92.5, 1.10, -0.16, bid=0.5, ask=1.7),        # wide spread ~109%
        _row("P", 88.0, 0.0, -0.10, bid=0.0, ask=0.0),         # unquotable
        _row("C", 105.0, 1.40, 0.20),
        # a later snapshot for repricing the 95P
        _row("P", 95.0, 0.55, -0.09, date=pd.Timestamp("2024-03-25")),
    ]
    d = tmp_path / "chains" / "TST"
    d.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(d / "2024.parquet")
    return HistoricalChainPremiumSource("TST", tmp_path / "chains")


def test_chain_returns_real_quotes_with_liquidity_fields(source):
    quotes = source.chain(DATE, spot=100.0, vol_proxy=0.3, cp="P")
    strikes = {q.strike for q in quotes}
    assert 95.0 in strikes and 90.0 in strikes
    assert 88.0 not in strikes                     # unquotable dropped
    q95 = next(q for q in quotes if q.strike == 95.0)
    assert q95.oi == 500 and q95.delta == -0.22 and q95.mid == 1.60
    assert q95.spread_pct == pytest.approx(0.06, abs=0.01)


def test_selector_applies_real_liquidity_filters(source):
    quotes = source.chain(DATE, spot=100.0, vol_proxy=0.3, cp="P")
    q, n = select_contract(CashAction.PUT_CONSERVATIVE, quotes, 100.0, 0.3,
                           ValuationState.FAIR)
    # band 0.10-0.18 contains 90P (delta -.12, liquid) and 92.5P (wide spread)
    # and 85P after widening (no OI). Only the 90P must survive.
    assert q is not None and q.strike == 90.0


def test_reprice_exact_then_fallback(source):
    # exact row on 2024-03-25
    mid = source.reprice("P", 95.0, EXP, pd.Timestamp("2024-03-25"), 103.0, 0.3)
    assert mid == pytest.approx(0.55)
    assert source.fallback_count == 0
    # missing date -> BS fallback, counted
    mid2 = source.reprice("P", 95.0, EXP, pd.Timestamp("2024-03-18"), 97.0, 0.3)
    assert mid2 > 0 and source.fallback_count == 1


def test_delta_now_from_row(source):
    d = source.delta_now("P", 95.0, EXP, pd.Timestamp("2024-03-25"), 103.0, 0.3)
    assert d == pytest.approx(-0.09)


def test_missing_day_gives_empty_chain(source):
    assert source.chain(pd.Timestamp("2024-07-01"), 100.0, 0.3, "P") == []


def test_missing_ticker_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        HistoricalChainPremiumSource("NOPE", tmp_path / "chains")


def test_env_runs_end_to_end_on_historical_source(tmp_path):
    """Full episode: open from a real-chain fixture, MTM via fallback, settle."""
    from rlbot.benchmarks.policies import FixedWheelPolicy
    from rlbot.simulator.environment import WheelEnv
    from tests.test_environment import make_frame

    frame = make_frame(n=80, seed=5)
    frame["close"] = 100.0                        # flat path, spot pinned
    d = tmp_path / "chains" / "TEST"
    d.mkdir(parents=True)
    dates = frame.index
    exp = dates[40]
    rows = [_row("P", 95.0, 1.60, -0.22, date=dt, exp=exp)
            for dt in dates[:35]] + [
           _row("C", 105.0, 1.40, 0.20, date=dt, exp=dates[70])
            for dt in dates[35:65]]
    for year, chunk in pd.DataFrame(rows).groupby(pd.DataFrame(rows)["snapshot_date"].dt.year):
        chunk.to_parquet(d / f"{year}.parquet")
    src = HistoricalChainPremiumSource("TEST", tmp_path / "chains")
    env = WheelEnv("TEST", frame, src)
    res = env.run(FixedWheelPolicy(), dates[0], dates[-1])
    opens = [dd for dd in res.decisions if dd.contract is not None]
    assert opens
    assert opens[0].contract["premium_source"] == "historical_chain"
    assert res.cycles
