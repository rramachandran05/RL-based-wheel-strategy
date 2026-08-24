"""FV-snapshot port: golden agreement with the sibling's fv_levels semantics."""
import pandas as pd
import pytest

from rlbot.data.fv_snapshot import (
    buy_sell_levels,
    etf_fv_proxy,
    fv_anchors,
    parse_fv_rows,
)
from tests.conftest import make_ohlcv


def test_anchor_semantics_match_sibling():
    """fv_buy = min(fmp, tipranks, so); fv_sell = max(fmp, tipranks, price)."""
    assert fv_anchors(110.0, 95.0, 105.0, 120.0) == (95.0, 120.0)
    # stock_oracle excluded from sell side; price included
    assert fv_anchors(110.0, 200.0, None, 90.0) == (110.0, 110.0)
    assert fv_anchors(None, 95.0, None, 120.0) == (95.0, 120.0)
    with pytest.raises(ValueError):
        fv_anchors(None, None, None, 100.0)   # buy side empty


def test_ladders_whole_dollar():
    buys, sells = buy_sell_levels(100.0, 200.0)
    assert buys == [94, 88, 82]
    assert sells == [212, 224, 236]


def test_parse_fv_rows_fixture():
    header = ["Stock", "low", "high", "median", "TipRanks (mean)",
              "Stock Oracle (Intrinsic Value)", "Min Buy Value", "Current Price"]
    pad = lambda cells: cells + [""] * (len(header) - len(cells))
    raw = [pad(["SET UP"]), header,
           pad(["AAPL", "", "", "", "$210", "$195.50", "", "$230"]),
           pad(["brk.b", "", "", "", "", "$480", "", "$495"]),
           pad(["STOCK"]),                   # skip-token
           pad(["Ticker"]),                  # stray sub-header
           pad(["", "", "", "", "$1"])]
    parsed = parse_fv_rows(raw)
    assert parsed["AAPL"] == {"tipranks": 210.0, "stock_oracle": 195.5}
    assert parsed["BRK-B"]["stock_oracle"] == 480.0
    assert "STOCK" not in parsed


def test_etf_proxy_brackets_price():
    df = make_ohlcv(n=260, seed=9)
    lo, hi = etf_fv_proxy(df)
    price = float(df["Close"].iloc[-1])
    assert lo <= price <= hi
    assert lo > 0 and hi > 0


def test_snapshot_schema_round_trips_into_valuation(tmp_path):
    from rlbot.config import RlbotConfig
    from rlbot.data.build import build_valuation

    cfg = RlbotConfig()
    snap = tmp_path / "snaps"
    snap.mkdir()
    pd.DataFrame([{
        "Ticker": "AAPL", "Price": 230.0, "FV_Buy": 195.5, "FV_Sell": 230.0,
        "Buy_L1": 184, "Buy_L2": 172, "Buy_L3": 160,
        "Sell_L1": 244, "Sell_L2": 258, "Sell_L3": 271,
        "FMP_Median": None, "Source": "sheet", "Confidence": "medium",
    }]).to_csv(snap / "fair_value_2026-08-23.csv", index=False)
    table = build_valuation(snap, cfg)
    row = table.loc[(pd.Timestamp("2026-08-23"), "AAPL")]
    assert row["fv_buy"] == 195.5
    assert row["valuation_state"] == 2      # price 17.6% above fv_buy -> EXPENSIVE
