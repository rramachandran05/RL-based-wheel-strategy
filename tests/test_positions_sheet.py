"""SPEC-008 §1b tests: monitor-sheet parsing, future-expiry filter, CSV
round-trip through the assistant's loader. Offline fixtures only."""
import pandas as pd
import pytest

from rlbot.assistant.daily import load_positions
from rlbot.data.positions_sheet import parse_active_positions, write_positions_csv

TODAY = pd.Timestamp("2026-08-22")

HEADER = ["Stock", "Date", "EarnDate", "Exp Date", "NDays", "#ofC", "P/C",
          "Strike", "Entry Premium", "PoW", "Collatoral", "AnRet", "Profit",
          "MOS", "200SMA", "Current Premium", "Action", "Notes"]


def _row(stock, exp, pc, strike, prem="$2.10", nofc="2"):
    r = [""] * len(HEADER)
    r[0], r[3], r[5], r[6], r[7], r[8] = stock, exp, nofc, pc, strike, prem
    return r


def _rows(*data_rows):
    section = ["SET UP"] + [""] * (len(HEADER) - 1)
    return [section, HEADER, *data_rows]


def test_future_expiry_kept_past_and_today_dropped():
    rows = _rows(
        _row("MSFT", "9/18/2026", "CSP", "$440"),     # future -> kept
        _row("AAPL", "8/21/2026", "CSP", "$270"),     # past -> dropped
        _row("NVDA", "8/22/2026", "CC", "$200"),      # today -> dropped (strict)
    )
    positions, warns = parse_active_positions(rows, TODAY)
    assert [p["ticker"] for p in positions] == ["MSFT"]
    p = positions[0]
    assert p == {"ticker": "MSFT", "type": "CSP", "strike": 440.0,
                 "expiration": "2026-09-18", "premium_fill": 2.10, "contracts": 2}
    assert not warns


def test_include_today_flag():
    rows = _rows(_row("NVDA", "8/22/2026", "CC", "$200"))
    positions, _ = parse_active_positions(rows, TODAY, include_today=True)
    assert len(positions) == 1


def test_junk_rows_and_bad_pc_skipped():
    rows = _rows(
        _row("", "9/18/2026", "CSP", "$100"),          # no ticker
        _row("Ticker", "", "", ""),                     # stray sub-header
        _row("TSLA", "9/18/2026", "PUT", "$250"),       # invalid P/C
        _row("MSFT", "not-a-date", "CSP", "$440"),      # bad date
        _row("brk.b ", "9/18/2026", "CSP", "$450"),     # normalized ticker
    )
    positions, _ = parse_active_positions(rows, TODAY)
    assert [p["ticker"] for p in positions] == ["BRK-B"]


def test_missing_premium_warns_and_defaults_zero():
    rows = _rows(_row("MSFT", "9/18/2026", "CSP", "$440", prem=""))
    positions, warns = parse_active_positions(rows, TODAY)
    assert positions[0]["premium_fill"] == 0.0
    assert warns and "Entry Premium" in warns[0]


def test_csv_round_trip_through_assistant_loader(tmp_path):
    rows = _rows(_row("MSFT", "9/18/2026", "CSP", "$440"))
    positions, _ = parse_active_positions(rows, TODAY)
    out = tmp_path / "positions.csv"
    write_positions_csv(positions, out)
    loaded, warn = load_positions(out)
    assert warn is None
    assert loaded[0]["ticker"] == "MSFT" and loaded[0]["strike"] == 440.0

    write_positions_csv([], out)                    # empty sheet -> empty file
    loaded, warn = load_positions(out)
    assert warn is None and loaded == []
