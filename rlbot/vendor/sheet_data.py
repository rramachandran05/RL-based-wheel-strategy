# VENDORED from ../wheel-strategy/sheet_data.py on 2026-08-22 (source sha256: 0ad3612fbe9a740094dd586a6904d55556cfea481b925f57f4360e446afff054)
# Do not edit — see SPEC-002 REQ-2.2. Changes belong in rlbot/, not here.
"""
sheet_data.py
=============
Generic helpers for reading the Google-Sheets-as-CSV inputs used elsewhere in
the toolkit (fair-value inputs, position monitoring). Network access and
string parsing live here so the modules that consume sheet data stay
offline-testable.

A published sheet tab is fetched as CSV via the `export?format=csv&gid=...`
URL. Two header-row quirks show up in both tabs used by this project:
  * row 0 is a stray section label, the real header is row 1.
  * trailing summary/notes rows have no usable ticker.
Both are handled by the caller (`rows_to_dicts` + a ticker-validity filter),
not here -- this module only does generic CSV fetch/parse.

Dependencies: requests.
"""

from __future__ import annotations

import csv
import io
import re
from typing import Dict, List, Optional

import requests

SHEET_EXPORT_URL = (
    "https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
)


def sheet_csv_url(sheet_id: str, gid: str) -> str:
    return SHEET_EXPORT_URL.format(sheet_id=sheet_id, gid=gid)


def fetch_sheet_rows(sheet_id: str, gid: str, timeout: int = 15) -> List[List[str]]:
    """
    Download a published Google Sheet tab as CSV and parse it into rows of
    raw strings. Returns [] on any failure -- network error, or a sign-in
    page returned instead of CSV (sharing set to private) -- so a dead sheet
    contributes nothing rather than crashing the pipeline.
    """
    try:
        resp = requests.get(sheet_csv_url(sheet_id, gid), timeout=timeout)
        resp.raise_for_status()
        text = resp.text
    except Exception:
        return []
    if "<html" in text[:200].lower():
        return []
    return list(csv.reader(io.StringIO(text)))


def rows_to_dicts(rows: List[List[str]], header_row: int) -> List[Dict[str, str]]:
    """
    Map every row after `header_row` (0-indexed) to a dict keyed by the
    header at `header_row`. Short rows are padded with "" so every dict has
    the full set of header keys.
    """
    if header_row >= len(rows):
        return []
    header = rows[header_row]
    out: List[Dict[str, str]] = []
    for raw in rows[header_row + 1:]:
        padded = raw + [""] * (len(header) - len(raw))
        out.append(dict(zip(header, padded)))
    return out


_MONEY_RE = re.compile(r"[^0-9.\-]")


def parse_dollar(s: str) -> Optional[float]:
    """'$1,003' -> 1003.0; '"$96,000.00"' -> 96000.0; '' / '-' -> None."""
    if not s:
        return None
    cleaned = _MONEY_RE.sub("", s.strip())
    if cleaned in ("", "-", "."):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_percent(s: str) -> Optional[float]:
    """'38.19%' -> 0.3819; '-36.39%' -> -0.3639; '' -> None."""
    if not s:
        return None
    cleaned = s.strip().rstrip("%")
    try:
        return float(cleaned) / 100.0
    except ValueError:
        return None


def normalize_ticker(s: str) -> str:
    """'brk.b ' -> 'BRK-B'. Sheet uses dot-class notation, config/Tiingo use dashes."""
    return s.strip().upper().replace(".", "-")


# ----------------------------------------------------------------------
# Tests (offline -- no network)
# ----------------------------------------------------------------------

def _run_tests() -> None:
    # --- dollar parsing ---
    assert parse_dollar("$291.58") == 291.58
    assert parse_dollar("$1,003") == 1003.0
    assert parse_dollar('$96,000.00') == 96000.0
    assert parse_dollar("-$5.00") == -5.0
    assert parse_dollar("") is None
    assert parse_dollar("-") is None

    # --- percent parsing ---
    assert abs(parse_percent("38.19%") - 0.3819) < 1e-9
    assert abs(parse_percent("-36.39%") - (-0.3639)) < 1e-9
    assert abs(parse_percent("92%") - 0.92) < 1e-9
    assert parse_percent("") is None

    # --- ticker normalization ---
    assert normalize_ticker(" brk.b ") == "BRK-B"
    assert normalize_ticker("aapl") == "AAPL"

    # --- rows_to_dicts: header offset + short-row padding ---
    rows = [
        ["stray label", "", ""],
        ["Stock", "low", "high"],
        ["AAPL", "100", "200"],
        ["AMZN", "50"],
    ]
    dicts = rows_to_dicts(rows, header_row=1)
    assert dicts[0] == {"Stock": "AAPL", "low": "100", "high": "200"}
    assert dicts[1] == {"Stock": "AMZN", "low": "50", "high": ""}

    # header_row past the end of the sheet -> empty, no crash
    assert rows_to_dicts(rows, header_row=10) == []

    print("sheet_data: all tests passed")


if __name__ == "__main__":
    _run_tests()
