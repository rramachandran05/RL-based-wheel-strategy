"""Sync active positions from the Google Sheet monitor tab (SPEC-008 §1b).

Reads the link-shared monitor tab (same sheet the sibling project's
position_monitor.py uses; header on row index 1, Exp Date in column D as
M/D/YYYY). A row is an active position when it has a valid ticker,
P/C in {CSP, CC}, a parseable strike, and an Exp Date strictly in the
future relative to `today` (per the 2026-08-22 request; positions expiring
today are excluded — flip `include_today` if that changes).

Run:  python -m rlbot.data.positions_sheet     # sync data_local/positions.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from rlbot.config import RlbotConfig
from rlbot.vendor.sheet_data import (
    fetch_sheet_rows,
    normalize_ticker,
    parse_dollar,
    rows_to_dicts,
)

MONITOR_SHEET_ID = "1IW4cNkUsTLgylGe5VkkxjOf-XBHtKWmTQ5xzCIbd_9Y"
MONITOR_SHEET_GID = "1242006061"
MONITOR_HEADER_ROW = 1          # row 0 is a stray section label
VALID_PC = {"CSP", "CC"}


def parse_active_positions(raw_rows: list, today: pd.Timestamp,
                           include_today: bool = False) -> tuple:
    """(positions, warnings) from raw sheet rows. Junk rows are skipped
    silently; parseable-but-expired rows are skipped silently too (that is
    the normal case for a long-lived sheet)."""
    positions, warnings = [], []
    today = pd.Timestamp(today).normalize()
    for row in rows_to_dicts(raw_rows, MONITOR_HEADER_ROW):
        ticker = normalize_ticker(row.get("Stock", ""))
        pc = row.get("P/C", "").strip().upper()
        strike = parse_dollar(row.get("Strike", ""))
        exp = pd.to_datetime(row.get("Exp Date", ""), format="%m/%d/%Y", errors="coerce")
        if not ticker or pc not in VALID_PC or strike is None or pd.isna(exp):
            continue
        exp = exp.normalize()
        if exp < today or (exp == today and not include_today):
            continue
        premium = parse_dollar(row.get("Entry Premium", ""))
        if premium is None:
            premium = 0.0
            warnings.append(f"{ticker} {pc} {strike}: no Entry Premium on sheet; "
                            "premium-captured will read 0%")
        contracts_raw = parse_dollar(row.get("#ofC", ""))
        contracts = int(contracts_raw) if contracts_raw else 1
        positions.append({
            "ticker": ticker, "type": pc, "strike": float(strike),
            "expiration": exp.strftime("%Y-%m-%d"),
            "premium_fill": float(premium), "contracts": contracts,
        })
    return positions, warnings


def fetch_active_positions(today: pd.Timestamp,
                           sheet_id: str = MONITOR_SHEET_ID,
                           gid: str = MONITOR_SHEET_GID) -> tuple:
    """(positions, warnings). Empty positions + warning when unreachable —
    callers fall back to the existing positions.csv (REQ-8.3 spirit)."""
    raw = fetch_sheet_rows(sheet_id, gid)
    if not raw:
        return [], ["monitor sheet unreachable; using existing positions.csv"]
    return parse_active_positions(raw, today)


def write_positions_csv(positions: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ("# Synced from the Google Sheet monitor tab by "
              "rlbot.data.positions_sheet — do not hand-edit; re-sync instead.\n")
    df = pd.DataFrame(positions,
                      columns=["ticker", "type", "strike", "expiration",
                               "premium_fill", "contracts"])
    path.write_text(header + df.to_csv(index=False))


if __name__ == "__main__":
    cfg = RlbotConfig()
    today = pd.Timestamp.now().normalize()
    positions, warns = fetch_active_positions(today)
    for w in warns:
        print(f"warning: {w}")
    out = cfg.data.base_path / "positions.csv"
    write_positions_csv(positions, out)
    print(f"wrote {out}: {len(positions)} active positions")
    for p in positions:
        print(f"  {p['ticker']:6s} {p['type']} {p['strike']} exp {p['expiration']} "
              f"x{p['contracts']} prem {p['premium_fill']}")
