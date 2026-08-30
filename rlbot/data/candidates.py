"""Feed B (SPEC-009 §3): momentum-monitor candidates.

Read-only consumer of the `monitor_rankings` table in the shared VMI store
(`../vmi-stock-search/data/vmi.sqlite`, override via MONITOR_DB_PATH).
Candidates are second-class: capped at PUT_CONSERVATIVE, max open positions
limited, never auto-promoted (Rahul promotes by editing the config/universe).
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from rlbot.config import PROJECT_ROOT

DEFAULT_DB = PROJECT_ROOT.parent / "vmi-stock-search" / "data" / "vmi.sqlite"
CANDIDATE_TOP_N = 10
MAX_OPEN_CANDIDATE_POSITIONS = 2


@dataclass(frozen=True)
class Candidate:
    ticker: str
    percentile: float
    r120: float | None
    run_date: str
    rank_change_4w: float | None   # percentile-points vs ~4 weeks earlier


def _db_path() -> Path:
    override = os.environ.get("MONITOR_DB_PATH")
    return Path(override) if override else DEFAULT_DB


def latest_candidates(exclude: set | None = None, top_n: int = CANDIDATE_TOP_N,
                      db_path: Path | None = None,
                      spy_only: bool = True) -> tuple:
    """(candidates, warning|None). Graceful empty on any failure (REQ-9.5)."""
    path = db_path or _db_path()
    if not path.exists():
        return [], f"momentum monitor store not found ({path}); no candidates"
    exclude = {t.upper() for t in (exclude or set())}
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        dates = [r[0] for r in conn.execute(
            "SELECT DISTINCT run_date FROM monitor_rankings ORDER BY run_date")]
        if not dates:
            return [], "momentum monitor has no ranking runs yet"
        latest = dates[-1]
        prior = next((d for d in reversed(dates)
                      if (pd.Timestamp(latest) - pd.Timestamp(d)).days >= 26), None)
        spy = set()
        if spy_only:
            try:
                spy = {r[0] for r in conn.execute(
                    "SELECT symbol FROM monitor_spy_holdings")}
            except Exception:
                spy = set()          # table absent -> filter no-ops
        cur = pd.read_sql_query(
            "SELECT symbol, percentile, r120 FROM monitor_rankings "
            "WHERE run_date = ? AND in_top_decile = 1 "
            "ORDER BY percentile DESC", conn, params=(latest,))
        if spy:
            cur = cur[cur["symbol"].isin(spy)]
        old = {}
        if prior:
            old = dict(conn.execute(
                "SELECT symbol, percentile FROM monitor_rankings WHERE run_date = ?",
                (prior,)))
        conn.close()
    except Exception as e:
        return [], f"momentum monitor store unreadable: {e}"

    out = []
    for r in cur.itertuples():
        sym = str(r.symbol).upper()
        if sym in exclude:
            continue
        change = (float(r.percentile) - float(old[sym])) if sym in old else None
        out.append(Candidate(sym, float(r.percentile),
                             float(r.r120) if pd.notna(r.r120) else None,
                             latest, change))
        if len(out) >= top_n:
            break
    return out, None


def cap_candidate_action(action):
    """Candidates never exceed PUT_CONSERVATIVE (SPEC-009 §3, reasoned
    default, labeled not-gate-validated)."""
    from rlbot.state.enums import CashAction
    if isinstance(action, CashAction) and int(action) > int(CashAction.PUT_CONSERVATIVE):
        return CashAction.PUT_CONSERVATIVE
    return action
