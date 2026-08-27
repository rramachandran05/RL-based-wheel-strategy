"""SPEC-010 Feed B tests: momentum-monitor candidate ingestion (fixture DB)."""
import sqlite3

import pytest

from rlbot.data.candidates import Candidate, cap_candidate_action, latest_candidates
from rlbot.state.enums import CashAction, StockAction

SCHEMA = """CREATE TABLE monitor_rankings (
    run_date TEXT NOT NULL, symbol TEXT NOT NULL,
    r20 REAL, r60 REAL, r120 REAL, percentile REAL, in_top_decile INTEGER,
    PRIMARY KEY (run_date, symbol));"""


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "vmi.sqlite"
    conn = sqlite3.connect(path)
    conn.execute(SCHEMA)
    rows = [
        # ~5 weeks ago
        ("2026-07-25", "PLTR", 0.4, 0.6, 0.90, 0.955, 1),
        ("2026-07-25", "APP", 0.3, 0.5, 0.85, 0.940, 1),
        ("2026-07-25", "XYZ", 0.1, 0.2, 0.30, 0.500, 0),
        # latest run
        ("2026-08-22", "PLTR", 0.5, 0.7, 0.95, 0.990, 1),
        ("2026-08-22", "APP", 0.2, 0.4, 0.80, 0.960, 1),
        ("2026-08-22", "NVDA", 0.4, 0.6, 0.88, 0.970, 1),   # in core -> excluded
        ("2026-08-22", "MEH", 0.0, 0.1, 0.20, 0.400, 0),    # not top decile
    ]
    conn.executemany("INSERT INTO monitor_rankings VALUES (?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return path


def test_latest_candidates_top_decile_ranked_and_excluded(db):
    cands, warn = latest_candidates(exclude={"NVDA"}, db_path=db)
    assert warn is None
    assert [c.ticker for c in cands] == ["PLTR", "APP"]      # by percentile desc
    assert all(c.run_date == "2026-08-22" for c in cands)


def test_rank_change_4w_from_history(db):
    cands, _ = latest_candidates(exclude=set(), db_path=db)
    pltr = next(c for c in cands if c.ticker == "PLTR")
    assert pltr.rank_change_4w == pytest.approx(0.990 - 0.955)
    nvda = next(c for c in cands if c.ticker == "NVDA")      # no prior row? has one? no
    assert nvda.rank_change_4w is None


def test_top_n_limit(db):
    cands, _ = latest_candidates(exclude=set(), top_n=1, db_path=db)
    assert len(cands) == 1 and cands[0].ticker == "PLTR"


def test_missing_db_degrades_gracefully(tmp_path):
    cands, warn = latest_candidates(db_path=tmp_path / "nope.sqlite")
    assert cands == [] and "not found" in warn


def test_cap_candidate_action():
    assert cap_candidate_action(CashAction.PUT_AGGRESSIVE) == CashAction.PUT_CONSERVATIVE
    assert cap_candidate_action(CashAction.PUT_BALANCED) == CashAction.PUT_CONSERVATIVE
    assert cap_candidate_action(CashAction.PUT_DEFENSIVE) == CashAction.PUT_DEFENSIVE
    assert cap_candidate_action(CashAction.WAIT) == CashAction.WAIT
    assert cap_candidate_action(StockAction.CALL_AGGRESSIVE) == StockAction.CALL_AGGRESSIVE
