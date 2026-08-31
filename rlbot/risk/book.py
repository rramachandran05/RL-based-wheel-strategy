"""Book-level risk inputs for the daily assistant (2026-08-30 review fix).

Aggregates the synced sheet positions into the numbers `validate_open`
needs to enforce RISK-4 (max positions), RISK-5 (expiry-week clustering)
and RISK-8 (aggregate assignment-at-once) across the WHOLE book — the
per-ticker loop previously evaluated every name in isolation.

Also estimates the next earnings date from the AV `reportedDate` history
(last report + ~91 days) so RISK-7 (earnings blackout) finally has a live
signal. Estimated dates carry a ±5-day tolerance; tickers without EPS
history get no estimate — constraint absent, per the house philosophy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from rlbot.config import RlbotConfig

QUARTER_DAYS = 91
EARNINGS_TOLERANCE_DAYS = 5


@dataclass
class BookState:
    n_open_positions: int = 0
    put_escrow: float = 0.0                    # Σ strike·100·contracts (CSPs)
    expiry_week_counts: dict = field(default_factory=dict)   # iso (yr, wk) -> n
    # Normalized views (2026-08-31): counts mislead when position sizes vary
    # 10x (one TQQQ put != one META put), so the risk rules read dollars.
    underlyings: set = field(default_factory=set)             # distinct tickers
    expiry_week_escrow: dict = field(default_factory=dict)    # (yr, wk) -> $ CSP escrow
    ticker_escrow: dict = field(default_factory=dict)         # ticker -> $ CSP escrow

    @property
    def n_underlyings(self) -> int:
        return len(self.underlyings)

    def has(self, ticker: str) -> bool:
        return str(ticker).upper() in self.underlyings

    def escrow_for(self, ticker: str) -> float:
        return self.ticker_escrow.get(str(ticker).upper(), 0.0)

    def same_week_count(self, expiration) -> int:
        iso = pd.Timestamp(expiration).isocalendar()
        return self.expiry_week_counts.get((iso.year, iso.week), 0)

    def same_week_escrow(self, expiration) -> float:
        iso = pd.Timestamp(expiration).isocalendar()
        return self.expiry_week_escrow.get((iso.year, iso.week), 0.0)


def build_book(positions: list) -> BookState:
    """positions: dicts from positions.csv (ticker/type/strike/expiration/
    contracts). Malformed rows are skipped — never break the brief."""
    book = BookState()
    for p in positions:
        try:
            strike = float(p["strike"])
            contracts = int(p.get("contracts", 1) or 1)
            exp = pd.Timestamp(p["expiration"])
        except Exception:
            continue
        book.n_open_positions += 1
        ticker = str(p.get("ticker", "")).upper()
        book.underlyings.add(ticker)
        iso = exp.isocalendar()
        key = (iso.year, iso.week)
        book.expiry_week_counts[key] = book.expiry_week_counts.get(key, 0) + 1
        if str(p.get("type", "")).upper() == "CSP":
            escrow = strike * 100 * contracts
            book.put_escrow += escrow
            book.ticker_escrow[ticker] = book.escrow_for(ticker) + escrow
            book.expiry_week_escrow[key] = \
                book.expiry_week_escrow.get(key, 0.0) + escrow
    return book


def next_earnings_estimate(ticker: str, cfg: RlbotConfig,
                           today=None) -> pd.Timestamp | None:
    """Last AV reportedDate + ~91d, rolled forward past today. None when no
    EPS history exists (constraint absent)."""
    path = cfg.data.external_path / "eps" / f"{ticker}.csv"
    if not path.exists():
        return None
    try:
        eps = pd.read_csv(path, parse_dates=["reportedDate"])
    except Exception:
        return None
    if eps.empty:
        return None
    today = pd.Timestamp(today or pd.Timestamp.now()).normalize()
    est = eps["reportedDate"].max().normalize()
    while est <= today:
        est += pd.Timedelta(days=QUARTER_DAYS)
    return est


def earnings_in_window(ticker: str, expiration, cfg: RlbotConfig,
                       today=None) -> bool:
    """True when the estimated next earnings (±tolerance) falls inside
    [today, expiration] — the RISK-7 blackout condition."""
    est = next_earnings_estimate(ticker, cfg, today)
    if est is None:
        return False
    today = pd.Timestamp(today or pd.Timestamp.now()).normalize()
    lo = est - pd.Timedelta(days=EARNINGS_TOLERANCE_DAYS)
    hi = est + pd.Timedelta(days=EARNINGS_TOLERANCE_DAYS)
    return not (hi < today or lo > pd.Timestamp(expiration))
