# VENDORED from ../wheel-strategy/portfolio_risk.py on 2026-08-21 (source sha256: 6cf673b440b9801c4af9625c92308aae167619120f99f0fe47a26b557d785fe3)
# Do not edit — see SPEC-002 REQ-2.2. Changes belong in rlbot/, not here.
"""
portfolio_risk.py
=================
Phase 4 -- Portfolio risk layer.

    Per-ticker analysis is necessary but not sufficient. This module assembles
    the portfolio view your strategy actually requires:
      * concentration   -- no single underlying above a capital cap
      * correlation     -- "8 names" that all move together is one bet
      * expiration      -- staggered weeks, not clustered into one
      * earnings        -- flag cycles that span an earnings report

Analogy: per-ticker analysis inspects each plank; this module checks whether
the raft actually floats once the planks are lashed together.

(Fair-value anchors now live in fv_levels.py.)

Dependencies: numpy, pandas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Phase 4: Portfolio risk
# ----------------------------------------------------------------------

@dataclass
class PortfolioConfig:
    max_pct_per_underlying: float = 0.15   # no name above 15% of deployed capital
    max_positions_per_week: int = 3        # expiration-clustering cap
    high_correlation: float = 0.80         # pairwise corr above this = "clustered"
    target_positions: int = 9              # diversification target (8-10)


@dataclass
class PortfolioReport:
    concentration_flags: List[str] = field(default_factory=list)
    correlation_flags: List[str] = field(default_factory=list)
    expiration_flags: List[str] = field(default_factory=list)
    earnings_flags: List[str] = field(default_factory=list)
    correlation_matrix: Optional[pd.DataFrame] = None
    cluster_groups: List[List[str]] = field(default_factory=list)

    @property
    def all_flags(self) -> List[str]:
        return (self.concentration_flags + self.correlation_flags
                + self.expiration_flags + self.earnings_flags)


def check_concentration(
    positions: Dict[str, float],
    cfg: PortfolioConfig,
) -> List[str]:
    """
    positions: {ticker: capital_at_risk_dollars}
    Flags any name above the per-underlying cap.
    """
    flags: List[str] = []
    total = sum(positions.values())
    if total <= 0:
        return flags
    for ticker, capital in positions.items():
        pct = capital / total
        if pct > cfg.max_pct_per_underlying:
            flags.append(
                f"CONCENTRATION: {ticker} is {pct:.0%} of deployed capital "
                f"(cap {cfg.max_pct_per_underlying:.0%})."
            )
    return flags


def correlation_matrix(price_data: Dict[str, pd.Series], lookback: int = 120) -> pd.DataFrame:
    """Pairwise correlation of daily returns over the last `lookback` bars."""
    returns = {}
    for ticker, close in price_data.items():
        r = close.pct_change().dropna()
        if len(r) >= 20:
            returns[ticker] = r.tail(lookback)
    if len(returns) < 2:
        return pd.DataFrame()
    return pd.DataFrame(returns).corr()


def find_correlation_clusters(
    corr: pd.DataFrame,
    cfg: PortfolioConfig,
) -> tuple:
    """
    Group tickers whose pairwise correlation exceeds the threshold.
    Returns (flags, cluster_groups). A cluster of 3+ names is the real warning:
    it means apparent diversification is one underlying bet.
    """
    flags: List[str] = []
    groups: List[List[str]] = []
    if corr.empty:
        return flags, groups

    tickers = list(corr.columns)
    seen = set()
    for t in tickers:
        if t in seen:
            continue
        cluster = [t]
        for other in tickers:
            if other == t or other in seen:
                continue
            if corr.loc[t, other] >= cfg.high_correlation:
                cluster.append(other)
        if len(cluster) >= 2:
            for c in cluster:
                seen.add(c)
            groups.append(cluster)
            if len(cluster) >= 3:
                flags.append(
                    f"CORRELATION: {', '.join(cluster)} move together "
                    f"(pairwise corr >= {cfg.high_correlation:.2f}) -- "
                    f"counts closer to ONE position than {len(cluster)}."
                )
    return flags, groups


def check_expiration_clustering(
    expiries: Dict[str, str],
    cfg: PortfolioConfig,
) -> List[str]:
    """
    expiries: {ticker: 'YYYY-MM-DD'} of proposed option expiry dates.
    Flags any ISO week holding more than the per-week cap.
    """
    flags: List[str] = []
    by_week: Dict[str, List[str]] = {}
    for ticker, date_str in expiries.items():
        d = pd.to_datetime(date_str)
        iso = d.isocalendar()
        key = f"{iso[0]}-W{iso[1]:02d}"
        by_week.setdefault(key, []).append(ticker)
    for week, tickers in by_week.items():
        if len(tickers) > cfg.max_positions_per_week:
            flags.append(
                f"EXPIRATION: {len(tickers)} positions expire in {week} "
                f"({', '.join(tickers)}) -- stagger across more weeks."
            )
    return flags


def check_earnings_proximity(
    earnings_dates: Dict[str, List[str]],
    cycle_windows: Dict[str, tuple],
) -> List[str]:
    """
    earnings_dates: {ticker: ['YYYY-MM-DD', ...]} known/expected report dates.
    cycle_windows:  {ticker: (open_date, expiry_date)} for the proposed trade.
    Flags any trade whose cycle spans an earnings report.
    """
    flags: List[str] = []
    for ticker, (open_d, expiry_d) in cycle_windows.items():
        o = pd.to_datetime(open_d)
        e = pd.to_datetime(expiry_d)
        for ed_str in earnings_dates.get(ticker, []):
            ed = pd.to_datetime(ed_str)
            if o <= ed <= e:
                flags.append(
                    f"EARNINGS: {ticker} reports on {ed_str}, inside the "
                    f"{open_d} -> {expiry_d} cycle -- expect a volatility gap."
                )
    return flags


def build_portfolio_report(
    positions: Dict[str, float],
    price_data: Dict[str, pd.Series],
    expiries: Dict[str, str],
    earnings_dates: Dict[str, List[str]],
    cycle_windows: Dict[str, tuple],
    cfg: PortfolioConfig = PortfolioConfig(),
) -> PortfolioReport:
    """Run all Phase 4 checks and return one consolidated report."""
    report = PortfolioReport()
    report.concentration_flags = check_concentration(positions, cfg)
    corr = correlation_matrix(price_data)
    report.correlation_matrix = corr
    corr_flags, groups = find_correlation_clusters(corr, cfg)
    report.correlation_flags = corr_flags
    report.cluster_groups = groups
    report.expiration_flags = check_expiration_clustering(expiries, cfg)
    report.earnings_flags = check_earnings_proximity(earnings_dates, cycle_windows)
    return report


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def _run_tests() -> None:
    cfg = PortfolioConfig()
    rng = np.random.default_rng(0)

    # --- concentration: one oversized name is flagged ---
    pos = {"NVDA": 50_000, "MSFT": 10_000, "WMT": 10_000}
    cflags = check_concentration(pos, cfg)
    assert any("NVDA" in f for f in cflags), "should flag NVDA over-concentration"
    balanced = {"A": 10_000, "B": 10_000, "C": 10_000, "D": 10_000,
                "E": 10_000, "F": 10_000, "G": 10_000}
    assert check_concentration(balanced, cfg) == [], "balanced book = no flags"

    # --- correlation: build 3 correlated + 1 independent series ---
    base = rng.normal(0, 0.01, 200)
    pdata = {
        "SEMI1": pd.Series(100 * np.cumprod(1 + base + rng.normal(0, 0.001, 200))),
        "SEMI2": pd.Series(100 * np.cumprod(1 + base + rng.normal(0, 0.001, 200))),
        "SEMI3": pd.Series(100 * np.cumprod(1 + base + rng.normal(0, 0.001, 200))),
        "INDEP": pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 200))),
    }
    corr = correlation_matrix(pdata)
    assert not corr.empty and corr.shape == (4, 4)
    corr_flags, groups = find_correlation_clusters(corr, cfg)
    assert any("SEMI" in f for f in corr_flags), "should flag the semi cluster"
    assert any(len(g) >= 3 for g in groups)

    # --- expiration clustering: 4 in one week is flagged ---
    expiries = {"A": "2026-06-19", "B": "2026-06-19", "C": "2026-06-19",
                "D": "2026-06-19", "E": "2026-07-17"}
    eflags = check_expiration_clustering(expiries, cfg)
    assert any("expire" in f for f in eflags), "should flag the clustered week"
    spread = {"A": "2026-06-19", "B": "2026-07-17", "C": "2026-08-21"}
    assert check_expiration_clustering(spread, cfg) == []

    # --- earnings proximity: a cycle spanning a report is flagged ---
    earnings = {"MSFT": ["2026-07-22"]}
    windows = {"MSFT": ("2026-07-01", "2026-07-31")}
    erflags = check_earnings_proximity(earnings, windows)
    assert any("MSFT" in f for f in erflags), "should flag earnings inside cycle"
    safe_windows = {"MSFT": ("2026-08-01", "2026-08-31")}
    assert check_earnings_proximity(earnings, safe_windows) == []

    # --- end to end ---
    report = build_portfolio_report(pos, pdata, expiries, earnings, windows, cfg)
    assert len(report.all_flags) > 0
    assert report.correlation_matrix is not None

    print("portfolio_risk: all tests passed")


if __name__ == "__main__":
    _run_tests()
