"""HistoricalChainPremiumSource (SPEC-002 §3.7) — real AV chains, same
interface as the synthetic source, so the simulator/selector/assistant swap
in without changes. Per-ticker (unlike the synthetic source): construct one
per ticker over data_local/chains/<TICKER>/<year>.parquet.

Marking rules:
- quote mid = mark when > 0, else (bid+ask)/2; rows with no positive price
  are dropped from chains (unquotable).
- reprice(): exact-contract row's mark on that date; missing row (holiday
  gap, filtered strike) falls back to BS on the ticker's vol proxy — counted
  in .fallback_count so runs can report data-coverage honesty.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from rlbot.options.premium_source import Quote, SyntheticBSPremiumSource

SOURCE_NAME = "historical_chain"


class HistoricalChainPremiumSource:
    source_name = SOURCE_NAME

    def __init__(self, ticker: str, chains_dir: Path,
                 fallback: SyntheticBSPremiumSource | None = None):
        self.ticker = ticker
        self.dir = Path(chains_dir) / ticker
        if not self.dir.exists():
            raise FileNotFoundError(f"no chain data for {ticker} under {chains_dir}")
        self.fallback = fallback or SyntheticBSPremiumSource()
        self.fallback_count = 0
        self.request_count = 0
        self._years: dict = {}

    # ------------------------------------------------------------------
    def _year_df(self, year: int) -> pd.DataFrame | None:
        if year not in self._years:
            path = self.dir / f"{year}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                df = df.set_index("snapshot_date").sort_index()
                self._years[year] = df
            else:
                self._years[year] = None
        return self._years[year]

    def _day(self, date: pd.Timestamp) -> pd.DataFrame | None:
        df = self._year_df(pd.Timestamp(date).year)
        if df is None:
            return None
        try:
            day = df.loc[[pd.Timestamp(date).normalize()]]
        except KeyError:
            return None
        return day

    # ------------------------------------------------------------------
    def chain(self, date, spot, vol_proxy, cp,
              dte_min: int = 25, dte_max: int = 45,
              strike_span: float = 0.45) -> list:
        day = self._day(date)
        if day is None or day.empty:
            return []          # no chain that day -> caller WAITs (honest gap)
        rows = day[(day["cp"] == cp)
                   & (day["dte"] >= dte_min) & (day["dte"] <= dte_max)
                   & (day["strike"] >= spot * (1 - strike_span))
                   & (day["strike"] <= spot * (1 + strike_span))]
        quotes = []
        for r in rows.itertuples():
            mid = r.mark if r.mark and r.mark > 0 else (r.bid + r.ask) / 2.0
            if not mid or mid <= 0.01 or pd.isna(r.delta):
                continue
            spread = (r.ask - r.bid) / mid if mid > 0 and r.ask >= r.bid else None
            quotes.append(Quote(
                cp=cp, strike=float(r.strike),
                expiration=pd.Timestamp(r.expiration), dte=int(r.dte),
                mid=float(mid), delta=float(r.delta),
                vol_used=float(r.iv) if pd.notna(r.iv) else 0.0,
                volume=float(r.volume), oi=float(r.open_interest),
                spread_pct=float(spread) if spread is not None else None,
            ))
        return quotes

    def _contract_row(self, cp, strike, expiration, date):
        day = self._day(date)
        if day is None or day.empty:
            return None
        rows = day[(day["cp"] == cp)
                   & (day["strike"].sub(strike).abs() < 1e-6)
                   & (day["expiration"] == pd.Timestamp(expiration))]
        return rows.iloc[0] if len(rows) else None

    def reprice(self, cp, strike, expiration, date, spot, vol_proxy) -> float:
        self.request_count += 1
        row = self._contract_row(cp, strike, expiration, date)
        if row is not None:
            mid = row["mark"] if row["mark"] and row["mark"] > 0 \
                else (row["bid"] + row["ask"]) / 2.0
            if mid and mid > 0:
                return float(mid)
        self.fallback_count += 1
        return self.fallback.reprice(cp, strike, expiration, date, spot, vol_proxy)

    def delta_now(self, cp, strike, expiration, date, spot, vol_proxy) -> float:
        row = self._contract_row(cp, strike, expiration, date)
        if row is not None and pd.notna(row["delta"]):
            return float(row["delta"])
        return self.fallback.delta_now(cp, strike, expiration, date, spot, vol_proxy)
