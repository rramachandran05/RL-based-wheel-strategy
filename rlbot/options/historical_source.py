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

from rlbot.options.premium_source import Quote, SyntheticBSPremiumSource, bs_delta

SOURCE_NAME = "historical_chain"


def _greeks_degenerate(delta, iv) -> bool:
    """Predecessor-symbol eras (FB, old GOOG) carry placeholder greeks:
    delta pinned at ±1/0 and IV ~0.015 flat. Prices are real; greeks are not."""
    return pd.isna(delta) or pd.isna(iv) or iv <= 0.03 or abs(delta) >= 0.995 \
        or delta == 0.0


class HistoricalChainPremiumSource:
    source_name = SOURCE_NAME

    def __init__(self, ticker: str, chains_dir: Path,
                 fallback: SyntheticBSPremiumSource | None = None,
                 adj_ratio: pd.Series | None = None):
        """adj_ratio: date -> adjClose/close. Chains store RAW strikes and
        premiums; the simulator lives in adjusted space, so quotes are scaled
        by the snapshot day's ratio (per-day moneyness preserved exactly;
        contract lookups use nearest-strike matching to absorb the <1%
        dividend-adjustment drift across a cycle — a mid-cycle split breaks
        the match and falls back to BS marks, counted)."""
        self.ticker = ticker
        self.dir = Path(chains_dir) / ticker
        if not self.dir.exists():
            raise FileNotFoundError(f"no chain data for {ticker} under {chains_dir}")
        self.fallback = fallback or SyntheticBSPremiumSource()
        self.adj_ratio = adj_ratio
        self.fallback_count = 0
        self.request_count = 0
        self.greek_fallback_count = 0   # real mark kept, BS delta substituted
        self._years: dict = {}

    def _ratio(self, date) -> float:
        if self.adj_ratio is None:
            return 1.0
        d = pd.Timestamp(date).normalize()
        try:
            return float(self.adj_ratio.loc[d])
        except KeyError:
            idx = self.adj_ratio.index
            pos = idx.searchsorted(d)
            return float(self.adj_ratio.iloc[max(pos - 1, 0)])

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
        ratio = self._ratio(date)
        spot_raw = spot / ratio
        rows = day[(day["cp"] == cp)
                   & (day["dte"] >= dte_min) & (day["dte"] <= dte_max)
                   & (day["strike"] >= spot_raw * (1 - strike_span))
                   & (day["strike"] <= spot_raw * (1 + strike_span))]
        quotes = []
        for r in rows.itertuples():
            mid = r.mark if r.mark and r.mark > 0 else (r.bid + r.ask) / 2.0
            if not mid or mid <= 0.01:
                continue
            spread = (r.ask - r.bid) / mid if mid > 0 and r.ask >= r.bid else None
            if _greeks_degenerate(r.delta, r.iv):
                # real price, junk greeks: substitute BS delta on the vol proxy
                if vol_proxy is None or vol_proxy <= 0:
                    continue
                self.greek_fallback_count += 1
                delta = bs_delta(cp, spot, float(r.strike) * ratio,
                                 int(r.dte) / 365.0, float(vol_proxy))
                vol_used = float(vol_proxy)
            else:
                delta = float(r.delta)
                vol_used = float(r.iv)
            quotes.append(Quote(
                cp=cp, strike=float(r.strike) * ratio,
                expiration=pd.Timestamp(r.expiration), dte=int(r.dte),
                mid=float(mid) * ratio, delta=delta, vol_used=vol_used,
                volume=float(r.volume), oi=float(r.open_interest),
                spread_pct=float(spread) if spread is not None else None,
            ))
        return quotes

    def _contract_row(self, cp, strike_adj, expiration, date):
        """Nearest raw strike to strike_adj/ratio(date), within 1% tolerance
        (absorbs dividend-adjustment drift; a mid-cycle split misses -> None)."""
        day = self._day(date)
        if day is None or day.empty:
            return None
        raw_guess = strike_adj / self._ratio(date)
        rows = day[(day["cp"] == cp)
                   & (day["expiration"] == pd.Timestamp(expiration))]
        if not len(rows):
            return None
        diffs = (rows["strike"] - raw_guess).abs()
        pos = int(diffs.values.argmin())      # positional: the day index is duplicated
        if diffs.iloc[pos] > 0.01 * raw_guess:
            return None
        return rows.iloc[pos]

    def reprice(self, cp, strike, expiration, date, spot, vol_proxy) -> float:
        self.request_count += 1
        row = self._contract_row(cp, strike, expiration, date)
        if row is not None:
            mid = row["mark"] if row["mark"] and row["mark"] > 0 \
                else (row["bid"] + row["ask"]) / 2.0
            if mid and mid > 0:
                return float(mid) * self._ratio(date)
        self.fallback_count += 1
        return self.fallback.reprice(cp, strike, expiration, date, spot, vol_proxy)

    def fallback_share(self) -> float:
        return self.fallback_count / self.request_count if self.request_count else 0.0

    def delta_now(self, cp, strike, expiration, date, spot, vol_proxy) -> float:
        row = self._contract_row(cp, strike, expiration, date)
        if row is not None and not _greeks_degenerate(row["delta"], row["iv"]):
            return float(row["delta"])
        return self.fallback.delta_now(cp, strike, expiration, date, spot, vol_proxy)


def historical_source_for(ticker: str, cfg, iv_uplift: float = 0.0
                          ) -> HistoricalChainPremiumSource:
    """Factory: real chains + adj_ratio + synthetic fallback for one ticker."""
    from rlbot.data.unadjusted import load_adj_ratio

    return HistoricalChainPremiumSource(
        ticker, cfg.data.base_path / "chains",
        fallback=SyntheticBSPremiumSource(iv_uplift=iv_uplift),
        adj_ratio=load_adj_ratio(ticker, cfg),
    )
