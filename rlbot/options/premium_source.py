"""Premium sources (SPEC-002 §3.7, APPROACH §12).

v1 ships only ``SyntheticBSPremiumSource`` — per the 2026-08-21 decision there
is no chain-data vendor; every premium is Black-Scholes on the ticker's
realized-vol proxy (vendored options_engine), with an optional global
``iv_uplift`` scalar fitted once by the PUT-index calibration gate
(SPEC-003 §7). A ``HistoricalChainPremiumSource`` would implement the same
interface if chain data ever arrives.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from rlbot.vendor.options_engine import _d1_d2, bs_call_price, bs_put_price

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class Quote:
    cp: str                      # "P" | "C"
    strike: float
    expiration: pd.Timestamp
    dte: int                     # calendar days
    mid: float                   # per-share price (model or market)
    delta: float                 # signed delta (puts negative)
    vol_used: float
    # real-chain liquidity fields; None on the synthetic track (SPEC-004 §1.1)
    volume: float | None = None
    oi: float | None = None
    spread_pct: float | None = None


def bs_delta(cp: str, spot: float, strike: float, t_years: float, vol: float, r: float = 0.0) -> float:
    d1, _ = _d1_d2(spot, strike, t_years, vol, r)
    return float(norm.cdf(d1)) if cp == "C" else float(norm.cdf(d1) - 1.0)


def strike_increment(spot: float) -> float:
    if spot < 25:
        return 0.5
    if spot < 100:
        return 1.0
    if spot < 250:
        return 2.5
    return 5.0


def expiration_fridays(date: pd.Timestamp, dte_min: int, dte_max: int) -> list:
    """All Fridays with calendar DTE in [dte_min, dte_max] (synthetic listings)."""
    out = []
    d = date + pd.Timedelta(days=dte_min)
    while (d - date).days <= dte_max:
        if d.dayofweek == 4:
            out.append(d.normalize())
        d += pd.Timedelta(days=1)
    return out


class SyntheticBSPremiumSource:
    """Synthetic chain + repricing from BS on a realized-vol proxy."""

    source_name = "synthetic_bs"

    def __init__(self, iv_uplift: float = 0.0, r: float = 0.0):
        self.iv_uplift = iv_uplift
        self.r = r

    def _vol(self, vol_proxy: float) -> float:
        return vol_proxy * (1.0 + self.iv_uplift)

    def price(self, cp: str, spot: float, strike: float, t_years: float, vol_proxy: float) -> float:
        vol = self._vol(vol_proxy)
        fn = bs_call_price if cp == "C" else bs_put_price
        return float(fn(spot, strike, t_years, vol, self.r))

    def chain(
        self,
        date: pd.Timestamp,
        spot: float,
        vol_proxy: float,
        cp: str,
        dte_min: int = 25,
        dte_max: int = 45,
        strike_span: float = 0.45,
    ) -> list:
        """Synthetic listed chain: strike grid ±strike_span around spot,
        every Friday expiration in the DTE window."""
        inc = strike_increment(spot)
        lo = np.ceil(spot * (1 - strike_span) / inc) * inc
        hi = np.floor(spot * (1 + strike_span) / inc) * inc
        strikes = np.arange(lo, hi + inc / 2, inc)
        vol = self._vol(vol_proxy)
        quotes = []
        for exp in expiration_fridays(date, dte_min, dte_max):
            dte = (exp - date.normalize()).days
            t = dte / 365.0
            for k in strikes:
                mid = self.price(cp, spot, float(k), t, vol_proxy)
                if mid < 0.01:
                    continue
                quotes.append(Quote(
                    cp=cp, strike=float(k), expiration=exp, dte=dte, mid=mid,
                    delta=bs_delta(cp, spot, float(k), t, vol, self.r),
                    vol_used=vol,
                ))
        return quotes

    def delta_now(
        self, cp: str, strike: float, expiration: pd.Timestamp,
        date: pd.Timestamp, spot: float, vol_proxy: float,
    ) -> float:
        """Signed BS delta of an open contract on a later date."""
        dte = max((expiration - date.normalize()).days, 0)
        if dte == 0:
            itm = (spot < strike) if cp == "P" else (spot > strike)
            return (-1.0 if cp == "P" else 1.0) if itm else 0.0
        return bs_delta(cp, spot, strike, dte / 365.0, self._vol(vol_proxy), self.r)

    def reprice(
        self, cp: str, strike: float, expiration: pd.Timestamp,
        date: pd.Timestamp, spot: float, vol_proxy: float,
    ) -> float:
        """Mark an open contract to model mid on a later date."""
        dte = max((expiration - date.normalize()).days, 0)
        if dte == 0:
            intrinsic = max(strike - spot, 0.0) if cp == "P" else max(spot - strike, 0.0)
            return float(intrinsic)
        return self.price(cp, spot, strike, dte / 365.0, vol_proxy)
