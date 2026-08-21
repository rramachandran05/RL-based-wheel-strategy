"""GATE G1 — simulator calibration vs. the CBOE PUT index (SPEC-003 §7).

Replicates PUT methodology inside the simulator: sell a one-month ATM SPY put
at each monthly expiration, fully cash-collateralized (fractional contracts,
index-style), hold to expiration, repeat. Compares monthly returns against the
real PUT index. A single global ``iv_uplift`` premium scalar is fitted ONCE on
the first half of the window and validated on the second half.

Run:  python -m rlbot.evaluation.put_gate
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rlbot.config import RlbotConfig
from rlbot.data import sources
from rlbot.features.regime import realized_vol
from rlbot.options.premium_source import SyntheticBSPremiumSource
from rlbot.simulator.portfolio import sell_open_fill, ExecutionConfig

UPLIFT_GRID = np.arange(0.0, 0.51, 0.05)


def third_fridays(start: pd.Timestamp, end: pd.Timestamp) -> list:
    out = []
    for period in pd.period_range(start, end, freq="M"):
        d = pd.Timestamp(period.year, period.month, 15)
        while d.dayofweek != 4:
            d += pd.Timedelta(days=1)
        if start <= d <= end:
            out.append(d)
    return out


def _next_trading_day(idx: pd.DatetimeIndex, date: pd.Timestamp):
    pos = idx.searchsorted(date)
    return idx[pos] if pos < len(idx) else None


def run_put_replication(spy: pd.DataFrame, iv_uplift: float,
                        exec_cfg: ExecutionConfig = ExecutionConfig()) -> pd.Series:
    """Daily NAV of the ATM-put-writing program. spy: close + vol_proxy."""
    ps = SyntheticBSPremiumSource(iv_uplift=iv_uplift)
    idx = spy.index
    fridays = third_fridays(idx[0], idx[-1])
    roll_dates = [d for d in (_next_trading_day(idx, f) for f in fridays) if d is not None]
    if len(roll_dates) < 3:
        raise ValueError("window too short for PUT replication")

    nav = 1.0
    navs = {}
    strike = premium = contracts = 0.0
    expiry = None
    k = 0
    for date in idx:
        row = spy.loc[date]
        spot, vol = float(row["close"]), row["vol_proxy"]
        if pd.isna(vol):
            navs[date] = nav
            continue
        # settle at expiration
        if expiry is not None and date >= expiry:
            payoff = max(strike - spot, 0.0)
            nav = collateral_base - payoff * contracts
            expiry = None
        # roll on schedule
        if k < len(roll_dates) and date >= roll_dates[k]:
            while k < len(roll_dates) and date >= roll_dates[k]:
                k += 1
            if k <= len(roll_dates) - 1 and expiry is None:
                next_exp = roll_dates[k]
                dte = (next_exp - date).days
                strike = spot
                mid = ps.price("P", spot, strike, dte / 365.0, float(vol))
                fill = sell_open_fill(mid, exec_cfg)
                contracts = nav / strike            # index-style full collateral
                collateral_base = nav + fill * contracts
                expiry = next_exp
        # daily MTM
        if expiry is not None:
            dte_left = max((expiry - date).days, 0)
            mark = ps.reprice("P", strike, expiry, date, spot, float(vol))
            nav_today = collateral_base - mark * contracts
        else:
            nav_today = nav
        navs[date] = nav_today
        if expiry is None:
            nav = nav_today
    return pd.Series(navs, name="replication_nav")


def monthly_metrics(rep_nav: pd.Series, put_index: pd.Series) -> dict:
    """Compare monthly (calendar month-end) returns of replication vs PUT."""
    joined = pd.DataFrame({"rep": rep_nav, "put": put_index}).dropna()
    monthly = joined.resample("ME").last().dropna()
    rets = monthly.pct_change().dropna()
    if len(rets) < 12:
        raise ValueError("not enough overlapping months")
    years = (monthly.index[-1] - monthly.index[0]).days / 365.25
    ann = {c: (monthly[c].iloc[-1] / monthly[c].iloc[0]) ** (1 / years) - 1 for c in monthly}
    vol = {c: rets[c].std() * np.sqrt(12) for c in rets}
    return {
        "n_months": int(len(rets)),
        "corr": float(rets["rep"].corr(rets["put"])),
        "ann_ret_rep": float(ann["rep"]),
        "ann_ret_put": float(ann["put"]),
        "ann_ret_diff": float(ann["rep"] - ann["put"]),
        "vol_rep": float(vol["rep"]),
        "vol_put": float(vol["put"]),
        "vol_ratio": float(vol["rep"] / vol["put"]),
    }


def run_gate(cfg: RlbotConfig | None = None,
             corr_min: float = 0.85, ret_diff_max: float = 0.03,
             vol_ratio_band: tuple = (0.75, 1.30)) -> dict:
    cfg = cfg or RlbotConfig()
    spy_bars = sources.load_bars(cfg.market_ticker, cfg.data.bars_path)
    spy = pd.DataFrame({"close": spy_bars["Close"]})
    spy.index = spy.index.tz_localize(None)
    spy["vol_proxy"] = realized_vol(spy["close"], cfg.data.realized_vol_ticker_window)
    spy = spy.loc["2010-01-01":].dropna()
    put = sources.load_put_index(cfg.data.external_path).loc[spy.index[0]:]

    mid = spy.index[len(spy) // 2]
    first, second = spy.loc[:mid], spy.loc[mid:]

    # fit uplift on first half
    best_uplift, best_err = 0.0, np.inf
    fit_metrics = {}
    for u in UPLIFT_GRID:
        m = monthly_metrics(run_put_replication(first, u), put)
        if abs(m["ann_ret_diff"]) < best_err:
            best_uplift, best_err, fit_metrics = float(u), abs(m["ann_ret_diff"]), m

    val_metrics = monthly_metrics(run_put_replication(second, best_uplift), put)
    full_metrics = monthly_metrics(run_put_replication(spy, best_uplift), put)

    passed = (
        val_metrics["corr"] >= corr_min
        and abs(val_metrics["ann_ret_diff"]) <= ret_diff_max
        and vol_ratio_band[0] <= val_metrics["vol_ratio"] <= vol_ratio_band[1]
    )
    verdict = {
        "pass": bool(passed),
        "iv_uplift": best_uplift,
        "window": [str(spy.index[0].date()), str(spy.index[-1].date())],
        "split_date": str(mid.date()),
        "thresholds": {"corr_min": corr_min, "ret_diff_max": ret_diff_max,
                        "vol_ratio_band": list(vol_ratio_band)},
        "fit_half": fit_metrics,
        "validation_half": val_metrics,
        "full_window": full_metrics,
    }
    out = cfg.data.base_path / "calibration" / "put_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict, indent=2))
    return verdict


def require_gate(cfg: RlbotConfig) -> dict:
    """Learning entry point guard (SPEC-003 AC-5)."""
    path = cfg.data.base_path / "calibration" / "put_gate.json"
    if not path.exists():
        raise RuntimeError("GATE G1 not run: execute python -m rlbot.evaluation.put_gate first")
    verdict = json.loads(path.read_text())
    if not verdict.get("pass"):
        raise RuntimeError(f"GATE G1 failed: {verdict['validation_half']}")
    return verdict


if __name__ == "__main__":
    v = run_gate()
    print(json.dumps(v, indent=2))
