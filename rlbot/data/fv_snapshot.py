"""Fair-value snapshotting, ported from ../wheel-strategy/fv_levels.py
(2026-08-23) so the sibling repo is no longer a runtime dependency.

Semantics preserved exactly (golden-tested); sheet columns updated
2026-08-23 — the analyst column is now "TipRanks (mean)" (was Morningstar):
  fv_buy  = min(fmp_median, tipranks_mean, stock_oracle)  over present inputs
  fv_sell = max(fmp_median, tipranks_mean, current_price) over present inputs
  ladders = whole-dollar rungs at ±6/12/18%
  ETFs (no sheet FV): 9-month volume-profile nearest support/resistance proxy
  confidence: high (3 inputs) / medium (2) / low (1)

Writes the sibling-schema `fair_value_<date>.csv` into
data_local/fv_snapshots/; rlbot's valuation table reads BOTH the legacy
sibling snapshots and this directory (newest wins per date+ticker).

Run:  python -m rlbot.data.fv_snapshot
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import requests

from rlbot.config import RlbotConfig
from rlbot.data.env import get_key
from rlbot.data.sources import DataUnavailable, load_bars
from rlbot.vendor.sheet_data import fetch_sheet_rows, normalize_ticker, parse_dollar, rows_to_dicts
from rlbot.vendor.technicals import volume_profile_levels

FV_SHEET_ID = "1IW4cNkUsTLgylGe5VkkxjOf-XBHtKWmTQ5xzCIbd_9Y"
FV_GID = "274275982"
FV_HEADER_ROW = 1
LADDER_PCTS = (0.06, 0.12, 0.18)
_TICKER_RE = re.compile(r"^[A-Z]{1,5}([.\-][A-Z]{1,2})?$")


def parse_fv_rows(raw_rows: list) -> dict:
    """{TICKER: {"tipranks": float|None, "stock_oracle": float|None}}"""
    out = {}
    for row in rows_to_dicts(raw_rows, FV_HEADER_ROW):
        t = normalize_ticker(row.get("Stock", ""))
        if not t or t in ("STOCK", "ETFS") or not _TICKER_RE.match(t):
            continue
        out[t] = {
            "tipranks": parse_dollar(row.get("TipRanks (mean)", "")),
            "stock_oracle": parse_dollar(row.get("Stock Oracle (Intrinsic Value)", "")),
        }
    return out


def fv_anchors(tipranks, stock_oracle, fmp_median, current_price):
    buys = [v for v in (fmp_median, tipranks, stock_oracle) if v]
    sells = [v for v in (fmp_median, tipranks, current_price) if v]
    if not buys or not sells:
        raise ValueError("no valuation inputs")
    return min(buys), max(sells)


def buy_sell_levels(fv_buy, fv_sell, pcts=LADDER_PCTS):
    return ([round(fv_buy * (1 - p)) for p in pcts],
            [round(fv_sell * (1 + p)) for p in pcts])


def etf_fv_proxy(df: pd.DataFrame, months: int = 9):
    """Nearest volume-profile support below / resistance above current price."""
    window = df.iloc[-21 * months:]
    price = float(window["Close"].iloc[-1])
    supports, resistances = volume_profile_levels(window)
    below = [s["price"] for s in supports if s["price"] < price]
    above = [r["price"] for r in resistances if r["price"] > price]
    return (max(below) if below else price), (min(above) if above else price)


def fmp_price_target_median(ticker: str, timeout: int = 12):
    key = get_key("FMP_API_KEY")
    if not key:
        return None
    try:
        r = requests.get("https://financialmodelingprep.com/stable/price-target-consensus",
                         params={"symbol": ticker, "apikey": key}, timeout=timeout)
        j = r.json()
        row = j[0] if isinstance(j, list) and j else (j if isinstance(j, dict) else {})
        v = row.get("targetMedian")
        return float(v) if v else None
    except Exception:
        return None


def snapshot_fair_value(cfg: RlbotConfig, date_str: str | None = None) -> Path:
    raw = fetch_sheet_rows(FV_SHEET_ID, FV_GID)
    if not raw:
        raise DataUnavailable("FV sheet unreachable")
    sheet = parse_fv_rows(raw)

    records = []
    for t in cfg.assistant_universe:
        try:
            bars = load_bars(t, cfg.data.bars_path)
        except DataUnavailable:
            continue
        price = float(bars["Close"].iloc[-1])
        row = sheet.get(t, {})
        tr, so = row.get("tipranks"), row.get("stock_oracle")
        fmp = fmp_price_target_median(t) if (tr or so) else None
        try:
            if tr or so or fmp:
                fv_buy, fv_sell = fv_anchors(tr, so, fmp, price)
                n_inputs = sum(1 for v in (tr, so, fmp) if v)
                source = "sheet+fmp" if fmp else "sheet"
                confidence = {3: "high", 2: "medium"}.get(n_inputs, "low")
            else:
                fv_buy, fv_sell = etf_fv_proxy(bars)
                source, confidence = "support-resistance", "low"
        except ValueError:
            continue
        buys, sells = buy_sell_levels(fv_buy, fv_sell)
        records.append({
            "Ticker": t, "Price": round(price, 2),
            "FV_Buy": round(fv_buy, 2), "FV_Sell": round(fv_sell, 2),
            "Buy_L1": buys[0], "Buy_L2": buys[1], "Buy_L3": buys[2],
            "Sell_L1": sells[0], "Sell_L2": sells[1], "Sell_L3": sells[2],
            "FMP_Median": fmp, "Source": source, "Confidence": confidence,
        })
    if not records:
        raise DataUnavailable("FV snapshot produced no rows")
    date_str = date_str or pd.Timestamp.now().strftime("%Y-%m-%d")
    out_dir = cfg.data.base_path / "fv_snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"fair_value_{date_str}.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def refresh_valuation(cfg: RlbotConfig) -> Path:
    """Rebuild the canonical valuation table from ALL snapshot dirs."""
    from rlbot.data.build import build_valuation
    frames = []
    for d in (cfg.data.fair_value_snapshot_dir, cfg.data.base_path / "fv_snapshots"):
        table = build_valuation(Path(d), cfg)
        if not table.empty:
            frames.append(table.reset_index())
    if not frames:
        raise DataUnavailable("no fair-value snapshots anywhere")
    merged = pd.concat(frames).drop_duplicates(subset=["date", "ticker"], keep="last") \
        .set_index(["date", "ticker"]).sort_index()
    path = cfg.data.canonical_path / "valuation.parquet"
    merged.to_parquet(path)
    return path


if __name__ == "__main__":
    cfg = RlbotConfig()
    p = snapshot_fair_value(cfg)
    print(f"wrote {p}")
    v = refresh_valuation(cfg)
    print(f"rebuilt {v}")
