"""Ingest ../fair-value-discount ensemble CSVs (SPEC-009 VREQ-1).

The sibling engine (its SPEC-002) publishes fair_value_ensemble_<date>.csv
daily: per-ticker wheel_fv (reliability-weighted blend of the intrinsic
ensemble and a haircut analyst target), IV reliability, put_required_mos,
and analyst sentiment. This module loads the newest file into
{ticker: WheelValuation} for the valuation gates.

Degradation (VREQ-7): missing directory, no eligible file, or a file older
than ValuationGateConfig.max_age_days -> ({}, [warning]); gates become
no-ops, mirroring the FAIR-unknown philosophy of SPEC-001 §3.1.

PIT discipline: only files dated on/before `as_of` are eligible, so a
future replay can never read tomorrow's valuation.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from rlbot.config import RlbotConfig
from rlbot.risk.valuation import WheelValuation

_FNAME_RE = re.compile(r"fair_value_ensemble_(\d{4}-\d{2}-\d{2})\.csv$")


def _eligible_files(directory: Path, as_of: pd.Timestamp) -> list:
    out = []
    if not directory.is_dir():
        return out
    for p in directory.glob("fair_value_ensemble_*.csv"):
        m = _FNAME_RE.search(p.name)
        if m and pd.Timestamp(m.group(1)) <= as_of:
            out.append((pd.Timestamp(m.group(1)), p))
    return sorted(out)


def load_wheel_valuations(
    cfg: RlbotConfig,
    as_of: pd.Timestamp | None = None,
) -> tuple[dict, list]:
    """-> ({ticker: WheelValuation}, warnings). Empty dict on any gap."""
    gate_cfg = cfg.val_gates
    as_of = (as_of or pd.Timestamp.now()).normalize()
    if not gate_cfg.enabled:
        return {}, []
    files = _eligible_files(Path(cfg.data.fv_ensemble_dir), as_of)
    if not files:
        return {}, [f"valuation gates off: no ensemble CSV in "
                    f"{cfg.data.fv_ensemble_dir}"]
    date, path = files[-1]
    age = (as_of - date).days
    if age > gate_cfg.max_age_days:
        return {}, [f"valuation gates off: newest ensemble CSV is {age}d old "
                    f"({path.name}); refresh fair-value-discount"]

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return {}, [f"valuation gates off: unreadable {path.name}: {e}"]
    need = {"ticker", "wheel_fv", "put_required_mos"}
    if not need.issubset(df.columns):
        return {}, [f"valuation gates off: {path.name} missing columns "
                    f"{sorted(need - set(df.columns))}"]

    out, warnings = {}, []
    for _, row in df.iterrows():
        ticker = str(row["ticker"]).upper()
        fv, mos = row.get("wheel_fv"), row.get("put_required_mos")
        if pd.isna(fv) or fv <= 0 or pd.isna(mos):
            continue                       # no Wheel_FV -> ungated (VREQ-7)
        cov = row.get("analyst_coverage")
        rel = row.get("iv_reliability")
        sent = row.get("analyst_sentiment")
        out[ticker] = WheelValuation(
            ticker=ticker,
            date=str(date.date()),
            wheel_fv=float(fv),
            put_required_mos=float(mos),
            reliability=None if pd.isna(rel) else float(rel),
            reliability_tier=str(row.get("iv_reliability_tier") or "med"),
            sentiment=None if pd.isna(sent) else float(sent),
            coverage=0 if pd.isna(cov) else int(cov),
        )
    if age > 0:
        warnings.append(f"valuation gates use {path.name} ({age}d old)")
    return out, warnings
