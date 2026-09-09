"""MCB feed: ../mcb-wheel's Maximum-Comfortable-Basis report replaces the
fair-value-discount Wheel-FV feed as the valuation-gate input.

Consumer contract v2 (SPEC-011 §2, 2026-09-09 — supersedes the v1 universal
hard ceiling):
  1. NetBasis = Strike − Premium ≤ wheel_entry, bindingness scaled by the
     producer's delta_posture (CONSERVATIVE never blocks; HIGHER always
     hard; MODERATE resolved by our own tier split) — see rlbot.risk.mcb_gates.
  2. Reachability is advisory (unchanged). 3. No momentum inputs.
  4. Reports older than 5 trading sessions are expired → gates no-op.

PIT discipline: only files dated on/before as_of are eligible.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from rlbot.config import PROJECT_ROOT, RlbotConfig

MCB_DIR = PROJECT_ROOT.parent / "mcb-wheel" / "outputs"
_FNAME_RE = re.compile(r"mcb_(\d{4}-\d{2}-\d{2})\.csv$")
STALE_SESSIONS = 5          # contract rule 4 (calendar-day approximation: 7)
TIERS = ("FAIR", "ATTRACTIVE", "EXCELLENT")   # loosest -> deepest


@dataclass(frozen=True)
class McbRow:
    ticker: str
    date: str
    mcb: dict                      # tier -> float (only valid tiers present)
    min_eligible_tier: str         # v2: data-quality-only, no longer the gate
    guardrail: str | None          # NORMAL / CAUTION / SEVERE / None (ETFs)
    layer_a: str                   # OWN / MONITOR_ONLY / HALT
    reachability: str | None      # NORMAL / PATIENCE / UNREACHABLE
    confidence: float | None
    # v2 fields (SPEC-011 §1/§2 rule 1) — the aggressiveness-pivot gate
    wheel_entry: float | None = None       # the actual ceiling to enforce
    delta_posture: str | None = None       # CONSERVATIVE / MODERATE / HIGHER
    dd_now: float | None = None            # current drawdown from trailing high
    drop_needed: float | None = None       # (spot - wheel_entry) / spot
    action: str | None = None              # producer advisory label
    shadow_min_tier: str | None = None     # old guardrail tier, calibration only
    # correction-history percentiles (drawdown from trailing high) — feed the
    # score-side valuation state (SPEC-011 §2 rule 6) with dd_now/drop_needed
    dd50: float | None = None
    dd75: float | None = None
    dd90: float | None = None

    def ceiling(self, tier: str) -> float | None:
        return self.mcb.get(tier)


def mcb_dir(cfg: RlbotConfig | None = None) -> Path:
    if cfg is not None and hasattr(cfg.data, "mcb_dir"):
        return Path(cfg.data.mcb_dir)
    return MCB_DIR


def _f(v) -> float | None:
    """Nullable float from a CSV cell (NaN/None -> None)."""
    return float(v) if v is not None and pd.notna(v) else None


def load_mcb(cfg: RlbotConfig | None = None, as_of=None) -> tuple:
    """-> ({ticker: McbRow}, warnings). Empty dict on any gap (gates no-op)."""
    as_of = pd.Timestamp(as_of or pd.Timestamp.now()).normalize()
    directory = mcb_dir(cfg)
    files = []
    if directory.is_dir():
        for p in directory.glob("mcb_*.csv"):
            m = _FNAME_RE.search(p.name)
            if m and pd.Timestamp(m.group(1)) <= as_of:
                files.append((pd.Timestamp(m.group(1)), p))
    if not files:
        return {}, [f"no MCB report found under {directory}; valuation gates inactive"]
    date, path = max(files)
    if (as_of - date).days > STALE_SESSIONS + 2:
        return {}, [f"MCB report {date.date()} older than {STALE_SESSIONS} "
                    "sessions — expired per contract; valuation gates inactive"]
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return {}, [f"MCB report unreadable ({e}); valuation gates inactive"]

    out, warnings = {}, []
    for r in df.itertuples():
        ticker = str(r.ticker).upper()
        mcb = {}
        for tier, col in (("FAIR", r.mcb_fair), ("ATTRACTIVE", r.mcb_attractive),
                          ("EXCELLENT", r.mcb_excellent)):
            if pd.notna(col) and float(col) > 0:
                mcb[tier] = float(col)
        if not mcb:
            warnings.append(f"{ticker}: MCB row has no usable zones "
                            "(constraint absent)")
            continue
        tier = str(r.min_eligible_tier) if pd.notna(r.min_eligible_tier) else "FAIR"
        # v2 columns (SPEC-011 §1): absent gracefully on an older-schema CSV
        # (getattr default), matching this repo's "constraint absent, never
        # a guess" philosophy rather than raising on a missing column.
        wheel_entry = getattr(r, "wheel_entry", None)
        posture = getattr(r, "delta_posture", None)
        shadow_tier = getattr(r, "shadow_min_tier", None)
        out[ticker] = McbRow(
            ticker=ticker, date=str(date.date()), mcb=mcb,
            min_eligible_tier=tier if tier in TIERS else "FAIR",
            guardrail=str(r.guardrail_status) if pd.notna(r.guardrail_status) else None,
            layer_a=str(r.layer_a) if pd.notna(r.layer_a) else "OWN",
            reachability=str(r.reachability) if pd.notna(r.reachability) else None,
            confidence=float(r.conf) if pd.notna(r.conf) else None,
            wheel_entry=float(wheel_entry) if pd.notna(wheel_entry) else None,
            delta_posture=str(posture) if pd.notna(posture) else None,
            dd_now=float(getattr(r, "dd_now", None)) if pd.notna(getattr(r, "dd_now", None)) else None,
            drop_needed=float(getattr(r, "drop_needed", None)) if pd.notna(getattr(r, "drop_needed", None)) else None,
            action=str(getattr(r, "action", None)) if pd.notna(getattr(r, "action", None)) else None,
            shadow_min_tier=str(shadow_tier) if pd.notna(shadow_tier) and str(shadow_tier) in TIERS else None,
            dd50=_f(getattr(r, "dd50", None)),
            dd75=_f(getattr(r, "dd75", None)),
            dd90=_f(getattr(r, "dd90", None)),
        )
        if wheel_entry is None or pd.isna(wheel_entry):
            warnings.append(f"{ticker}: no wheel_entry (v2 schema) — "
                            "MCB gate inactive for this ticker (constraint "
                            "absent, not guessed)")
    return out, warnings
