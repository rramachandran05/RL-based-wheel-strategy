"""MCB feed (2026-08-30): ../mcb-wheel's Maximum-Comfortable-Basis report
replaces the fair-value-discount Wheel-FV feed as the valuation-gate input.

Consumer contract (mcb-wheel/specs/mcb-spec-outline.md):
  1. HARD: NetBasis = Strike − Premium ≤ MCB(required tier), required tier =
     deeper of the row's min_eligible_tier and our regime posture.
  2. Reachability is advisory (UNREACHABLE → skip strike scan; PATIENCE →
     elevated-IV setups only). 3. No momentum inputs. 4. Reports older than
     5 trading sessions are expired → gates no-op with a warning.

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
    min_eligible_tier: str         # guardrail-resolved by the producer
    guardrail: str | None          # NORMAL / CAUTION / SEVERE / None (ETFs)
    layer_a: str                   # OWN / MONITOR_ONLY / HALT
    reachability: str | None      # NORMAL / PATIENCE / UNREACHABLE
    confidence: float | None

    def ceiling(self, tier: str) -> float | None:
        return self.mcb.get(tier)


def mcb_dir(cfg: RlbotConfig | None = None) -> Path:
    if cfg is not None and hasattr(cfg.data, "mcb_dir"):
        return Path(cfg.data.mcb_dir)
    return MCB_DIR


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
        out[ticker] = McbRow(
            ticker=ticker, date=str(date.date()), mcb=mcb,
            min_eligible_tier=tier if tier in TIERS else "FAIR",
            guardrail=str(r.guardrail_status) if pd.notna(r.guardrail_status) else None,
            layer_a=str(r.layer_a) if pd.notna(r.layer_a) else "OWN",
            reachability=str(r.reachability) if pd.notna(r.reachability) else None,
            confidence=float(r.conf) if pd.notna(r.conf) else None,
        )
    return out, warnings
