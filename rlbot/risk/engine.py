"""Hard risk engine (SPEC-004 §2). RL proposes, this disposes.

v1 scope: single-ticker episodes. Portfolio-level rules (RISK-3/4/5/8/9) are
implemented against the provided portfolio snapshot; in single-ticker
simulation the caps that only make sense for a multi-name book are relaxed via
``RiskConfig.single_ticker()`` (documented deviation, SPEC-004).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RiskConfig:
    # SPEC-004 §2 v2 (2026-08-31): capital rules read exposure/NAV, never
    # position counts (counts mislead when contract sizes differ 10x). The
    # one surviving count is distinct underlyings — an attention cap.
    # Two-tier disposition: hard blocks vs human-review warnings.
    max_pct_per_underlying: float = 0.15   # RISK-3: potential exposure/NAV
    max_underlyings: int = 12               # RISK-4: distinct active tickers
    max_week_assignment_pct: float = 0.15   # RISK-5: same-ISO-week escrow/NAV
    # RISK-8 assignment-stress liquidity reserve (§2.6 — replaces the old
    # blanket 40%-escrow cap): stress = 100% nearest expiry week + 50%
    # following week + 100% later ITM puts; then cash − stress ≥ reserve·NAV.
    stress_week1_pct: float = 1.0
    stress_week2_pct: float = 0.5
    stress_itm_pct: float = 1.0
    min_stress_reserve_pct: float = 0.15
    earnings_warning: bool = True           # RISK-7: warn, never block (§2.5)
    corr_threshold: float = 0.80            # RISK-9: warn, never block (§2.7)
    corr_lookback: int = 120

    @staticmethod
    def single_ticker() -> "RiskConfig":
        """Episode preset: one underlying uses the whole sleeve by design."""
        return RiskConfig(
            max_pct_per_underlying=1.0,
            max_underlyings=1,
            max_week_assignment_pct=1.0,
            min_stress_reserve_pct=0.0,
        )


@dataclass
class RiskDecision:
    """passed reflects HARD blocks only; warnings are human-review items
    (RISK-7/9) that ride along with a passing decision (§2.8)."""
    passed: bool
    flags: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def validate_open(
    quote,                      # Quote or None (None → trivially fine: WAIT)
    contracts: int,
    cash: float,
    shares: int,
    nav: float,
    open_put_escrow: float,     # Σ strike·100·contracts of already-open short puts
    event_in_window: bool,
    cfg: RiskConfig,
    *,
    valuation=None,             # WheelValuation or None (SPEC-009 VREQ-4)
    spot: float | None = None,
    cost_basis: float | None = None,
    val_cfg=None,
    # Book-level inputs (SPEC-004 §2 v2), fed by the daily assistant from the
    # synced positions. Defaults keep single-sleeve simulation callers
    # unchanged (event_in_window=False, no book context).
    n_underlyings: int = 0,          # distinct tickers already in the book
    is_new_underlying: bool = True,  # would this trade add a ticker?
    underlying_exposure: float = 0.0,   # RISK-3 §2.2: existing share market
                                        # value + existing put assignment
                                        # value on THIS ticker
    same_week_escrow: float = 0.0,   # existing CSP escrow expiring same week
    stressed_assignment: float | None = None,  # RISK-8 §2.6: precomputed
                                        # stress INCLUDING the proposed put
                                        # (BookState.stressed_assignment);
                                        # None -> conservative fallback
    earnings_info: dict | None = None,  # RISK-7 §2.5: {"date", "source",
                                        # "expiration"} for the warning text
    correlated: list | None = None,     # RISK-9 §2.7: [{"ticker", "corr",
                                        # "exposure_pct"}] above threshold
) -> RiskDecision:
    if quote is None:
        return RiskDecision(True)
    flags, warnings = [], []
    if valuation is not None:   # valuation gates (RL proposes, this disposes)
        from rlbot.config import ValuationGateConfig
        from rlbot.risk.valuation import call_gate_flags, put_gate_flags
        vcfg = val_cfg or ValuationGateConfig()
        if quote.cp == "P":
            flags += put_gate_flags(quote.strike, quote.mid, valuation,
                                    spot or 0.0, vcfg)
        else:
            flags += call_gate_flags(quote.strike, quote.mid, valuation,
                                     cost_basis, vcfg)
    notional = quote.strike * 100 * contracts

    if quote.cp == "P":
        if cash < notional:
            flags.append("RISK-1:cash_secured")
        # RISK-8 §2.6: assignment-stress liquidity reserve. With no
        # precomputed stress (single-sleeve callers) fall back to the most
        # conservative reading: everything open assigns at once.
        if nav > 0 and cfg.min_stress_reserve_pct > 0:
            stress = stressed_assignment if stressed_assignment is not None \
                else open_put_escrow + notional
            if (cash - stress) / nav < cfg.min_stress_reserve_pct:
                flags.append("RISK-8:stress_reserve")
        # RISK-5 §2.4 is put-side: it bounds how much stock one expiry Friday
        # can force onto the book (CC assignment is the wheel's intended exit).
        if nav > 0 and (same_week_escrow + notional) / nav > cfg.max_week_assignment_pct:
            flags.append("RISK-5:week_assignment_pct")
    else:
        if shares < 100 * contracts:
            flags.append("RISK-2:naked_call")

    # RISK-3 §2.2: potential exposure = shares + existing puts + this put.
    # Covered calls add no new underlying exposure (put-side only).
    if nav > 0 and quote.cp == "P" and \
            (underlying_exposure + notional) / nav > cfg.max_pct_per_underlying:
        flags.append("RISK-3:concentration")
    if n_underlyings + (1 if is_new_underlying else 0) > cfg.max_underlyings:
        flags.append("RISK-4:max_underlyings")

    # ---- human-review warnings (§2.8): surfaced, never blocking ----
    if cfg.earnings_warning and event_in_window:
        info = earnings_info or {}
        warnings.append(
            "RISK-7:earnings_review — EARNINGS RISK: earnings "
            f"{info.get('source', 'estimated')} for {info.get('date', '?')}; "
            f"proposed option expires {info.get('expiration', '?')} — the "
            "position may remain open through earnings. Human approval "
            "required (APPROVE / REJECT).")
    if correlated:
        pairs = ", ".join(f"{c['ticker']} corr {c['corr']:.2f} "
                          f"(exposure {c.get('exposure_pct', 0):.0%} NAV)"
                          for c in correlated)
        combined = sum(c.get("exposure_pct", 0) for c in correlated) \
            + (notional / nav if nav > 0 else 0)
        warnings.append(
            f"RISK-9:correlation_review — CORRELATION WARNING: proposed put "
            f"has {cfg.corr_lookback}d correlation ≥ {cfg.corr_threshold:.2f} "
            f"with {pairs}; proposed exposure "
            f"{(notional / nav if nav > 0 else 0):.0%} NAV; combined related "
            f"exposure ≈ {combined:.0%} NAV. Human approval required "
            "(APPROVE / REJECT).")

    return RiskDecision(passed=not flags, flags=flags, warnings=warnings)
