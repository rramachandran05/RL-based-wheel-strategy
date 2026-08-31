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
    # Normalized caps (2026-08-31): position COUNTS mislead when sizes vary
    # 10x (one TQQQ put escrows ~$5K, one META put ~$62K), so every capital
    # rule reads escrow/NAV. The one remaining count is distinct underlyings
    # — an attention cap, which genuinely doesn't scale with dollars.
    max_pct_per_underlying: float = 0.15      # RISK-3: per-ticker escrow/NAV
    max_underlyings: int = 12                  # RISK-4: distinct tickers
    max_week_assignment_pct: float = 0.15      # RISK-5: same-expiry-week escrow/NAV
    max_assignment_at_once: float = 0.40       # RISK-8: total escrow/NAV
    earnings_blackout: bool = True             # RISK-7 (live era only; DATA-GAP-2)

    @staticmethod
    def single_ticker() -> "RiskConfig":
        """Episode preset: one underlying uses the whole sleeve by design."""
        return RiskConfig(
            max_pct_per_underlying=1.0,
            max_underlyings=1,
            max_week_assignment_pct=1.0,
            max_assignment_at_once=1.0,
        )


@dataclass
class RiskDecision:
    passed: bool
    flags: list = field(default_factory=list)


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
    # Book-level inputs (2026-08-30 review fix; normalized to escrow dollars
    # 2026-08-31), fed by the daily assistant from the synced positions.
    # Defaults keep single-sleeve simulation callers unchanged.
    n_underlyings: int = 0,          # distinct tickers already in the book
    is_new_underlying: bool = True,  # would this trade add a ticker?
    underlying_escrow: float = 0.0,  # existing CSP escrow on THIS ticker
    same_week_escrow: float = 0.0,   # existing CSP escrow expiring same week
) -> RiskDecision:
    if quote is None:
        return RiskDecision(True)
    flags = []
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
        if nav > 0 and (open_put_escrow + notional) / nav > cfg.max_assignment_at_once:
            flags.append("RISK-8:assignment_at_once")
        # RISK-5 is put-side: it bounds how much stock one expiry Friday can
        # force onto the book (CC assignment is the wheel's intended exit).
        if nav > 0 and (same_week_escrow + notional) / nav > cfg.max_week_assignment_pct:
            flags.append("RISK-5:week_assignment_pct")
    else:
        if shares < 100 * contracts:
            flags.append("RISK-2:naked_call")

    if nav > 0 and (underlying_escrow + (notional if quote.cp == "P" else 0)) \
            / nav > cfg.max_pct_per_underlying:
        flags.append("RISK-3:concentration")
    if n_underlyings + (1 if is_new_underlying else 0) > cfg.max_underlyings:
        flags.append("RISK-4:max_underlyings")
    if cfg.earnings_blackout and event_in_window:
        flags.append("RISK-7:earnings_blackout")

    return RiskDecision(passed=not flags, flags=flags)
