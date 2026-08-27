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
    max_pct_per_underlying: float = 0.15      # RISK-3
    max_positions: int = 9                     # RISK-4
    max_new_per_expiry_week: int = 3           # RISK-5
    max_assignment_at_once: float = 0.40       # RISK-8
    earnings_blackout: bool = True             # RISK-7 (live era only; DATA-GAP-2)

    @staticmethod
    def single_ticker() -> "RiskConfig":
        """Episode preset: one underlying uses the whole sleeve by design."""
        return RiskConfig(
            max_pct_per_underlying=1.0,
            max_positions=1,
            max_new_per_expiry_week=1,
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
    else:
        if shares < 100 * contracts:
            flags.append("RISK-2:naked_call")

    if nav > 0 and notional / nav > cfg.max_pct_per_underlying:
        flags.append("RISK-3:concentration")
    if cfg.earnings_blackout and event_in_window:
        flags.append("RISK-7:earnings_blackout")

    return RiskDecision(passed=not flags, flags=flags)
