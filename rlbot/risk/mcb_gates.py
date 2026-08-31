"""MCB-based put gates (2026-08-30) — the consumer side of mcb-wheel's
contract, replacing the Wheel-FV gates as the live valuation constraint.

The tier constraint is HARD and not learnable (contract rule 1); the
reachability handling is the contract's *recommended* reading (rule 2),
implemented as default behavior behind honor_reachability.
"""
from __future__ import annotations

from rlbot.data.mcb_feed import TIERS, McbRow
from rlbot.state.enums import MarketRegime, VolCompensation

# Consumer regime posture (contract rule 1: "defensive -> at least
# ATTRACTIVE"): calm bull is non-defensive; everything else is defensive.
POSTURE_TIER = {
    int(MarketRegime.BULL_LOW_VOL): "FAIR",
    int(MarketRegime.BULL_HIGH_VOL): "ATTRACTIVE",
    int(MarketRegime.SIDEWAYS): "ATTRACTIVE",
    int(MarketRegime.BEAR_STRESS): "ATTRACTIVE",
}


def required_tier(row: McbRow, market_regime: int) -> str:
    """Deeper of the report's guardrail-resolved tier and our regime posture,
    limited to tiers the row actually has (missing deeper tier -> deepest
    available: stay conservative, never loosen)."""
    posture = POSTURE_TIER.get(int(market_regime), "ATTRACTIVE")
    tier = max((row.min_eligible_tier, posture), key=TIERS.index)
    while tier not in row.mcb and TIERS.index(tier) > 0:
        tier = TIERS[TIERS.index(tier) - 1]
    return tier if tier in row.mcb else TIERS[-1]


def mcb_ceiling(row: McbRow | None, market_regime: int) -> float | None:
    """The hard net-basis ceiling, or None (constraint absent)."""
    if row is None:
        return None
    return row.ceiling(required_tier(row, market_regime))


def tradeable(row: McbRow | None) -> tuple:
    """(ok, reason). Contract rule 1: never trade MONITOR_ONLY or HALT."""
    if row is None:
        return True, None
    if row.layer_a in ("MONITOR_ONLY", "HALT"):
        return False, f"MCB gate: layer A = {row.layer_a} — never trade"
    return True, None


def reachability_advice(row: McbRow | None, vol_comp: int,
                        honor: bool = True) -> str | None:
    """Contract rule 2 recommended reading, as a WAIT reason or None.
    UNREACHABLE -> skip the strike scan; PATIENCE -> elevated-IV only."""
    if not honor or row is None or row.reachability is None:
        return None
    if row.reachability == "UNREACHABLE":
        return ("MCB reachability UNREACHABLE: FAIR basis sits below a "
                "bear-correction price — strike scan skipped (advisory)")
    if row.reachability == "PATIENCE" and int(vol_comp) != int(VolCompensation.ATTRACTIVE):
        return ("MCB reachability PATIENCE: only elevated-IV setups — "
                "vol-comp not ATTRACTIVE today (advisory)")
    return None


def net_basis_flag(strike: float, premium: float, ceiling: float | None) -> str | None:
    """HARD constraint: strike − premium must sit at/below the ceiling."""
    if ceiling is None or ceiling <= 0:
        return None
    if strike - premium > ceiling + 1e-9:
        return "MCB-1:net_basis_above_ceiling"
    return None


def premium_required(strike: float, ceiling: float | None) -> float | None:
    """Minimum live premium making this strike acceptable."""
    if ceiling is None:
        return None
    return max(0.0, strike - ceiling)
