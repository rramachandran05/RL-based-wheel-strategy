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


LOW_YIELD_ROC = 0.07    # SPEC-011 §6.2: LOW YIELD flag threshold, /yr —
                        # decision support, never a verdict or blocker
                        # (user decision 2026-09-01)


def opportunity_scan(chain: list, ceiling: float,
                     low_yield_roc: float = LOW_YIELD_ROC,
                     sel_cfg=None) -> dict | None:
    """SPEC-011 §6: below-band advisory scan, run only after the normal
    tier scan found no MCB-compliant candidate.

    Delta describes risk; economics inform the human: among MCB-compliant
    puts in the DTE window (no delta floor), pick the best annualized return
    on escrow ROC = premium/(strike−premium) × 365/DTE and ALWAYS surface it
    — the system renders the economics and makes no accept/reject judgment.
    ROC below low_yield_roc carries a LOW YIELD flag; liquidity is reported,
    not filtered. Opportunity cost is the user's call.

    Returns None when NO compliant strike exists in the chain window
    (geometrically unreachable); else the §6.3 advisory dict. Advisory only:
    the result is never executable and never enters the decision record.
    """
    from rlbot.options.selector import TARGET_DTE, SelectorConfig, _liquid
    dte_min, dte_max, _ = TARGET_DTE
    sel_cfg = sel_cfg or SelectorConfig()
    best = None
    for q in chain:
        if q.cp != "P" or not (dte_min <= q.dte <= dte_max):
            continue
        net_basis = q.strike - q.mid
        if net_basis > ceiling + 1e-9 or net_basis <= 0:
            continue
        roc_ann = q.mid / net_basis * 365.0 / q.dte
        if best is None or roc_ann > best["roc_ann"]:
            liquid_ok = _liquid(q, sel_cfg)
            best = {
                "strike": q.strike,
                "premium": round(q.mid, 2),
                "net_basis": round(net_basis, 2),
                "delta": round(abs(q.delta), 4),
                "dte": q.dte,
                "roc_ann": round(roc_ann, 4),
                "oi": q.oi,
                "spread_pct": round(q.spread_pct, 4)
                if getattr(q, "spread_pct", None) is not None else None,
                "liquidity": ("n/a (model quote)" if q.oi is None
                              else ("acceptable" if liquid_ok else "poor")),
                "mcb_headroom": round(ceiling - net_basis, 2),
                "low_yield": bool(roc_ann < low_yield_roc),
            }
    if best is not None:
        flags = []
        if best["low_yield"]:
            flags.append(f"LOW YIELD (< {low_yield_roc:.0%}/yr)")
        if best["liquidity"] == "poor":
            flags.append("liquidity poor")
        best["flags"] = flags
    return best
