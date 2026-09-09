"""MCB-based put gates — the consumer side of mcb-wheel's contract,
replacing the Wheel-FV gates as the live valuation constraint.

v2 (SPEC-011 §2 rule 1, 2026-09-09): MCB is an aggressiveness pivot, not a
uniform rejection boundary. A trending, compounding stock can be safely
sold at conservative delta far above its comfortable acquisition price —
assignment there is a low-probability tail event, not the trade's purpose.
Bindingness now scales with the producer's per-ticker `delta_posture`:
CONSERVATIVE never blocks, HIGHER always blocks, MODERATE is resolved by
our own tier split (0.10-0.18Δ boundary, already frozen by SPEC-001). A
confirmed downtrend overrides all of this to conservative-or-wait — the
producer computes no trend, so that override is ours alone to apply.

The layer_a mask (never trade MONITOR_ONLY/HALT) and reachability handling
are unchanged from v1.
"""
from __future__ import annotations

from rlbot.data.mcb_feed import TIERS, McbRow
from rlbot.state.enums import CashAction, MarketRegime, VolCompensation

# SPEC-011 §2 rule 1: downtrend structures that override delta_posture to
# conservative-or-wait ("don't catch a falling knife"). Matches the vendored
# classify_structure labels (rlbot/vendor/technicals.py), already computed
# per-day in the underlying table (features/technicals_series.py) and
# logged-only until now (APPROACH.md §5).
DOWNTREND_STRUCTURES = {"Pullback in Uptrend", "Breakdown"}

# Our own tier split resolving the producer's ambiguous MODERATE case
# (SPEC-011 §2 rule 1): income tiers stay advisory, acquisition tiers hard.
ACQUISITION_TIERS = {CashAction.PUT_BALANCED, CashAction.PUT_AGGRESSIVE,
                     CashAction.PUT_VERY_AGGRESSIVE}

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
    """The v1-era ceiling (deeper-of tier). Retained for callers that don't
    yet pass an action/trend context; superseded live by mcb_binding()."""
    if row is None:
        return None
    return row.ceiling(required_tier(row, market_regime))


def mcb_binding(row: McbRow | None, action, downtrend: bool = False) -> tuple:
    """SPEC-011 §2 rule 1 (v2): (ceiling, hard) for the given put action.

    ceiling = row.wheel_entry — None means constraint absent (no v1
    fallback: an older-schema row with no wheel_entry gates nothing rather
    than guessing at a different anchor; mcb_feed.py warns on this).
    hard = whether that ceiling actually blocks a trade at this action:

      - downtrend=True             -> hard, always ("don't catch a falling
        knife" overrides delta_posture; ours to apply, the producer computes
        no trend).
      - delta_posture == HIGHER    -> hard, always (acquisition sought).
      - delta_posture == MODERATE  -> hard only for BALANCED-and-up actions
        (our own tier split resolves the producer's one ambiguous case).
      - delta_posture == CONSERVATIVE, or missing -> never hard (advisory).
    """
    if row is None:
        return None, False
    ceiling = row.wheel_entry
    if ceiling is None:
        return None, False           # constraint absent, never guessed
    if downtrend:
        return ceiling, True
    posture = row.delta_posture
    if posture == "HIGHER":
        return ceiling, True
    if posture == "MODERATE":
        return ceiling, action in ACQUISITION_TIERS
    return ceiling, False            # CONSERVATIVE or unknown -> advisory


def mcb_valuation_state(row: McbRow | None, market_regime: int):
    """SPEC-011 §2 rule 6 / SPEC-004 §1.2: the ValuationState fed to the
    selector's assignment-penalty multiplier — replacing the Google-Sheet
    FV anchor in that role. Combines MCB position (drop_needed: how far spot
    sits above wheel_entry) with drawdown severity judged against the
    ticker's own correction history (dd50/dd75/dd90) and the regime:

        drop_needed ≤ dd50          -> ATTRACTIVE  (entry within a typical correction)
        dd50 < drop_needed ≤ dd90   -> FAIR
        drop_needed > dd90          -> EXPENSIVE   (needs a beyond-90th-pct correction)
        relief notch: dd_now ≥ dd75 and regime != BEAR_STRESS -> one step
        less penalizing (a severe correction is already underway; not
        granted in a stressed market where it may continue)
        any input missing           -> FAIR (constraint absent -> neutral)

    Score-only: the frozen Q-state axis the policy conditions on is untouched,
    and backtests (no MCB history) never reach this function.
    """
    from rlbot.state.enums import ValuationState
    if row is None:
        return ValuationState.FAIR
    dn, d50, d90 = row.drop_needed, row.dd50, row.dd90
    if dn is None or d50 is None or d90 is None:
        return ValuationState.FAIR
    if dn <= d50:
        state = ValuationState.ATTRACTIVE
    elif dn <= d90:
        state = ValuationState.FAIR
    else:
        state = ValuationState.EXPENSIVE
    relief = (row.dd_now is not None and row.dd75 is not None
              and row.dd_now >= row.dd75
              and int(market_regime) != int(MarketRegime.BEAR_STRESS))
    if relief and state != ValuationState.ATTRACTIVE:
        state = ValuationState(int(state) - 1)
    return state


def is_downtrend(structure: str | None) -> bool:
    """SPEC-011 §2 rule 1 trend override input: classify_structure's label
    for today (vendored technicals.py; computed daily, previously
    logged-only per APPROACH.md §5). None (warmup/missing) -> not a
    confirmed downtrend, never guessed into one."""
    return structure in DOWNTREND_STRUCTURES


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
