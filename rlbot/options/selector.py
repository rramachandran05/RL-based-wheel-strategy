"""Deterministic contract selector (SPEC-004 §1).

Filter by the action's delta band and the DTE window, score with the SPEC
formula, tie-break by premium then DTE-nearest-30. On the synthetic track,
VolPremium and SpreadCost are identically zero (no chain-level IV or quoted
spread information); friction is charged at execution instead (SPEC-003 §4).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rlbot.state.enums import (
    CALL_DELTA_BANDS,
    PUT_DELTA_BANDS,
    TARGET_DTE,
    CashAction,
    StockAction,
    ValuationState,
)
from rlbot.vendor.options_engine import expected_put_payout, put_assignment_probability, call_assignment_probability


@dataclass(frozen=True)
class SelectorConfig:
    w_premium_yield: float = 1.0
    w_vol_premium: float = 0.5
    w_spread_cost: float = 0.5
    w_downside: float = 1.0
    w_assignment: float = 1.0
    valuation_multiplier: dict = field(default_factory=lambda: {
        ValuationState.ATTRACTIVE: 0.25,
        ValuationState.FAIR: 1.0,
        ValuationState.EXPENSIVE: 2.0,
    })
    band_widen: float = 0.02      # single widening step (SPEC-004 §1.3)
    payout_grid: int = 2000       # expected_put_payout resolution


def _band_for(action) -> tuple:
    if isinstance(action, CashAction):
        return PUT_DELTA_BANDS[action]
    return CALL_DELTA_BANDS[action]


def _filter(quotes, band, dte_min, dte_max):
    lo, hi = band
    return [q for q in quotes if lo <= abs(q.delta) <= hi and dte_min <= q.dte <= dte_max]


def score_quote(q, spot: float, vol_proxy: float, valuation_state, cost_basis, cfg: SelectorConfig) -> float:
    ann = 365.0 / q.dte
    premium_yield = (q.mid / q.strike) * ann
    t = q.dte / 365.0
    if q.cp == "P":
        downside = (expected_put_payout(spot, q.strike, t, q.vol_used, n_grid=cfg.payout_grid)
                    / q.strike) * ann
        assign_prob = put_assignment_probability(spot, q.strike, t, q.vol_used)
    else:
        assign_prob = call_assignment_probability(spot, q.strike, t, q.vol_used)
        downside = assign_prob * max(0.0, (spot - q.strike) / spot) * ann
    val_mult = cfg.valuation_multiplier[ValuationState(int(valuation_state))]
    return (
        cfg.w_premium_yield * premium_yield
        - cfg.w_downside * downside
        - cfg.w_assignment * assign_prob * val_mult
    )


def select_contract(
    action,
    quotes: list,
    spot: float,
    vol_proxy: float,
    valuation_state,
    cost_basis: float | None = None,
    cfg: SelectorConfig = SelectorConfig(),
):
    """Returns (best_quote_or_None, n_candidates). None => tier unimplementable → WAIT."""
    if isinstance(action, CashAction) and action == CashAction.WAIT:
        return None, 0
    if isinstance(action, StockAction) and action == StockAction.WAIT:
        return None, 0

    dte_min, dte_max, dte_pref = TARGET_DTE
    band = _band_for(action)
    candidates = _filter(quotes, band, dte_min, dte_max)
    if not candidates:  # single widen step, then give up (SPEC-004 §1.3)
        widened = (max(band[0] - cfg.band_widen, 0.01), band[1] + cfg.band_widen)
        candidates = _filter(quotes, widened, dte_min, dte_max)
    if cost_basis is not None:
        candidates = [q for q in candidates if q.cp != "C" or q.strike >= cost_basis]
    if not candidates:
        return None, 0

    scored = sorted(
        candidates,
        key=lambda q: (
            -score_quote(q, spot, vol_proxy, valuation_state, cost_basis, cfg),
            -q.mid,
            abs(q.dte - dte_pref),
        ),
    )
    return scored[0], len(candidates)
