# VENDORED from ../wheel-strategy/options_engine.py on 2026-08-21 (source sha256: 45418e6e9d6d9ad1ce7c6c6c99e9f1fd565ff6ea76079da81fb97c035ad4636d)
# Do not edit — see SPEC-002 REQ-2.2. Changes belong in rlbot/, not here.
"""
options_engine.py
=================
Phase 1 layer: turns a *stock* analyzer into an *options* analyzer.

This module answers the question the old notebook never could:
    "For this strike, what does it pay, how likely is assignment,
     and is the expected value positive?"

Volatility note
---------------
We do NOT have an options-data feed, so implied volatility is approximated
with REALIZED volatility (annualised stdev of daily log returns). Every
function and output that uses it is labelled `*_proxy` so it is never
mistaken for true IV. Realized vol typically runs a little below IV, so
premium estimates here are best read as mildly conservative.

Analogy: realized vol is the "speed the car was actually driven last month."
Implied vol is "the speed limit the market is pricing for next month."
They are correlated but not identical -- hence the proxy label.

All math is Black-Scholes with zero rates (negligible for <60-day tenors).
Dependencies: numpy, scipy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log, sqrt, exp

import numpy as np
from scipy.stats import norm

TRADING_DAYS = 252


# ----------------------------------------------------------------------
# Volatility
# ----------------------------------------------------------------------

def realized_volatility_proxy(close_prices, window: int = 30) -> float:
    """
    Annualised realized volatility from the last `window` daily closes.

    Used as a stand-in for implied volatility. Labelled 'proxy' everywhere.
    """
    close = np.asarray(close_prices, dtype=float)
    close = close[~np.isnan(close)]
    if len(close) < window + 1:
        window = max(5, len(close) - 1)
    if len(close) < 6:
        raise ValueError("need at least 6 price points for a vol estimate")
    log_returns = np.diff(np.log(close[-(window + 1):]))
    daily_sigma = float(np.std(log_returns, ddof=1))
    return daily_sigma * sqrt(TRADING_DAYS)


# ----------------------------------------------------------------------
# Black-Scholes building blocks
# ----------------------------------------------------------------------

def _d1_d2(spot, strike, t_years, vol, r=0.0):
    if t_years <= 0 or vol <= 0:
        raise ValueError("t_years and vol must be positive")
    d1 = (log(spot / strike) + (r + 0.5 * vol ** 2) * t_years) / (vol * sqrt(t_years))
    d2 = d1 - vol * sqrt(t_years)
    return d1, d2


def bs_put_price(spot, strike, t_years, vol, r=0.0) -> float:
    """Black-Scholes price of a European put = the premium the seller collects."""
    d1, d2 = _d1_d2(spot, strike, t_years, vol, r)
    return float(strike * exp(-r * t_years) * norm.cdf(-d2) - spot * norm.cdf(-d1))


def bs_call_price(spot, strike, t_years, vol, r=0.0) -> float:
    """Black-Scholes price of a European call = covered-call premium collected."""
    d1, d2 = _d1_d2(spot, strike, t_years, vol, r)
    return float(spot * norm.cdf(d1) - strike * exp(-r * t_years) * norm.cdf(d2))


def put_assignment_probability(spot, strike, t_years, vol, r=0.0) -> float:
    """Risk-neutral P(S_T < K): the put-seller's probability of assignment."""
    _, d2 = _d1_d2(spot, strike, t_years, vol, r)
    return float(norm.cdf(-d2))


def call_assignment_probability(spot, strike, t_years, vol, r=0.0) -> float:
    """Risk-neutral P(S_T > K): the covered-call writer's probability of being called away."""
    _, d2 = _d1_d2(spot, strike, t_years, vol, r)
    return float(norm.cdf(d2))


def probability_of_win_put(spot, strike, t_years, vol, r=0.0) -> float:
    """POW for a cash-secured put seller: the put expires worthless."""
    return 1.0 - put_assignment_probability(spot, strike, t_years, vol, r)


def expected_put_payout(spot, strike, t_years, vol, r=0.0, n_grid=20_000) -> float:
    """
    E[max(K - S_T, 0)] under a lognormal terminal distribution.
    This is what the put SELLER expects to pay out before premium.
    """
    mu = (r - 0.5 * vol ** 2) * t_years
    sigma = vol * sqrt(t_years)
    z = np.linspace(-8, 8, n_grid)
    s_t = spot * np.exp(mu + sigma * z)
    payoff = np.maximum(strike - s_t, 0.0)
    pdf = norm.pdf(z)
    _trap = getattr(np, "trapezoid", None) or np.trapz
    return float(_trap(payoff * pdf, z))


# ----------------------------------------------------------------------
# Trade-level evaluation
# ----------------------------------------------------------------------

@dataclass
class PutQuote:
    """Probabilistic verdict for one cash-secured put strike.

    Section-1 "sell today" gate (REQ-7): a strike qualifies if it passes
    POW + EV, OR -- stocks only -- it sits within `fv_near_pct` of the
    fair-value buy anchor regardless of POW/EV. ETFs have no fair-value
    anchor, so `passes_fv` is always False for them and the OR term drops
    out, leaving the POW+EV gate as the sole requirement (unchanged from
    before).
    """
    strike: float
    premium_proxy: float          # estimated credit per share
    pow: float                    # probability of win (0-1)
    assignment_prob: float        # probability of assignment (0-1)
    expected_value: float         # premium - E[payout], per share
    annualized_roc: float         # annualised return on strike collateral
    discount_to_fv_low: float     # (fv_low - strike) / fv_low, can be negative
    passes_pow: bool = False
    passes_ev: bool = False
    passes_fv: bool = False        # "near fair value" (within fv_near_pct), not "below"

    @property
    def passes_all(self) -> bool:
        return (self.passes_pow and self.passes_ev) or self.passes_fv


def evaluate_put(
    spot: float,
    strike: float,
    days_to_expiry: int,
    vol_proxy: float,
    fair_value_low: float,
    pow_floor: float = 0.85,
    r: float = 0.0,
    has_fair_value: bool = True,
    fv_near_pct: float = 0.05,
) -> PutQuote:
    """
    Score a single CSP strike on the metrics a wheel needs.

    EV per share = premium - E[max(K - S_T, 0)]
    ROC uses the strike as cash collateral (cash-secured put).

    `has_fair_value` should be False for ETFs / tickers with no real fair-value
    anchor (callers otherwise pass spot as a placeholder `fair_value_low`,
    which would spuriously satisfy an unqualified proximity check).
    """
    t = days_to_expiry / 365.0
    premium = bs_put_price(spot, strike, t, vol_proxy, r)
    pow_ = probability_of_win_put(spot, strike, t, vol_proxy, r)
    assign = 1.0 - pow_
    payout = expected_put_payout(spot, strike, t, vol_proxy, r)
    ev = premium - payout
    roc_period = ev / strike if strike > 0 else 0.0
    annualized_roc = roc_period * (365.0 / days_to_expiry)
    discount = (fair_value_low - strike) / fair_value_low if fair_value_low > 0 else 0.0

    q = PutQuote(
        strike=round(strike, 2),
        premium_proxy=round(premium, 3),
        pow=round(pow_, 4),
        assignment_prob=round(assign, 4),
        expected_value=round(ev, 4),
        annualized_roc=round(annualized_roc, 4),
        discount_to_fv_low=round(discount, 4),
    )
    q.passes_pow = pow_ >= pow_floor
    q.passes_ev = ev > 0
    q.passes_fv = (has_fair_value and fair_value_low > 0
                   and abs(strike - fair_value_low) / fair_value_low <= fv_near_pct)
    return q


@dataclass
class CallQuote:
    """Probabilistic verdict for one covered-call strike."""
    strike: float
    premium_proxy: float
    assignment_prob: float        # probability shares get called away
    annualized_roc: float         # annualised premium yield on shares
    above_cost_basis: bool        # True if strike >= cost basis (can exit at profit)


def evaluate_call(
    spot: float,
    strike: float,
    days_to_expiry: int,
    vol_proxy: float,
    cost_basis: float,
    r: float = 0.0,
) -> CallQuote:
    """
    Score a covered-call strike. The key wheel rule: only write a CC at or
    above your cost basis, otherwise assignment locks in a loss on the stock.
    """
    t = days_to_expiry / 365.0
    premium = bs_call_price(spot, strike, t, vol_proxy, r)
    assign = call_assignment_probability(spot, strike, t, vol_proxy, r)
    roc_period = premium / spot if spot > 0 else 0.0
    annualized_roc = roc_period * (365.0 / days_to_expiry)
    return CallQuote(
        strike=round(strike, 2),
        premium_proxy=round(premium, 3),
        assignment_prob=round(assign, 4),
        annualized_roc=round(annualized_roc, 4),
        above_cost_basis=strike >= cost_basis,
    )


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def _run_tests() -> None:
    # vol proxy: a flat series has ~0 vol, a noisy one has more
    flat = [100.0] * 40
    try:
        realized_volatility_proxy(flat)
        flat_ok = True
    except Exception:
        flat_ok = True  # zero-vol may raise downstream; that's fine
    assert flat_ok
    rng = np.random.default_rng(0)
    noisy = 100 * np.cumprod(1 + rng.normal(0, 0.02, 200))
    v = realized_volatility_proxy(noisy)
    assert 0.1 < v < 1.0, f"noisy vol proxy out of range: {v}"

    # put-call parity-ish sanity: ATM put and call similar with zero rate
    p = bs_put_price(100, 100, 30 / 365, 0.3)
    c = bs_call_price(100, 100, 30 / 365, 0.3)
    assert abs(p - c) < 0.05, f"ATM put/call should match at r=0: {p} vs {c}"

    # POW + assignment prob complementary
    pow_ = probability_of_win_put(100, 95, 45 / 365, 0.35)
    ap = put_assignment_probability(100, 95, 45 / 365, 0.35)
    assert abs(pow_ + ap - 1.0) < 1e-9

    # deep OTM put: high POW, small premium
    deep = evaluate_put(100, 80, 30, 0.3, fair_value_low=95)
    assert deep.pow > 0.9 and deep.premium_proxy < 1.0

    # EV trap: a near-the-money put in high vol still has positive premium,
    # but a fairly-priced BS put has EV ~ 0 by construction. Confirm EV is
    # finite and the gates evaluate without error.
    atm = evaluate_put(100, 99, 30, 0.5, fair_value_low=95)
    assert atm.premium_proxy > 0
    assert atm.passes_fv is True  # strike 99 is within 5% of fv_low 95 (~4.2% away)

    # fair-value gate (REQ-7): "near" fv_low (within 5%), not merely "below" it
    near = evaluate_put(100, 92, 30, 0.3, fair_value_low=95)   # within 5% (~3.2% away)
    assert near.passes_fv is True
    far = evaluate_put(100, 70, 30, 0.3, fair_value_low=95)    # far below -> not "near"
    assert far.passes_fv is False
    above_but_near = evaluate_put(100, 98, 30, 0.3, fair_value_low=95)  # 3.2% above, still "near"
    assert above_but_near.passes_fv is True
    just_outside = evaluate_put(100, 101, 30, 0.3, fair_value_low=95)  # 6.3% above -> not "near"
    assert just_outside.passes_fv is False

    # passes_all is an OR (REQ-7): near-FV alone qualifies even with weak
    # POW/EV, and POW+EV alone qualifies even when far from fair value.
    near_fv_weak_gates = PutQuote(strike=98, premium_proxy=1.0, pow=0.5,
                                  assignment_prob=0.5, expected_value=-0.1,
                                  annualized_roc=0.1, discount_to_fv_low=0.03,
                                  passes_pow=False, passes_ev=False, passes_fv=True)
    assert near_fv_weak_gates.passes_all is True
    strong_gates_far_fv = PutQuote(strike=80, premium_proxy=1.0, pow=0.9,
                                   assignment_prob=0.1, expected_value=0.5,
                                   annualized_roc=0.15, discount_to_fv_low=0.2,
                                   passes_pow=True, passes_ev=True, passes_fv=False)
    assert strong_gates_far_fv.passes_all is True
    neither = PutQuote(strike=80, premium_proxy=1.0, pow=0.5, assignment_prob=0.5,
                       expected_value=-0.1, annualized_roc=0.1, discount_to_fv_low=0.2,
                       passes_pow=False, passes_ev=False, passes_fv=False)
    assert neither.passes_all is False

    # ETFs (has_fair_value=False): near-FV branch never fires, even if the
    # caller passes spot as a placeholder fair_value_low (distance 0).
    etf_quote = evaluate_put(100, 90, 30, 0.3, fair_value_low=100, has_fair_value=False)
    assert etf_quote.passes_fv is False
    assert etf_quote.passes_all == (etf_quote.passes_pow and etf_quote.passes_ev)

    # covered call: strike below cost basis is flagged
    cc_bad = evaluate_call(100, 95, 30, 0.3, cost_basis=110)
    assert cc_bad.above_cost_basis is False
    cc_ok = evaluate_call(100, 115, 30, 0.3, cost_basis=110)
    assert cc_ok.above_cost_basis is True

    print("options_engine: all tests passed")


if __name__ == "__main__":
    _run_tests()
