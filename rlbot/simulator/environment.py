"""Event-driven wheel episode runner (SPEC-003 §2).

One implementation serves baselines, counterfactual sweeps, learned-policy
evaluation, and live replay (REQ-3.1). Rewards are attached post-hoc from
reference NAV series (SPEC-001 §4) so a policy can be evaluated against any
reference — including itself (REQ-3.4).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from rlbot.options.selector import SelectorConfig, select_contract
from rlbot.risk.engine import RiskConfig, RiskDecision, validate_open
from rlbot.simulator.portfolio import (
    ExecutionConfig,
    Portfolio,
    buy_to_close,
    nav as nav_of,
    open_short_option,
    settle_expiration,
)
from rlbot.state.encoder import encode_q_state
from rlbot.state.enums import (
    CallMgmtAction,
    CashAction,
    PositionState,
    PutMgmtAction,
    StockAction,
    legal_actions,
)
from rlbot.state.mgmt import CHALLENGE_DELTA, encode_mgmt_state, moneyness_bucket, premium_captured

RAW_FEATURE_COLS = [
    "close", "vol_proxy", "vix_close", "vix_pct_5y", "vrp", "fv_dist",
    "rsi14", "sma50", "sma200", "atr20", "drawdown",
]


@dataclass
class Decision:
    date: pd.Timestamp
    position_state: str
    q_state: tuple
    features_raw: dict
    available_actions: list
    chosen_action: int
    action_source: str
    contract: dict | None
    risk: RiskDecision
    n_candidates: int
    portfolio_before: dict
    # finalized when the next epoch (or episode end) is known:
    next_epoch_date: pd.Timestamp | None = None
    portfolio_after: dict | None = None
    next_q_state: tuple | None = None
    next_position_state: str | None = None
    terminal: bool = False
    reward: float | None = None
    reference_return: float | None = None
    mgmt_state: tuple | None = None      # SPEC-001A: set only on management decisions


@dataclass
class EpisodeResult:
    ticker: str
    start: pd.Timestamp
    end: pd.Timestamp
    nav: pd.Series
    decisions: list
    final_portfolio: Portfolio
    cycles: list = field(default_factory=list)   # (leg, open, close, strike, assigned)


def _port_dict(port: Portfolio, nav_val: float) -> dict:
    return {"cash": port.cash, "shares": port.shares,
            "cost_basis": port.cost_basis, "nav": nav_val}


def _step_down(action):
    """SPEC-004 §2.2: one tier toward WAIT on risk rejection."""
    cls = type(action)
    return cls(int(action) - 1) if int(action) > 0 else action


class WheelEnv:
    def __init__(
        self,
        ticker: str,
        frame: pd.DataFrame,
        premium_source,
        exec_cfg: ExecutionConfig = ExecutionConfig(),
        risk_cfg: RiskConfig | None = None,
        selector_cfg: SelectorConfig = SelectorConfig(),
        epoch_cadence: int = 5,
    ):
        self.ticker = ticker
        self.frame = frame
        self.ps = premium_source
        self.exec_cfg = exec_cfg
        self.risk_cfg = risk_cfg or RiskConfig.single_ticker()
        self.selector_cfg = selector_cfg
        self.cadence = epoch_cadence

    # ------------------------------------------------------------------
    def run(
        self,
        policy,
        start,
        end,
        starting_cash: float = 100_000.0,
        initial_portfolio: Portfolio | None = None,
        action_source: str = "policy",
        mgmt_policy=None,
    ) -> EpisodeResult:
        window = self.frame.loc[pd.Timestamp(start):pd.Timestamp(end)]
        port = initial_portfolio or Portfolio(cash=starting_cash)
        navs, decisions, cycles = {}, [], []
        last_decision_i = -(10 ** 6)
        last_regime = None
        last_vol = None
        pending: Decision | None = None
        # management trackers (SPEC-001A §4) — inert when mgmt_policy is None
        last_mgmt_i = -(10 ** 6)
        prev_m_bucket = prev_m_regime = None
        prev_delta_high = False
        expiry_week_fired = committed = False

        for i, (date, row) in enumerate(window.iterrows()):
            spot = float(row["close"])
            vol = float(row["vol_proxy"]) if pd.notna(row["vol_proxy"]) else last_vol
            last_vol = vol

            epoch = False
            # E2: expiration settlement
            if port.option is not None and date >= port.option.expiration:
                opt = port.option
                shares_before = port.shares
                port = settle_expiration(port, spot)
                assigned = (port.shares > shares_before) if opt.cp == "P" \
                    else (port.shares < shares_before)
                cycles.append({"leg": "CSP" if opt.cp == "P" else "CC",
                               "close_date": date, "strike": opt.strike,
                               "assigned": assigned})
                epoch = True
                prev_m_bucket = prev_m_regime = None
                prev_delta_high = False
                expiry_week_fired = committed = False

            q_state = encode_q_state(
                row["market_regime"], row["valuation_state"], row["vol_compensation"]
            )

            flat = port.option is None
            if flat and q_state is not None and vol is not None and vol > 0:
                if i - last_decision_i >= self.cadence:
                    epoch = True                                   # E1
                if last_regime is not None and pd.notna(row["market_regime"]) \
                        and int(row["market_regime"]) != last_regime:
                    epoch = True                                   # E3

            mark = self._mark(port, date, spot, vol)
            current_nav = nav_of(port, spot, mark)

            # ---- management epochs M1-M5 (SPEC-001A §4) ----
            if (mgmt_policy is not None and port.option is not None and not committed
                    and vol is not None and vol > 0 and q_state is not None):
                opt = port.option
                dte = max((opt.expiration - date.normalize()).days, 0)
                m_bucket = moneyness_bucket(opt.cp, spot, opt.strike)
                delta_high = abs(self.ps.delta_now(
                    opt.cp, opt.strike, opt.expiration, date, spot, vol)) >= CHALLENGE_DELTA
                regime_now = int(row["market_regime"])
                m_epoch = (
                    (i - last_mgmt_i >= self.cadence)                             # M1
                    or (prev_m_bucket is not None and m_bucket != prev_m_bucket)  # M2
                    or (delta_high and not prev_delta_high)                        # M3
                    or (prev_m_regime is not None and regime_now != prev_m_regime)  # M4
                    or (dte <= 7 and not expiry_week_fired)                        # M5
                )
                if dte <= 7:
                    expiry_week_fired = True
                prev_m_bucket, prev_m_regime, prev_delta_high = m_bucket, regime_now, delta_high
                if m_epoch and dte > 0:
                    if pending is not None:
                        self._finalize(pending, date, q_state, port, current_nav)
                        decisions.append(pending)
                        pending = None
                    port, pending, committed, extra_cycle = self._manage(
                        mgmt_policy, date, row, q_state, port, spot, vol,
                        current_nav, mark, dte, action_source,
                    )
                    if extra_cycle is not None:
                        cycles.append(extra_cycle)
                    last_mgmt_i = i
                    if port.option is not None:
                        prev_m_bucket = moneyness_bucket(port.option.cp, spot, port.option.strike)
                    else:
                        prev_m_bucket = prev_m_regime = None
                        prev_delta_high = False
                        expiry_week_fired = False
                    mark = self._mark(port, date, spot, vol)
                    current_nav = nav_of(port, spot, mark)
                    flat = port.option is None

            if epoch and flat and q_state is not None and vol is not None and vol > 0:
                if pending is not None:
                    self._finalize(pending, date, q_state, port, current_nav)
                    decisions.append(pending)
                    pending = None
                port, pending = self._decide(
                    policy, date, row, q_state, port, spot, vol, current_nav, action_source
                )
                last_decision_i = i
                last_regime = int(row["market_regime"])
                mark = self._mark(port, date, spot, vol)
                current_nav = nav_of(port, spot, mark)

            navs[date] = current_nav

        nav_series = pd.Series(navs, name="nav")
        if pending is not None:
            last_date = nav_series.index[-1]
            self._finalize(pending, last_date, None, port, nav_series.iloc[-1], terminal=True)
            decisions.append(pending)
        return EpisodeResult(self.ticker, window.index[0], window.index[-1],
                             nav_series, decisions, port, cycles)

    # ------------------------------------------------------------------
    def _mark(self, port: Portfolio, date, spot, vol) -> float:
        if port.option is None or vol is None:
            return 0.0
        return self.ps.reprice(port.option.cp, port.option.strike,
                               port.option.expiration, date, spot, vol)

    def _decide(self, policy, date, row, q_state, port, spot, vol, current_nav, source):
        pos = port.position_state
        action = policy.decide(pos, q_state, row)
        assert action in legal_actions(pos), f"illegal action {action} in {pos}"

        contract_dict, risk, n_cands = None, RiskDecision(True), 0
        quote = None
        if (isinstance(action, CashAction) and action != CashAction.WAIT) or (
            isinstance(action, StockAction) and action != StockAction.WAIT
        ):
            cp = "P" if isinstance(action, CashAction) else "C"
            chain = self.ps.chain(date, spot, vol, cp)
            attempt = action
            for _ in range(2):  # original tier, then one step-down (SPEC-004 §2.2)
                quote, n_cands = select_contract(
                    attempt, chain, spot, vol, row["valuation_state"]
                    if pd.notna(row["valuation_state"]) else 1,
                    cost_basis=port.cost_basis, cfg=self.selector_cfg,
                )
                risk = validate_open(
                    quote, self.exec_cfg.contracts, port.cash, port.shares,
                    current_nav, open_put_escrow=0.0, event_in_window=False,
                    cfg=self.risk_cfg,
                )
                if risk.passed and quote is not None:
                    break
                nxt = _step_down(attempt)
                if nxt == attempt or int(nxt) == 0:
                    quote = None
                    break
                attempt = nxt
            if quote is not None and risk.passed:
                port = open_short_option(port, quote, self.exec_cfg, tier=int(attempt))
                contract_dict = {
                    "type": "PUT" if quote.cp == "P" else "CALL",
                    "strike": quote.strike,
                    "expiration": str(quote.expiration.date()),
                    "dte": quote.dte, "delta": quote.delta,
                    "premium_fill": port.option.premium_fill,
                    "premium_source": "synthetic_bs",
                    "candidates_considered": n_cands,
                }
                action = attempt   # record the tier actually executed

        pending = Decision(
            date=date, position_state=pos.value, q_state=q_state,
            features_raw={k: (float(row[k]) if pd.notna(row[k]) else None)
                          for k in RAW_FEATURE_COLS if k in row.index},
            available_actions=[int(a) for a in legal_actions(pos)],
            chosen_action=int(action), action_source=source,
            contract=contract_dict, risk=risk, n_candidates=n_cands,
            portfolio_before=_port_dict(port, current_nav),
        )
        return port, pending

    def _manage(self, mgmt_policy, date, row, q_state, port, spot, vol,
                current_nav, mark, dte, source):
        """Execute one management decision (SPEC-001A §3). Returns
        (port, pending_decision, committed, extra_cycle_or_None)."""
        opt = port.option
        pos = port.position_state
        m_state = encode_mgmt_state(row["market_regime"], opt.cp, spot, opt.strike, dte)
        ctx = {"cp": opt.cp, "strike": opt.strike, "dte": dte, "mark": mark,
               "premium_fill": opt.premium_fill, "spot": spot,
               "premium_captured": premium_captured(mark, opt.premium_fill),
               "tier": opt.tier, "expiration": opt.expiration,
               "cash": port.cash, "shares": port.shares,
               "cost_basis": port.cost_basis, "vol": vol}
        action = mgmt_policy.decide_mgmt(pos, m_state, ctx, row)
        assert action in legal_actions(pos), f"illegal mgmt action {action} in {pos}"

        # NOTE: Put/Call mgmt enums share int values — dispatch by type, never
        # by cross-class membership (IntEnum equality is by value).
        if isinstance(action, PutMgmtAction):
            committed = action == PutMgmtAction.ACCEPT_ASSIGNMENT
            roll_delta = {PutMgmtAction.ROLL_SAME_RISK: 0,
                          PutMgmtAction.ROLL_LOWER_RISK: -1,
                          PutMgmtAction.ROLL_HIGHER_RISK: +1}.get(action)
            is_close = action == PutMgmtAction.CLOSE
        else:
            committed = action == CallMgmtAction.ALLOW_CALL_AWAY
            roll_delta = {CallMgmtAction.ROLL_OUT: 0,
                          CallMgmtAction.ROLL_UP_AND_OUT: -1}.get(action)
            is_close = action == CallMgmtAction.CLOSE
        contract_dict, extra_cycle = None, None
        risk = RiskDecision(True)
        n_cands = 0

        if is_close or roll_delta is not None:
            leg = "CSP" if opt.cp == "P" else "CC"
            port = buy_to_close(port, mark, self.exec_cfg)
            extra_cycle = {"leg": leg, "close_date": date, "strike": opt.strike,
                           "assigned": False, "early_close": True}
            if roll_delta is not None:
                base = opt.tier if opt.tier is not None else 3
                cls = CashAction if opt.cp == "P" else StockAction
                hi = 5 if opt.cp == "P" else 4
                new_tier = cls(max(1, min(hi, base + roll_delta)))
                chain = self.ps.chain(date, spot, vol, opt.cp)
                quote, n_cands = select_contract(
                    new_tier, chain, spot, vol,
                    row["valuation_state"] if pd.notna(row["valuation_state"]) else 1,
                    cost_basis=port.cost_basis, cfg=self.selector_cfg,
                )
                risk = validate_open(quote, self.exec_cfg.contracts, port.cash,
                                     port.shares, current_nav, 0.0, False, self.risk_cfg)
                if quote is not None and risk.passed:
                    port = open_short_option(port, quote, self.exec_cfg, tier=int(new_tier))
                    contract_dict = {
                        "type": "PUT" if quote.cp == "P" else "CALL",
                        "strike": quote.strike,
                        "expiration": str(quote.expiration.date()),
                        "dte": quote.dte, "delta": quote.delta,
                        "premium_fill": port.option.premium_fill,
                        "premium_source": "synthetic_bs",
                        "candidates_considered": n_cands,
                    }
                # rejected roll leg degrades to CLOSE (SPEC-001A §3)

        raw = {k: (float(row[k]) if pd.notna(row[k]) else None)
               for k in RAW_FEATURE_COLS if k in row.index}
        raw.update({"open_strike": opt.strike, "open_dte": dte, "mark": mark,
                    "premium_captured": ctx["premium_captured"]})
        pending = Decision(
            date=date, position_state=pos.value, q_state=q_state,
            features_raw=raw,
            available_actions=[int(a) for a in legal_actions(pos)],
            chosen_action=int(action), action_source=source,
            contract=contract_dict, risk=risk, n_candidates=n_cands,
            portfolio_before=_port_dict(port, current_nav),
            mgmt_state=m_state,
        )
        return port, pending, bool(committed), extra_cycle

    @staticmethod
    def _finalize(d: Decision, date, q_state, port: Portfolio, nav_val, terminal=False):
        d.next_epoch_date = date
        d.portfolio_after = _port_dict(port, nav_val)
        d.next_q_state = q_state
        d.next_position_state = port.position_state.value
        d.terminal = terminal


# ----------------------------------------------------------------------
def attach_rewards(result: EpisodeResult, cash_ref_nav: pd.Series, frame: pd.DataFrame):
    """Differential reward (SPEC-001 §4), computed once per finalized decision.

    cash_ref_nav: reference NAV series for CASH-state decisions (Baseline 1).
    Stock-state reference: buy-and-hold = the underlying close itself.
    """
    close = frame["close"]
    for d in result.decisions:
        if d.next_epoch_date is None:
            continue
        if d.mgmt_state is not None:
            # management decisions carry diff_v2 rewards only in sweep records
            continue
        t0, t1 = d.date, d.next_epoch_date
        pol = result.nav.loc[t0:t1]
        if len(pol) < 2:
            d.reward, d.reference_return = 0.0, 0.0
            continue
        pol_ret = pol.iloc[-1] / pol.iloc[0] - 1.0
        ref = cash_ref_nav if d.position_state == PositionState.CASH.value else close
        ref_w = ref.loc[t0:t1]
        ref_ret = ref_w.iloc[-1] / ref_w.iloc[0] - 1.0 if len(ref_w) >= 2 else 0.0
        d.reference_return = float(ref_ret)
        d.reward = float(pol_ret - ref_ret)
    return result
