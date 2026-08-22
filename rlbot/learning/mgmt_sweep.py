"""Management counterfactual sweep (SPEC-007 §2.2, SPEC-001A §5).

At every management epoch of a B3-openings + HOLD-management rollout, branch
each candidate action and measure NAV at the ORIGINAL contract's resolution
date T — a common horizon for all branches. target(a) = NAV%_T(a) − NAV%_T(HOLD),
so HOLD ≡ 0 by construction (REQ-7.2 is enforced, not just tested). Rolled
branches (whose new expiration is beyond T) are marked to model at T.
ROLL_HIGHER_RISK is excluded (SPEC-001A §3); ACCEPT/ALLOW ≡ HOLD in NAV terms.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from rlbot.benchmarks.policies import AdaptiveRulePolicy, HoldMgmtPolicy
from rlbot.options.selector import SelectorConfig, select_contract
from rlbot.risk.engine import RiskConfig, validate_open
from rlbot.simulator.environment import WheelEnv
from rlbot.simulator.portfolio import (
    ExecutionConfig,
    OpenOption,
    Portfolio,
    buy_to_close,
    nav as nav_of,
    open_short_option,
    settle_expiration,
)
from rlbot.state.enums import CallMgmtAction, CashAction, PositionState, PutMgmtAction, StockAction

PUT_SWEEP_ACTIONS = [PutMgmtAction.CLOSE, PutMgmtAction.ROLL_SAME_RISK,
                     PutMgmtAction.ROLL_LOWER_RISK]
CALL_SWEEP_ACTIONS = [CallMgmtAction.CLOSE, CallMgmtAction.ROLL_OUT,
                      CallMgmtAction.ROLL_UP_AND_OUT]


@dataclass
class MgmtEpoch:
    date: pd.Timestamp
    pos_state: str
    m_state: tuple
    ctx: dict


@dataclass
class MgmtSweepTarget:
    ticker: str
    date: pd.Timestamp
    pos_state: str
    m_state: tuple
    action: int
    target: float
    cluster_id: str


class CapturingHoldPolicy(HoldMgmtPolicy):
    """HOLD everywhere, recording every management epoch's full context."""

    def __init__(self):
        self.epochs: list = []

    def decide_mgmt(self, position_state, m_state, ctx, row):
        if m_state is not None:
            self.epochs.append(MgmtEpoch(None, position_state.value, m_state, dict(ctx)))
        return super().decide_mgmt(position_state, m_state, ctx, row)


def _port_from_ctx(ctx: dict) -> Portfolio:
    opt = OpenOption(ctx["cp"], ctx["strike"], ctx["expiration"], 1,
                     ctx["premium_fill"], ctx["tier"])
    return Portfolio(cash=ctx["cash"], shares=ctx["shares"],
                     cost_basis=ctx["cost_basis"], option=opt)


def apply_mgmt_action(port: Portfolio, action, date, spot, vol, valuation_state,
                      ps, exec_cfg: ExecutionConfig, selector_cfg: SelectorConfig,
                      risk_cfg: RiskConfig) -> Portfolio:
    """Standalone twin of WheelEnv._manage's execution core (sweep use)."""
    opt = port.option
    mark = ps.reprice(opt.cp, opt.strike, opt.expiration, date, spot, vol)
    if isinstance(action, PutMgmtAction):
        roll_delta = {PutMgmtAction.ROLL_SAME_RISK: 0,
                      PutMgmtAction.ROLL_LOWER_RISK: -1,
                      PutMgmtAction.ROLL_HIGHER_RISK: +1}.get(action)
        is_close = action == PutMgmtAction.CLOSE
    else:
        roll_delta = {CallMgmtAction.ROLL_OUT: 0,
                      CallMgmtAction.ROLL_UP_AND_OUT: -1}.get(action)
        is_close = action == CallMgmtAction.CLOSE
    if not is_close and roll_delta is None:
        return port                                    # HOLD / commit
    port = buy_to_close(port, mark, exec_cfg)
    if roll_delta is not None:
        base = opt.tier if opt.tier is not None else 3
        cls = CashAction if opt.cp == "P" else StockAction
        hi = 5 if opt.cp == "P" else 4
        new_tier = cls(max(1, min(hi, base + roll_delta)))
        chain = ps.chain(date, spot, vol, opt.cp)
        quote, _ = select_contract(new_tier, chain, spot, vol, valuation_state,
                                   cost_basis=port.cost_basis, cfg=selector_cfg)
        nav0 = nav_of(port, spot, 0.0)
        risk = validate_open(quote, exec_cfg.contracts, port.cash, port.shares,
                             nav0, 0.0, False, risk_cfg)
        if quote is not None and risk.passed:
            port = open_short_option(port, quote, exec_cfg, tier=int(new_tier))
    return port


def simulate_mgmt_branch(frame: pd.DataFrame, ps, t, ctx: dict, action,
                         exec_cfg=ExecutionConfig(), selector_cfg=SelectorConfig(),
                         risk_cfg=None) -> float | None:
    """NAV% of one branch from t to T (original contract's resolution)."""
    risk_cfg = risk_cfg or RiskConfig.single_ticker()
    idx = frame.index
    i0 = idx.get_loc(t)
    row0 = frame.iloc[i0]
    spot0, vol0 = float(row0["close"]), ctx["vol"]
    val_state = int(row0["valuation_state"]) if pd.notna(row0["valuation_state"]) else 1

    port = _port_from_ctx(ctx)
    nav0 = nav_of(port, spot0, ps.reprice(port.option.cp, port.option.strike,
                                          port.option.expiration, t, spot0, vol0))
    if nav0 <= 0:
        return None
    orig_exp = ctx["expiration"]
    port = apply_mgmt_action(port, action, t, spot0, vol0, val_state,
                             ps, exec_cfg, selector_cfg, risk_cfg)

    last_vol = vol0
    j = i0 + 1
    while j < len(idx):
        d = idx[j]
        spot = float(frame["close"].iloc[j])
        v = frame["vol_proxy"].iloc[j]
        last_vol = float(v) if pd.notna(v) else last_vol
        if port.option is not None and d >= port.option.expiration:
            port = settle_expiration(port, spot)
        if d >= orig_exp:                       # common horizon T
            mark = 0.0
            if port.option is not None:
                mark = ps.reprice(port.option.cp, port.option.strike,
                                  port.option.expiration, d, spot, last_vol)
            return nav_of(port, spot, mark) / nav0 - 1.0
        j += 1
    return None                                  # ran out of data before T


def sweep_mgmt_ticker(ticker: str, frame: pd.DataFrame, ps, train_start, train_end) -> list:
    """All-action management targets from a B3+HOLD rollout over the window."""
    capture = CapturingHoldPolicy()
    env = WheelEnv(ticker, frame, ps)
    res = env.run(AdaptiveRulePolicy(), train_start, train_end,
                  action_source="sweep", mgmt_policy=capture)
    decisions = [d for d in res.decisions if d.mgmt_state is not None]
    epochs = capture.epochs
    assert len(decisions) == len(epochs), "epoch capture out of sync"

    window = frame.loc[pd.Timestamp(train_start):pd.Timestamp(train_end)]
    regime = window["market_regime"].fillna(-1).astype(int)
    episode_id = (regime != regime.shift()).cumsum().astype(int)

    guard_end = pd.Timestamp(train_end) - pd.Timedelta(days=60)
    targets = []
    for d, ep in zip(decisions, epochs):
        t = d.date
        if t > guard_end or t not in window.index:
            continue
        cluster = f"r{int(episode_id.loc[t])}-{t.year}Q{t.quarter}"
        is_put = ep.pos_state == PositionState.SHORT_PUT.value
        actions = PUT_SWEEP_ACTIONS if is_put else CALL_SWEEP_ACTIONS
        hold_a = PutMgmtAction.HOLD if is_put else CallMgmtAction.HOLD
        hold_ret = simulate_mgmt_branch(frame, ps, t, ep.ctx, hold_a)
        if hold_ret is None:
            continue
        # HOLD is the diff_v2 reference: its own target is identically 0
        targets.append(MgmtSweepTarget(ticker, t, ep.pos_state, ep.m_state,
                                       int(hold_a), 0.0, cluster))
        for a in actions:
            ret = simulate_mgmt_branch(frame, ps, t, ep.ctx, a)
            if ret is None:
                continue
            targets.append(MgmtSweepTarget(ticker, t, ep.pos_state, ep.m_state,
                                           int(a), ret - hold_ret, cluster))
    return targets
