"""Counterfactual sweep engine (SPEC-005 §2).

The environment is exogenous, so at every decision epoch we simulate EVERY
legal cash action against the same historical path. Branches close at the
branch's own end (expiration settle, or +cadence for WAIT) with a Baseline-3
continuation value; targets are differential vs. the SPEC-001 references
(B1 wheel for cash windows, buy-and-hold for stock continuations).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from rlbot.benchmarks.policies import RULE_TABLE, AdaptiveRulePolicy, FixedWheelPolicy
from rlbot.options.selector import SelectorConfig, select_contract
from rlbot.risk.engine import RiskConfig, validate_open
from rlbot.simulator.environment import WheelEnv
from rlbot.simulator.portfolio import ExecutionConfig, Portfolio, nav as nav_of, open_short_option, settle_expiration
from rlbot.state.encoder import encode_q_state
from rlbot.state.enums import CashAction, PositionState, StockAction


@dataclass
class SweepTarget:
    ticker: str
    date: pd.Timestamp
    q_state: tuple
    action: int
    target: float          # branch differential + continuation
    branch_diff: float
    continuation: float
    end_state: str
    cluster_id: str


CONTINUATION_DAYS = 63     # trading days
BRANCH_GUARD_DAYS = 170    # calendar buffer so branch+continuation stay in-window


def run_cc_program(frame: pd.DataFrame, ps, exec_cfg: ExecutionConfig = ExecutionConfig()) -> pd.Series:
    """BXM-style covered-call overlay: always long 100 shares, write calls per
    B3 stock rules, cash-settle ITM expirations (share exposure constant).
    Used as the LONG_STOCK continuation reference program."""
    policy = AdaptiveRulePolicy()
    cash = 0.0
    strike = fill = 0.0
    expiry = None
    navs = {}
    last_vol = None
    for date, row in frame.iterrows():
        spot = float(row["close"])
        vol = float(row["vol_proxy"]) if pd.notna(row["vol_proxy"]) else last_vol
        last_vol = vol
        if expiry is not None and date >= expiry:
            cash -= max(spot - strike, 0.0) * 100      # cash settlement
            expiry = None
        q = encode_q_state(row["market_regime"], row["valuation_state"], row["vol_compensation"])
        if expiry is None and q is not None and vol is not None and vol > 0:
            action = policy.decide(PositionState.LONG_STOCK, q, row)
            if action != StockAction.WAIT:
                chain = ps.chain(date, spot, vol, "C")
                quote, _ = select_contract(action, chain, spot, vol, q[1])
                if quote is not None:
                    from rlbot.simulator.portfolio import sell_open_fill
                    fill = sell_open_fill(quote.mid, exec_cfg)
                    cash += fill * 100 - exec_cfg.commission_per_contract
                    strike, expiry = quote.strike, quote.expiration
        mark = ps.reprice("C", strike, expiry, date, spot, vol) if (expiry is not None and vol) else 0.0
        navs[date] = cash + spot * 100 - mark * 100
    return pd.Series(navs, name="cc_program_nav")


def simulate_branch(frame, ps, t0, action, valuation_state, cadence,
                    exec_cfg=ExecutionConfig(), selector_cfg=SelectorConfig(),
                    starting_cash=100_000.0):
    """One action branch from t0 to its own next epoch.
    Returns (end_date, end_state, branch_return) or None if unimplementable."""
    idx = frame.index
    i0 = idx.get_loc(t0)
    row = frame.iloc[i0]
    spot = float(row["close"])
    vol = float(row["vol_proxy"])
    port = Portfolio(cash=starting_cash)

    if action == CashAction.WAIT:
        i_end = min(i0 + cadence, len(idx) - 1)
        return idx[i_end], PositionState.CASH.value, 0.0

    chain = ps.chain(t0, spot, vol, "P")
    quote, _ = select_contract(action, chain, spot, vol, valuation_state, cfg=selector_cfg)
    risk = validate_open(quote, exec_cfg.contracts, port.cash, 0, port.cash, 0.0,
                         False, RiskConfig.single_ticker())
    if quote is None or not risk.passed:
        return None
    port = open_short_option(port, quote, exec_cfg)

    j = i0 + 1
    last_vol = vol
    while j < len(idx):
        d = idx[j]
        spot_j = float(frame["close"].iloc[j])
        v = frame["vol_proxy"].iloc[j]
        last_vol = float(v) if pd.notna(v) else last_vol
        if d >= port.option.expiration:
            port = settle_expiration(port, spot_j)
            end_nav = nav_of(port, spot_j, 0.0)
            return d, port.position_state.value, end_nav / starting_cash - 1.0
        j += 1
    # ran out of data before expiry: mark at last day
    spot_j = float(frame["close"].iloc[-1])
    mark = ps.reprice(port.option.cp, port.option.strike, port.option.expiration,
                      idx[-1], spot_j, last_vol)
    return idx[-1], port.position_state.value, nav_of(port, spot_j, mark) / starting_cash - 1.0


def _window_return(series: pd.Series, t0, t1) -> float:
    w = series.loc[t0:t1].dropna()
    return float(w.iloc[-1] / w.iloc[0] - 1.0) if len(w) >= 2 else 0.0


def _fwd_date(idx: pd.DatetimeIndex, t, n: int):
    i = idx.get_loc(t)
    return idx[min(i + n, len(idx) - 1)]


def sweep_ticker(ticker: str, frame: pd.DataFrame, ps, train_start, train_end,
                 cadence: int = 10) -> list:
    """All-action targets for every CASH-state epoch on the training window."""
    guard_end = pd.Timestamp(train_end) - pd.Timedelta(days=BRANCH_GUARD_DAYS)
    window = frame.loc[pd.Timestamp(train_start):pd.Timestamp(train_end)]
    env = WheelEnv(ticker, frame, ps)
    b1_nav = env.run(FixedWheelPolicy(), train_start, train_end).nav
    b3_nav = env.run(AdaptiveRulePolicy(), train_start, train_end).nav
    cc_nav = run_cc_program(window, ps)
    close = frame["close"]

    regime = window["market_regime"].fillna(-1).astype(int)
    episode_id = (regime != regime.shift()).cumsum().astype(int)

    targets = []
    dates = window.index
    for i in range(0, len(dates), cadence):
        t = dates[i]
        if t > guard_end:
            break
        row = window.loc[t]
        q = encode_q_state(row["market_regime"], row["valuation_state"], row["vol_compensation"])
        if q is None or pd.isna(row["vol_proxy"]) or row["vol_proxy"] <= 0:
            continue
        val_state = q[1]
        # SPEC-005 §3: cluster = (regime_episode, quarter) — shared across
        # co-moving tickers on purpose; n_eff counts market episodes, not rows.
        cluster = f"r{int(episode_id.loc[t])}-{t.year}Q{t.quarter}"
        for action in CashAction:
            branch = simulate_branch(frame, ps, t, action, val_state, cadence)
            if branch is None:
                continue
            end_date, end_state, branch_ret = branch
            branch_diff = branch_ret - _window_return(b1_nav, t, end_date)
            cont_end = _fwd_date(frame.index, end_date, CONTINUATION_DAYS)
            if end_state == PositionState.CASH.value:
                cont = _window_return(b3_nav, end_date, cont_end) \
                    - _window_return(b1_nav, end_date, cont_end)
            else:
                cont = _window_return(cc_nav, end_date, cont_end) \
                    - _window_return(close, end_date, cont_end)
            targets.append(SweepTarget(
                ticker=ticker, date=t, q_state=q, action=int(action),
                target=branch_diff + cont, branch_diff=branch_diff,
                continuation=cont, end_state=end_state, cluster_id=cluster,
            ))
    return targets
