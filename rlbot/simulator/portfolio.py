"""Portfolio state, execution fills, expiration settlement (SPEC-003 §3-5)."""
from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from rlbot.state.enums import PositionState


@dataclass(frozen=True)
class ExecutionConfig:
    slippage_pct_of_premium: float = 0.03   # synthetic-track stand-in for spread
    commission_per_contract: float = 0.65
    contracts: int = 1

    @property
    def shares_per_position(self) -> int:
        return self.contracts * 100


@dataclass(frozen=True)
class OpenOption:
    cp: str
    strike: float
    expiration: pd.Timestamp
    contracts: int
    premium_fill: float          # per-share fill actually received


@dataclass(frozen=True)
class Portfolio:
    cash: float
    shares: int = 0
    cost_basis: float | None = None
    option: OpenOption | None = None

    @property
    def position_state(self) -> PositionState:
        if self.option is not None:
            return PositionState.SHORT_PUT if self.option.cp == "P" else PositionState.COVERED_CALL
        return PositionState.LONG_STOCK if self.shares > 0 else PositionState.CASH


def nav(port: Portfolio, spot: float, option_mark: float) -> float:
    """cash + stock − short-option liability (SPEC-003 §3).
    option_mark: per-share model mid of the open option (0 if none)."""
    liability = 0.0
    if port.option is not None:
        liability = option_mark * 100 * port.option.contracts
    return port.cash + port.shares * spot - liability


def sell_open_fill(mid: float, cfg: ExecutionConfig) -> float:
    return mid * (1.0 - cfg.slippage_pct_of_premium)


def open_short_option(port: Portfolio, quote, cfg: ExecutionConfig) -> Portfolio:
    fill = sell_open_fill(quote.mid, cfg)
    proceeds = fill * 100 * cfg.contracts - cfg.commission_per_contract * cfg.contracts
    return replace(
        port,
        cash=port.cash + proceeds,
        option=OpenOption(quote.cp, quote.strike, quote.expiration, cfg.contracts, fill),
    )


def settle_expiration(port: Portfolio, close: float) -> Portfolio:
    """Expiration-settled assignment (SPEC-003 §5). Boundary close == strike
    does NOT assign."""
    opt = port.option
    assert opt is not None
    n = opt.contracts * 100
    if opt.cp == "P":
        if close < opt.strike:
            return replace(
                port,
                cash=port.cash - opt.strike * n,
                shares=port.shares + n,
                cost_basis=opt.strike - opt.premium_fill,
                option=None,
            )
        return replace(port, option=None)
    # covered call
    if close > opt.strike:
        return replace(port, cash=port.cash + opt.strike * n, shares=port.shares - n,
                       cost_basis=None, option=None)
    return replace(port, option=None)
