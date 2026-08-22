"""Configuration tree for the RL wheel-strategy project (SPEC-000 §7, SPEC-002).

All thresholds referenced by SPEC-001 §3.1 live here so the state-encoding
rules are tunable in one place without touching frozen enum identities.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Training universe v1 (SPEC-002 §4): SPY + 10 liquid mega-caps.
DEFAULT_TRAINING_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "V", "MA", "WMT", "UNH",
]
MARKET_TICKER = "SPY"


@dataclass(frozen=True)
class RegimeThresholds:
    """SPEC-001 §3.1 market_regime rule parameters."""
    vix_pct_high: float = 0.60      # bull high-vol split
    vix_pct_stress: float = 0.85    # stress trigger
    dd_bull: float = -0.10          # bull requires drawdown above this
    dd_stress: float = -0.15        # stress trigger regardless of SMA position


@dataclass(frozen=True)
class VolCompThresholds:
    """SPEC-001 §3.1 vol_compensation rule parameters (market-level proxy)."""
    vrp_attractive: float = 2.0     # vol points
    vix_pct_attractive: float = 0.60
    vix_pct_poor: float = 0.20


@dataclass(frozen=True)
class ValuationThresholds:
    """SPEC-001 §3.1 valuation_state band around fv_buy."""
    band: float = 0.05


@dataclass
class DataConfig:
    base_path: Path = field(default_factory=lambda: _default_base_path())
    spy_years: int = 20
    ticker_years: int = 15
    vix_percentile_window: int = 1260   # ~5 trading years
    vix_percentile_min_obs: int = 252
    realized_vol_market_window: int = 20
    realized_vol_ticker_window: int = 30
    # Sibling project whose fair_value_*.csv snapshots we ingest (SPEC-002 §3.4)
    fair_value_snapshot_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT.parent / "wheel-strategy" / "wheel_outputs"
    )

    @property
    def bars_path(self) -> Path:
        return self.base_path / "bars"

    @property
    def external_path(self) -> Path:
        return self.base_path / "external"

    @property
    def canonical_path(self) -> Path:
        return self.base_path / "canonical"


def _default_base_path() -> Path:
    env = os.getenv("RLBOT_DATA_PATH")
    return Path(env) if env else PROJECT_ROOT / "data_local"


@dataclass
class RlbotConfig:
    tickers: list = field(default_factory=lambda: list(DEFAULT_TRAINING_TICKERS))
    market_ticker: str = MARKET_TICKER
    # SPEC-007 Track B: fill the historically-dead valuation axis from the
    # AV EPS percentile proxy where no sheet-based FV snapshot exists
    use_valuation_proxy: bool = False
    data: DataConfig = field(default_factory=DataConfig)
    regime: RegimeThresholds = field(default_factory=RegimeThresholds)
    vol_comp: VolCompThresholds = field(default_factory=VolCompThresholds)
    valuation: ValuationThresholds = field(default_factory=ValuationThresholds)
