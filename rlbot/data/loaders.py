"""Load canonical tables and assemble per-ticker decision frames."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from rlbot.config import RlbotConfig
from rlbot.state.encoder import build_ticker_frame


def load_canonical(cfg: RlbotConfig) -> dict:
    p = cfg.data.canonical_path
    return {
        "market": pd.read_parquet(p / "market.parquet"),
        "underlying": pd.read_parquet(p / "underlying.parquet"),
        "valuation": pd.read_parquet(p / "valuation.parquet"),
    }


class FrameStore:
    """Caches assembled per-ticker frames for a config."""

    def __init__(self, cfg: RlbotConfig):
        self.cfg = cfg
        self.tables = load_canonical(cfg)
        self._cache: dict = {}

    def frame(self, ticker: str) -> pd.DataFrame:
        if ticker not in self._cache:
            self._cache[ticker] = build_ticker_frame(
                ticker, self.tables["underlying"], self.tables["market"],
                self.tables["valuation"], self.cfg,
            )
        return self._cache[ticker]
