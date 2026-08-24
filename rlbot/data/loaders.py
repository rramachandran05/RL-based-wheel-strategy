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
        self._proxy = None
        if cfg.use_valuation_proxy:
            path = cfg.data.canonical_path / "valuation_proxy.parquet"
            if not path.exists():
                raise FileNotFoundError(
                    f"use_valuation_proxy=True but {path} missing; "
                    "run python -m rlbot.data.eps_proxy")
            self._proxy = pd.read_parquet(path)
        self._ticker_iv = None
        if getattr(cfg, "vol_comp_source", "market") == "ticker_iv":
            tiv_path = cfg.data.canonical_path / "ticker_iv.parquet"
            if not tiv_path.exists():
                raise FileNotFoundError(
                    f"vol_comp_source='ticker_iv' but {tiv_path} missing; "
                    "run python -m rlbot.features.ticker_iv")
            self._ticker_iv = pd.read_parquet(tiv_path)
        self._cache: dict = {}

    def frame(self, ticker: str) -> pd.DataFrame:
        if ticker not in self._cache:
            self._cache[ticker] = build_ticker_frame(
                ticker, self.tables["underlying"], self.tables["market"],
                self.tables["valuation"], self.cfg, valuation_proxy=self._proxy,
                ticker_iv=self._ticker_iv,
            )
        return self._cache[ticker]
