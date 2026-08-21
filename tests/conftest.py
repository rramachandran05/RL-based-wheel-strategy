import numpy as np
import pandas as pd
import pytest


def make_ohlcv(n: int = 500, seed: int = 7, start: str = "2020-01-01") -> pd.DataFrame:
    """Seeded synthetic OHLCV random walk on business days."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    log_ret = rng.normal(0.0004, 0.015, n)
    close = 100.0 * np.exp(np.cumsum(log_ret))
    spread = np.abs(rng.normal(0.006, 0.003, n))
    high = close * (1 + spread)
    low = close * (1 - spread)
    open_ = np.concatenate([[100.0], close[:-1]]) * (1 + rng.normal(0, 0.002, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def make_vix(index: pd.DatetimeIndex, seed: int = 11) -> pd.Series:
    """Mean-reverting synthetic VIX aligned to the given dates."""
    rng = np.random.default_rng(seed)
    n = len(index)
    vix = np.empty(n)
    vix[0] = 18.0
    for i in range(1, n):
        vix[i] = max(9.0, vix[i - 1] + 0.15 * (18.0 - vix[i - 1]) + rng.normal(0, 1.2))
    return pd.Series(vix, index=index, name="vix_close")


@pytest.fixture
def ohlcv():
    return make_ohlcv()


@pytest.fixture
def vix_series(ohlcv):
    return make_vix(ohlcv.index)
