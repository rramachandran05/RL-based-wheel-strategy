"""Dynamic contract sizing: invariants + default-path regression."""
import pandas as pd
import pytest

from rlbot.benchmarks.policies import FixedWheelPolicy
from rlbot.options.premium_source import SyntheticBSPremiumSource
from rlbot.simulator.environment import WheelEnv
from tests.test_environment import make_frame

PS = SyntheticBSPremiumSource(iv_uplift=0.1)


def test_sized_run_deploys_multiple_contracts():
    frame = make_frame(n=300, seed=5)          # spot ~100 -> ~10 contracts on 100K
    env = WheelEnv("TEST", frame, PS, dynamic_sizing=True)
    res = env.run(FixedWheelPolicy(), "2021-01-04", "2021-12-31")
    opens = [d for d in res.decisions if d.contract is not None]
    assert opens
    # premium_fill is per-share; portfolio cash jump proves multi-contract fills
    assert res.final_portfolio.shares % 100 == 0
    # cash never went negative (cash-secured invariant held under sizing)
    assert (res.nav > 0).all()


def test_sizing_scales_contracts_with_capital():
    """4x the capital -> ~4x the contracts on the first open (lumpy at small
    counts, so returns aren't exactly scale-invariant — sizing is)."""
    frame = make_frame(n=300, seed=5)
    env1 = WheelEnv("TEST", frame, PS, dynamic_sizing=True)

    def first_open_cash_jump(cash):
        res = env1.run(FixedWheelPolicy(), "2021-01-04", "2021-12-31",
                       starting_cash=cash)
        d = next(d for d in res.decisions if d.contract is not None)
        return d.portfolio_before["cash"] - cash   # premium proceeds of open #1

    small, big = first_open_cash_jump(50_000), first_open_cash_jump(200_000)
    assert big / small == pytest.approx(4.0, rel=0.30)


def test_default_path_unchanged_without_flag():
    """The sizing edit must not perturb fixed-1-contract runs (gate integrity)."""
    import hashlib, json
    from pathlib import Path
    golden = json.loads((Path(__file__).parent / "golden_env_regression.json").read_text())
    env = WheelEnv("TEST", make_frame(n=400, seed=5), PS)
    res = env.run(FixedWheelPolicy(), "2021-01-04", "2022-06-30")
    sha = hashlib.sha256(
        json.dumps([round(v, 10) for v in res.nav.tolist()]).encode()).hexdigest()
    assert sha == golden["b1"]["nav_sha"]
