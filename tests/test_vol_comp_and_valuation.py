"""SPEC-001 §3.1 vol_compensation and valuation_state golden tests."""
import pandas as pd
import pytest

from rlbot.features.valuation import classify_valuation_series
from rlbot.features.vol_comp import classify_vol_comp_series
from rlbot.state.enums import ValuationState, VolCompensation


def _vc(vrp, vix_pct):
    result = classify_vol_comp_series(pd.Series([vrp]), pd.Series([vix_pct]))
    return result.iloc[0]


@pytest.mark.parametrize("vrp,vix_pct,expected", [
    (-0.5, 0.50, VolCompensation.POOR),        # negative VRP
    (5.0, 0.10, VolCompensation.POOR),         # too-quiet VIX regime
    (1.0, 0.50, VolCompensation.NORMAL),
    (2.0, 0.59, VolCompensation.NORMAL),       # vrp ok, pct just under
    (2.0, 0.60, VolCompensation.ATTRACTIVE),   # both boundaries inclusive
    (10.0, 0.90, VolCompensation.ATTRACTIVE),
    (0.0, 0.90, VolCompensation.NORMAL),       # vrp=0 is not poor, not attractive
])
def test_vol_comp_golden(vrp, vix_pct, expected):
    assert _vc(vrp, vix_pct) == expected


def test_vol_comp_nan_is_na():
    assert pd.isna(_vc(float("nan"), 0.5))


def _val(fv_dist):
    return classify_valuation_series(pd.Series([fv_dist])).iloc[0]


@pytest.mark.parametrize("fv_dist,expected", [
    (-0.20, ValuationState.ATTRACTIVE),
    (-0.051, ValuationState.ATTRACTIVE),
    (-0.05, ValuationState.FAIR),   # band inclusive
    (0.0, ValuationState.FAIR),
    (0.05, ValuationState.FAIR),
    (0.051, ValuationState.EXPENSIVE),
    (0.30, ValuationState.EXPENSIVE),
])
def test_valuation_golden(fv_dist, expected):
    assert _val(fv_dist) == expected


def test_valuation_unknown_degrades_to_fair():
    """SPEC-001 §3.1: FV unknown -> FAIR (DATA-GAP-3 degradation path)."""
    assert _val(float("nan")) == ValuationState.FAIR
