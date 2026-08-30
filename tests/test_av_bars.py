"""AV daily-adjusted bar source: adjustment math on a split+dividend fixture."""
import pandas as pd
import pytest

from rlbot.data.av_bars import parse_daily_adjusted, to_caches


@pytest.fixture
def series():
    """3 days around a 4:1 split on day 3 (AV stamps the coefficient on the
    split day); day 2 pays a dividend so adj ratio != split ratio alone."""
    def row(o, h, l, c, adj, v, split="1.0"):
        return {"1. open": o, "2. high": h, "3. low": l, "4. close": c,
                "5. adjusted close": adj, "6. volume": v,
                "7. dividend amount": "0", "8. split coefficient": split}
    return {
        "2024-06-10": row("400", "410", "396", "404", "100.0", "1000000"),
        "2024-06-11": row("404", "412", "400", "408", "101.5", "1200000"),
        "2024-06-12": row("102", "104", "100", "103", "103.0", "4000000",
                          split="4.0"),
    }


def test_parse_and_adjust(series):
    df = parse_daily_adjusted(series)
    assert list(df.index) == [pd.Timestamp("2024-06-10"),
                              pd.Timestamp("2024-06-11"),
                              pd.Timestamp("2024-06-12")]
    bars, unadj = to_caches(df, years=10)

    # adjusted close taken directly; OHL scaled by the per-day ratio
    assert bars.loc["2024-06-10", "Close"] == pytest.approx(100.0)
    ratio_d1 = 100.0 / 404.0
    assert bars.loc["2024-06-10", "Open"] == pytest.approx(400 * ratio_d1)
    assert bars.loc["2024-06-10", "High"] == pytest.approx(410 * ratio_d1)
    # post-split day: raw == adjusted (ratio 1)
    assert bars.loc["2024-06-12", "Open"] == pytest.approx(102.0)

    # volume: pre-split days x4 (the day-3 coefficient), split day x1
    assert bars.loc["2024-06-10", "Volume"] == 4_000_000
    assert bars.loc["2024-06-11", "Volume"] == 4_800_000
    assert bars.loc["2024-06-12", "Volume"] == 4_000_000

    # unadjusted cache: raw close + ratio
    assert unadj.loc["2024-06-11", "close_unadj"] == pytest.approx(408.0)
    assert unadj.loc["2024-06-11", "adj_ratio"] == pytest.approx(101.5 / 408.0)


def test_years_truncation(series):
    df = parse_daily_adjusted(series)
    bars, unadj = to_caches(df, years=0)      # window excludes 2024
    assert bars.empty and unadj.empty
