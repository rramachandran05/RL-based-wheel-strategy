"""2026-08-30 review fixes: book-level RISK-4/5/8 enforcement and the
reportedDate-based earnings blackout."""
import pandas as pd
import pytest

from rlbot.config import RlbotConfig
from rlbot.options.premium_source import Quote
from rlbot.risk.book import build_book, earnings_in_window, next_earnings_estimate
from rlbot.risk.engine import RiskConfig, validate_open

EXP = pd.Timestamp("2026-09-18")


def _quote(strike=100.0):
    return Quote("P", strike, EXP, 21, 2.0, -0.2, 0.3)


def _pos(ticker, typ, strike, exp, contracts=1):
    return {"ticker": ticker, "type": typ, "strike": strike,
            "expiration": exp, "premium_fill": 1.0, "contracts": contracts}


def test_build_book_aggregates():
    book = build_book([
        _pos("TQQQ", "CSP", 50.0, "2026-09-18", contracts=5),
        _pos("NVDA", "CSP", 195.0, "2026-09-18"),
        _pos("MSFT", "CC", 500.0, "2026-09-04", contracts=4),
        {"ticker": "BAD", "type": "CSP", "strike": "junk",
         "expiration": "2026-09-18"},                    # skipped, not fatal
    ])
    assert book.n_open_positions == 3
    assert book.put_escrow == pytest.approx(50 * 100 * 5 + 195 * 100)
    assert book.same_week_count("2026-09-18") == 2
    assert book.same_week_count("2026-09-04") == 1
    assert book.same_week_count("2026-12-18") == 0
    # normalized views (2026-08-31)
    assert book.n_underlyings == 3 and book.has("tqqq")
    assert book.escrow_for("TQQQ") == pytest.approx(25_000)
    assert book.same_week_escrow("2026-09-18") == pytest.approx(44_500)
    assert book.same_week_escrow("2026-09-04") == 0.0        # CC: no escrow


def test_risk4_max_underlyings_enforced():
    # 12 names already held; a NEW ticker would be the 13th -> flag
    bad = validate_open(_quote(), 1, 1_000_000, 0, 1_000_000, 0.0, False,
                        RiskConfig(), n_underlyings=12, is_new_underlying=True)
    assert "RISK-4:max_underlyings" in bad.flags
    # same trade on a ticker ALREADY in the book adds no name -> fine
    ok = validate_open(_quote(), 1, 1_000_000, 0, 1_000_000, 0.0, False,
                       RiskConfig(), n_underlyings=12, is_new_underlying=False)
    assert "RISK-4:max_underlyings" not in ok.flags


def test_risk5_week_escrow_normalized():
    # NAV 100k, cap 15%: 10k new + 6k already expiring that week = 16% -> flag
    bad = validate_open(_quote(100.0), 1, 1_000_000, 0, 100_000, 0.0, False,
                        RiskConfig(), same_week_escrow=6_000)
    assert "RISK-5:week_assignment_pct" in bad.flags
    ok = validate_open(_quote(100.0), 1, 1_000_000, 0, 100_000, 0.0, False,
                       RiskConfig(), same_week_escrow=4_000)
    assert "RISK-5:week_assignment_pct" not in ok.flags


def test_risk3_counts_existing_ticker_escrow():
    # NAV 100k, cap 15%: 10k new + 8k already on this ticker = 18% -> flag
    bad = validate_open(_quote(100.0), 1, 1_000_000, 0, 100_000, 0.0, False,
                        RiskConfig(), underlying_escrow=8_000)
    assert "RISK-3:concentration" in bad.flags
    ok = validate_open(_quote(100.0), 1, 1_000_000, 0, 100_000, 0.0, False,
                       RiskConfig(), underlying_escrow=2_000)
    assert "RISK-3:concentration" not in ok.flags


def test_risk8_uses_aggregate_book_escrow():
    # 100 strike x 100 = 10k notional; escrow already 35k; NAV 100k; cap 40%
    bad = validate_open(_quote(100.0), 1, 100_000, 0, 100_000,
                        open_put_escrow=35_000, event_in_window=False,
                        cfg=RiskConfig())
    assert "RISK-8:assignment_at_once" in bad.flags


def test_defaults_keep_simulator_paths_unchanged():
    ok = validate_open(_quote(), 1, 1_000_000, 0, 1_000_000, 0.0, False,
                       RiskConfig.single_ticker())
    assert ok.passed


@pytest.fixture
def eps_cfg(tmp_path):
    cfg = RlbotConfig()
    cfg.data.base_path = tmp_path
    d = tmp_path / "external" / "eps"
    d.mkdir(parents=True)
    pd.DataFrame({
        "fiscalDateEnding": ["2026-06-30"],
        "reportedDate": ["2026-07-30"],
        "reportedEPS": [2.02],
    }).to_csv(d / "AAPL.csv", index=False)
    return cfg


def test_next_earnings_estimate_rolls_forward(eps_cfg):
    est = next_earnings_estimate("AAPL", eps_cfg, today="2026-08-30")
    assert est == pd.Timestamp("2026-10-29")          # 07-30 + 91d
    assert next_earnings_estimate("NOPE", eps_cfg) is None


def test_earnings_in_window_blackout(eps_cfg):
    # est 10-29 (+/-5d): a 11-20 expiry contains it; a 09-18 expiry does not
    assert earnings_in_window("AAPL", "2026-11-20", eps_cfg, today="2026-08-30")
    assert not earnings_in_window("AAPL", "2026-09-18", eps_cfg, today="2026-08-30")
    assert not earnings_in_window("NOPE", "2026-11-20", eps_cfg, today="2026-08-30")
