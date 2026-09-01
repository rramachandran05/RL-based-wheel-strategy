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


def test_risk3_potential_exposure_shares_plus_puts():
    # SPEC-004 §2.2 example scaled: NAV 100k, cap 15%; existing shares 4.5k +
    # existing puts 1k + proposed 10k = 15.5k -> 15.5% -> reject
    bad = validate_open(_quote(100.0), 1, 1_000_000, 0, 100_000, 0.0, False,
                        RiskConfig(), underlying_exposure=5_500)
    assert "RISK-3:concentration" in bad.flags
    ok = validate_open(_quote(100.0), 1, 1_000_000, 0, 100_000, 0.0, False,
                       RiskConfig(), underlying_exposure=4_000)
    assert "RISK-3:concentration" not in ok.flags


def test_risk8_assignment_stress_reserve():
    # SPEC-004 §2.6 example scaled to NAV 500k: stress 125k + proposed put
    # joins a later week OTM (adds 0 via precomputed value). cash 150k:
    # (150k - 125k)/500k = 5% < 15% -> reject; cash 250k -> 25% -> pass
    q = _quote(100.0)
    bad = validate_open(q, 1, 150_000, 0, 500_000, 0.0, False, RiskConfig(),
                        stressed_assignment=125_000)
    assert "RISK-8:stress_reserve" in bad.flags
    ok = validate_open(q, 1, 250_000, 0, 500_000, 0.0, False, RiskConfig(),
                       stressed_assignment=125_000)
    assert "RISK-8:stress_reserve" not in ok.flags


def test_risk9_correlation_is_warning_not_block():
    # SPEC-004 §2.7: corr >= 0.80 surfaces exposures for human review
    d = validate_open(_quote(100.0), 1, 1_000_000, 0, 1_000_000, 0.0, False,
                      RiskConfig(),
                      correlated=[{"ticker": "META", "corr": 0.84,
                                   "exposure_pct": 0.11},
                                  {"ticker": "GOOGL", "corr": 0.82,
                                   "exposure_pct": 0.09}])
    assert d.passed
    w = [x for x in d.warnings if "RISK-9:correlation_review" in x]
    assert w and "META" in w[0] and "0.84" in w[0] and "11%" in w[0]


def test_book_stress_and_exposure_helpers():
    # weeks: W1 = 9/18 (TQQQ 25k + NVDA 19.5k), no W2, so a later put would
    # be ITM-tested. Build a book with a later ITM put to exercise all legs.
    book = build_book([
        _pos("TQQQ", "CSP", 50.0, "2026-09-18", contracts=5),
        _pos("NVDA", "CSP", 195.0, "2026-09-25"),
        _pos("META", "CSP", 500.0, "2026-10-30"),   # later; ITM iff spot<=500
        _pos("MSFT", "CC", 500.0, "2026-09-04", contracts=4),
    ])
    spots = {"META": 480.0}                          # 500 >= 480 -> ITM
    stress = book.stressed_assignment(spots)
    # 100% x 25k (W1) + 50% x 19.5k (W2) + 100% x 50k (later ITM)
    assert stress == pytest.approx(25_000 + 9_750 + 50_000)
    spots_otm = {"META": 520.0}                      # 500 < 520 -> OTM
    assert book.stressed_assignment(spots_otm) == pytest.approx(34_750)
    # RISK-3 leg: MSFT CC implies 400 shares; exposure = 400*spot + 0 puts
    assert book.potential_exposure("MSFT", 500.0) == pytest.approx(200_000)
    assert book.potential_exposure("TQQQ", 70.0) == pytest.approx(25_000)


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


def test_brief_renders_review_warnings():
    # SPEC-004 §2.8: warnings render as ⚠ REVIEW plus the full text block
    from rlbot.assistant.daily import render_brief
    rec = {"ticker": "META", "date": "2026-08-28", "spot": 578.0,
           "action": "SELL_PUT", "state_names": ["BULL_LOW_VOL", "FAIR", "POOR"],
           "contract": {"strike": 500.0, "dte": 30, "delta": -0.2,
                        "model_premium": 5.0},
           "review_warnings": ["RISK-7:earnings_review — EARNINGS RISK: ..."]}
    text = render_brief("2026-08-28", [rec], [], [])
    assert "SELL_PUT ⚠ REVIEW" in text
    assert "Human-review warnings" in text
    assert "RISK-7:earnings_review" in text
