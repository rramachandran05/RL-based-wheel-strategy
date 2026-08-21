"""REQ-2.3 / SPEC-002 AC-3: vectorized structure/momentum classifiers agree
exactly with the vendored scalar versions on >= 200 sampled rows."""
import numpy as np

from rlbot.features.technicals_series import build_feature_frame
from rlbot.vendor.technicals import classify_momentum, classify_structure

N_SAMPLES = 200


def test_golden_agreement_with_vendored_classifiers(ohlcv):
    feat = build_feature_frame(ohlcv)
    valid = feat.dropna(subset=["sma200", "atr20", "rsi14", "adx14", "structure"])
    assert len(valid) >= N_SAMPLES, "fixture too short for the golden sample"

    rng = np.random.default_rng(3)
    sampled = valid.iloc[sorted(rng.choice(len(valid), N_SAMPLES, replace=False))]

    for date, row in sampled.iterrows():
        ctx = {
            "price": row["close"], "sma50": row["sma50"], "sma200": row["sma200"],
            "atr20": row["atr20"], "rsi14": row["rsi14"], "adx14": row["adx14"],
            "di_plus": row["di_plus"], "di_minus": row["di_minus"],
        }
        assert row["structure"] == classify_structure(ctx), f"structure mismatch @ {date}"
        assert row["momentum"] == classify_momentum(ctx), f"momentum mismatch @ {date}"


def test_warmup_rows_have_na_classifications(ohlcv):
    feat = build_feature_frame(ohlcv)
    warmup = feat[feat["sma200"].isna()]
    assert warmup["structure"].isna().all()
    assert warmup["trend_bucket"].isna().all()


def test_bucket_columns_are_nullable_int(ohlcv):
    feat = build_feature_frame(ohlcv)
    assert str(feat["trend_bucket"].dtype) == "Int8"
    assert str(feat["momentum_bucket"].dtype) == "Int8"
    ok = feat["trend_bucket"].dropna()
    assert ok.isin([0, 1, 2, 3, 4]).all()
