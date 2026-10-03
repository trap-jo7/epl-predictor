import pandas as pd

from src.data import synthetic_matches
from src.features import FeatureBuilder, build_features


def test_first_match_uses_priors_only():
    row = FeatureBuilder().row("A", "B", pd.Timestamp("2024-08-01"))
    assert row["elo_home"] == row["elo_away"] == 1500.0 and row["h_ppg5"] == row["a_ppg5"]


def test_row_does_not_change_until_update():
    fb, d = FeatureBuilder(), pd.Timestamp("2024-08-01")
    before = fb.row("A", "B", d)
    assert fb.row("A", "B", d) == before
    fb.update("A", "B", d, 3, 0, 6, 1)
    assert fb.row("A", "B", d + pd.Timedelta(days=7)) != before


def test_elo_is_zero_sum():
    fb = FeatureBuilder()
    fb.update("A", "B", pd.Timestamp("2024-08-01"), 2, 1, None, None)
    assert abs(fb.elo["A"] + fb.elo["B"] - 3000.0) < 1e-9


def test_no_leakage_when_future_matches_removed():
    m = synthetic_matches(seasons=2).head(300)
    X_full, _ = build_features(m)
    X_cut, _ = build_features(m.head(200))
    pd.testing.assert_frame_equal(X_full.head(200), X_cut)
