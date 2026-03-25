"""Tests for src.constants — ensure locked values remain stable."""

from src.constants import FEATURE_COLS, RR_MIN_MS, RR_MAX_MS, MIN_RR_COUNT


class TestLockedConstants:
    def test_feature_cols_count(self):
        assert len(FEATURE_COLS) == 8

    def test_feature_cols_names(self):
        expected = [
            "rmssd", "sdnn", "mean_rr", "pnn50",
            "hr_mean", "hr_std", "rr_skewness", "rr_kurtosis",
        ]
        assert FEATURE_COLS == expected

    def test_rr_bounds(self):
        assert RR_MIN_MS == 300
        assert RR_MAX_MS == 2000

    def test_min_rr_count(self):
        assert MIN_RR_COUNT == 10
