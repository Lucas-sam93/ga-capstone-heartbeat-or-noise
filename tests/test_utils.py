"""Tests for src.utils — RR interval filtering and feature computation."""

import numpy as np
import pytest

from src.utils import filter_rr_intervals, compute_rr_features


# ── filter_rr_intervals ──────────────────────────────────────────────


class TestFilterRrIntervals:
    def test_normal_range_passes_through(self):
        rr = np.array([600.0, 800.0, 1000.0])
        result = filter_rr_intervals(rr)
        np.testing.assert_array_equal(result, rr)

    def test_below_minimum_removed(self):
        rr = np.array([200.0, 600.0, 800.0])
        result = filter_rr_intervals(rr)
        np.testing.assert_array_equal(result, [600.0, 800.0])

    def test_above_maximum_removed(self):
        rr = np.array([600.0, 2500.0, 800.0])
        result = filter_rr_intervals(rr)
        np.testing.assert_array_equal(result, [600.0, 800.0])

    def test_boundary_values_inclusive(self):
        rr = np.array([300.0, 2000.0])
        result = filter_rr_intervals(rr)
        np.testing.assert_array_equal(result, [300.0, 2000.0])

    def test_all_removed_returns_empty(self):
        rr = np.array([100.0, 5000.0])
        result = filter_rr_intervals(rr)
        assert len(result) == 0

    def test_empty_input(self):
        rr = np.array([])
        result = filter_rr_intervals(rr)
        assert len(result) == 0


# ── compute_rr_features ──────────────────────────────────────────────


class TestComputeRrFeatures:
    def test_returns_all_eight_features(self):
        rr = np.array([800.0, 810.0, 790.0, 820.0, 780.0])
        result = compute_rr_features(rr)
        expected_keys = {
            "rmssd", "sdnn", "mean_rr", "pnn50",
            "hr_mean", "hr_std", "rr_skewness", "rr_kurtosis",
        }
        assert set(result.keys()) == expected_keys

    def test_known_values_rmssd(self):
        # RR = [800, 860, 800] -> diffs = [60, -60] -> squared = [3600, 3600]
        # RMSSD = sqrt(mean([3600, 3600])) = 60.0
        rr = np.array([800.0, 860.0, 800.0])
        result = compute_rr_features(rr)
        assert result["rmssd"] == pytest.approx(60.0)

    def test_known_values_pnn50(self):
        # Diffs: [60, -60] -> abs > 50: both True -> pNN50 = 2/2 = 1.0
        rr = np.array([800.0, 860.0, 800.0])
        result = compute_rr_features(rr)
        assert result["pnn50"] == pytest.approx(1.0)

    def test_pnn50_none_above_threshold(self):
        # Diffs: [10, -10] -> abs > 50: both False -> pNN50 = 0.0
        rr = np.array([800.0, 810.0, 800.0])
        result = compute_rr_features(rr)
        assert result["pnn50"] == pytest.approx(0.0)

    def test_mean_rr(self):
        rr = np.array([800.0, 1000.0])
        result = compute_rr_features(rr)
        assert result["mean_rr"] == pytest.approx(900.0)

    def test_hr_mean_inverse_of_mean_rr(self):
        rr = np.array([600.0, 600.0, 600.0])
        result = compute_rr_features(rr)
        # HR = 60000 / 600 = 100 bpm
        assert result["hr_mean"] == pytest.approx(100.0)

    def test_constant_rr_zero_variability(self):
        rr = np.array([800.0, 800.0, 800.0, 800.0])
        result = compute_rr_features(rr)
        assert result["rmssd"] == pytest.approx(0.0)
        assert result["sdnn"] == pytest.approx(0.0)
        assert result["hr_std"] == pytest.approx(0.0)
        assert result["pnn50"] == pytest.approx(0.0)

    def test_single_interval_edge_case(self):
        rr = np.array([800.0])
        result = compute_rr_features(rr)
        assert result["mean_rr"] == pytest.approx(800.0)
        assert result["rmssd"] == pytest.approx(0.0)
        assert result["sdnn"] == pytest.approx(0.0)
        assert result["pnn50"] == pytest.approx(0.0)
        assert result["rr_skewness"] == pytest.approx(0.0)
        assert result["rr_kurtosis"] == pytest.approx(0.0)

    def test_two_intervals_no_skew_kurtosis(self):
        rr = np.array([800.0, 900.0])
        result = compute_rr_features(rr)
        # Skewness/kurtosis require >2 values, should default to 0
        assert result["rr_skewness"] == pytest.approx(0.0)
        assert result["rr_kurtosis"] == pytest.approx(0.0)

    def test_all_features_are_finite(self):
        rr = np.array([750.0, 800.0, 850.0, 900.0, 780.0, 820.0, 810.0])
        result = compute_rr_features(rr)
        for key, val in result.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"
