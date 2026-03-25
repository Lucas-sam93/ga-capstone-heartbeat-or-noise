"""Tests for app.pipeline — CSV parsing and inference pipeline."""

import io

import numpy as np
import pandas as pd
import pytest

from app.pipeline import parse_apple_health_export, process_and_predict


# ── parse_apple_health_export ────────────────────────────────────────


class TestParseAppleHealthExport:
    def _make_csv(self, rows, header="type,startDate,value"):
        lines = [header] + rows
        return "\n".join(lines).encode("utf-8")

    def test_extracts_heart_rate_rows(self):
        csv = self._make_csv([
            "HKQuantityTypeIdentifierHeartRate,2025-06-01 10:00:00 +0800,72",
            "HKQuantityTypeIdentifierStepCount,2025-06-01 10:00:00 +0800,100",
            "HKQuantityTypeIdentifierHeartRate,2025-06-01 10:05:00 +0800,75",
        ])
        df = parse_apple_health_export(csv)
        assert len(df) == 2
        assert list(df.columns) == ["startDate", "Value"]

    def test_renames_lowercase_value(self):
        csv = self._make_csv([
            "HKQuantityTypeIdentifierHeartRate,2025-06-01 10:00:00 +0800,72",
        ])
        df = parse_apple_health_export(csv)
        assert "Value" in df.columns

    def test_missing_type_column_raises(self):
        csv = b"startDate,value\n2025-06-01,72"
        with pytest.raises(ValueError, match="Missing 'type' column"):
            parse_apple_health_export(csv)

    def test_no_heart_rate_rows_raises(self):
        csv = self._make_csv([
            "HKQuantityTypeIdentifierStepCount,2025-06-01 10:00:00 +0800,100",
        ])
        with pytest.raises(ValueError, match="No heart rate records"):
            parse_apple_health_export(csv)

    def test_drops_non_numeric_values(self):
        csv = self._make_csv([
            "HKQuantityTypeIdentifierHeartRate,2025-06-01 10:00:00 +0800,72",
            "HKQuantityTypeIdentifierHeartRate,2025-06-01 10:05:00 +0800,INVALID",
        ])
        df = parse_apple_health_export(csv)
        assert len(df) == 1


# ── process_and_predict ──────────────────────────────────────────────


class TestProcessAndPredict:
    def _make_hr_dataframe(self, n_records=500, days=30):
        """Generate a realistic heart rate DataFrame spanning `days` days."""
        rng = np.random.default_rng(42)
        start = pd.Timestamp("2025-06-01")
        timestamps = pd.date_range(start, periods=n_records, freq="3min")
        hr_values = rng.normal(loc=72, scale=8, size=n_records).clip(40, 180)
        return pd.DataFrame({"startDate": timestamps, "Value": hr_values})

    def test_returns_expected_keys(self):
        df = self._make_hr_dataframe()
        result = process_and_predict(df)
        expected = {"pct_flagged", "total_windows", "flagged_windows", "risk_tier", "days_analysed"}
        assert set(result.keys()) == expected

    def test_risk_tier_is_valid(self):
        df = self._make_hr_dataframe()
        result = process_and_predict(df)
        assert result["risk_tier"] in ("Low", "Intermediate", "High")

    def test_pct_flagged_in_range(self):
        df = self._make_hr_dataframe()
        result = process_and_predict(df)
        assert 0 <= result["pct_flagged"] <= 100

    def test_flagged_leq_total(self):
        df = self._make_hr_dataframe()
        result = process_and_predict(df)
        assert result["flagged_windows"] <= result["total_windows"]

    def test_insufficient_data_raises(self):
        # Only 3 records — not enough for any 30-min window with 10+ readings
        df = pd.DataFrame({
            "startDate": pd.date_range("2025-06-01", periods=3, freq="1min"),
            "Value": [72.0, 75.0, 70.0],
        })
        with pytest.raises(ValueError, match="Insufficient data"):
            process_and_predict(df)

    def test_90_day_truncation(self):
        # Data spanning 180 days at 3-min intervals — should only analyse most recent 90
        n = 86400  # 180 days * 480 readings/day (3-min intervals)
        df = pd.DataFrame({
            "startDate": pd.date_range("2025-01-01", periods=n, freq="3min"),
            "Value": np.random.default_rng(42).normal(72, 8, n).clip(40, 180),
        })
        result = process_and_predict(df)
        assert result["days_analysed"] <= 91  # 90 days + possible boundary day

    def test_tier_boundaries(self):
        # Low: < 10%, Intermediate: 10-40%, High: > 40%
        df = self._make_hr_dataframe()
        result = process_and_predict(df)
        pct = result["pct_flagged"]
        tier = result["risk_tier"]
        if pct < 10:
            assert tier == "Low"
        elif pct < 40:
            assert tier == "Intermediate"
        else:
            assert tier == "High"
