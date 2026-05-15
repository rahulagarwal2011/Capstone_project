"""Unit tests for model calibration."""

from __future__ import annotations

from reason_reduce.models.calibration import ConfidenceCalibrator
from reason_reduce.reduce.platt_calibration import PlattCalibrator


class TestConfidenceCalibrator:
    """Tests for the confidence calibrator."""

    def test_unfitted_returns_raw(self) -> None:
        cal = ConfidenceCalibrator()
        assert cal.calibrate(0.7) == 0.7
        assert cal.calibrate(0.3) == 0.3

    def test_output_in_range(self) -> None:
        cal = ConfidenceCalibrator()
        cal._is_fitted = True
        cal._a = 2.0
        cal._b = -1.0
        result = cal.calibrate(0.8)
        assert 0.0 <= result <= 1.0


class TestPlattCalibrator:
    """Tests for Platt scaling."""

    def test_unfitted_identity(self) -> None:
        cal = PlattCalibrator()
        assert cal.calibrate(0.5) == 0.5

    def test_fitted_changes_output(self) -> None:
        cal = PlattCalibrator(a=2.0, b=-0.5, is_fitted=True)
        raw = 0.7
        result = cal.calibrate(raw)
        assert result != raw
        assert 0.0 <= result <= 1.0
