"""Unit tests for the configuration system."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from reason_reduce.config.loader import Settings, ThresholdSettings, load_settings


class TestThresholdSettings:
    """Tests for threshold configuration validation."""

    def test_valid_defaults(self) -> None:
        settings = ThresholdSettings()
        assert settings.tau_confidence == 0.7
        assert settings.tau_entropy == 0.5
        assert settings.max_workers == 4

    def test_invalid_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            ThresholdSettings(tau_confidence=1.5)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            ThresholdSettings(tau_confidence=-0.1)

    def test_conflict_high_must_exceed_low(self) -> None:
        with pytest.raises(ValidationError):
            ThresholdSettings(tau_conflict_low=0.5, tau_conflict_high=0.3)

    def test_conflict_equal_raises(self) -> None:
        with pytest.raises(ValidationError):
            ThresholdSettings(tau_conflict_low=0.5, tau_conflict_high=0.5)


class TestSettings:
    """Tests for root settings loading."""

    def test_load_settings_returns_valid(self) -> None:
        settings = load_settings()
        assert isinstance(settings, Settings)
        assert settings.thresholds.tau_confidence >= 0.0
        assert settings.default_seed == 42

    def test_default_model(self) -> None:
        settings = load_settings()
        assert settings.default_model == "mistral-7b"
