"""Unit tests for API adapter and registry integration."""

from __future__ import annotations

import os

import pytest

from reason_reduce.models.api_adapter import APIConfig, OpenAICompatibleAdapter
from reason_reduce.models.registry import MockLLMAdapter, ModelRegistry


class TestAPIConfig:
    """Tests for API configuration."""

    def test_default_deepseek_endpoint(self) -> None:
        config = APIConfig(provider="deepseek")
        assert "deepseek.com" in config.endpoint

    def test_custom_base_url(self) -> None:
        config = APIConfig(base_url="http://localhost:8000/v1")
        assert config.endpoint == "http://localhost:8000/v1"

    def test_openai_endpoint(self) -> None:
        config = APIConfig(provider="openai")
        assert "openai.com" in config.endpoint


class TestOpenAICompatibleAdapter:
    """Tests for the API adapter (without real API calls)."""

    def test_missing_api_key_logs_warning(self) -> None:
        os.environ.pop("DEEPSEEK_API_KEY", None)
        adapter = OpenAICompatibleAdapter(APIConfig(provider="deepseek"))
        assert adapter._api_key == ""

    def test_stats_start_at_zero(self) -> None:
        adapter = OpenAICompatibleAdapter(APIConfig())
        assert adapter.stats["total_requests"] == 0
        assert adapter.stats["estimated_cost_usd"] == 0.0

    def test_total_cost_calculation(self) -> None:
        adapter = OpenAICompatibleAdapter(
            APIConfig(cost_per_1k_input=0.001, cost_per_1k_output=0.002)
        )
        adapter._total_input_tokens = 1000
        adapter._total_output_tokens = 500
        expected = (1000 / 1000) * 0.001 + (500 / 1000) * 0.002
        assert abs(adapter.total_cost - expected) < 1e-6


class TestRegistryWithAPIModels:
    """Tests for model registry API model integration."""

    def test_get_deepseek_without_key_falls_back_to_mock(self) -> None:
        os.environ.pop("DEEPSEEK_API_KEY", None)
        registry = ModelRegistry()
        adapter = registry.get("deepseek")
        assert isinstance(adapter, MockLLMAdapter)

    def test_get_deepseek_with_key(self) -> None:
        os.environ["DEEPSEEK_API_KEY"] = "test-key-not-real"
        try:
            registry = ModelRegistry()
            adapter = registry.get("deepseek")
            assert isinstance(adapter, OpenAICompatibleAdapter)
        finally:
            del os.environ["DEEPSEEK_API_KEY"]

    def test_get_mock_still_works(self) -> None:
        registry = ModelRegistry()
        adapter = registry.get("mock")
        assert isinstance(adapter, MockLLMAdapter)

    def test_unknown_model_raises(self) -> None:
        registry = ModelRegistry()
        with pytest.raises(KeyError):
            registry.get("totally-fake-model-xyz")
