"""Unit tests for model registry and adapters."""

from __future__ import annotations

import asyncio

import pytest

from reason_reduce.models.registry import (
    LLMResponse,
    MockLLMAdapter,
    ModelRegistry,
)


class TestMockLLMAdapter:
    """Tests for the mock LLM adapter."""

    def test_returns_default_response(self) -> None:
        adapter = MockLLMAdapter()
        response = asyncio.get_event_loop().run_until_complete(adapter.generate("test prompt"))
        assert isinstance(response, LLMResponse)
        assert response.finish_reason == "stop"
        assert response.model_id == "mock"

    def test_deterministic_with_seed(self) -> None:
        adapter = MockLLMAdapter(seed=42)
        r1 = asyncio.get_event_loop().run_until_complete(adapter.generate("prompt"))
        r2 = asyncio.get_event_loop().run_until_complete(adapter.generate("prompt"))
        assert r1.text == r2.text
        assert r1.logprobs == r2.logprobs

    def test_pattern_matching(self) -> None:
        adapter = MockLLMAdapter()
        adapter.register_pattern("medical", '{"key": "DRUG", "value": "aspirin"}')
        response = asyncio.get_event_loop().run_until_complete(
            adapter.generate("This is a medical document")
        )
        assert "aspirin" in response.text

    def test_logprobs_present(self) -> None:
        adapter = MockLLMAdapter()
        response = asyncio.get_event_loop().run_until_complete(adapter.generate("test", logprobs=5))
        assert len(response.logprobs) == 5
        assert all(lp < 0 for lp in response.logprobs)


class TestModelRegistry:
    """Tests for the model registry."""

    def test_get_mock(self) -> None:
        registry = ModelRegistry()
        adapter = registry.get("mock")
        assert isinstance(adapter, MockLLMAdapter)

    def test_get_unknown_raises(self) -> None:
        registry = ModelRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent-model-xyz")

    def test_list_available(self) -> None:
        registry = ModelRegistry()
        models = registry.list_available()
        assert len(models) > 0

    def test_can_fit(self) -> None:
        registry = ModelRegistry()
        assert registry.can_fit("mistral-7b", 8.0) is True
        assert registry.can_fit("llama-3.1-70b", 8.0) is False
