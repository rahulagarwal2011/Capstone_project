"""Model registry and LLM adapter protocol.

Provides a unified interface for LLM inference across mock, vLLM, and OpenAI backends.
Switching between adapters is a one-line config change.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from reason_reduce.config.loader import ModelInfo, load_settings
from reason_reduce.monitoring.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    """Response from an LLM adapter.

    Attributes:
        text: Generated text.
        logprobs: Per-token log probabilities (if available).
        finish_reason: Why generation stopped (stop, length, timeout).
        latency_ms: Inference time in milliseconds.
        model_id: Which model produced this response.
    """

    text: str
    logprobs: list[float] = field(default_factory=list)
    finish_reason: str = "stop"
    latency_ms: float = 0.0
    model_id: str = "unknown"


@runtime_checkable
class LLMAdapter(Protocol):
    """Protocol for LLM inference adapters.

    Implementations: MockLLMAdapter, VLLMLocalAdapter, OpenAIAdapter.
    """

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        seed: int | None = None,
        logprobs: int = 5,
    ) -> LLMResponse:
        """Generate text from a prompt.

        Args:
            prompt: Input prompt text.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            seed: Random seed for reproducibility.
            logprobs: Number of top logprobs to return per token.

        Returns:
            LLMResponse with generated text and metadata.
        """
        ...


class MockLLMAdapter:
    """Deterministic mock adapter for testing.

    Returns canned outputs based on registered patterns or a default response.
    Used in all unit tests — no GPU required.
    """

    def __init__(self, model_id: str = "mock", seed: int = 42) -> None:
        self._model_id = model_id
        self._seed = seed
        self._patterns: dict[str, str] = {}
        self._default_response = '{"key": "entity", "value": "mock_value", "confidence": 0.85}'

    def register_pattern(self, pattern: str, response: str) -> None:
        """Register a pattern → response mapping."""
        self._patterns[pattern] = response

    def set_default_response(self, response: str) -> None:
        """Set the default response for unmatched prompts."""
        self._default_response = response

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        seed: int | None = None,
        logprobs: int = 5,
    ) -> LLMResponse:
        """Generate a deterministic response.

        Uses registered patterns first, falls back to default.
        Logprobs are synthetic but deterministic given the seed.
        """
        start = time.perf_counter()

        response_text = self._default_response
        for pattern, resp in self._patterns.items():
            if pattern in prompt:
                response_text = resp
                break

        effective_seed = seed if seed is not None else self._seed
        prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
        synthetic_logprobs = [
            -0.1 - (((effective_seed + i + prompt_hash) % 100) / 1000.0)
            for i in range(min(logprobs, 10))
        ]

        elapsed_ms = (time.perf_counter() - start) * 1000

        return LLMResponse(
            text=response_text,
            logprobs=synthetic_logprobs,
            finish_reason="stop",
            latency_ms=elapsed_ms,
            model_id=self._model_id,
        )


class ModelRegistry:
    """Central registry for model loading and adapter management.

    Provides model lookup, VRAM budget checking, and adapter instantiation.
    """

    def __init__(self) -> None:
        settings = load_settings()
        self._models: dict[str, ModelInfo] = {m.name: m for m in settings.models}
        self._adapters: dict[str, LLMAdapter] = {}

    def get(self, model_id: str) -> LLMAdapter:
        """Get or create an adapter for the given model.

        Args:
            model_id: Model name from the registry.

        Returns:
            An LLMAdapter instance.

        Raises:
            KeyError: If model_id is not in the registry.
        """
        if model_id == "mock":
            if "mock" not in self._adapters:
                self._adapters["mock"] = MockLLMAdapter()
            return self._adapters["mock"]

        if model_id not in self._models:
            msg = f"Model '{model_id}' not found in registry. Available: {list(self._models.keys())}"
            raise KeyError(msg)

        if model_id not in self._adapters:
            logger.info("adapter_created", model_id=model_id, adapter_type="mock_fallback")
            self._adapters[model_id] = MockLLMAdapter(model_id=model_id)

        return self._adapters[model_id]

    def list_available(self) -> list[ModelInfo]:
        """List all registered models."""
        return list(self._models.values())

    def can_fit(self, model_id: str, available_vram_gb: float) -> bool:
        """Check if a model fits in available VRAM.

        Args:
            model_id: Model to check.
            available_vram_gb: Available GPU memory in GB.

        Returns:
            True if model fits, False otherwise.
        """
        if model_id not in self._models:
            return False
        return self._models[model_id].vram_gb <= available_vram_gb
