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

_API_MODELS: dict[str, dict[str, str | float]] = {
    "deepseek": {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "cost_per_1k_input": 0.0014,
        "cost_per_1k_output": 0.0028,
    },
    "deepseek-reasoner": {
        "provider": "deepseek",
        "model": "deepseek-reasoner",
        "api_key_env": "DEEPSEEK_API_KEY",
        "cost_per_1k_input": 0.0055,
        "cost_per_1k_output": 0.016,
    },
    "gpt-4o": {
        "provider": "openai",
        "model": "gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
        "cost_per_1k_input": 0.005,
        "cost_per_1k_output": 0.015,
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
        "cost_per_1k_input": 0.00015,
        "cost_per_1k_output": 0.0006,
    },
    "groq-llama": {
        "provider": "groq",
        "model": "llama-3.1-70b-versatile",
        "api_key_env": "GROQ_API_KEY",
        "cost_per_1k_input": 0.00059,
        "cost_per_1k_output": 0.00079,
    },
}


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

        Supports:
            - "mock": deterministic mock (no network, no GPU)
            - "deepseek": DeepSeek API via DEEPSEEK_API_KEY
            - "deepseek-reasoner": DeepSeek R1 reasoning model
            - "gpt-4o" / "gpt-4o-mini": OpenAI API via OPENAI_API_KEY
            - Any model in registry: falls back to mock if no GPU

        Args:
            model_id: Model name from the registry or a known API model.

        Returns:
            An LLMAdapter instance.

        Raises:
            KeyError: If model_id is not recognized.
        """
        if model_id in self._adapters:
            return self._adapters[model_id]

        if model_id == "mock":
            self._adapters["mock"] = MockLLMAdapter()
            return self._adapters["mock"]

        api_adapter = self._try_create_api_adapter(model_id)
        if api_adapter is not None:
            self._adapters[model_id] = api_adapter
            return api_adapter

        if model_id not in self._models:
            msg = (
                f"Model '{model_id}' not found. "
                f"Available: {list(self._models.keys()) + list(_API_MODELS.keys())}"
            )
            raise KeyError(msg)

        logger.info("adapter_created", model_id=model_id, adapter_type="mock_fallback")
        self._adapters[model_id] = MockLLMAdapter(model_id=model_id)
        return self._adapters[model_id]

    def _try_create_api_adapter(self, model_id: str) -> LLMAdapter | None:
        """Try to create an API adapter for the given model ID."""
        from reason_reduce.models.api_adapter import APIConfig, OpenAICompatibleAdapter

        if model_id not in _API_MODELS:
            return None

        config_kwargs = _API_MODELS[model_id]
        config = APIConfig(**config_kwargs)

        import os

        if not os.environ.get(config.api_key_env):
            logger.warning(
                "api_key_not_set",
                model=model_id,
                env_var=config.api_key_env,
                hint=f"Set {config.api_key_env} to use {model_id}",
            )
            return None

        logger.info("api_adapter_created", model=model_id, provider=config.provider)
        return OpenAICompatibleAdapter(config)

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
