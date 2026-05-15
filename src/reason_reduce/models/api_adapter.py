"""OpenAI-compatible API adapter for external LLM inference.

Supports any provider with an OpenAI-compatible API:
    - DeepSeek (deepseek-chat, deepseek-reasoner)
    - OpenAI (gpt-4o, gpt-4o-mini)
    - Groq (llama, mistral)
    - Together AI
    - Mistral API
    - Any OpenAI-compatible endpoint

Used for:
    - Real inference on MacBook (no local GPU needed)
    - Baseline comparison (Spark + GPT-4)
    - Fallback when vLLM is unavailable
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx

from reason_reduce.models.registry import LLMResponse
from reason_reduce.monitoring.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_ENDPOINTS: dict[str, str] = {
    "deepseek": "https://api.deepseek.com/v1",
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
    "mistral": "https://api.mistral.ai/v1",
}


@dataclass
class APIConfig:
    """Configuration for an OpenAI-compatible API.

    Attributes:
        provider: Provider name (deepseek, openai, groq, together, mistral).
        model: Model ID at the provider (e.g., "deepseek-chat").
        api_key_env: Environment variable name holding the API key.
        base_url: API base URL (auto-detected from provider if not set).
        max_retries: Number of retries on transient failures.
        timeout_seconds: Per-request timeout.
        cost_per_1k_input: Cost per 1k input tokens (for tracking).
        cost_per_1k_output: Cost per 1k output tokens (for tracking).
    """

    provider: str = "deepseek"
    model: str = "deepseek-chat"
    api_key_env: str = "DEEPSEEK_API_KEY"
    base_url: str | None = None
    max_retries: int = 3
    timeout_seconds: float = 60.0
    cost_per_1k_input: float = 0.0014
    cost_per_1k_output: float = 0.0028

    @property
    def endpoint(self) -> str:
        if self.base_url:
            return self.base_url
        env_url = os.environ.get(f"{self.provider.upper()}_BASE_URL")
        if env_url:
            return env_url
        return _DEFAULT_ENDPOINTS.get(self.provider, "https://api.deepseek.com/v1")


class OpenAICompatibleAdapter:
    """Adapter for any OpenAI-compatible API.

    Works with DeepSeek, OpenAI, Groq, Together, Mistral, and any
    provider implementing the /v1/chat/completions endpoint.

    Tracks token usage for cost estimation.
    """

    def __init__(self, config: APIConfig | None = None) -> None:
        self._config = config or APIConfig()
        self._api_key = os.environ.get(self._config.api_key_env, "")
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_requests = 0

        if not self._api_key:
            logger.warning(
                "api_key_missing",
                env_var=self._config.api_key_env,
                provider=self._config.provider,
            )

    @property
    def total_cost(self) -> float:
        """Estimated total cost so far."""
        input_cost = (self._total_input_tokens / 1000) * self._config.cost_per_1k_input
        output_cost = (self._total_output_tokens / 1000) * self._config.cost_per_1k_output
        return input_cost + output_cost

    @property
    def stats(self) -> dict[str, float | int]:
        """Usage statistics."""
        return {
            "total_requests": self._total_requests,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "estimated_cost_usd": round(self.total_cost, 4),
        }

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        seed: int | None = None,
        logprobs: int = 5,
    ) -> LLMResponse:
        """Generate text via the API.

        Args:
            prompt: Input prompt (sent as user message).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            seed: Random seed (supported by DeepSeek and OpenAI).
            logprobs: Request logprobs (not all providers support this).

        Returns:
            LLMResponse with generated text and metadata.
        """
        start = time.perf_counter()

        messages = [{"role": "user", "content": prompt}]

        body: dict = {
            "model": self._config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if seed is not None:
            body["seed"] = seed
        if logprobs > 0:
            body["logprobs"] = True
            body["top_logprobs"] = min(logprobs, 5)

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(self._config.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self._config.timeout_seconds) as client:
                    response = await client.post(
                        f"{self._config.endpoint}/chat/completions",
                        headers=headers,
                        json=body,
                    )

                if response.status_code == 429:
                    wait = 2**attempt
                    logger.warning("rate_limited", wait_seconds=wait, attempt=attempt)
                    import asyncio

                    await asyncio.sleep(wait)
                    continue

                if response.status_code != 200:
                    logger.error(
                        "api_error",
                        status=response.status_code,
                        body=response.text[:200],
                    )
                    if attempt == self._config.max_retries - 1:
                        elapsed_ms = (time.perf_counter() - start) * 1000
                        return LLMResponse(
                            text="",
                            logprobs=[],
                            finish_reason="error",
                            latency_ms=elapsed_ms,
                            model_id=self._config.model,
                        )
                    continue

                data = response.json()
                choice = data["choices"][0]
                text = choice["message"]["content"]
                finish_reason = choice.get("finish_reason", "stop")

                usage = data.get("usage", {})
                self._total_input_tokens += usage.get("prompt_tokens", 0)
                self._total_output_tokens += usage.get("completion_tokens", 0)
                self._total_requests += 1

                token_logprobs: list[float] = []
                if "logprobs" in choice and choice["logprobs"]:
                    content_logprobs = choice["logprobs"].get("content", [])
                    for token_info in content_logprobs[:logprobs]:
                        if "logprob" in token_info:
                            token_logprobs.append(token_info["logprob"])

                elapsed_ms = (time.perf_counter() - start) * 1000

                logger.info(
                    "api_request_complete",
                    provider=self._config.provider,
                    model=self._config.model,
                    latency_ms=round(elapsed_ms, 1),
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                )

                return LLMResponse(
                    text=text,
                    logprobs=token_logprobs,
                    finish_reason=finish_reason,
                    latency_ms=elapsed_ms,
                    model_id=self._config.model,
                )

            except httpx.TimeoutException:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.warning("api_timeout", attempt=attempt, elapsed_ms=elapsed_ms)
                if attempt == self._config.max_retries - 1:
                    return LLMResponse(
                        text="",
                        logprobs=[],
                        finish_reason="timeout",
                        latency_ms=elapsed_ms,
                        model_id=self._config.model,
                    )

            except Exception as e:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.error("api_exception", error=str(e), attempt=attempt)
                if attempt == self._config.max_retries - 1:
                    return LLMResponse(
                        text="",
                        logprobs=[],
                        finish_reason="error",
                        latency_ms=elapsed_ms,
                        model_id=self._config.model,
                    )

        elapsed_ms = (time.perf_counter() - start) * 1000
        return LLMResponse(
            text="",
            logprobs=[],
            finish_reason="error",
            latency_ms=elapsed_ms,
            model_id=self._config.model,
        )
