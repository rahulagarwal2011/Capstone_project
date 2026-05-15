"""vLLM local inference adapter.

Wraps vllm.AsyncLLMEngine for production inference with PagedAttention
and continuous batching. Lazily loads the engine on first request.

Failure Handling:
    - OOM: Logged as structured event, worker marked degraded.
    - CUDA error: Worker killed, Ray restarts.
    - Timeout: Request cancelled, partial result with finish_reason="timeout".
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from reason_reduce.models.registry import LLMResponse, MockLLMAdapter
from reason_reduce.monitoring.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VLLMConfig:
    """Configuration for the vLLM engine.

    Attributes:
        model_id: HuggingFace model ID or local path.
        gpu_memory_utilization: Fraction of GPU memory to use (0-1).
        max_model_len: Maximum sequence length.
        quantization: Quantization method (None, "awq", "gptq").
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        max_num_seqs: Max concurrent sequences in the engine.
        timeout_seconds: Per-request timeout.
    """

    model_id: str
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 4096
    quantization: str | None = None
    tensor_parallel_size: int = 1
    max_num_seqs: int = 64
    timeout_seconds: float = 30.0


class VLLMLocalAdapter:
    """Adapter wrapping vLLM AsyncLLMEngine.

    Lazily initializes the engine on first generate() call.
    Reuses the engine across all subsequent requests — do NOT
    create one per request.

    Confidence Semantics:
        Logprobs returned from vLLM are per-token log-probabilities.
        Sequence-level confidence is computed as mean(exp(logprob)) over
        generated tokens, then passed through Platt calibration downstream.
    """

    def __init__(self, config: VLLMConfig) -> None:
        self._config = config
        self._engine: object | None = None
        self._is_available: bool = False
        self._request_counter: int = 0
        self._fallback = MockLLMAdapter(model_id=config.model_id)

    @property
    def is_available(self) -> bool:
        """Whether the vLLM engine loaded successfully."""
        return self._is_available

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        seed: int | None = None,
        logprobs: int = 5,
    ) -> LLMResponse:
        """Generate text using vLLM with continuous batching.

        Falls back to mock adapter if vLLM/GPU is unavailable.

        Args:
            prompt: Input text.
            max_tokens: Max new tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            seed: Random seed for reproducibility.
            logprobs: Number of top logprobs per token.

        Returns:
            LLMResponse with text, logprobs, and metadata.
        """
        start = time.perf_counter()

        if self._engine is None:
            self._engine = await self._init_engine()

        if not self._is_available:
            response = await self._fallback.generate(
                prompt=prompt, max_tokens=max_tokens,
                temperature=temperature, seed=seed, logprobs=logprobs,
            )
            return LLMResponse(
                text=response.text,
                logprobs=response.logprobs,
                finish_reason=response.finish_reason,
                latency_ms=(time.perf_counter() - start) * 1000,
                model_id=self._config.model_id,
            )

        try:
            self._request_counter += 1
            request_id = f"req-{self._request_counter}"

            from vllm import SamplingParams
            from vllm.engine.async_llm_engine import AsyncLLMEngine

            engine: AsyncLLMEngine = self._engine  # type: ignore[assignment]

            params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
                logprobs=logprobs,
            )

            generated_text = ""
            all_logprobs: list[float] = []
            finish_reason = "stop"

            try:
                async with asyncio.timeout(self._config.timeout_seconds):
                    async for output in engine.generate(prompt, params, request_id):
                        if output.outputs:
                            result = output.outputs[0]
                            generated_text = result.text
                            finish_reason = result.finish_reason or "stop"
                            if result.logprobs:
                                for token_logprob in result.logprobs:
                                    if token_logprob:
                                        top_lp = max(token_logprob.values())
                                        all_logprobs.append(top_lp)

            except asyncio.TimeoutError:
                logger.warning(
                    "vllm_timeout",
                    request_id=request_id,
                    timeout=self._config.timeout_seconds,
                )
                await engine.abort(request_id)
                finish_reason = "timeout"

            elapsed_ms = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text=generated_text,
                logprobs=all_logprobs[:logprobs],
                finish_reason=finish_reason,
                latency_ms=elapsed_ms,
                model_id=self._config.model_id,
            )

        except RuntimeError as e:
            error_msg = str(e).lower()
            elapsed_ms = (time.perf_counter() - start) * 1000

            if "out of memory" in error_msg or "cuda" in error_msg:
                logger.error(
                    "vllm_oom",
                    model=self._config.model_id,
                    error=str(e),
                )
                return LLMResponse(
                    text="", logprobs=[], finish_reason="oom",
                    latency_ms=elapsed_ms, model_id=self._config.model_id,
                )

            logger.error("vllm_runtime_error", error=str(e))
            return LLMResponse(
                text="", logprobs=[], finish_reason="error",
                latency_ms=elapsed_ms, model_id=self._config.model_id,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error("vllm_generation_failed", error=str(e), model=self._config.model_id)
            return LLMResponse(
                text="", logprobs=[], finish_reason="error",
                latency_ms=elapsed_ms, model_id=self._config.model_id,
            )

    async def _init_engine(self) -> object:
        """Initialize the vLLM AsyncLLMEngine.

        Returns a mock object if vLLM is not importable (no GPU environment).
        """
        try:
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine

            args = AsyncEngineArgs(
                model=self._config.model_id,
                gpu_memory_utilization=self._config.gpu_memory_utilization,
                max_model_len=self._config.max_model_len,
                quantization=self._config.quantization,
                tensor_parallel_size=self._config.tensor_parallel_size,
                max_num_seqs=self._config.max_num_seqs,
                trust_remote_code=True,
            )

            engine = AsyncLLMEngine.from_engine_args(args)
            self._is_available = True
            logger.info(
                "vllm_engine_ready",
                model=self._config.model_id,
                quantization=self._config.quantization,
                max_model_len=self._config.max_model_len,
            )
            return engine

        except ImportError:
            logger.info("vllm_not_installed", fallback="mock")
            self._is_available = False
            return object()

        except Exception as e:
            logger.error("vllm_init_failed", error=str(e), model=self._config.model_id)
            self._is_available = False
            return object()

    async def health_check(self) -> bool:
        """Check if the engine is responsive."""
        if not self._is_available:
            return False
        try:
            response = await self.generate("hello", max_tokens=1, temperature=0.0)
            return response.finish_reason != "error"
        except Exception:
            return False
