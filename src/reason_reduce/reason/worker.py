"""Reason worker: Ray actor that processes documents through an LLM.

Each worker holds one LLM adapter instance and processes documents
from its assigned partition, producing ReasonOutput 4-tuples.

In production, this is a @ray.remote actor that batches requests
for throughput and handles failure modes explicitly.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass

import numpy as np

from reason_reduce.ingestion.batch import Doc
from reason_reduce.models.registry import LLMAdapter, LLMResponse
from reason_reduce.reason.prompts import render_prompt
from reason_reduce.monitoring.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TaskSpec:
    """Specification for a reasoning task.

    Attributes:
        task_type: One of "ner", "summarization", "classification", "relation_extraction".
        output_schema: Expected JSON schema for the output.
        prompt_version: Version of the prompt template to use.
        categories: Category list for classification tasks.
    """

    task_type: str = "ner"
    output_schema: dict[str, str] | None = None
    prompt_version: str = "v1.0"
    categories: list[str] | None = None


@dataclass(frozen=True)
class ReasonOutput:
    """Output of a single reason() operation.

    The canonical 4-tuple: (key, value, confidence, reasoning_trace).

    Confidence Semantics:
        confidence is in [0, 1] and represents the model's calibrated certainty.
        confidence=0 indicates a failure (timeout, parse error, etc.).
        confidence=1 is only possible with perfect calibration (rare).
        Downstream reason_reduce() uses these for Dempster-Shafer aggregation.

    Attributes:
        key: Extracted key (e.g., entity type).
        value: Extracted value (e.g., entity name).
        confidence: Calibrated confidence score in [0, 1].
        trace: Chain-of-thought reasoning trace.
        model_id: Which model produced this output.
        latency_ms: Processing time in milliseconds.
        doc_id: Source document ID.
    """

    key: str
    value: str | None
    confidence: float
    trace: str
    model_id: str = "unknown"
    latency_ms: float = 0.0
    doc_id: str = ""


class ReasonWorker:
    """Worker that processes documents through an LLM.

    Can be used directly (testing) or wrapped as a @ray.remote actor (production).

    Args:
        adapter: LLM adapter for inference.
        partition_id: ID of the partition this worker handles.
        partition_context: Semantic context for this partition.
        max_retries: Max retries on JSON parse failure.
    """

    def __init__(
        self,
        adapter: LLMAdapter,
        partition_id: int = 0,
        partition_context: str | None = None,
        max_retries: int = 2,
    ) -> None:
        self._adapter = adapter
        self._partition_id = partition_id
        self._partition_context = partition_context
        self._max_retries = max_retries
        self._processed_count = 0

    async def process(self, doc: Doc, task: TaskSpec) -> ReasonOutput:
        """Process a single document through the LLM.

        Pipeline: render prompt → call LLM → parse JSON → compute confidence.
        On parse failure, retries with a stricter prompt (up to max_retries).

        Args:
            doc: Input document.
            task: Task specification.

        Returns:
            ReasonOutput with extracted information.
        """
        start = time.perf_counter()
        self._processed_count += 1

        prompt = self._render_prompt(doc, task)

        for attempt in range(1 + self._max_retries):
            try:
                response: LLMResponse = await asyncio.wait_for(
                    self._adapter.generate(
                        prompt=prompt,
                        max_tokens=256,
                        temperature=0.0,
                    ),
                    timeout=30.0,
                )

                parsed = self._parse_response(response)
                if parsed is not None:
                    confidence = self._compute_confidence(parsed, response)
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    return ReasonOutput(
                        key=parsed.get("key", task.task_type),
                        value=parsed.get("value"),
                        confidence=confidence,
                        trace=parsed.get("trace", response.text[:500]),
                        model_id=response.model_id,
                        latency_ms=elapsed_ms,
                        doc_id=doc.id,
                    )

                if attempt < self._max_retries:
                    prompt = self._stricter_prompt(doc, task, response.text)

            except asyncio.TimeoutError:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.error(
                    "worker_timeout",
                    doc_id=doc.id,
                    partition=self._partition_id,
                    attempt=attempt,
                )
                return ReasonOutput(
                    key=task.task_type, value=None, confidence=0.0,
                    trace="TIMEOUT", latency_ms=elapsed_ms, doc_id=doc.id,
                )

            except Exception as e:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.error(
                    "worker_error",
                    doc_id=doc.id,
                    error=str(e),
                    attempt=attempt,
                )
                if attempt == self._max_retries:
                    return ReasonOutput(
                        key=task.task_type, value=None, confidence=0.0,
                        trace=f"ERROR: {e}", latency_ms=elapsed_ms, doc_id=doc.id,
                    )

        elapsed_ms = (time.perf_counter() - start) * 1000
        return ReasonOutput(
            key=task.task_type, value=None, confidence=0.0,
            trace=f"PARSE_FAILURE after {self._max_retries} retries",
            latency_ms=elapsed_ms, doc_id=doc.id,
        )

    async def process_batch(self, docs: list[Doc], task: TaskSpec) -> list[ReasonOutput]:
        """Process a batch of documents concurrently.

        Uses asyncio.gather for within-worker parallelism (vLLM handles
        continuous batching on the backend).

        Args:
            docs: Batch of documents.
            task: Task specification.

        Returns:
            List of ReasonOutputs in same order as input docs.
        """
        tasks = [self.process(doc, task) for doc in docs]
        return await asyncio.gather(*tasks)

    def _render_prompt(self, doc: Doc, task: TaskSpec) -> str:
        """Render a prompt using versioned templates."""
        try:
            return render_prompt(
                task_type=task.task_type,
                document_text=doc.text,
                version=task.prompt_version,
                context=self._partition_context,
                categories=", ".join(task.categories) if task.categories else "",
            )
        except KeyError:
            return (
                f"Task: {task.task_type}\n"
                f"Document: {doc.text[:2000]}\n"
                f"Extract structured information as JSON with keys: key, value, confidence.\n"
            )

    def _stricter_prompt(self, doc: Doc, task: TaskSpec, previous_output: str) -> str:
        """Generate a stricter retry prompt after parse failure."""
        return (
            f"Task: {task.task_type}\n"
            f"Document: {doc.text[:1500]}\n\n"
            f"Your previous response was not valid JSON: {previous_output[:200]}\n\n"
            f"You MUST respond with ONLY a JSON object in this exact format:\n"
            f'{{"key": "<type>", "value": "<extracted>", "confidence": <0.0-1.0>, "trace": "<reasoning>"}}\n'
        )

    def _parse_response(self, response: LLMResponse) -> dict[str, str] | None:
        """Parse LLM response as JSON. Returns None on failure."""
        text = response.text.strip()

        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            result = json.loads(text)
            if isinstance(result, dict) and "key" in result:
                return result
            return None
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            return None

    def _compute_confidence(
        self, parsed: dict[str, str], response: LLMResponse
    ) -> float:
        """Compute confidence from parsed output + logprobs.

        Priority:
        1. If the LLM returned an explicit "confidence" field, use it
        2. Otherwise, compute from logprobs (mean exp(logprob))
        3. Fallback: 0.5

        The raw score will be passed through Platt calibration downstream.
        """
        explicit = parsed.get("confidence")
        if explicit is not None:
            try:
                conf = float(explicit)
                return min(1.0, max(0.0, conf))
            except (ValueError, TypeError):
                pass

        if response.logprobs:
            mean_lp = float(np.mean(response.logprobs))
            return min(1.0, max(0.0, float(np.exp(mean_lp))))

        return 0.5
