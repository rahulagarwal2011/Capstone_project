"""Public API for the reason() operator.

reason() is the core operator that replaces map() in MapReduce.
It processes documents through an embedded LLM and produces
probabilistic 4-tuples: (key, value, confidence, reasoning_trace).

Supports two execution modes:
    - Local (default): direct async execution, for testing and small datasets.
    - Distributed: Ray-based parallel execution across worker actors.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from reason_reduce.ingestion.batch import Doc
from reason_reduce.models.registry import ModelRegistry
from reason_reduce.reason.partitioner import partition_documents
from reason_reduce.reason.context_propagator import get_partition_context
from reason_reduce.reason.worker import ReasonOutput, ReasonWorker, TaskSpec
from reason_reduce.monitoring.logger import get_logger

if TYPE_CHECKING:
    from reason_reduce.api.rdd import ReasonRDD

logger = get_logger(__name__)


def reason(
    data: list[Doc],
    task: TaskSpec | None = None,
    model: str = "mock",
    n_partitions: int | None = None,
    seed: int = 42,
    distributed: bool = False,
    n_workers: int = 4,
) -> list[ReasonOutput]:
    """Apply LLM-powered reasoning to a collection of documents.

    This is the core Reason-Reduce operator. It:
    1. Partitions documents semantically
    2. Assigns partitions to workers
    3. Each worker processes its docs through an LLM
    4. Returns a list of ReasonOutput 4-tuples

    Confidence Semantics:
        Each output carries a confidence in [0, 1]. A confidence of 0
        indicates failure (timeout, parse error). Downstream reduce()
        uses these confidences for Dempster-Shafer aggregation.

    Args:
        data: Input documents to process.
        task: Task specification (defaults to NER).
        model: Model ID from the registry ("mock" for testing).
        n_partitions: Number of partitions (None = auto).
        seed: Random seed for reproducibility.
        distributed: If True, use Ray for parallel execution.
        n_workers: Number of Ray workers (distributed mode only).

    Returns:
        List of ReasonOutput objects, one per document.
    """
    if not data:
        return []

    if task is None:
        task = TaskSpec()

    partitions = partition_documents(data, n_partitions=n_partitions, seed=seed)

    if distributed:
        results = _run_distributed(partitions, task, model, n_workers)
    else:
        results = _run_local(partitions, task, model)

    logger.info(
        "reason_complete",
        n_docs=len(data),
        n_results=len(results),
        model=model,
        distributed=distributed,
        mean_confidence=sum(r.confidence for r in results) / max(len(results), 1),
    )
    return results


def _run_local(
    partitions: list,
    task: TaskSpec,
    model: str,
) -> list[ReasonOutput]:
    """Execute reason() locally (single process, async)."""
    registry = ModelRegistry()
    adapter = registry.get(model)
    all_partitions = partitions

    results: list[ReasonOutput] = []

    loop = asyncio.new_event_loop()
    try:
        for partition in partitions:
            context = get_partition_context(partition, all_partitions)
            worker = ReasonWorker(
                adapter=adapter,
                partition_id=partition.id,
                partition_context=context,
            )
            batch_results = loop.run_until_complete(
                worker.process_batch(partition.docs, task)
            )
            results.extend(batch_results)
    finally:
        loop.close()

    return results


def _run_distributed(
    partitions: list,
    task: TaskSpec,
    model: str,
    n_workers: int,
) -> list[ReasonOutput]:
    """Execute reason() on a Ray cluster with worker pool."""
    import ray

    from reason_reduce.orchestration.scheduler import init_cluster

    init_cluster(local=True, n_workers=n_workers)

    from reason_reduce.orchestration.scheduler import create_reason_worker_pool

    actors = create_reason_worker_pool(n_workers=min(n_workers, len(partitions)), model_id=model)

    task_data = {
        "task_type": task.task_type,
        "prompt_version": task.prompt_version,
    }
    if task.categories:
        task_data["categories"] = task.categories

    futures = []
    for i, partition in enumerate(partitions):
        actor = actors[i % len(actors)]
        for doc in partition.docs:
            doc_data = {"id": doc.id, "text": doc.text, "metadata": doc.metadata}
            future = actor.process.remote(doc_data, task_data)
            futures.append(future)

    raw_results = ray.get(futures)

    results = [
        ReasonOutput(
            key=r["key"],
            value=r["value"],
            confidence=r["confidence"],
            trace=r["trace"],
            model_id=r["model_id"],
            latency_ms=r["latency_ms"],
            doc_id=r["doc_id"],
        )
        for r in raw_results
    ]
    return results
