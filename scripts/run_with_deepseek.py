"""Run the Reason-Reduce pipeline with DeepSeek API.

Usage:
    export DEEPSEEK_API_KEY=your-key-here
    PYTHONPATH=src python scripts/run_with_deepseek.py

This runs real LLM inference on your MacBook via DeepSeek's API.
No GPU needed — all inference happens server-side.
"""

from __future__ import annotations

import os
import sys
import time

from reason_reduce.ingestion.batch import Doc
from reason_reduce.reason.api import reason
from reason_reduce.reason.worker import TaskSpec
from reason_reduce.reduce.api import reason_reduce
from reason_reduce.models.registry import ModelRegistry
from reason_reduce.monitoring.logger import get_logger

logger = get_logger(__name__)


def main() -> int:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        logger.error("missing_api_key", hint="export DEEPSEEK_API_KEY=your-key-here")
        return 1

    docs = [
        Doc(id="1", text="Aspirin (acetylsalicylic acid) is a nonsteroidal anti-inflammatory drug used to treat pain, fever, and inflammation. It is also used as a blood thinner to prevent heart attacks and strokes."),
        Doc(id="2", text="Metformin hydrochloride is the first-line medication for type 2 diabetes mellitus. It works by decreasing glucose production in the liver and improving insulin sensitivity."),
        Doc(id="3", text="Lisinopril is an angiotensin-converting enzyme (ACE) inhibitor used primarily for the treatment of hypertension, heart failure, and post-myocardial infarction."),
        Doc(id="4", text="CRISPR-Cas9 gene editing technology was developed by Jennifer Doudna and Emmanuelle Charpentier, who received the Nobel Prize in Chemistry in 2020."),
        Doc(id="5", text="Tesla Inc., founded by Elon Musk, reported Q4 2025 revenue of $25.7 billion, exceeding analyst expectations of $24.1 billion."),
    ]

    logger.info("pipeline_start", n_docs=len(docs), model="deepseek")
    start = time.perf_counter()

    reason_results = reason(
        docs,
        task=TaskSpec(task_type="ner"),
        model="deepseek",
        n_partitions=2,
        seed=42,
    )

    logger.info("reason_phase_complete", n_outputs=len(reason_results))

    for r in reason_results:
        logger.info(
            "reason_output",
            doc_id=r.doc_id,
            key=r.key,
            value=r.value,
            confidence=round(r.confidence, 3),
            latency_ms=round(r.latency_ms, 1),
        )

    reduce_results = reason_reduce(reason_results, strategy="ds", seed=42)

    elapsed = time.perf_counter() - start

    logger.info("pipeline_complete", elapsed_seconds=round(elapsed, 2))
    logger.info("results_summary", n_consensus=len(reduce_results))

    for r in reduce_results:
        logger.info(
            "consensus",
            key=r.key,
            value=r.value,
            confidence=round(r.confidence, 3),
            n_sources=r.n_sources,
        )

    registry = ModelRegistry()
    adapter = registry.get("deepseek")
    if hasattr(adapter, "stats"):
        logger.info("cost_summary", **adapter.stats)

    return 0


if __name__ == "__main__":
    sys.exit(main())
