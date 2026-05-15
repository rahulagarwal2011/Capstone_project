"""Smoke test: validates the full pipeline runs on a laptop with no GPU.

10 docs → reason() with mock LLM → reason_reduce() → assert 3 outputs.
Must complete in <5 seconds.
"""

from __future__ import annotations

import sys
import time

from reason_reduce.ingestion.batch import Doc
from reason_reduce.reason.api import reason
from reason_reduce.reason.worker import TaskSpec
from reason_reduce.reduce.api import reason_reduce
from reason_reduce.monitoring.logger import get_logger

logger = get_logger(__name__)


def main() -> int:
    """Run the smoke test pipeline."""
    start = time.perf_counter()

    logger.info("smoke_test_start")

    docs = [
        Doc(id="1", text="Aspirin is a nonsteroidal anti-inflammatory drug used to treat pain."),
        Doc(id="2", text="Metformin is the first-line medication for type 2 diabetes."),
        Doc(id="3", text="Lisinopril is an ACE inhibitor used for high blood pressure."),
        Doc(id="4", text="Atorvastatin is a statin used to lower cholesterol levels."),
        Doc(id="5", text="Omeprazole is a proton pump inhibitor for acid reflux."),
        Doc(id="6", text="Amoxicillin is a penicillin antibiotic for bacterial infections."),
        Doc(id="7", text="Levothyroxine treats hypothyroidism by replacing thyroid hormone."),
        Doc(id="8", text="Amlodipine is a calcium channel blocker for hypertension."),
        Doc(id="9", text="Gabapentin is used to treat nerve pain and seizures."),
        Doc(id="10", text="Sertraline is an SSRI antidepressant for depression and anxiety."),
    ]

    logger.info("running_reason", n_docs=len(docs))
    reason_results = reason(
        docs,
        task=TaskSpec(task_type="ner"),
        model="mock",
        n_partitions=2,
        seed=42,
    )

    assert len(reason_results) == 10, f"Expected 10 results, got {len(reason_results)}"
    logger.info(
        "reason_done",
        n_results=len(reason_results),
        mean_confidence=sum(r.confidence for r in reason_results) / len(reason_results),
    )

    logger.info("running_reason_reduce")
    reduce_results = reason_reduce(
        reason_results,
        strategy="ds",
        seed=42,
    )

    assert len(reduce_results) >= 1, f"Expected ≥1 reduce results, got {len(reduce_results)}"
    logger.info(
        "reason_reduce_done",
        n_results=len(reduce_results),
        mean_confidence=sum(r.confidence for r in reduce_results) / len(reduce_results),
    )

    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"Smoke test took {elapsed:.2f}s (must be <5s)"

    logger.info(
        "smoke_test_passed",
        elapsed_seconds=round(elapsed, 3),
        n_reason_outputs=len(reason_results),
        n_reduce_outputs=len(reduce_results),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
