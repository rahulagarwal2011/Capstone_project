"""Smoke test: full pipeline small-scale validation."""

from __future__ import annotations

from reason_reduce.ingestion.batch import Doc
from reason_reduce.reason.api import reason
from reason_reduce.reason.worker import TaskSpec
from reason_reduce.reduce.api import reason_reduce


class TestFullPipelineSmall:
    """Full pipeline test at small scale (no GPU, no external services)."""

    def test_10_docs_end_to_end(self) -> None:
        """10 docs → reason() → reason_reduce() → assert outputs exist."""
        drugs = ["aspirin", "ibuprofen", "paracetamol"]
        docs = [
            Doc(id=str(i), text=f"Patient received {drugs[i % 3]} for treatment.")
            for i in range(10)
        ]

        reason_results = reason(
            docs,
            task=TaskSpec(task_type="ner"),
            model="mock",
            n_partitions=2,
            seed=42,
        )

        assert len(reason_results) == 10
        for r in reason_results:
            assert r.confidence >= 0.0
            assert r.doc_id != ""

        reduce_results = reason_reduce(
            reason_results,
            strategy="ds",
            seed=42,
        )

        assert len(reduce_results) > 0
        for r in reduce_results:
            assert r.confidence >= 0.0
            assert r.n_sources > 0
