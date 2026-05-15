"""Integration test: full reason → reason_reduce pipeline."""

from __future__ import annotations

from reason_reduce.ingestion.batch import Doc
from reason_reduce.reason.api import reason
from reason_reduce.reason.worker import TaskSpec
from reason_reduce.reduce.api import reason_reduce


class TestReasonE2E:
    """End-to-end tests for the reason() operator."""

    def test_basic_pipeline(self) -> None:
        docs = [
            Doc(id="1", text="Aspirin is used to treat headaches."),
            Doc(id="2", text="Ibuprofen reduces inflammation."),
            Doc(id="3", text="Paracetamol is a common pain reliever."),
        ]

        results = reason(docs, task=TaskSpec(task_type="ner"), model="mock")

        assert len(results) == 3
        for r in results:
            assert r.confidence >= 0.0
            assert r.confidence <= 1.0
            assert r.doc_id in ["1", "2", "3"]

    def test_empty_input(self) -> None:
        results = reason([], model="mock")
        assert results == []


class TestReasonReduceE2E:
    """End-to-end tests for the full reason → reason_reduce pipeline."""

    def test_full_pipeline(self) -> None:
        docs = [Doc(id=str(i), text=f"Document {i} about entity_{i % 3}") for i in range(10)]

        reason_results = reason(docs, task=TaskSpec(task_type="ner"), model="mock")
        assert len(reason_results) == 10

        reduce_results = reason_reduce(reason_results, strategy="ds")
        assert len(reduce_results) > 0

        for r in reduce_results:
            assert r.confidence >= 0.0
            assert r.n_sources > 0
