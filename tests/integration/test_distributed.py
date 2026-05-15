"""Integration test: Ray distributed execution."""

from __future__ import annotations

import pytest

from reason_reduce.ingestion.batch import Doc
from reason_reduce.reason.api import reason
from reason_reduce.reason.worker import TaskSpec


@pytest.fixture
def sample_docs() -> list[Doc]:
    return [
        Doc(id=str(i), text=f"Patient {i} was prescribed drug_{i % 5} for condition_{i % 3}.")
        for i in range(20)
    ]


class TestDistributedReason:
    """Tests for Ray-distributed reason() execution."""

    @pytest.mark.integration
    def test_distributed_matches_local(self, sample_docs: list[Doc]) -> None:
        """Distributed mode produces same number of results as local."""
        pytest.importorskip("ray")
        task = TaskSpec(task_type="ner")

        local_results = reason(sample_docs, task=task, model="mock", n_partitions=2, seed=42)
        distributed_results = reason(
            sample_docs,
            task=task,
            model="mock",
            n_partitions=2,
            seed=42,
            distributed=True,
            n_workers=2,
        )

        assert len(distributed_results) == len(local_results)
        assert all(r.confidence > 0 for r in distributed_results)

    @pytest.mark.integration
    def test_distributed_four_workers(self, sample_docs: list[Doc]) -> None:
        """Four workers can process documents in parallel."""
        pytest.importorskip("ray")
        results = reason(
            sample_docs,
            task=TaskSpec(task_type="ner"),
            model="mock",
            n_partitions=4,
            seed=42,
            distributed=True,
            n_workers=4,
        )

        assert len(results) == 20
        for r in results:
            assert r.doc_id != ""
            assert r.confidence >= 0.0
