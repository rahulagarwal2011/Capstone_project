"""Unit tests for the semantic partitioner."""

from __future__ import annotations

import pytest

from reason_reduce.ingestion.batch import Doc
from reason_reduce.reason.partitioner import partition_documents


@pytest.fixture
def sample_docs() -> list[Doc]:
    return [Doc(id=str(i), text=f"Document {i} about topic {i % 3}") for i in range(30)]


class TestPartitioner:
    """Tests for document partitioning."""

    def test_empty_docs_returns_empty(self) -> None:
        result = partition_documents([], n_partitions=3)
        assert result == []

    def test_single_doc_single_partition(self) -> None:
        docs = [Doc(id="1", text="hello world")]
        result = partition_documents(docs, n_partitions=1)
        assert len(result) == 1
        assert len(result[0].docs) == 1

    def test_n_partitions_respected(self, sample_docs: list[Doc]) -> None:
        result = partition_documents(sample_docs, n_partitions=5)
        assert len(result) == 5

    def test_all_docs_assigned(self, sample_docs: list[Doc]) -> None:
        result = partition_documents(sample_docs, n_partitions=3)
        total_docs = sum(len(p.docs) for p in result)
        assert total_docs == len(sample_docs)

    def test_deterministic_with_seed(self, sample_docs: list[Doc]) -> None:
        r1 = partition_documents(sample_docs, n_partitions=3, seed=42)
        r2 = partition_documents(sample_docs, n_partitions=3, seed=42)
        for p1, p2 in zip(r1, r2):
            assert [d.id for d in p1.docs] == [d.id for d in p2.docs]

    def test_more_partitions_than_docs(self) -> None:
        docs = [Doc(id=str(i), text=f"doc {i}") for i in range(3)]
        result = partition_documents(docs, n_partitions=10)
        assert len(result) == 3
