"""Integration test: reason_reduce operator end-to-end."""

from __future__ import annotations

from reason_reduce.reason.worker import ReasonOutput
from reason_reduce.reduce.api import reason_reduce


class TestReasonReduceIntegration:
    """Integration tests for reason_reduce()."""

    def test_multiple_workers_agree(self) -> None:
        outputs = [
            ReasonOutput(key="DRUG", value="aspirin", confidence=0.8, trace="", doc_id=str(i))
            for i in range(5)
        ]

        results = reason_reduce(outputs, strategy="ds")
        assert len(results) == 1
        assert results[0].value == "aspirin"
        assert results[0].confidence > 0.8

    def test_workers_disagree(self) -> None:
        outputs = [
            ReasonOutput(key="DRUG", value="aspirin", confidence=0.7, trace="", doc_id="1"),
            ReasonOutput(key="DRUG", value="aspirin", confidence=0.8, trace="", doc_id="2"),
            ReasonOutput(key="DRUG", value="ibuprofen", confidence=0.6, trace="", doc_id="3"),
        ]

        results = reason_reduce(outputs, strategy="ds")
        assert len(results) == 1
        assert results[0].value == "aspirin"

    def test_majority_strategy(self) -> None:
        outputs = [
            ReasonOutput(key="CAT", value="medical", confidence=0.9, trace="", doc_id="1"),
            ReasonOutput(key="CAT", value="legal", confidence=0.3, trace="", doc_id="2"),
            ReasonOutput(key="CAT", value="medical", confidence=0.7, trace="", doc_id="3"),
        ]

        results = reason_reduce(outputs, strategy="majority")
        assert len(results) == 1
        assert results[0].value == "medical"
