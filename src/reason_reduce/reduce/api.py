"""Public API for the reason_reduce() operator.

reason_reduce() is the aggregation operator that replaces reduce() in MapReduce.
It combines multiple ReasonOutputs using Dempster-Shafer evidence combination,
producing posterior confidence scores.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from reason_reduce.monitoring.logger import get_logger
from reason_reduce.reason.worker import ReasonOutput
from reason_reduce.reduce.aggregator import aggregate
from reason_reduce.reduce.conflict import ConflictPolicy
from reason_reduce.reduce.consensus import ConsensusResult

logger = get_logger(__name__)


def reason_reduce(
    reasoned: list[ReasonOutput],
    key_fn: Callable[[ReasonOutput], str] | None = None,
    strategy: Literal["majority", "ds", "bayesian"] = "ds",
    conflict_policy: ConflictPolicy = ConflictPolicy.ESCALATE,
    seed: int = 42,
) -> list[ConsensusResult]:
    """Aggregate ReasonOutputs using probabilistic combination.

    This is the core reduce operator. It groups outputs by key,
    then applies Dempster-Shafer (or alternative) aggregation
    to produce consensus results with propagated uncertainty.

    Confidence Semantics:
        Output confidence represents the posterior belief after combining
        evidence from multiple workers. Higher conflict (K) between workers
        reduces posterior confidence.

    Args:
        reasoned: Output from reason() — list of ReasonOutput 4-tuples.
        key_fn: Function to extract grouping key. Defaults to output.key.
        strategy: Aggregation strategy ("majority", "ds", "bayesian").
        conflict_policy: How to handle high-conflict situations.
        seed: Random seed for reproducibility.

    Returns:
        List of ConsensusResult objects, one per unique key.
    """
    if not reasoned:
        return []

    if key_fn is None:
        key_fn = lambda o: o.key  # noqa: E731

    results = aggregate(reasoned, key_fn=key_fn, strategy=strategy)

    logger.info(
        "reason_reduce_complete",
        n_inputs=len(reasoned),
        n_results=len(results),
        strategy=strategy,
        mean_confidence=sum(r.confidence for r in results) / max(len(results), 1),
    )

    return results
