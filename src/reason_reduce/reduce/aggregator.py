"""Hierarchical tree-reduction aggregator.

Reduces ReasonOutputs in a tree structure: local (per partition) then global.

TODO[Phase-3]: Full Ray-based tree reduction with configurable fanout.
"""

from __future__ import annotations

from reason_reduce.reason.worker import ReasonOutput
from reason_reduce.reduce.consensus import ConsensusResult, reach_consensus
from reason_reduce.monitoring.logger import get_logger

logger = get_logger(__name__)


def aggregate(
    outputs: list[ReasonOutput],
    key_fn: callable = lambda o: o.key,  # type: ignore[type-arg]
    strategy: str = "ds",
) -> list[ConsensusResult]:
    """Aggregate ReasonOutputs by key using tree reduction.

    Groups outputs by key_fn, then runs consensus within each group.

    Args:
        outputs: All ReasonOutputs from the reason() phase.
        key_fn: Function to extract grouping key from a ReasonOutput.
        strategy: Consensus strategy ("majority", "ds", "bayesian").

    Returns:
        List of ConsensusResults, one per unique key.
    """
    groups: dict[str, list[ReasonOutput]] = {}
    for output in outputs:
        key = key_fn(output)
        if key not in groups:
            groups[key] = []
        groups[key].append(output)

    results: list[ConsensusResult] = []
    for key, group_outputs in groups.items():
        result = reach_consensus(group_outputs, strategy=strategy)  # type: ignore[arg-type]
        results.append(result)

    logger.info(
        "aggregation_complete",
        n_inputs=len(outputs),
        n_groups=len(groups),
        n_results=len(results),
    )
    return results
