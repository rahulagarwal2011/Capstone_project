"""Multi-worker consensus mechanisms.

Provides three strategies for combining multiple ReasonOutputs for the same key:
- Confidence-weighted majority vote (baseline)
- Dempster-Shafer combination (ours)
- Bayesian model averaging (ablation)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from reason_reduce.reason.worker import ReasonOutput
from reason_reduce.reduce.dempster_shafer import (
    MassFunction,
    combine_multiple,
    reason_output_to_mass,
    combine_murphy,
    CombinationRule,
)
from reason_reduce.monitoring.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ConsensusResult:
    """Result of aggregating multiple ReasonOutputs.

    Attributes:
        key: The aggregation key.
        value: Consensus value.
        confidence: Posterior confidence after aggregation.
        n_sources: Number of ReasonOutputs combined.
        conflict_mass: Dempster-Shafer conflict mass (0 if not using DS).
        strategy: Which consensus strategy was used.
        trace: Aggregation trace for debugging.
    """

    key: str
    value: str
    confidence: float
    n_sources: int
    conflict_mass: float = 0.0
    strategy: str = "ds"
    trace: str = ""


def reach_consensus(
    outputs: list[ReasonOutput],
    strategy: Literal["majority", "ds", "bayesian"] = "ds",
    frame: frozenset[str] | None = None,
) -> ConsensusResult:
    """Combine multiple ReasonOutputs into a single consensus.

    Args:
        outputs: List of outputs for the same logical key.
        strategy: Aggregation strategy.
        frame: Frame of discernment for DS (auto-inferred from values if None).

    Returns:
        ConsensusResult with aggregated confidence.
    """
    if not outputs:
        return ConsensusResult(key="", value="", confidence=0.0, n_sources=0)

    if strategy == "majority":
        return _majority_vote(outputs)
    elif strategy == "ds":
        return _dempster_shafer(outputs, frame)
    elif strategy == "bayesian":
        return _bayesian_average(outputs)
    else:
        msg = f"Unknown strategy: {strategy}"
        raise ValueError(msg)


def _majority_vote(outputs: list[ReasonOutput]) -> ConsensusResult:
    """Confidence-weighted majority vote."""
    vote_weights: dict[str, float] = {}
    for o in outputs:
        if o.value is not None:
            vote_weights[o.value] = vote_weights.get(o.value, 0.0) + o.confidence

    if not vote_weights:
        return ConsensusResult(
            key=outputs[0].key, value="", confidence=0.0,
            n_sources=len(outputs), strategy="majority",
        )

    winner = max(vote_weights, key=vote_weights.get)  # type: ignore[arg-type]
    total_weight = sum(vote_weights.values())
    confidence = vote_weights[winner] / total_weight if total_weight > 0 else 0.0

    return ConsensusResult(
        key=outputs[0].key,
        value=winner,
        confidence=confidence,
        n_sources=len(outputs),
        strategy="majority",
    )


def _dempster_shafer(
    outputs: list[ReasonOutput], frame: frozenset[str] | None
) -> ConsensusResult:
    """Dempster-Shafer evidence combination."""
    if frame is None:
        values = {o.value for o in outputs if o.value is not None}
        frame = frozenset(values) if values else frozenset(["unknown"])

    mass_functions = [
        reason_output_to_mass(o.value, o.confidence, frame)
        for o in outputs
        if o.confidence > 0.0
    ]

    if not mass_functions:
        return ConsensusResult(
            key=outputs[0].key, value="", confidence=0.0,
            n_sources=len(outputs), strategy="ds",
        )

    if len(mass_functions) == 1:
        mf = mass_functions[0]
        best_singleton = max(
            ((s, m) for s, m in mf.masses.items() if len(s) == 1),
            key=lambda x: x[1],
            default=(frozenset(), 0.0),
        )
        value = next(iter(best_singleton[0])) if best_singleton[0] else ""
        return ConsensusResult(
            key=outputs[0].key, value=value, confidence=best_singleton[1],
            n_sources=len(outputs), conflict_mass=0.0, strategy="ds",
        )

    combined, max_k = combine_multiple(mass_functions)

    best_singleton = max(
        ((s, m) for s, m in combined.masses.items() if len(s) == 1),
        key=lambda x: x[1],
        default=(frozenset(), 0.0),
    )
    value = next(iter(best_singleton[0])) if best_singleton[0] else ""

    return ConsensusResult(
        key=outputs[0].key,
        value=value,
        confidence=best_singleton[1],
        n_sources=len(outputs),
        conflict_mass=max_k,
        strategy="ds",
    )


def _bayesian_average(outputs: list[ReasonOutput]) -> ConsensusResult:
    """Bayesian model averaging with uniform Dirichlet prior.

    TODO[Phase-3]: Full implementation with proper Dirichlet posterior.
    Currently equivalent to normalized confidence weighting.
    """
    return _majority_vote(outputs)
