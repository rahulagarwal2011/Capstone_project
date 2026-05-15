"""Conflict resolution and escalation logic.

Decision tree for resolving conflicts when Dempster-Shafer conflict mass K is high.

TODO[Phase-3]: Full implementation with domain rules and HITL queue.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from reason_reduce.monitoring.logger import get_logger

logger = get_logger(__name__)


class ConflictPolicy(Enum):
    """Policy for handling high-conflict situations."""

    ESCALATE = "escalate"
    DOMAIN_RULES = "domain_rules"
    HUMAN_IN_LOOP = "human_in_loop"
    MARK_UNRESOLVED = "mark_unresolved"


@dataclass(frozen=True)
class ConflictResolution:
    """Result of conflict resolution.

    Attributes:
        resolved: Whether the conflict was resolved.
        value: Resolved value (if resolved).
        method: Which resolution method succeeded.
        trace: Explanation of the resolution.
    """

    resolved: bool
    value: str | None
    method: str
    trace: str


def resolve_conflict(
    candidates: list[tuple[str, float]],
    policy: ConflictPolicy = ConflictPolicy.ESCALATE,
    conflict_mass: float = 0.0,
) -> ConflictResolution:
    """Attempt to resolve a high-conflict aggregation.

    Decision tree (first that fires wins):
    1. Domain rule check
    2. Escalate to larger model
    3. Human-in-the-loop
    4. Mark as unresolved

    TODO[Phase-3]: Implement each branch fully.

    Args:
        candidates: List of (value, confidence) pairs in conflict.
        policy: Resolution policy to apply.
        conflict_mass: The Dempster-Shafer conflict mass K.

    Returns:
        ConflictResolution describing the outcome.
    """
    logger.info(
        "conflict_resolution_triggered",
        n_candidates=len(candidates),
        conflict_mass=conflict_mass,
        policy=policy.value,
    )

    if policy == ConflictPolicy.MARK_UNRESOLVED:
        return ConflictResolution(
            resolved=False,
            value=None,
            method="mark_unresolved",
            trace=f"Unresolved conflict with K={conflict_mass:.3f}",
        )

    # Fallback: pick highest confidence candidate
    if candidates:
        best = max(candidates, key=lambda x: x[1])
        return ConflictResolution(
            resolved=True,
            value=best[0],
            method="highest_confidence_fallback",
            trace=f"Resolved via highest confidence ({best[1]:.3f})",
        )

    return ConflictResolution(
        resolved=False,
        value=None,
        method="no_candidates",
        trace="No candidates to resolve",
    )
