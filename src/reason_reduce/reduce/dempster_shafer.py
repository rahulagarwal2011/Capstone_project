"""Dempster-Shafer evidence combination.

Implements Dempster's rule of combination for mass functions over a
frame of discernment. Used to aggregate multiple workers' outputs
into a consensus result with propagated uncertainty.

For numerical stability with many sources (>10), log-space combination
is used via Yager's rule as a fallback option.

Reference: Shafer, "A Mathematical Theory of Evidence", 1976.
Reference: Smets, "The Transferable Belief Model", 1994.

### Thesis Note
When positioned in the thesis, emphasize: "while DS is well-known in sensor
fusion (Smets, 1994), its application to LLM output aggregation in a
data-parallel processing framework is, to our knowledge, novel."
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np

from reason_reduce.monitoring.logger import get_logger

logger = get_logger(__name__)


class CombinationRule(Enum):
    """Available combination rules for ablation."""

    DEMPSTER = "dempster"
    YAGER = "yager"
    MURPHY = "murphy"


@dataclass
class MassFunction:
    """A Dempster-Shafer mass function m: 2^Theta -> [0,1].

    The frame of discernment Theta is represented as a frozenset.
    Focal elements are subsets of Theta with non-zero mass.

    Confidence Semantics:
        Mass assigned to a singleton {x} represents direct evidence for x.
        Mass assigned to Theta represents ignorance (don't know).
        The sum of all masses must equal 1.0 (within floating-point tolerance).

    Attributes:
        masses: Mapping from subsets of Theta to mass values.
        frame: The full frame of discernment.
    """

    masses: dict[frozenset[str], float] = field(default_factory=dict)
    frame: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.masses:
            self.masses = {self.frame: 1.0}

        self.masses = {k: v for k, v in self.masses.items() if v > 1e-12}
        if not self.masses:
            self.masses = {self.frame: 1.0}

        total = sum(self.masses.values())
        if abs(total - 1.0) > 1e-6:
            msg = f"Mass function must sum to 1.0, got {total}"
            raise ValueError(msg)

    def belief(self, hypothesis: frozenset[str]) -> float:
        """Compute belief (lower probability) for a hypothesis.

        Bel(A) = sum of m(B) for all B that are subsets of A.

        Args:
            hypothesis: A subset of the frame.

        Returns:
            Belief value in [0, 1].
        """
        return sum(
            mass for subset, mass in self.masses.items()
            if subset <= hypothesis and len(subset) > 0
        )

    def plausibility(self, hypothesis: frozenset[str]) -> float:
        """Compute plausibility (upper probability) for a hypothesis.

        Pl(A) = sum of m(B) for all B that intersect A.

        Args:
            hypothesis: A subset of the frame.

        Returns:
            Plausibility value in [0, 1].
        """
        return sum(
            mass for subset, mass in self.masses.items()
            if subset & hypothesis and len(subset) > 0
        )

    def pignistic_probability(self) -> dict[str, float]:
        """Compute the pignistic probability transformation.

        Projects the mass function onto a probability distribution over
        singletons. Used for downstream decision-making.

        Reference: Smets, 1990.

        Returns:
            Dict mapping each element of the frame to its pignistic probability.
        """
        prob: dict[str, float] = {x: 0.0 for x in self.frame}
        for subset, mass in self.masses.items():
            if not subset:
                continue
            share = mass / len(subset)
            for element in subset:
                prob[element] += share
        return prob

    def combine(
        self,
        other: MassFunction,
        rule: CombinationRule = CombinationRule.DEMPSTER,
    ) -> tuple[MassFunction, float]:
        """Combine this mass function with another.

        Args:
            other: Another mass function over the same frame.
            rule: Combination rule to use.

        Returns:
            Tuple of (combined MassFunction, conflict mass K).

        Raises:
            ValueError: If frames don't match or K = 1.0 with Dempster's rule.
        """
        if self.frame != other.frame:
            msg = "Cannot combine mass functions over different frames"
            raise ValueError(msg)

        if rule == CombinationRule.DEMPSTER:
            return self._dempster_combine(other)
        elif rule == CombinationRule.YAGER:
            return self._yager_combine(other)
        elif rule == CombinationRule.MURPHY:
            msg = "Murphy's rule requires a list; use combine_murphy()"
            raise ValueError(msg)
        else:
            return self._dempster_combine(other)

    def _dempster_combine(self, other: MassFunction) -> tuple[MassFunction, float]:
        """Dempster's rule: normalize by 1/(1-K)."""
        combined_raw: dict[frozenset[str], float] = {}
        conflict = 0.0

        for a_set, a_mass in self.masses.items():
            for b_set, b_mass in other.masses.items():
                intersection = a_set & b_set
                product = a_mass * b_mass

                if not intersection:
                    conflict += product
                else:
                    combined_raw[intersection] = (
                        combined_raw.get(intersection, 0.0) + product
                    )

        if abs(conflict - 1.0) < 1e-10:
            msg = "Total conflict (K=1.0): evidence sources are completely contradictory"
            raise ValueError(msg)

        normalizer = 1.0 / (1.0 - conflict)
        normalized: dict[frozenset[str], float] = {
            subset: mass * normalizer
            for subset, mass in combined_raw.items()
            if mass * normalizer > 1e-12
        }

        if conflict >= 0.7:
            logger.warning("high_conflict", K=round(conflict, 4))
        elif conflict >= 0.3:
            logger.info("moderate_conflict", K=round(conflict, 4))

        result = MassFunction(masses=normalized, frame=self.frame)
        return result, conflict

    def _yager_combine(self, other: MassFunction) -> tuple[MassFunction, float]:
        """Yager's rule: conflict mass goes to Theta (no normalization).

        More conservative than Dempster's rule when conflict is high.
        Preserves total mass without normalization.
        """
        combined_raw: dict[frozenset[str], float] = {}
        conflict = 0.0

        for a_set, a_mass in self.masses.items():
            for b_set, b_mass in other.masses.items():
                intersection = a_set & b_set
                product = a_mass * b_mass

                if not intersection:
                    conflict += product
                else:
                    combined_raw[intersection] = (
                        combined_raw.get(intersection, 0.0) + product
                    )

        combined_raw[self.frame] = combined_raw.get(self.frame, 0.0) + conflict

        cleaned = {k: v for k, v in combined_raw.items() if v > 1e-12}
        result = MassFunction(masses=cleaned, frame=self.frame)
        return result, conflict


def combine_multiple(
    mass_functions: list[MassFunction],
    rule: CombinationRule = CombinationRule.DEMPSTER,
    log_space_threshold: int = 10,
) -> tuple[MassFunction, float]:
    """Combine multiple mass functions sequentially.

    For many sources (> log_space_threshold), automatically switches
    to Yager's rule to avoid numerical instability from repeated
    normalization.

    Args:
        mass_functions: List of mass functions to combine.
        rule: Combination rule to use.
        log_space_threshold: Switch to Yager above this many sources.

    Returns:
        Tuple of (final combined MassFunction, maximum conflict encountered).

    Raises:
        ValueError: If list is empty or frames don't match.
    """
    if not mass_functions:
        msg = "Cannot combine empty list of mass functions"
        raise ValueError(msg)

    if len(mass_functions) == 1:
        return mass_functions[0], 0.0

    effective_rule = rule
    if len(mass_functions) > log_space_threshold and rule == CombinationRule.DEMPSTER:
        logger.info(
            "switching_to_yager",
            reason="many_sources",
            n_sources=len(mass_functions),
        )
        effective_rule = CombinationRule.YAGER

    result = mass_functions[0]
    max_conflict = 0.0

    for mf in mass_functions[1:]:
        result, k = result.combine(mf, rule=effective_rule)
        max_conflict = max(max_conflict, k)

    return result, max_conflict


def combine_murphy(mass_functions: list[MassFunction]) -> tuple[MassFunction, float]:
    """Murphy's averaging rule for high-conflict scenarios.

    Averages all mass functions first, then combines the average with itself.
    Handles high-conflict better than Dempster's rule.

    Reference: Murphy, 2000.

    Args:
        mass_functions: List of mass functions to combine.

    Returns:
        Tuple of (combined MassFunction, estimated conflict).
    """
    if not mass_functions:
        msg = "Cannot combine empty list"
        raise ValueError(msg)

    if len(mass_functions) == 1:
        return mass_functions[0], 0.0

    frame = mass_functions[0].frame
    n = len(mass_functions)

    all_subsets: set[frozenset[str]] = set()
    for mf in mass_functions:
        all_subsets.update(mf.masses.keys())

    averaged: dict[frozenset[str], float] = {}
    for subset in all_subsets:
        total = sum(mf.masses.get(subset, 0.0) for mf in mass_functions)
        averaged[subset] = total / n

    cleaned = {k: v for k, v in averaged.items() if v > 1e-12}
    total = sum(cleaned.values())
    if abs(total - 1.0) > 1e-6:
        cleaned = {k: v / total for k, v in cleaned.items()}

    avg_mf = MassFunction(masses=cleaned, frame=frame)

    result = avg_mf
    max_k = 0.0
    for _ in range(n - 1):
        result, k = result.combine(avg_mf, rule=CombinationRule.DEMPSTER)
        max_k = max(max_k, k)

    return result, max_k


def reason_output_to_mass(
    value: str | None,
    confidence: float,
    frame: frozenset[str],
) -> MassFunction:
    """Convert a ReasonOutput's confidence into a mass function.

    For a classification with classes {A, B, C}, an output with value=A
    and confidence=0.7 becomes m({A}) = 0.7, m({A,B,C}) = 0.3.

    The "I don't know" residual goes to the full frame (vacuous evidence).

    Args:
        value: The predicted value (maps to a singleton in the frame).
        confidence: Confidence score in [0, 1].
        frame: Full frame of discernment.

    Returns:
        A MassFunction representing the evidence.
    """
    if value is None or confidence <= 0.0:
        return MassFunction(masses={frame: 1.0}, frame=frame)

    if value not in frame:
        return MassFunction(masses={frame: 1.0}, frame=frame)

    singleton = frozenset([value])

    if singleton == frame:
        return MassFunction(masses={frame: 1.0}, frame=frame)

    confidence = min(1.0, max(0.0, confidence))
    masses: dict[frozenset[str], float] = {
        singleton: confidence,
        frame: 1.0 - confidence,
    }
    return MassFunction(masses=masses, frame=frame)


def compute_expected_calibration_error(
    confidences: list[float],
    correct: list[bool],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Reference: Guo et al. 2017.

    Args:
        confidences: Predicted confidence scores.
        correct: Whether each prediction was correct.
        n_bins: Number of bins for calibration histogram.

    Returns:
        ECE value in [0, 1]. Lower is better.
    """
    if not confidences:
        return 0.0

    conf_arr = np.array(confidences)
    correct_arr = np.array(correct, dtype=float)
    bin_edges = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    total = len(confidences)

    for i in range(n_bins):
        mask = (conf_arr > bin_edges[i]) & (conf_arr <= bin_edges[i + 1])
        if not mask.any():
            continue
        bin_size = mask.sum()
        bin_confidence = conf_arr[mask].mean()
        bin_accuracy = correct_arr[mask].mean()
        ece += (bin_size / total) * abs(bin_accuracy - bin_confidence)

    return float(ece)
