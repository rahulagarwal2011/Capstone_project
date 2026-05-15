"""Unit tests for Dempster-Shafer evidence combination."""

from __future__ import annotations

import pytest

from reason_reduce.reduce.dempster_shafer import (
    MassFunction,
    combine_multiple,
    reason_output_to_mass,
)


class TestMassFunction:
    """Tests for the MassFunction class."""

    def test_valid_construction(self) -> None:
        frame = frozenset(["A", "B", "C"])
        masses = {frozenset(["A"]): 0.6, frame: 0.4}
        mf = MassFunction(masses=masses, frame=frame)
        assert abs(sum(mf.masses.values()) - 1.0) < 1e-6

    def test_invalid_sum_raises(self) -> None:
        frame = frozenset(["A", "B"])
        with pytest.raises(ValueError, match="sum to 1.0"):
            MassFunction(masses={frozenset(["A"]): 0.5}, frame=frame)

    def test_belief_singleton(self) -> None:
        frame = frozenset(["A", "B", "C"])
        masses = {frozenset(["A"]): 0.6, frame: 0.4}
        mf = MassFunction(masses=masses, frame=frame)
        assert abs(mf.belief(frozenset(["A"])) - 0.6) < 1e-6

    def test_plausibility_ge_belief(self) -> None:
        frame = frozenset(["A", "B", "C"])
        masses = {frozenset(["A"]): 0.6, frame: 0.4}
        mf = MassFunction(masses=masses, frame=frame)
        hyp = frozenset(["A"])
        assert mf.plausibility(hyp) >= mf.belief(hyp)

    def test_combination_basic(self) -> None:
        frame = frozenset(["A", "B"])
        mf1 = MassFunction(masses={frozenset(["A"]): 0.7, frame: 0.3}, frame=frame)
        mf2 = MassFunction(masses={frozenset(["A"]): 0.8, frame: 0.2}, frame=frame)
        combined, k = mf1.combine(mf2)
        assert combined.masses[frozenset(["A"])] > 0.7
        assert k >= 0.0

    def test_combination_total_conflict_raises(self) -> None:
        frame = frozenset(["A", "B"])
        mf1 = MassFunction(masses={frozenset(["A"]): 1.0}, frame=frame)
        mf2 = MassFunction(masses={frozenset(["B"]): 1.0}, frame=frame)
        with pytest.raises(ValueError, match="Total conflict"):
            mf1.combine(mf2)

    def test_shafer_burglar_example(self) -> None:
        """Reproduce Shafer's textbook 'burglar' example.

        Two witnesses observe a burglar. Witness 1: P(burglar)=0.8, P(unknown)=0.2.
        Witness 2: P(burglar)=0.5, P(unknown)=0.5.
        Combined: P(burglar) should be higher than either individual.
        """
        frame = frozenset(["burglar", "not_burglar"])
        w1 = MassFunction(masses={frozenset(["burglar"]): 0.8, frame: 0.2}, frame=frame)
        w2 = MassFunction(masses={frozenset(["burglar"]): 0.5, frame: 0.5}, frame=frame)
        combined, k = w1.combine(w2)
        assert combined.masses[frozenset(["burglar"])] > 0.8
        assert k < 1.0


class TestCombineMultiple:
    """Tests for combining multiple mass functions."""

    def test_single_function(self) -> None:
        frame = frozenset(["A", "B"])
        mf = MassFunction(masses={frozenset(["A"]): 0.7, frame: 0.3}, frame=frame)
        result, k = combine_multiple([mf])
        assert result.masses == mf.masses
        assert k == 0.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            combine_multiple([])

    def test_many_agreeing_converges(self) -> None:
        frame = frozenset(["A", "B"])
        mfs = [
            MassFunction(masses={frozenset(["A"]): 0.6, frame: 0.4}, frame=frame) for _ in range(5)
        ]
        result, _ = combine_multiple(mfs)
        assert result.masses[frozenset(["A"])] > 0.9


class TestReasonOutputToMass:
    """Tests for converting ReasonOutput confidences to mass functions."""

    def test_high_confidence(self) -> None:
        frame = frozenset(["A", "B", "C"])
        mf = reason_output_to_mass("A", 0.9, frame)
        assert abs(mf.masses[frozenset(["A"])] - 0.9) < 1e-6
        assert abs(mf.masses[frame] - 0.1) < 1e-6

    def test_zero_confidence_is_ignorance(self) -> None:
        frame = frozenset(["A", "B", "C"])
        mf = reason_output_to_mass("A", 0.0, frame)
        assert mf.masses[frame] == 1.0

    def test_none_value_is_ignorance(self) -> None:
        frame = frozenset(["A", "B", "C"])
        mf = reason_output_to_mass(None, 0.8, frame)
        assert mf.masses[frame] == 1.0
