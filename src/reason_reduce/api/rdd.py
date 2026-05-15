"""ReasonRDD: A Spark-RDD-like lazy wrapper over reasoning results.

Provides familiar functional API (collect, filter, map, take) over
Ray ObjectRefs holding ReasonOutput or ConsensusResult objects.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

T = TypeVar("T")


class ReasonRDD(Generic[T]):
    """Lazy wrapper around a collection of reasoning results.

    Mirrors Spark RDD semantics: transformations are lazy,
    actions (collect, take, count) trigger execution.

    Args:
        data: The underlying data (list for now; Ray ObjectRefs in production).
    """

    def __init__(self, data: list[T]) -> None:
        self._data = data

    def collect(self) -> list[T]:
        """Materialize all results.

        Returns:
            List of all elements.
        """
        return list(self._data)

    def take(self, n: int) -> list[T]:
        """Take the first n elements.

        Args:
            n: Number of elements to take.

        Returns:
            List of up to n elements.
        """
        return self._data[:n]

    def filter(self, predicate: Callable[[T], bool]) -> ReasonRDD[T]:
        """Filter elements by a predicate (lazy).

        Args:
            predicate: Function returning True for elements to keep.

        Returns:
            New ReasonRDD with filtered elements.
        """
        return ReasonRDD([x for x in self._data if predicate(x)])

    def map(self, fn: Callable[[T], T]) -> ReasonRDD[T]:
        """Apply a function to each element (lazy).

        Args:
            fn: Transformation function.

        Returns:
            New ReasonRDD with transformed elements.
        """
        return ReasonRDD([fn(x) for x in self._data])

    def count(self) -> int:
        """Count elements.

        Returns:
            Number of elements.
        """
        return len(self._data)

    def __iter__(self) -> Iterator[T]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)
