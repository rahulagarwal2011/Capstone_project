"""Cost-aware model routing.

Selects the cheapest model that can handle each document accurately.

TODO[Phase-4]: Full implementation with LightGBM difficulty classifier.
"""

from __future__ import annotations

from reason_reduce.ingestion.batch import Doc
from reason_reduce.monitoring.logger import get_logger

logger = get_logger(__name__)


class CostAwareRouter:
    """Routes documents to models based on predicted difficulty.

    Two-stage routing:
    1. Difficulty classifier (CPU, fast): predicts P(easy | doc)
    2. Routing policy: maps difficulty to model choice

    TODO[Phase-4]: Replace with trained LightGBM classifier.
    """

    def __init__(
        self,
        threshold_easy: float = 0.8,
        threshold_medium: float = 0.4,
    ) -> None:
        self._threshold_easy = threshold_easy
        self._threshold_medium = threshold_medium

    def route(self, doc: Doc) -> str:
        """Select a model for the given document.

        Current heuristic: route by document length.
        TODO[Phase-4]: Replace with learned classifier.

        Args:
            doc: Document to route.

        Returns:
            Model ID string.
        """
        n_tokens = len(doc.text.split())

        if n_tokens < 200:
            return "mistral-7b"
        elif n_tokens < 500:
            return "llama-3.1-8b"
        else:
            return "llama-3.1-70b"
