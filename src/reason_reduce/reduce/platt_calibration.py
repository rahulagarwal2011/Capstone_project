"""Platt calibration for LLM confidence scores.

Learns a sigmoid mapping from raw LLM logprobs/confidences to calibrated
probabilities. Reduces Expected Calibration Error (ECE).

Reference: Guo et al. 2017, "On Calibration of Modern Neural Networks".

TODO[Phase-3]: Full implementation with sklearn LogisticRegression.
"""

from __future__ import annotations

from dataclasses import dataclass

from reason_reduce.monitoring.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PlattCalibrator:
    """Platt scaling calibrator: sigmoid(a * score + b).

    Trained on held-out labeled data to map raw confidences
    to calibrated probabilities.

    TODO[Phase-3]: Implement fit() with sklearn.
    """

    a: float = 1.0
    b: float = 0.0
    is_fitted: bool = False

    def calibrate(self, raw_score: float) -> float:
        """Apply Platt scaling to a raw confidence score.

        Args:
            raw_score: Raw confidence in [0, 1].

        Returns:
            Calibrated probability in [0, 1].
        """
        if not self.is_fitted:
            return raw_score

        import math

        logit = self.a * raw_score + self.b
        return 1.0 / (1.0 + math.exp(-logit))
