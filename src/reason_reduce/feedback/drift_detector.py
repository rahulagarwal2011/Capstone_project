"""Confidence distribution drift detection.

Monitors KL-divergence between current confidence distribution and
a rolling baseline to trigger retraining.

TODO[Phase-4]: Full implementation with KL threshold.
"""

from __future__ import annotations

from reason_reduce.monitoring.logger import get_logger

logger = get_logger(__name__)
