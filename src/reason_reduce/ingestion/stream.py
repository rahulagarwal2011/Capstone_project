"""Kafka-compatible streaming interface.

TODO[Phase-4]: Implement streaming ingestion with confluent-kafka.
"""

from __future__ import annotations

from reason_reduce.monitoring.logger import get_logger

logger = get_logger(__name__)
