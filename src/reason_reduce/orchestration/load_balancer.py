"""Adaptive, complexity-aware load balancer.

TODO[Phase-4]: Full implementation with EMA latency tracking and
straggler mitigation.
"""

from __future__ import annotations

from reason_reduce.monitoring.logger import get_logger

logger = get_logger(__name__)
