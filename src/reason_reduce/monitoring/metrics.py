"""Prometheus metrics exporters for Reason-Reduce."""

from __future__ import annotations

from prometheus_client import Counter, Histogram, Gauge

REASON_REQUESTS_TOTAL = Counter(
    "reason_reduce_reason_requests_total",
    "Total reason() calls",
    ["model", "task", "status"],
)

REASON_LATENCY = Histogram(
    "reason_reduce_reason_latency_seconds",
    "Latency of reason() calls",
    ["model", "task"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

REASON_CONFIDENCE = Histogram(
    "reason_reduce_confidence",
    "Distribution of confidence scores from reason()",
    ["model", "task"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)

REDUCE_CONFLICT_MASS = Histogram(
    "reason_reduce_conflict_mass",
    "Dempster-Shafer conflict mass K during reduce()",
    ["strategy"],
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0),
)

ACTIVE_WORKERS = Gauge(
    "reason_reduce_active_workers",
    "Number of active Ray workers",
)

MODEL_ROUTING = Counter(
    "reason_reduce_model_routing_total",
    "Model routing decisions by the cost-aware router",
    ["target_model", "reason"],
)
