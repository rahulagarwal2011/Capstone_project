"""Prometheus metrics HTTP server for Reason-Reduce.

Exposes a /metrics endpoint that Prometheus can scrape.
Start with `start_metrics_server()` — runs in a background thread.
"""

from __future__ import annotations

import threading

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

from reason_reduce.monitoring.logger import get_logger

logger = get_logger(__name__)

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

DOCS_PROCESSED = Counter(
    "reason_reduce_docs_processed_total",
    "Total documents processed",
    ["phase"],
)

_server_started = False
_server_lock = threading.Lock()


def start_metrics_server(port: int = 8000) -> None:
    """Start the Prometheus metrics HTTP server in a background thread.

    Safe to call multiple times — only starts once.

    Args:
        port: Port to serve /metrics on. Default 8000.
    """
    global _server_started
    with _server_lock:
        if _server_started:
            return
        try:
            start_http_server(port)
            _server_started = True
            logger.info(
                "metrics_server_started", port=port, endpoint=f"http://localhost:{port}/metrics"
            )
        except OSError as e:
            if "Address already in use" in str(e):
                logger.info("metrics_server_already_running", port=port)
                _server_started = True
            else:
                logger.error("metrics_server_failed", error=str(e))


def record_reason_request(
    model: str,
    task: str,
    status: str,
    latency_s: float,
    confidence: float,
) -> None:
    """Record metrics for a reason() request."""
    REASON_REQUESTS_TOTAL.labels(model=model, task=task, status=status).inc()
    REASON_LATENCY.labels(model=model, task=task).observe(latency_s)
    REASON_CONFIDENCE.labels(model=model, task=task).observe(confidence)
    DOCS_PROCESSED.labels(phase="reason").inc()


def record_reduce_conflict(strategy: str, conflict_mass: float) -> None:
    """Record conflict mass from a reduce() operation."""
    REDUCE_CONFLICT_MASS.labels(strategy=strategy).observe(conflict_mass)
    DOCS_PROCESSED.labels(phase="reduce").inc()
