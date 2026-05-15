"""OpenTelemetry tracing for Reason-Reduce.

Provides distributed tracing with correlation IDs that propagate across
Ray actor boundaries. Spans capture partition_id, confidence, model_used.
"""

from __future__ import annotations

import os
from contextvars import ContextVar

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from reason_reduce.monitoring.logger import get_logger

logger = get_logger(__name__)

_provider: TracerProvider | None = None
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


def init_tracing(
    service_name: str = "reason-reduce",
    export_to_console: bool = False,
) -> TracerProvider:
    """Initialize OpenTelemetry tracing.

    Args:
        service_name: Name of the service for trace identification.
        export_to_console: If True, print spans to stdout (dev mode).

    Returns:
        The configured TracerProvider.
    """
    global _provider
    if _provider is not None:
        return _provider

    resource = Resource.create({"service.name": service_name})
    _provider = TracerProvider(resource=resource)

    if export_to_console or os.environ.get("RR_TRACE_CONSOLE"):
        _provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            _provider.add_span_processor(SimpleSpanProcessor(exporter))
            logger.info("otlp_tracing_enabled", endpoint=otlp_endpoint)
        except ImportError:
            logger.info("otlp_exporter_not_installed", fallback="console_only")

    trace.set_tracer_provider(_provider)
    return _provider


def get_tracer(name: str) -> trace.Tracer:
    """Get a tracer instance.

    Args:
        name: Instrumentation name, typically __name__.

    Returns:
        An OpenTelemetry Tracer.
    """
    if _provider is None:
        init_tracing()
    return trace.get_tracer(name)


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID for the current context.

    Propagated across Ray actor calls via structlog context.
    """
    _correlation_id.set(correlation_id)
    import structlog

    structlog.contextvars.bind_contextvars(correlation_id=correlation_id)


def get_correlation_id() -> str:
    """Get the current correlation ID."""
    return _correlation_id.get()
