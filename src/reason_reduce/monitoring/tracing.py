"""OpenTelemetry tracing for Reason-Reduce.

Spans are created for reason() and reason_reduce() operations with custom
attributes for partition ID, confidence, and model used.
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

_provider: TracerProvider | None = None


def init_tracing(service_name: str = "reason-reduce") -> TracerProvider:
    """Initialize OpenTelemetry tracing.

    Args:
        service_name: Name of the service for trace identification.

    Returns:
        The configured TracerProvider.
    """
    global _provider  # noqa: PLW0603
    if _provider is not None:
        return _provider

    _provider = TracerProvider()
    _provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
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
