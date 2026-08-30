"""OpenTelemetry observability (Phase 3.2.4)."""

from membrane.otel_tracer.otel import (
    SERVICE_NAME,
    TracerFactory,
    get_default_tracer,
    membrane_span,
)

__all__ = [
    "SERVICE_NAME",
    "TracerFactory",
    "get_default_tracer",
    "membrane_span",
]
