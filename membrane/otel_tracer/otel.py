"""OpenTelemetry tracer + OTLP exporter (Phase 3.2.4).

The v3.0.0 release adopts OpenTelemetry as the canonical
distributed tracing layer. The :func:`configure` helper reads
``OTEL_EXPORTER_OTLP_ENDPOINT`` from the environment and
installs a :class:`TracerProvider` with an OTLP exporter when
the env var is set; absent the env var the helper installs the
:class:`NoOpTracer` so single-process tests continue to work.

The :func:`instrument_async` decorator wraps a coroutine so the
trace span covers the entire coroutine lifetime; the
:class:`MembraneSpan` context manager is the synchronous form.
Both spans land under the ``membrane`` instrumentation scope
and carry the standard service attributes
(``service.name``, ``service.version``).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


SERVICE_NAME: str = "membrane"
"""The OTel ``service.name`` value."""


def _noop_tracer() -> Any:
    """Return a tracer whose spans are inert."""
    from opentelemetry.trace import NoOpTracer

    return NoOpTracer()


class TracerFactory:
    """Process-wide OTel tracer factory.

    Attributes:
        provider: The configured :class:`TracerProvider`, or
            ``None`` when no exporter is configured.
        endpoint: The OTLP endpoint derived from the
            environment. ``None`` means the system runs with
            a no-op tracer.
    """

    def __init__(self) -> None:
        self.provider: Any | None = None
        self.endpoint: str | None = None
        self._tracer: Any | None = None

    def configure(self, endpoint: str | None = None) -> Any:
        """Configure the tracer from an OTLP endpoint.

        Args:
            endpoint: OTLP endpoint. Defaults to the value of
                ``OTEL_EXPORTER_OTLP_ENDPOINT``; ``None`` when
                the env var is unset.

        Returns:
            The tracer instance.
        """
        target = endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        if not target:
            self._tracer = _noop_tracer()
            return self._tracer
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError as exc:  # pragma: no cover - import guard
            logger.warning("OTel SDK not installed: %s", exc)
            self._tracer = _noop_tracer()
            return self._tracer
        resource = Resource.create({"service.name": SERVICE_NAME, "service.version": "3.0.0"})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=target)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        self.provider = provider
        self.endpoint = target
        self._tracer = trace.get_tracer(SERVICE_NAME)
        return self._tracer

    @property
    def tracer(self) -> Any:
        """Return the configured tracer.

        Returns:
            Either the OTel tracer or the no-op tracer.
        """
        if self._tracer is None:
            self._tracer = _noop_tracer()
        return self._tracer


_default_factory: TracerFactory | None = None


def get_default_tracer() -> Any:
    """Return the process-wide default tracer.

    Returns:
        The :class:`TracerFactory` may be configured or not;
        if it is, returns the configured tracer; otherwise
        installs the no-op tracer the first time it is asked.
    """
    global _default_factory
    if _default_factory is None:
        _default_factory = TracerFactory()
        _default_factory.configure()
    return _default_factory.tracer


@contextmanager
def membrane_span(name: str, **attributes: Any) -> Iterator[Any]:
    """Open a synchronous :class:`membrane` span.

    Args:
        name: Span name (e.g., ``"transfer.kv"``).
        **attributes: Span attributes recorded on the span
            events.

    Yields:
        The active span, or ``None`` when the no-op tracer is
        configured.
    """
    tracer = get_default_tracer()
    if hasattr(tracer, "start_as_current_span"):
        with tracer.start_as_current_span(name) as span:
            for key, value in attributes.items():
                if span is not None and hasattr(span, "set_attribute"):
                    span.set_attribute(key, value)
            yield span
    else:
        yield None


__all__ = [
    "SERVICE_NAME",
    "TracerFactory",
    "get_default_tracer",
    "membrane_span",
]
