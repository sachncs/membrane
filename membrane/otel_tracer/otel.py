"""OpenTelemetry tracing: spans for requests, writes, and replication.

Tracing is off until :meth:`Tracing.configure` runs (``membrane serve
--otel-endpoint URL``, or ``OTEL_EXPORTER_OTLP_ENDPOINT``). Until then
every helper here is a cheap no-op that never imports OpenTelemetry, so
the ``otel`` extra stays optional.

When on, each HTTP request gets a server span (tagged with its request
ID), strong writes and hand-offs get child spans, and peer calls carry
the W3C ``traceparent`` header, so one write traces across nodes.
"""

import logging
import os
from collections.abc import Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

SERVICE_NAME: str = "membrane"
"""The OTel ``service.name`` value."""


class Tracing:
    """Process-wide tracer state.

    Attributes:
        provider: The SDK ``TracerProvider`` once configured.
        endpoint: The OTLP endpoint in use, if any.
    """

    def __init__(self) -> None:
        """Start unconfigured: every span is a no-op."""
        self.provider: Any = None
        self.endpoint = ""
        self.__tracer: Any = None

    @property
    def enabled(self) -> bool:
        """Whether spans are recorded."""
        return self.__tracer is not None

    def configure(self, endpoint: str = "", node_id: str = "", exporter: Any = None) -> bool:
        """Install a tracer provider that exports spans.

        Args:
            endpoint: OTLP/gRPC endpoint; defaults to
                ``OTEL_EXPORTER_OTLP_ENDPOINT``.
            node_id: Recorded as ``service.instance.id``.
            exporter: A span exporter to use instead of OTLP (tests).

        Returns:
            bool: True when tracing is on; False when no endpoint is set or
            the ``otel`` extra is not installed.
        """
        target = endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
        if not target and exporter is None:
            return False
        try:
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
        except ImportError as exc:
            logger.warning("tracing requested but the OpenTelemetry SDK is missing (install membrane[otel]): %s", exc)
            return False
        from membrane import __version__

        resource = Resource.create(
            {"service.name": SERVICE_NAME, "service.version": __version__, "service.instance.id": node_id}
        )
        provider = TracerProvider(resource=resource)
        if exporter is not None:
            provider.add_span_processor(SimpleSpanProcessor(exporter))
        else:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=target)))
        self.provider = provider
        self.endpoint = target
        self.__tracer = provider.get_tracer(SERVICE_NAME)
        logger.info("tracing to %s", target or "a custom exporter")
        return True

    def shutdown(self) -> None:
        """Flush pending spans and stop recording."""
        if self.provider is not None:
            self.provider.shutdown()
        self.provider = None
        self.__tracer = None

    @contextmanager
    def span(
        self, name: str, kind: str = "internal", context: Any = None, attributes: Mapping[str, Any] | None = None
    ) -> Iterator[Any]:
        """Open a span, or do nothing when tracing is off.

        Args:
            name: Span name (e.g. ``"quorum.replicate"``).
            kind: ``"internal"``, ``"server"``, or ``"client"``.
            context: Parent context (from :meth:`extract`); the current
                span when ``None``.
            attributes: Span attributes.

        Yields:
            Any: The span, or ``None`` when tracing is off.
        """
        if self.__tracer is None:
            yield None
            return
        from opentelemetry.trace import SpanKind

        span_kind = {"server": SpanKind.SERVER, "client": SpanKind.CLIENT}.get(kind, SpanKind.INTERNAL)
        with self.__tracer.start_as_current_span(name, context=context, kind=span_kind) as span:
            for key, value in (attributes or {}).items():
                span.set_attribute(key, value)
            yield span

    def inject(self, headers: MutableMapping[str, str]) -> None:
        """Add trace-propagation headers (``traceparent``) for an outgoing call.

        Args:
            headers: Outgoing request headers, updated in place.
        """
        if self.__tracer is not None:
            from opentelemetry.propagate import inject

            inject(headers)

    def extract(self, headers: MutableMapping[str, str]) -> Any:
        """Return the remote parent context carried by incoming headers.

        Args:
            headers: Incoming request headers (lower-case names).

        Returns:
            Any: The parent context, or ``None`` when tracing is off.
        """
        if self.__tracer is None:
            return None
        from opentelemetry.propagate import extract

        return extract(headers)


#: The process-wide tracing state.
TRACING = Tracing()


@contextmanager
def membrane_span(name: str, **attributes: Any) -> Iterator[Any]:
    """Open an internal span on :data:`TRACING` (a no-op when tracing is off).

    Args:
        name: Span name.
        **attributes: Span attributes.

    Yields:
        Any: The span, or ``None`` when tracing is off.
    """
    with TRACING.span(name, attributes=attributes) as span:
        yield span


__all__ = ["SERVICE_NAME", "TRACING", "Tracing", "membrane_span"]
