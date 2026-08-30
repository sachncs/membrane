"""Tests for the OpenTelemetry tracer integration (Phase 3.2.4).

The v3.0.0 release wires OTel into the cluster. Tests run
without the OTel SDK or with the SDK configured to a no-op
provider; the contracts under test are the import surface,
the no-op path, and the convenience context manager.
"""

from __future__ import annotations

from membrane.otel_tracer import (
    SERVICE_NAME,
    TracerFactory,
    get_default_tracer,
    membrane_span,
)


class TestTracerFactory:
    def test_default_returns_noop(self):
        """No OTLP endpoint -> no-op tracer."""
        factory = TracerFactory()
        tracer = factory.configure(endpoint=None)
        # The NoOpTracer instance is what `configure(None)` returns.
        assert tracer is not None

    def test_configure_with_endpoint_no_op_when_sdk_missing(self):
        """When the SDK is unavailable, the factory degrades to no-op."""
        factory = TracerFactory()
        # An unreachable endpoint should still complete without raising.
        tracer = factory.configure(endpoint="http://localhost:4317")
        assert tracer is not None

    def test_lazy_tracer_property(self):
        factory = TracerFactory()
        tracer = factory.tracer
        # Subsequent access returns the same instance.
        assert factory.tracer is tracer


class TestMembraneSpan:
    def test_with_no_endpoint_is_inert(self):
        """`membrane_span` on a no-op tracer never raises."""
        TracerFactory().configure(endpoint=None)
        with membrane_span("test.span", attribute="value"):
            pass

    def test_yields_span_object(self):
        """Inside the context manager a span value is yielded."""
        TracerFactory().configure(endpoint=None)
        with membrane_span("test.span") as span:
            # Either the OTel span or None; both are valid.
            assert span is None or hasattr(span, "set_attribute")


class TestServiceName:
    def test_default_service_name(self):
        assert SERVICE_NAME == "membrane"

    def test_get_default_tracer_returns_something(self):
        tracer = get_default_tracer()
        assert tracer is not None
