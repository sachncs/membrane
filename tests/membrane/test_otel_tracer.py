"""OpenTelemetry tracing: no-op when off; request, write, and peer spans when on."""

import sys

import pytest
from fastapi.testclient import TestClient

from membrane.network.peer import Peer
from membrane.node import Node
from membrane.otel_tracer import SERVICE_NAME, TRACING, membrane_span
from membrane.quorum import QuorumReplicator
from membrane.server import Server
from tests.conftest import make_fragment

pytest.importorskip("opentelemetry.sdk")
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture
def spans():
    exporter = InMemorySpanExporter()
    assert TRACING.configure(exporter=exporter, node_id="t0")
    yield exporter
    TRACING.shutdown()


def test_off_by_default_and_never_imports_the_sdk(monkeypatch) -> None:
    TRACING.shutdown()
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    assert not TRACING.configure()
    monkeypatch.setitem(sys.modules, "opentelemetry", None)  # importing would fail
    with membrane_span("noop", a=1) as span:
        assert span is None
    headers: dict[str, str] = {}
    TRACING.inject(headers)
    assert headers == {}
    assert SERVICE_NAME == "membrane"


def test_each_request_gets_a_server_span(spans) -> None:
    server = Server(node=Node("t0"), port=0, load_hooks=False)
    response = TestClient(server.transport.app).get("/inventory", headers={"X-Request-ID": "req-123"})
    assert response.status_code == 200
    span = next(s for s in spans.get_finished_spans() if s.name == "GET /inventory")
    assert span.attributes["http.response.status_code"] == 200
    assert span.attributes["membrane.request_id"] == "req-123"
    assert span.resource.attributes["service.instance.id"] == "t0"


def test_incoming_traceparent_is_continued(spans) -> None:
    server = Server(node=Node("t1"), port=0, load_hooks=False)
    parent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    TestClient(server.transport.app).get("/livez", headers={"traceparent": parent})
    span = next(s for s in spans.get_finished_spans() if s.name == "GET /livez")
    assert format(span.context.trace_id, "032x") == "0af7651916cd43dd8448eb211c80319c"


class RecordingTransport:
    """Peer transport double that records outgoing headers."""

    def __init__(self) -> None:
        self.headers: list[dict[str, str]] = []

    def request(self, method, url, body, headers, timeout_sec):
        self.headers.append(dict(headers))
        return {"success": True, "stored": True}

    def request_bytes(self, method, url, body, headers, timeout_sec):
        raise NotImplementedError


def test_quorum_write_traces_and_propagates_to_peers(spans) -> None:
    transport = RecordingTransport()
    peer = Peer("http://peer:8080", transport=transport, max_retries=1)
    node = Node("w")
    frag = make_fragment("q1")
    result = QuorumReplicator()(frag, [peer], quorum_count=1, timeout_sec=2.0, blob=b"kv")
    assert result.success
    names = [s.name for s in spans.get_finished_spans()]
    assert "quorum.replicate" in names and "replication.push" in names
    quorum = next(s for s in spans.get_finished_spans() if s.name == "quorum.replicate")
    assert quorum.attributes["membrane.acks"] == 1
    # Both the blob upload and the metadata call carry the trace.
    assert all("traceparent" in h for h in transport.headers) and len(transport.headers) == 2
    trace_ids = {h["traceparent"].split("-")[1] for h in transport.headers}
    assert trace_ids == {format(quorum.context.trace_id, "032x")}
    assert node is not None
