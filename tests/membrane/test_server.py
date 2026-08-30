"""Tests for Server."""

import time

import pytest

from membrane.compute.cpu import CPU
from membrane.node import Node
from membrane.server import Server


class TestMembraneServer:
    """Test suite for Server."""

    def test_create_http_server(self):
        node = Node("s1")
        srv = Server(node=node, transport="http", compute="cpu")
        assert srv.transport_type == "http"
        assert srv.compute_backend.device_name() == "cpu"
        assert srv.node.node_id == "s1"

    def test_start_and_stop(self):
        node = Node("s2")
        srv = Server(node=node, transport="http", compute="cpu", port=18081)
        srv.start()
        assert srv.running
        time.sleep(0.1)
        srv.stop()
        assert not srv.running

    def test_diagnostics(self):
        node = Node("s3")
        srv = Server(node=node, transport="http", compute="cpu")
        diag = srv.diagnostics()
        assert diag.node_id == "s3"
        assert diag.uptime_seconds >= 0
        assert diag.backend_name == "cpu"
        assert diag.redis_connected is False

    def test_log_event(self):
        node = Node("s4")
        srv = Server(node=node, transport="http", compute="cpu")
        srv.log_event("info", "hello", node_id="s4", bytes_affected=100)
        events = srv.recent_events(n=5)
        assert len(events) == 1
        assert events[0].message == "hello"
        assert events[0].level == "info"

    def test_peer_lifecycle(self):
        node = Node("s5")
        srv = Server(node=node, transport="http", compute="cpu")
        srv.connected_nodes.add("peer-1")
        assert "peer-1" in srv.connected_nodes
        diag = srv.diagnostics()
        assert diag.connected_nodes == 1
        srv.connected_nodes.discard("peer-1")
        assert "peer-1" not in srv.connected_nodes

    def test_event_rolloff(self):
        node = Node("s7")
        srv = Server(node=node, transport="http", compute="cpu")
        for i in range(20):
            srv.log_event("info", f"event-{i}")
        events = srv.recent_events(n=5)
        assert len(events) == 5
        assert events[-1].message == "event-19"

    def test_persistence_is_caching_wrapper(self):
        """setup_persistence must wrap the underlying backend in
        :class:`CachingPersistence` so repeated reads hit the local
        cache instead of crossing the network on every call.
        """
        from membrane.persistence.cache import CachingPersistence

        node = Node("s-cache")
        srv = Server(node=node, transport="http", compute="cpu")
        assert isinstance(srv.persistence, CachingPersistence)

    def test_persistence_caching_serves_repeated_reads(self):
        """The CachingPersistence wrapping should make the second
        positive retrieve hit the in-memory cache instead of
        crossing the inner backend."""
        from membrane.fragment import Fragment
        from membrane.persistence.cache import CachingPersistence
        from membrane.signature import Signature

        node = Node("s-cache2")
        srv = Server(node=node, transport="http", compute="cpu")
        assert isinstance(srv.persistence, CachingPersistence)

        # Seed the inner backend with a known fragment, then
        # instrument its retrieve to count invocations.
        frag = Fragment(
            content_hash="cache-test",
            embedding=(0.0,),
            structural_signature=Signature("m", (0, 1), (0, 0)),
            size=10,
            ttl=3600.0,
            reuse_score=0.5,
            version_id=1,
        )
        inner = srv.persistence.inner
        inner.store_fragment(frag, "s-cache2", is_primary=True)

        calls = {"n": 0}
        original = inner.retrieve_fragment

        def counting_retrieve(content_hash):
            calls["n"] += 1
            return original(content_hash)

        inner.retrieve_fragment = counting_retrieve  # type: ignore[method-assign]

        # First retrieve: cache miss → inner backend (1 call).
        # Second retrieve: cache hit → inner backend skipped (still 1).
        srv.persistence.retrieve_fragment("cache-test")
        srv.persistence.retrieve_fragment("cache-test")
        assert calls["n"] == 1
