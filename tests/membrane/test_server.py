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
        assert srv._running
        time.sleep(0.1)
        srv.stop()
        assert not srv._running

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

    def test_register_peer(self):
        node = Node("s5")
        srv = Server(node=node, transport="http", compute="cpu")
        srv.register_peer("peer-1")
        assert "peer-1" in srv.connected_nodes
        diag = srv.diagnostics()
        assert diag.connected_nodes == 1

    def test_unregister_peer(self):
        node = Node("s6")
        srv = Server(node=node, transport="http", compute="cpu")
        srv.register_peer("peer-1")
        srv.unregister_peer("peer-1")
        assert "peer-1" not in srv.connected_nodes

    def test_event_rolloff(self):
        node = Node("s7")
        srv = Server(node=node, transport="http", compute="cpu")
        for i in range(20):
            srv.log_event("info", f"event-{i}")
        events = srv.recent_events(n=5)
        assert len(events) == 5
        assert events[-1].message == "event-19"
