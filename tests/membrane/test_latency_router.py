from tests.conftest import make_fragment

"""Tests for latency_router module."""

import pytest

from membrane.fragment import Fragment
from membrane.latency import Latency
from membrane.node import Node


class TestLatencyRouter:
    """Test suite for Latency."""

    def test_route_local_hit(self):
        local = Node("local")
        local.store(make_fragment("abc"))
        router = Latency()
        node_id = router.pick_target("abc", local, [])
        assert node_id == "local"

    def test_route_fallback_to_origin_when_no_candidates(self):
        """When no candidate has the fragment, should fall back to origin."""
        local = Node("local")
        router = Latency(origin_node_id="origin-1")
        node_id = router.pick_target("abc", local, [])
        assert node_id == "origin-1"

    def test_route_fallback_to_local_when_no_origin_set(self):
        """When no origin is configured, fallback to local node."""
        local = Node("local")
        router = Latency()
        node_id = router.pick_target("abc", local, [])
        assert node_id == "local"

    def test_route_nearest_replica(self):
        local = Node("local")
        r1 = Node("replica-east")
        r2 = Node("replica-west")
        r1.store(make_fragment("abc"))
        r2.store(make_fragment("abc"))
        router = Latency(latency_table={"replica-east": 10.0, "replica-west": 50.0})
        node_id = router.pick_target("abc", local, [r1, r2])
        assert node_id == "replica-east"

    def test_add_latency_updates_table(self):
        router = Latency()
        router.add_latency("n1", 20.0)
        assert router.get_latency("n1") == 20.0

    def test_get_latency_unknown_returns_inf(self):
        router = Latency()
        assert router.get_latency("unknown") == float("inf")

    def test_route_prefers_local_even_if_replica_lower_latency(self):
        local = Node("local")
        local.store(make_fragment("abc"))
        r1 = Node("replica")
        r1.store(make_fragment("abc"))
        router = Latency(latency_table={"local": 100.0, "replica": 1.0})
        node_id = router.pick_target("abc", local, [r1])
        assert node_id == "local"

    def test_origin_fallback_overrides_local_when_no_replica(self):
        """Origin fallback should be preferred over local when no replica holds fragment."""
        local = Node("local")
        router = Latency(origin_node_id="origin-1")
        node_id = router.pick_target("abc", local, [])
        assert node_id == "origin-1"

    def test_origin_id_stored_in_attribute(self):
        router = Latency(origin_node_id="my-origin")
        assert router.origin_node_id == "my-origin"
