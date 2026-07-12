"""Tests for latency_router module."""

import pytest

from membrane.fragment import Fragment
from membrane.latency_router import LatencyRouter
from membrane.membrane_node import MembraneNode
from membrane.structural_signature import StructuralSignature


def make_fragment(content_hash, size=10):
    return Fragment(
        content_hash=content_hash,
        embedding=(0.0,),
        structural_signature=StructuralSignature(model_id="m", layer_range=(0, 1), token_span=(0, 1)),
        size=size,
        ttl=3600.0,
        reuse_score=0.5,
        version_id=1,
    )


class TestLatencyRouter:
    """Test suite for LatencyRouter."""

    def test_route_local_hit(self):
        local = MembraneNode("local")
        local.store(make_fragment("abc"))
        router = LatencyRouter()
        node_id = router.route_local_or_replica("abc", local, [])
        assert node_id == "local"

    def test_route_fallback_to_origin_when_no_candidates(self):
        """When no candidate has the fragment, should fall back to origin."""
        local = MembraneNode("local")
        router = LatencyRouter(origin_node_id="origin-1")
        node_id = router.route_local_or_replica("abc", local, [])
        assert node_id == "origin-1"

    def test_route_fallback_to_local_when_no_origin_set(self):
        """When no origin is configured, fallback to local node."""
        local = MembraneNode("local")
        router = LatencyRouter()
        node_id = router.route_local_or_replica("abc", local, [])
        assert node_id == "local"

    def test_route_nearest_replica(self):
        local = MembraneNode("local")
        r1 = MembraneNode("replica-east")
        r2 = MembraneNode("replica-west")
        r1.store(make_fragment("abc"))
        r2.store(make_fragment("abc"))
        router = LatencyRouter(latency_table={"replica-east": 10.0, "replica-west": 50.0})
        node_id = router.route_local_or_replica("abc", local, [r1, r2])
        assert node_id == "replica-east"

    def test_add_latency_updates_table(self):
        router = LatencyRouter()
        router.add_latency("n1", 20.0)
        assert router.get_latency("n1") == 20.0

    def test_get_latency_unknown_returns_inf(self):
        router = LatencyRouter()
        assert router.get_latency("unknown") == float("inf")

    def test_route_prefers_local_even_if_replica_lower_latency(self):
        local = MembraneNode("local")
        local.store(make_fragment("abc"))
        r1 = MembraneNode("replica")
        r1.store(make_fragment("abc"))
        router = LatencyRouter(latency_table={"local": 100.0, "replica": 1.0})
        node_id = router.route_local_or_replica("abc", local, [r1])
        assert node_id == "local"

    def test_origin_fallback_overrides_local_when_no_replica(self):
        """Origin fallback should be preferred over local when no replica holds fragment."""
        local = MembraneNode("local")
        router = LatencyRouter(origin_node_id="origin-1")
        node_id = router.route_local_or_replica("abc", local, [])
        assert node_id == "origin-1"

    def test_origin_id_stored_in_attribute(self):
        router = LatencyRouter(origin_node_id="my-origin")
        assert router.origin_node_id == "my-origin"
