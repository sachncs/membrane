"""Tests for origin_node module."""

import pytest

from membrane.fragment import Fragment
from membrane.membrane_node import MembraneNode
from membrane.origin_node import OriginNode
from membrane.structural_signature import StructuralSignature
from membrane.transfer_service import TransferService


def make_fragment(content_hash, size=100):
    return Fragment(
        content_hash=content_hash,
        embedding=(0.0, 0.0),
        structural_signature=StructuralSignature(
            model_id="m", layer_range=(0, 1), token_span=(0, 1)
        ),
        size=size,
        ttl=3600.0,
        reuse_score=0.5,
        version_id=1,
    )


class TestOriginNode:
    """Test suite for OriginNode."""

    def test_promote_to_replica_copiesfragment(self):
        origin = OriginNode("origin-1")
        replica = MembraneNode("replica-1")
        frag = make_fragment("abc", size=50)
        assert origin.promote_to_replica(frag, replica)
        assert replica.retrieve("abc") is not None

    def test_promote_to_replica_stores_on_origin_first(self):
        origin = OriginNode("origin-1")
        replica = MembraneNode("replica-1")
        frag = make_fragment("xyz", size=50)
        assert frag.content_hash not in origin.fragments
        origin.promote_to_replica(frag, replica)
        assert origin.retrieve("xyz") is not None

    def test_bulk_promote_partial_success(self):
        origin = OriginNode("origin-1")
        replica = MembraneNode("replica-1", max_memory_bytes=80)
        f1 = make_fragment("a", size=40)
        f2 = make_fragment("b", size=40)
        f3 = make_fragment("c", size=40)
        origin.store(f1, is_primary=True)
        origin.store(f2, is_primary=True)
        origin.store(f3, is_primary=True)
        transferred = origin.bulk_promote(["a", "b", "c"], replica)
        assert len(transferred) >= 2

    def test_origin_is_membrane_node_subclass(self):
        origin = OriginNode("o")
        assert isinstance(origin, MembraneNode)

    def test_transfer_service_injection(self):
        ts = TransferService()
        origin = OriginNode("o", transfer_service=ts)
        assert origin.transfer_service is ts

    def test_promote_to_replica_missingfragment_fails(self):
        origin = OriginNode("origin-1")
        replica = MembraneNode("replica-1")
        assert not origin.transfer_service.transfer_fragment(origin, replica, "missing")
