from tests.conftest import make_fragment

"""Tests for origin_node module."""

import pytest

from membrane.fragment import Fragment
from membrane.node import Node
from membrane.origin import Origin
from membrane.signature import Signature
from membrane.transfer import TransferService


class TestOriginNode:
    """Test suite for Origin."""

    def test_promote_to_replica_copiesfragment(self):
        origin = Origin("origin-1")
        replica = Node("replica-1")
        frag = make_fragment("abc", size=50)
        assert origin.promote_to_replica(frag, replica)
        assert replica.retrieve("abc") is not None

    def test_promote_to_replica_stores_on_origin_first(self):
        origin = Origin("origin-1")
        replica = Node("replica-1")
        frag = make_fragment("xyz", size=50)
        assert frag.content_hash not in origin.fragments
        origin.promote_to_replica(frag, replica)
        assert origin.retrieve("xyz") is not None

    def test_bulk_promote_partial_success(self):
        origin = Origin("origin-1")
        replica = Node("replica-1", max_memory_bytes=80)
        f1 = make_fragment("a", size=40)
        f2 = make_fragment("b", size=40)
        f3 = make_fragment("c", size=40)
        origin.store(f1, is_primary=True)
        origin.store(f2, is_primary=True)
        origin.store(f3, is_primary=True)
        transferred = origin.bulk_promote(["a", "b", "c"], replica)
        assert len(transferred) >= 2

    def test_origin_is_membrane_node_subclass(self):
        origin = Origin("o")
        assert isinstance(origin, Node)

    def test_transfer_service_injection(self):
        ts = TransferService()
        origin = Origin("o", transfer_service=ts)
        assert origin.transfer_service is ts

    def test_promote_to_replica_missingfragment_fails(self):
        origin = Origin("origin-1")
        replica = Node("replica-1")
        assert not origin.transfer_service.transfer_fragment(origin, replica, "missing")
