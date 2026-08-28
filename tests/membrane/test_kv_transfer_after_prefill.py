"""Tests for kv_transfer_after_prefill module."""

import pytest

from membrane.fragment import Fragment
from membrane.kvreturn import KVReturn
from membrane.node import Node
from membrane.adapter import Adapter
from membrane.signature import Signature


def make_fragment(content_hash, size=10):
    return Fragment(
        content_hash=content_hash,
        embedding=(0.0,),
        structural_signature=Signature(model_id="m", layer_range=(0, 1), token_span=(0, 1)),
        size=size,
        ttl=3600.0,
        reuse_score=0.5,
        version_id=1,
    )


class TestKVTransferAfterPrefill:
    """Test suite for KVReturn."""

    def test_ship_kv_transfers_allfragments(self):
        shipper = KVReturn()
        source = Node("source")
        target = Node("target")
        f1 = make_fragment("a", size=10)
        f2 = make_fragment("b", size=10)
        source.store(f1, is_primary=True)
        source.store(f2, is_primary=True)
        adapter = Adapter()
        result = adapter.prefill(list(range(10)), "m")
        # override fragments with known stored ones
        from dataclasses import replace

        result = replace(result, fragments=[f1, f2])
        transferred = shipper.ship_kv(result, source, target)
        assert "a" in transferred
        assert "b" in transferred
        assert target.retrieve("a") is not None

    def test_ship_kv_missing_on_source(self):
        shipper = KVReturn()
        source = Node("source")
        target = Node("target")
        adapter = Adapter()
        result = adapter.prefill(list(range(10)), "m")
        transferred = shipper.ship_kv(result, source, target)
        assert transferred == []

    def test_custom_transfer_service(self):
        from membrane.transfer import TransferService

        ts = TransferService()
        shipper = KVReturn(transfer_service=ts)
        assert shipper.transfer_service is ts
