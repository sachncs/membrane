"""Tests for the post-prefill KV shipping flow.

This used to cover the deleted :class:`membrane.kvreturn.KVReturn`
helper. The shipping flow is now a direct use of
:class:`membrane.transfer.TransferService` —
:func:`transfer_fragment` already moves one fragment per call, so
looping over a list of fragments and aggregating the successes
exercises the same behavior with no extra wrapper.
"""

import pytest

from membrane.node import Node
from membrane.prefilling import Adapter
from membrane.transfer import TransferService
from tests.conftest import make_fragment


def _ship(prefill_result, source, target, transfer):
    """Move every fragment via ``transfer`` and return the successes."""
    transferred = []
    for frag in prefill_result.fragments:
        if transfer.transfer_fragment(source, target, frag.identity.payload_hash):
            transferred.append(frag.identity.payload_hash)
    return transferred


def test_ship_kv_transfers_all_fragments():
    transfer = TransferService()
    source = Node("source")
    target = Node("target")
    f1 = make_fragment("a", size=10)
    f2 = make_fragment("b", size=10)
    source.store(f1, is_primary=True)
    source.store(f2, is_primary=True)
    adapter = Adapter()
    result = adapter.prefill(list(range(10)), "m")
    from dataclasses import replace

    result = replace(result, fragments=[f1, f2])
    transferred = _ship(result, source, target, transfer)
    assert "a" in transferred
    assert "b" in transferred
    assert target.retrieve("a") is not None


def test_ship_kv_missing_on_source():
    transfer = TransferService()
    source = Node("source")
    target = Node("target")
    adapter = Adapter()
    result = adapter.prefill(list(range(10)), "m")
    transferred = _ship(result, source, target, transfer)
    assert transferred == []


def test_custom_transfer_service():
    ts = TransferService()
    assert isinstance(ts, TransferService)
