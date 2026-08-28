"""Tests for TransferService."""

from membrane.fragment import Fragment
from membrane.node import Node
from membrane.signature import Signature
from membrane.transfer import TransferService

def test_inventory_digest_reflects_state():
    svc = TransferService()
    node = Node("n1")
    node.store(make_fragment("h1"))
    node.store(make_fragment("h2", size=200))
    digest = svc.inventory_digest(node)
    assert digest == {"h1": 1, "h2": 1}

def test_compare_finds_missing():
    svc = TransferService()
    local = {"h1": 1}
    remote = {"h1": 1, "h2": 1}
    missing = svc.compare_inventories(local, remote)
    assert missing == {"h2"}

def test_compare_ignores_same_version():
    svc = TransferService()
    local = {"h1": 1}
    remote = {"h1": 1}
    assert svc.compare_inventories(local, remote) == set()

def test_compare_detects_newer_version():
    svc = TransferService()
    local = {"h1": 1}
    remote = {"h1": 2}
    assert svc.compare_inventories(local, remote) == {"h1"}

def test_transfer_fragment_copies():
    svc = TransferService()
    source = Node("s1")
    target = Node("t1")
    frag = make_fragment("h1")
    source.store(frag)
    assert svc.transfer_fragment(source, target, "h1")
    assert target.retrieve("h1") == frag

def test_transfer_missing_returns_false():
    svc = TransferService()
    source = Node("s1")
    target = Node("t1")
    assert not svc.transfer_fragment(source, target, "missing")

def test_sync_nodes_transfers_all():
    svc = TransferService()
    source = Node("s1")
    target = Node("t1")
    source.store(make_fragment("h1"))
    source.store(make_fragment("h2"))
    transferred = svc.sync_nodes(source, target)
    assert set(transferred) == {"h1", "h2"}
    assert target.retrieve("h1") is not None
    assert target.retrieve("h2") is not None

def test_sync_nodes_is_idempotent():
    svc = TransferService()
    source = Node("s1")
    target = Node("t1")
    source.store(make_fragment("h1"))
    svc.sync_nodes(source, target)
    second = svc.sync_nodes(source, target)
    assert second == []
