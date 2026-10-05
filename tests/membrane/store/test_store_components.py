"""FragmentTable, eviction policies, and TenantGuard, alone and inside Node."""

import pytest

from membrane.decision import TinyLFU
from membrane.errors import TenantScopeError
from membrane.node import Node
from membrane.store.eviction import FrequencyLRU, WeightedLRU
from membrane.store.table import FragmentTable
from membrane.store.tenant_guard import TenantGuard
from tests.conftest import make_fragment


def test_table_tracks_memory_and_primaries() -> None:
    table = FragmentTable()
    frag = make_fragment("t1", payload_size=40)
    table.add(frag, now=100.0)
    table.touch("t1", 101.0, is_primary=True)
    assert "t1" in table and len(table) == 1
    assert table.memory_usage == 40 and table.primaries() == {"t1"}
    assert table.pop("t1") is frag
    assert table.memory_usage == 0 and table.primaries() == set()
    with pytest.raises(KeyError):
        table.pop("t1")


def test_table_expiry() -> None:
    table = FragmentTable()
    table.add(make_fragment("t2", ttl=10.0), now=0.0)
    assert not table.is_expired("t2", 5.0)
    assert table.expired(11.0) == ["t2"]
    assert not table.is_expired("absent", 11.0)


def test_weighted_lru_prefers_old_and_low_reuse() -> None:
    old = make_fragment("old", reuse_score=0.5)
    hot = make_fragment("hot", reuse_score=0.9)
    order = WeightedLRU().order([("hot", hot), ("old", old)], {"old": 1.0, "hot": 100.0}, now=200.0)
    assert order == ["old", "hot"]


def test_frequency_lru_keeps_frequently_hit_hashes() -> None:
    policy = FrequencyLRU(TinyLFU())
    for _ in range(5):
        policy.touch("popular")
    a, b = make_fragment("popular"), make_fragment("rare")
    # "popular" is older but hit more often: "rare" goes first.
    assert policy.order([("popular", a), ("rare", b)], {"popular": 1.0, "rare": 50.0}, now=60.0) == [
        "rare",
        "popular",
    ]


def test_node_uses_tinylfu_when_configured() -> None:
    node = Node("n", max_memory_bytes=250, eviction_strategy=TinyLFU())
    for name in ("a", "b"):
        node.store(make_fragment(name, payload_size=100))
    for _ in range(5):
        node.record_hit("a")
    node.store(make_fragment("c", payload_size=100))  # forces one eviction
    assert "a" in node.fragments and "b" not in node.fragments


def test_tenant_guard_rules() -> None:
    TenantGuard.check_write("acme", "", frozenset())  # auth off
    TenantGuard.check_write("acme", "acme", frozenset({"write"}))
    with pytest.raises(TenantScopeError):
        TenantGuard.check_write("acme", "globex", frozenset({"write"}))
    TenantGuard.check_write("acme", "ops", frozenset({"admin"}))
    assert TenantGuard.can_read("acme", "acme", frozenset({"read"}))
    assert not TenantGuard.can_read("acme", "globex", frozenset({"read"}))


def test_node_facade_forwards_to_table() -> None:
    node = Node("n")
    node.store(make_fragment("f"), is_primary=True)
    assert node.fragments is node.table.fragments
    assert node.primary_hashes == {"f"} and node.lock is node.table.lock
    node.memory_usage = 7
    assert node.table.memory_usage == 7
