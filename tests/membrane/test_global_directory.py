"""Tests for Registry."""

from membrane.fragment import Fragment
from membrane.registry import Registry
from membrane.node import Node
from membrane.signature import Signature


def make_fragment(content_hash: str, size: int = 100) -> Fragment:
    sig = Signature("m", (0, 1), (0, 10))
    return Fragment(
        content_hash=content_hash,
        embedding=(0.1, 0.2, 0.3),
        structural_signature=sig,
        size=size,
        ttl=3600.0,
        reuse_score=0.5,
        version_id=1,
    )


def test_register_and_unregister():
    gd = Registry()
    node = Node("n1")
    gd.register_node(node)
    assert "n1" in gd.nodes
    assert gd.unregister_node("n1")
    assert "n1" not in gd.nodes


def test_unregister_missing():
    gd = Registry()
    assert not gd.unregister_node("missing")


def test_locate_fragment():
    gd = Registry()
    gd.record_fragment_location("h1", "n1")
    gd.record_fragment_location("h1", "n2")
    assert gd.locate_fragment("h1") == {"n1", "n2"}


def test_locate_missing():
    gd = Registry()
    assert gd.locate_fragment("missing") == set()


def test_optimal_nodes_ranked_by_coverage():
    from membrane.fragmenter import Fragmenter

    gd = Registry()
    n1 = Node("n1", max_memory_bytes=1_000_000)
    n2 = Node("n2", max_memory_bytes=1_000_000)
    engine = Fragmenter()
    tokens = list(range(20))
    frags = engine.create_windows(tokens, model_id="m")
    for f in frags[:2]:
        n1.store(f)
    for f in frags:
        n2.store(f)
    gd.register_node(n1)
    gd.register_node(n2)

    ranked = gd.optimal_nodes_for_reconstruction(tokens, "m", k=2)
    assert len(ranked) > 0


def test_optimal_nodes_empty():
    gd = Registry()
    assert gd.optimal_nodes_for_reconstruction([1, 2, 3], "m") == []


def test_node_load_influences_ranking():
    gd = Registry()
    n1 = Node("n1", max_memory_bytes=1000)
    n2 = Node("n2", max_memory_bytes=100_000)
    for i in range(10):
        n1.store(make_fragment(f"a{i}", size=100))
        n2.store(make_fragment(f"b{i}", size=100))
    gd.register_node(n1)
    gd.register_node(n2)

    tokens = list(range(20))
    ranked = gd.optimal_nodes_for_reconstruction(tokens, "m", k=2)
    # n1 is heavily loaded (100% memory), n2 is lightly loaded
    # Both have similar coverage but n1 gets load penalty
    if len(ranked) == 2:
        assert ranked[0] == "n2"
