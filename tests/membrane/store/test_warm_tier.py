"""Fragments evicted from memory move to the warm tier and come back on read."""

from pathlib import Path

import pytest

from membrane.content_store import FilesystemBlob
from membrane.node import Node
from membrane.runtime.components import load_data_key
from membrane.runtime.settings import ServerSettings, SettingsError, build_server
from membrane.store.tiered import WarmTier
from membrane.tiers import TierPolicy
from tests.conftest import make_fragment


def warm(tmp_path: Path, capacity: int = 10_000) -> WarmTier:
    store = FilesystemBlob(tmp_path / "warm", tenant_id="w", key_provider=load_data_key(tmp_path, ""))
    return WarmTier(store, capacity)


def store_with_bytes(node: Node, name: str, size: int = 100, reuse: float = 0.5) -> None:
    frag = make_fragment(name, size=size, reuse_score=reuse)
    node.content_store.put(frag.payload_ref, name.encode() * (size // len(name) + 1))
    assert node.store(frag)


def test_evicted_fragment_is_promoted_with_its_bytes(tmp_path: Path) -> None:
    node = Node("n", max_memory_bytes=250)
    node.lower_tier = warm(tmp_path)
    store_with_bytes(node, "a1")
    store_with_bytes(node, "b1")
    store_with_bytes(node, "c1")  # evicts one
    node.lower_tier.flush()
    evicted = next(h for h in ("a1", "b1") if h not in node.fragments)
    assert evicted in node.lower_tier and node.lower_tier.demotions == 1
    fragment = node.retrieve(evicted)
    assert fragment is not None and evicted in node.fragments
    assert node.content_store.get(fragment.payload_ref).startswith(evicted.encode())
    assert evicted not in node.lower_tier and node.lower_tier.promotions == 1


def test_capacity_drops_the_oldest(tmp_path: Path) -> None:
    tier = warm(tmp_path, capacity=250)
    for name in ("x1", "x2", "x3"):
        frag = make_fragment(name, size=100)
        tier.demote(frag, b"z" * 100)
    assert tier.size_bytes == 200 and "x1" not in tier and "x3" in tier


def test_index_survives_restart(tmp_path: Path) -> None:
    tier = warm(tmp_path)
    tier.demote(make_fragment("r1", size=50), b"q" * 50)
    reopened = warm(tmp_path)
    assert "r1" in reopened and reopened.size_bytes == 50
    fragment, payload = reopened.promote("r1")
    assert fragment.identity.payload_hash == "r1" and payload == b"q" * 50


def test_deletes_and_archival_fragments_are_not_demoted(tmp_path: Path) -> None:
    node = Node("n", max_memory_bytes=10_000)
    node.lower_tier = warm(tmp_path)
    store_with_bytes(node, "d1")
    node.remove_fragment("d1")
    node.lower_tier.flush()
    assert "d1" not in node.lower_tier
    picky = WarmTier(node.lower_tier.store, 10_000, TierPolicy(warm_threshold=0.4, archive_threshold=0.2))
    assert not picky.demote(make_fragment("arch", reuse_score=0.1), b"x")
    assert picky.demote(make_fragment("keep", reuse_score=0.3), b"x")


def test_settings_wire_the_tier_and_encrypted_memory_store(tmp_path: Path) -> None:
    server, _ = build_server(ServerSettings(port=0, data_dir=str(tmp_path), warm_tier_bytes=1 << 20, load_hooks=False))
    assert isinstance(server.node.lower_tier, WarmTier)
    with pytest.raises(SettingsError, match="needs --data-dir"):
        ServerSettings(warm_tier_bytes=1)
    server, _ = build_server(
        ServerSettings(port=0, data_dir=str(tmp_path / "m"), content_store="encrypted-memory", load_hooks=False)
    )
    store = server.node.content_store
    store.put("k" * 8, b"secret")
    assert store.get("k" * 8) == b"secret" and type(store).__name__ == "EncryptedInProcessBytes"
