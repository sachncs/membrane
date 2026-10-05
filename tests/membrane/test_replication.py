"""Replication carries KV bytes; hand-offs are verified before ownership moves."""

import hashlib
import threading
from unittest.mock import MagicMock

from membrane.network.config import ClusterConfig
from membrane.node import Node
from membrane.replication import hand_off_primary, replicate_fragment, verify_replica
from membrane.replicator import Replicator
from membrane.ring import Ring
from membrane.server import Server
from tests.conftest import make_fragment


class BlobPeer:
    """Peer double with a content store and a fragment table."""

    def __init__(self, corrupt: bool = False) -> None:
        self.blobs: dict[str, bytes] = {}
        self.fragments: dict[str, object] = {}
        self.primary: set[str] = set()
        self.calls: list[str] = []
        self.corrupt = corrupt

    def put_blob(self, payload_ref: str, data: bytes) -> bool:
        self.calls.append("put_blob")
        self.blobs[payload_ref] = data + b"!" if self.corrupt else data
        return True

    def request_replicate(self, fragment, is_primary: bool = False) -> bool:
        self.calls.append("replicate")
        if fragment.payload_ref is not None and fragment.payload_ref not in self.blobs:
            return False  # mirrors op_replicate's 422
        self.fragments[fragment.identity.payload_hash] = fragment
        if is_primary:
            self.primary.add(fragment.identity.payload_hash)
        return True

    def blob_digest(self, payload_ref: str) -> str | None:
        self.calls.append("digest")
        data = self.blobs.get(payload_ref)
        return None if data is None else hashlib.sha256(data).hexdigest()

    def retrieve_fragment(self, content_hash: str):
        return self.fragments.get(content_hash)


def test_replicate_sends_bytes_before_metadata() -> None:
    peer = BlobPeer()
    frag = make_fragment("r1")
    assert replicate_fragment(peer, frag, b"kv")
    assert peer.calls == ["put_blob", "replicate"]
    assert peer.blobs[frag.payload_ref] == b"kv"


def test_replicate_refuses_without_local_bytes() -> None:
    peer = BlobPeer()
    assert not replicate_fragment(peer, make_fragment("r2"), None)
    assert peer.calls == []


def test_metadata_only_fragments_skip_the_blob() -> None:
    peer = BlobPeer()
    base = make_fragment("r3")
    frag = type(base)(
        identity=base.identity,
        payload_ref=None,
        payload_size=base.payload_size,
        ttl=base.ttl,
        reuse_score=base.reuse_score,
        version_id=base.version_id,
    )
    assert replicate_fragment(peer, frag, None)
    assert peer.calls == ["replicate"]


def test_hand_off_skips_identical_bytes_and_verifies() -> None:
    peer = BlobPeer()
    frag = make_fragment("r4")
    peer.blobs[frag.payload_ref] = b"kv"  # already a replica
    assert hand_off_primary(peer, frag, b"kv")
    assert "put_blob" not in peer.calls
    assert frag.identity.payload_hash in peer.primary


def test_hand_off_fails_when_peer_copy_differs() -> None:
    peer = BlobPeer(corrupt=True)
    frag = make_fragment("r5")
    assert not hand_off_primary(peer, frag, b"kv")
    assert not verify_replica(peer, frag, b"kv")


def make_replicator(node: Node, peer: BlobPeer, ring_owner: str) -> Replicator:
    membership = MagicMock()
    membership.get_client.return_value = peer
    shard = MagicMock()
    shard.hash_ring.get_node.return_value = ring_owner
    return Replicator(
        membership=membership,
        shard=shard,
        node=node,
        config=ClusterConfig(node_id=node.node_id),
        stop_event=threading.Event(),
        running=[True],
    )


def test_rebalance_moves_misplaced_primaries_after_verification() -> None:
    node = Node("local", max_memory_bytes=10**6)
    frag = make_fragment("r6")
    node.content_store.put(frag.payload_ref, b"kv")
    node.store(frag, is_primary=True)
    peer = BlobPeer()
    replicator = make_replicator(node, peer, ring_owner="peer-1")
    assert replicator.rebalance({"peer-1"}) == 1
    assert "r6" not in node.primary_hashes
    assert "r6" in peer.primary
    replicator.shard.migrate_primary.assert_called_once()


def test_rebalance_keeps_ownership_when_verification_fails() -> None:
    node = Node("local", max_memory_bytes=10**6)
    frag = make_fragment("r7")
    node.content_store.put(frag.payload_ref, b"kv")
    node.store(frag, is_primary=True)
    replicator = make_replicator(node, BlobPeer(corrupt=True), ring_owner="peer-1")
    assert replicator.rebalance({"peer-1"}) == 0
    assert "r7" in node.primary_hashes


def test_rebalance_ignores_unhealthy_or_local_owners() -> None:
    node = Node("local", max_memory_bytes=10**6)
    node.content_store.put("blob-r8", b"kv")
    node.store(make_fragment("r8"), is_primary=True)
    assert make_replicator(node, BlobPeer(), ring_owner="local").rebalance({"peer-1"}) == 0
    assert make_replicator(node, BlobPeer(), ring_owner="peer-9").rebalance({"peer-1"}) == 0


def test_drain_target_follows_ring_order() -> None:
    ring = Ring()
    for node_id in ("a", "b", "c"):
        ring.add_node(node_id)
    server = Server(node=Node("a"), port=0)
    order = ring.get_nodes("some-hash", n=3)
    expected = next(n for n in order if n in {"b", "c"})
    assert server.drain_target("some-hash", ring, {"b", "c"}) == expected
    assert server.drain_target("some-hash", ring, set()) is None
