"""Scale out: incremental inventory digests, bucketed repair, bounded locations."""

import random

from fastapi.testclient import TestClient

from membrane.network.membership import Membership
from membrane.node import Node
from membrane.registry import Registry
from membrane.replicator import Replicator
from membrane.ring import Ring
from membrane.server import Server
from membrane.services.placement import ClusterView, LatencyPlacement, PlacementService
from membrane.shard import Shard
from membrane.store.digest import BUCKETS, InventoryDigest, bucket_of, fold
from membrane.wire.v3.chunks import sha256_hex
from tests.conftest import make_fragment


def test_digest_is_order_independent_and_tracks_removals() -> None:
    hashes = [f"{i:032x}" for i in range(500)]
    forward, backward = InventoryDigest(), InventoryDigest()
    for h in hashes:
        forward.add(h, 1)
    for h in reversed(hashes):
        backward.add(h, 1)
    assert forward.root() == backward.root()
    assert forward.buckets() == backward.buckets()
    before = forward.root()
    forward.add("extra", 1)
    assert forward.root() != before
    forward.remove("extra")
    assert forward.root() == before
    forward.add(hashes[0], 2)  # a new version changes its bucket only
    changed = [b for b, (x, y) in enumerate(zip(forward.buckets(), backward.buckets(), strict=True)) if x != y]
    assert changed == [bucket_of(hashes[0])]
    assert len(forward) == 500


def test_bucket_digest_equals_fold_of_members() -> None:
    digest = InventoryDigest()
    members = {f"h{i}": i % 3 + 1 for i in range(200)}
    for h, v in members.items():
        digest.add(h, v)
    for bucket in range(BUCKETS):
        page, cursor = digest.page(bucket)
        assert cursor == ""
        assert digest.buckets()[bucket] == fold(page)


def test_digest_pages_and_samples() -> None:
    digest = InventoryDigest()
    for i in range(50):
        digest.add(f"same-bucket-{i}", 1)
    bucket = bucket_of("same-bucket-0")
    seen, cursor = {}, ""
    while True:
        page, cursor = digest.page(bucket, cursor, limit=1)
        seen.update(page)
        if not cursor:
            break
    assert all(bucket_of(h) == bucket for h in seen)
    assert len(digest.sample(10)) == 10
    assert set(digest.sample(1000)) == {f"same-bucket-{i}" for i in range(50)}


def test_node_and_http_expose_the_digest() -> None:
    node = Node("d")
    for i in range(30):
        node.store(make_fragment(f"f{i:02d}"))
    client = TestClient(Server(node=node, port=0, load_hooks=False).transport.app)
    body = client.get("/inventory/buckets").json()
    assert len(body["buckets"]) == BUCKETS and body["count"] == 30
    assert body["root"] == node.digest.root().hex()
    bucket = bucket_of("f07")
    page = client.get("/inventory", params={"bucket": bucket}).json()
    assert "f07" in page["digest"] and all(bucket_of(h) == bucket for h in page["digest"])
    assert client.get("/inventory", params={"bucket": BUCKETS}).status_code == 400
    with node.lock:
        node.remove_fragment("f07")
    assert client.get("/inventory/buckets").json()["count"] == 29


class CountingPeer:
    """A peer node behind the calls the replicator makes."""

    def __init__(self, node: Node) -> None:
        self.node = node
        self.bucket_requests: list[int] = []
        self.pushed: list[str] = []

    def bucket_digests(self):
        return self.node.digest.buckets()

    def inventory_bucket(self, bucket, page_size=10_000):
        self.bucket_requests.append(bucket)
        return self.node.digest.page(bucket)[0]

    def put_blob(self, payload_ref, data):
        self.node.content_store.put(payload_ref, data)
        return True

    def blob_digest(self, payload_ref):
        data = self.node.content_store.get(payload_ref)
        return sha256_hex(data) if data is not None else None

    def request_replicate(self, fragment, is_primary=False):
        self.pushed.append(fragment.identity.payload_hash)
        return self.node.store(fragment, is_primary=is_primary)

    def retrieve_fragment(self, content_hash):
        return self.node.retrieve(content_hash)


def replicator_with_peer() -> tuple[Replicator, Node, CountingPeer]:
    local = Node("a", max_memory_bytes=1 << 30)
    ring = Ring()
    membership = Membership("a", ring, Shard(ring))
    membership.add("b", "127.0.0.1", 9100)
    peer = CountingPeer(Node("b", max_memory_bytes=1 << 30))
    membership.clients["b"] = peer  # type: ignore[assignment]
    return Replicator(membership=membership, node=local), local, peer


def test_repair_pages_only_buckets_that_changed() -> None:
    replicator, local, peer = replicator_with_peer()
    hashes = [f"{i:032x}" for i in range(300)]
    for h in hashes:
        local.content_store.put(f"blob-{h}", b"kv")
        local.store(make_fragment(h, (0, 3)), is_primary=True)
    assert replicator.push_missing("b", hashes) == set()
    assert sorted(peer.pushed) == sorted(hashes)
    first_pass = len(peer.bucket_requests)

    # Verify pass: buckets the pushes changed are paged once and marked verified.
    replicator.push_missing("b", hashes)
    # Quiet cluster: nothing changed, so nothing is paged.
    peer.bucket_requests.clear()
    assert replicator.push_missing("b", hashes) == set()
    assert peer.bucket_requests == []
    assert first_pass > 0

    # The peer loses one fragment: only its bucket is paged, and it is pushed back.
    lost = random.choice(hashes)
    with peer.node.lock:
        peer.node.remove_fragment(lost)
    peer.pushed.clear()
    replicator.push_missing("b", hashes)
    assert peer.bucket_requests == [bucket_of(lost)]
    assert peer.pushed == [lost]
    assert replicator.drift["b"] == 1


def test_repair_falls_back_to_a_full_inventory_for_old_peers() -> None:
    replicator, local, peer = replicator_with_peer()
    local.content_store.put("blob-x", b"kv")
    local.store(make_fragment("x", (0, 3)), is_primary=True)
    peer.bucket_digests = lambda: None  # type: ignore[method-assign]
    peer.inventory_digest = lambda: {}  # type: ignore[attr-defined]
    assert replicator.push_missing("b", ["x"]) == set()
    assert peer.pushed == ["x"]


def test_location_registry_is_bounded_lru() -> None:
    registry = Registry(max_entries=3)
    for h in ("a", "b", "c"):
        registry.record_fragment_location(h, "n1")
    registry.record_fragment_location("a", "n2")  # refresh a
    registry.record_fragment_location("d", "n1")  # evicts b, the least recent
    assert registry.locate_fragment("b") == set()
    assert registry.locate_fragment("a") == {"n1", "n2"}
    assert len(registry.fragment_locations) == 3


def test_unknown_locations_fall_back_to_ring_owners() -> None:
    from types import SimpleNamespace

    local = Node("a")
    ring = Ring()
    ring.add_node("a")
    membership = Membership("a", ring, Shard(ring))
    for peer_id in ("b", "c"):
        membership.add(peer_id, "127.0.0.1", 9200 + ord(peer_id))
    cluster = SimpleNamespace(membership=membership, directory=Registry(), hash_ring=ring)
    view = ClusterView(local, cluster, replica_count=1)
    content_hash = "never-recorded"
    owners = [n for n in view.owners(content_hash) if n != "a"]
    assert view.holders(content_hash) == []
    assert view.probable_holders(content_hash) == owners
    assert LatencyPlacement().place(view, content_hash, []).fetch_from in owners
    # A prompt's cached prefix counts only recorded locations.
    service = PlacementService(view, LatencyPlacement())
    assert service.route(tokens=list(range(256)), model_id="m").matched_tokens == 0
