"""TransferService between local nodes and peers (peers faked at their client API)."""

import dataclasses
import types

from membrane.node import Node
from membrane.replication import sha256_hex
from membrane.transfer import ClusterPeerEndpoint, NodeEndpoint, TransferService
from tests.conftest import make_fragment


class PeerClient:
    """The peer-client calls the transfer plane makes, served by a local node."""

    def __init__(self, node: Node) -> None:
        self.node = node

    def get_inventory(self):
        return {
            "node_id": self.node.node_id,
            "digest": {h: f.version_id for h, f in self.node.fragment_snapshot().items()},
        }

    def retrieve_fragment(self, content_hash):
        return self.node.retrieve(content_hash)

    def get_blob(self, payload_ref):
        return self.node.content_store.get(payload_ref)

    def blob_digest(self, payload_ref):
        data = self.node.content_store.get(payload_ref)
        return sha256_hex(data) if data is not None else None

    def put_blob(self, payload_ref, data):
        self.node.content_store.put(payload_ref, data)
        return True

    def request_replicate(self, fragment, is_primary=False):
        return self.node.store(fragment, is_primary=is_primary)


def cluster_of(**peers: Node):
    clients = {name: PeerClient(node) for name, node in peers.items()}
    return types.SimpleNamespace(membership=types.SimpleNamespace(get_client=clients.get))


def node_with(name: str, *hashes: str) -> Node:
    node = Node(name, max_memory_bytes=1 << 20)
    for h in hashes:
        node.content_store.put(f"blob-{h}", f"kv-{h}".encode())
        node.store(make_fragment(h, (0, 3)))
    return node


def test_local_to_local_copies_bytes_and_syncs() -> None:
    source, target = node_with("a", "1" * 32, "2" * 32), node_with("b")
    service = TransferService(local_node=target)
    assert service.transfer_fragment(source, target, "1" * 32)
    assert target.content_store.get("blob-" + "1" * 32) == b"kv-" + b"1" * 32
    assert service.transfer_fragment(source, target, "missing") is False
    assert service.sync_nodes(source, target) == ["2" * 32]
    assert service.sync_local(source, target) == []


def test_pull_from_a_peer_brings_the_bytes() -> None:
    peer, local = node_with("p", "3" * 32), node_with("l")
    service = TransferService(cluster_manager=cluster_of(p=peer), local_node=local)  # type: ignore[arg-type]
    assert service.inventory_digest("p") == {"3" * 32: 1}
    assert service.pull_from_remote("p", local, "3" * 32)
    assert local.content_store.get("blob-" + "3" * 32) == b"kv-" + b"3" * 32
    assert service.pull_from_remote("p", local, "nope") is False
    assert service.transfer_fragment("p", local, "3" * 32)
    assert service.transfer_fragment("p", local, "nope") is False


def test_push_and_peer_to_peer() -> None:
    local, p1, p2 = node_with("l", "4" * 32), node_with("p1", "5" * 32), node_with("p2")
    service = TransferService(cluster_manager=cluster_of(p1=p1, p2=p2), local_node=local)  # type: ignore[arg-type]
    assert service.push_to_remote(local, "p2", "4" * 32)
    assert "4" * 32 in p2.fragments and p2.content_store.has("blob-" + "4" * 32)
    assert service.transfer_fragment(local, "p1", "4" * 32)
    assert service.transfer_fragment("p1", "p2", "5" * 32)
    assert service.pull_from_remote("p1", "p2", "5" * 32)
    p1.content_store.put("blob-" + "9" * 32, b"kv-9")
    p1.store(make_fragment("9" * 32, (0, 3)))
    assert service.sync_nodes("p1", "p2") == ["9" * 32]
    assert service.sync_nodes(local, "p1") == []
    local.content_store.put("blob-" + "a" * 32, b"kv-a")
    local.store(make_fragment("a" * 32, (0, 3)))
    assert service.sync_nodes(local, "p1") == ["a" * 32]
    assert service.transfer_fragment("p1", "gone", "5" * 32) is False


def test_unknown_or_unconfigured_peers_fail_cleanly() -> None:
    local = node_with("l", "6" * 32)
    standalone = TransferService(local_node=local)
    assert standalone.transfer_fragment(local, "peer", "6" * 32) is False
    assert standalone.sync_nodes(local, "peer") == []
    assert standalone.pull_from_remote("peer", local, "6" * 32) is False
    assert standalone.push_to_remote(local, "peer", "6" * 32) is False
    service = TransferService(cluster_manager=cluster_of(), local_node=local)  # type: ignore[arg-type]
    ghost = ClusterPeerEndpoint("ghost", service.cluster_manager)  # type: ignore[arg-type]
    fragment = make_fragment("6" * 32, (0, 3))
    assert ghost.inventory() is None and ghost.retrieve("x") is None
    assert ghost.payload(fragment) is None and ghost.push(fragment, b"x") is False
    assert service.inventory_digest("ghost") is None
    assert service.sync_nodes("ghost", local) == []
    assert service.pull_from_remote("ghost", "ghost", "x") is False


def test_endpoint_payloads() -> None:
    node = node_with("n", "7" * 32)
    endpoint = NodeEndpoint(node)
    fragment = node.fragments["7" * 32]
    assert endpoint.payload(fragment) == b"kv-" + b"7" * 32
    assert endpoint.payload(dataclasses.replace(fragment, payload_ref=None)) is None
