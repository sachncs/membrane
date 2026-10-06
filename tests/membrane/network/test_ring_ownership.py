"""Every node sees the same ring, itself included, so primaries settle on their owners."""

import logging

from membrane.network.cluster import Cluster
from membrane.network.config import ClusterConfig
from membrane.node import Node
from tests.conftest import make_fragment

NODES = [f"membrane-{i}" for i in range(5)]


def cluster(node_id: str) -> Cluster:
    logging.disable(logging.CRITICAL)
    try:
        c = Cluster(node_id=node_id, host="127.0.0.1", port=1, node=Node(node_id), config=ClusterConfig())
    finally:
        logging.disable(logging.NOTSET)
    for peer in NODES:
        c.membership.add(peer, "127.0.0.1", 2)
    return c


def test_the_local_node_is_on_its_own_ring() -> None:
    c = cluster("membrane-0")
    assert c.hash_ring.node_ids == set(NODES)
    assert "membrane-0" not in c.membership.peers  # on the ring, not a peer


def test_all_nodes_agree_on_owners_and_owners_keep_their_primaries() -> None:
    clusters = {n: cluster(n) for n in NODES}
    hashes = [f"{i:032x}" for i in range(500)]
    owners = {h: clusters[NODES[0]].hash_ring.get_node(h) for h in hashes}
    for c in clusters.values():
        assert all(c.hash_ring.get_node(h) == owners[h] for h in hashes)
    # Give every node every hash as a primary: each must hand off exactly
    # the ones it does not own, so nothing ever comes back to it.
    healthy = set(NODES)
    for node_id, c in clusters.items():
        for h in hashes:
            c.node.store(make_fragment(h, (0, 3)), is_primary=True)
        misplaced = c.replicator.misplaced_primaries(healthy)
        assert set(misplaced) == {h for h in hashes if owners[h] != node_id}
        assert all(misplaced[h] == owners[h] for h in misplaced)
    # The owners share the work roughly evenly.
    counts = [sum(1 for o in owners.values() if o == n) for n in NODES]
    assert max(counts) < 0.35 * len(hashes)
