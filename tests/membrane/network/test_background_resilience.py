"""Background cluster threads survive concurrent writes, restarts, and bugs.

Found on a kind cluster during a rolling restart: the gossip thread died
with "dictionary changed size during iteration" and the replication
thread with EmptyRingError, leaving the node without gossip or
replication and with only a raw traceback on stderr.
"""

import contextlib
import logging
import sys
import threading
import time

from membrane.gc import TombstoneTable
from membrane.logging import configure_logging
from membrane.network.cluster import Cluster
from membrane.network.config import ClusterConfig
from membrane.network.gossip import Gossip
from membrane.network.membership import Membership
from membrane.node import Node
from membrane.registry import Registry
from membrane.replicator import Replicator
from membrane.ring import Ring
from membrane.shard import Shard
from tests.conftest import make_fragment


def test_gossip_state_is_safe_under_concurrent_writes() -> None:
    node = Node("local", max_memory_bytes=10**9)
    ring = Ring()
    shard = Shard(ring)
    gossip = Gossip(
        membership=Membership("local", ring, shard),
        node=node,
        config=ClusterConfig(node_id="local"),
        directory=Registry(),
        tombstones=TombstoneTable(),
        stop_event=threading.Event(),
        running=[False],
    )
    stop = threading.Event()

    def writer() -> None:
        # Keep the table small but constantly changing size.
        i = 0
        while not stop.is_set():
            node.store(make_fragment(f"w{i % 500}"))
            with contextlib.suppress(KeyError):
                node.remove_fragment(f"w{(i + 250) % 500}")
            i += 1

    # Switch threads often so the writer interleaves with iteration.
    previous_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    thread = threading.Thread(target=writer)
    thread.start()
    try:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            state = gossip.build_state()
            assert state.inventory_size >= 0
    finally:
        stop.set()
        thread.join()
        sys.setswitchinterval(previous_interval)
    assert writes_survived(node)


def writes_survived(node: Node) -> bool:
    """The writer kept running (it never empties the 500-key window)."""
    return len(node.fragments) > 0


def test_replicator_sweep_tolerates_an_empty_ring() -> None:
    node = Node("local", max_memory_bytes=10**6)
    node.store(make_fragment("p1"), is_primary=True)
    ring = Ring()
    stop = threading.Event()
    replicator = Replicator(
        membership=Membership("local", ring, Shard(ring)),
        shard=Shard(ring),
        node=node,
        config=ClusterConfig(node_id="local", gossip_interval_sec=0.01),
        stop_event=stop,
        running=[True],
    )
    thread = threading.Thread(target=replicator.loop)
    thread.start()
    time.sleep(0.1)
    stop.set()
    thread.join(timeout=2)
    assert not thread.is_alive()


def test_cluster_restarts_a_crashed_loop(caplog) -> None:
    cluster = Cluster(node_id="c0", host="127.0.0.1", port=0, node=Node("c0"), config=ClusterConfig(node_id="c0"))
    calls: list[int] = []

    def flaky_loop() -> None:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")

    cluster.running[0] = True
    supervise = cluster._Cluster__supervise  # name-mangled private, exercised directly
    thread = threading.Thread(target=supervise, args=("flaky", flaky_loop))
    thread.start()
    thread.join(timeout=5)
    assert len(calls) == 2
    assert "cluster flaky loop crashed; restarting it" in caplog.text


def test_uncaught_thread_exceptions_are_logged(caplog) -> None:
    configure_logging(force=True)
    logging.getLogger().addHandler(caplog.handler)

    def explode() -> None:
        raise ValueError("thread failure")

    thread = threading.Thread(target=explode, name="membrane-test")
    thread.start()
    thread.join()
    record = next(r for r in caplog.records if r.name == "membrane.threads")
    assert record.levelno == logging.CRITICAL
    assert "membrane-test" in record.getMessage()
    assert record.exc_info is not None and record.exc_info[0] is ValueError


class CountingPeer:
    """Peer double that records calls and holds what it is sent."""

    def __init__(self) -> None:
        self.held: set[str] = set()
        self.calls: list[str] = []

    def inventory_digest(self) -> dict:
        self.calls.append("inventory")
        return dict.fromkeys(self.held, 1)

    def retrieve_fragment(self, content_hash: str):
        self.calls.append("retrieve")
        return object() if content_hash in self.held else None

    def put_blob(self, payload_ref: str, data: bytes) -> bool:
        self.calls.append("put_blob")
        return True

    def request_replicate(self, fragment, is_primary: bool = False) -> bool:
        self.calls.append("replicate")
        self.held.add(fragment.identity.payload_hash)
        return True


def test_replication_sweeps_scale_with_writes_not_data() -> None:
    from unittest.mock import MagicMock

    node = Node("local", max_memory_bytes=10**7)
    for i in range(50):
        node.content_store.put(f"blob-p{i}", b"kv")
        node.store(make_fragment(f"p{i}"), is_primary=True)
    peer = CountingPeer()
    membership = MagicMock()
    membership.get_client.return_value = peer
    shard = MagicMock()
    shard.get_replicas.return_value = ["local", "peer-1"]
    stop = threading.Event()
    replicator = Replicator(
        membership=membership,
        shard=shard,
        node=node,
        config=ClusterConfig(node_id="local", gossip_interval_sec=0.05, repair_interval_sec=3600),
        stop_event=stop,
        running=[True],
    )
    thread = threading.Thread(target=replicator.loop)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while len(peer.held) < 50 and time.monotonic() < deadline:
            time.sleep(0.01)
        # Full pass: one inventory request, then bytes and metadata per
        # missing fragment.
        assert peer.calls.count("inventory") == 1
        assert peer.calls.count("put_blob") == 50
        assert peer.calls.count("replicate") == 50
        assert peer.calls.count("retrieve") == 0
        time.sleep(0.3)  # several idle sweeps
        idle_calls = len(peer.calls)
        assert idle_calls == 101
        node.content_store.put("blob-new", b"kv")
        node.store(make_fragment("new"), is_primary=True)
        deadline = time.monotonic() + 5
        while "new" not in peer.held and time.monotonic() < deadline:
            time.sleep(0.01)
        assert peer.calls[idle_calls:] == ["retrieve", "put_blob", "replicate"]
    finally:
        stop.set()
        thread.join(timeout=2)


def make_gossip(node: Node, **config) -> Gossip:
    ring = Ring()
    shard = Shard(ring)
    return Gossip(
        membership=Membership("local", ring, shard),
        node=node,
        config=ClusterConfig(node_id="local", **config),
        directory=Registry(),
        tombstones=TombstoneTable(),
        stop_event=threading.Event(),
        running=[False],
    )


def test_inventory_summary_tracks_every_change_without_rebuilding() -> None:
    node = Node("local", max_memory_bytes=10**7)
    gossip = make_gossip(node)
    first = gossip.build_state()
    node.store(make_fragment("a"))
    second = gossip.build_state()
    assert second.inventory_size == first.inventory_size + 1
    assert second.inventory_merkle_root != first.inventory_merkle_root
    with node.lock:
        node.remove_fragment("a")
    assert gossip.build_state().inventory_merkle_root == first.inventory_merkle_root  # same set, same root
