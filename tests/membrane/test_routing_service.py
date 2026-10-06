"""Routing and background policies: /route, placement plugins, promotion, roles, origin read-through."""

import json
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

from membrane.compute.hashing import token_hash
from membrane.fragment import Fragment
from membrane.identity import PayloadIdentity
from membrane.network.membership import Membership
from membrane.network.peer import Peer, RawResponse
from membrane.node import Node, NodeAttributes
from membrane.registry import Registry
from membrane.replica import Replica
from membrane.ring import Ring
from membrane.roles import NodeRole
from membrane.runtime.plugins import PLACEMENT
from membrane.runtime.settings import ServerSettings, SettingsError, build_server
from membrane.server import Server
from membrane.services import ServiceOptions
from membrane.services.memory import MemoryService
from membrane.services.placement import (
    ClusterView,
    LatencyPlacement,
    PlacementService,
    RingPlacement,
    SelectorPlacement,
)
from membrane.services.policies import OriginLink, Promoter, RolePolicy
from membrane.shard import Shard
from membrane.wire.v3.chunks import sha256_hex


class FakeCluster:
    """The parts of a cluster manager the services read."""

    def __init__(self, local: str) -> None:
        self.hash_ring = Ring()
        self.hash_ring.add_node(local)
        self.directory = Registry()
        self.membership = Membership(local, self.hash_ring, Shard(self.hash_ring), self.directory)


class FakePeer:
    """A peer backed by an in-process node."""

    def __init__(self, node: Node) -> None:
        self.node = node

    def put_blob(self, payload_ref, data):
        self.node.content_store.put(payload_ref, data)
        return True

    def blob_digest(self, payload_ref):
        data = self.node.content_store.get(payload_ref)
        return sha256_hex(data) if data is not None else None

    def request_replicate(self, fragment, is_primary=False):
        return self.node.store(fragment, is_primary=is_primary)

    def retrieve_fragment(self, content_hash):
        return self.node.retrieve(content_hash)


def fragment(content_hash: str, reuse: float = 0.5, span: tuple[int, int] = (0, 3)) -> Fragment:
    identity = PayloadIdentity(
        payload_hash=content_hash,
        model_id="m",
        model_revision="",
        tokenizer_name="m",
        tokenizer_revision="",
        layer_range=(0, 1),
        head_range=(-1, -1),
        token_span=span,
        dtype="float16",
        shape=(1, 1, 4, 1, 64),
    )
    return Fragment(
        identity=identity, payload_ref=content_hash, payload_size=4, ttl=600.0, reuse_score=reuse, version_id=1
    )


def cluster_with_peers(local: Node) -> FakeCluster:
    cluster = FakeCluster(local.node_id)
    for name, rtt, load in (("b", 10.0, 0.5), ("c", 50.0, 0.1)):
        cluster.membership.add(name, "127.0.0.1", 9000 + ord(name))
        cluster.membership.record_heartbeat(
            name, report={"load": load, "gpu_load": 0.2, "attributes": {"region": "eu"}}, round_trip_ms=rtt
        )
    return cluster


def test_heartbeat_report_becomes_telemetry() -> None:
    local = Node("a", attributes=NodeAttributes(region="us"))
    view = ClusterView(local, cluster_with_peers(local))
    telemetry = view.telemetry()
    assert set(telemetry) == {"a", "b", "c"}
    assert telemetry["b"].latency_ms == 10.0
    assert telemetry["b"].memory_pressure == 0.5
    assert telemetry["b"].bandwidth_cost == 0.5  # another region
    assert telemetry["a"].latency_ms == 0.0
    info = view.cluster.membership.find("b")
    assert (info.region, info.gpu_load) == ("eu", 0.2)
    # A later slow round trip moves the average, not the whole value.
    view.cluster.membership.record_heartbeat("b", round_trip_ms=110.0)
    assert view.cluster.membership.find("b").latency_ms == pytest.approx(30.0)


def test_placement_policies() -> None:
    local = Node("a")
    view = ClusterView(local, cluster_with_peers(local))
    view.cluster.directory.record_fragment_location("h1", "b")
    view.cluster.directory.record_fragment_location("h1", "c")
    assert LatencyPlacement().place(view, "h1", []).fetch_from == "b"  # 10 ms beats 50 ms
    assert SelectorPlacement().place(view, "h1", []).fetch_from == "c"  # less loaded
    ring = RingPlacement().place(view, "h1", [])
    assert ring.fetch_from in {"b", "c"} and ring.prefill_on == "a"
    for name in ("ring", "latency", "selector", "economic", "joint"):
        decision = PLACEMENT.get(name)().place(view, "h1", [])
        assert decision.fetch_from in {"b", "c"}, name
        assert decision.store_on in {"a", "b", "c"}, name
    # A copy on this node always wins on latency.
    local.content_store.put("h1", b"data")
    local.store(fragment("h1"))
    assert LatencyPlacement().place(view, "h1", []).fetch_from == "a"


def test_route_matches_prompt_across_the_cluster() -> None:
    local = Node("a")
    memory = MemoryService(local)
    view = ClusterView(local, cluster_with_peers(local))
    service = PlacementService(view, LatencyPlacement(), "latency", memory=memory, route_threshold=100)
    tokens = list(range(300))
    first = Fragment(
        identity=fragment(token_hash(tokens[:128]), span=(0, 127)).identity,
        payload_ref=None,
        payload_size=0,
        ttl=600.0,
        reuse_score=0.5,
        version_id=1,
    )
    local.store(first)
    view.cluster.directory.record_fragment_location(token_hash(tokens[128:256]), "c")
    answer = service.route(tokens=tokens, model_id="m", local_cached_tokens=0)
    assert answer.matched_tokens == 256
    assert [(f["node_id"], f["url"]) for f in answer.fragments] == [("a", None), ("c", "http://127.0.0.1:9099")]
    assert answer.offload == {
        "target": "membrane",
        "incremental_length": 44,
        "cached_prefix_length": 256,
        "cross_cluster_cache_transfer": False,
        "threshold": 100,
    }
    body = service.body(answer)
    assert body["placement"]["prefill_on_url"] is None
    assert body["node_id"] == "a"
    # Short prompts stay on the caller's own engine.
    assert service.route(tokens=list(range(50)), model_id="m").offload["target"] == "pd-p"


def test_routing_threshold_adapts_to_load() -> None:
    local = Node("a")
    service = PlacementService(ClusterView(local), RingPlacement(), route_threshold=1000)
    assert service.adjust(queue_depth=50, max_queue_depth=50) == 1100  # saturated: offload less
    for _ in range(50):
        service.adjust(queue_depth=0, max_queue_depth=50)
    assert service.scheduler.state.effective_threshold < 1100
    assert service.reoptimize() is None  # too few observations yet
    assert PlacementService(ClusterView(local), RingPlacement()).adjust(1, 1) is None


def test_route_endpoint() -> None:
    server = Server(node=Node("r"), port=0, load_hooks=False, services=ServiceOptions(route_threshold=64))
    client = TestClient(server.transport.app)
    client.post("/prefill", json={"prompt_tokens": list(range(128)), "model_id": "m"})
    body = client.post("/route", json={"tokens": list(range(200)), "model_id": "m"}).json()
    assert body["matched_tokens"] == 128
    assert body["reuse"] is True
    assert body["offload"]["target"] == "membrane"
    assert body["placement"]["prefill_on"] == "r"
    hashed = client.post("/route", json={"content_hash": token_hash(list(range(128)))}).json()
    assert hashed["placement"]["fetch_from"] == "r"
    assert client.post("/route", json={}).status_code == 400


def test_promoter_copies_hot_fragments_to_least_loaded_peers() -> None:
    local = Node("a")
    cluster = cluster_with_peers(local)
    peers = {name: Node(name) for name in ("b", "c")}
    for name, node in peers.items():
        cluster.membership.clients[name] = FakePeer(node)  # type: ignore[assignment]
    memory = MemoryService(local)
    hot, cold = fragment("hot" * 8, reuse=0.9), fragment("cold" * 8, reuse=0.9)
    for frag in (hot, cold):
        local.content_store.put(frag.payload_ref, b"kv-bytes")
        local.store(frag)
    for _ in range(3):
        memory.record_access(hot.identity.payload_hash)
    memory.record_access(cold.identity.payload_hash)  # below the demand threshold
    promoter = Promoter(memory, ClusterView(local, cluster), max_replicas=2)
    assert promoter.run_once() == 1
    assert "hot" * 8 in peers["c"].fragments  # c is less loaded than b
    assert peers["c"].content_store.get("hot" * 8) == b"kv-bytes"
    assert not peers["b"].fragments
    assert cluster.directory.locate_fragment("hot" * 8) == {"c"}
    assert promoter.run_once() == 0  # counts were consumed


def test_role_policy_follows_load() -> None:
    policy = RolePolicy(ClusterView(Node("a"), gpu_load=lambda: 0.9))
    assert policy.run_once() is NodeRole.PREFILL_WORKER
    assert policy.changes == 1
    assert RolePolicy(ClusterView(Node("a"))).run_once() is NodeRole.MEMORY_HOST


class AppTransport:
    """Peer transport that calls an in-process app."""

    def __init__(self, client: TestClient) -> None:
        self.client = client

    def request(self, method, url, body, headers, timeout_sec):
        parts = urlsplit(url)
        path = parts.path + (f"?{parts.query}" if parts.query else "")
        response = self.client.request(method, path, content=body, headers=headers)
        return json.loads(response.text) if response.text else {}

    def request_bytes(self, method, url, body, headers, timeout_sec):
        response = self.client.request(method, urlsplit(url).path, headers=headers)
        return RawResponse(response.status_code, {k.lower(): v for k, v in response.headers.items()}, response.content)


def test_regional_cache_reads_through_to_origin() -> None:
    origin = Server(node=Node("origin"), port=0, load_hooks=False)
    origin_client = TestClient(origin.transport.app)
    stored = origin_client.post("/prefill", json={"prompt_tokens": list(range(16)), "model_id": "m"}).json()
    content_hash = stored["fragments"][0]["identity"]["payload_hash"]

    cache = Server(node=Replica("eu-cache"), port=0, load_hooks=False)
    peer = Peer("http://origin:8080", transport=AppTransport(origin_client), max_retries=1, retry_delay_sec=0.0)
    cache.services.origin = OriginLink(cache.node, "origin:8080", peer=peer)
    client = TestClient(cache.transport.app)
    assert client.get(f"/retrieve?content_hash={content_hash}").json()["found"] is True
    assert content_hash in cache.node.fragments
    assert content_hash not in cache.node.primary_hashes  # a regional copy is never primary
    assert cache.services.origin.fetched == 1
    assert client.get("/retrieve?content_hash=" + "f" * 32).json()["found"] is False


def test_settings_for_services() -> None:
    with pytest.raises(SettingsError, match="placement"):
        ServerSettings(placement="nope")
    with pytest.raises(SettingsError, match="regional cache"):
        ServerSettings(origin="o:1", peers=("p:2",))
    with pytest.raises(SettingsError, match="HOST:PORT"):
        ServerSettings(origin="origin")
    server, _mode = build_server(
        ServerSettings(port=0, region="eu", origin="127.0.0.1:1", placement="latency", load_hooks=False)
    )
    assert isinstance(server.node, Replica)
    assert server.node.attributes.region == "eu"
    assert server.services.placement.policy_name == "latency"
    assert server.services.origin is not None
