"""Tenants that store byte-identical content each keep, and read, their own copy."""

from fastapi.testclient import TestClient

from membrane.auth import AuthContext
from membrane.auth.apikey import APIKeyAuthenticator
from membrane.content_store import FilesystemBlob
from membrane.fragment import Fragment, fragment_key, split_key
from membrane.identity import PayloadIdentity
from membrane.node import Node
from membrane.ring import Ring
from membrane.runtime.components import load_data_key
from membrane.serialization import to_dict
from membrane.server import Server
from membrane.store.tiered import WarmTier
from membrane.transport.ops import op_retrieve, op_store

HASH = "c" * 64
KEYS = "alice-key:alice:read,write\nbob-key:bob:read,write\nroot-key:root:admin\n"


def identity(payload_hash: str = HASH) -> PayloadIdentity:
    return PayloadIdentity(
        payload_hash=payload_hash,
        model_id="m",
        model_revision="",
        tokenizer_name="m",
        tokenizer_revision="",
        layer_range=(0, 1),
        head_range=(-1, -1),
        token_span=(0, 1),
        dtype="float16",
        shape=(1, 1, 1, 1, 64),
    )


def fragment(tenant: str, payload_hash: str = HASH, size: int = 10) -> Fragment:
    return Fragment(
        identity=identity(payload_hash),
        payload_ref=payload_hash,
        payload_size=size,
        ttl=60.0,
        reuse_score=0.5,
        version_id=1,
        tenant_id=tenant,
    )


def caller(tenant: str, *scopes: str) -> AuthContext:
    return AuthContext(subject=tenant, scopes=frozenset(scopes or ("read", "write")))


def test_keys_scope_every_tenant_but_the_default() -> None:
    assert fragment_key("public", HASH) == HASH
    assert fragment_key("acme", HASH) == f"acme:{HASH}"
    assert fragment("acme").key == f"acme:{HASH}"
    assert split_key(f"acme:{HASH}") == ("acme", HASH)
    assert split_key(HASH) == ("public", HASH)


def test_ring_places_every_tenant_copy_with_the_content() -> None:
    ring = Ring()
    for node_id in ("a", "b", "c", "d"):
        ring.add_node(node_id)
    assert ring.get_node(f"acme:{HASH}") == ring.get_node(HASH)
    assert ring.get_nodes(f"acme:{HASH}", 3) == ring.get_nodes(HASH, 3)


def test_two_tenants_with_identical_content_both_hit() -> None:
    node = Node("n", max_memory_bytes=10_000)
    node.content_store.put(HASH, b"kv")
    assert node.store(fragment("alice"), caller_tenant="alice", caller_scopes=frozenset({"write"}))
    assert node.store(fragment("bob"), caller_tenant="bob", caller_scopes=frozenset({"write"}))
    assert set(node.fragments) == {f"alice:{HASH}", f"bob:{HASH}"}
    alice = node.retrieve(HASH, caller_tenant="alice", caller_scopes=frozenset({"read"}))
    bob = node.retrieve(HASH, caller_tenant="bob", caller_scopes=frozenset({"read"}))
    assert alice is not None and alice.tenant_id == "alice"
    assert bob is not None and bob.tenant_id == "bob"
    assert node.memory_usage == 20


def test_a_tenant_cannot_read_another_tenants_copy() -> None:
    node = Node("n", max_memory_bytes=10_000)
    node.store(fragment("alice"))
    read = frozenset({"read"})
    assert node.retrieve(HASH, caller_tenant="bob", caller_scopes=read) is None
    assert node.retrieve(f"alice:{HASH}", caller_tenant="bob", caller_scopes=read) is None
    assert node.locate(HASH, "bob", read) is None
    # The owner, an admin, and an unauthenticated caller all find it.
    assert node.retrieve(HASH, caller_tenant="alice", caller_scopes=read) is not None
    assert node.retrieve(HASH, caller_tenant="ops", caller_scopes=frozenset({"admin"})) is not None
    assert node.retrieve(HASH) is not None
    assert node.locate(HASH) == f"alice:{HASH}"


def test_own_copy_first_then_the_default_tenant() -> None:
    node = Node("n", max_memory_bytes=10_000)
    node.store(fragment("public"))
    read = frozenset({"read"})
    found = node.retrieve(HASH, caller_tenant="alice", caller_scopes=read)
    assert found is not None and found.tenant_id == "public"
    node.store(fragment("alice"))
    found = node.retrieve(HASH, caller_tenant="alice", caller_scopes=read)
    assert found is not None and found.tenant_id == "alice"
    assert node.locate(HASH, "alice", read) == f"alice:{HASH}"


def test_shared_blob_outlives_one_tenants_copy() -> None:
    node = Node("n", max_memory_bytes=10_000)
    node.content_store.put(HASH, b"kv")
    node.store(fragment("alice"))
    node.store(fragment("bob"))
    node.remove_fragment(f"alice:{HASH}")
    assert node.content_store.has(HASH)
    assert node.retrieve(HASH, caller_tenant="bob", caller_scopes=frozenset({"read"})) is not None
    node.remove_fragment(f"bob:{HASH}")
    assert not node.content_store.has(HASH)
    assert node.table.tenant_keys == {} and node.table.payload_refs == {}


def test_store_and_retrieve_ops_per_tenant() -> None:
    node = Node("n", max_memory_bytes=10_000)
    node.content_store.put(HASH, b"kv")
    for tenant in ("alice", "bob"):
        status, body = op_store(node, to_dict(fragment(tenant)), cluster_metrics=None, auth_context=caller(tenant))
        assert status == 200 and body["success"] is True
    for tenant in ("alice", "bob"):
        _status, body = op_retrieve(node, HASH, auth_context=caller(tenant))
        assert body["found"] is True
        assert body["fragment"]["tenant_id"] == tenant
    _status, body = op_retrieve(node, HASH, auth_context=caller("carol"))
    assert body["found"] is False


def test_warm_tier_keeps_a_shared_blob_until_the_last_copy_goes(tmp_path) -> None:
    store = FilesystemBlob(tmp_path / "warm", tenant_id="w", key_provider=load_data_key(tmp_path, ""))
    tier = WarmTier(store, 1_000)
    alice, bob = fragment("alice"), fragment("bob")
    assert tier.demote(alice, b"kv") and tier.demote(bob, b"kv")
    promoted = tier.promote(alice.key)
    assert promoted is not None and promoted[1] == b"kv"
    # Promoting alice's copy must not take the bytes bob's copy still needs.
    promoted = tier.promote(bob.key)
    assert promoted is not None and promoted[1] == b"kv"
    assert tier.size_bytes == 0 and not store.has(HASH)


def test_memory_api_serves_each_tenant_its_own_prefill() -> None:
    server = Server(
        node=Node("mem", max_memory_bytes=1 << 26),
        port=0,
        load_hooks=False,
        authenticator=APIKeyAuthenticator(keyfile_text=KEYS),
    )
    client = TestClient(server.transport.app)
    tokens = list(range(256))
    for key in ("alice-key", "bob-key"):
        headers = {"Authorization": f"Bearer {key}"}
        assert (
            client.post("/prefill", json={"prompt_tokens": tokens, "model_id": "m"}, headers=headers).status_code == 200
        )
    for key, tenant in (("alice-key", "alice"), ("bob-key", "bob")):
        headers = {"Authorization": f"Bearer {key}"}
        body = client.post("/reconstruct", json={"tokens": tokens, "model_id": "m"}, headers=headers).json()
        assert body["coverage"] == 1.0
        assert {f["tenant_id"] for f in body["fragments"]} == {tenant}
    tenants = sorted({split_key(k)[0] for k in server.node.fragments})
    assert tenants == ["alice", "bob"]


def test_repair_replicates_every_tenants_copy() -> None:
    from tests.membrane.test_scale_out import replicator_with_peer

    replicator, local, peer = replicator_with_peer()
    local.content_store.put(HASH, b"kv")
    for tenant in ("alice", "bob"):
        local.store(fragment(tenant), is_primary=True)
    keys = sorted(local.fragments)
    assert replicator.push_missing("b", keys) == set()
    assert sorted(peer.node.fragments) == keys
    assert peer.node.content_store.get(HASH) == b"kv"
    for tenant in ("alice", "bob"):
        copy = peer.node.retrieve(HASH, caller_tenant=tenant, caller_scopes=frozenset({"read"}))
        assert copy is not None and copy.tenant_id == tenant
