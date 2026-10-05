"""A node configured with --redis and --data-dir survives a restart.

Before this, ``--redis`` built a persistence backend that nothing used,
and Redis records dropped ``tenant_id`` (a restored fragment would
have become public).
"""

import pytest

from membrane.auth import AuthContext
from membrane.compute.cpu import CPU
from membrane.node import Node
from membrane.persistence.redis import Redis
from membrane.server import Server, build_content_store
from membrane.transport.ops import op_prefill, op_retrieve

REDIS_URL = "redis://localhost:6379/0"


@pytest.fixture(autouse=True)
def _redis():
    backend = Redis(REDIS_URL)
    if not backend.ping():
        pytest.skip("Redis server not available")
    backend.flush()
    yield
    backend.flush()


def _server(data_dir: str) -> Server:
    node = Node("durable-0", max_memory_bytes=10_000_000, content_store=build_content_store(data_dir))
    return Server(node=node, redis_url=REDIS_URL, host="127.0.0.1", port=0)


def test_fragments_survive_restart_with_tenant(tmp_path) -> None:
    first = _server(str(tmp_path))
    assert first.durable
    acme = AuthContext(subject="acme", scopes=frozenset({"read", "write"}))
    status, body = op_prefill(first.node, CPU(), list(range(300)), "m", auth_context=acme)
    assert status == 200
    hashes = [f["identity"]["payload_hash"] for f in body["fragments"]]

    second = _server(str(tmp_path))  # fresh process: empty node, same Redis + data dir
    assert second.node.fragments == {}
    assert second.restore_fragments() == len(hashes)
    for content_hash in hashes:
        _, found = op_retrieve(second.node, content_hash, auth_context=acme)
        assert found["found"] is True
        assert found["fragment"]["tenant_id"] == "acme"
        globex = AuthContext(subject="globex", scopes=frozenset({"read"}))
        assert op_retrieve(second.node, content_hash, auth_context=globex)[1]["found"] is False


def test_evicted_fragments_are_not_restored(tmp_path) -> None:
    first = _server(str(tmp_path))
    _, body = op_prefill(first.node, CPU(), list(range(10)), "m")
    content_hash = body["fragments"][0]["identity"]["payload_hash"]
    first.node.remove_fragment(content_hash)

    second = _server(str(tmp_path))
    assert second.restore_fragments() == 0


def test_data_dir_key_is_private_and_stable(tmp_path) -> None:
    import stat

    build_content_store(str(tmp_path))
    key = tmp_path / "master.key"
    assert stat.S_IMODE(key.stat().st_mode) == 0o600
    first_key = key.read_bytes()
    build_content_store(str(tmp_path))
    assert key.read_bytes() == first_key
