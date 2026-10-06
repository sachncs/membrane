"""The Redis backend against Sentinel and Redis Cluster.

URL parsing is tested everywhere. The live tests run when
``MEMBRANE_TEST_SENTINEL_URL`` / ``MEMBRANE_TEST_CLUSTER_URL`` point at real
deployments (``scripts/redis_topologies.sh`` starts both in Docker; CI runs it).
"""

import os

import pytest

from membrane.persistence.redis import Redis, connect, parse_multi_host
from tests.conftest import make_fragment


def test_parse_multi_host_urls() -> None:
    assert parse_multi_host("redis+sentinel://:pw@s1:26379,s2,[::1]:26380/mymaster/2?socket_timeout=1") == (
        "redis+sentinel",
        None,
        "pw",
        [("s1", 26379), ("s2", 26379), ("::1", 26380)],
        ["mymaster", "2"],
        {"socket_timeout": "1"},
    )
    assert parse_multi_host("rediss+cluster://app:secret@n1:7000,n2")[1:4] == (
        "app",
        "secret",
        [("n1", 7000), ("n2", 6379)],
    )
    with pytest.raises(ValueError, match="no hosts"):
        parse_multi_host("redis+cluster://")


def test_connect_builds_the_right_client(monkeypatch) -> None:
    import redis.cluster
    import redis.sentinel

    created: dict[str, object] = {}

    class FakeCluster:
        def __init__(self, **kwargs) -> None:
            created["cluster"] = kwargs

    class FakeSentinel:
        def __init__(self, hosts, **kwargs) -> None:
            created["sentinel"] = (hosts, kwargs)

        def master_for(self, service, **kwargs):
            created["master"] = (service, kwargs)
            return "master-client"

    monkeypatch.setattr(redis.cluster, "RedisCluster", FakeCluster)
    monkeypatch.setattr(redis.sentinel, "Sentinel", FakeSentinel)

    client, is_cluster = connect("rediss+cluster://:pw@n1:7000,n2:7001")
    assert is_cluster and isinstance(client, FakeCluster)
    nodes = created["cluster"]["startup_nodes"]  # type: ignore[index]
    assert [(n.host, n.port) for n in nodes] == [("n1", 7000), ("n2", 7001)]
    assert created["cluster"]["ssl"] is True and created["cluster"]["password"] == "pw"  # type: ignore[index]

    client, is_cluster = connect("redis+sentinel://s1:26379/mymaster/3")
    assert client == "master-client" and not is_cluster
    assert created["sentinel"][0] == [("s1", 26379)]  # type: ignore[index]
    assert created["master"][0] == "mymaster" and created["master"][1]["db"] == 3  # type: ignore[index]
    with pytest.raises(ValueError, match="service"):
        connect("redis+sentinel://s1:26379")


def exercise(backend: Redis) -> None:
    """Write, read, list, and delete through the backend."""
    backend.flush()
    fragment = make_fragment("topology-a", (0, 3))
    assert backend.store_fragment(fragment, node_id="n1", is_primary=True)
    assert backend.retrieve_fragment("topology-a") is not None
    assert backend.get_primary("topology-a") == "n1"
    assert backend.list_node_fragments("n1") == {"topology-a"}
    assert backend.inventory_digest("n1") == {"topology-a": 1}
    assert backend.delete_fragment("topology-a")
    assert backend.retrieve_fragment("topology-a") is None
    assert backend.ping()


@pytest.mark.skipif("MEMBRANE_TEST_SENTINEL_URL" not in os.environ, reason="no Sentinel deployment")
def test_backend_through_sentinel() -> None:
    backend = Redis(os.environ["MEMBRANE_TEST_SENTINEL_URL"], prefix="membrane-sentinel-test:")
    assert not backend.cluster
    exercise(backend)


@pytest.mark.skipif("MEMBRANE_TEST_CLUSTER_URL" not in os.environ, reason="no Redis Cluster deployment")
def test_backend_on_redis_cluster() -> None:
    backend = Redis(os.environ["MEMBRANE_TEST_CLUSTER_URL"], prefix="membrane-cluster-test:")
    assert backend.cluster
    exercise(backend)
