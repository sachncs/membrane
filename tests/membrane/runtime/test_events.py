"""EventBus delivery, server events, hooks, and the new plugin registries."""

from importlib.metadata import EntryPoint, EntryPoints

import pytest

from membrane.network.membership import Membership
from membrane.node import Node
from membrane.ring import Ring
from membrane.runtime import plugins
from membrane.runtime.events import EventBus, FragmentRemoved, FragmentStored, PeerJoined, PeerLeft
from membrane.runtime.settings import ServerSettings, SettingsError, build_server
from membrane.server import Server
from membrane.shard import Shard
from membrane.store.eviction import FrequencyLRU
from tests.conftest import make_fragment


def sample_hook(bus: EventBus, server: Server) -> None:
    """A ``membrane.hooks`` factory used through a fake entry point."""
    calls: list[object] = [server]
    server.hook_calls = calls  # type: ignore[attr-defined]
    bus.subscribe(FragmentStored, calls.append)


def test_bus_delivers_in_order_and_isolates_failures() -> None:
    bus = EventBus()
    seen: list[object] = []

    def broken(_event: object) -> None:
        raise RuntimeError("subscriber bug")

    bus.subscribe(FragmentStored, broken)
    bus.subscribe(FragmentStored, seen.append)
    bus.subscribe(object, lambda e: seen.append(("any", type(e).__name__)))
    for i in range(3):
        bus.publish(FragmentStored(f"h{i}", "t", False))
    bus.publish(FragmentRemoved("h0"))
    assert bus.flush(2.0)
    stored = [e.content_hash for e in seen if isinstance(e, FragmentStored)]
    assert stored == ["h0", "h1", "h2"]
    assert ("any", "FragmentRemoved") in seen
    bus.close()


def test_bus_drops_when_full() -> None:
    bus = EventBus(capacity=1)
    bus.subscribe(FragmentStored, lambda _e: __import__("time").sleep(0.2))
    for i in range(5):
        bus.publish(FragmentStored(f"h{i}", "t", False))
    assert bus.dropped >= 3
    bus.close()


def test_server_publishes_node_events() -> None:
    server = Server(node=Node("ev-0"), port=0, load_hooks=False)
    seen: list[object] = []
    server.event_bus.subscribe(object, seen.append)
    server.node.store(make_fragment("e1"), is_primary=True)
    server.node.remove_fragment("e1")
    assert server.event_bus.flush(2.0)
    assert seen == [FragmentStored("e1", "public", True), FragmentRemoved("e1")]
    server.stop(1.0)


def test_installed_hooks_run_at_build(monkeypatch) -> None:
    fake = EntryPoint("sample", "tests.membrane.runtime.test_events:sample_hook", "membrane.hooks")
    monkeypatch.setattr(
        plugins,
        "entry_points",
        lambda *, group, name=None: EntryPoints(
            [fake] if group == "membrane.hooks" and name in (None, "sample") else []
        ),
    )
    server = Server(node=Node("hook-0"), port=0)
    assert server.hook_calls == [server]
    server.node.store(make_fragment("e2"))
    assert server.event_bus.flush(2.0)
    assert any(isinstance(e, FragmentStored) for e in server.hook_calls)
    server.stop(1.0)


def test_membership_listeners_see_joins_and_leaves() -> None:
    ring = Ring()
    membership = Membership("self", ring, Shard(ring))
    changes: list[tuple[str, str]] = []
    membership.listeners.append(lambda change, node_id: changes.append((change, node_id)))
    membership.add("p1", "127.0.0.1", 9001)
    membership.add("self", "127.0.0.1", 9000)  # the local node: ignored
    membership.remove("p1")
    membership.remove("p1")  # already gone: no event
    assert changes == [("joined", "p1"), ("left", "p1")]
    assert PeerJoined("p1") != PeerLeft("p1")


def test_settings_select_eviction_and_validate_plugins() -> None:
    server, _ = build_server(ServerSettings(port=0, eviction="tinylfu", load_hooks=False))
    assert isinstance(server.node.eviction_policy, FrequencyLRU)
    assert not server.durable
    with pytest.raises(SettingsError, match="unknown eviction policy"):
        ServerSettings(eviction="random")
    with pytest.raises(SettingsError, match="unknown persistence backend"):
        ServerSettings(persistence="postgres")
    assert {"memory", "redis"} <= set(plugins.PERSISTENCE.names())
    assert {"env", "aws", "gcp", "vault"} <= set(plugins.SECRET_PROVIDERS.names())
