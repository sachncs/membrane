"""Tests for membrane.runtime: plugins, write-behind persistence, lifecycle, file checks."""

import os
import threading
import time
from importlib.metadata import EntryPoint, EntryPoints

import pytest
from fastapi.testclient import TestClient

from membrane.metrics import MetricsCollector, PersistenceMetrics
from membrane.node import Node
from membrane.runtime import plugins
from membrane.runtime.lifecycle import PeriodicTask
from membrane.runtime.observability import EventLog
from membrane.runtime.persistence_writer import PersistenceWriter
from membrane.runtime.plugins import COMPUTE_BACKENDS, PluginRegistry, UnknownPluginError
from membrane.security.files import InsecureFileError, require_private_file
from membrane.server import Server
from tests.conftest import make_fragment


class TestPluginRegistry:
    def test_builtins_are_listed(self) -> None:
        assert {"cpu", "gpu", "ollama", "openai", "anthropic", "transformers"} <= set(COMPUTE_BACKENDS.names())
        assert "cpu" in COMPUTE_BACKENDS

    def test_unknown_name_lists_available(self) -> None:
        with pytest.raises(UnknownPluginError, match=r"available: .*cpu"):
            COMPUTE_BACKENDS.get("no-such-backend")

    def test_entry_point_plugin_loads(self, monkeypatch) -> None:
        fake = EntryPoint("echo", "membrane.runtime.plugins:cpu_backend", "membrane.test")

        def fake_entry_points(*, group: str, name: str | None = None) -> EntryPoints:
            found = [fake] if group == "membrane.test" and name in (None, "echo") else []
            return EntryPoints(found)

        monkeypatch.setattr(plugins, "entry_points", fake_entry_points)
        registry = PluginRegistry[object]("membrane.test", "test plugin")
        assert registry.names() == ["echo"]
        assert registry.get("echo") is plugins.cpu_backend

    def test_builtin_wins_over_entry_point(self, monkeypatch) -> None:
        shadow = EntryPoint("cpu", "membrane.runtime.plugins:gpu_backend", "membrane.compute")
        monkeypatch.setattr(plugins, "entry_points", lambda *, group, name=None: EntryPoints([shadow]))
        assert COMPUTE_BACKENDS.get("cpu") is plugins.cpu_backend

    def test_server_resolves_compute_by_name(self) -> None:
        server = Server(node=Node("plug-0"), compute="cpu", port=0)
        assert server.compute_type == "cpu"
        with pytest.raises(ValueError, match="unknown compute backend"):
            Server(node=Node("plug-1"), compute="nope", port=0)


class FlakyPersistence:
    """Fails the first ``failures`` writes, then records them in order."""

    def __init__(self, failures: int = 0, delay_sec: float = 0.0) -> None:
        self.failures = failures
        self.delay_sec = delay_sec
        self.log: list[tuple[str, str]] = []

    def store_fragment(self, fragment, node_id: str, is_primary: bool = False) -> bool:
        time.sleep(self.delay_sec)
        if self.failures > 0:
            self.failures -= 1
            return False
        self.log.append(("store", fragment.identity.payload_hash))
        return True

    def forget_on_node(self, content_hash: str, node_id: str) -> bool:
        if self.failures > 0:
            self.failures -= 1
            return False
        self.log.append(("forget", content_hash))
        return True


def writer_for(backend: FlakyPersistence, **kwargs) -> tuple[PersistenceWriter, PersistenceMetrics]:
    metrics = PersistenceMetrics(MetricsCollector())
    return PersistenceWriter(backend, "n0", metrics, initial_backoff_sec=0.01, max_backoff_sec=0.02, **kwargs), metrics


class TestPersistenceWriter:
    def test_applies_operations_in_order(self) -> None:
        backend = FlakyPersistence()
        writer, _ = writer_for(backend)
        first, second = make_fragment("h1"), make_fragment("h2")
        writer.store(first, True)
        writer.forget(first.identity.payload_hash)
        writer.store(second, False)
        assert writer.flush(2.0)
        assert backend.log == [
            ("store", first.identity.payload_hash),
            ("forget", first.identity.payload_hash),
            ("store", second.identity.payload_hash),
        ]
        assert writer.stop(1.0)

    def test_survives_outage_and_flushes_after(self) -> None:
        backend = FlakyPersistence(failures=5)
        writer, metrics = writer_for(backend)
        writer.store(make_fragment("h3"), True)
        writer.forget("abc")
        assert writer.flush(3.0)
        assert [kind for kind, _ in backend.log] == ["store", "forget"]
        assert metrics.operations.get(kind="store", outcome="error") == 5
        writer.stop(1.0)

    def test_full_queue_drops_and_counts(self) -> None:
        backend = FlakyPersistence(delay_sec=0.2)
        writer, metrics = writer_for(backend, capacity=1)
        for seed in range(5):
            writer.store(make_fragment(f"h{seed}"), False)
        assert metrics.dropped.get(kind="store") >= 3
        writer.stop(2.0)

    def test_hook_does_not_block_node_lock(self) -> None:
        backend = FlakyPersistence(delay_sec=0.5)
        writer, _ = writer_for(backend)
        node = Node("wb-0")
        node.set_persistence_hooks(writer.store, writer.forget)
        start = time.monotonic()
        for seed in range(5):
            assert node.store(make_fragment(f"h{seed}"))
        assert time.monotonic() - start < 0.4
        writer.stop(0.1)

    def test_stop_counts_unwritten(self) -> None:
        backend = FlakyPersistence(failures=10_000)
        writer, metrics = writer_for(backend)
        writer.store(make_fragment("h9"), True)
        writer.forget("x")
        assert writer.stop(0.1) is False
        assert metrics.dropped.get(kind="store") + metrics.dropped.get(kind="forget") == 2
        assert writer.pending == 0


class TestLifecycle:
    def test_periodic_task_runs_and_stops(self) -> None:
        ran = threading.Event()
        calls: list[int] = []

        def action() -> None:
            calls.append(1)
            ran.set()
            raise RuntimeError("logged, not fatal")

        task = PeriodicTask("t", 0.01, action)
        task.start()
        assert ran.wait(1.0)
        time.sleep(0.05)
        assert task.stop(1.0)
        assert len(calls) >= 2
        assert not task.running

    def test_event_log_is_bounded(self) -> None:
        log = EventLog(capacity=3)
        for i in range(5):
            log.record("info", f"e{i}")
        assert len(log) == 3
        assert [e.message for e in log.recent(10)] == ["e2", "e3", "e4"]
        assert log.recent(0) == []

    def test_readyz_reports_draining(self) -> None:
        server = Server(node=Node("drain-0"), port=0)
        client = TestClient(server.transport.app)
        assert client.get("/readyz").status_code == 200
        server.is_draining = True
        response = client.get("/readyz")
        assert response.status_code == 503
        assert response.json() == {"status": "draining"}

    def test_stop_is_idempotent(self) -> None:
        server = Server(node=Node("stop-0"), port=0)
        assert server.stop(1.0)
        assert server.stop(1.0)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
class TestPrivateFiles:
    def test_owner_only_passes(self, tmp_path) -> None:
        secret = tmp_path / "k"
        secret.write_text("x")
        secret.chmod(0o600)
        require_private_file(secret, "key")

    def test_world_readable_refused(self, tmp_path) -> None:
        secret = tmp_path / "k"
        secret.write_text("x")
        secret.chmod(0o644)
        with pytest.raises(InsecureFileError, match="chmod 600"):
            require_private_file(secret, "key")

    def test_group_readable_warns(self, tmp_path, caplog) -> None:
        secret = tmp_path / "k"
        secret.write_text("x")
        secret.chmod(0o640)
        require_private_file(secret, "key")
        assert "group-readable" in caplog.text
