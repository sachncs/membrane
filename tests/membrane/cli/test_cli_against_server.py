"""CLI commands run against a live in-process server, plus the dashboard and setup wizard."""

import importlib
import json
import time
import types
from collections.abc import Iterator

import pytest
from typer.testing import CliRunner

from membrane.auth.apikey import APIKeyAuthenticator
from membrane.cli import app, wizard
from membrane.cli.formatters import fmt_bytes, fmt_duration
from membrane.node import Node
from membrane.policy import Promotion
from membrane.server import Server
from tests.conftest import make_fragment

# The package exports the ``dashboard`` command under the same name.
dashboard = importlib.import_module("membrane.cli.dashboard")
ADMIN_KEY = "root-key"


@pytest.fixture(scope="module")
def server() -> Iterator[Server]:
    srv = Server(
        node=Node("cli", max_memory_bytes=1 << 20),
        port=0,
        load_hooks=False,
        authenticator=APIKeyAuthenticator(keyfile_text=f"{ADMIN_KEY}:root:admin\n"),
    )
    srv.start()
    deadline = time.monotonic() + 15
    while not srv.transport.port or dashboard.fetch_json("127.0.0.1", srv.transport.port, "/livez") == {}:
        assert time.monotonic() < deadline, "server did not start"
        time.sleep(0.05)
    yield srv
    srv.stop()


@pytest.fixture
def run(server: Server, monkeypatch):
    monkeypatch.setenv("MEMBRANE_API_KEY", ADMIN_KEY)
    runner = CliRunner()

    def invoke(*args: str):
        return runner.invoke(app, [*args, "--host", "127.0.0.1", "--port", str(server.transport.port)])

    return invoke


def test_llm_status_names_the_backend(run) -> None:
    result = run("llm-status")
    assert result.exit_code == 0, result.output
    assert "cpu" in result.stdout and "LLM Backend Status" in result.stdout


def test_cluster_status_without_peers(run) -> None:
    result = run("cluster-status")
    assert result.exit_code == 0
    assert "No peers connected." in result.stdout


def test_cluster_status_renders_peers(monkeypatch) -> None:
    peers = {"peers": [{"node_id": "n1", "host": "10.0.0.1", "port": 8080, "healthy": True}, {"node_id": "n2"}]}
    monkeypatch.setattr("membrane.cli.commands.cluster.fetch_json", lambda *a: peers)
    result = CliRunner().invoke(app, ["cluster-status"])
    assert result.exit_code == 0
    assert "n1" in result.stdout and "10.0.0.1" in result.stdout and "n2" in result.stdout


@pytest.mark.parametrize("command", ["cluster-status", "llm-status"])
def test_status_commands_fail_when_unreachable(command: str) -> None:
    result = CliRunner().invoke(app, [command, "--host", "127.0.0.1", "--port", "1"])
    assert result.exit_code == 1


def test_admin_commands_reach_the_admin_routes(run, server: Server) -> None:
    server.node.content_store.put("blob-" + "e" * 32, b"kv")
    server.node.store(make_fragment("e" * 32, (0, 3)))
    inspected = run("admin", "inspect", "e" * 32)
    assert inspected.exit_code == 0 and inspected.stdout.startswith("200 ")
    assert json.loads(inspected.stdout[4:])["content_hash"] == "e" * 32
    assert run("admin", "placement", "e" * 32, "other").stdout.startswith("503 ")  # no cluster manager
    assert run("admin", "repair", "peer-x").exit_code == 0
    evicted = run("admin", "evict", "e" * 32)
    assert evicted.stdout.startswith("200 ")
    assert "e" * 32 not in server.node.fragments
    assert run("admin", "inspect", "e" * 32).stdout.startswith("404 ")


def test_admin_policy_reads_and_updates(run, server: Server, monkeypatch) -> None:
    assert "promotion is off" in json.loads(run("admin", "policy").stdout)["detail"]
    monkeypatch.setattr(server.services, "promoter", types.SimpleNamespace(policy=Promotion()))
    updated = run("admin", "policy", "--min-reuse-score", "0.25", "--demand-threshold", "3")
    assert updated.exit_code == 0, updated.output
    current = json.loads(run("admin", "policy").stdout)
    assert current == {"min_reuse_score": 0.25, "demand_threshold": 3, "max_replicas": 3}
    assert server.services.promoter.policy.config.reuse_threshold == 0.25


def test_admin_requires_a_key(server: Server, monkeypatch) -> None:
    monkeypatch.delenv("MEMBRANE_API_KEY", raising=False)
    result = CliRunner().invoke(
        app, ["admin", "inspect", "x", "--host", "127.0.0.1", "--port", str(server.transport.port)]
    )
    assert result.stdout.startswith("401 ")


def test_remote_dashboard_polls_until_interrupted(server: Server, monkeypatch) -> None:
    monkeypatch.setenv("MEMBRANE_API_KEY", ADMIN_KEY)
    polled = []

    def stop(seconds: float) -> None:
        polled.append(seconds)
        raise KeyboardInterrupt

    monkeypatch.setattr(dashboard.time, "sleep", stop)
    dashboard.run_remote_dashboard("127.0.0.1", server.transport.port, refresh=0.5)
    assert polled == [0.5]
    data = dashboard.fetch_json("127.0.0.1", server.transport.port, "/heartbeat")
    assert "HEALTHY" in str(dashboard.header_panel_remote(data).renderable.renderable)


def test_in_process_dashboard_renders_and_stops_the_server(monkeypatch) -> None:
    srv = Server(node=Node("dash"), port=0, load_hooks=False)
    srv.start()
    events = [
        types.SimpleNamespace(timestamp=time.time(), level=level, message=f"{level} event")
        for level in ("info", "warn", "error", "debug", "other")
    ]
    monkeypatch.setattr(srv, "recent_events", lambda n=15: events)
    monkeypatch.setattr(type(srv), "connected_nodes", property(lambda self: ["peer-1"]), raising=False)

    def stop(seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(dashboard.time, "sleep", stop)
    dashboard.run_dashboard(srv)
    assert not srv.running
    assert dashboard.peers_panel(srv).title == "[bold]Peers[/bold]"


def test_dashboard_panels_for_an_idle_server() -> None:
    srv = Server(node=Node("idle"), port=0, load_hooks=False)
    diag = srv.diagnostics()
    assert "idle" in str(dashboard.header_panel_inproc(diag).renderable.renderable)
    assert dashboard.metrics_panel(diag).title == "[bold]Server Metrics[/bold]"
    assert "No peers" in str(dashboard.peers_panel(srv).renderable)
    assert dashboard.metrics_panel_remote({}).title == "[bold]Metrics[/bold]"
    assert "UNHEALTHY" in str(dashboard.header_panel_remote({}).renderable.renderable)
    monkeypatched_events = srv.recent_events
    assert callable(monkeypatched_events)


def test_formatters() -> None:
    assert fmt_bytes(512) == "512.0 B"
    assert fmt_bytes(3 * 1024**2) == "3.0 MiB"
    assert fmt_bytes(2 * 1024**5) == "2.0 PiB"
    assert fmt_duration(42) == "42s"
    assert fmt_duration(600) == "10m"
    assert fmt_duration(5400) == "1.5h"


def answers(monkeypatch, *values: str) -> None:
    replies = iter(values)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))


def test_wizard_accepts_defaults(monkeypatch) -> None:
    answers(monkeypatch, *[""] * 11)
    config = wizard.interactive_setup()
    assert config["node_id"] == "membrane-0" and config["port"] == 8080
    assert config["transport"] == "http" and config["compute"] == "cpu"
    assert config["redis_url"] == "redis://localhost:6379/0" and config["peers"] == []


def test_wizard_reprompts_invalid_choices(monkeypatch) -> None:
    answers(
        monkeypatch,
        "n7", "0.0.0.0", "9000",
        "grpc", "http",  # gRPC is not a transport the server offers
        "tpu", "ollama", "http://ollama:11434", "llama3",
        "n",
        "a:1, b:2",
        "1024", "DEBUG",
    )  # fmt: skip
    config = wizard.interactive_setup()
    assert config["transport"] == "http"
    assert config["compute"] == "ollama" and config["llm_url"] == "http://ollama:11434"
    assert config["redis_url"] == "" and config["peers"] == ["a:1", "b:2"]
    assert config["max_memory"] == 1024 and config["log_level"] == "DEBUG"


@pytest.mark.parametrize(
    ("compute", "replies", "expected"),
    [
        ("openai", ["sk-1", "gpt-x"], {"api_key": "sk-1", "llm_model": "gpt-x"}),
        ("anthropic", ["ak-1", ""], {"api_key": "ak-1", "llm_model": "claude-sonnet-5-5"}),
        ("transformers", ["tiny"], {"llm_model": "tiny"}),
    ],
)
def test_wizard_asks_for_backend_settings(monkeypatch, compute: str, replies: list[str], expected: dict) -> None:
    answers(monkeypatch, "", "", "", "", compute, *replies, "yes", "redis://r:6379/1", "", "", "")
    config = wizard.interactive_setup()
    assert {k: config[k] for k in expected} == expected
    assert config["redis_url"] == "redis://r:6379/1"
