"""``membrane serve`` builds, starts, and stops a real server along each of its paths."""

import types

import pytest
from typer.testing import CliRunner

from membrane.cli import app
from membrane.cli.commands import serve


@pytest.fixture
def stopped(monkeypatch) -> list[str]:
    """Stop the server as soon as serve hands it to the daemon loop or the dashboard."""
    calls: list[str] = []

    def daemon(server, drain_timeout):
        calls.append(f"daemon:{server.node.node_id}:{drain_timeout:g}")
        server.stop()

    def dashboard(server):
        calls.append(f"dashboard:{server.node.node_id}")
        server.stop()

    monkeypatch.setattr(serve, "run_until_signalled", daemon)
    monkeypatch.setattr(serve, "run_dashboard", dashboard)
    return calls


def test_daemon_mode_serves_until_signalled(stopped: list[str]) -> None:
    result = CliRunner().invoke(
        app, ["serve", "--daemon", "--node-id", "d1", "--port", "0", "--allow-unauthenticated", "--drain-timeout", "2"]
    )
    assert result.exit_code == 0, result.output
    assert stopped == ["daemon:d1:2"]


def test_a_terminal_gets_the_dashboard(stopped: list[str], monkeypatch) -> None:
    tty = types.SimpleNamespace(isatty=lambda: True)
    monkeypatch.setattr(serve, "sys", types.SimpleNamespace(stdin=tty, stdout=tty))
    result = CliRunner().invoke(app, ["serve", "--node-id", "t1", "--port", "0", "--allow-unauthenticated"])
    assert result.exit_code == 0, result.output
    assert stopped == ["dashboard:t1"]


def test_interactive_setup_supplies_the_settings(stopped: list[str], monkeypatch) -> None:
    config = {
        "node_id": "wizard-node",
        "host": "127.0.0.1",
        "port": 0,
        "transport": "http",
        "compute": "cpu",
        "llm_url": "",
        "llm_model": "",
        "api_key": "",
        "redis_url": "",
        "peers": [],
        "max_memory": 1 << 20,
        "log_level": "WARNING",
    }
    monkeypatch.setattr(serve, "interactive_setup", lambda: config)
    result = CliRunner().invoke(app, ["serve", "--interactive", "--daemon", "--allow-unauthenticated"])
    assert result.exit_code == 0, result.output
    assert len(stopped) == 1 and stopped[0].startswith("daemon:wizard-node:")


def test_invalid_settings_exit_with_the_reason(stopped: list[str]) -> None:
    result = CliRunner().invoke(app, ["serve", "--daemon", "--port", "0", "--eviction", "nope"])
    assert result.exit_code == 2  # configuration error
    assert "unknown eviction policy" in result.output
    assert stopped == []


def test_unknown_log_format_is_refused(stopped: list[str]) -> None:
    result = CliRunner().invoke(app, ["serve", "--daemon", "--port", "0", "--log-format", "xml"])
    assert result.exit_code == 2
    assert "--log-format" in result.output
    assert stopped == []


def test_split_list_flattens_repeated_and_comma_separated_values() -> None:
    assert serve.split_list(["a, b", "c", " "]) == ["a", "b", "c"]
    assert serve.split_list(None) == []
