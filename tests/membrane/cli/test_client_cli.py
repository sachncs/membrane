"""Tests for the `membrane client` CLI subcommand (Phase 3.6.1 follow-up)."""

from __future__ import annotations

import io
import json
import sys
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from membrane.cli.commands.client import client_app


def _combined(result) -> str:
    """Return stdout + stderr for the test to assert on."""
    return (result.stdout or "") + (getattr(result, "stderr", None) or "")


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestClientStore:
    def test_store_emits_json_on_2xx(self, runner: CliRunner) -> None:
        # Patch the MembraneClient inside the cli module.
        with patch("membrane.cli.commands.client.MembraneClient") as cls:
            instance = MagicMock()
            instance.store.return_value = {"success": True, "content_hash": "h" * 64}
            cls.return_value = instance
            result = runner.invoke(
                client_app,
                [
                    "store",
                    "--body",
                    json.dumps(
                        {
                            "schema_version": 5,
                            "tenant_id": "acme",
                            "identity": {},
                            "payload_size": 0,
                            "ttl": 0,
                            "reuse_score": 0,
                            "version_id": 1,
                        }
                    ),
                    "--base-url",
                    "http://n1",
                ],
            )
        assert result.exit_code == 0
        instance.store.assert_called_once()
        # The result was JSON-printed.
        body = json.loads(result.stdout)
        assert body["success"] is True
        assert body["content_hash"] == "h" * 64

    def test_store_invalid_json_exits_1(self, runner: CliRunner) -> None:
        result = runner.invoke(
            client_app,
            ["store", "--body", "not-json", "--base-url", "http://n1"],
        )
        assert result.exit_code == 1
        assert "invalid JSON" in _combined(result)

    def test_store_client_error_exits_1(self, runner: CliRunner) -> None:
        from membrane.client import MembraneClientError

        with patch("membrane.cli.commands.client.MembraneClient") as cls:
            instance = MagicMock()
            instance.store.side_effect = MembraneClientError("draining")
            cls.return_value = instance
            result = runner.invoke(
                client_app,
                [
                    "store",
                    "--body",
                    json.dumps({"schema_version": 5}),
                    "--base-url",
                    "http://n1",
                ],
            )
        assert result.exit_code == 1
        assert "draining" in _combined(result)


class TestClientRetrieve:
    def test_retrieve_emits_dict(self, runner: CliRunner) -> None:
        with patch("membrane.cli.commands.client.MembraneClient") as cls:
            instance = MagicMock()
            instance.retrieve.return_value = {"found": True, "fragment": None}
            cls.return_value = instance
            result = runner.invoke(
                client_app,
                ["retrieve", "--hash", "h" * 64, "--base-url", "http://n1"],
            )
        assert result.exit_code == 0
        body = json.loads(result.stdout)
        assert body["found"] is True


class TestClientInventory:
    def test_inventory_emits_dict(self, runner: CliRunner) -> None:
        with patch("membrane.cli.commands.client.MembraneClient") as cls:
            instance = MagicMock()
            instance.inventory.return_value = {"digest": {"a": 1}}
            cls.return_value = instance
            result = runner.invoke(
                client_app, ["inventory", "--base-url", "http://n1"]
            )
        assert result.exit_code == 0
        body = json.loads(result.stdout)
        assert body == {"digest": {"a": 1}}


class TestClientPrefill:
    def test_prefill_emits_dict(self, runner: CliRunner) -> None:
        with patch("membrane.cli.commands.client.MembraneClient") as cls:
            instance = MagicMock()
            instance.prefill.return_value = {"fragments": []}
            cls.return_value = instance
            result = runner.invoke(
                client_app,
                [
                    "prefill",
                    "--prompt-tokens",
                    "1 2 3",
                    "--model-id",
                    "m",
                    "--base-url",
                    "http://n1",
                ],
            )
        assert result.exit_code == 0
        body = json.loads(result.stdout)
        assert body == {"fragments": []}

    def test_prefill_token_parse(self, runner: CliRunner) -> None:
        with patch("membrane.cli.commands.client.MembraneClient") as cls:
            instance = MagicMock()
            instance.prefill.return_value = {"fragments": []}
            cls.return_value = instance
            runner.invoke(
                client_app,
                [
                    "prefill",
                    "--prompt-tokens",
                    "1 2 3 4 5",
                    "--base-url",
                    "http://n1",
                ],
            )
        # Typer parses "--prompt-tokens 1 2 3 4 5" as a single
        # string by default; the CLI splits on whitespace.
        instance.prefill.assert_called_once()
        args, _kwargs = instance.prefill.call_args
        assert args[0] == [1, 2, 3, 4, 5]
