"""The admin audit log survives restarts, keeps its chain, and detects tampering."""

import json
import os
from pathlib import Path

from typer.testing import CliRunner

from membrane.audit import verify_chain
from membrane.cli import app
from membrane.node import Node
from membrane.server import Server


def test_chain_continues_across_restarts(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    first = Server(node=Node("a0"), port=0, audit_path=str(path), load_hooks=False)
    first.audit_log.record("ops", "policy.update", {"ttl": 60})
    first.audit_log.record("ops", "fragment.evict", {"hash": "h1"})
    assert (os.stat(path).st_mode & 0o777) == 0o600

    second = Server(node=Node("a0"), port=0, audit_path=str(path), load_hooks=False)
    assert second.audit_log.broken_at is None
    second.audit_log.record("ops", "fragment.evict", {"hash": "h2"})
    entries = second.audit_log.storage.all()
    assert [e.index for e in entries] == [0, 1, 2]
    assert verify_chain(entries) is None
    assert second.metrics_peers.audit_chain_valid.value == 1.0
    assert entries[0].timestamp > 1_700_000_000  # wall-clock time


def test_tampering_is_detected_on_start(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    server = Server(node=Node("a1"), port=0, audit_path=str(path), load_hooks=False)
    server.audit_log.record("ops", "policy.update", {"ttl": 60})
    server.audit_log.record("ops", "policy.update", {"ttl": 120})
    lines = path.read_text().splitlines()
    record = json.loads(lines[0])
    record["payload"]["ttl"] = 999_999
    lines[0] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n")

    reopened = Server(node=Node("a1"), port=0, audit_path=str(path), load_hooks=False)
    assert reopened.audit_log.broken_at == 0
    assert reopened.metrics_peers.audit_chain_valid.value == 0.0


def test_openapi_command(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["openapi"])
    assert result.exit_code == 0
    spec = json.loads(result.stdout)
    assert spec["openapi"].startswith("3.") and "/store" in spec["paths"] and "/blobs/{payload_ref}" in spec["paths"]
    out = tmp_path / "spec.json"
    assert CliRunner().invoke(app, ["openapi", "-o", str(out)]).exit_code == 0
    assert json.loads(out.read_text())["paths"] == spec["paths"]
