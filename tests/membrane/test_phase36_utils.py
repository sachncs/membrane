"""Tests for OpenAPI spec, demo entry-point, and ClusterConfig validation (Phase 3.6.2 + 3.6.4 + 3.6.5)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from membrane.network.config import ClusterConfig, validate_config


class TestValidateConfig:
    def test_accepts_existing_config(self):
        cfg = ClusterConfig()
        assert validate_config(cfg) is cfg

    def test_validates_minimal_dict(self):
        cfg = validate_config({"node_id": "n1"})
        assert cfg.node_id == "n1"
        assert cfg.port == 8080

    def test_rejects_invalid_port(self):
        with pytest.raises(ValueError, match="ClusterConfig validation failed"):
            validate_config({"node_id": "n1", "port": 0})

    def test_rejects_negative_interval(self):
        with pytest.raises(ValueError, match="ClusterConfig validation failed"):
            validate_config({"node_id": "n1", "heartbeat_interval_sec": -1.0})

    def test_rejects_multiple_errors(self):
        with pytest.raises(ValueError) as exc_info:
            validate_config({"node_id": "n1", "port": 99999, "quorum_count": 0})
        # Both fields should be mentioned.
        msg = str(exc_info.value)
        assert "port" in msg
        assert "quorum_count" in msg


class TestOpenAPISpec:
    def test_generate_spec_returns_openapi_dict(self):
        from fastapi import FastAPI

        from membrane.openapi import generate_spec

        app = FastAPI(title="Test")

        @app.get("/ping")
        def ping():  # type: ignore[no-redef]
            return {"pong": True}

        spec = generate_spec(app)
        assert "openapi" in spec
        assert "paths" in spec
        assert "/ping" in spec["paths"]

    def test_write_spec_round_trip(self, tmp_path: Path):
        from fastapi import FastAPI

        from membrane.openapi import write_spec

        app = FastAPI(title="Test")
        path = tmp_path / "openapi.json"
        write_spec(app, str(path))
        loaded = json.loads(path.read_text())
        assert loaded["info"]["title"] == "Test"


class TestDemoEntryPoint:
    def test_runs(self, capsys):
        result = subprocess.run(
            [sys.executable, "-m", "membrane.demo"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        # The demo writes to stdout via print; the exit code is 0.
        assert result.returncode == 0
        assert "demo:" in result.stdout
