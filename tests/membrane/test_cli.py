"""Tests for Membrane CLI."""

import subprocess
import sys


class TestCLI:
    """Test suite for CLI commands."""

    def test_config_command(self):
        result = subprocess.run(
            [sys.executable, "-m", "membrane.cli", "config"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Membrane Configuration" in result.stdout

    def test_serve_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "membrane.cli", "serve", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--peer" in result.stdout
        assert "--heartbeat-interval" in result.stdout
        assert "--replica-count" in result.stdout

    def test_cluster_status_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "membrane.cli", "cluster-status", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--host" in result.stdout
        assert "--port" in result.stdout

    def test_llm_status_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "membrane.cli", "llm-status", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--host" in result.stdout
        assert "--port" in result.stdout

    def test_serve_help_includes_llm_flags(self):
        result = subprocess.run(
            [sys.executable, "-m", "membrane.cli", "serve", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--compute" in result.stdout
        assert "--llm-url" in result.stdout
        assert "--llm-model" in result.stdout
        assert "--api-key" in result.stdout
