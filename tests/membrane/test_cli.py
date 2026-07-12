"""Tests for Membrane CLI."""

from typer.testing import CliRunner

from membrane.cli import app

# Use Typer's CliRunner because it does not depend on the calling
# terminal's COLUMNS / width environment variables. The previous
# implementation used subprocess + --help and was brittle on CI
# runners where Rich detects an extremely wide terminal and
# wraps long option names past the visible region.
runner = CliRunner()


class TestCLI:
    """Test suite for CLI commands."""

    def test_config_command(self):
        result = runner.invoke(app, ["config"])
        assert result.exit_code == 0
        # Rich may wrap or colorize the output, so check for the
        # canonical substring rather than the exact header.
        assert "Membrane Configuration" in result.stdout

    def test_serve_help(self):
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--peer" in result.stdout
        assert "--heartbeat-interval" in result.stdout
        assert "--replica-count" in result.stdout

    def test_cluster_status_help(self):
        result = runner.invoke(app, ["cluster-status", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.stdout
        assert "--port" in result.stdout

    def test_llm_status_help(self):
        result = runner.invoke(app, ["llm-status", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.stdout
        assert "--port" in result.stdout

    def test_serve_help_includes_llm_flags(self):
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--compute" in result.stdout
        assert "--llm-url" in result.stdout
        assert "--llm-model" in result.stdout
        assert "--api-key" in result.stdout
