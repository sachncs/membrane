"""Tests for Membrane CLI."""

import re

from typer.testing import CliRunner

from membrane.cli import app

# Match CSI / control-introduced escape sequences that Rich emits
# when it syntax-highlights help output. Rich's OptionHighlighter
# re-styles long flags at render time and splits them across SGR
# boundaries (e.g. ``--host`` becomes ``\x1b[36m-\x1b[0m\x1b[36m-host\x1b[0m``),
# so the literal substring never appears in ``result.stdout`` even
# though the user sees it on the terminal. Strip ANSI before
# substring matching so tests assert on what a human would read.
ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def strip_ansi(text: str) -> str:
    """Return ``text`` with ANSI / Rich styling escape sequences removed."""
    return ANSI_ESCAPE_RE.sub("", text)


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
        assert "Membrane Configuration" in strip_ansi(result.stdout)

    def test_serve_help(self):
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        rendered = strip_ansi(result.stdout)
        assert "--peer" in rendered
        assert "--heartbeat-interval" in rendered
        assert "--replica-count" in rendered

    def test_cluster_status_help(self):
        result = runner.invoke(app, ["cluster-status", "--help"])
        assert result.exit_code == 0
        rendered = strip_ansi(result.stdout)
        assert "--host" in rendered
        assert "--port" in rendered

    def test_llm_status_help(self):
        result = runner.invoke(app, ["llm-status", "--help"])
        assert result.exit_code == 0
        rendered = strip_ansi(result.stdout)
        assert "--host" in rendered
        assert "--port" in rendered

    def test_serve_help_includes_llm_flags(self):
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        rendered = strip_ansi(result.stdout)
        assert "--compute" in rendered
        assert "--llm-url" in rendered
        assert "--llm-model" in rendered
        assert "--api-key" in rendered
