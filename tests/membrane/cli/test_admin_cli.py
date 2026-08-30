"""Tests for the admin CLI subcommands (Phase 3.2.7)."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from membrane.cli.commands.admin import admin_app


class TestAdminCli:
    def setup_method(self):
        self.runner = CliRunner()

    def test_admin_help(self):
        result = self.runner.invoke(admin_app, ["--help"])
        assert result.exit_code == 0
        assert "Admin operations" in result.stdout

    def test_inspect_help(self):
        result = self.runner.invoke(admin_app, ["inspect", "--help"])
        assert result.exit_code == 0

    def test_evict_help(self):
        result = self.runner.invoke(admin_app, ["evict", "--help"])
        assert result.exit_code == 0

    def test_repair_help(self):
        result = self.runner.invoke(admin_app, ["repair", "--help"])
        assert result.exit_code == 0

    def test_policy_help(self):
        result = self.runner.invoke(admin_app, ["policy", "--help"])
        assert result.exit_code == 0
