"""Tests for the JSON wire dict migration tool (Phase 3.0+ follow-up)."""

from __future__ import annotations

import json

import pytest

from tools.upgrade_v2_to_v5_json import migrate_dict, migrate_file


class TestMigrateDict:
    def test_bumps_schema_version(self):
        d = {"schema_version": 4, "identity": {}, "tenant_id": "acme"}
        out = migrate_dict(d, default_tenant="acme")
        assert out["schema_version"] == 5
        assert out["tenant_id"] == "acme"

    def test_inserts_default_tenant_when_missing(self):
        d = {"schema_version": 4, "identity": {}}
        out = migrate_dict(d, default_tenant="globex")
        assert out["tenant_id"] == "globex"

    def test_does_not_mutate_input(self):
        d = {"schema_version": 4, "identity": {}, "tenant_id": "acme"}
        snapshot = dict(d)
        migrate_dict(d, default_tenant="globex")
        assert d == snapshot

    def test_preserves_other_fields(self):
        d = {
            "schema_version": 4,
            "identity": {"payload_hash": "h" * 64},
            "payload_size": 10,
            "tenant_id": "acme",
        }
        out = migrate_dict(d, default_tenant="acme")
        assert out["identity"] == {"payload_hash": "h" * 64}
        assert out["payload_size"] == 10


class TestMigrateFile:
    def test_dry_run_does_not_modify(self, tmp_path):
        path = tmp_path / "wire.jsonl"
        path.write_text(
            '{"schema_version": 4, "tenant_id": "a"}\n'
            '{"schema_version": 4, "tenant_id": "b"}\n'
        )
        rewritten, _skipped = migrate_file(
            path, default_tenant="public", write=False
        )
        assert rewritten == 2
        # File unchanged.
        assert path.read_text().count('"schema_version": 4') == 2

    def test_write_rewrites_in_place(self, tmp_path):
        path = tmp_path / "wire.jsonl"
        path.write_text('{"schema_version": 4, "tenant_id": "a"}\n')
        rewritten, _skipped = migrate_file(
            path, default_tenant="public", write=True
        )
        assert rewritten == 1
        content = path.read_text()
        assert '"schema_version": 5' in content
        assert '"tenant_id"' in content

    def test_skips_already_v5(self, tmp_path):
        path = tmp_path / "wire.jsonl"
        path.write_text('{"schema_version": 5, "tenant_id": "a"}\n')
        rewritten, _skipped = migrate_file(
            path, default_tenant="public", write=False
        )
        assert rewritten == 0
        assert _skipped == 1

    def test_skips_malformed_lines(self, tmp_path):
        path = tmp_path / "wire.jsonl"
        path.write_text(
            'not-json\n{"schema_version": 4}\n'
        )
        rewritten, _skipped = migrate_file(
            path, default_tenant="public", write=True
        )
        # The malformed line is preserved unchanged; the v4 line
        # is migrated.
        assert rewritten == 1
        # The v4 line is now v5.
        parsed = json.loads(path.read_text().strip().splitlines()[1])
        assert parsed["schema_version"] == 5

    def test_missing_path_is_noop(self, tmp_path):
        path = tmp_path / "nope.jsonl"
        rewritten, _skipped = migrate_file(
            path, default_tenant="public", write=False
        )
        assert rewritten == 0

    def test_empty_lines_preserved(self, tmp_path):
        path = tmp_path / "wire.jsonl"
        path.write_text(
            '{"schema_version": 4}\n\n{"schema_version": 4}\n'
        )
        rewritten, _skipped = migrate_file(
            path, default_tenant="public", write=True
        )
        assert rewritten == 2
