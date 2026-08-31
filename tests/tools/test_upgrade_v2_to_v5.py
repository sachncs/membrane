"""Tests for the v2 / v4 / v5 magic migration tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.upgrade_v2_to_v5 import (
    detect_magic,
    migrate,
    migrate_v2_to_v5,
    migrate_v4_to_v5,
)


class TestDetectMagic:
    def test_detect_v5(self):
        assert detect_magic(b"\xc0\xde\x01\x05hello") == 5

    def test_detect_v4(self):
        assert detect_magic(b"\xc0\xde\x01\x04hello") == 4

    def test_detect_v2(self):
        assert detect_magic(b"\xc0\xde\x01\x02hello") == 2

    def test_detect_unknown(self):
        assert detect_magic(b"unknown" * 2) == 0

    def test_detect_short(self):
        assert detect_magic(b"") == 0


class TestRewrite:
    def test_v4_to_v5(self):
        original = b"\xc0\xde\x01\x04" + b"body"
        rewritten = migrate_v4_to_v5(original)
        assert rewritten.startswith(b"\xc0\xde\x01\x05")
        assert rewritten[4:] == original[4:]

    def test_v2_to_v5(self):
        original = b"\xc0\xde\x01\x02" + b"body"
        rewritten = migrate_v2_to_v5(
            original, v2_tenant="acme", v3_tenant="globex"
        )
        assert rewritten.startswith(b"\xc0\xde\x01\x05")
        assert rewritten[4:] == original[4:]


class TestMigrateCli:
    def test_dry_run_does_not_modify(self, tmp_path: Path):
        # Seed a v4 file.
        (tmp_path / "aa").mkdir()
        blob = tmp_path / "aa" / "ab" / "abcd.blob"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(b"\xc0\xde\x01\x04" + b"v4-content")

        counts = migrate(tmp_path, dry_run=True)
        assert counts["v4"] == 1
        assert counts["v5_noop"] == 0
        # The on-disk bytes are still the v4 magic.
        assert blob.read_bytes().startswith(b"\xc0\xde\x01\x04")

    def test_write_rewrites_v4(self, tmp_path: Path):
        blob = tmp_path / "aa" / "ab" / "abcd.blob"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(b"\xc0\xde\x01\x04" + b"v4-content")

        counts = migrate(tmp_path, dry_run=False)
        assert counts["v4"] == 1
        assert blob.read_bytes().startswith(b"\xc0\xde\x01\x05")

    def test_v5_noop_count(self, tmp_path: Path):
        blob = tmp_path / "aa" / "ab" / "abcd.blob"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(b"\xc0\xde\x01\x05" + b"v5-content")

        counts = migrate(tmp_path, dry_run=False)
        assert counts["v5_noop"] == 1
        # The v5 file is byte-identical.
        assert blob.read_bytes() == b"\xc0\xde\x01\x05" + b"v5-content"
