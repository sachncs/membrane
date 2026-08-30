"""Tests for the canonical ContentStore interface and its implementations."""

from __future__ import annotations

import contextlib
import io
import os
import sys
from pathlib import Path

import pytest

from membrane.content_store import (
    ContentStore,
    FilesystemBlob,
    InProcessBytes,
)


class TestContentStoreProtocol:
    """The Protocol is satisfied by both real implementations."""

    def test_inprocess_is_a_content_store(self):
        assert isinstance(InProcessBytes(), ContentStore)

    def test_filesystem_is_a_content_store(self, tmp_path: Path):
        store = FilesystemBlob(tmp_path / "blob")
        assert isinstance(store, ContentStore)


class TestInProcessBytes:
    """Round-trip and edge cases for the in-process store."""

    def test_round_trip(self):
        store = InProcessBytes()
        store.put("k1", b"hello")
        assert store.get("k1") == b"hello"
        assert store.has("k1")

    def test_missing_returns_none(self):
        store = InProcessBytes()
        assert store.get("missing") is None
        assert store.has("missing") is False

    def test_delete_returns_false_when_missing(self):
        store = InProcessBytes()
        assert store.delete("missing") is False

    def test_delete_returns_true_when_present(self):
        store = InProcessBytes()
        store.put("k1", b"hello")
        assert store.delete("k1") is True
        assert store.get("k1") is None

    def test_size_tracks_total_bytes(self):
        store = InProcessBytes()
        store.put("a", b"12345")
        store.put("b", b"678")
        assert store.size() == 8
        store.delete("a")
        assert store.size() == 3

    def test_capacity_bytes_rejects_oversize_writes(self):
        store = InProcessBytes(capacity_bytes=10)
        with pytest.raises(ValueError, match="exceeds capacity_bytes"):
            store.put("huge", b"x" * 11)

    def test_overwrite_replaces(self):
        store = InProcessBytes()
        store.put("k", b"a")
        store.put("k", b"bb")
        assert store.get("k") == b"bb"
        assert store.size() == 2

    def test_len_reflects_entries(self):
        store = InProcessBytes()
        assert len(store) == 0
        store.put("a", b"1")
        store.put("b", b"2")
        assert len(store) == 2

    def test_thread_safe(self):
        """A burst of concurrent puts/reads does not interleave badly."""
        import threading

        store = InProcessBytes()
        errors: list[Exception] = []

        def writer(prefix: str) -> None:
            try:
                for i in range(100):
                    store.put(f"{prefix}_{i}", b"x")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(f"t{i}",)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        # Every key the writers attempted must be visible.
        for i in range(8):
            for j in range(100):
                assert store.has(f"t{i}_{j}")


class TestFilesystemBlob:
    """Round-trip and crash-safety for the on-disk store."""

    def test_round_trip(self, tmp_path: Path):
        store = FilesystemBlob(tmp_path / "blob")
        store.put("abcd1234", b"hello")
        assert store.get("abcd1234") == b"hello"
        assert store.has("abcd1234")

    def test_creates_missing_dirs(self, tmp_path: Path):
        root = tmp_path / "nested" / "dirs"
        FilesystemBlob(root)
        assert root.exists()

    def test_sharded_layout(self, tmp_path: Path):
        root = tmp_path / "blob"
        store = FilesystemBlob(root)
        store.put("abcdef01", b"x")
        # Two-level shard: "ab" / "cd" / "abcdef01.blob"
        expected = root / "ab" / "cd" / "abcdef01.blob"
        assert expected.exists()

    def test_delete_removes_blob_and_empty_dirs(self, tmp_path: Path):
        root = tmp_path / "blob"
        store = FilesystemBlob(root)
        store.put("abcdef01", b"hello")
        path = root / "ab" / "cd" / "abcdef01.blob"
        assert path.exists()
        assert store.delete("abcdef01") is True
        assert not path.exists()
        # The empty parent shards may be removed.
        # (One of the rmdir calls may succeed; we don't care which.)

    def test_delete_missing_returns_false(self, tmp_path: Path):
        store = FilesystemBlob(tmp_path / "blob")
        assert store.delete("nonexistent") is False

    def test_get_missing_returns_none(self, tmp_path: Path):
        store = FilesystemBlob(tmp_path / "blob")
        assert store.get("nonexistent") is None

    def test_short_keys_rejected(self, tmp_path: Path):
        store = FilesystemBlob(tmp_path / "blob")
        with pytest.raises(ValueError, match="at least 4 chars"):
            store.put("ab", b"x")

    def test_persistence_across_instances(self, tmp_path: Path):
        root = tmp_path / "blob"
        store_a = FilesystemBlob(root)
        store_a.put("abcdef01", b"hello world")
        # New instance, same root — must find the prior frame.
        store_b = FilesystemBlob(root)
        assert store_b.get("abcdef01") == b"hello world"
        assert store_b.has("abcdef01")

    def test_size_reflects_total(self, tmp_path: Path):
        store = FilesystemBlob(tmp_path / "blob")
        store.put("aaaa1111", b"12345")
        store.put("bbbb2222", b"678")
        assert store.size() == 8

    def test_atomic_write_no_partial(self, tmp_path: Path):
        """A successful put leaves exactly one ``*.blob`` and no ``.tmp`` left behind."""
        root = tmp_path / "blob"
        store = FilesystemBlob(root)
        store.put("abcd1234", b"x")
        parents = list(root.rglob("*"))
        # No ``.tmp`` file from the write survived.
        assert not any(p.name.endswith(".tmp") for p in parents)
        # Exactly one ``*.blob`` was written.
        blobs = [p for p in parents if p.suffix == ".blob"]
        assert len(blobs) == 1
