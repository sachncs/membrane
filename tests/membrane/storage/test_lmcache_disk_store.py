"""Tests for the LMCacheDiskStore reimplementation (Phase 0.4)."""

from __future__ import annotations

from pathlib import Path

from membrane.content_store import FilesystemBlob, LMCacheDiskStore


def test_lmcache_disk_store_is_a_filesystem_blob_subclass():
    """The v1 surface stays consistent with the v1.0.x FilesystemBlob."""
    assert issubclass(LMCacheDiskStore, FilesystemBlob)


def test_round_trip(tmp_path: Path):
    store = LMCacheDiskStore(tmp_path / "lmcache", tenant_id="acme")
    store.put("alpha", b"hello lmcache")
    assert store.get("alpha") == b"hello lmcache"
    assert store.has("alpha") is True
    assert store.size() == len(b"hello lmcache")


def test_atomic_writes(tmp_path: Path):
    store = LMCacheDiskStore(tmp_path / "lmcache", tenant_id="acme")
    store.put("k1aa", b"v1")
    store.put("k1aa", b"v2-longer")
    assert store.get("k1aa") == b"v2-longer"
