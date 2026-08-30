"""Tests for the LMCache-backed ContentStore (Phase 0.2)."""

from __future__ import annotations

import pytest

lmcache = pytest.importorskip("lmcache")


def test_import_path():
    from membrane.storage.lmcache import LMCacheContentStore

    assert LMCacheContentStore is not None


def test_requires_lmcache_to_be_present():
    """The import is lazy so a Membrane install without the
    [lmcache] extras must keep working."""
    pytest.importorskip("lmcache")  # already verified above


def test_round_trip_bytes():
    from membrane.storage.lmcache import LMCacheContentStore

    store = LMCacheContentStore()
    store.put("alpha", b"hello world")
    assert store.get("alpha") == b"hello world"
    assert store.has("alpha") is True
    assert store.size() == len(b"hello world")


def test_get_missing_key_returns_none():
    from membrane.storage.lmcache import LMCacheContentStore

    store = LMCacheContentStore()
    assert store.get("never") is None
    assert store.has("never") is False


def test_delete_returns_true_on_present_key():
    from membrane.storage.lmcache import LMCacheContentStore

    store = LMCacheContentStore()
    store.put("k", b"v")
    assert store.delete("k") is True
    assert store.has("k") is False


def test_delete_missing_returns_false():
    from membrane.storage.lmcache import LMCacheContentStore

    store = LMCacheContentStore()
    assert store.delete("never") is False


def test_size_increments_per_put():
    from membrane.storage.lmcache import LMCacheContentStore

    store = LMCacheContentStore()
    store.put("a", b"x" * 10)
    store.put("b", b"y" * 5)
    assert store.size() == 15


def test_content_store_protocol_conformance():
    from membrane.content_store import ContentStore
    from membrane.storage.lmcache import LMCacheContentStore

    assert isinstance(LMCacheContentStore(), ContentStore)
