"""Tests for the LMCache distributed connector factory (Phase 0.3)."""

from __future__ import annotations

import pytest

from membrane.storage.lmcache_connectors import (
    LMCacheConnectorError,
    build_distributed_store,
)


def test_remote_without_url_raises():
    with pytest.raises(LMCacheConnectorError, match="requires a remote_url"):
        build_distributed_store(kind="remote")


def test_remote_v1_raises_not_implemented():
    with pytest.raises(LMCacheConnectorError, match=r"Phase 5\+"):
        build_distributed_store(kind="remote", remote_url="localhost:50051")


def test_gds_v1_raises_not_implemented():
    with pytest.raises(LMCacheConnectorError, match=r"Phase 5\+"):
        build_distributed_store(kind="gds", remote_url="weka://example")
