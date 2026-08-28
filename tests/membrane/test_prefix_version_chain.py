"""Tests for prefix_version_chain module."""

import pytest

from membrane.versions import Versions, VersionEntry


class TestPrefixVersionChain:
    """Test suite for Versions."""

    def test_append_version(self):
        chain = Versions()
        chain.append_version("hash-a", version_id=1)
        entry = chain.get_version(1)
        assert entry == VersionEntry(prefix_hash="hash-a", version_id=1, parent_version=None)

    def test_get_version_missing(self):
        chain = Versions()
        assert chain.get_version(99) is None

    def test_common_ancestor_linear(self):
        chain = Versions()
        chain.append_version("h1", 1)
        chain.append_version("h2", 2, parent_version=1)
        chain.append_version("h3", 3, parent_version=2)
        assert chain.get_common_ancestor(1, 3) == 1

    def test_common_ancestor_branched(self):
        chain = Versions()
        chain.append_version("h1", 1)
        chain.append_version("h2", 2, parent_version=1)
        chain.append_version("h3", 3, parent_version=1)
        assert chain.get_common_ancestor(2, 3) == 1

    def test_common_ancestor_same_version(self):
        chain = Versions()
        chain.append_version("h1", 1)
        assert chain.get_common_ancestor(1, 1) == 1

    def test_common_ancestor_no_common(self):
        chain = Versions()
        chain.append_version("h1", 1)
        chain.append_version("h2", 2)
        assert chain.get_common_ancestor(1, 2) is None

    def test_latest_version(self):
        chain = Versions()
        chain.append_version("h1", 1)
        chain.append_version("h2", 2, parent_version=1)
        assert chain.latest_version("h2") == 2

    def test_latest_version_unknown(self):
        chain = Versions()
        assert chain.latest_version("unknown") is None
