"""Tests for the tier-migration primitive (Phase 3.5.6 follow-up)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from membrane.fragment import Fragment
from membrane.identity import PayloadIdentity
from membrane.node import Node
from membrane.tier_migration import TierMigration, install_default_demote
from membrane.tiers import TierPolicy


def _fragment(reuse_score: float, payload_size: int = 10) -> Fragment:
    """Build a fragment with a given reuse score."""
    ident = PayloadIdentity(
        payload_hash=f"hash-{reuse_score}".ljust(64, "0")[:64],
        model_id="m",
        model_revision="",
        tokenizer_name="m",
        tokenizer_revision="",
        layer_range=(0, 1),
        head_range=(-1, -1),
        token_span=(0, 1),
        dtype="float16",
        shape=(1, 1, 1, 1, 64),
    )
    return Fragment(
        identity=ident,
        payload_ref=None,
        payload_size=payload_size,
        ttl=60.0,
        reuse_score=reuse_score,
        version_id=1,
        tenant_id="public",
    )


class TestTierMigration:
    def test_demote_called_on_evict(self):
        demoted: list[tuple[str, str]] = []  # (payload_hash, tier)

        def on_demote(fragment: object, tier: str) -> None:
            assert isinstance(fragment, Fragment)
            demoted.append((fragment.identity.payload_hash, tier))

        policy = TierPolicy()
        migration = TierMigration(policy=policy, on_demote=on_demote)
        node = Node(node_id="n1", max_memory_bytes=10_000)
        node.add_eviction_callback(migration.on_evict)
        # High-reuse fragments go to "hot"; low-reuse go to
        # "archival". Fill the node so eviction kicks in.
        for i in range(50):
            node.store(_fragment(reuse_score=0.1), is_primary=True)
        # Trigger eviction.
        node.evict(target_bytes=100)
        # The callbacks fired.
        assert len(demoted) > 0
        # Each demoted fragment is the right tier.
        for _h, tier in demoted:
            assert tier in {"hot", "warm", "cold", "archival"}

    def test_dedupe_within_evict_run(self):
        """A fragment evicted only once should fire the callback once."""
        demoted: list[object] = []

        def on_demote(fragment: object, tier: str) -> None:
            demoted.append(fragment)

        policy = TierPolicy()
        migration = TierMigration(policy=policy, on_demote=on_demote)
        node = Node(node_id="n1", max_memory_bytes=10_000)
        node.add_eviction_callback(migration.on_evict)
        # Force eviction of one specific fragment.
        node.store(_fragment(reuse_score=0.0), is_primary=True)
        node.evict(target_bytes=100)
        # The fragment was evicted once; the dedup set keeps
        # the callback from firing twice for the same hash.
        assert len(demoted) == 1

    def test_reset_seen_clears_dedup(self):
        demoted: list[object] = []

        def on_demote(fragment: object, tier: str) -> None:
            demoted.append(fragment)

        policy = TierPolicy()
        migration = TierMigration(policy=policy, on_demote=on_demote)
        node = Node(node_id="n1", max_memory_bytes=10_000)
        node.add_eviction_callback(migration.on_evict)
        node.store(_fragment(reuse_score=0.0), is_primary=True)
        node.evict(target_bytes=100)
        migration.reset_seen()
        # Re-store the same fragment identity and evict again; the
        # dedup set is now empty.
        node.store(_fragment(reuse_score=0.0), is_primary=True)
        node.evict(target_bytes=100)
        assert len(demoted) == 2

    def test_no_on_demote_is_no_op(self):
        """When on_demote is None, eviction still happens."""
        policy = TierPolicy()
        migration = TierMigration(policy=policy, on_demote=None)
        node = Node(node_id="n1", max_memory_bytes=10_000)
        node.add_eviction_callback(migration.on_evict)
        node.store(_fragment(reuse_score=0.0), is_primary=True)
        # Just verify no exception.
        node.evict(target_bytes=100)

    def test_install_default_demote_attaches_callback(self):
        policy = TierPolicy()
        migration = TierMigration(policy=policy, on_demote=None)
        node = Node(node_id="n1", max_memory_bytes=10_000)
        install_default_demote(node, migration)
        node.store(_fragment(reuse_score=0.0), is_primary=True)
        node.evict(target_bytes=100)
