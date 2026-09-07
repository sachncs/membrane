"""Tests for the v3.0.0 Node integration of cost + admission + tier surfaces."""

from __future__ import annotations

import pytest

from membrane.decision import AdmissionPolicy, TenantQuota, TinyLFU
from membrane.fragment import Fragment
from membrane.identity import PayloadIdentity
from membrane.node import Node
from membrane.tiers import TierPolicy
from membrane.transfer_engine_ext import AdaptiveFragmenter


def _fragment(tenant: str, reuse: float = 0.5) -> Fragment:
    """Build a fragment for the tests.

    Args:
        tenant: Tenant namespace the fragment lives in.
        reuse: Initial reuse score.

    Returns:
        Fragment: A unique fragment.
    """
    ident = PayloadIdentity(
        payload_hash=f"hash-{tenant}".ljust(64, "0")[:64],
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
        payload_size=10,
        ttl=60.0,
        reuse_score=reuse,
        version_id=1,
        tenant_id=tenant,
    )


class TestNodeAdmissionPolicy:
    def test_disabled_admits_everything(self):
        node = Node(node_id="n1", max_memory_bytes=10_000)
        node.store(_fragment("acme", reuse=0.0))
        assert len(node.fragments) == 1

    def test_low_reuse_rejected_when_enabled(self):
        policy = AdmissionPolicy(enabled=True, min_reuse_score=0.6)
        node = Node(node_id="n1", max_memory_bytes=10_000, admission_policy=policy)
        node.store(_fragment("acme", reuse=0.1))
        assert len(node.fragments) == 0

    def test_high_reuse_admitted_when_enabled(self):
        policy = AdmissionPolicy(enabled=True, min_reuse_score=0.6)
        node = Node(node_id="n1", max_memory_bytes=10_000, admission_policy=policy)
        node.store(_fragment("acme", reuse=0.9))
        assert len(node.fragments) == 1


class TestNodeTenantQuota:
    def test_byte_cap_rejects_second_write(self):
        quota = TenantQuota(tenant_id="acme", max_bytes=15)
        node = Node(node_id="n1", max_memory_bytes=10_000, quotas={"acme": quota})
        node.store(_fragment("acme"))  # 10 bytes
        node.store(_fragment("acme"))  # another 10 bytes -> over budget
        assert len(node.fragments) == 1

    def test_entry_cap(self):
        quota = TenantQuota(tenant_id="acme", max_entries=1)
        node = Node(node_id="n1", max_memory_bytes=10_000, quotas={"acme": quota})
        node.store(_fragment("acme", reuse=0.5))
        # Second fragment with a different hash but same tenant
        ident = _fragment("acme", reuse=0.5)
        ident = Fragment(
            identity=PayloadIdentity(
                payload_hash="h" * 64,
                model_id="m",
                model_revision="",
                tokenizer_name="m",
                tokenizer_revision="",
                layer_range=(0, 1),
                head_range=(-1, -1),
                token_span=(0, 1),
                dtype="float16",
                shape=(1, 1, 1, 1, 64),
            ),
            payload_ref=None,
            payload_size=10,
            ttl=60.0,
            reuse_score=0.5,
            version_id=1,
            tenant_id="acme",
        )
        node.store(ident)
        assert len(node.fragments) == 1


class TestNodeTierPolicy:
    def test_tier_assigned_on_store(self):
        policy = TierPolicy(hot_threshold=0.7, warm_threshold=0.4, archive_threshold=0.0)
        node = Node(node_id="n1", max_memory_bytes=10_000, tier_policy=policy)
        node.store(_fragment("acme", reuse=0.9))
        assert node.tier_of(_fragment("acme").identity.payload_hash) == "hot"

    def test_no_tier_when_policy_unset(self):
        node = Node(node_id="n1", max_memory_bytes=10_000)
        node.store(_fragment("acme"))
        assert node.tier_of(_fragment("acme").identity.payload_hash) is None


class TestNodeAdaptiveFragmenter:
    def test_window_size_default(self):
        node = Node(node_id="n1", max_memory_bytes=10_000)
        assert node.window_size() == 128

    def test_window_size_with_fragmenter(self):
        frag = AdaptiveFragmenter(enabled=True, model_id="llama-3-8b")
        node = Node(node_id="n1", max_memory_bytes=10_000, fragmenter=frag)
        assert node.window_size() == 128


class TestNodeRecordHit:
    def test_hit_observer_increments_reuse_score(self):
        node = Node(node_id="n1", max_memory_bytes=10_000)
        frag = _fragment("acme", reuse=0.0)
        node.store(frag)
        node.record_hit(frag.identity.payload_hash)
        assert node.fragments[frag.identity.payload_hash].reuse_score > 0.0

    def test_record_hit_touches_tinylfu(self):
        cache = TinyLFU(capacity=10)
        node = Node(node_id="n1", max_memory_bytes=10_000, eviction_strategy=cache)
        frag = _fragment("acme")
        node.store(frag)
        node.record_hit(frag.identity.payload_hash)
        # The TinyLFU sketch bucket should now be incremented.
        assert cache._estimate(frag.identity.payload_hash) >= 1
