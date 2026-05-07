"""Tests for tenant_isolation module."""

import pytest

from membrane.fragment import Fragment
from membrane.structural_signature import StructuralSignature
from membrane.tenant_isolation import TenantIsolation, TenantPolicy


def make_fragment(model_id="prefix", reuse_score=0.5):
    return Fragment(
        content_hash="abc",
        embedding=(0.0,),
        structural_signature=StructuralSignature(
            model_id=model_id, layer_range=(0, 1), token_span=(0, 1)
        ),
        size=10,
        ttl=3600.0,
        reuse_score=reuse_score,
        version_id=1,
    )


class TestTenantIsolation:
    """Test suite for TenantIsolation."""

    def test_same_tenant_always_true(self):
        ti = TenantIsolation()
        frag = make_fragment()
        assert ti.can_share(frag, "t1", "t1")

    def test_low_reuse_score_blocks_share(self):
        ti = TenantIsolation()
        frag = make_fragment(reuse_score=0.1)
        assert not ti.can_share(frag, "t1", "t2")

    def test_public_prefixes_allowed_by_default(self):
        ti = TenantIsolation()
        frag = make_fragment(model_id="prefix", reuse_score=0.8)
        assert ti.can_share(frag, "t1", "t2")

    def test_public_prefixes_blocked_when_policy_false(self):
        policy = TenantPolicy(allow_public_prefixes=False)
        ti = TenantIsolation(policy=policy)
        frag = make_fragment(model_id="prefix", reuse_score=0.8)
        assert not ti.can_share(frag, "t1", "t2")

    def test_tool_traces_blocked_by_default(self):
        ti = TenantIsolation()
        frag = make_fragment(model_id="tool", reuse_score=0.8)
        assert not ti.can_share(frag, "t1", "t2")

    def test_tool_traces_allowed_when_policy_true(self):
        policy = TenantPolicy(allow_tool_traces=True)
        ti = TenantIsolation(policy=policy)
        frag = make_fragment(model_id="tool", reuse_score=0.8)
        assert ti.can_share(frag, "t1", "t2")

    def test_artifacts_allowed_by_default(self):
        ti = TenantIsolation()
        frag = make_fragment(model_id="artifact", reuse_score=0.8)
        assert ti.can_share(frag, "t1", "t2")

    def test_artifacts_blocked_when_policy_false(self):
        policy = TenantPolicy(allow_artifacts=False)
        ti = TenantIsolation(policy=policy)
        frag = make_fragment(model_id="artifact", reuse_score=0.8)
        assert not ti.can_share(frag, "t1", "t2")

    def test_default_policy_values(self):
        policy = TenantPolicy()
        assert policy.allow_public_prefixes is True
        assert policy.allow_tool_traces is False
        assert policy.allow_artifacts is True
        assert policy.min_reuse_score_for_share == 0.6
